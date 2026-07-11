from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ema_filter import EMAFilter
from fsm import DrowsinessSignals
from perclos import PERCLOSCalculator
from runtime.config import RuntimeConfig
from runtime.perception import RawPerception

from runtime.dynamic_ear import DynamicEAR as SqadDynamicEAR


@dataclass
class FeatureState:
    calibrated: bool = False
    ear_threshold: float = 0.23
    calibration_count: int = 0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SignalFeaturePipeline:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.feature_backend = str(getattr(config, "feature_backend", "phuong")).strip().lower()
        self.state = FeatureState(ear_threshold=config.thresholds.ear_default)

        self.ear_ema = EMAFilter(alpha=0.3)
        self.mar_ema = EMAFilter(alpha=0.3)
        self.pitch_ema = EMAFilter(alpha=0.3)
        self.dynamic_ear = SqadDynamicEAR() if self.feature_backend == "phuong" and SqadDynamicEAR is not None else None

        self.perclos_long = PERCLOSCalculator(
            window_seconds=config.windows.perclos_seconds,
            fps=config.runtime.fps,
        )
        self.perclos_short = PERCLOSCalculator(
            window_seconds=config.windows.perclos_short_seconds,
            fps=config.runtime.fps,
        )

        self._pitch_samples: list[float] = []
        self._yaw_samples: list[float] = []
        self._ear_samples: list[float] = []
        self._base_pitch = 0.0
        self._base_yaw = 0.0

        self._prev_rel_pitch: float | None = None
        self._eyes_closed_previous = False
        self._eyes_closed_consecutive = 0

        self._blink_timestamps: deque[float] = deque(maxlen=1000)
        self._yawn_timestamps: deque[float] = deque(maxlen=1000)
        self._gaze_points: deque[tuple[float, float]] = deque(maxlen=100)

        self._current_yawn_frames = 0
        self._head_nod_counter = 0

    def update(self, raw: RawPerception, now: float) -> tuple[DrowsinessSignals, dict[str, Any]]:
        if raw.face_detected and not self.state.calibrated:
            self._ear_samples.append(raw.ear)
            self._pitch_samples.append(raw.pitch)
            self._yaw_samples.append(raw.yaw)
            self.state.calibration_count += 1
            if self.state.calibration_count >= self.config.runtime.calibration_frames:
                baseline_ear = float(np.mean(self._ear_samples)) if self._ear_samples else self.config.thresholds.ear_default
                self.state.ear_threshold = _clamp(
                    baseline_ear * self.config.thresholds.ear_calibration_factor,
                    self.config.thresholds.ear_min,
                    self.config.thresholds.ear_max,
                )
                self._base_pitch = float(np.mean(self._pitch_samples)) if self._pitch_samples else 0.0
                self._base_yaw = float(np.mean(self._yaw_samples)) if self._yaw_samples else 0.0
                self.state.calibrated = True

        if raw.face_detected:
            rel_pitch = raw.pitch - self._base_pitch
            rel_yaw = raw.yaw - self._base_yaw
            ear_raw = raw.ear
            mar_raw = raw.mar
            if raw.gaze_center is not None:
                self._gaze_points.append(raw.gaze_center)
        else:
            rel_pitch = 0.0
            rel_yaw = 0.0
            ear_raw = self.state.ear_threshold + 0.02
            mar_raw = 0.0

        eye_status = None
        eye_detector_source = "legacy_calibration"
        effective_ear_threshold = self.state.ear_threshold
        if raw.face_detected and self.dynamic_ear is not None:
            eye_status = self.dynamic_ear.update(float(ear_raw))
            if getattr(eye_status, "locked", False) and eye_status.T_low is not None:
                effective_ear_threshold = float(eye_status.T_low)
                eye_detector_source = "sqad_dynamic_ear"

        ear_smooth = self.ear_ema.update(ear_raw)
        mar_smooth = self.mar_ema.update(mar_raw)
        pitch_smooth = self.pitch_ema.update(rel_pitch)

        if eye_status is not None and eye_detector_source == "sqad_dynamic_ear":
            eye_closed = bool(eye_status.is_closed)
        else:
            eye_closed = ear_raw < effective_ear_threshold
        if eye_closed:
            self._eyes_closed_consecutive += 1
        else:
            if self._eyes_closed_previous and self._eyes_closed_consecutive >= 2:
                self._blink_timestamps.append(now)
            self._eyes_closed_consecutive = 0
        self._eyes_closed_previous = eye_closed

        if mar_raw > self.config.thresholds.mar:
            self._current_yawn_frames += 1
        else:
            if self._current_yawn_frames >= self.config.thresholds.yawn_frames:
                self._yawn_timestamps.append(now)
            self._current_yawn_frames = 0

        if (
            rel_pitch > self.config.thresholds.pitch
            and abs(rel_yaw) < self.config.thresholds.head_yaw
            and mar_raw <= self.config.thresholds.mar
        ):
            self._head_nod_counter += 1
        else:
            self._head_nod_counter = max(0, self._head_nod_counter - 1)

        head_nod_detected = self._head_nod_counter >= self.config.thresholds.head_nod_frames

        self._trim_timestamps(self._blink_timestamps, now, self.config.windows.blink_window_seconds)
        self._trim_timestamps(self._yawn_timestamps, now, self.config.windows.yawn_window_seconds)

        blink_frequency = len(self._blink_timestamps)
        yawn_frequency = len(self._yawn_timestamps)

        movement_score = None
        gaze_stable = False
        if len(self._gaze_points) >= 10:
            points = np.array(self._gaze_points)
            std = np.std(points, axis=0)
            movement_score = float(np.mean(std))
            gaze_stable = movement_score < self.config.thresholds.gaze_move

        perclos = self.perclos_long.update(eye_closed)
        perclos_short = self.perclos_short.update(eye_closed)

        if self._prev_rel_pitch is None:
            pitch_velocity = 0.0
        else:
            pitch_velocity = pitch_smooth - self._prev_rel_pitch
        self._prev_rel_pitch = pitch_smooth

        signals = DrowsinessSignals(
            ear=float(ear_smooth),
            mar=float(mar_smooth),
            pitch=float(pitch_smooth),
            pitch_velocity=float(pitch_velocity),
            perclos=float(perclos),
            perclos_short=float(perclos_short),
            yawn_frequency=int(yawn_frequency),
            blink_frequency=int(blink_frequency),
            gaze_stable=bool(gaze_stable),
            head_nod_detected=bool(head_nod_detected),
            eyes_closed_consecutive=int(self._eyes_closed_consecutive),
            face_detected=bool(raw.face_detected),
            ear_below_threshold=bool(eye_closed),
            mar_above_threshold=bool(mar_raw > self.config.thresholds.mar),
            pitch_above_threshold=bool(rel_pitch > self.config.thresholds.pitch),
        )

        debug = {
            "feature_backend": self.feature_backend,
            "calibrated": self.state.calibrated,
            "ear_threshold": effective_ear_threshold,
            "eye_detector_source": eye_detector_source,
            "eye_dynamic_phase": self._eye_dynamic_phase(eye_status),
            "eye_dynamic_threshold": getattr(eye_status, "T_low", None) if eye_status is not None else None,
            "eye_dynamic_progress": getattr(eye_status, "progress", 0.0) if eye_status is not None else 0.0,
            "ear_raw": ear_raw,
            "mar_raw": mar_raw,
            "pitch_raw": rel_pitch,
            "rel_yaw": rel_yaw,
            "movement_score": movement_score,
            "blink_frequency": blink_frequency,
            "yawn_frequency": yawn_frequency,
            "head_nod_counter": self._head_nod_counter,
            "face_detected": raw.face_detected,
        }
        return signals, debug

    def _eye_dynamic_phase(self, eye_status: Any) -> str:
        if self.feature_backend == "legacy":
            return "DISABLED"
        return getattr(eye_status, "phase", "UNAVAILABLE") if eye_status is not None else "UNAVAILABLE"

    @staticmethod
    def _trim_timestamps(queue: deque[float], now: float, window_seconds: float) -> None:
        while queue and (now - queue[0]) > window_seconds:
            queue.popleft()
