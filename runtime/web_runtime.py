from __future__ import annotations

from collections import Counter, OrderedDict, deque
import hashlib
import json
import copy
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable
import uuid

from runtime.config import RuntimeConfig, load_runtime_config
from runtime.contracts import DecisionResult, EngineContext
from runtime.engines.registry import create_engine
from runtime.features import SignalFeaturePipeline
from runtime.landmark_adapter import LandmarkPacketAdapter, REQUIRED_LANDMARKS
from runtime.model_bundle import WinnerModelBundle
from runtime.perception import RawPerception


LOGGER = logging.getLogger(__name__)
REFERENCE_FPS = 30.0
REFERENCE_STEP_MS = 1000.0 / REFERENCE_FPS
ALLOWED_TARGET_FPS = {10, 15, 20, 30}
CAPTURE_STALL_TOLERANCE_MS = 3000.0
MAX_GAP_VIRTUAL_FRAMES = 96


class ProtocolError(ValueError):
    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


class TimestampNormalizer:
    def __init__(
        self,
        reference_fps: float = REFERENCE_FPS,
        max_hold_ms: float = 250.0,
        max_gap_ms: float = CAPTURE_STALL_TOLERANCE_MS,
        max_generated_frames: int = MAX_GAP_VIRTUAL_FRAMES,
    ) -> None:
        self.step_ms = 1000.0 / float(reference_fps)
        self.max_hold_ms = float(max_hold_ms)
        self.max_gap_ms = float(max_gap_ms)
        self.max_generated_frames = int(max_generated_frames)
        self.next_grid_ms: float | None = None
        self.last_input_ms: float | None = None
        self.held_packet: dict[str, Any] | None = None

    def push(self, packet: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
        timestamp_ms = float(packet["timestamp_ms"])
        if not math.isfinite(timestamp_ms):
            raise ProtocolError("timestamp_ms must be finite", 400)
        if self.last_input_ms is not None:
            if timestamp_ms <= self.last_input_ms:
                raise ProtocolError("Frame timestamps must be strictly increasing")
            if timestamp_ms - self.last_input_ms > self.max_gap_ms:
                raise ProtocolError(
                    f"Frame timestamp gap exceeds {self.max_gap_ms:.0f}ms; reset the session",
                    409,
                )
        if self.next_grid_ms is not None:
            generated = max(0, math.ceil((timestamp_ms - self.next_grid_ms) / self.step_ms))
            if generated > self.max_generated_frames:
                raise ProtocolError("Frame timestamp gap would generate too many virtual frames", 409)

        current = copy.deepcopy(packet)
        if self.next_grid_ms is None:
            self.next_grid_ms = timestamp_ms
            self.last_input_ms = timestamp_ms
            self.held_packet = current
            output = [(timestamp_ms, current)]
            self.next_grid_ms += self.step_ms
            return output

        output: list[tuple[float, dict[str, Any]]] = []
        epsilon = 1e-6
        while self.next_grid_ms < timestamp_ms - epsilon:
            output.append((self.next_grid_ms, self._held_for_grid(self.next_grid_ms)))
            self.next_grid_ms += self.step_ms

        if abs(self.next_grid_ms - timestamp_ms) <= epsilon:
            output.append((timestamp_ms, current))
            self.next_grid_ms += self.step_ms

        self.last_input_ms = timestamp_ms
        self.held_packet = current
        return output

    def reset(self) -> None:
        self.next_grid_ms = None
        self.last_input_ms = None
        self.held_packet = None

    def _held_for_grid(self, grid_ms: float) -> dict[str, Any]:
        if self.held_packet is None or self.last_input_ms is None:
            return {"timestamp_ms": grid_ms, "face_detected": False, "landmarks": {}}
        if grid_ms - self.last_input_ms > self.max_hold_ms:
            return {"timestamp_ms": grid_ms, "face_detected": False, "landmarks": {}}
        held = copy.deepcopy(self.held_packet)
        held["timestamp_ms"] = grid_ms
        return held


class AlertCommandController:
    def __init__(self, cooldown_seconds: float) -> None:
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.last_double_at: float | None = None
        self.continuous_active = False

    def update(self, requested_sound: str, now: float) -> str:
        requested = str(requested_sound or "none").strip().lower()
        clock = float(now)
        if requested == "continuous":
            if self.continuous_active:
                return "none"
            self.continuous_active = True
            return "continuous_start"

        if self.continuous_active:
            self.continuous_active = False
            return "continuous_stop"

        if requested == "double":
            if self.last_double_at is None or clock - self.last_double_at >= self.cooldown_seconds:
                self.last_double_at = clock
                return "double"
        return "none"

    def reset(self) -> None:
        self.last_double_at = None
        self.continuous_active = False


@dataclass(frozen=True)
class SessionLimits:
    max_input_fps: int = 40
    max_batches_per_second: int = 12
    max_frames_per_batch: int = 4
    max_payload_bytes: int = 64 * 1024
    idle_timeout_seconds: float = 5 * 60.0
    max_active_sessions: int = 16
    max_session_creates_per_minute: int = 60
    max_active_sessions_per_client: int = 4
    max_session_creates_per_client_per_minute: int = 12
    max_virtual_frames_per_second: int = 96


class SlidingRateLimiter:
    def __init__(self, limit: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = int(limit)
        self.clock = clock
        self.events: list[float] = []

    def consume(self, count: int = 1) -> bool:
        now = self.clock()
        cutoff = now - 1.0
        self.events = [value for value in self.events if value > cutoff]
        if len(self.events) + int(count) > self.limit:
            return False
        self.events.extend([now] * int(count))
        return True


class WinnerRuntime:
    def __init__(self, root: Path, profile_name: str = "recommended") -> None:
        self.root = Path(root).resolve()
        requested = str(profile_name or "recommended").strip().lower()
        if requested not in {"recommended", "protected"}:
            LOGGER.warning("Unknown DMS_RUNTIME_PROFILE=%s; falling back to protected", requested)
            requested = "protected"
        self.profile_name = requested
        self.profile_path = self.root / "configs" / f"{requested}.json"
        self.config = load_runtime_config(str(self.profile_path))
        self.config.runtime.fps = REFERENCE_FPS
        self.config.runtime_profile = requested
        self.config.camera_model.model_path = str((self.root / "models" / "camera_hybrid_winner.joblib").resolve())
        self.bundle = WinnerModelBundle.load(
            self.root / "models" / "camera_hybrid_winner.joblib",
            self.root / "models" / "winner_manifest.json",
        )
        if self.bundle.feature_columns != list(self.bundle.manifest["feature_columns"]):
            raise ValueError("Winner feature contract is not ready")
        if float(self.config.camera_model.probability_threshold or 0.0) != self.bundle.runtime_threshold:
            raise ValueError("Runtime threshold does not match winner manifest")

    def create_session(self, source_mode: str, target_fps: int) -> "WinnerSession":
        mode = str(source_mode or "camera").strip().lower()
        if mode not in {"camera", "file"}:
            raise ProtocolError("source_mode must be camera or file", 400)
        if isinstance(target_fps, bool) or not isinstance(target_fps, int):
            raise ProtocolError("target_fps must be an integer", 400)
        fps = target_fps
        if fps not in ALLOWED_TARGET_FPS:
            raise ProtocolError("target_fps must be one of 10, 15, 20, 30", 400)
        return WinnerSession(self, source_mode=mode, target_fps=fps)


class WinnerSession:
    def __init__(self, runtime: WinnerRuntime, source_mode: str, target_fps: int) -> None:
        self.runtime = runtime
        self.session_id = str(uuid.uuid4())
        self.source_mode = source_mode
        self.target_fps = int(target_fps)
        self.lock = threading.RLock()
        self.limits = SessionLimits()
        self.created_at = time.monotonic()
        self.last_activity = self.created_at
        self.batch_limiter = SlidingRateLimiter(self.limits.max_batches_per_second)
        self.frame_limiter = SlidingRateLimiter(self.limits.max_input_fps)
        self.virtual_frame_limiter = SlidingRateLimiter(self.limits.max_virtual_frames_per_second)
        self.batch_cache: OrderedDict[int, tuple[str, dict[str, Any]]] = OrderedDict()
        self.last_batch_seq = -1
        self.last_frame_seq = -1
        self.last_frame_timestamp_ms: float | None = None
        self.state_counts: Counter[str] = Counter()
        self.audio_counts: Counter[str] = Counter()
        self.virtual_frame_count = 0
        self.input_frame_count = 0
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        self.normalizer = TimestampNormalizer()
        self.adapter = LandmarkPacketAdapter()
        self.features = SignalFeaturePipeline(self.runtime.config)
        self.engine = create_engine("camera_hybrid", self.runtime.config)
        self.engine.initialize(EngineContext(fps=REFERENCE_FPS, metadata={"source": self.source_mode}))
        self.alerts = AlertCommandController(self.runtime.config.alerts.drowsy_cooldown_seconds)
        self.latest_result: DecisionResult | None = None
        self.latest_signals = None
        self.latest_debug: dict[str, Any] = {}
        self.latest_audio_command = "none"

    def process_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._reject_unknown_keys(payload, {"batch_seq", "frames"}, "Batch")
            batch_seq = self._required_int(payload, "batch_seq")
            payload_digest = self._payload_digest(payload)
            if batch_seq in self.batch_cache:
                cached_digest, cached_response = self.batch_cache[batch_seq]
                if cached_digest != payload_digest:
                    raise ProtocolError("Repeated batch_seq does not match the original payload")
                self.last_activity = time.monotonic()
                return copy.deepcopy(cached_response)
            if batch_seq <= self.last_batch_seq:
                raise ProtocolError("Old or out-of-order batch_seq")

            frames = payload.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ProtocolError("frames must be a non-empty list", 400)
            if len(frames) > self.limits.max_frames_per_batch:
                raise ProtocolError("A batch can contain at most 4 frames", 400)
            audio_commands: list[str] = []
            pending_last_seq = self.last_frame_seq
            pending_last_timestamp = self.last_frame_timestamp_ms
            for frame in frames:
                if not isinstance(frame, dict):
                    raise ProtocolError("Each frame must be a JSON object", 400)
                seq = self._required_int(frame, "seq")
                timestamp_ms = self._required_float(frame, "timestamp_ms")
                if seq <= pending_last_seq:
                    raise ProtocolError("Old, duplicate, or out-of-order frame seq")
                if pending_last_timestamp is not None and timestamp_ms <= pending_last_timestamp:
                    raise ProtocolError("Old or out-of-order frame timestamp")
                self._validate_frame(frame)
                pending_last_seq = seq
                pending_last_timestamp = timestamp_ms

            batch_events_before = list(self.batch_limiter.events)
            frame_events_before = list(self.frame_limiter.events)
            virtual_events_before = list(self.virtual_frame_limiter.events)
            if not self.batch_limiter.consume(1):
                raise ProtocolError("Batch rate limit exceeded", 429)
            if not self.frame_limiter.consume(len(frames)):
                self.batch_limiter.events = batch_events_before
                raise ProtocolError("Frame rate limit exceeded", 429)

            processing_snapshot = self._capture_processing_state(
                batch_events_before=batch_events_before,
                frame_events_before=frame_events_before,
                virtual_events_before=virtual_events_before,
            )
            try:
                for frame in frames:
                    virtual_rows = self.normalizer.push(frame)
                    if not self.virtual_frame_limiter.consume(len(virtual_rows)):
                        raise ProtocolError("Virtual frame work limit exceeded", 429)
                    for grid_ms, virtual_packet in virtual_rows:
                        command = self._process_virtual_frame(grid_ms, virtual_packet)
                        if command != "none":
                            audio_commands.append(command)
                    self.input_frame_count += 1
            except Exception:
                self._restore_processing_state(processing_snapshot)
                raise
            self.last_frame_seq = pending_last_seq
            self.last_frame_timestamp_ms = pending_last_timestamp
            self.last_batch_seq = batch_seq
            self.last_activity = time.monotonic()
            response = self._response(audio_commands)
            self.batch_cache[batch_seq] = (payload_digest, copy.deepcopy(response))
            while len(self.batch_cache) > 32:
                self.batch_cache.popitem(last=False)
            return response

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self._log_summary("reset")
            self.last_batch_seq = -1
            self.last_frame_seq = -1
            self.last_frame_timestamp_ms = None
            self.batch_cache.clear()
            self.state_counts.clear()
            self.audio_counts.clear()
            self.virtual_frame_count = 0
            self.input_frame_count = 0
            self._initialize_pipeline()
            self.last_activity = time.monotonic()
            return self._response([])

    def close(self) -> dict[str, Any]:
        with self.lock:
            summary = self.summary("closed")
            self._log_summary("closed")
            return summary

    def summary(self, reason: str) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "reason": reason,
            "profile": self.runtime.profile_name,
            "source_mode": self.source_mode,
            "target_fps": self.target_fps,
            "input_frames": self.input_frame_count,
            "virtual_frames": self.virtual_frame_count,
            "state_counts": dict(sorted(self.state_counts.items())),
            "audio_command_counts": dict(sorted(self.audio_counts.items())),
        }

    @staticmethod
    def _copy_deque(value: deque) -> deque:
        return deque(value, maxlen=value.maxlen)

    def _copy_features_for_rollback(self) -> SignalFeaturePipeline:
        source = self.features
        snapshot = copy.copy(source)
        snapshot.state = copy.copy(source.state)
        for name in ("ear_ema", "mar_ema", "pitch_ema"):
            setattr(snapshot, name, copy.copy(getattr(source, name)))
        for name in ("perclos_long", "perclos_short"):
            calculator = copy.copy(getattr(source, name))
            calculator.eye_closed_flags = self._copy_deque(calculator.eye_closed_flags)
            setattr(snapshot, name, calculator)
        if source.dynamic_ear is not None:
            snapshot.dynamic_ear = copy.copy(source.dynamic_ear)
            snapshot.dynamic_ear._window = self._copy_deque(source.dynamic_ear._window)
        for name in ("_pitch_samples", "_yaw_samples", "_ear_samples"):
            setattr(snapshot, name, list(getattr(source, name)))
        for name in ("_blink_timestamps", "_yawn_timestamps", "_gaze_points"):
            setattr(snapshot, name, self._copy_deque(getattr(source, name)))
        return snapshot

    def _capture_processing_state(
        self,
        *,
        batch_events_before: list[float],
        frame_events_before: list[float],
        virtual_events_before: list[float],
    ) -> dict[str, Any]:
        engine_state: dict[str, Any] = {
            "frame_index": self.engine.frame_index,
            "rows": self._copy_deque(self.engine.rows),
            "previous_window": copy.copy(self.engine.previous_window),
            "feature_fsm": copy.deepcopy(self.engine.feature_fsm),
        }
        for name in ("smoother", "ml_smoother", "hybrid_policy", "macroevent_seed_times"):
            if hasattr(self.engine, name):
                value = getattr(self.engine, name)
                engine_state[name] = self._copy_deque(value) if isinstance(value, deque) else copy.deepcopy(value)
        return {
            "normalizer": copy.deepcopy(self.normalizer),
            "features": self._copy_features_for_rollback(),
            "engine": engine_state,
            "alerts": copy.copy(self.alerts),
            "latest_result": self.latest_result,
            "latest_signals": self.latest_signals,
            "latest_debug": dict(self.latest_debug),
            "latest_audio_command": self.latest_audio_command,
            "state_counts": self.state_counts.copy(),
            "audio_counts": self.audio_counts.copy(),
            "virtual_frame_count": self.virtual_frame_count,
            "input_frame_count": self.input_frame_count,
            "batch_limiter_events": batch_events_before,
            "frame_limiter_events": frame_events_before,
            "virtual_limiter_events": virtual_events_before,
        }

    def _restore_processing_state(self, snapshot: dict[str, Any]) -> None:
        self.normalizer = snapshot["normalizer"]
        self.features = snapshot["features"]
        for name, value in snapshot["engine"].items():
            setattr(self.engine, name, value)
        self.alerts = snapshot["alerts"]
        self.latest_result = snapshot["latest_result"]
        self.latest_signals = snapshot["latest_signals"]
        self.latest_debug = snapshot["latest_debug"]
        self.latest_audio_command = snapshot["latest_audio_command"]
        self.state_counts = snapshot["state_counts"]
        self.audio_counts = snapshot["audio_counts"]
        self.virtual_frame_count = snapshot["virtual_frame_count"]
        self.input_frame_count = snapshot["input_frame_count"]
        self.batch_limiter.events = snapshot["batch_limiter_events"]
        self.frame_limiter.events = snapshot["frame_limiter_events"]
        self.virtual_frame_limiter.events = snapshot["virtual_limiter_events"]

    def _process_virtual_frame(self, grid_ms: float, packet: dict[str, Any]) -> str:
        if bool(packet.get("face_detected", False)):
            raw = self.adapter.from_normalized(
                packet.get("landmarks", {}),
                width=int(packet["width"]),
                height=int(packet["height"]),
                face_detected=True,
            )
        else:
            raw = RawPerception(face_detected=False)
        signals, feature_debug = self.features.update(raw, now=grid_ms / 1000.0)
        result = self.engine.update(signals)
        command = self.alerts.update(result.alert_sound, now=grid_ms / 1000.0)
        self.latest_result = result
        self.latest_signals = signals
        self.latest_debug = {**feature_debug, **result.debug}
        self.latest_audio_command = command
        self.virtual_frame_count += 1
        self.state_counts[result.state.value] += 1
        self.audio_counts[command] += 1
        return command

    def _response(self, audio_commands: list[str]) -> dict[str, Any]:
        result = self.latest_result
        debug = self.latest_debug
        state = result.state.value if result is not None else "ALERT"
        probability = debug.get("sleepy_probability")
        dynamic_progress = float(debug.get("eye_dynamic_progress", 0.0) or 0.0)
        dynamic_phase = str(debug.get("eye_dynamic_phase", "WARMUP"))
        if dynamic_phase == "LOCKED":
            dynamic_total_progress = 1.0
        elif dynamic_phase == "CALIBRATING":
            dynamic_total_progress = min(1.0, (30.0 + 270.0 * dynamic_progress) / 300.0)
        else:
            dynamic_total_progress = min(0.1, 30.0 * dynamic_progress / 300.0)
        signals = self.latest_signals
        metrics = {
            "ear": float(signals.ear) if signals is not None else None,
            "mar": float(signals.mar) if signals is not None else None,
            "pitch": float(signals.pitch) if signals is not None else None,
            "pitch_velocity": float(signals.pitch_velocity) if signals is not None else None,
            "perclos": float(signals.perclos) if signals is not None else None,
            "perclos_short": float(signals.perclos_short) if signals is not None else None,
            "blink_frequency": int(signals.blink_frequency) if signals is not None else 0,
            "yawn_frequency": int(signals.yawn_frequency) if signals is not None else 0,
            "eyes_closed_consecutive": int(signals.eyes_closed_consecutive) if signals is not None else 0,
            "face_detected": bool(debug.get("face_detected", False)),
            "ear_threshold": float(debug.get("ear_threshold", 0.0) or 0.0),
            "mar_threshold": float(self.runtime.config.thresholds.mar),
            "head_nod_detected": bool(signals.head_nod_detected) if signals is not None else False,
        }
        return {
            "session_id": self.session_id,
            "state": state,
            "label": result.label if result is not None else "MODEL WARMUP",
            "probability": None if probability is None else float(probability),
            "threshold": float(self.runtime.config.camera_model.probability_threshold or self.runtime.bundle.runtime_threshold),
            "hybrid_guard": str(debug.get("hybrid_guard", "warmup")),
            "reasons": list(result.reasons) if result is not None else ["CAMERA_MODEL_WARMUP"],
            "metrics": metrics,
            "calibration": {
                "valid_face_frames": int(self.features.state.calibration_count),
                "runtime_target_frames": int(self.runtime.config.runtime.calibration_frames),
                "runtime_calibrated": bool(self.features.state.calibrated),
                "dynamic_phase": dynamic_phase,
                "dynamic_progress": dynamic_progress,
                "dynamic_total_progress": dynamic_total_progress,
                "dynamic_target_frames": 300,
                "application_warmup_seconds": float(self.runtime.config.runtime.warmup_seconds),
                "model_ready": probability is not None,
            },
            "profile": self.runtime.profile_name,
            "model_hash": self.runtime.bundle.sha256,
            "runtime_alert_semantic": str(debug.get("runtime_alert_semantic", "standard")),
            "visual_alert_mode": str(debug.get("visual_alert_mode", "")),
            "audio_command": audio_commands[-1] if audio_commands else "none",
            "audio_commands": list(audio_commands),
            "input_frames": self.input_frame_count,
            "virtual_frames": self.virtual_frame_count,
        }

    @staticmethod
    def _payload_digest(payload: dict[str, Any]) -> str:
        try:
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Batch payload must be canonical JSON", 400) from exc
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _required_int(payload: dict[str, Any], key: str) -> int:
        try:
            return int(payload[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"{key} must be an integer", 400) from exc

    @staticmethod
    def _required_float(payload: dict[str, Any], key: str) -> float:
        try:
            value = float(payload[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"{key} must be numeric", 400) from exc
        if not math.isfinite(value):
            raise ProtocolError(f"{key} must be finite", 400)
        return value

    @staticmethod
    def _reject_unknown_keys(payload: dict[str, Any], allowed: set[str], label: str) -> None:
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ProtocolError(f"{label} contains unsupported fields: {', '.join(unknown)}", 400)

    @staticmethod
    def _validate_frame(frame: dict[str, Any]) -> None:
        WinnerSession._reject_unknown_keys(
            frame,
            {"seq", "timestamp_ms", "width", "height", "face_detected", "landmarks"},
            "Frame",
        )
        try:
            width = int(frame["width"])
            height = int(frame["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("width and height must be integers", 400) from exc
        if width <= 0 or height <= 0:
            raise ProtocolError("width and height must be positive", 400)
        if not bool(frame.get("face_detected", False)):
            landmarks = frame.get("landmarks")
            if landmarks not in (None, {}):
                raise ProtocolError("No-face frames require absent or empty landmarks", 400)
            return
        landmarks = frame.get("landmarks")
        if not isinstance(landmarks, dict):
            raise ProtocolError("Detected faces require a landmarks object", 400)
        try:
            indices = {int(key) for key in landmarks}
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Landmark keys must be integer indices", 400) from exc
        if indices != set(REQUIRED_LANDMARKS):
            raise ProtocolError("Landmark packet must contain exactly the required 20 indices", 400)
        for key, point in landmarks.items():
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ProtocolError(f"Landmark {key} must contain exactly x and y", 400)
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError) as exc:
                raise ProtocolError(f"Landmark {key} coordinates must be numeric", 400) from exc
            if not math.isfinite(x) or not math.isfinite(y) or not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                raise ProtocolError(f"Landmark {key} x/y must be finite normalized coordinates", 400)

    def _log_summary(self, reason: str) -> None:
        LOGGER.info("winner_session_summary %s", self.summary(reason))