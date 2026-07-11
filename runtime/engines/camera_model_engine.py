from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from fsm import ALERT_CONFIGS, DrowsinessFSM, DrowsinessSignals, DrowsinessState
from runtime.config import RuntimeConfig
from runtime.contracts import DecisionResult, EngineContext
from runtime.engines.base import DecisionEngine
from runtime.model_bundle import load_model_artifact
from runtime.window_features import aggregate_runtime_window, numeric_feature_value


class ProbabilityStateSmoother:
    def __init__(self) -> None:
        self.state_index = 0
        self.elevated_streak = 0
        self.high_streak = 0
        self.low_streak = 0
        self.names = [
            DrowsinessState.ALERT,
            DrowsinessState.SUSPICIOUS,
            DrowsinessState.DROWSY,
            DrowsinessState.CRITICAL,
        ]

    def update(self, probability: float) -> DrowsinessState:
        if probability >= 0.85:
            self.high_streak += 1
        else:
            self.high_streak = 0

        if probability >= 0.60:
            self.elevated_streak += 1
        else:
            self.elevated_streak = 0

        if probability < 0.35:
            self.low_streak += 1
        else:
            self.low_streak = 0

        if self.high_streak >= 2:
            self.state_index = 3
        elif self.elevated_streak >= 2:
            self.state_index = max(self.state_index, 2)
        elif probability >= 0.35:
            self.state_index = max(self.state_index, 1)
        elif self.low_streak >= 1:
            self.state_index = max(0, self.state_index - 1)

        return self.names[self.state_index]

    def reset(self) -> None:
        self.state_index = 0
        self.elevated_streak = 0
        self.high_streak = 0
        self.low_streak = 0


class CameraModelDecisionEngine(DecisionEngine):
    """Runtime engine for models trained on live camera-available window features."""

    name = "camera_model"

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.model_path = Path(config.camera_model.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Camera model artifact not found: {self.model_path}")

        artifact = load_model_artifact(str(self.model_path.resolve()))
        if not isinstance(artifact, dict) or "model" not in artifact or "feature_columns" not in artifact:
            raise ValueError("Camera model artifact must contain 'model' and 'feature_columns'.")

        self.model = artifact["model"]
        self.feature_columns = [str(feature) for feature in artifact["feature_columns"]]
        artifact_threshold = float(artifact.get("probability_threshold", 0.5))
        config_threshold = config.camera_model.probability_threshold
        self.threshold = float(config_threshold) if config_threshold is not None else artifact_threshold
        self.feature_set = str(artifact.get("feature_set", "camera"))
        self.quality_guard = config.camera_model.quality_guard

        self.window_seconds = float(config.camera_model.window_seconds)
        self.min_window_seconds = float(config.camera_model.min_window_seconds)
        self.min_frames = int(config.camera_model.min_frames)

        self.fps = config.runtime.fps
        self.frame_index = 0
        self.rows: deque[dict[str, Any]] = deque()
        self.previous_window: dict[str, Any] | None = None
        self.smoother = ProbabilityStateSmoother()
        self.feature_fsm = self._new_feature_fsm(self.fps)

    def initialize(self, context: EngineContext) -> None:
        self.fps = context.fps
        self.feature_fsm = self._new_feature_fsm(context.fps)

    def update(self, signals: DrowsinessSignals) -> DecisionResult:
        timestamp_sec = self.frame_index / max(self.fps, 1e-6)
        self.frame_index += 1

        base_state = self.feature_fsm.update(signals)
        base_evidence = float(self.feature_fsm.evidence_score)
        self._append_frame_row(timestamp_sec, signals, base_state, base_evidence)

        if not self._window_ready(timestamp_sec):
            return self._warmup_result(base_state, base_evidence)

        window = self._build_window_row()
        if window is None:
            return self._warmup_result(base_state, base_evidence)
        quality_guard_reason = self._quality_guard_reason(window)
        if quality_guard_reason is not None:
            return self._quality_guard_result(base_state, base_evidence, window, quality_guard_reason)

        probability = self._sleepy_probability(window)
        state = self.smoother.update(probability)
        alert_cfg = ALERT_CONFIGS[state]
        predicted_sleepy = probability >= self.threshold

        reasons = ["CAMERA_MODEL"]
        if predicted_sleepy:
            reasons.append("MODEL_SLEEPY_PROBABILITY")
        if state in {DrowsinessState.DROWSY, DrowsinessState.CRITICAL}:
            reasons.append("SMOOTHED_SLEEPY_STATE")

        return DecisionResult(
            state=state,
            evidence=probability,
            reasons=reasons,
            alert_sound=alert_cfg.sound_type,
            color=alert_cfg.color,
            label=f"MODEL {alert_cfg.text}",
            debug={
                "sleepy_probability": probability,
                "probability_threshold": self.threshold,
                "predicted_sleepy": int(predicted_sleepy),
                "feature_set": self.feature_set,
                "feature_count": len(self.feature_columns),
                "model_feature_columns": list(self.feature_columns),
                "model_feature_vector": self._model_feature_debug(window),
                "valid_face_ratio": numeric_feature_value(window, "valid_face_ratio"),
                "quality_guard_triggered": 0,
                "base_fsm_state": base_state.value,
                "base_fsm_evidence": base_evidence,
            },
        )

    def reset(self) -> None:
        self.frame_index = 0
        self.rows.clear()
        self.previous_window = None
        self.smoother.reset()
        self.feature_fsm = self._new_feature_fsm(self.fps)

    def _append_frame_row(
        self,
        timestamp_sec: float,
        signals: DrowsinessSignals,
        base_state: DrowsinessState,
        base_evidence: float,
    ) -> None:
        self.rows.append(
            {
                "subject_id": "runtime",
                "session_id": "runtime",
                "video_id": "runtime",
                "timestamp_sec": timestamp_sec,
                "face_detected": int(getattr(signals, "face_detected", True)),
                "ear": signals.ear,
                "mar": signals.mar,
                "eye_closed": int(signals.ear_below_threshold),
                "mouth_open": int(signals.mar_above_threshold),
                "head_nod_detected": int(signals.head_nod_detected),
                "perclos_60s": signals.perclos,
                "perclos_5s": signals.perclos_short,
                "blink_frequency": signals.blink_frequency,
                "yawn_frequency": signals.yawn_frequency,
                "pitch_velocity": signals.pitch_velocity,
                "gaze_stable": int(signals.gaze_stable),
                "fsm_state": base_state.value,
                "fsm_evidence": base_evidence,
            }
        )
        while self.rows and timestamp_sec - float(self.rows[0]["timestamp_sec"]) > self.window_seconds:
            self.rows.popleft()

    def _window_ready(self, timestamp_sec: float) -> bool:
        if len(self.rows) < self.min_frames:
            return False
        oldest = float(self.rows[0]["timestamp_sec"])
        return (timestamp_sec - oldest) >= self.min_window_seconds

    def _build_window_row(self) -> dict[str, Any] | None:
        aggregated = aggregate_runtime_window(list(self.rows), window_seconds=self.window_seconds)
        if aggregated is None:
            return None
        window = dict(aggregated)
        self._add_delta_features(window)
        return window

    def _add_delta_features(self, window: dict[str, Any]) -> None:
        previous = self.previous_window
        for feature, value in list(window.items()):
            if feature == "fsm_state_mode":
                continue
            try:
                current_value = float(value)
            except (TypeError, ValueError):
                continue
            if previous is None:
                window[f"{feature}__delta_prev"] = 0.0
            else:
                window[f"{feature}__delta_prev"] = current_value - numeric_feature_value(previous, feature)
        self.previous_window = dict(window)

    def _sleepy_probability(self, window: dict[str, Any]) -> float:
        values = [[numeric_feature_value(window, feature) for feature in self.feature_columns]]
        probabilities = self.model.predict_proba(values)[0]
        class_index = self._positive_class_index()
        return float(probabilities[class_index])

    def _model_feature_debug(self, window: dict[str, Any]) -> dict[str, float]:
        return {feature: numeric_feature_value(window, feature) for feature in self.feature_columns}

    def _quality_guard_reason(self, window: dict[str, Any]) -> str | None:
        if not getattr(self.quality_guard, "enabled", False):
            return None
        if numeric_feature_value(window, "valid_face_ratio") < float(self.quality_guard.min_valid_face_ratio):
            return "low_valid_face_ratio"
        degenerate_features = [
            "mean_ear",
            "min_ear",
            "mean_mar",
            "max_mar",
            "perclos_60s",
            "perclos_5s",
            "max_fsm_evidence",
        ]
        if all(abs(numeric_feature_value(window, feature)) <= 1e-9 for feature in degenerate_features):
            return "degenerate_window"
        return None

    def _quality_guard_result(
        self,
        base_state: DrowsinessState,
        base_evidence: float,
        window: dict[str, Any],
        reason: str,
    ) -> DecisionResult:
        alert_cfg = ALERT_CONFIGS[base_state]
        return DecisionResult(
            state=base_state,
            evidence=base_evidence,
            reasons=[self._engine_label(), "QUALITY_GUARD", reason.upper()],
            alert_sound=alert_cfg.sound_type,
            color=alert_cfg.color,
            label=f"{self._label_prefix()} {alert_cfg.text}",
            debug={
                "predicted_sleepy": 0,
                "feature_set": self.feature_set,
                "feature_count": len(self.feature_columns),
                "model_feature_columns": list(self.feature_columns),
                "model_feature_vector": self._model_feature_debug(window),
                "valid_face_ratio": numeric_feature_value(window, "valid_face_ratio"),
                "quality_guard_triggered": 1,
                "quality_guard_reason": reason,
                "base_fsm_state": base_state.value,
                "base_fsm_evidence": base_evidence,
            },
        )

    def _positive_class_index(self) -> int:
        classes = list(getattr(self.model, "classes_", []))
        if 1 in classes:
            return classes.index(1)
        return min(1, max(0, len(classes) - 1))

    def _warmup_result(self, base_state: DrowsinessState, base_evidence: float) -> DecisionResult:
        suppress = bool(getattr(self.config.camera_model, "suppress_warmup_alerts", False)) and base_state != DrowsinessState.ALERT
        state = DrowsinessState.ALERT if suppress else base_state
        evidence = 0.0 if suppress else base_evidence
        alert_cfg = ALERT_CONFIGS[state]
        return DecisionResult(
            state=state,
            evidence=evidence,
            reasons=["CAMERA_MODEL_WARMUP"],
            alert_sound=alert_cfg.sound_type,
            color=alert_cfg.color,
            label="MODEL WARMUP",
            debug={
                "window_frame_count": len(self.rows),
                "min_frames": self.min_frames,
                "feature_set": self.feature_set,
                "base_fsm_state": base_state.value,
                "base_fsm_evidence": base_evidence,
                "warmup_alert_suppression_triggered": int(suppress),
                "warmup_original_state": base_state.value,
                "warmup_original_evidence": base_evidence,
            },
        )

    def _new_feature_fsm(self, fps: float) -> DrowsinessFSM:
        return DrowsinessFSM(
            fps=fps,
            ear_threshold=self.config.thresholds.ear_default,
            mar_threshold=self.config.thresholds.mar,
            pitch_threshold=self.config.thresholds.pitch,
        )

    def _label_prefix(self) -> str:
        return "MODEL"

    def _engine_label(self) -> str:
        return "CAMERA_MODEL"
