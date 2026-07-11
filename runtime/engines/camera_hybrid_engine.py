from __future__ import annotations

from collections import deque
from typing import Any

from fsm import ALERT_CONFIGS, DrowsinessSignals, DrowsinessState
from runtime.contracts import DecisionResult
from runtime.engines.camera_hybrid_policy import HybridDecision, HybridDecisionPolicy, HybridEvidence
from runtime.engines.camera_model_engine import CameraModelDecisionEngine, ProbabilityStateSmoother


class CameraHybridDecisionEngine(CameraModelDecisionEngine):
    """Runtime hybrid: learned camera risk guarded by FSM/recent-signal recovery."""

    name = "camera_hybrid"
    VALID_ALERT_SOUND_OVERRIDES = {"none", "double", "continuous"}

    def __init__(self, config):
        super().__init__(config)
        self.ml_smoother = ProbabilityStateSmoother()
        self.hybrid_policy = HybridDecisionPolicy(config.hybrid_policy)
        self.macroevent_seed_times: deque[float] = deque()

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
        ml_only_state = self.ml_smoother.update(probability)
        predicted_sleepy = probability >= self.threshold
        evidence = HybridEvidence(
            probability=probability,
            threshold=self.threshold,
            ml_only_state=ml_only_state,
            base_fsm_state=base_state,
            base_fsm_evidence=base_evidence,
            perclos_short=float(signals.perclos_short),
            perclos_long=float(signals.perclos),
            eyes_closed_consecutive=int(signals.eyes_closed_consecutive),
            ear_below_threshold=bool(signals.ear_below_threshold),
            mar_above_threshold=bool(signals.mar_above_threshold),
            yawn_count=float(signals.yawn_frequency),
            head_drop_count=1.0 if signals.head_nod_detected else 0.0,
            max_eye_closed_duration_sec=float(window.get("max_eye_closed_duration_sec", 0.0) or 0.0),
            max_fsm_evidence=float(window.get("max_fsm_evidence", base_evidence) or base_evidence),
        )
        decision = self.hybrid_policy.update(evidence)
        adjusted = self._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=predicted_sleepy,
            probability=probability,
            window=window,
            timestamp_sec=timestamp_sec,
            base_fsm_state=base_state,
        )
        state = adjusted["state"]
        predicted_sleepy = bool(adjusted["predicted_sleepy"])
        alert_cfg = ALERT_CONFIGS[state]
        alert_sound, bridge_alert_sound_override = self._resolve_alert_sound(
            default_sound=alert_cfg.sound_type,
            rule_action=str(adjusted["rule_action"]),
        )
        runtime_alert_semantic = self._runtime_alert_semantic(
            state=state,
            alert_sound=alert_sound,
            rule_action=str(adjusted["rule_action"]),
            bridge_alert_sound_override=bridge_alert_sound_override,
        )
        visual_alert_mode = "visual_only" if runtime_alert_semantic == "visual_only_drowsy_bridge" else ""
        label = self._display_label(alert_cfg.text, runtime_alert_semantic)

        reasons = ["CAMERA_HYBRID", str(adjusted["guard"]).upper()]
        if predicted_sleepy:
            reasons.append("MODEL_SLEEPY_PROBABILITY")
        if decision.recent_evidence:
            reasons.append("HYBRID_RECENT_EVIDENCE")
        if adjusted["rule_action"]:
            reasons.append(f"POST_RULE_{str(adjusted['rule_action']).upper()}")
        if runtime_alert_semantic == "visual_only_drowsy_bridge":
            reasons.append("VISUAL_ONLY_DROWSY_BRIDGE")

        return DecisionResult(
            state=state,
            evidence=probability,
            reasons=reasons,
            alert_sound=alert_sound,
            color=alert_cfg.color,
            label=label,
            debug={
                "sleepy_probability": probability,
                "probability_threshold": self.threshold,
                "predicted_sleepy": int(predicted_sleepy),
                "feature_set": self.feature_set,
                "feature_count": len(self.feature_columns),
                "model_feature_columns": list(self.feature_columns),
                "model_feature_vector": self._model_feature_debug(window),
                "base_fsm_state": base_state.value,
                "base_fsm_evidence": base_evidence,
                "ml_only_state": ml_only_state.value,
                "hybrid_guard": adjusted["guard"],
                "hybrid_recent_evidence": decision.recent_evidence,
                "hybrid_support": decision.support_summary,
                "hybrid_support_model": int(decision.support_model),
                "hybrid_support_fsm": int(decision.support_fsm),
                "hybrid_support_recent": int(decision.support_recent),
                "hybrid_support_severe": int(decision.support_severe),
                "hybrid_clean_streak": decision.clean_streak,
                "hybrid_severe_streak": decision.severe_streak,
                "hybrid_elevated_streak": decision.elevated_streak,
                "hybrid_release_clean_streak": decision.release_clean_streak,
                "hybrid_hold_decay_streak": decision.hold_decay_streak,
                "hybrid_dwell_relief_credit": decision.dwell_relief_credit,
                "hybrid_state_age_frames": decision.state_age_frames,
                "hybrid_high_state_age_frames": decision.high_state_age_frames,
                "hybrid_hold_age_frames": decision.hold_age_frames,
                "hybrid_hold_source_guard": decision.hold_source_guard,
                "hybrid_frames_since_hold_source": decision.frames_since_hold_source,
                "hybrid_rule_action": adjusted["rule_action"],
                "subtle_rescue_triggered": int(adjusted["rule_action"] == "rescue"),
                "fp_suppression_triggered": int(adjusted["rule_action"] == "suppress"),
                "seeded_macroevent_bridge_triggered": int(adjusted["rule_action"] == "bridge"),
                "seeded_macroevent_bridge_seed_count": adjusted["macroevent_seed_count"],
                "seeded_macroevent_bridge_active_until_sec": adjusted["macroevent_active_until_sec"],
                "seeded_macroevent_bridge_alert_sound_override": bridge_alert_sound_override,
                "seeded_macroevent_bridge_visual_only": int(
                    runtime_alert_semantic == "visual_only_drowsy_bridge"
                ),
                "runtime_alert_semantic": runtime_alert_semantic,
                "visual_alert_mode": visual_alert_mode,
                "low_probability_release_triggered": int(adjusted["guard"] == "low_probability_release"),
                "dwell_relief_triggered": int(adjusted["guard"] == "dwell_relief_release"),
                "hybrid_window_features": self._window_debug_features(window),
            },
        )

    def reset(self) -> None:
        super().reset()
        self.ml_smoother.reset()
        self.hybrid_policy.reset()
        self.macroevent_seed_times.clear()

    def _label_prefix(self) -> str:
        return "HYBRID"

    def _engine_label(self) -> str:
        return "CAMERA_HYBRID"

    @staticmethod
    def _window_debug_features(window: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "mean_ear",
            "min_ear",
            "ear_std",
            "ear_p10",
            "mean_mar",
            "max_mar",
            "perclos_60s",
            "perclos_5s",
            "max_eye_closed_duration_sec",
            "yawn_count",
            "head_drop_count",
            "mean_fsm_evidence",
            "max_fsm_evidence",
            "fsm_state_mode",
        ]
        return {key: window.get(key, "") for key in keys if key in window}

    def _apply_post_hybrid_rules(
        self,
        decision: HybridDecision,
        predicted_sleepy: bool,
        probability: float,
        window: dict[str, Any],
        timestamp_sec: float = 0.0,
        base_fsm_state: DrowsinessState = DrowsinessState.ALERT,
    ) -> dict[str, Any]:
        state = decision.state
        guard = decision.guard
        predicted = bool(predicted_sleepy)
        rule_action = ""
        order = str(getattr(self.config.hybrid_policy, "post_rule_order", "rescue_then_suppress")).strip().lower()

        if order == "suppress_then_rescue":
            state, predicted, guard, rule_action = self._apply_fp_suppression_rule(state, predicted, probability, window, guard, rule_action)
            state, predicted, guard, rule_action = self._apply_subtle_rescue_rule(state, predicted, probability, window, guard, rule_action)
        else:
            state, predicted, guard, rule_action = self._apply_subtle_rescue_rule(state, predicted, probability, window, guard, rule_action)
            state, predicted, guard, rule_action = self._apply_fp_suppression_rule(state, predicted, probability, window, guard, rule_action)

        state, predicted, guard, rule_action = self._apply_guarded_severe_rescue_rule(
            state, predicted, window, guard, rule_action
        )
        state, predicted, guard, rule_action = self._apply_seeded_macroevent_bridge_rule(
            timestamp_sec=timestamp_sec,
            state=state,
            predicted_sleepy=predicted,
            probability=probability,
            window=window,
            base_fsm_state=base_fsm_state,
            guard=guard,
            rule_action=rule_action,
        )

        return {
            "state": state,
            "predicted_sleepy": int(predicted),
            "guard": guard,
            "rule_action": rule_action,
            "macroevent_seed_count": len(self.macroevent_seed_times),
            "macroevent_active_until_sec": self._macroevent_active_until_sec(),
        }

    def _apply_subtle_rescue_rule(
        self,
        state: DrowsinessState,
        predicted_sleepy: bool,
        probability: float,
        window: dict[str, Any],
        guard: str,
        rule_action: str,
    ) -> tuple[DrowsinessState, bool, str, str]:
        cfg = self.config.hybrid_policy.subtle_rescue
        if not getattr(cfg, "enabled", False):
            return state, predicted_sleepy, guard, rule_action
        if predicted_sleepy or state != DrowsinessState.SUSPICIOUS:
            return state, predicted_sleepy, guard, rule_action
        if not (float(cfg.probability_low) <= probability <= float(cfg.probability_high)):
            return state, predicted_sleepy, guard, rule_action
        if float(window.get("ear_std", 0.0) or 0.0) < float(cfg.ear_std_min):
            return state, predicted_sleepy, guard, rule_action
        if float(window.get("ear_p10", 0.0) or 0.0) > float(cfg.ear_p10_max):
            return state, predicted_sleepy, guard, rule_action
        if int(round(float(window.get("head_drop_count", 0.0) or 0.0))) > int(cfg.head_drop_max):
            return state, predicted_sleepy, guard, rule_action
        return DrowsinessState.DROWSY, True, "subtle_rescue", "rescue"

    def _apply_fp_suppression_rule(
        self,
        state: DrowsinessState,
        predicted_sleepy: bool,
        probability: float,
        window: dict[str, Any],
        guard: str,
        rule_action: str,
    ) -> tuple[DrowsinessState, bool, str, str]:
        cfg = self.config.hybrid_policy.fp_suppression
        if not getattr(cfg, "enabled", False):
            return state, predicted_sleepy, guard, rule_action
        if not predicted_sleepy or state != DrowsinessState.DROWSY:
            return state, predicted_sleepy, guard, rule_action
        if probability > float(cfg.probability_high):
            return state, predicted_sleepy, guard, rule_action
        if int(round(float(window.get("head_drop_count", 0.0) or 0.0))) < int(cfg.head_drop_min):
            return state, predicted_sleepy, guard, rule_action
        if float(window.get("max_eye_closed_duration_sec", 0.0) or 0.0) > float(cfg.maxeye_max):
            return state, predicted_sleepy, guard, rule_action
        return DrowsinessState.SUSPICIOUS, False, "fp_suppression", "suppress"

    def _apply_guarded_severe_rescue_rule(
        self,
        state: DrowsinessState,
        predicted_sleepy: bool,
        window: dict[str, Any],
        guard: str,
        rule_action: str,
    ) -> tuple[DrowsinessState, bool, str, str]:
        cfg = self.config.hybrid_policy.guarded_severe_rescue
        if not getattr(cfg, "enabled", False):
            return state, predicted_sleepy, guard, rule_action
        if rule_action or predicted_sleepy:
            return state, predicted_sleepy, guard, rule_action
        target_state = self._state_from_name(getattr(cfg, "target_state", "DROWSY"), DrowsinessState.DROWSY)
        if self._state_index(state) >= self._state_index(target_state):
            return state, predicted_sleepy, guard, rule_action
        if float(window.get("mean_ear", 1.0) or 1.0) > float(cfg.max_mean_ear):
            return state, predicted_sleepy, guard, rule_action
        if float(window.get("max_fsm_evidence", 0.0) or 0.0) < float(cfg.min_max_fsm_evidence):
            return state, predicted_sleepy, guard, rule_action
        if float(window.get("max_eye_closed_duration_sec", 0.0) or 0.0) < float(cfg.min_max_eye_closed_duration_sec):
            return state, predicted_sleepy, guard, rule_action
        return target_state, True, "guarded_severe_rescue", "guarded_rescue"

    def _apply_seeded_macroevent_bridge_rule(
        self,
        timestamp_sec: float,
        state: DrowsinessState,
        predicted_sleepy: bool,
        probability: float,
        window: dict[str, Any],
        base_fsm_state: DrowsinessState,
        guard: str,
        rule_action: str,
    ) -> tuple[DrowsinessState, bool, str, str]:
        cfg = self.config.hybrid_policy.seeded_macroevent_bridge
        if not getattr(cfg, "enabled", False):
            self.macroevent_seed_times.clear()
            return state, predicted_sleepy, guard, rule_action

        self._expire_macroevent_seeds(timestamp_sec)
        if self._matches_macroevent_seed(predicted_sleepy, probability, window, base_fsm_state):
            self.macroevent_seed_times.append(float(timestamp_sec))
            self._expire_macroevent_seeds(timestamp_sec)

        if rule_action or predicted_sleepy:
            return state, predicted_sleepy, guard, rule_action
        if self._state_index(state) >= self._state_index(DrowsinessState.DROWSY):
            return state, predicted_sleepy, guard, rule_action
        if self._state_index(base_fsm_state) >= self._state_index(DrowsinessState.DROWSY):
            return state, predicted_sleepy, guard, rule_action
        if len(self.macroevent_seed_times) < int(cfg.min_seed_hits):
            return state, predicted_sleepy, guard, rule_action
        if timestamp_sec > self._macroevent_active_until_sec():
            return state, predicted_sleepy, guard, rule_action
        if not self._matches_macroevent_bridge_candidate(window):
            return state, predicted_sleepy, guard, rule_action
        target_state = self._state_from_name(getattr(cfg, "target_state", "DROWSY"), DrowsinessState.DROWSY)
        if self._state_index(state) >= self._state_index(target_state):
            return state, predicted_sleepy, guard, rule_action
        return (
            target_state,
            self._state_index(target_state) >= self._state_index(DrowsinessState.DROWSY),
            "seeded_macroevent_bridge",
            "bridge",
        )

    def _expire_macroevent_seeds(self, timestamp_sec: float) -> None:
        cfg = self.config.hybrid_policy.seeded_macroevent_bridge
        active_seconds = max(0.0, float(getattr(cfg, "active_seconds", 0.0)))
        while self.macroevent_seed_times and timestamp_sec - self.macroevent_seed_times[0] > active_seconds:
            self.macroevent_seed_times.popleft()

    def _macroevent_active_until_sec(self) -> float:
        cfg = self.config.hybrid_policy.seeded_macroevent_bridge
        if not self.macroevent_seed_times:
            return 0.0
        return float(self.macroevent_seed_times[-1]) + max(0.0, float(getattr(cfg, "active_seconds", 0.0)))

    def _matches_macroevent_seed(
        self,
        predicted_sleepy: bool,
        probability: float,
        window: dict[str, Any],
        base_fsm_state: DrowsinessState,
    ) -> bool:
        cfg = self.config.hybrid_policy.seeded_macroevent_bridge
        return (
            not predicted_sleepy
            and self._state_index(base_fsm_state) < self._state_index(DrowsinessState.DROWSY)
            and probability >= float(cfg.probability_min)
            and float(window.get("head_drop_count", 0.0) or 0.0) >= float(cfg.head_drop_min)
            and float(window.get("max_mar", 0.0) or 0.0) >= float(cfg.max_mar_min)
        )

    def _matches_macroevent_bridge_candidate(self, window: dict[str, Any]) -> bool:
        cfg = self.config.hybrid_policy.seeded_macroevent_bridge
        min_max_fsm_evidence = getattr(cfg, "bridge_min_max_fsm_evidence", None)
        return (
            float(window.get("perclos_60s", 0.0) or 0.0) <= float(cfg.bridge_perclos_max)
            and float(window.get("max_eye_closed_duration_sec", 0.0) or 0.0) <= float(cfg.bridge_maxeye_max)
            and float(window.get("mean_mar", 0.0) or 0.0) >= float(cfg.bridge_mean_mar_min)
            and (
                min_max_fsm_evidence is None
                or float(window.get("max_fsm_evidence", 0.0) or 0.0) >= float(min_max_fsm_evidence)
            )
        )

    def _resolve_alert_sound(self, default_sound: str, rule_action: str) -> tuple[str, str]:
        if rule_action != "bridge":
            return default_sound, ""
        override = getattr(self.config.hybrid_policy.seeded_macroevent_bridge, "alert_sound_override", None)
        if override is None:
            return default_sound, ""
        normalized = str(override).strip().lower()
        if normalized in self.VALID_ALERT_SOUND_OVERRIDES:
            return normalized, normalized
        return default_sound, ""

    @staticmethod
    def _runtime_alert_semantic(
        state: DrowsinessState,
        alert_sound: str,
        rule_action: str,
        bridge_alert_sound_override: str,
    ) -> str:
        if (
            rule_action == "bridge"
            and bridge_alert_sound_override == "none"
            and alert_sound == "none"
            and state == DrowsinessState.DROWSY
        ):
            return "visual_only_drowsy_bridge"
        return "standard"

    @staticmethod
    def _display_label(alert_text: str, runtime_alert_semantic: str) -> str:
        if runtime_alert_semantic == "visual_only_drowsy_bridge":
            return "HYBRID VISUAL DROWSY"
        return f"HYBRID {alert_text}"

    @staticmethod
    def _state_from_name(name: str, default: DrowsinessState) -> DrowsinessState:
        try:
            return DrowsinessState[str(name).strip().upper()]
        except (KeyError, TypeError):
            return default

    @staticmethod
    def _state_index(state: DrowsinessState) -> int:
        order = [
            DrowsinessState.ALERT,
            DrowsinessState.SUSPICIOUS,
            DrowsinessState.DROWSY,
            DrowsinessState.CRITICAL,
        ]
        return order.index(state)
