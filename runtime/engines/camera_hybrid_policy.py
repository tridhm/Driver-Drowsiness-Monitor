from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fsm import DrowsinessState
from runtime.config import HybridPolicyConfig


STATE_ORDER = [
    DrowsinessState.ALERT,
    DrowsinessState.SUSPICIOUS,
    DrowsinessState.DROWSY,
    DrowsinessState.CRITICAL,
]


@dataclass
class HybridEvidence:
    probability: float
    threshold: float
    ml_only_state: DrowsinessState
    base_fsm_state: DrowsinessState
    base_fsm_evidence: float
    perclos_short: float = 0.0
    perclos_long: float = 0.0
    eyes_closed_consecutive: int = 0
    ear_below_threshold: bool = False
    mar_above_threshold: bool = False
    yawn_count: float = 0.0
    head_drop_count: float = 0.0
    max_eye_closed_duration_sec: float = 0.0
    max_fsm_evidence: float = 0.0


@dataclass
class HybridDecision:
    state: DrowsinessState
    guard: str
    recent_evidence: bool
    clean_streak: int
    severe_streak: int
    elevated_streak: int
    release_clean_streak: int
    hold_decay_streak: int = 0
    dwell_relief_credit: float = 0.0
    severe_evidence: bool = False
    support_model: bool = False
    support_fsm: bool = False
    support_recent: bool = False
    support_severe: bool = False
    support_summary: str = "none"
    state_age_frames: int = 0
    high_state_age_frames: int = 0
    hold_age_frames: int = 0
    hold_source_guard: str = ""
    frames_since_hold_source: int = 0


class HybridDecisionPolicy:
    """Guard learned camera-model risk with current FSM/recent-signal evidence."""

    def __init__(self, config: HybridPolicyConfig | None = None) -> None:
        self.config = config or HybridPolicyConfig()
        self.fsm_safety_min_state = parse_state(self.config.fsm_safety_min_state, DrowsinessState.DROWSY)
        self.fsm_safety_cap_state = (
            parse_state(self.config.fsm_safety_cap_state, DrowsinessState.CRITICAL)
            if str(self.config.fsm_safety_cap_state).strip()
            else None
        )
        self.state = DrowsinessState.ALERT
        self.clean_streak = 0
        self.severe_streak = 0
        self.elevated_streak = 0
        self.release_clean_streak = 0
        self.hold_decay_streak = 0
        self.dwell_relief_credit = 0.0
        self.state_age_frames = 0
        self.high_state_age_frames = 0
        self.hold_age_frames = 0
        self.hold_source_guard = ""
        self.frames_since_hold_source = 0
        self._last_output_state = self.state
        self._support_model = False
        self._support_fsm = False
        self._support_recent = False
        self._support_severe = False

    def update(self, evidence: HybridEvidence) -> HybridDecision:
        was_sleepy_state = self._state_index(self.state) >= self._state_index(DrowsinessState.DROWSY)
        predicted_sleepy = evidence.probability >= evidence.threshold
        clean = self._is_clean(evidence)
        recent = self._has_recent_evidence(evidence)
        severe = self._has_severe_evidence(evidence)
        release_candidate = self._is_low_probability_release_candidate(evidence, predicted_sleepy)
        hold_decay_candidate = self._is_hold_decay_candidate(evidence, predicted_sleepy, severe)
        dwell_relief_gain = self._dwell_relief_gain(evidence, predicted_sleepy, severe)
        self._set_support_context(
            predicted_sleepy=predicted_sleepy,
            base_fsm_state=evidence.base_fsm_state,
            recent=recent,
            severe=severe,
        )

        if clean:
            self.clean_streak += 1
        else:
            self.clean_streak = 0

        if severe:
            self.severe_streak += 1
        else:
            self.severe_streak = 0

        if predicted_sleepy and recent:
            self.elevated_streak += 1
        else:
            self.elevated_streak = 0

        if release_candidate:
            self.release_clean_streak += 1
        else:
            self.release_clean_streak = 0

        if hold_decay_candidate:
            self.hold_decay_streak += 1
        else:
            self.hold_decay_streak = 0

        if dwell_relief_gain is not None:
            self.dwell_relief_credit += dwell_relief_gain
        elif getattr(self.config.dwell_relief, "enabled", False) and was_sleepy_state and not predicted_sleepy and not severe:
            self.dwell_relief_credit = max(0.0, self.dwell_relief_credit - float(self.config.dwell_relief.decay))
        else:
            self.dwell_relief_credit = 0.0

        if clean:
            if predicted_sleepy:
                self.state = self._lower_to_at_most(DrowsinessState.SUSPICIOUS)
            else:
                self.state = DrowsinessState.ALERT
            return self._decision("clean_cap", recent)

        if self._state_index(evidence.base_fsm_state) >= self._state_index(self.fsm_safety_min_state) and not predicted_sleepy:
            self.state = self._apply_optional_cap(evidence.base_fsm_state, self.fsm_safety_cap_state)
            return self._decision("fsm_safety", recent)

        if (
            predicted_sleepy
            and evidence.probability >= float(self.config.critical_probability_threshold)
            and self.severe_streak >= int(self.config.critical_severe_streak)
        ):
            self.state = DrowsinessState.CRITICAL
            return self._decision("critical_sustained", recent)

        if predicted_sleepy and recent:
            if self._state_index(evidence.base_fsm_state) >= self._state_index(DrowsinessState.DROWSY):
                self.state = DrowsinessState.DROWSY
                return self._decision("model_plus_fsm", recent)
            if self.elevated_streak >= int(self.config.elevated_streak_threshold):
                recent_floor = getattr(self.config, "model_plus_recent_drowsy_probability_min", None)
                recent_eye_cap = getattr(self.config, "model_plus_recent_max_eye_closed_duration_sec", None)
                if (
                    recent_floor is not None
                    and evidence.probability < float(recent_floor)
                    and (
                        recent_eye_cap is None
                        or evidence.max_eye_closed_duration_sec <= float(recent_eye_cap)
                    )
                    and self._state_index(self.state) < self._state_index(DrowsinessState.DROWSY)
                ):
                    self.state = self._raise_to_at_least(DrowsinessState.SUSPICIOUS)
                    return self._decision("model_recent_cap", recent)
                self.state = self._raise_to_at_least(DrowsinessState.DROWSY)
                return self._decision("model_plus_recent", recent)
            self.state = self._raise_to_at_least(DrowsinessState.SUSPICIOUS)
            return self._decision("model_caution", recent)

        if self._state_index(evidence.base_fsm_state) > self._state_index(self.state):
            self.state = evidence.base_fsm_state
            return self._decision("fsm_elevated", recent)

        if self._should_low_probability_release():
            self.state = self._apply_low_probability_release_target()
            return self._decision("low_probability_release", recent)

        if self._should_hold_decay_release():
            self.state = self._apply_hold_decay_target()
            return self._decision("hold_decay_release", recent)

        if self._should_dwell_relief_release():
            self.state = self._apply_dwell_relief_target()
            self.dwell_relief_credit = 0.0
            return self._decision("dwell_relief_release", recent)

        if self.clean_streak >= int(self.config.recovery_clean_streak):
            self.state = self._downgrade_one()
            return self._decision("recovery", recent)

        return self._decision("hold", recent)

    def reset(self) -> None:
        self.state = DrowsinessState.ALERT
        self.clean_streak = 0
        self.severe_streak = 0
        self.elevated_streak = 0
        self.release_clean_streak = 0
        self.hold_decay_streak = 0
        self.dwell_relief_credit = 0.0
        self.state_age_frames = 0
        self.high_state_age_frames = 0
        self.hold_age_frames = 0
        self.hold_source_guard = ""
        self.frames_since_hold_source = 0
        self._last_output_state = self.state
        self._support_model = False
        self._support_fsm = False
        self._support_recent = False
        self._support_severe = False

    def _decision(self, guard: str, recent: bool) -> HybridDecision:
        self._refresh_provenance_diagnostics(guard)
        return HybridDecision(
            state=self.state,
            guard=guard,
            recent_evidence=recent,
            clean_streak=self.clean_streak,
            severe_streak=self.severe_streak,
            elevated_streak=self.elevated_streak,
            release_clean_streak=self.release_clean_streak,
            hold_decay_streak=self.hold_decay_streak,
            dwell_relief_credit=self.dwell_relief_credit,
            severe_evidence=self._support_severe,
            support_model=self._support_model,
            support_fsm=self._support_fsm,
            support_recent=self._support_recent,
            support_severe=self._support_severe,
            support_summary=self._support_summary(),
            state_age_frames=self.state_age_frames,
            high_state_age_frames=self.high_state_age_frames,
            hold_age_frames=self.hold_age_frames,
            hold_source_guard=self.hold_source_guard,
            frames_since_hold_source=self.frames_since_hold_source,
        )

    @staticmethod
    def _state_index(state: DrowsinessState) -> int:
        return STATE_ORDER.index(state)

    def _set_support_context(
        self,
        *,
        predicted_sleepy: bool,
        base_fsm_state: DrowsinessState,
        recent: bool,
        severe: bool,
    ) -> None:
        self._support_model = bool(predicted_sleepy)
        self._support_fsm = self._state_index(base_fsm_state) >= self._state_index(DrowsinessState.DROWSY)
        self._support_recent = bool(recent)
        self._support_severe = bool(severe)

    def _support_summary(self) -> str:
        labels: list[str] = []
        if self._support_model:
            labels.append("model")
        if self._support_fsm:
            labels.append("fsm")
        if self._support_recent:
            labels.append("recent")
        if self._support_severe:
            labels.append("severe")
        return "+".join(labels) if labels else "none"

    def _refresh_provenance_diagnostics(self, guard: str) -> None:
        was_high = self._state_index(self._last_output_state) >= self._state_index(DrowsinessState.DROWSY)
        is_high = self._state_index(self.state) >= self._state_index(DrowsinessState.DROWSY)

        if self.state == self._last_output_state:
            self.state_age_frames += 1
        else:
            self.state_age_frames = 1

        if is_high:
            self.high_state_age_frames = self.high_state_age_frames + 1 if was_high else 1
        else:
            self.high_state_age_frames = 0
            self.hold_source_guard = ""
            self.frames_since_hold_source = 0

        if guard == "hold":
            self.hold_age_frames += 1
        else:
            self.hold_age_frames = 0

        if is_high:
            if guard != "hold":
                self.hold_source_guard = guard
                self.frames_since_hold_source = 0
            elif self.hold_source_guard:
                self.frames_since_hold_source += 1
            else:
                self.frames_since_hold_source = 0

        self._last_output_state = self.state

    def _raise_to_at_least(self, minimum: DrowsinessState) -> DrowsinessState:
        return STATE_ORDER[max(self._state_index(self.state), self._state_index(minimum))]

    def _lower_to_at_most(self, maximum: DrowsinessState) -> DrowsinessState:
        return STATE_ORDER[min(self._state_index(self.state), self._state_index(maximum))]

    def _apply_optional_cap(self, state: DrowsinessState, maximum: DrowsinessState | None) -> DrowsinessState:
        if maximum is None:
            return state
        return STATE_ORDER[min(self._state_index(state), self._state_index(maximum))]

    def _downgrade_one(self) -> DrowsinessState:
        return STATE_ORDER[max(0, self._state_index(self.state) - 1)]

    def _should_low_probability_release(self) -> bool:
        cfg = self.config.low_probability_release
        return (
            getattr(cfg, "enabled", False)
            and self._state_index(self.state) >= self._state_index(DrowsinessState.DROWSY)
            and self.release_clean_streak >= int(cfg.clean_streak)
        )

    def _apply_low_probability_release_target(self) -> DrowsinessState:
        target_state = parse_state(
            getattr(self.config.low_probability_release, "target_state", DrowsinessState.SUSPICIOUS.value),
            DrowsinessState.SUSPICIOUS,
        )
        return self._lower_to_at_most(target_state)

    def _should_hold_decay_release(self) -> bool:
        cfg = self.config.hold_decay
        return (
            getattr(cfg, "enabled", False)
            and self._state_index(self.state) >= self._state_index(DrowsinessState.DROWSY)
            and self.hold_decay_streak >= int(cfg.hold_streak)
        )

    def _apply_hold_decay_target(self) -> DrowsinessState:
        target_state = parse_state(
            getattr(self.config.hold_decay, "target_state", DrowsinessState.SUSPICIOUS.value),
            DrowsinessState.SUSPICIOUS,
        )
        return self._lower_to_at_most(target_state)

    def _should_dwell_relief_release(self) -> bool:
        cfg = self.config.dwell_relief
        return (
            getattr(cfg, "enabled", False)
            and self._state_index(self.state) >= self._state_index(DrowsinessState.DROWSY)
            and self.dwell_relief_credit >= float(cfg.min_credit)
        )

    def _apply_dwell_relief_target(self) -> DrowsinessState:
        target_state = parse_state(
            getattr(self.config.dwell_relief, "target_state", DrowsinessState.SUSPICIOUS.value),
            DrowsinessState.SUSPICIOUS,
        )
        return self._lower_to_at_most(target_state)

    def _is_clean(self, evidence: HybridEvidence) -> bool:
        return (
            evidence.base_fsm_state == DrowsinessState.ALERT
            and evidence.base_fsm_evidence <= float(self.config.clean_max_base_fsm_evidence)
            and evidence.perclos_short <= float(self.config.clean_max_perclos_short)
            and not evidence.ear_below_threshold
            and not evidence.mar_above_threshold
            and evidence.eyes_closed_consecutive <= int(self.config.clean_max_eyes_closed_consecutive)
            and evidence.yawn_count <= float(self.config.clean_max_yawn_count)
            and evidence.head_drop_count <= float(self.config.clean_max_head_drop_count)
            and evidence.max_eye_closed_duration_sec <= float(self.config.clean_max_eye_closed_duration_sec)
        )

    def _has_recent_evidence(self, evidence: HybridEvidence) -> bool:
        return (
            evidence.base_fsm_state != DrowsinessState.ALERT
            or evidence.base_fsm_evidence >= float(self.config.recent_min_base_fsm_evidence)
            or evidence.max_fsm_evidence >= float(self.config.recent_min_max_fsm_evidence)
            or evidence.perclos_short >= float(self.config.recent_min_perclos_short)
            or evidence.eyes_closed_consecutive >= int(self.config.recent_min_eyes_closed_consecutive)
            or evidence.ear_below_threshold
            or evidence.mar_above_threshold
            or evidence.yawn_count > 0
            or evidence.head_drop_count > 0
            or evidence.max_eye_closed_duration_sec >= float(self.config.recent_min_max_eye_closed_duration_sec)
        )

    def _has_severe_evidence(self, evidence: HybridEvidence) -> bool:
        return (
            evidence.base_fsm_state in {DrowsinessState.DROWSY, DrowsinessState.CRITICAL}
            or evidence.perclos_short >= float(self.config.severe_min_perclos_short)
            or evidence.eyes_closed_consecutive >= int(self.config.severe_min_eyes_closed_consecutive)
            or evidence.max_eye_closed_duration_sec >= float(self.config.severe_min_max_eye_closed_duration_sec)
            or evidence.max_fsm_evidence >= float(self.config.severe_min_max_fsm_evidence)
            or (
                evidence.base_fsm_evidence >= float(self.config.severe_combined_min_base_fsm_evidence)
                and evidence.perclos_long >= float(self.config.severe_combined_min_perclos_long)
            )
        )

    def _is_low_probability_release_candidate(self, evidence: HybridEvidence, predicted_sleepy: bool) -> bool:
        cfg = self.config.low_probability_release
        max_perclos_long = getattr(cfg, "max_perclos_long", None)
        max_max_eye_closed_duration_sec = getattr(cfg, "max_max_eye_closed_duration_sec", None)
        max_max_fsm_evidence = getattr(cfg, "max_max_fsm_evidence", None)
        return (
            getattr(cfg, "enabled", False)
            and not predicted_sleepy
            and evidence.probability <= float(cfg.probability_max)
            and evidence.base_fsm_state == DrowsinessState.ALERT
            and evidence.base_fsm_evidence <= float(cfg.max_base_fsm_evidence)
            and evidence.perclos_short <= float(cfg.max_perclos_short)
            and (max_perclos_long is None or evidence.perclos_long <= float(max_perclos_long))
            and not evidence.ear_below_threshold
            and not evidence.mar_above_threshold
            and evidence.eyes_closed_consecutive <= int(cfg.max_eyes_closed_consecutive)
            and evidence.yawn_count <= float(cfg.max_yawn_count)
            and evidence.head_drop_count >= float(getattr(cfg, "min_head_drop_count", 0.0))
            and evidence.head_drop_count <= float(cfg.max_head_drop_count)
            and (
                max_max_eye_closed_duration_sec is None
                or evidence.max_eye_closed_duration_sec <= float(max_max_eye_closed_duration_sec)
            )
            and (max_max_fsm_evidence is None or evidence.max_fsm_evidence <= float(max_max_fsm_evidence))
        )

    def _is_hold_decay_candidate(
        self,
        evidence: HybridEvidence,
        predicted_sleepy: bool,
        severe: bool,
    ) -> bool:
        cfg = self.config.hold_decay
        max_perclos_long = getattr(cfg, "max_perclos_long", None)
        max_max_eye_closed_duration_sec = getattr(cfg, "max_max_eye_closed_duration_sec", None)
        max_max_fsm_evidence = getattr(cfg, "max_max_fsm_evidence", None)
        max_base_state = parse_state(getattr(cfg, "max_base_state", DrowsinessState.SUSPICIOUS.value), DrowsinessState.SUSPICIOUS)
        return (
            getattr(cfg, "enabled", False)
            and not predicted_sleepy
            and not severe
            and evidence.probability <= float(cfg.probability_max)
            and self._state_index(self.state) >= self._state_index(DrowsinessState.DROWSY)
            and self._state_index(evidence.base_fsm_state) <= self._state_index(max_base_state)
            and evidence.base_fsm_evidence <= float(cfg.max_base_fsm_evidence)
            and evidence.perclos_short <= float(cfg.max_perclos_short)
            and (max_perclos_long is None or evidence.perclos_long <= float(max_perclos_long))
            and evidence.eyes_closed_consecutive <= int(cfg.max_eyes_closed_consecutive)
            and evidence.yawn_count <= float(cfg.max_yawn_count)
            and evidence.head_drop_count <= float(cfg.max_head_drop_count)
            and (
                max_max_eye_closed_duration_sec is None
                or evidence.max_eye_closed_duration_sec <= float(max_max_eye_closed_duration_sec)
            )
            and (max_max_fsm_evidence is None or evidence.max_fsm_evidence <= float(max_max_fsm_evidence))
        )

    def _dwell_relief_gain(
        self,
        evidence: HybridEvidence,
        predicted_sleepy: bool,
        severe: bool,
    ) -> float | None:
        cfg = self.config.dwell_relief
        max_perclos_long = getattr(cfg, "max_perclos_long", None)
        max_max_eye_closed_duration_sec = getattr(cfg, "max_max_eye_closed_duration_sec", None)
        max_max_fsm_evidence = getattr(cfg, "max_max_fsm_evidence", None)
        max_base_state = parse_state(getattr(cfg, "max_base_state", DrowsinessState.SUSPICIOUS.value), DrowsinessState.SUSPICIOUS)
        if (
            not getattr(cfg, "enabled", False)
            or predicted_sleepy
            or severe
            or evidence.probability > float(cfg.probability_max)
            or self._state_index(self.state) < self._state_index(DrowsinessState.DROWSY)
            or self._state_index(evidence.base_fsm_state) > self._state_index(max_base_state)
            or evidence.base_fsm_evidence > float(cfg.max_base_fsm_evidence)
            or evidence.perclos_short > float(cfg.max_perclos_short)
            or (max_perclos_long is not None and evidence.perclos_long > float(max_perclos_long))
            or evidence.eyes_closed_consecutive > int(cfg.max_eyes_closed_consecutive)
            or evidence.yawn_count > float(cfg.max_yawn_count)
            or evidence.head_drop_count > float(cfg.max_head_drop_count)
            or (
                max_max_eye_closed_duration_sec is not None
                and evidence.max_eye_closed_duration_sec > float(max_max_eye_closed_duration_sec)
            )
            or (max_max_fsm_evidence is not None and evidence.max_fsm_evidence > float(max_max_fsm_evidence))
        ):
            return None
        if evidence.base_fsm_state == DrowsinessState.ALERT:
            return float(cfg.alert_gain)
        if evidence.base_fsm_state == DrowsinessState.SUSPICIOUS:
            return float(cfg.suspicious_gain)
        return None


def parse_state(value: Any, default: DrowsinessState = DrowsinessState.ALERT) -> DrowsinessState:
    text = str(value).strip().upper()
    for state in STATE_ORDER:
        if state.value == text:
            return state
    return default
