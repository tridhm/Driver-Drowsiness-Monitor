from __future__ import annotations

from fsm import DrowsinessSignals, DrowsinessState
from runtime.config import RuntimeConfig
from runtime.contracts import DecisionResult, EngineContext
from runtime.engines.base import DecisionEngine


class LegacyRuleEngine(DecisionEngine):
    """Bundle-style rule engine isolated from FSM internals."""

    name = "legacy"

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.state = DrowsinessState.ALERT
        self._recovery_counter = 0

    def initialize(self, context: EngineContext) -> None:
        _ = context

    def update(self, signals: DrowsinessSignals) -> DecisionResult:
        score = 0.0
        reasons: list[str] = []

        if signals.eyes_closed_consecutive >= 15:
            score += 0.35
            reasons.append("EYES_CLOSED")

        if signals.perclos_short >= 0.60:
            score += 0.30
            reasons.append("PERCLOS_5S_HIGH")
        elif signals.perclos >= 0.35:
            score += 0.20
            reasons.append("PERCLOS_HIGH")

        if signals.head_nod_detected:
            score += 0.20
            reasons.append("HEAD_NOD")

        if signals.yawn_frequency >= self.config.thresholds.max_yawns_window:
            score += 0.15
            reasons.append("FREQUENT_YAWN")

        if signals.blink_frequency > self.config.thresholds.blink_freq:
            score += 0.10
            reasons.append("RAPID_BLINK")

        if signals.gaze_stable and not signals.ear_below_threshold:
            score += 0.05
            reasons.append("FIXED_GAZE")

        target_state = self._score_to_state(score)
        state_order = [
            DrowsinessState.ALERT,
            DrowsinessState.SUSPICIOUS,
            DrowsinessState.DROWSY,
            DrowsinessState.CRITICAL,
        ]
        target_rank = state_order.index(target_state)
        current_rank = state_order.index(self.state)

        if target_rank < current_rank:
            self._recovery_counter += 1
            if self._recovery_counter >= int(self.config.runtime.fps * 1.0):
                self.state = target_state
                self._recovery_counter = 0
        else:
            self.state = target_state
            self._recovery_counter = 0

        color_map = {
            DrowsinessState.ALERT: (0, 255, 0),
            DrowsinessState.SUSPICIOUS: (0, 255, 255),
            DrowsinessState.DROWSY: (0, 165, 255),
            DrowsinessState.CRITICAL: (0, 0, 255),
        }
        sound_map = {
            DrowsinessState.ALERT: "none",
            DrowsinessState.SUSPICIOUS: "none",
            DrowsinessState.DROWSY: "double",
            DrowsinessState.CRITICAL: "continuous",
        }

        return DecisionResult(
            state=self.state,
            evidence=score,
            reasons=reasons,
            alert_sound=sound_map[self.state],
            color=color_map[self.state],
            label=f"LEGACY {self.state.value}",
            debug={"legacy_score": score},
        )

    def reset(self) -> None:
        self.state = DrowsinessState.ALERT
        self._recovery_counter = 0

    @staticmethod
    def _score_to_state(score: float) -> DrowsinessState:
        if score >= 0.80:
            return DrowsinessState.CRITICAL
        if score >= 0.55:
            return DrowsinessState.DROWSY
        if score >= 0.30:
            return DrowsinessState.SUSPICIOUS
        return DrowsinessState.ALERT
