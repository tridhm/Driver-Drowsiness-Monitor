from __future__ import annotations

from fsm import ALERT_CONFIGS, DrowsinessFSM, DrowsinessSignals
from runtime.config import RuntimeConfig
from runtime.contracts import DecisionResult, EngineContext
from runtime.engines.base import DecisionEngine


class FSMDecisionEngine(DecisionEngine):
    name = "fsm"

    def __init__(self, config: RuntimeConfig):
        self.config = config
        fsm_cfg = config.fsm
        self.fsm = DrowsinessFSM(
            fps=config.runtime.fps,
            ear_threshold=config.thresholds.ear_default,
            mar_threshold=config.thresholds.mar,
            pitch_threshold=config.thresholds.pitch,
            perclos_threshold=fsm_cfg.perclos_threshold,
            pitch_velocity_evidence_threshold=fsm_cfg.pitch_velocity_evidence_threshold,
            yawn_frequency_threshold=fsm_cfg.yawn_frequency_threshold,
            blink_frequency_threshold=fsm_cfg.blink_frequency_threshold,
            hysteresis_alert_to_suspicious=fsm_cfg.hysteresis_alert_to_suspicious,
            hysteresis_suspicious_to_alert=fsm_cfg.hysteresis_suspicious_to_alert,
            hysteresis_suspicious_to_drowsy=fsm_cfg.hysteresis_suspicious_to_drowsy,
            hysteresis_drowsy_to_suspicious=fsm_cfg.hysteresis_drowsy_to_suspicious,
            hysteresis_drowsy_to_critical=fsm_cfg.hysteresis_drowsy_to_critical,
            hysteresis_critical_to_drowsy=fsm_cfg.hysteresis_critical_to_drowsy,
            extreme_critical_perclos_short_drowsy=fsm_cfg.extreme_critical_perclos_short_drowsy,
            extreme_critical_perclos_long=fsm_cfg.extreme_critical_perclos_long,
            extreme_drowsy_perclos_short=fsm_cfg.extreme_drowsy_perclos_short,
            extreme_drowsy_perclos_long=fsm_cfg.extreme_drowsy_perclos_long,
            extreme_drowsy_ear=fsm_cfg.extreme_drowsy_ear,
            extreme_drowsy_eye_closed_seconds=fsm_cfg.extreme_drowsy_eye_closed_seconds,
            extreme_suspicious_perclos_long=fsm_cfg.extreme_suspicious_perclos_long,
            extreme_suspicious_pitch_velocity=fsm_cfg.extreme_suspicious_pitch_velocity,
            frames_to_suspicious_seconds=fsm_cfg.frames_to_suspicious_seconds,
            frames_to_recovery_seconds=fsm_cfg.frames_to_recovery_seconds,
            sustained_evidence_seconds=fsm_cfg.sustained_evidence_seconds,
            min_dwell_alert_seconds=fsm_cfg.min_dwell_alert_seconds,
            min_dwell_suspicious_seconds=fsm_cfg.min_dwell_suspicious_seconds,
            recovery_grace_period_seconds=fsm_cfg.recovery_grace_period_seconds,
            extreme_signal_seconds=fsm_cfg.extreme_signal_seconds,
            recovery_perclos_short_threshold=fsm_cfg.recovery_perclos_short_threshold,
            drowsy_recovery_seconds=fsm_cfg.drowsy_recovery_seconds,
        )

    def initialize(self, context: EngineContext) -> None:
        self.fsm.set_fps(context.fps)

    def update(self, signals: DrowsinessSignals) -> DecisionResult:
        state = self.fsm.update(signals)
        alert_cfg = ALERT_CONFIGS[state]

        reasons: list[str] = []
        if signals.ear_below_threshold:
            reasons.append("EAR_BELOW_THRESHOLD")
        if signals.mar_above_threshold:
            reasons.append("MAR_ABOVE_THRESHOLD")
        if signals.head_nod_detected:
            reasons.append("HEAD_NOD_DETECTED")
        if signals.perclos_short >= self.config.fsm.extreme_drowsy_perclos_short:
            reasons.append("PERCLOS_5S_HIGH")

        return DecisionResult(
            state=state,
            evidence=self.fsm.evidence_score,
            reasons=reasons,
            alert_sound=alert_cfg.sound_type,
            color=alert_cfg.color,
            label=alert_cfg.text,
            debug={
                "perclos": signals.perclos,
                "perclos_short": signals.perclos_short,
            },
        )

    def reset(self) -> None:
        self.fsm.reset()
