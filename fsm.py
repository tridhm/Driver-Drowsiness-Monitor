"""
Finite State Machine (FSM) for Drowsiness Detection
Thesis §5.3–5.4: Multi-state decision engine with hysteresis to prevent alert oscillation.

States: ALERT → SUSPICIOUS → DROWSY → CRITICAL
Hysteresis: Different thresholds for entering vs. leaving each state.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class DrowsinessState(Enum):
    """Four-level drowsiness state as defined in thesis §5.3."""
    ALERT = "ALERT"              # TỈNH TÁO — driver is awake
    SUSPICIOUS = "SUSPICIOUS"    # NGHI NGỜ — early signs detected
    DROWSY = "DROWSY"            # BUỒN NGỦ — clear drowsiness
    CRITICAL = "CRITICAL"        # NGUY KỊCH — sustained danger


@dataclass
class DrowsinessSignals:
    """Container for all drowsiness signals from the perception pipeline."""
    # Smoothed signal values
    ear: float = 0.25
    mar: float = 0.3
    pitch: float = 0.0
    pitch_velocity: float = 0.0  # degrees/frame - rapid drop = head nod

    # Temporal features
    perclos: float = 0.0  # 60s window
    perclos_short: float = 0.0  # 5s window for rapid escalation
    yawn_frequency: int = 0
    blink_frequency: int = 0

    # Event flags
    gaze_stable: bool = False
    head_nod_detected: bool = False
    eyes_closed_consecutive: int = 0  # frames

    # Raw detection flags (for FSM evidence scoring)
    face_detected: bool = True
    ear_below_threshold: bool = False
    mar_above_threshold: bool = False
    pitch_above_threshold: bool = False


@dataclass
class AlertConfig:
    """Configuration for how to alert at each state level."""
    color: tuple  # BGR color for on-screen text
    text: str
    sound_type: str  # "none", "double", "continuous"
    font_scale: float = 1.0
    thickness: int = 3


# State-specific alert configurations
ALERT_CONFIGS = {
    DrowsinessState.ALERT: AlertConfig(
        color=(0, 255, 0),  # Green
        text="ALERT",
        sound_type="none",
    ),
    DrowsinessState.SUSPICIOUS: AlertConfig(
        color=(0, 255, 255),  # Yellow/Cyan
        text="CAUTION",
        sound_type="none",  # Visual only — no sound for early warning
    ),
    DrowsinessState.DROWSY: AlertConfig(
        color=(0, 165, 255),  # Orange
        text="DROWSY",
        sound_type="double",
    ),
    DrowsinessState.CRITICAL: AlertConfig(
        color=(0, 0, 255),  # Red
        text="CRITICAL — STOP",
        sound_type="continuous",
        font_scale=1.2,
        thickness=4,
    ),
}


class DrowsinessFSM:
    """
    Finite State Machine for drowsiness level decision-making.

    Implements the 4-state model from thesis §5.3 with hysteresis (§5.4)
    to prevent state oscillation when signals hover near thresholds.
    """

    def __init__(
        self,
        fps: float = 30.0,
        ear_threshold: float = 0.23,
        mar_threshold: float = 0.6,
        pitch_threshold: float = 20.0,
        perclos_threshold: float = 0.2,
        hysteresis_margin: float = 0.15,
        pitch_velocity_evidence_threshold: float = 5.0,
        yawn_frequency_threshold: int = 3,
        blink_frequency_threshold: int = 10,
        hysteresis_alert_to_suspicious: float = 0.45,
        hysteresis_suspicious_to_alert: float = 0.25,
        hysteresis_suspicious_to_drowsy: float = 0.60,
        hysteresis_drowsy_to_suspicious: float = 0.40,
        hysteresis_drowsy_to_critical: float = 0.75,
        hysteresis_critical_to_drowsy: float = 0.55,
        extreme_critical_perclos_short_drowsy: float = 0.60,
        extreme_critical_perclos_long: float = 0.70,
        extreme_drowsy_perclos_short: float = 0.60,
        extreme_drowsy_perclos_long: float = 0.50,
        extreme_drowsy_ear: float = 0.10,
        extreme_drowsy_eye_closed_seconds: float = 1.0,
        extreme_suspicious_perclos_long: float = 0.35,
        extreme_suspicious_pitch_velocity: float = 10.0,
        frames_to_suspicious_seconds: float = 0.5,
        frames_to_recovery_seconds: float = 1.0,
        sustained_evidence_seconds: float = 1.0,
        min_dwell_alert_seconds: float = 5.0,
        min_dwell_suspicious_seconds: float = 3.0,
        recovery_grace_period_seconds: float = 10.0,
        extreme_signal_seconds: float = 1.0,
        recovery_perclos_short_threshold: float = 0.40,
        drowsy_recovery_seconds: float = 10.0,
    ):
        """
        Args:
            fps: Measured frames per second for time-based calculations.
            ear_threshold: EAR value below which eyes are considered closed.
            mar_threshold: MAR value above which mouth is considered open (yawning).
            pitch_threshold: Relative pitch angle (degrees) for head nodding.
            perclos_threshold: PERCLOS fraction indicating significant drowsiness.
            hysteresis_margin: Margin for hysteresis (τ_high - τ_low).
        """
        self.fps = fps
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.pitch_threshold = pitch_threshold
        self.perclos_threshold = perclos_threshold
        self.hysteresis_margin = hysteresis_margin
        self.pitch_velocity_evidence_threshold = pitch_velocity_evidence_threshold
        self.yawn_frequency_threshold = yawn_frequency_threshold
        self.blink_frequency_threshold = blink_frequency_threshold

        self.hysteresis_alert_to_suspicious = hysteresis_alert_to_suspicious
        self.hysteresis_suspicious_to_alert = hysteresis_suspicious_to_alert
        self.hysteresis_suspicious_to_drowsy = hysteresis_suspicious_to_drowsy
        self.hysteresis_drowsy_to_suspicious = hysteresis_drowsy_to_suspicious
        self.hysteresis_drowsy_to_critical = hysteresis_drowsy_to_critical
        self.hysteresis_critical_to_drowsy = hysteresis_critical_to_drowsy

        self.extreme_critical_perclos_short_drowsy = extreme_critical_perclos_short_drowsy
        self.extreme_critical_perclos_long = extreme_critical_perclos_long
        self.extreme_drowsy_perclos_short = extreme_drowsy_perclos_short
        self.extreme_drowsy_perclos_long = extreme_drowsy_perclos_long
        self.extreme_drowsy_ear = extreme_drowsy_ear
        self.extreme_drowsy_eye_closed_seconds = extreme_drowsy_eye_closed_seconds
        self.extreme_suspicious_perclos_long = extreme_suspicious_perclos_long
        self.extreme_suspicious_pitch_velocity = extreme_suspicious_pitch_velocity

        self.frames_to_suspicious_seconds = frames_to_suspicious_seconds
        self.frames_to_recovery_seconds = frames_to_recovery_seconds
        self.sustained_evidence_seconds = sustained_evidence_seconds
        self.min_dwell_alert_seconds = min_dwell_alert_seconds
        self.min_dwell_suspicious_seconds = min_dwell_suspicious_seconds
        self.recovery_grace_period_seconds = recovery_grace_period_seconds
        self.extreme_signal_seconds = extreme_signal_seconds
        self.recovery_perclos_short_threshold = recovery_perclos_short_threshold
        self.drowsy_recovery_seconds = drowsy_recovery_seconds

        # State tracking
        self.state = DrowsinessState.ALERT
        self.prev_state = DrowsinessState.ALERT

        # Evidence accumulators (for multi-signal transitions)
        self.evidence_score = 0.0  # Normalized [0, 1]
        self.suspicious_frames = 0
        self.drowsy_frames = 0

        # Timing thresholds are derived from FPS so replay/runtime can tune them honestly.
        self.frames_to_suspicious = 0
        self.frames_to_recovery = 0
        self.sustained_evidence_frames = 0
        self.min_dwell_alert = 0
        self.min_dwell_suspicious = 0
        self.recovery_grace_period = 0
        self.extreme_signal_frames = 0
        self.drowsy_recovery_frames = 0

        # Recovery counter (must stay clean for N frames to downgrade)
        self.recovery_counter = 0

        # Sustained evidence counter: evidence must be above threshold for N consecutive frames
        self.sustained_evidence_counter = 0
        self.frames_in_current_state = 0

        self.recovery_grace_counter = 0

        # Extreme signal tracking (for direct escalation)
        self.extreme_signal_counter = 0
        self.last_extreme_state = None
        self.set_fps(fps)

    def set_fps(self, fps: float) -> None:
        self.fps = fps
        self.frames_to_suspicious = self._seconds_to_frames(self.frames_to_suspicious_seconds)
        self.frames_to_recovery = self._seconds_to_frames(self.frames_to_recovery_seconds)
        self.sustained_evidence_frames = self._seconds_to_frames(self.sustained_evidence_seconds)
        self.min_dwell_alert = self._seconds_to_frames(self.min_dwell_alert_seconds)
        self.min_dwell_suspicious = self._seconds_to_frames(self.min_dwell_suspicious_seconds)
        self.recovery_grace_period = self._seconds_to_frames(self.recovery_grace_period_seconds)
        self.extreme_signal_frames = self._seconds_to_frames(self.extreme_signal_seconds)
        self.drowsy_recovery_frames = self._seconds_to_frames(self.drowsy_recovery_seconds)

    def _seconds_to_frames(self, seconds: float) -> int:
        return max(0, int(self.fps * seconds))

    def _compute_evidence(self, signals: DrowsinessSignals) -> float:
        """
        Compute a normalized evidence score [0, 1] from all signals.

        Weighted combination of:
        - Eye closure (EAR + consecutive frames): 40%
        - PERCLOS (cumulative): 25%
        - Head nodding: 15%
        - Yawning: 10%
        - Gaze / blink anomalies: 10%
        """
        score = 0.0

        # 1. Eye closure evidence (40%)
        eye_evidence = 0.0
        if signals.ear_below_threshold:
            eye_evidence += 0.2
        # Bonus for sustained closure
        if signals.eyes_closed_consecutive > self.frames_to_suspicious:
            eye_evidence += 0.2
        score += min(eye_evidence, 0.4)

        # 2. PERCLOS evidence (25%)
        if signals.perclos > self.perclos_threshold:
            perclos_evidence = min(
                (signals.perclos - self.perclos_threshold) / (0.5 - self.perclos_threshold),
                1.0
            )
            score += perclos_evidence * 0.25

        # 3. Head nodding evidence (15%)
        head_evidence = 0.0
        if signals.head_nod_detected:
            head_evidence += 0.10
        # Pitch velocity: rapid head drop = strong nod indicator
        if abs(signals.pitch_velocity) > self.pitch_velocity_evidence_threshold:
            head_evidence += 0.05
        score += min(head_evidence, 0.15)

        # 4. Yawning evidence (10%)
        if signals.yawn_frequency >= self.yawn_frequency_threshold:
            score += 0.10
        elif signals.mar_above_threshold:
            score += 0.05

        # 5. Gaze / blink anomalies (10%)
        # Gaze: eyes open but fixed/unmoving (blank stare) - independent from PERCLOS
        # PERCLOS catches eyes closing, gaze catches eyes open but unresponsive
        if signals.gaze_stable:
            score += 0.10  # Fixed gaze = tunnel vision, regardless of EAR
        elif signals.blink_frequency > self.blink_frequency_threshold:
            score += 0.05

        return min(score, 1.0)

    def _check_extreme_escalation(self, signals: DrowsinessSignals) -> Optional[DrowsinessState]:
        """
        Check if any single signal is extreme enough to force escalation.

        Returns the forced target state, or None if no extreme signal detected.
        Requires 1 second of sustained extreme signal to avoid false triggers.
        Respects the recovery grace period.
        """
        # Don't escalate during grace period
        if self.recovery_grace_counter > 0:
            return None

        # === CRITICAL-level extremes (direct jump to CRITICAL) ===

        # 5s-PERCLOS >= 0.60 = eyes closed 60% of last 5 seconds (already drowsy or worse)
        state_order = [DrowsinessState.ALERT, DrowsinessState.SUSPICIOUS, DrowsinessState.DROWSY, DrowsinessState.CRITICAL]
        if (
            state_order.index(self.state) >= state_order.index(DrowsinessState.DROWSY)
            and signals.perclos_short >= self.extreme_critical_perclos_short_drowsy
        ):
            return DrowsinessState.CRITICAL

        # PERCLOS >= 0.70 = eyes closed 70% of last 60 seconds
        if signals.perclos >= self.extreme_critical_perclos_long:
            return DrowsinessState.CRITICAL

        # === DROWSY-level extremes (direct jump to DROWSY) ===
        # 5s-PERCLOS >= 0.60 = eyes closed 60% of last 5 seconds (rapid escalation)
        # Works from ALERT or SUSPICIOUS state
        if signals.perclos_short >= self.extreme_drowsy_perclos_short:
            return DrowsinessState.DROWSY

        # PERCLOS >= 0.50 = eyes closed half the time
        if signals.perclos >= self.extreme_drowsy_perclos_long:
            return DrowsinessState.DROWSY

        # EAR extremely low (<0.10) for 1+ second = eyes fully shut
        if signals.ear < self.extreme_drowsy_ear and signals.eyes_closed_consecutive >= self._seconds_to_frames(self.extreme_drowsy_eye_closed_seconds):
            return DrowsinessState.DROWSY

        # === SUSPICIOUS-level extremes (direct jump to SUSPICIOUS) ===

        # PERCLOS >= 0.35 = significant drowsiness (only from ALERT)
        if signals.perclos >= self.extreme_suspicious_perclos_long and self.state == DrowsinessState.ALERT:
            return DrowsinessState.SUSPICIOUS

        # Violent head nod (>=10 degrees/frame) = nodding off
        if abs(signals.pitch_velocity) >= self.extreme_suspicious_pitch_velocity and self.state == DrowsinessState.ALERT:
            return DrowsinessState.SUSPICIOUS

        return None

    def _apply_hysteresis(self, evidence: float) -> DrowsinessState:
        """
        Apply hysteresis logic to determine target state.

        Uses different thresholds for ascending vs. descending transitions
        to prevent state oscillation (thesis §5.4).
        """
        current = self.state

        # Ascending thresholds (τ_high — harder to escalate)
        if current == DrowsinessState.ALERT:
            if evidence > self.hysteresis_alert_to_suspicious:
                return DrowsinessState.SUSPICIOUS
        elif current == DrowsinessState.SUSPICIOUS:
            if evidence > self.hysteresis_suspicious_to_drowsy:
                return DrowsinessState.DROWSY
            elif evidence < self.hysteresis_suspicious_to_alert:
                return DrowsinessState.ALERT
        elif current == DrowsinessState.DROWSY:
            if evidence > self.hysteresis_drowsy_to_critical:
                return DrowsinessState.CRITICAL
            elif evidence < self.hysteresis_drowsy_to_suspicious:
                return DrowsinessState.SUSPICIOUS
        elif current == DrowsinessState.CRITICAL:
            if evidence < self.hysteresis_critical_to_drowsy:
                return DrowsinessState.DROWSY

        return current

    def update(self, signals: DrowsinessSignals) -> DrowsinessState:
        """
        Process signals and update FSM state.

        Priority order:
        1. Extreme signal escalation (bypasses normal evidence)
        2. Normal evidence-based escalation
        3. Recovery (downgrade)
        """
        self.prev_state = self.state

        # Compute evidence score (always, for logging/display)
        self.evidence_score = self._compute_evidence(signals)

        # === STEP 1: Check extreme escalation ===
        forced_state = self._check_extreme_escalation(signals)

        # Compare states by their order in the escalation chain
        state_order = [DrowsinessState.ALERT, DrowsinessState.SUSPICIOUS, DrowsinessState.DROWSY, DrowsinessState.CRITICAL]
        if forced_state is not None and state_order.index(forced_state) > state_order.index(self.state):
            # Track sustained extreme signal (1 second required)
            if forced_state == self.last_extreme_state:
                self.extreme_signal_counter += 1
            else:
                self.extreme_signal_counter = 1
                self.last_extreme_state = forced_state

            # Require 1 second of sustained extreme signal
            if self.extreme_signal_counter >= self.extreme_signal_frames:
                self.state = forced_state
                self.recovery_counter = 0
                self.sustained_evidence_counter = 0
                self.frames_in_current_state = 0
                self.extreme_signal_counter = 0
                self.last_extreme_state = None
                return self.state
        else:
            # Extreme signal not present or not sustained
            self.extreme_signal_counter = 0
            self.last_extreme_state = None

        # === STEP 2: Normal evidence-based logic ===

        # Track sustained evidence (decaying accumulator — tolerates brief dips)
        escalation_threshold = self.hysteresis_alert_to_suspicious if self.state == DrowsinessState.ALERT else self.hysteresis_suspicious_to_drowsy
        if self.evidence_score > escalation_threshold:
            self.sustained_evidence_counter += 1
        else:
            # Decay instead of hard reset: brief dips don't erase all progress
            # Decay rate of 2 means ~15 frames of strong evidence can recover from 1 dip
            self.sustained_evidence_counter = max(0, self.sustained_evidence_counter - 2)

        # Track frames in current state
        self.frames_in_current_state += 1

        # Apply hysteresis to determine target state
        target_state = self._apply_hysteresis(self.evidence_score)

        # Recovery logic: require sustained clean signal to downgrade
        # Use PERCLOS_5s for faster recovery response
        state_order = [DrowsinessState.ALERT, DrowsinessState.SUSPICIOUS, DrowsinessState.DROWSY, DrowsinessState.CRITICAL]
        if state_order.index(target_state) < state_order.index(self.state):
            # DROWSY->SUSPICIOUS: recover when PERCLOS_5s drops below 0.40 (eyes open for ~2s in 5s window)
            if signals.perclos_short < self.recovery_perclos_short_threshold:
                self.recovery_counter += 1
                recovery_frames_needed = self.drowsy_recovery_frames if self.state == DrowsinessState.DROWSY else self.frames_to_recovery
                if self.recovery_counter >= recovery_frames_needed:
                    self.state = target_state
                    self.recovery_counter = 0
                    self.sustained_evidence_counter = 0
                    self.frames_in_current_state = 0
                    if self.state == DrowsinessState.ALERT:
                        self.recovery_grace_counter = self.recovery_grace_period
            else:
                self.recovery_counter = 0
        else:
            # Escalating or same level — check conditions
            state_order = [DrowsinessState.ALERT, DrowsinessState.SUSPICIOUS, DrowsinessState.DROWSY, DrowsinessState.CRITICAL]
            if state_order.index(target_state) > state_order.index(self.state):
                has_sustained_evidence = self.sustained_evidence_counter >= self.sustained_evidence_frames

                min_dwell = self.min_dwell_alert if self.state == DrowsinessState.ALERT else self.min_dwell_suspicious
                has_min_dwell = self.frames_in_current_state >= min_dwell

                in_grace_period = self.recovery_grace_counter > 0

                if has_sustained_evidence and has_min_dwell and not in_grace_period:
                    self.state = target_state
                    self.recovery_counter = 0
                    self.sustained_evidence_counter = 0
                    self.frames_in_current_state = 0
            else:
                self.state = target_state

        # Decrement grace period counter
        if self.recovery_grace_counter > 0:
            self.recovery_grace_counter -= 1

        return self.state

    def reset(self):
        """Reset FSM to initial state."""
        self.state = DrowsinessState.ALERT
        self.prev_state = DrowsinessState.ALERT
        self.evidence_score = 0.0
        self.recovery_counter = 0
        self.sustained_evidence_counter = 0
        self.frames_in_current_state = 0
        self.recovery_grace_counter = 0
        self.extreme_signal_counter = 0
        self.last_extreme_state = None

    def get_alert_config(self) -> AlertConfig:
        """Get the alert configuration for the current state."""
        return ALERT_CONFIGS[self.state]
