import tempfile
import unittest
from pathlib import Path

from joblib import dump

from fsm import DrowsinessSignals, DrowsinessState
from runtime.config import default_runtime_config
from runtime.contracts import EngineContext
from runtime.engines.camera_hybrid_policy import HybridDecision, HybridDecisionPolicy, HybridEvidence
from runtime.engines.registry import create_engine


class FixedProbabilityModel:
    classes_ = [0, 1]

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, rows):
        return [[1.0 - self.probability, self.probability] for _row in rows]


class CameraHybridEngineTests(unittest.TestCase):
    def _engine(self, probability: float):
        tmp = tempfile.TemporaryDirectory()
        model_path = Path(tmp.name) / "camera_model.joblib"
        dump(
            {
                "model": FixedProbabilityModel(probability),
                "feature_columns": ["mean_ear", "perclos_60s", "perclos_5s", "fsm_state_mode"],
                "probability_threshold": 0.5,
                "feature_set": "camera",
            },
            model_path,
        )
        config = default_runtime_config()
        config.camera_model.model_path = str(model_path)
        config.camera_model.window_seconds = 0.4
        config.camera_model.min_window_seconds = 0.0
        config.camera_model.min_frames = 2
        engine = create_engine("camera_hybrid", config)
        engine.initialize(EngineContext(fps=10.0))
        self.addCleanup(tmp.cleanup)
        return engine

    def test_high_model_probability_with_clean_signals_cannot_produce_critical(self):
        engine = self._engine(0.95)
        clean = DrowsinessSignals(ear=0.30, mar=0.10, perclos=0.02, perclos_short=0.02)

        result = None
        for _ in range(6):
            result = engine.update(clean)

        self.assertIsNotNone(result)
        self.assertNotEqual(result.state, DrowsinessState.CRITICAL)
        self.assertIn(result.state, {DrowsinessState.ALERT, DrowsinessState.SUSPICIOUS})
        self.assertEqual(result.debug["hybrid_guard"], "clean_cap")

    def test_isolated_yawn_recovers_after_clean_streak(self):
        engine = self._engine(0.95)
        yawn = DrowsinessSignals(
            ear=0.28,
            mar=0.85,
            perclos=0.10,
            perclos_short=0.10,
            mar_above_threshold=True,
        )
        clean = DrowsinessSignals(ear=0.30, mar=0.10, perclos=0.02, perclos_short=0.02)

        result = None
        for _ in range(2):
            result = engine.update(yawn)
        self.assertIsNotNone(result)
        self.assertNotEqual(result.state, DrowsinessState.CRITICAL)

        for _ in range(8):
            result = engine.update(clean)

        self.assertIn(result.state, {DrowsinessState.ALERT, DrowsinessState.SUSPICIOUS})
        self.assertGreaterEqual(result.debug["hybrid_clean_streak"], 3)
        self.assertEqual(result.debug["hybrid_guard"], "clean_cap")

    def test_sustained_severe_evidence_can_escalate_to_critical(self):
        engine = self._engine(0.95)
        severe = DrowsinessSignals(
            ear=0.08,
            mar=0.80,
            perclos=0.70,
            perclos_short=0.75,
            yawn_frequency=2,
            head_nod_detected=True,
            eyes_closed_consecutive=20,
            ear_below_threshold=True,
            mar_above_threshold=True,
        )

        result = None
        for _ in range(6):
            result = engine.update(severe)

        self.assertEqual(result.state, DrowsinessState.CRITICAL)
        self.assertEqual(result.debug["hybrid_guard"], "critical_sustained")
        self.assertGreaterEqual(result.debug["hybrid_severe_streak"], 2)

    def test_hybrid_policy_config_can_raise_critical_probability_requirement(self):
        engine = self._engine(0.95)
        engine.config.hybrid_policy.critical_probability_threshold = 0.99
        engine.hybrid_policy = engine.hybrid_policy.__class__(engine.config.hybrid_policy)
        severe = DrowsinessSignals(
            ear=0.08,
            mar=0.80,
            perclos=0.70,
            perclos_short=0.75,
            yawn_frequency=2,
            head_nod_detected=True,
            eyes_closed_consecutive=20,
            ear_below_threshold=True,
            mar_above_threshold=True,
        )

        result = None
        for _ in range(6):
            result = engine.update(severe)

        self.assertNotEqual(result.state, DrowsinessState.CRITICAL)
        self.assertNotEqual(result.debug["hybrid_guard"], "critical_sustained")

    def test_low_model_probability_respects_severe_fsm_safety_state(self):
        engine = self._engine(0.10)
        engine.config.hybrid_policy.fsm_safety_cap_state = ""
        engine.hybrid_policy = engine.hybrid_policy.__class__(engine.config.hybrid_policy)
        severe = DrowsinessSignals(
            ear=0.08,
            mar=0.80,
            perclos=0.70,
            perclos_short=0.75,
            yawn_frequency=2,
            head_nod_detected=True,
            eyes_closed_consecutive=20,
            ear_below_threshold=True,
            mar_above_threshold=True,
        )

        result = None
        for _ in range(12):
            result = engine.update(severe)

        self.assertIn(result.state, {DrowsinessState.DROWSY, DrowsinessState.CRITICAL})
        self.assertEqual(result.debug["predicted_sleepy"], 0)
        self.assertIn(result.debug["hybrid_guard"], {"fsm_safety", "critical_sustained"})

    def test_fsm_safety_cap_state_can_hold_to_suspicious(self):
        engine = self._engine(0.10)
        engine.config.hybrid_policy.fsm_safety_cap_state = "SUSPICIOUS"
        engine.hybrid_policy = engine.hybrid_policy.__class__(engine.config.hybrid_policy)
        severe = DrowsinessSignals(
            ear=0.08,
            mar=0.80,
            perclos=0.70,
            perclos_short=0.75,
            yawn_frequency=2,
            head_nod_detected=True,
            eyes_closed_consecutive=20,
            ear_below_threshold=True,
            mar_above_threshold=True,
        )

        result = None
        for _ in range(12):
            result = engine.update(severe)

        self.assertEqual(result.state, DrowsinessState.SUSPICIOUS)
        self.assertEqual(result.debug["predicted_sleepy"], 0)
        self.assertEqual(result.debug["hybrid_guard"], "fsm_safety")

    def test_quality_guard_suppresses_model_when_face_ratio_is_low(self):
        engine = self._engine(0.95)
        engine.config.camera_model.quality_guard.enabled = True
        engine.config.camera_model.quality_guard.min_valid_face_ratio = 0.75
        missing_face = DrowsinessSignals(face_detected=False)

        result = None
        for _ in range(6):
            result = engine.update(missing_face)

        self.assertEqual(result.state, DrowsinessState.ALERT)
        self.assertEqual(result.debug["quality_guard_triggered"], 1)
        self.assertEqual(result.debug["quality_guard_reason"], "low_valid_face_ratio")
        self.assertIn("QUALITY_GUARD", result.reasons)

    def test_quality_guard_suppresses_model_for_degenerate_window(self):
        engine = self._engine(0.95)
        engine.config.camera_model.quality_guard.enabled = True
        degenerate = DrowsinessSignals(face_detected=True, ear=0.0, mar=0.0, perclos=0.0, perclos_short=0.0)

        result = None
        for _ in range(6):
            result = engine.update(degenerate)

        self.assertEqual(result.state, DrowsinessState.ALERT)
        self.assertEqual(result.debug["quality_guard_triggered"], 1)
        self.assertEqual(result.debug["quality_guard_reason"], "degenerate_window")
        self.assertIn("QUALITY_GUARD", result.reasons)

    def test_camera_model_warmup_preserves_base_state_by_default(self):
        engine = self._engine(0.95)

        result = engine._warmup_result(DrowsinessState.SUSPICIOUS, 0.45)

        self.assertEqual(result.state, DrowsinessState.SUSPICIOUS)
        self.assertAlmostEqual(result.evidence, 0.45)
        self.assertEqual(result.debug["warmup_alert_suppression_triggered"], 0)

    def test_camera_model_warmup_can_suppress_high_state_when_opted_in(self):
        engine = self._engine(0.95)
        engine.config.camera_model.suppress_warmup_alerts = True

        result = engine._warmup_result(DrowsinessState.SUSPICIOUS, 0.45)

        self.assertEqual(result.state, DrowsinessState.ALERT)
        self.assertEqual(result.evidence, 0.0)
        self.assertEqual(result.debug["base_fsm_state"], "SUSPICIOUS")
        self.assertEqual(result.debug["warmup_alert_suppression_triggered"], 1)
        self.assertEqual(result.debug["warmup_original_state"], "SUSPICIOUS")
        self.assertAlmostEqual(result.debug["warmup_original_evidence"], 0.45)

    def test_post_hybrid_rules_can_rescue_subtle_sleepy_window(self):
        engine = self._engine(0.50)

        adjusted = engine._apply_post_hybrid_rules(
            decision=HybridDecision(
                state=DrowsinessState.SUSPICIOUS,
                guard="hold",
                recent_evidence=True,
                clean_streak=0,
                severe_streak=0,
                elevated_streak=0,
                release_clean_streak=0,
            ),
            predicted_sleepy=False,
            probability=0.50,
            window={
                "ear_std": 0.060,
                "ear_p10": 0.090,
                "head_drop_count": 0,
                "max_eye_closed_duration_sec": 0.80,
            },
        )

        self.assertEqual(adjusted["state"], DrowsinessState.DROWSY)
        self.assertEqual(adjusted["predicted_sleepy"], 1)
        self.assertEqual(adjusted["guard"], "subtle_rescue")
        self.assertEqual(adjusted["rule_action"], "rescue")

    def test_post_hybrid_rules_can_suppress_false_positive_window(self):
        engine = self._engine(0.40)

        adjusted = engine._apply_post_hybrid_rules(
            decision=HybridDecision(
                state=DrowsinessState.DROWSY,
                guard="model_plus_fsm",
                recent_evidence=True,
                clean_streak=0,
                severe_streak=0,
                elevated_streak=0,
                release_clean_streak=0,
            ),
            predicted_sleepy=True,
            probability=0.40,
            window={
                "ear_std": 0.030,
                "ear_p10": 0.220,
                "head_drop_count": 800,
                "max_eye_closed_duration_sec": 0.70,
            },
        )

        self.assertEqual(adjusted["state"], DrowsinessState.SUSPICIOUS)
        self.assertEqual(adjusted["predicted_sleepy"], 0)
        self.assertEqual(adjusted["guard"], "fp_suppression")
        self.assertEqual(adjusted["rule_action"], "suppress")

    def test_guarded_severe_rescue_is_opt_in_and_requires_strict_eye_evidence(self):
        engine = self._engine(0.40)
        decision = HybridDecision(
            state=DrowsinessState.SUSPICIOUS,
            guard="hold",
            recent_evidence=True,
            clean_streak=0,
            severe_streak=0,
            elevated_streak=0,
            release_clean_streak=0,
        )
        strict_window = {
            "mean_ear": 0.19,
            "max_fsm_evidence": 0.86,
            "max_eye_closed_duration_sec": 5.2,
        }

        disabled = engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.40,
            window=strict_window,
        )

        self.assertEqual(disabled["state"], DrowsinessState.SUSPICIOUS)
        self.assertEqual(disabled["rule_action"], "")

        engine.config.hybrid_policy.guarded_severe_rescue.enabled = True
        enabled = engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.40,
            window=strict_window,
        )
        weak_eye = engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.40,
            window={
                "mean_ear": 0.21,
                "max_fsm_evidence": 0.86,
                "max_eye_closed_duration_sec": 5.2,
            },
        )

        self.assertEqual(enabled["state"], DrowsinessState.DROWSY)
        self.assertEqual(enabled["predicted_sleepy"], 1)
        self.assertEqual(enabled["guard"], "guarded_severe_rescue")
        self.assertEqual(enabled["rule_action"], "guarded_rescue")
        self.assertEqual(weak_eye["state"], DrowsinessState.SUSPICIOUS)
        self.assertEqual(weak_eye["rule_action"], "")

    def test_seeded_macroevent_bridge_is_causal_and_opt_in(self):
        engine = self._engine(0.40)
        decision = HybridDecision(
            state=DrowsinessState.SUSPICIOUS,
            guard="hold",
            recent_evidence=True,
            clean_streak=0,
            severe_streak=0,
            elevated_streak=0,
            release_clean_streak=0,
        )
        seed_window = {
            "head_drop_count": 450.0,
            "max_mar": 0.20,
            "perclos_60s": 0.35,
            "max_eye_closed_duration_sec": 0.8,
            "mean_mar": 0.02,
        }
        bridge_candidate_window = {
            "head_drop_count": 0.0,
            "max_mar": 0.05,
            "perclos_60s": 0.20,
            "max_eye_closed_duration_sec": 0.7,
            "mean_mar": 0.02,
        }

        disabled_seed = engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.43,
            window=seed_window,
            timestamp_sec=0.0,
            base_fsm_state=DrowsinessState.ALERT,
        )
        self.assertEqual(disabled_seed["rule_action"], "")
        self.assertEqual(disabled_seed["macroevent_seed_count"], 0)

        cfg = engine.config.hybrid_policy.seeded_macroevent_bridge
        cfg.enabled = True
        cfg.active_seconds = 120.0

        first_seed = engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.43,
            window=seed_window,
            timestamp_sec=0.0,
            base_fsm_state=DrowsinessState.ALERT,
        )
        between_seeds = engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.34,
            window=bridge_candidate_window,
            timestamp_sec=30.0,
            base_fsm_state=DrowsinessState.ALERT,
        )
        second_seed = engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.43,
            window=seed_window,
            timestamp_sec=60.0,
            base_fsm_state=DrowsinessState.ALERT,
        )
        after_second_seed = engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.34,
            window=bridge_candidate_window,
            timestamp_sec=90.0,
            base_fsm_state=DrowsinessState.ALERT,
        )

        self.assertEqual(first_seed["rule_action"], "")
        self.assertEqual(between_seeds["rule_action"], "")
        self.assertEqual(second_seed["rule_action"], "")
        self.assertEqual(after_second_seed["state"], DrowsinessState.DROWSY)
        self.assertEqual(after_second_seed["predicted_sleepy"], 1)
        self.assertEqual(after_second_seed["guard"], "seeded_macroevent_bridge")
        self.assertEqual(after_second_seed["rule_action"], "bridge")
        self.assertEqual(after_second_seed["macroevent_seed_count"], 2)

        already_sleepy = engine._apply_post_hybrid_rules(
            decision=HybridDecision(
                state=DrowsinessState.DROWSY,
                guard="hold",
                recent_evidence=True,
                clean_streak=0,
                severe_streak=0,
                elevated_streak=0,
                release_clean_streak=0,
            ),
            predicted_sleepy=False,
            probability=0.34,
            window=bridge_candidate_window,
            timestamp_sec=100.0,
            base_fsm_state=DrowsinessState.ALERT,
        )

        self.assertEqual(already_sleepy["state"], DrowsinessState.DROWSY)
        self.assertEqual(already_sleepy["guard"], "hold")
        self.assertEqual(already_sleepy["rule_action"], "")

    def test_seeded_macroevent_bridge_strict_guard_requires_candidate_fsm_evidence(self):
        engine = self._engine(0.40)
        decision = HybridDecision(
            state=DrowsinessState.SUSPICIOUS,
            guard="hold",
            recent_evidence=True,
            clean_streak=0,
            severe_streak=0,
            elevated_streak=0,
            release_clean_streak=0,
        )
        seed_window = {
            "head_drop_count": 450.0,
            "max_mar": 0.20,
            "perclos_60s": 0.35,
            "max_eye_closed_duration_sec": 0.8,
            "mean_mar": 0.02,
        }
        bridge_candidate_window = {
            "head_drop_count": 0.0,
            "max_mar": 0.05,
            "perclos_60s": 0.20,
            "max_eye_closed_duration_sec": 0.7,
            "mean_mar": 0.02,
        }
        cfg = engine.config.hybrid_policy.seeded_macroevent_bridge
        cfg.enabled = True
        cfg.active_seconds = 120.0
        cfg.bridge_min_max_fsm_evidence = 0.35

        engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.43,
            window=seed_window,
            timestamp_sec=0.0,
            base_fsm_state=DrowsinessState.ALERT,
        )
        engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.43,
            window=seed_window,
            timestamp_sec=60.0,
            base_fsm_state=DrowsinessState.ALERT,
        )
        weak_candidate = engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.34,
            window={**bridge_candidate_window, "max_fsm_evidence": 0.30},
            timestamp_sec=90.0,
            base_fsm_state=DrowsinessState.ALERT,
        )
        strong_candidate = engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.34,
            window={**bridge_candidate_window, "max_fsm_evidence": 0.35},
            timestamp_sec=100.0,
            base_fsm_state=DrowsinessState.ALERT,
        )

        self.assertEqual(weak_candidate["state"], DrowsinessState.SUSPICIOUS)
        self.assertEqual(weak_candidate["rule_action"], "")
        self.assertEqual(weak_candidate["macroevent_seed_count"], 2)
        self.assertEqual(strong_candidate["state"], DrowsinessState.DROWSY)
        self.assertEqual(strong_candidate["guard"], "seeded_macroevent_bridge")
        self.assertEqual(strong_candidate["rule_action"], "bridge")

    def test_seeded_macroevent_bridge_can_target_suspicious_for_visual_only_warning(self):
        engine = self._engine(0.40)
        decision = HybridDecision(
            state=DrowsinessState.ALERT,
            guard="hold",
            recent_evidence=False,
            clean_streak=0,
            severe_streak=0,
            elevated_streak=0,
            release_clean_streak=0,
        )
        seed_window = {
            "head_drop_count": 450.0,
            "max_mar": 0.20,
            "perclos_60s": 0.35,
            "max_eye_closed_duration_sec": 0.8,
            "mean_mar": 0.02,
            "max_fsm_evidence": 0.35,
        }
        bridge_candidate_window = {
            "head_drop_count": 0.0,
            "max_mar": 0.05,
            "perclos_60s": 0.20,
            "max_eye_closed_duration_sec": 0.7,
            "mean_mar": 0.02,
            "max_fsm_evidence": 0.35,
        }
        cfg = engine.config.hybrid_policy.seeded_macroevent_bridge
        cfg.enabled = True
        cfg.active_seconds = 120.0
        cfg.bridge_min_max_fsm_evidence = 0.35
        cfg.target_state = "SUSPICIOUS"

        engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.43,
            window=seed_window,
            timestamp_sec=0.0,
            base_fsm_state=DrowsinessState.ALERT,
        )
        engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.43,
            window=seed_window,
            timestamp_sec=60.0,
            base_fsm_state=DrowsinessState.ALERT,
        )
        tempered_bridge = engine._apply_post_hybrid_rules(
            decision=decision,
            predicted_sleepy=False,
            probability=0.34,
            window=bridge_candidate_window,
            timestamp_sec=90.0,
            base_fsm_state=DrowsinessState.ALERT,
        )

        self.assertEqual(tempered_bridge["state"], DrowsinessState.SUSPICIOUS)
        self.assertEqual(tempered_bridge["predicted_sleepy"], 0)
        self.assertEqual(tempered_bridge["guard"], "seeded_macroevent_bridge")
        self.assertEqual(tempered_bridge["rule_action"], "bridge")

    def test_seeded_macroevent_bridge_alert_override_only_applies_to_bridge_rows(self):
        engine = self._engine(0.40)

        self.assertEqual(engine._resolve_alert_sound("double", "bridge"), ("double", ""))
        self.assertEqual(
            engine._runtime_alert_semantic(DrowsinessState.DROWSY, "double", "bridge", ""),
            "standard",
        )

        engine.config.hybrid_policy.seeded_macroevent_bridge.alert_sound_override = "none"
        self.assertEqual(engine._resolve_alert_sound("double", "bridge"), ("none", "none"))
        self.assertEqual(engine._resolve_alert_sound("double", "guarded_rescue"), ("double", ""))
        self.assertEqual(
            engine._runtime_alert_semantic(DrowsinessState.DROWSY, "none", "bridge", "none"),
            "visual_only_drowsy_bridge",
        )
        self.assertEqual(
            engine._display_label("DROWSY", "visual_only_drowsy_bridge"),
            "HYBRID VISUAL DROWSY",
        )
        self.assertEqual(engine._display_label("DROWSY", "standard"), "HYBRID DROWSY")

        engine.config.hybrid_policy.seeded_macroevent_bridge.alert_sound_override = "invalid"
        self.assertEqual(engine._resolve_alert_sound("double", "bridge"), ("double", ""))

    def test_low_probability_release_can_drop_stuck_drowsy_state_when_current_cues_are_clean(self):
        engine = self._engine(0.90)
        engine.config.hybrid_policy.low_probability_release.enabled = True
        engine.config.hybrid_policy.low_probability_release.clean_streak = 2
        engine.config.hybrid_policy.low_probability_release.target_state = "SUSPICIOUS"
        engine.hybrid_policy = engine.hybrid_policy.__class__(engine.config.hybrid_policy)

        recent_sleepy = DrowsinessSignals(
            ear=0.12,
            mar=0.10,
            perclos=0.20,
            perclos_short=0.25,
            eyes_closed_consecutive=4,
            ear_below_threshold=True,
        )
        clean = DrowsinessSignals(
            ear=0.30,
            mar=0.10,
            perclos=0.08,
            perclos_short=0.08,
            eyes_closed_consecutive=0,
            ear_below_threshold=False,
            mar_above_threshold=False,
            head_nod_detected=False,
            yawn_frequency=0,
        )

        for _ in range(6):
            engine.update(recent_sleepy)

        engine.model.probability = 0.05
        first = engine.update(clean)
        second = engine.update(clean)

        self.assertEqual(first.state, DrowsinessState.DROWSY)
        self.assertEqual(first.debug["low_probability_release_triggered"], 0)
        self.assertEqual(second.state, DrowsinessState.SUSPICIOUS)
        self.assertEqual(second.debug["hybrid_guard"], "low_probability_release")
        self.assertEqual(second.debug["predicted_sleepy"], 0)
        self.assertEqual(second.debug["low_probability_release_triggered"], 1)

    def test_low_probability_release_respects_min_head_drop_count(self):
        engine = self._engine(0.90)
        engine.config.hybrid_policy.low_probability_release.enabled = True
        engine.config.hybrid_policy.low_probability_release.clean_streak = 2
        engine.config.hybrid_policy.low_probability_release.min_head_drop_count = 1.0
        engine.hybrid_policy = engine.hybrid_policy.__class__(engine.config.hybrid_policy)

        recent_sleepy = DrowsinessSignals(
            ear=0.12,
            mar=0.10,
            perclos=0.20,
            perclos_short=0.25,
            eyes_closed_consecutive=4,
            ear_below_threshold=True,
        )
        clean = DrowsinessSignals(
            ear=0.30,
            mar=0.10,
            perclos=0.08,
            perclos_short=0.08,
            eyes_closed_consecutive=0,
            ear_below_threshold=False,
            mar_above_threshold=False,
            head_nod_detected=False,
            yawn_frequency=0,
        )

        for _ in range(6):
            engine.update(recent_sleepy)

        engine.model.probability = 0.05
        first = engine.update(clean)
        second = engine.update(clean)

        self.assertEqual(first.state, DrowsinessState.DROWSY)
        self.assertEqual(second.state, DrowsinessState.DROWSY)
        self.assertEqual(second.debug["hybrid_guard"], "hold")
        self.assertEqual(second.debug["low_probability_release_triggered"], 0)

    def test_low_probability_release_respects_max_perclos_long(self):
        engine = self._engine(0.90)
        engine.config.hybrid_policy.low_probability_release.enabled = True
        engine.config.hybrid_policy.low_probability_release.clean_streak = 2
        engine.config.hybrid_policy.low_probability_release.max_perclos_long = 0.05
        engine.hybrid_policy = engine.hybrid_policy.__class__(engine.config.hybrid_policy)

        recent_sleepy = DrowsinessSignals(
            ear=0.12,
            mar=0.10,
            perclos=0.20,
            perclos_short=0.25,
            eyes_closed_consecutive=4,
            ear_below_threshold=True,
        )
        clean = DrowsinessSignals(
            ear=0.30,
            mar=0.10,
            perclos=0.08,
            perclos_short=0.08,
            eyes_closed_consecutive=0,
            ear_below_threshold=False,
            mar_above_threshold=False,
            head_nod_detected=False,
            yawn_frequency=0,
        )

        for _ in range(6):
            engine.update(recent_sleepy)

        engine.model.probability = 0.05
        first = engine.update(clean)
        second = engine.update(clean)

        self.assertEqual(first.state, DrowsinessState.DROWSY)
        self.assertEqual(second.state, DrowsinessState.DROWSY)
        self.assertEqual(second.debug["hybrid_guard"], "hold")
        self.assertEqual(second.debug["low_probability_release_triggered"], 0)

    def test_hold_decay_release_can_drop_stuck_drowsy_state_when_low_probability_persists_with_only_suspicious_base_state(self):
        cfg = default_runtime_config().hybrid_policy
        cfg.hold_decay.enabled = True
        cfg.hold_decay.hold_streak = 2
        cfg.hold_decay.max_base_state = "SUSPICIOUS"
        cfg.hold_decay.max_base_fsm_evidence = 0.35
        cfg.hold_decay.max_perclos_short = 0.25
        cfg.hold_decay.max_eyes_closed_consecutive = 3
        policy = HybridDecisionPolicy(cfg)

        recent_sleepy = HybridEvidence(
            probability=0.90,
            threshold=0.50,
            ml_only_state=DrowsinessState.DROWSY,
            base_fsm_state=DrowsinessState.ALERT,
            base_fsm_evidence=0.40,
            perclos_short=0.25,
            perclos_long=0.20,
            eyes_closed_consecutive=4,
            ear_below_threshold=True,
            max_fsm_evidence=0.40,
        )
        for _ in range(3):
            policy.update(recent_sleepy)

        low_probability_hold = HybridEvidence(
            probability=0.05,
            threshold=0.50,
            ml_only_state=DrowsinessState.ALERT,
            base_fsm_state=DrowsinessState.SUSPICIOUS,
            base_fsm_evidence=0.30,
            perclos_short=0.20,
            perclos_long=0.20,
            eyes_closed_consecutive=2,
            ear_below_threshold=False,
            mar_above_threshold=False,
            yawn_count=0.0,
            head_drop_count=0.0,
            max_eye_closed_duration_sec=0.40,
            max_fsm_evidence=0.30,
        )

        first = policy.update(low_probability_hold)
        second = policy.update(low_probability_hold)

        self.assertEqual(first.state, DrowsinessState.DROWSY)
        self.assertEqual(first.guard, "hold")
        self.assertEqual(second.state, DrowsinessState.SUSPICIOUS)
        self.assertEqual(second.guard, "hold_decay_release")

    def test_hold_decay_release_does_not_fire_while_base_fsm_state_is_still_drowsy(self):
        cfg = default_runtime_config().hybrid_policy
        cfg.hold_decay.enabled = True
        cfg.hold_decay.hold_streak = 2
        cfg.hold_decay.max_base_state = "SUSPICIOUS"
        cfg.hold_decay.max_base_fsm_evidence = 0.35
        cfg.hold_decay.max_perclos_short = 0.25
        cfg.hold_decay.max_eyes_closed_consecutive = 3
        policy = HybridDecisionPolicy(cfg)

        recent_sleepy = HybridEvidence(
            probability=0.90,
            threshold=0.50,
            ml_only_state=DrowsinessState.DROWSY,
            base_fsm_state=DrowsinessState.ALERT,
            base_fsm_evidence=0.40,
            perclos_short=0.25,
            perclos_long=0.20,
            eyes_closed_consecutive=4,
            ear_below_threshold=True,
            max_fsm_evidence=0.40,
        )
        for _ in range(3):
            policy.update(recent_sleepy)

        low_probability_but_drowsy_base = HybridEvidence(
            probability=0.05,
            threshold=0.50,
            ml_only_state=DrowsinessState.ALERT,
            base_fsm_state=DrowsinessState.DROWSY,
            base_fsm_evidence=0.30,
            perclos_short=0.20,
            perclos_long=0.20,
            eyes_closed_consecutive=2,
            ear_below_threshold=False,
            mar_above_threshold=False,
            yawn_count=0.0,
            head_drop_count=0.0,
            max_eye_closed_duration_sec=0.40,
            max_fsm_evidence=0.30,
        )

        first = policy.update(low_probability_but_drowsy_base)
        second = policy.update(low_probability_but_drowsy_base)

        self.assertEqual(first.state, DrowsinessState.DROWSY)
        self.assertEqual(second.state, DrowsinessState.DROWSY)
        self.assertEqual(second.guard, "hold")

    def test_dwell_relief_can_accumulate_credit_across_interrupted_low_risk_windows(self):
        cfg = default_runtime_config().hybrid_policy
        cfg.dwell_relief.enabled = True
        cfg.dwell_relief.min_credit = 2.0
        cfg.dwell_relief.alert_gain = 1.0
        cfg.dwell_relief.suspicious_gain = 0.5
        cfg.dwell_relief.decay = 0.5
        cfg.dwell_relief.max_base_state = "SUSPICIOUS"
        cfg.dwell_relief.max_base_fsm_evidence = 0.35
        cfg.dwell_relief.max_perclos_short = 0.25
        cfg.dwell_relief.max_eyes_closed_consecutive = 2
        policy = HybridDecisionPolicy(cfg)

        recent_sleepy = HybridEvidence(
            probability=0.90,
            threshold=0.50,
            ml_only_state=DrowsinessState.DROWSY,
            base_fsm_state=DrowsinessState.ALERT,
            base_fsm_evidence=0.40,
            perclos_short=0.25,
            perclos_long=0.20,
            eyes_closed_consecutive=4,
            ear_below_threshold=True,
            max_fsm_evidence=0.40,
        )
        for _ in range(3):
            policy.update(recent_sleepy)

        low_probability_alert = HybridEvidence(
            probability=0.05,
            threshold=0.50,
            ml_only_state=DrowsinessState.ALERT,
            base_fsm_state=DrowsinessState.ALERT,
            base_fsm_evidence=0.20,
            perclos_short=0.12,
            perclos_long=0.15,
            eyes_closed_consecutive=0,
            ear_below_threshold=False,
            mar_above_threshold=False,
            yawn_count=0.0,
            head_drop_count=0.0,
            max_eye_closed_duration_sec=0.30,
            max_fsm_evidence=0.20,
        )
        interrupted_low_probability = HybridEvidence(
            probability=0.05,
            threshold=0.50,
            ml_only_state=DrowsinessState.ALERT,
            base_fsm_state=DrowsinessState.ALERT,
            base_fsm_evidence=0.20,
            perclos_short=0.12,
            perclos_long=0.15,
            eyes_closed_consecutive=0,
            ear_below_threshold=False,
            mar_above_threshold=False,
            yawn_count=0.0,
            head_drop_count=1.0,
            max_eye_closed_duration_sec=0.30,
            max_fsm_evidence=0.20,
        )
        low_probability_suspicious = HybridEvidence(
            probability=0.05,
            threshold=0.50,
            ml_only_state=DrowsinessState.ALERT,
            base_fsm_state=DrowsinessState.SUSPICIOUS,
            base_fsm_evidence=0.20,
            perclos_short=0.15,
            perclos_long=0.15,
            eyes_closed_consecutive=1,
            ear_below_threshold=False,
            mar_above_threshold=False,
            yawn_count=0.0,
            head_drop_count=0.0,
            max_eye_closed_duration_sec=0.30,
            max_fsm_evidence=0.20,
        )

        first = policy.update(low_probability_alert)
        second = policy.update(interrupted_low_probability)
        third = policy.update(low_probability_alert)
        fourth = policy.update(low_probability_suspicious)

        self.assertEqual(first.state, DrowsinessState.DROWSY)
        self.assertEqual(first.guard, "hold")
        self.assertEqual(second.state, DrowsinessState.DROWSY)
        self.assertEqual(second.guard, "hold")
        self.assertEqual(third.state, DrowsinessState.DROWSY)
        self.assertEqual(third.guard, "hold")
        self.assertEqual(fourth.state, DrowsinessState.SUSPICIOUS)
        self.assertEqual(fourth.guard, "dwell_relief_release")

    def test_dwell_relief_does_not_accumulate_credit_under_drowsy_base_state(self):
        cfg = default_runtime_config().hybrid_policy
        cfg.dwell_relief.enabled = True
        cfg.dwell_relief.min_credit = 2.0
        cfg.dwell_relief.alert_gain = 1.0
        cfg.dwell_relief.suspicious_gain = 1.0
        cfg.dwell_relief.decay = 0.5
        cfg.dwell_relief.max_base_state = "SUSPICIOUS"
        cfg.dwell_relief.max_base_fsm_evidence = 0.35
        cfg.dwell_relief.max_perclos_short = 0.25
        cfg.dwell_relief.max_eyes_closed_consecutive = 2
        policy = HybridDecisionPolicy(cfg)

        recent_sleepy = HybridEvidence(
            probability=0.90,
            threshold=0.50,
            ml_only_state=DrowsinessState.DROWSY,
            base_fsm_state=DrowsinessState.ALERT,
            base_fsm_evidence=0.40,
            perclos_short=0.25,
            perclos_long=0.20,
            eyes_closed_consecutive=4,
            ear_below_threshold=True,
            max_fsm_evidence=0.40,
        )
        for _ in range(3):
            policy.update(recent_sleepy)

        low_probability_drowsy_base = HybridEvidence(
            probability=0.05,
            threshold=0.50,
            ml_only_state=DrowsinessState.ALERT,
            base_fsm_state=DrowsinessState.DROWSY,
            base_fsm_evidence=0.20,
            perclos_short=0.15,
            perclos_long=0.15,
            eyes_closed_consecutive=1,
            ear_below_threshold=False,
            mar_above_threshold=False,
            yawn_count=0.0,
            head_drop_count=0.0,
            max_eye_closed_duration_sec=0.30,
            max_fsm_evidence=0.20,
        )

        first = policy.update(low_probability_drowsy_base)
        second = policy.update(low_probability_drowsy_base)
        third = policy.update(low_probability_drowsy_base)

        self.assertEqual(first.state, DrowsinessState.DROWSY)
        self.assertEqual(second.state, DrowsinessState.DROWSY)
        self.assertEqual(third.state, DrowsinessState.DROWSY)
        self.assertEqual(third.guard, "hold")

    def test_model_plus_recent_drowsy_floor_can_keep_recent_promotion_at_suspicious(self):
        cfg = default_runtime_config().hybrid_policy
        cfg.elevated_streak_threshold = 2
        cfg.model_plus_recent_drowsy_probability_min = 0.75
        policy = HybridDecisionPolicy(cfg)

        recent_but_subfloor = HybridEvidence(
            probability=0.60,
            threshold=0.50,
            ml_only_state=DrowsinessState.DROWSY,
            base_fsm_state=DrowsinessState.ALERT,
            base_fsm_evidence=0.40,
            perclos_short=0.22,
            perclos_long=0.18,
            eyes_closed_consecutive=0,
            ear_below_threshold=False,
            mar_above_threshold=False,
            yawn_count=0.0,
            head_drop_count=0.0,
            max_eye_closed_duration_sec=0.20,
            max_fsm_evidence=0.40,
        )

        first = policy.update(recent_but_subfloor)
        second = policy.update(recent_but_subfloor)

        self.assertEqual(first.state, DrowsinessState.SUSPICIOUS)
        self.assertEqual(first.guard, "model_caution")
        self.assertEqual(second.state, DrowsinessState.SUSPICIOUS)
        self.assertEqual(second.guard, "model_recent_cap")

    def test_model_plus_recent_drowsy_floor_respects_max_eye_closed_duration_cap(self):
        cfg = default_runtime_config().hybrid_policy
        cfg.elevated_streak_threshold = 2
        cfg.model_plus_recent_drowsy_probability_min = 0.75
        cfg.model_plus_recent_max_eye_closed_duration_sec = 0.90
        policy = HybridDecisionPolicy(cfg)

        recent_but_long_eye_closure = HybridEvidence(
            probability=0.60,
            threshold=0.50,
            ml_only_state=DrowsinessState.DROWSY,
            base_fsm_state=DrowsinessState.ALERT,
            base_fsm_evidence=0.40,
            perclos_short=0.22,
            perclos_long=0.18,
            eyes_closed_consecutive=4,
            ear_below_threshold=False,
            mar_above_threshold=False,
            yawn_count=0.0,
            head_drop_count=0.0,
            max_eye_closed_duration_sec=1.20,
            max_fsm_evidence=0.40,
        )

        first = policy.update(recent_but_long_eye_closure)
        second = policy.update(recent_but_long_eye_closure)

        self.assertEqual(first.state, DrowsinessState.SUSPICIOUS)
        self.assertEqual(first.guard, "model_caution")
        self.assertEqual(second.state, DrowsinessState.DROWSY)
        self.assertEqual(second.guard, "model_plus_recent")

    def test_hybrid_policy_reports_hold_provenance_and_current_support(self):
        cfg = default_runtime_config().hybrid_policy
        cfg.elevated_streak_threshold = 1
        policy = HybridDecisionPolicy(cfg)

        model_supported = HybridEvidence(
            probability=0.90,
            threshold=0.50,
            ml_only_state=DrowsinessState.DROWSY,
            base_fsm_state=DrowsinessState.ALERT,
            base_fsm_evidence=0.40,
            perclos_short=0.25,
            perclos_long=0.20,
            eyes_closed_consecutive=1,
            max_fsm_evidence=0.40,
        )
        unsupported_hold = HybridEvidence(
            probability=0.10,
            threshold=0.50,
            ml_only_state=DrowsinessState.ALERT,
            base_fsm_state=DrowsinessState.ALERT,
            base_fsm_evidence=0.25,
            perclos_short=0.16,
            perclos_long=0.10,
            eyes_closed_consecutive=0,
            ear_below_threshold=False,
            mar_above_threshold=False,
            yawn_count=0.0,
            head_drop_count=0.0,
            max_eye_closed_duration_sec=0.20,
            max_fsm_evidence=0.25,
        )

        first = policy.update(model_supported)
        second = policy.update(unsupported_hold)
        third = policy.update(unsupported_hold)

        self.assertEqual(first.state, DrowsinessState.DROWSY)
        self.assertEqual(first.guard, "model_plus_recent")
        self.assertEqual(first.support_summary, "model+recent")
        self.assertEqual(first.high_state_age_frames, 1)
        self.assertEqual(first.hold_source_guard, "model_plus_recent")
        self.assertEqual(first.frames_since_hold_source, 0)

        self.assertEqual(second.state, DrowsinessState.DROWSY)
        self.assertEqual(second.guard, "hold")
        self.assertEqual(second.support_summary, "none")
        self.assertEqual(second.hold_age_frames, 1)
        self.assertEqual(second.high_state_age_frames, 2)
        self.assertEqual(second.hold_source_guard, "model_plus_recent")
        self.assertEqual(second.frames_since_hold_source, 1)

        self.assertEqual(third.guard, "hold")
        self.assertEqual(third.hold_age_frames, 2)
        self.assertEqual(third.high_state_age_frames, 3)
        self.assertEqual(third.frames_since_hold_source, 2)


if __name__ == "__main__":
    unittest.main()
