from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WinnerFoundationTests(unittest.TestCase):
    def test_model_manifest_matches_packaged_artifact(self) -> None:
        from runtime.model_bundle import WinnerModelBundle

        bundle = WinnerModelBundle.load(
            ROOT / "models" / "camera_hybrid_winner.joblib",
            ROOT / "models" / "winner_manifest.json",
        )

        artifact_hash = hashlib.sha256(bundle.model_path.read_bytes()).hexdigest()
        self.assertEqual(artifact_hash, bundle.manifest["sha256"])
        self.assertEqual(bundle.feature_columns, bundle.manifest["feature_columns"])
        self.assertEqual(bundle.artifact_threshold, 0.44)
        self.assertEqual(bundle.runtime_threshold, 0.55)
        self.assertEqual(bundle.manifest["default_profile"], "recommended")

    def test_model_artifact_is_cached_across_bundle_loads(self) -> None:
        from runtime.model_bundle import WinnerModelBundle

        first = WinnerModelBundle.load(
            ROOT / "models" / "camera_hybrid_winner.joblib",
            ROOT / "models" / "winner_manifest.json",
        )
        second = WinnerModelBundle.load(
            ROOT / "models" / "camera_hybrid_winner.joblib",
            ROOT / "models" / "winner_manifest.json",
        )
        self.assertIs(first.artifact, second.artifact)
    def test_lightweight_dynamic_ear_locks_with_source_parameters(self) -> None:
        from runtime.dynamic_ear import DynamicEAR

        detector = DynamicEAR()
        statuses = [detector.update(0.30) for _ in range(300)]

        self.assertEqual(statuses[29].phase, "WARMUP")
        self.assertEqual(statuses[30].phase, "CALIBRATING")
        self.assertEqual(statuses[299].phase, "CALIBRATING")
        self.assertIsNone(statuses[299].T_low)
        self.assertTrue(detector.locked)
        locked_status = detector.update(0.30)
        self.assertTrue(locked_status.locked)
        self.assertAlmostEqual(locked_status.T_low, 0.2925, places=4)
        self.assertFalse(locked_status.is_closed)
        self.assertTrue(detector.update(0.20).is_closed)
        self.assertFalse(detector.update(0.31).is_closed)

    def test_window_features_match_expected_camera_vector(self) -> None:
        from runtime.window_features import aggregate_frame_rows, numeric_feature_value

        rows = []
        for index in range(180):
            closed = 30 <= index < 60
            rows.append(
                {
                    "subject_id": "fixture",
                    "session_id": "fixture",
                    "video_id": "fixture",
                    "timestamp_sec": index / 30.0,
                    "face_detected": 1,
                    "ear": 0.12 if closed else 0.30,
                    "mar": 0.60 if index >= 150 else 0.20,
                    "eye_closed": int(closed),
                    "mouth_open": int(index >= 150),
                    "head_nod_detected": int(index >= 120),
                    "perclos_60s": 30 / 180,
                    "perclos_5s": 30 / 150,
                    "blink_frequency": 1,
                    "yawn_frequency": 1,
                    "pitch_velocity": 0.0,
                    "gaze_stable": 1,
                    "fsm_state": "DROWSY" if closed else "ALERT",
                    "fsm_evidence": 0.8 if closed else 0.0,
                }
            )

        windows = aggregate_frame_rows(rows, window_seconds=60.0, stride_seconds=60.0)
        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual(window["frame_count"], 180)
        self.assertAlmostEqual(window["valid_face_ratio"], 1.0)
        self.assertAlmostEqual(window["max_eye_closed_duration_sec"], 1.0)
        self.assertEqual(window["head_drop_count"], 60)
        self.assertEqual(numeric_feature_value(window, "fsm_state_mode"), 0.0)

    def test_runtime_window_fast_path_matches_canonical_aggregator(self) -> None:
        from runtime.window_features import aggregate_frame_rows, aggregate_runtime_window

        rows = []
        for index in range(1800):
            closed = 300 <= index < 345 or 900 <= index < 930
            rows.append({
                "subject_id": "fixture", "session_id": "fixture", "video_id": "fixture",
                "timestamp_sec": index / 30.0, "face_detected": int(index % 17 != 0),
                "ear": 0.11 if closed else 0.28 + (index % 5) * 0.002,
                "mar": 0.55 if index % 200 < 15 else 0.18,
                "eye_closed": int(closed), "mouth_open": int(index % 200 < 15),
                "head_nod_detected": int(index % 90 == 0),
                "perclos_60s": 0.25, "perclos_5s": 0.35,
                "blink_frequency": index // 150, "yawn_frequency": index // 600,
                "pitch_velocity": float(index % 7), "gaze_stable": 1,
                "fsm_state": "DROWSY" if closed else ("SUSPICIOUS" if index % 40 == 0 else "ALERT"),
                "fsm_evidence": 0.8 if closed else (0.4 if index % 40 == 0 else 0.0),
            })

        canonical = aggregate_frame_rows(rows, window_seconds=60.0, stride_seconds=60.0)[-1]
        fast = aggregate_runtime_window(rows, window_seconds=60.0, rows_are_ordered=True)
        self.assertIsNotNone(fast)
        for key in (
            "frame_count", "valid_face_ratio", "mean_ear", "min_ear", "ear_std", "ear_p10",
            "mean_mar", "max_mar", "perclos_60s", "perclos_5s", "max_eye_closed_duration_sec",
            "yawn_count", "head_drop_count", "mean_fsm_evidence", "max_fsm_evidence",
        ):
            self.assertAlmostEqual(float(fast[key]), float(canonical[key]), places=12, msg=key)
        self.assertEqual(fast["fsm_state_mode"], canonical["fsm_state_mode"])

    def test_fast_isotonic_predictor_matches_packaged_sklearn_model(self) -> None:
        import numpy as np

        from runtime.model_bundle import FastIsotonicBinaryPredictor, WinnerModelBundle

        bundle = WinnerModelBundle.load(
            ROOT / "models" / "camera_hybrid_winner.joblib",
            ROOT / "models" / "winner_manifest.json",
        )
        predictor = FastIsotonicBinaryPredictor.from_model(bundle.model)
        base = np.array([3600.0, 0.75, 1200.0, 2.4, 0.55, 0.25, 0.22, 0.28, 0.999, 0.2])
        scale = np.array([120.0, 1.2, 1600.0, 4.5, 0.2, 0.25, 0.04, 0.24, 0.002, 0.5])
        rows = np.vstack([base + ((index % 11) - 5) * scale / 5.0 for index in range(101)])

        expected = bundle.model.predict_proba(rows)
        actual = predictor.predict_proba(rows)

        self.assertLessEqual(float(np.max(np.abs(expected - actual))), 1e-12)

    def test_landmark_adapter_computes_ear_mar_and_pose(self) -> None:
        from runtime.landmark_adapter import LandmarkPacketAdapter

        adapter = LandmarkPacketAdapter()
        landmarks = {
            33: (0.30, 0.40), 160: (0.32, 0.38), 158: (0.36, 0.38),
            133: (0.40, 0.40), 153: (0.36, 0.42), 144: (0.32, 0.42),
            362: (0.60, 0.40), 385: (0.62, 0.38), 387: (0.66, 0.38),
            263: (0.70, 0.40), 373: (0.66, 0.42), 380: (0.62, 0.42),
            61: (0.40, 0.62), 291: (0.60, 0.62),
            13: (0.50, 0.60), 14: (0.50, 0.66),
            1: (0.50, 0.48), 152: (0.50, 0.78),
            468: (0.35, 0.40), 473: (0.65, 0.40),
        }

        raw = adapter.from_normalized(landmarks, width=1000, height=800, face_detected=True)
        self.assertTrue(raw.face_detected)
        self.assertAlmostEqual(raw.ear, 0.32, places=6)
        self.assertAlmostEqual(raw.mar, 0.24, places=6)
        self.assertTrue(math.isfinite(raw.pitch))
        self.assertTrue(math.isfinite(raw.yaw))
        self.assertEqual(raw.gaze_center, (500.0, 320.0))

    def test_web_runtime_profile_registry_exposes_only_packaged_profiles(self) -> None:
        from runtime.config import available_runtime_profiles, runtime_profile_config_path

        self.assertEqual(available_runtime_profiles(), ("protected", "recommended"))
        for profile in available_runtime_profiles():
            self.assertTrue((ROOT / runtime_profile_config_path(profile)).is_file())
    def test_recommended_config_is_packaged_without_runtime_logging(self) -> None:
        recommended = json.loads((ROOT / "configs" / "recommended.json").read_text(encoding="utf-8"))
        protected = json.loads((ROOT / "configs" / "protected.json").read_text(encoding="utf-8"))

        self.assertEqual(recommended["decision_engine"], "camera_hybrid")
        self.assertEqual(recommended["feature_backend"], "phuong")
        self.assertEqual(recommended["alerts"]["drowsy_cooldown_seconds"], 15.0)
        self.assertTrue(recommended["camera_model"]["suppress_warmup_alerts"])
        self.assertTrue(recommended["hybrid_policy"]["seeded_macroevent_bridge"]["enabled"])
        self.assertFalse(recommended["runtime_log"]["enabled"])
        self.assertEqual(protected["alerts"]["drowsy_cooldown_seconds"], 5.0)
        self.assertNotIn("seeded_macroevent_bridge", protected["hybrid_policy"])

    def test_feature_pipeline_uses_lightweight_dynamic_ear(self) -> None:
        from runtime.config import load_runtime_config
        from runtime.dynamic_ear import DynamicEAR
        from runtime.features import SignalFeaturePipeline

        config = load_runtime_config(str(ROOT / "configs" / "recommended.json"))
        pipeline = SignalFeaturePipeline(config)
        self.assertIsInstance(pipeline.dynamic_ear, DynamicEAR)

    def test_production_runtime_has_no_heavy_or_audio_dependencies(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        production_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in [ROOT / "web_server.py", *(ROOT / "runtime").rglob("*.py")]
        ).lower()
        for forbidden in ("mediapipe", "torch", "playsound", "pandas"):
            self.assertNotIn(forbidden, requirements)
            self.assertNotIn(f"import {forbidden}", production_sources)
            self.assertNotIn(f"from {forbidden}", production_sources)
    def test_production_perception_module_has_no_mediapipe_or_torch(self) -> None:
        source = (ROOT / "runtime" / "perception.py").read_text(encoding="utf-8")
        self.assertNotIn("import mediapipe", source)
        self.assertNotIn("import torch", source)
    def test_camera_engine_uses_production_window_aggregator(self) -> None:
        import inspect
        from runtime.engines import camera_model_engine

        source = inspect.getsource(camera_model_engine)
        self.assertIn("from runtime.window_features import", source)
        self.assertNotIn("tools.research.common", source)

if __name__ == "__main__":
    unittest.main()
