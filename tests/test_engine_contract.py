import tempfile
import unittest
from pathlib import Path

from fsm import DrowsinessSignals, DrowsinessState
from runtime.config import default_runtime_config
from runtime.contracts import EngineContext
from runtime.engines.registry import create_engine


class FixedProbabilityModel:
    classes_ = [0, 1]

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, rows):
        return [[1.0 - self.probability, self.probability] for _row in rows]


class EngineContractTests(unittest.TestCase):
    def test_fsm_engine_contract(self):
        config = default_runtime_config()
        engine = create_engine("fsm", config)
        engine.initialize(EngineContext(fps=config.runtime.fps))

        result = engine.update(DrowsinessSignals())
        self.assertIsInstance(result.state, DrowsinessState)
        self.assertIsInstance(result.evidence, float)
        self.assertTrue(hasattr(result, "alert_sound"))

    def test_legacy_engine_contract(self):
        config = default_runtime_config()
        engine = create_engine("legacy", config)
        engine.initialize(EngineContext(fps=config.runtime.fps))

        signals = DrowsinessSignals(
            ear=0.1,
            mar=0.8,
            perclos=0.5,
            perclos_short=0.7,
            yawn_frequency=4,
            blink_frequency=15,
            head_nod_detected=True,
            eyes_closed_consecutive=25,
            ear_below_threshold=True,
            mar_above_threshold=True,
            pitch_above_threshold=True,
        )
        result = engine.update(signals)
        self.assertIn(result.state, list(DrowsinessState))
        self.assertGreaterEqual(result.evidence, 0.0)

    def test_camera_model_engine_contract_with_fixture_model(self):
        from joblib import dump
        from sklearn.ensemble import RandomForestClassifier

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "camera_model.joblib"
            model = RandomForestClassifier(n_estimators=5, random_state=1)
            model.fit(
                [
                    [0.30, 0.10, 0.0],
                    [0.28, 0.15, 1.0],
                    [0.12, 0.70, 2.0],
                    [0.10, 0.80, 3.0],
                ],
                [0, 0, 1, 1],
            )
            dump(
                {
                    "model": model,
                    "feature_columns": ["mean_ear", "perclos_60s", "fsm_state_mode"],
                    "probability_threshold": 0.5,
                    "feature_set": "camera",
                },
                model_path,
            )

            config = default_runtime_config()
            config.camera_model.model_path = str(model_path)
            config.camera_model.window_seconds = 0.2
            config.camera_model.min_window_seconds = 0.0
            config.camera_model.min_frames = 2
            engine = create_engine("camera_model", config)
            engine.initialize(EngineContext(fps=10.0))

            signals = DrowsinessSignals(
                ear=0.10,
                mar=0.80,
                perclos=0.75,
                perclos_short=0.80,
                yawn_frequency=2,
                head_nod_detected=True,
                eyes_closed_consecutive=15,
                ear_below_threshold=True,
                mar_above_threshold=True,
            )
            engine.update(signals)
            result = engine.update(signals)

            self.assertIn(result.state, list(DrowsinessState))
            self.assertIn("sleepy_probability", result.debug)
            self.assertEqual(result.debug["model_feature_columns"], ["mean_ear", "perclos_60s", "fsm_state_mode"])
            self.assertEqual(set(result.debug["model_feature_vector"]), {"mean_ear", "perclos_60s", "fsm_state_mode"})
            self.assertGreaterEqual(result.debug["sleepy_probability"], 0.0)
            self.assertLessEqual(result.debug["sleepy_probability"], 1.0)

    def test_camera_hybrid_contract_with_fixture_model(self):
        from joblib import dump

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "camera_hybrid.joblib"
            dump(
                {
                    "model": FixedProbabilityModel(0.90),
                    "feature_columns": ["mean_ear", "perclos_60s", "fsm_state_mode"],
                    "probability_threshold": 0.5,
                    "feature_set": "camera",
                },
                model_path,
            )

            config = default_runtime_config()
            config.camera_model.model_path = str(model_path)
            config.camera_model.window_seconds = 0.2
            config.camera_model.min_window_seconds = 0.0
            config.camera_model.min_frames = 2
            engine = create_engine("camera_hybrid", config)
            engine.initialize(EngineContext(fps=10.0))

            signals = DrowsinessSignals(ear=0.28, mar=0.10, perclos=0.02, perclos_short=0.02)
            engine.update(signals)
            result = engine.update(signals)

            self.assertIn(result.state, list(DrowsinessState))
            self.assertIn("sleepy_probability", result.debug)
            self.assertIn("ml_only_state", result.debug)
            self.assertIn("hybrid_guard", result.debug)
            self.assertIn("hybrid_clean_streak", result.debug)
            self.assertIn("hybrid_severe_streak", result.debug)
            self.assertIn("hybrid_window_features", result.debug)
            self.assertEqual(result.debug["model_feature_columns"], ["mean_ear", "perclos_60s", "fsm_state_mode"])
            self.assertEqual(set(result.debug["model_feature_vector"]), {"mean_ear", "perclos_60s", "fsm_state_mode"})


if __name__ == "__main__":
    unittest.main()
