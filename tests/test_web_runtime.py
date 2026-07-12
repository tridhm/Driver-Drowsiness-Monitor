from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def landmark_fixture() -> dict[str, list[float]]:
    return {
        "33": [0.30, 0.40], "160": [0.32, 0.38], "158": [0.36, 0.38],
        "133": [0.40, 0.40], "153": [0.36, 0.42], "144": [0.32, 0.42],
        "362": [0.60, 0.40], "385": [0.62, 0.38], "387": [0.66, 0.38],
        "263": [0.70, 0.40], "373": [0.66, 0.42], "380": [0.62, 0.42],
        "61": [0.40, 0.62], "291": [0.60, 0.62],
        "13": [0.50, 0.60], "14": [0.50, 0.66],
        "1": [0.50, 0.48], "152": [0.50, 0.78],
        "468": [0.35, 0.40], "473": [0.65, 0.40],
    }


class TimestampNormalizerTests(unittest.TestCase):
    def test_uniform_30hz_uses_current_packet_on_matching_grid(self) -> None:
        from runtime.web_runtime import TimestampNormalizer

        normalizer = TimestampNormalizer()
        first = normalizer.push({"timestamp_ms": 0.0, "face_detected": True, "marker": "a"})
        second = normalizer.push({"timestamp_ms": 1000.0 / 30.0, "face_detected": True, "marker": "b"})

        self.assertEqual(first[0][1]["marker"], "a")
        self.assertEqual(second[0][1]["marker"], "b")
        self.assertAlmostEqual(second[0][0], 1000.0 / 30.0, places=6)

    def test_20fps_is_zero_order_held_on_30hz_grid(self) -> None:
        from runtime.web_runtime import TimestampNormalizer

        normalizer = TimestampNormalizer()
        normalizer.push({"timestamp_ms": 0.0, "face_detected": True, "marker": "a"})
        at_50 = normalizer.push({"timestamp_ms": 50.0, "face_detected": True, "marker": "b"})
        at_100 = normalizer.push({"timestamp_ms": 100.0, "face_detected": True, "marker": "c"})

        self.assertEqual([row[1]["marker"] for row in at_50], ["a"])
        self.assertEqual([row[1]["marker"] for row in at_100], ["b", "c"])

    def test_gap_over_250ms_emits_missing_face_instead_of_stale_landmarks(self) -> None:
        from runtime.web_runtime import TimestampNormalizer

        normalizer = TimestampNormalizer(max_hold_ms=250.0)
        normalizer.push({"timestamp_ms": 0.0, "face_detected": True, "marker": "a"})
        rows = normalizer.push({"timestamp_ms": 400.0, "face_detected": True, "marker": "b"})

        missing = [packet for timestamp, packet in rows if timestamp > 250.0 and timestamp < 400.0]
        self.assertTrue(missing)
        self.assertTrue(all(not packet["face_detected"] for packet in missing))
        self.assertEqual(rows[-1][1]["marker"], "b")

    def test_default_normalizer_preserves_session_across_1500ms_stall(self) -> None:
        from runtime.web_runtime import TimestampNormalizer

        normalizer = TimestampNormalizer()
        normalizer.push({"timestamp_ms": 0.0, "face_detected": True, "marker": "a"})
        rows = normalizer.push({"timestamp_ms": 1500.0, "face_detected": True, "marker": "b"})

        self.assertGreater(len(rows), 32)
        self.assertTrue(any(not packet["face_detected"] for timestamp, packet in rows if timestamp > 250.0))
        self.assertEqual(normalizer.last_input_ms, 1500.0)

    def test_excessive_timestamp_gap_is_rejected_before_virtual_expansion(self) -> None:
        from runtime.web_runtime import ProtocolError, TimestampNormalizer

        normalizer = TimestampNormalizer(max_gap_ms=1000.0)
        normalizer.push({"timestamp_ms": 0.0, "face_detected": False})

        with self.assertRaises(ProtocolError) as caught:
            normalizer.push({"timestamp_ms": 2000.0, "face_detected": False})

        self.assertEqual(caught.exception.status_code, 409)
        self.assertIn("timestamp gap", str(caught.exception).lower())


class AlertCommandTests(unittest.TestCase):
    def test_double_alert_respects_recommended_cooldown(self) -> None:
        from runtime.web_runtime import AlertCommandController

        controller = AlertCommandController(cooldown_seconds=15.0)
        self.assertEqual(controller.update("double", now=0.0), "double")
        self.assertEqual(controller.update("double", now=5.0), "none")
        self.assertEqual(controller.update("double", now=15.0), "double")

    def test_continuous_alert_has_explicit_start_and_stop(self) -> None:
        from runtime.web_runtime import AlertCommandController

        controller = AlertCommandController(cooldown_seconds=15.0)
        self.assertEqual(controller.update("continuous", now=0.0), "continuous_start")
        self.assertEqual(controller.update("continuous", now=0.1), "none")
        self.assertEqual(controller.update("none", now=0.2), "continuous_stop")


class WinnerSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        from runtime.web_runtime import WinnerRuntime

        self.runtime = WinnerRuntime(ROOT, profile_name="recommended")

    def test_runtime_loads_recommended_profile_and_validated_model(self) -> None:
        self.assertEqual(self.runtime.profile_name, "recommended")
        self.assertEqual(self.runtime.config.decision_engine, "camera_hybrid")
        from runtime.model_bundle import FastIsotonicBinaryPredictor

        self.assertEqual(self.runtime.bundle.sha256, "8958d2d4dd0a0757b5a922adb11df263144e253873909ac8816cd26c248bc89c")
        self.assertEqual(self.runtime.config.camera_model.probability_threshold, 0.55)
        session = self.runtime.create_session(source_mode="camera", target_fps=20)
        self.assertIsInstance(session.engine.predictor, FastIsotonicBinaryPredictor)

    def test_session_batch_is_idempotent_and_out_of_order_is_rejected(self) -> None:
        from runtime.web_runtime import ProtocolError

        session = self.runtime.create_session(source_mode="camera", target_fps=20)
        payload = {
            "batch_seq": 1,
            "frames": [{
                "seq": 1,
                "timestamp_ms": 0.0,
                "width": 1000,
                "height": 800,
                "face_detected": True,
                "landmarks": landmark_fixture(),
            }],
        }
        first = session.process_batch(payload)
        duplicate = session.process_batch(payload)
        self.assertEqual(first, duplicate)
        changed_replay = {**payload, "frames": [{**payload["frames"][0], "timestamp_ms": 1.0}]}
        with self.assertRaises(ProtocolError):
            session.process_batch(changed_replay)
        self.assertEqual(first["profile"], "recommended")
        self.assertEqual(first["model_hash"], self.runtime.bundle.sha256)
        self.assertIn(first["audio_command"], {"none", "double", "continuous_start", "continuous_stop"})

        with self.assertRaises(ProtocolError):
            session.process_batch({"batch_seq": 2, "frames": [{**payload["frames"][0], "seq": 0, "timestamp_ms": 1.0}]})

    def test_failed_batch_rolls_back_processing_state(self) -> None:
        session = self.runtime.create_session(source_mode="camera", target_fps=20)
        original_adapter = session.adapter.from_normalized
        calls = 0

        def fail_after_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise RuntimeError("synthetic adapter failure")
            return original_adapter(*args, **kwargs)

        session.adapter.from_normalized = fail_after_first
        before = {
            "input": session.input_frame_count,
            "virtual": session.virtual_frame_count,
            "calibration": session.features.state.calibration_count,
            "frame_index": session.engine.frame_index,
            "normalizer_next": session.normalizer.next_grid_ms,
            "batch_events": list(session.batch_limiter.events),
            "frame_events": list(session.frame_limiter.events),
        }
        with self.assertRaises(RuntimeError):
            session.process_batch({
                "batch_seq": 1,
                "frames": [
                    {
                        "seq": 1, "timestamp_ms": 0.0, "width": 1000, "height": 800,
                        "face_detected": True, "landmarks": landmark_fixture(),
                    },
                    {
                        "seq": 2, "timestamp_ms": 100.0, "width": 1000, "height": 800,
                        "face_detected": True, "landmarks": landmark_fixture(),
                    },
                ],
            })
        after = {
            "input": session.input_frame_count,
            "virtual": session.virtual_frame_count,
            "calibration": session.features.state.calibration_count,
            "frame_index": session.engine.frame_index,
            "normalizer_next": session.normalizer.next_grid_ms,
            "batch_events": list(session.batch_limiter.events),
            "frame_events": list(session.frame_limiter.events),
        }
        self.assertEqual(after, before)
    def test_reset_restarts_calibration_and_preserves_session_identity(self) -> None:
        session = self.runtime.create_session(source_mode="file", target_fps=30)
        session.process_batch({
            "batch_seq": 1,
            "frames": [{
                "seq": 1, "timestamp_ms": 0.0, "width": 1000, "height": 800,
                "face_detected": True, "landmarks": landmark_fixture(),
            }],
        })
        session_id = session.session_id
        reset = session.reset()
        self.assertEqual(reset["session_id"], session_id)
        self.assertEqual(reset["calibration"]["valid_face_frames"], 0)
        self.assertEqual(reset["calibration"]["runtime_target_frames"], 60)
        self.assertEqual(reset["calibration"]["application_warmup_seconds"], 3.0)
        self.assertEqual(reset["calibration"]["dynamic_total_progress"], 0.0)


if __name__ == "__main__":
    unittest.main()