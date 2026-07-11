from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def landmarks(closed: bool) -> dict[str, list[float]]:
    upper = 0.399 if closed else 0.38
    lower = 0.401 if closed else 0.42
    return {
        "33": [0.30, 0.40], "160": [0.32, upper], "158": [0.36, upper],
        "133": [0.40, 0.40], "153": [0.36, lower], "144": [0.32, lower],
        "362": [0.60, 0.40], "385": [0.62, upper], "387": [0.66, upper],
        "263": [0.70, 0.40], "373": [0.66, lower], "380": [0.62, lower],
        "61": [0.40, 0.62], "291": [0.60, 0.62], "13": [0.50, 0.60], "14": [0.50, 0.64],
        "1": [0.50, 0.48], "152": [0.50, 0.78], "468": [0.35, 0.40], "473": [0.65, 0.40],
    }


def replay(runtime, input_fps: int, sleepy_start: float = 3.0, sleepy_end: float = 9.0, duration: float = 12.0):
    session = runtime.create_session(source_mode="file", target_fps=input_fps)
    trace = []
    frame_count = int(duration * input_fps) + 1
    for index in range(frame_count):
        timestamp_ms = index * (1000.0 / input_fps)
        second = timestamp_ms / 1000.0
        packet = {
            "seq": index,
            "timestamp_ms": timestamp_ms,
            "width": 1000,
            "height": 800,
            "face_detected": True,
            "landmarks": landmarks(sleepy_start <= second < sleepy_end),
        }
        for grid_ms, virtual in session.normalizer.push(packet):
            session._process_virtual_frame(grid_ms, virtual)
            result = session.latest_result
            trace.append({
                "timestamp_ms": grid_ms,
                "state": result.state.value,
                "guard": result.debug.get("hybrid_guard", "warmup"),
                "probability": result.debug.get("sleepy_probability"),
            })
    return trace


class ResamplingParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from runtime.web_runtime import WinnerRuntime
        cls.runtime = WinnerRuntime(ROOT, profile_name="recommended")

    def test_uniform_30hz_engine_matches_protected_source_golden(self) -> None:
        from fsm import DrowsinessSignals
        from runtime.contracts import EngineContext
        from runtime.engines.registry import create_engine
        from runtime.web_runtime import WinnerRuntime

        protected = WinnerRuntime(ROOT, profile_name="protected")
        engine = create_engine("camera_hybrid", protected.config)
        engine.initialize(EngineContext(fps=30.0))
        golden = json.loads((ROOT / "tests" / "data" / "source_uniform_30hz_golden.json").read_text(encoding="utf-8"))

        for index, expected in enumerate(golden):
            severe = 100 <= index < 220
            signals = DrowsinessSignals(
                ear=0.08 if severe else 0.30,
                mar=0.80 if severe else 0.10,
                pitch=25.0 if severe else 0.0,
                pitch_velocity=8.0 if severe else 0.0,
                perclos=0.70 if severe else 0.02,
                perclos_short=0.75 if severe else 0.02,
                yawn_frequency=4 if severe else 0,
                blink_frequency=12 if severe else 0,
                gaze_stable=not severe,
                head_nod_detected=severe,
                eyes_closed_consecutive=20 if severe else 0,
                face_detected=True,
                ear_below_threshold=severe,
                mar_above_threshold=severe,
                pitch_above_threshold=severe,
            )
            actual = engine.update(signals)
            self.assertEqual(actual.state.value, expected["state"], index)
            self.assertEqual(actual.debug.get("hybrid_guard", "warmup"), expected["guard"], index)
            self.assertEqual(actual.alert_sound, expected["alert_sound"], index)
            self.assertEqual(actual.debug.get("runtime_alert_semantic", "standard"), expected["runtime_alert_semantic"], index)
            self.assertEqual(actual.label, expected["label"], index)
            actual_probability = actual.debug.get("sleepy_probability")
            if expected["probability"] is None:
                self.assertIsNone(actual_probability, index)
            else:
                self.assertAlmostEqual(actual_probability, expected["probability"], delta=1e-9, msg=str(index))
    def test_downsampled_landmark_trace_stays_within_runtime_guardrails(self) -> None:
        reference = replay(self.runtime, 30)
        reference_high = sum(row["state"] in {"DROWSY", "CRITICAL"} for row in reference)
        reference_onset = next((row["timestamp_ms"] for row in reference if row["state"] in {"DROWSY", "CRITICAL"}), None)

        for fps in (10, 15, 20):
            candidate = replay(self.runtime, fps)
            common = min(len(reference), len(candidate))
            agreement = sum(reference[index]["state"] == candidate[index]["state"] for index in range(common)) / common
            candidate_high = sum(row["state"] in {"DROWSY", "CRITICAL"} for row in candidate)
            candidate_onset = next((row["timestamp_ms"] for row in candidate if row["state"] in {"DROWSY", "CRITICAL"}), None)

            self.assertGreaterEqual(agreement, 0.95, fps)
            self.assertIsNotNone(reference_onset)
            self.assertIsNotNone(candidate_onset)
            self.assertLessEqual(abs(candidate_onset - reference_onset), 500.0, fps)
            self.assertGreaterEqual(candidate_high, reference_high * 0.98, fps)

    def test_clean_clip_does_not_gain_high_alert_states_when_downsampled(self) -> None:
        for fps in (10, 15, 20, 30):
            trace = replay(self.runtime, fps, sleepy_start=99.0, sleepy_end=100.0, duration=6.0)
            high = sum(row["state"] in {"DROWSY", "CRITICAL"} for row in trace)
            self.assertEqual(high, 0, fps)


if __name__ == "__main__":
    unittest.main()