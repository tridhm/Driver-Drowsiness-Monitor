from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"


class WinnerEvidenceGateTests(unittest.TestCase):
    def test_canonical_heldout_356_metrics_remain_protected_winner(self) -> None:
        with (DATA / "winner_heldout_356_system_predictions.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        truth = [int(row["target_sleepy"]) for row in rows]
        predicted = [int(row["hybrid_predicted_sleepy"]) for row in rows]
        tp = sum(t == 1 and p == 1 for t, p in zip(truth, predicted))
        tn = sum(t == 0 and p == 0 for t, p in zip(truth, predicted))
        fp = sum(t == 0 and p == 1 for t, p in zip(truth, predicted))
        fn = sum(t == 1 and p == 0 for t, p in zip(truth, predicted))
        balanced_accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2.0

        self.assertEqual(len(rows), 356)
        self.assertEqual((tp, tn, fp, fn), (87, 214, 18, 37))
        self.assertAlmostEqual(balanced_accuracy, 0.812013, places=6)

    def test_recommended_package_keeps_silent_bridge_audio_gates_clean(self) -> None:
        gate = json.loads((DATA / "recommended_silent_bridge_gate_summary.json").read_text(encoding="utf-8"))
        decision = json.loads((DATA / "recommended_package_decision_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(gate["clip_pair_count"], 4)
        self.assertEqual(gate["paired_frames"], 33600)
        self.assertEqual(gate["visual_only_drowsy_bridge_rows"], 18879)
        self.assertEqual(gate["audible_delta"], 0)
        self.assertEqual(gate["actual_double_fire_delta"], 0)
        self.assertEqual(gate["double_alert_suppressed_delta"], 0)
        self.assertEqual((gate["baseline_model_unsupported_hold_rows_total"], gate["candidate_model_unsupported_hold_rows_total"]), (19095, 6366))
        self.assertEqual(decision["final_decision"], "recommended_silent_bridge_packaged")
        self.assertFalse(decision["root_config_changed"])


if __name__ == "__main__":
    unittest.main()