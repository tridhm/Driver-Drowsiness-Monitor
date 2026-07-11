from __future__ import annotations

import unittest
from pathlib import Path

from tools.load_acceptance import nearest_rank_percentile, run


class LoadAcceptanceTests(unittest.TestCase):
    def test_nearest_rank_p95_uses_the_29th_value_for_30_samples(self) -> None:
        self.assertEqual(nearest_rank_percentile(list(range(1, 31)), 0.95), 29)

    def test_short_smoke_can_keep_production_rate_limits_enabled(self) -> None:
        result = run(
            Path(__file__).resolve().parents[1],
            sessions=1,
            duration_seconds=1,
            input_fps=10,
            batch_size=4,
            enforce_production_limits=True,
        )

        self.assertEqual(result["results"][0]["http_statuses"], [200])
        self.assertIn("enforced", result["rate_limit_note"].lower())


if __name__ == "__main__":
    unittest.main()
