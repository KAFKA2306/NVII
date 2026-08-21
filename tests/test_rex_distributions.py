from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.validate_rex_distributions import validate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/rex-growth-income-distributions.json"


class RexDistributionHistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(DATA.read_text(encoding="utf-8"))

    def test_repository_history_is_valid(self) -> None:
        self.assertEqual(validate(self.payload), [])
        by_ticker = {row["ticker"]: row for row in self.payload["funds"]}
        self.assertEqual(set(by_ticker), {"NVII", "TSII", "WMTI"})
        self.assertTrue(all(len(row["distributions"]) >= 12 for row in by_ticker.values()))
        self.assertEqual(sum(len(row["distributions"]) for row in by_ticker.values()), 36)

    def test_missing_twelfth_week_fails_closed(self) -> None:
        broken = copy.deepcopy(self.payload)
        broken["funds"][0]["distributions"] = broken["funds"][0]["distributions"][:11]
        self.assertTrue(validate(broken))

    def test_future_unrealized_row_fails_closed(self) -> None:
        broken = copy.deepcopy(self.payload)
        broken["funds"][0]["distributions"][0]["payable_date"] = "2026-08-22"
        self.assertTrue(validate(broken))

    def test_duplicate_distribution_fails_closed(self) -> None:
        broken = copy.deepcopy(self.payload)
        broken["funds"][1]["distributions"][1] = copy.deepcopy(broken["funds"][1]["distributions"][0])
        self.assertTrue(validate(broken))

    def test_non_positive_distribution_fails_closed(self) -> None:
        broken = copy.deepcopy(self.payload)
        broken["funds"][2]["distributions"][0]["distribution_per_share_usd"] = 0
        self.assertTrue(validate(broken))


if __name__ == "__main__":
    unittest.main()
