from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.validate_rex_active_observations import validate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/rex-growth-income-active-observations.json"


class ActiveRexObservationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(DATA.read_text(encoding="utf-8"))

    def test_repository_snapshot_is_valid(self) -> None:
        self.assertEqual(validate(self.payload), [])
        by_ticker = {row["ticker"]: row for row in self.payload["funds"]}
        self.assertEqual(set(by_ticker), {"NVII", "TSII", "WMTI"})
        self.assertFalse(any(row["distribution"]["distribution_rate_is_total_return"] for row in by_ticker.values()))
        self.assertEqual(by_ticker["NVII"]["observation"]["number_of_holdings"], 8)
        self.assertEqual(by_ticker["TSII"]["observation"]["number_of_holdings"], 6)
        self.assertEqual(by_ticker["WMTI"]["observation"]["number_of_holdings"], 6)

    def test_missing_active_fund_fails_closed(self) -> None:
        broken = copy.deepcopy(self.payload)
        broken["funds"] = broken["funds"][:-1]
        self.assertTrue(validate(broken))

    def test_distribution_rate_cannot_be_total_return(self) -> None:
        broken = copy.deepcopy(self.payload)
        broken["funds"][0]["distribution"]["distribution_rate_is_total_return"] = True
        self.assertTrue(validate(broken))

    def test_holding_count_and_weights_must_reconcile(self) -> None:
        broken = copy.deepcopy(self.payload)
        broken["funds"][1]["holdings"]["rows"].pop()
        self.assertTrue(validate(broken))


if __name__ == "__main__":
    unittest.main()
