import csv
import json
from pathlib import Path
import unittest


class OptionIncomeSourcesTest(unittest.TestCase):
    def test_yieldmax_snapshot_is_official_and_internally_consistent(self):
        payload = json.loads(
            Path("data/yieldmax-option-income-observations.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["schema_version"], "yieldmax-option-income-observations.v1")
        funds = {fund["ticker"]: fund for fund in payload["funds"]}
        self.assertEqual(set(funds), {"NVDY", "NVIT", "TSLY", "TEST"})
        for ticker, fund in funds.items():
            self.assertTrue(fund["source_url"].startswith("https://yieldmaxetfs.com/"))
            rows = fund["distributions"]
            self.assertGreaterEqual(len(rows), 3)
            newest = max(row["declaration_date"] for row in rows)
            self.assertEqual(newest, fund["distribution"]["as_of"], ticker)
            roc = fund["distribution"]["estimated_roc_percent"]
            self.assertGreaterEqual(roc, 0)
            self.assertLessEqual(roc, 100)

    def test_instrument_master_contains_required_underlying_groups(self):
        with Path("seeds/instrument_master.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        groups = {}
        for row in rows:
            groups.setdefault(row["underlying_ticker"], set()).add(row["ticker"])
        self.assertEqual(groups["NVDA"], {"NVDA", "NVII", "NVDY", "NVIT"})
        self.assertEqual(groups["TSLA"], {"TSLA", "TSII", "TSLY", "TEST"})


if __name__ == "__main__":
    unittest.main()
