import copy
import json
import unittest
from pathlib import Path

from scripts.validate_rex_fund_master import validate


class RexFundMasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(Path("data/rex-growth-income-funds.json").read_text(encoding="utf-8"))

    def test_repository_master_is_valid(self):
        self.assertEqual(validate(self.payload), [])

    def test_missing_fund_fails_closed(self):
        payload = copy.deepcopy(self.payload)
        payload["funds"].pop()
        self.assertTrue(validate(payload))

    def test_liquidated_fund_cannot_be_active(self):
        payload = copy.deepcopy(self.payload)
        next(f for f in payload["funds"] if f["ticker"] == "MSII")["lifecycle_status"] = "ACTIVE"
        self.assertTrue(any("MSII" in error for error in validate(payload)))

    def test_liquidation_date_is_required(self):
        payload = copy.deepcopy(self.payload)
        next(f for f in payload["funds"] if f["ticker"] == "COII").pop("liquidation_date")
        self.assertTrue(any("COII" in error for error in validate(payload)))

    def test_non_official_product_source_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        next(f for f in payload["funds"] if f["ticker"] == "NVII")["product_url"] = "https://example.com/nvii"
        self.assertTrue(any("NVII" in error for error in validate(payload)))


if __name__ == "__main__":
    unittest.main()
