from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.validate_rex_compare_pair import PairValidationError, validate_pair


class RexComparePairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations = json.loads(
            Path("data/rex-growth-income-active-observations.json").read_text(encoding="utf-8")
        )
        cls.distributions = json.loads(
            Path("data/rex-growth-income-distributions.json").read_text(encoding="utf-8")
        )

    def test_current_pair_passes(self):
        metadata = validate_pair(self.observations, self.distributions)
        self.assertEqual(metadata["schema_version"], "rex-compare-pair.v1")
        self.assertEqual(metadata["tickers"], ["NVII", "TSII", "WMTI"])
        self.assertEqual(
            metadata["active_observations"]["observed_at"], self.observations["observed_at"]
        )
        self.assertEqual(
            metadata["distribution_history"]["observed_at"], self.distributions["observed_at"]
        )

    def test_missing_ticker_is_rejected(self):
        distributions = deepcopy(self.distributions)
        distributions["funds"] = [
            fund for fund in distributions["funds"] if fund["ticker"] != "WMTI"
        ]
        with self.assertRaisesRegex(PairValidationError, "ticker set mismatch"):
            validate_pair(self.observations, distributions)

    def test_source_identity_mismatch_is_rejected(self):
        distributions = deepcopy(self.distributions)
        distributions["funds"][0]["source_url"] = "https://example.invalid/nvii/"
        with self.assertRaisesRegex(PairValidationError, "source identity mismatch"):
            validate_pair(self.observations, distributions)

    def test_newer_history_than_active_snapshot_is_rejected(self):
        distributions = deepcopy(self.distributions)
        distributions["funds"][0]["distributions"][0]["declaration_date"] = "2026-08-24"
        with self.assertRaisesRegex(PairValidationError, "newest realized declaration_date"):
            validate_pair(self.observations, distributions)

    def test_same_pair_produces_identical_metadata(self):
        first = validate_pair(self.observations, self.distributions)
        second = validate_pair(deepcopy(self.observations), deepcopy(self.distributions))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
