import json
from pathlib import Path
import unittest


class OptionIncomePageTest(unittest.TestCase):
    def test_static_contract_is_generated_from_dbt_mart(self):
        payload = json.loads(
            Path("data/option-income-comparison.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["schema_version"], "option-income-comparison.v1")
        self.assertEqual(
            payload["generated_from"], "main.mart_underlying_strategy_comparison"
        )
        groups = {}
        for row in payload["rows"]:
            groups.setdefault(row["underlying_ticker"], set()).add(row["ticker"])
        self.assertEqual(groups["NVDA"], {"NVDA", "NVII", "NVDY", "NVIT"})
        self.assertEqual(groups["TSLA"], {"TSLA", "TSII", "TSLY", "TEST"})

    def test_page_reads_only_canonical_comparison_artifact(self):
        html = Path("option-income/index.html").read_text(encoding="utf-8")
        self.assertIn("../data/option-income-comparison.json", html)
        self.assertNotIn("rex-growth-income-active-observations.json", html)
        self.assertNotIn("yieldmax-option-income-observations.json", html)
        self.assertIn("Option Income BI", html)
        self.assertIn('id="kpi-underlying-tr"', html)
        self.assertIn('id="bars"', html)
        self.assertIn('id="scatter"', html)
        self.assertIn('id="matrix"', html)
        self.assertIn("row.total_return_percent", html)
        self.assertIn("row.upside_capture_percent", html)
        self.assertIn("row.downside_capture_percent", html)
        self.assertIn("OBSERVED_SNAPSHOT_MARKET_DATA_GAP", html)
        self.assertNotIn("Distribution Rate は total return ではありません", html)

    def test_pages_workflow_packages_option_income_surface(self):
        workflow = Path(".github/workflows/weekly-update.yml").read_text(encoding="utf-8")
        self.assertIn("cp -R option-income _site/option-income", workflow)
        self.assertIn("option-income-comparison.json", workflow)


if __name__ == "__main__":
    unittest.main()
