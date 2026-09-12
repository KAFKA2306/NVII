from pathlib import Path
import unittest


class RexComparePageTest(unittest.TestCase):
    def test_compare_page_uses_canonical_data_and_source_links(self):
        html = Path("compare/index.html").read_text(encoding="utf-8")
        self.assertIn("../data/rex-growth-income-active-observations.json", html)
        self.assertIn("../data/rex-growth-income-distributions.json", html)
        self.assertIn("Distribution Rate", html)
        self.assertIn("total return ではありません", html)
        self.assertIn("REX公式一次情報", html)
        self.assertIn("NVII / TSII / WMTI", html)

    def test_compare_page_exposes_both_input_observation_times(self):
        html = Path("compare/index.html").read_text(encoding="utf-8")
        self.assertIn('id="observation-status"', html)
        self.assertIn("observations.observed_at", html)
        self.assertIn("distributions.observed_at", html)
        self.assertIn("基準値", html)
        self.assertIn("分配履歴", html)
        self.assertIn("それぞれの観測日時点の正本を組み合わせた比較です", html)
        self.assertIn("観測日時を確認できません。現在値として扱わないでください。", html)

    def test_compare_page_updates_when_selection_changes(self):
        html = Path("compare/index.html").read_text(encoding="utf-8")
        self.assertIn("el.addEventListener('change',render)", html)
        self.assertNotIn('id="compare"', html)
        self.assertIn('<label for="left">比較1</label>', html)
        self.assertIn('<label for="right">比較2</label>', html)

    def test_compare_page_records_alert_demand_as_github_issue(self):
        html = Path("compare/index.html").read_text(encoding="utf-8")
        self.assertIn("週次アラートを希望する", html)
        self.assertIn("https://github.com/KAFKA2306/NVII/issues/new?", html)
        self.assertIn("実際に送信された要望だけを記録します", html)

    def test_compare_page_does_not_embed_model_outputs(self):
        html = Path("compare/index.html").read_text(encoding="utf-8").lower()
        self.assertNotIn("value at risk", html)
        self.assertNotIn("expected shortfall", html)
        self.assertNotIn("black-scholes", html)

    def test_pages_workflow_packages_compare_and_validates_pair(self):
        workflow = Path(".github/workflows/weekly-update.yml").read_text(encoding="utf-8")
        self.assertIn("cp -R compare _site/compare", workflow)
        self.assertIn("rex-growth-income-active-observations.json", workflow)
        self.assertIn("rex-growth-income-distributions.json", workflow)
        self.assertIn("validate_rex_compare_pair.py", workflow)
        self.assertIn("path: ./_site", workflow)
        self.assertIn("Verify deployed compare page", workflow)
        self.assertIn("- 'compare/**'", workflow)
        self.assertIn("- 'data/rex-growth-income-active-observations.json'", workflow)
        self.assertIn("- 'data/rex-growth-income-distributions.json'", workflow)


if __name__ == "__main__":
    unittest.main()
