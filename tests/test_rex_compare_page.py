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

    def test_compare_page_does_not_embed_model_outputs(self):
        html = Path("compare/index.html").read_text(encoding="utf-8").lower()
        self.assertNotIn("value at risk", html)
        self.assertNotIn("expected shortfall", html)
        self.assertNotIn("black-scholes", html)

    def test_pages_workflow_packages_compare_and_canonical_data(self):
        workflow = Path(".github/workflows/weekly-update.yml").read_text(encoding="utf-8")
        self.assertIn("cp -R compare _site/compare", workflow)
        self.assertIn("rex-growth-income-active-observations.json", workflow)
        self.assertIn("rex-growth-income-distributions.json", workflow)
        self.assertIn("path: ./_site", workflow)
        self.assertIn("Verify deployed compare page", workflow)


if __name__ == "__main__":
    unittest.main()
