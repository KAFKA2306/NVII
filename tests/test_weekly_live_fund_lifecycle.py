"""The weekly live-price fetch must respect the canonical fund lifecycle."""
import json
import tempfile
import unittest
from pathlib import Path

# The lightweight compare-pair job does not install analyzer dependencies.
# The weekly dashboard contract job installs them and runs this integration test.
try:
    from scripts.unified_rex_system_english import UnifiedREXSystemEnglish
except ModuleNotFoundError as exc:
    if exc.name not in {'pandas', 'numpy', 'yfinance', 'matplotlib', 'seaborn'}:
        raise
    UnifiedREXSystemEnglish = None


class WeeklyLiveFundLifecycleTests(unittest.TestCase):
    @unittest.skipIf(UnifiedREXSystemEnglish is None, 'analyzer dependencies not installed in lightweight job')
    def test_only_canonical_active_funds_are_enabled_for_live_analysis(self):
        master = json.loads(Path('data/rex-growth-income-funds.json').read_text(encoding='utf-8'))
        statuses = {fund['ticker']: fund['lifecycle_status'] for fund in master['funds']}
        with tempfile.TemporaryDirectory() as directory:
            system = UnifiedREXSystemEnglish(output_dir=directory)
            actual = {ticker for ticker, config in system.rex_family.items() if config['active']}
        expected = {ticker for ticker in system.rex_family if statuses[ticker] == 'ACTIVE'}
        self.assertEqual(actual, expected)
        self.assertNotIn('MSII', actual)
        self.assertNotIn('COII', actual)
        self.assertEqual(actual, {'NVII', 'TSII'})


if __name__ == '__main__':
    unittest.main()
