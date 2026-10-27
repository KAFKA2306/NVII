"""Regression for transient yfinance cache locks and incomplete underlying data."""
import tempfile
import unittest

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scripts.unified_rex_system_english import UnifiedREXSystemEnglish
except ModuleNotFoundError as exc:
    if exc.name not in {"pandas", "numpy", "yfinance", "matplotlib", "seaborn"}:
        raise
    UnifiedREXSystemEnglish = None


@unittest.skipIf(UnifiedREXSystemEnglish is None, "analyzer dependencies not installed")
class WeeklyPartialDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.system = UnifiedREXSystemEnglish(output_dir=self.tmp.name)

    def test_cache_lock_retries_missing_history_after_workers_finish(self):
        attempts = {}

        def fetch(symbol, _period):
            attempts[symbol] = attempts.get(symbol, 0) + 1
            if symbol == "NVDA" and attempts[symbol] == 1:
                return pd.DataFrame()
            return pd.DataFrame({"Close": [1.0]})

        self.system._fetch_single_symbol = fetch
        result = self.system.fetch_all_data_parallel()
        self.assertEqual(attempts["NVDA"], 2)
        self.assertFalse(result["NVII"]["underlying_data"].empty)

    def test_persistent_missing_history_fails_instead_of_publishing_partial_data(self):
        def fetch(symbol, _period):
            return pd.DataFrame() if symbol == "NVDA" else pd.DataFrame({"Close": [1.0]})

        self.system._fetch_single_symbol = fetch
        with self.assertRaisesRegex(RuntimeError, "NVDA"):
            self.system.fetch_all_data_parallel()

    def test_underlying_chart_matches_parent_etf_when_first_underlying_is_missing(self):
        def row(symbol, kind, underlying, price):
            return {
                "Symbol": symbol, "Type": kind, "Underlying": underlying,
                "Total_Return": 0.1, "Dividend_Yield": 0.03,
                "Sharpe_Ratio": 1.0, "Volatility": 0.2,
                "Max_Drawdown": -0.1, "Leverage_Estimate": 1.2,
                "Weekly_Dividends": 4, "Avg_Weekly_Div": 0.2,
                "Dividend_Consistency": 0.8, "Current_Price": price,
                "Excess_vs_Underlying": 0.02, "Overall_Rank": 1,
            }

        self.system.comparative_df = pd.DataFrame([
            row("NVII", "REX ETF", "NVDA", 20),
            row("TSII", "REX ETF", "TSLA", 30),
            row("TSLA", "Underlying", "TSLA", 300),
        ])
        self.system.processed_data = {}
        figure = self.system.create_comprehensive_english_dashboard()
        self.addCleanup(plt.close, figure)
        # Two ETF bars and one underlying bar, located next to TSII (index 1).
        bars = figure.axes[1].patches
        self.assertEqual(len(bars), 3)
        self.assertAlmostEqual(bars[-1].get_x() + bars[-1].get_width() / 2, 1.175)
        # The scatter plot must use one underlying color for one underlying point.
        self.assertEqual(len(figure.axes[7].collections[1].get_offsets()), 1)


if __name__ == "__main__":
    unittest.main()
