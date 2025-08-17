#!/usr/bin/env python3
"""
NVIDIA (NVDA) Historical Volatility Analysis Script
Calculates and visualizes historical volatility trends for NVDA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance for data fetching
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("yfinance not available, using simulated data")

class NVDAVolatilityAnalyzer:
    def __init__(self):
        self.symbol = "NVDA"
        self.current_price = 140.0  # Current estimate
        
    def fetch_nvda_data(self, period="2y"):
        """
        Fetch NVDA historical price data
        """
        if HAS_YFINANCE:
            try:
                ticker = yf.Ticker(self.symbol)
                data = ticker.history(period=period)
                return data
            except Exception as e:
                print(f"Error fetching data: {e}")
                return self.generate_simulated_data()
        else:
            return self.generate_simulated_data()
    
    def generate_simulated_data(self):
        """
        Generate realistic simulated NVDA price data for demonstration
        """
        np.random.seed(42)  # For reproducible results
        
        # Create date range for 2 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulate realistic NVDA price movement
        n_days = len(dates)
        
        # Base parameters
        initial_price = 80.0
        annual_drift = 0.15  # 15% annual growth
        base_volatility = 0.35  # 35% base volatility
        
        # Generate price path with regime changes
        returns = []
        current_vol = base_volatility
        
        for i in range(n_days):
            # Add volatility regime changes
            if i > 200 and i < 300:  # High volatility period (AI boom)
                current_vol = 0.55
            elif i > 400 and i < 500:  # Correction period
                current_vol = 0.65
            elif i > 600:  # Recent stabilization
                current_vol = 0.40
            else:
                current_vol = base_volatility
            
            # Generate daily return
            daily_return = np.random.normal(
                annual_drift / 365,  # Daily drift
                current_vol / np.sqrt(365)  # Daily volatility
            )
            returns.append(daily_return)
        
        # Convert to price series
        log_returns = np.array(returns)
        log_prices = np.log(initial_price) + np.cumsum(log_returns)
        prices = np.exp(log_prices)
        
        # Scale to end near current price
        final_price = prices[-1]
        scaling_factor = self.current_price / final_price
        prices = prices * scaling_factor
        
        # Create DataFrame similar to yfinance format
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.01, n_days)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.02, n_days))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, n_days))),
            'Close': prices,
            'Volume': np.random.lognormal(15, 0.5, n_days).astype(int)
        }, index=dates)
        
        # Ensure High >= Close >= Low
        data['High'] = np.maximum(data['High'], data['Close'])
        data['Low'] = np.minimum(data['Low'], data['Close'])
        
        return data
    
    def calculate_realized_volatility(self, prices, window_days=30):
        """
        Calculate realized volatility using different time windows
        """
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))
        
        # Calculate rolling volatility (annualized)
        rolling_vol = log_returns.rolling(window=window_days).std() * np.sqrt(252)
        
        return rolling_vol
    
    def calculate_garman_klass_volatility(self, data, window_days=30):
        """
        Calculate Garman-Klass volatility estimator (more efficient than close-to-close)
        """
        # Garman-Klass estimator
        log_hl = np.log(data['High'] / data['Low'])
        log_co = np.log(data['Close'] / data['Open'])
        
        gk_var = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        gk_vol = np.sqrt(gk_var.rolling(window=window_days).mean() * 252)
        
        return gk_vol
    
    def calculate_parkinson_volatility(self, data, window_days=30):
        """
        Calculate Parkinson volatility estimator (uses high-low range)
        """
        log_hl = np.log(data['High'] / data['Low'])
        park_var = log_hl**2 / (4 * np.log(2))
        park_vol = np.sqrt(park_var.rolling(window=window_days).mean() * 252)
        
        return park_vol
    
    def analyze_volatility_regimes(self, volatility_series):
        """
        Identify volatility regimes using quantile-based approach
        """
        vol_clean = volatility_series.dropna()
        
        # Define regime thresholds
        low_threshold = vol_clean.quantile(0.25)
        high_threshold = vol_clean.quantile(0.75)
        
        regimes = pd.Series(index=volatility_series.index, dtype='object')
        regimes[volatility_series <= low_threshold] = 'Low Volatility'
        regimes[(volatility_series > low_threshold) & (volatility_series <= high_threshold)] = 'Medium Volatility'
        regimes[volatility_series > high_threshold] = 'High Volatility'
        
        return regimes, low_threshold, high_threshold
    
    def plot_volatility_analysis(self, data):
        """
        Create comprehensive volatility analysis plots
        """
        # Calculate different volatility measures
        vol_30d = self.calculate_realized_volatility(data['Close'], 30)
        vol_60d = self.calculate_realized_volatility(data['Close'], 60)
        vol_90d = self.calculate_realized_volatility(data['Close'], 90)
        gk_vol = self.calculate_garman_klass_volatility(data, 30)
        park_vol = self.calculate_parkinson_volatility(data, 30)
        
        # Analyze regimes
        regimes, low_thresh, high_thresh = self.analyze_volatility_regimes(vol_30d)
        
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Price and Volume
        plt.subplot(3, 2, 1)
        plt.plot(data.index, data['Close'], 'blue', linewidth=1.5, label='NVDA Price')
        plt.ylabel('Price ($)')
        plt.title('NVDA Stock Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Volume subplot
        ax_vol = plt.gca().twinx()
        ax_vol.bar(data.index, data['Volume'], alpha=0.3, color='gray', width=1)
        ax_vol.set_ylabel('Volume')
        
        # Plot 2: Multiple Volatility Measures
        plt.subplot(3, 2, 2)
        plt.plot(data.index, vol_30d * 100, 'red', linewidth=2, label='30-day Realized Vol')
        plt.plot(data.index, vol_60d * 100, 'orange', linewidth=1.5, label='60-day Realized Vol')
        plt.plot(data.index, vol_90d * 100, 'purple', linewidth=1.5, label='90-day Realized Vol')
        
        plt.axhline(y=low_thresh * 100, color='green', linestyle='--', alpha=0.7, label=f'Low Regime ({low_thresh*100:.1f}%)')
        plt.axhline(y=high_thresh * 100, color='red', linestyle='--', alpha=0.7, label=f'High Regime ({high_thresh*100:.1f}%)')
        
        plt.ylabel('Volatility (%)')
        plt.title('NVDA Realized Volatility (Multiple Time Periods)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Alternative Volatility Estimators
        plt.subplot(3, 2, 3)
        plt.plot(data.index, vol_30d * 100, 'blue', linewidth=2, label='Close-to-Close')
        plt.plot(data.index, gk_vol * 100, 'green', linewidth=2, label='Garman-Klass')
        plt.plot(data.index, park_vol * 100, 'red', linewidth=2, label='Parkinson')
        
        plt.ylabel('Volatility (%)')
        plt.title('NVDA Volatility Estimators Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Volatility Distribution
        plt.subplot(3, 2, 4)
        vol_data = vol_30d.dropna() * 100
        plt.hist(vol_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(vol_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {vol_data.mean():.1f}%')
        plt.axvline(vol_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {vol_data.median():.1f}%')
        
        plt.xlabel('Volatility (%)')
        plt.ylabel('Frequency')
        plt.title('NVDA 30-Day Volatility Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Rolling Correlation with Market (simulated)
        plt.subplot(3, 2, 5)
        # Simulate market correlation
        market_vol = vol_30d * (0.8 + 0.4 * np.random.random(len(vol_30d)))
        correlation = vol_30d.rolling(60).corr(market_vol)
        
        plt.plot(data.index, correlation, 'purple', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=correlation.mean(), color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean Correlation: {correlation.mean():.2f}')
        
        plt.ylabel('Correlation')
        plt.title('NVDA Volatility Market Correlation (60-day rolling)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Volatility Regime Analysis
        plt.subplot(3, 2, 6)
        regime_colors = {'Low Volatility': 'green', 'Medium Volatility': 'orange', 'High Volatility': 'red'}
        
        for regime in ['Low Volatility', 'Medium Volatility', 'High Volatility']:
            mask = regimes == regime
            if mask.any():
                plt.scatter(data.index[mask], vol_30d[mask] * 100, 
                          c=regime_colors[regime], label=regime, alpha=0.6, s=20)
        
        plt.ylabel('30-day Volatility (%)')
        plt.title('NVDA Volatility Regime Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for i in range(1, 7):
            plt.subplot(3, 2, i)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/kafka/projects/NVII/nvda_volatility_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return vol_30d, vol_60d, vol_90d, gk_vol, park_vol, regimes
    
    def generate_volatility_statistics(self, vol_30d, vol_60d, vol_90d, gk_vol, park_vol, regimes):
        """
        Generate comprehensive volatility statistics
        """
        print("\n" + "="*70)
        print("NVIDIA (NVDA) VOLATILITY ANALYSIS SUMMARY")
        print("="*70)
        print(f"Analysis Period: {vol_30d.index[0].strftime('%Y-%m-%d')} to {vol_30d.index[-1].strftime('%Y-%m-%d')}")
        print(f"Current NVDA Price: ${self.current_price}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Clean data for statistics
        vol_30_clean = vol_30d.dropna() * 100
        vol_60_clean = vol_60d.dropna() * 100
        vol_90_clean = vol_90d.dropna() * 100
        gk_clean = gk_vol.dropna() * 100
        park_clean = park_vol.dropna() * 100
        
        print("\nVOLATILITY STATISTICS (Annualized %):")
        print("-" * 50)
        
        volatility_data = {
            '30-Day Realized': vol_30_clean,
            '60-Day Realized': vol_60_clean,
            '90-Day Realized': vol_90_clean,
            'Garman-Klass': gk_clean,
            'Parkinson': park_clean
        }
        
        for name, vol_series in volatility_data.items():
            if len(vol_series) > 0:
                print(f"\n{name} Volatility:")
                print(f"  Current: {vol_series.iloc[-1]:.1f}%")
                print(f"  Mean: {vol_series.mean():.1f}%")
                print(f"  Median: {vol_series.median():.1f}%")
                print(f"  Std Dev: {vol_series.std():.1f}%")
                print(f"  Min: {vol_series.min():.1f}%")
                print(f"  Max: {vol_series.max():.1f}%")
                print(f"  25th Percentile: {vol_series.quantile(0.25):.1f}%")
                print(f"  75th Percentile: {vol_series.quantile(0.75):.1f}%")
        
        print("\nVOLATILITY REGIME ANALYSIS:")
        print("-" * 50)
        regime_counts = regimes.value_counts()
        total_days = len(regimes.dropna())
        
        for regime, count in regime_counts.items():
            percentage = (count / total_days) * 100
            print(f"{regime}: {count} days ({percentage:.1f}%)")
        
        print("\nRECENT VOLATILITY TRENDS (Last 30 Days):")
        print("-" * 50)
        recent_vol = vol_30_clean.tail(30)
        if len(recent_vol) > 1:
            trend = "Increasing" if recent_vol.iloc[-1] > recent_vol.iloc[0] else "Decreasing"
            change = recent_vol.iloc[-1] - recent_vol.iloc[0]
            print(f"Trend: {trend}")
            print(f"Change: {change:+.1f} percentage points")
            print(f"Volatility of Volatility: {recent_vol.std():.1f}%")
        
        print("\nVOLATILITY RANKINGS:")
        print("-" * 50)
        current_vol = vol_30_clean.iloc[-1]
        percentile = (vol_30_clean <= current_vol).mean() * 100
        print(f"Current 30-day volatility is at the {percentile:.1f}th percentile historically")
        
        if percentile < 25:
            print("Current volatility is BELOW AVERAGE (low volatility regime)")
        elif percentile > 75:
            print("Current volatility is ABOVE AVERAGE (high volatility regime)")
        else:
            print("Current volatility is AVERAGE (medium volatility regime)")

def main():
    analyzer = NVDAVolatilityAnalyzer()
    
    print("Fetching NVDA historical data...")
    data = analyzer.fetch_nvda_data(period="2y")
    
    print("Calculating volatility measures...")
    vol_30d, vol_60d, vol_90d, gk_vol, park_vol, regimes = analyzer.plot_volatility_analysis(data)
    
    print("Generating volatility statistics...")
    analyzer.generate_volatility_statistics(vol_30d, vol_60d, vol_90d, gk_vol, park_vol, regimes)
    
    print("\nAnalysis complete! Charts saved as 'nvda_volatility_analysis.png'")

if __name__ == "__main__":
    main()