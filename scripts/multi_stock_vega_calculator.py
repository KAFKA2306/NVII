#!/usr/bin/env python3
"""
Multi-Stock Vega Calculator
Calculates and compares vega for NVDA, TSLA, MSTR, and COIN options.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

class MultiStockVegaCalculator:
    def __init__(self):
        self.symbols = ['NVDA', 'TSLA', 'MSTR', 'COIN']
        self.risk_free_rate = 0.045  # 4.5%
        self.dividend_yields = {
            'NVDA': 0.0025,  # 0.25%
            'TSLA': 0.0,     # 0%
            'MSTR': 0.0,     # 0%
            'COIN': 0.0      # 0%
        }
        self.stock_data = {}
        
    def fetch_market_data(self):
        """
        Fetch current market data for all stocks
        """
        print("Fetching market data for all stocks...")
        
        if HAS_YFINANCE:
            for symbol in self.symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1mo")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        
                        # Calculate realized volatility
                        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                        volatility = returns.std() * np.sqrt(252)
                        
                        self.stock_data[symbol] = {
                            'price': current_price,
                            'volatility': volatility,
                            'dividend_yield': self.dividend_yields[symbol]
                        }
                        print(f"{symbol}: ${current_price:.2f}, Vol: {volatility*100:.1f}%")
                    else:
                        self.use_fallback_data(symbol)
                        
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
                    self.use_fallback_data(symbol)
        else:
            for symbol in self.symbols:
                self.use_fallback_data(symbol)
        
        return True
    
    def use_fallback_data(self, symbol):
        """
        Use fallback data for a specific symbol
        """
        fallback_data = {
            'NVDA': {'price': 180.45, 'volatility': 0.35},
            'TSLA': {'price': 220.50, 'volatility': 0.45},
            'MSTR': {'price': 1650.00, 'volatility': 0.65},
            'COIN': {'price': 185.00, 'volatility': 0.55}
        }
        
        if symbol in fallback_data:
            self.stock_data[symbol] = {
                'price': fallback_data[symbol]['price'],
                'volatility': fallback_data[symbol]['volatility'],
                'dividend_yield': self.dividend_yields[symbol]
            }
            print(f"{symbol}: ${fallback_data[symbol]['price']:.2f}, Vol: {fallback_data[symbol]['volatility']*100:.1f}% (fallback)")
    
    def black_scholes_vega(self, S, K, T, r, sigma, q=0):
        """
        Calculate vega using Black-Scholes formula with dividend adjustment
        """
        if T <= 0:
            return 0
            
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        
        # Vega = S * sqrt(T) * N'(d1) * exp(-q*T)
        vega = S * np.sqrt(T) * norm.pdf(d1) * np.exp(-q*T)
        
        return vega / 100  # Convert to percentage points
    
    def calculate_vega_matrix(self):
        """
        Calculate vega matrix for all stocks
        """
        print("Calculating vega matrix for all stocks...")
        
        # Define relative strike prices (percentage of current price)
        strike_percentages = np.arange(0.85, 1.16, 0.05)  # 85% to 115% in 5% increments
        
        # Define expiration dates
        expirations = [7, 14, 21, 30, 45, 60, 90]  # Days to expiration
        
        results = []
        
        for symbol in self.symbols:
            stock_info = self.stock_data[symbol]
            S = stock_info['price']
            sigma = stock_info['volatility']
            q = stock_info['dividend_yield']
            
            for days in expirations:
                T = days / 365.0
                
                for strike_pct in strike_percentages:
                    K = S * strike_pct
                    moneyness = K / S
                    
                    # Calculate vega
                    vega = self.black_scholes_vega(S, K, T, self.risk_free_rate, sigma, q)
                    
                    # Calculate other metrics for context
                    d1 = (np.log(S/K) + (self.risk_free_rate - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    d2 = d1 - sigma*np.sqrt(T)
                    
                    # Call price for reference
                    call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-self.risk_free_rate*T)*norm.cdf(d2)
                    
                    # Delta for reference
                    delta = np.exp(-q*T) * norm.cdf(d1)
                    
                    results.append({
                        'symbol': symbol,
                        'stock_price': S,
                        'strike_price': K,
                        'strike_percentage': strike_pct,
                        'moneyness': moneyness,
                        'days_to_expiry': days,
                        'time_to_expiry': T,
                        'volatility': sigma,
                        'vega': vega,
                        'call_price': call_price,
                        'delta': delta,
                        'vega_per_dollar': vega / S,  # Normalized by stock price
                        'vega_per_vol': vega / sigma  # Normalized by volatility
                    })
        
        return pd.DataFrame(results)
    
    def analyze_vega_rankings(self, vega_df):
        """
        Analyze and rank stocks by vega characteristics
        """
        print("Analyzing vega rankings...")
        
        # ATM vega analysis (moneyness close to 1.0)
        atm_mask = abs(vega_df['moneyness'] - 1.0) < 0.025
        atm_vega = vega_df[atm_mask]
        
        # Group by symbol and expiration
        rankings = {}
        
        for days in [7, 30, 90]:
            day_data = atm_vega[atm_vega['days_to_expiry'] == days]
            
            if not day_data.empty:
                ranking = day_data.groupby('symbol').agg({
                    'vega': 'mean',
                    'vega_per_dollar': 'mean',
                    'vega_per_vol': 'mean',
                    'volatility': 'mean',
                    'stock_price': 'mean'
                }).round(4)
                
                ranking['vega_rank'] = ranking['vega'].rank(ascending=False)
                ranking['vega_per_dollar_rank'] = ranking['vega_per_dollar'].rank(ascending=False)
                
                rankings[f'{days}_days'] = ranking.sort_values('vega', ascending=False)
        
        return rankings
    
    def plot_vega_analysis(self, vega_df, rankings):
        """
        Create comprehensive vega visualization
        """
        print("Creating vega visualizations...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # Color scheme for stocks
        colors = {'NVDA': 'green', 'TSLA': 'red', 'MSTR': 'orange', 'COIN': 'blue'}
        
        # Plot 1: ATM Vega by Expiration for all stocks
        plt.subplot(3, 3, 1)
        atm_data = vega_df[abs(vega_df['moneyness'] - 1.0) < 0.025]
        
        for symbol in self.symbols:
            symbol_data = atm_data[atm_data['symbol'] == symbol]
            if not symbol_data.empty:
                plt.plot(symbol_data['days_to_expiry'], symbol_data['vega'], 
                        'o-', label=symbol, color=colors[symbol], linewidth=2, markersize=6)
        
        plt.xlabel('Days to Expiration')
        plt.ylabel('ATM Vega')
        plt.title('ATM Vega by Expiration Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Vega vs Moneyness (30-day options)
        plt.subplot(3, 3, 2)
        data_30d = vega_df[vega_df['days_to_expiry'] == 30]
        
        for symbol in self.symbols:
            symbol_data = data_30d[data_30d['symbol'] == symbol]
            if not symbol_data.empty:
                plt.plot(symbol_data['moneyness'], symbol_data['vega'], 
                        'o-', label=symbol, color=colors[symbol], linewidth=2, alpha=0.8)
        
        plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='ATM')
        plt.xlabel('Moneyness (Strike/Spot)')
        plt.ylabel('Vega')
        plt.title('Vega vs Moneyness (30-Day Options)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Vega per Dollar Invested
        plt.subplot(3, 3, 3)
        atm_30d = vega_df[(abs(vega_df['moneyness'] - 1.0) < 0.025) & (vega_df['days_to_expiry'] == 30)]
        
        symbols = atm_30d['symbol'].values
        vega_per_dollar = atm_30d['vega_per_dollar'].values
        
        bars = plt.bar(symbols, vega_per_dollar, color=[colors[s] for s in symbols], alpha=0.7)
        plt.ylabel('Vega per Dollar of Stock Price')
        plt.title('ATM Vega Efficiency (30-Day)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, vega_per_dollar):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001, 
                    f'{val:.5f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Vega Heat Map for NVDA
        plt.subplot(3, 3, 4)
        nvda_data = vega_df[vega_df['symbol'] == 'NVDA']
        try:
            pivot_nvda = nvda_data.pivot_table(index='days_to_expiry', columns='strike_percentage', 
                                             values='vega', aggfunc='mean')
            im = plt.imshow(pivot_nvda.values, aspect='auto', cmap='viridis')
            plt.colorbar(im, label='Vega')
            plt.xlabel('Strike % of Spot')
            plt.ylabel('Days to Expiry')
            plt.title('NVDA Vega Heat Map')
            
            # Set tick labels
            strike_labels = [f'{int(p*100)}%' for p in pivot_nvda.columns[::2]]
            plt.xticks(range(0, len(pivot_nvda.columns), 2), strike_labels, rotation=45)
            plt.yticks(range(len(pivot_nvda.index)), pivot_nvda.index)
        except:
            plt.text(0.5, 0.5, 'Error creating heat map', transform=plt.gca().transAxes, 
                    ha='center', va='center')
        
        # Plot 5: Vega Heat Map for TSLA
        plt.subplot(3, 3, 5)
        tsla_data = vega_df[vega_df['symbol'] == 'TSLA']
        try:
            pivot_tsla = tsla_data.pivot_table(index='days_to_expiry', columns='strike_percentage', 
                                             values='vega', aggfunc='mean')
            im = plt.imshow(pivot_tsla.values, aspect='auto', cmap='plasma')
            plt.colorbar(im, label='Vega')
            plt.xlabel('Strike % of Spot')
            plt.ylabel('Days to Expiry')
            plt.title('TSLA Vega Heat Map')
            
            strike_labels = [f'{int(p*100)}%' for p in pivot_tsla.columns[::2]]
            plt.xticks(range(0, len(pivot_tsla.columns), 2), strike_labels, rotation=45)
            plt.yticks(range(len(pivot_tsla.index)), pivot_tsla.index)
        except:
            plt.text(0.5, 0.5, 'Error creating heat map', transform=plt.gca().transAxes, 
                    ha='center', va='center')
        
        # Plot 6: Maximum Vega by Stock
        plt.subplot(3, 3, 6)
        max_vega_by_stock = vega_df.groupby('symbol')['vega'].max().sort_values(ascending=False)
        
        bars = plt.bar(max_vega_by_stock.index, max_vega_by_stock.values, 
                      color=[colors[s] for s in max_vega_by_stock.index], alpha=0.7)
        plt.ylabel('Maximum Vega')
        plt.title('Maximum Vega by Stock')
        plt.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, max_vega_by_stock.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 7: Vega vs Volatility Relationship
        plt.subplot(3, 3, 7)
        atm_30d_all = vega_df[(abs(vega_df['moneyness'] - 1.0) < 0.025) & 
                             (vega_df['days_to_expiry'] == 30)]
        
        for symbol in self.symbols:
            symbol_data = atm_30d_all[atm_30d_all['symbol'] == symbol]
            if not symbol_data.empty:
                plt.scatter(symbol_data['volatility'] * 100, symbol_data['vega'], 
                           color=colors[symbol], s=100, alpha=0.8, label=symbol)
        
        plt.xlabel('Implied Volatility (%)')
        plt.ylabel('ATM Vega (30-day)')
        plt.title('Vega vs Volatility Relationship')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 8: Vega Rankings Bar Chart
        plt.subplot(3, 3, 8)
        if '30_days' in rankings:
            ranking_30d = rankings['30_days']
            
            x_pos = np.arange(len(ranking_30d))
            bars = plt.bar(x_pos, ranking_30d['vega'], 
                          color=[colors[s] for s in ranking_30d.index], alpha=0.7)
            
            plt.xlabel('Stock')
            plt.ylabel('ATM Vega (30-day)')
            plt.title('30-Day ATM Vega Rankings')
            plt.xticks(x_pos, ranking_30d.index)
            plt.grid(True, alpha=0.3)
            
            for i, (bar, val) in enumerate(zip(bars, ranking_30d['vega'])):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 9: Normalized Vega Comparison
        plt.subplot(3, 3, 9)
        if '30_days' in rankings:
            ranking_30d = rankings['30_days']
            
            x_pos = np.arange(len(ranking_30d))
            width = 0.35
            
            bars1 = plt.bar(x_pos - width/2, ranking_30d['vega'], width, 
                           label='Absolute Vega', alpha=0.8)
            bars2 = plt.bar(x_pos + width/2, ranking_30d['vega_per_dollar'] * 1000, width, 
                           label='Vega per $1000', alpha=0.8)
            
            plt.xlabel('Stock')
            plt.ylabel('Vega')
            plt.title('Absolute vs Normalized Vega (30-day ATM)')
            plt.xticks(x_pos, ranking_30d.index)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/kafka/projects/NVII/multi_stock_vega_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_vega_report(self, vega_df, rankings):
        """
        Generate comprehensive vega analysis report
        """
        print("\n" + "="*80)
        print("MULTI-STOCK VEGA ANALYSIS REPORT")
        print("="*80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Stocks Analyzed: {', '.join(self.symbols)}")
        print(f"Risk-Free Rate: {self.risk_free_rate*100:.1f}%")
        
        print("\nCURRENT MARKET DATA:")
        print("-" * 50)
        for symbol in self.symbols:
            data = self.stock_data[symbol]
            print(f"{symbol:4s}: ${data['price']:8.2f} | Vol: {data['volatility']*100:5.1f}% | "
                  f"Div Yield: {data['dividend_yield']*100:4.2f}%")
        
        print("\nVEGA RANKINGS BY EXPIRATION:")
        print("-" * 50)
        
        for period, ranking in rankings.items():
            days = period.replace('_days', '')
            print(f"\n{days}-Day ATM Options:")
            print(f"{'Rank':<4} {'Symbol':<6} {'Vega':<8} {'Per $':<8} {'Vol%':<6} {'Price':<8}")
            print("-" * 45)
            
            for i, (symbol, row) in enumerate(ranking.iterrows(), 1):
                print(f"{i:<4} {symbol:<6} {row['vega']:<8.4f} {row['vega_per_dollar']:<8.6f} "
                      f"{row['volatility']*100:<6.1f} ${row['stock_price']:<8.2f}")
        
        print("\nVEGA EFFICIENCY ANALYSIS:")
        print("-" * 50)
        
        # Calculate vega efficiency metrics
        atm_30d = vega_df[(abs(vega_df['moneyness'] - 1.0) < 0.025) & 
                         (vega_df['days_to_expiry'] == 30)]
        
        for symbol in self.symbols:
            symbol_data = atm_30d[atm_30d['symbol'] == symbol]
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                vega_efficiency = row['vega'] / (row['stock_price'] * row['volatility'])
                
                print(f"\n{symbol} Vega Efficiency:")
                print(f"  Absolute Vega: {row['vega']:.4f}")
                print(f"  Vega per Dollar: {row['vega_per_dollar']:.6f}")
                print(f"  Vega per Vol Point: {row['vega_per_vol']:.4f}")
                print(f"  Efficiency Ratio: {vega_efficiency:.8f}")
        
        print("\nMAXIMUM VEGA ANALYSIS:")
        print("-" * 50)
        
        for symbol in self.symbols:
            symbol_data = vega_df[vega_df['symbol'] == symbol]
            max_vega_row = symbol_data.loc[symbol_data['vega'].idxmax()]
            
            print(f"\n{symbol} Maximum Vega:")
            print(f"  Vega: {max_vega_row['vega']:.4f}")
            print(f"  Strike: ${max_vega_row['strike_price']:.2f} "
                  f"({max_vega_row['strike_percentage']*100:.0f}% of spot)")
            print(f"  Days to Expiry: {int(max_vega_row['days_to_expiry'])}")
            print(f"  Moneyness: {max_vega_row['moneyness']:.3f}")
        
        print("\nVOLATILITY IMPACT ANALYSIS:")
        print("-" * 50)
        
        print("Vega sensitivity to 1% volatility change (ATM, 30-day):")
        for symbol in self.symbols:
            symbol_data = atm_30d[atm_30d['symbol'] == symbol]
            if not symbol_data.empty:
                vega = symbol_data.iloc[0]['vega']
                option_price = symbol_data.iloc[0]['call_price']
                pct_change = (vega / option_price) * 100 if option_price > 0 else 0
                
                print(f"  {symbol}: ${vega:.4f} ({pct_change:.1f}% of option price)")
        
        print("\nTRADING IMPLICATIONS:")
        print("-" * 50)
        
        highest_vega = rankings['30_days'].index[0]
        lowest_vega = rankings['30_days'].index[-1]
        
        print(f"• Highest Vega Exposure: {highest_vega}")
        print(f"  - Best for volatility trading strategies")
        print(f"  - Higher premium collection in covered calls")
        print(f"  - Greater sensitivity to IV changes")
        
        print(f"\n• Lowest Vega Exposure: {lowest_vega}")
        print(f"  - More stable option prices")
        print(f"  - Lower volatility risk")
        print(f"  - Potentially more predictable strategies")
        
        print(f"\n• Price-Normalized Leader: "
              f"{rankings['30_days']['vega_per_dollar'].idxmax()}")
        print(f"  - Best vega exposure per dollar invested")
        print(f"  - Efficient for smaller accounts")
        
        print("\nRISK CONSIDERATIONS:")
        print("-" * 50)
        print("• High vega stocks (MSTR, COIN) offer:")
        print("  - Higher potential profits from volatility expansion")
        print("  - Greater losses during volatility contraction")
        print("  - More dramatic option price swings")
        
        print("\n• Lower vega stocks (NVDA, TSLA) offer:")
        print("  - More predictable option behavior")
        print("  - Lower volatility risk")
        print("  - Steadier income from option strategies")

def main():
    calculator = MultiStockVegaCalculator()
    
    print("Multi-Stock Vega Analysis")
    print("Analyzing: NVDA, TSLA, MSTR, COIN")
    
    # Fetch market data
    calculator.fetch_market_data()
    
    # Calculate vega matrix
    vega_df = calculator.calculate_vega_matrix()
    
    # Analyze rankings
    rankings = calculator.analyze_vega_rankings(vega_df)
    
    # Create visualizations
    calculator.plot_vega_analysis(vega_df, rankings)
    
    # Generate report
    calculator.generate_vega_report(vega_df, rankings)
    
    print(f"\nAnalysis complete!")
    print(f"Charts saved as 'multi_stock_vega_analysis.png'")
    print(f"Total options analyzed: {len(vega_df)}")

if __name__ == "__main__":
    main()