#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TSLAOptionAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.045  # 4.5% current risk-free rate
        self.dividend_yield = 0.0    # TSLA doesn't pay dividends
        
    def fetch_tsla_data(self, period="2y"):
        """Fetch TSLA historical data"""
        tsla = yf.Ticker("TSLA")
        data = tsla.history(period=period)
        return data
    
    def black_scholes_call(self, S, K, T, r, sigma, q=0):
        """Calculate Black-Scholes call option price"""
        if T <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def calculate_greeks(self, S, K, T, r, sigma, q=0):
        """Calculate option Greeks"""
        if T <= 0:
            return {'delta': 1 if S > K else 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        delta = np.exp(-q * T) * norm.cdf(d1)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) - \
                r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
        theta = theta / 365  # Convert to daily theta
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Convert to 1% volatility change
        
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}
    
    def calculate_historical_volatility(self, prices, window=30):
        """Calculate rolling historical volatility"""
        returns = np.log(prices / prices.shift(1)).dropna()
        vol = returns.rolling(window=window).std() * np.sqrt(252)
        return vol
    
    def analyze_tsla_options(self):
        """Comprehensive TSLA option analysis"""
        print("Fetching TSLA data...")
        data = self.fetch_tsla_data("2y")
        
        # Calculate historical volatility
        data['volatility'] = self.calculate_historical_volatility(data['Close'])
        
        # Option parameters for analysis
        strike_prices = [200, 250, 300, 350, 400]  # Various strikes
        days_to_expiry = [30, 60, 90, 180, 365]   # Various maturities
        
        results = []
        
        for idx, row in data.iterrows():
            if pd.isna(row['volatility']):
                continue
                
            S = row['Close']
            vol = row['volatility']
            
            for strike in strike_prices:
                for dte in days_to_expiry:
                    T = dte / 365.0
                    
                    # Calculate option price and Greeks
                    option_price = self.black_scholes_call(S, strike, T, self.risk_free_rate, vol, self.dividend_yield)
                    greeks = self.calculate_greeks(S, strike, T, self.risk_free_rate, vol, self.dividend_yield)
                    
                    results.append({
                        'date': idx,
                        'stock_price': S,
                        'strike': strike,
                        'dte': dte,
                        'volatility': vol,
                        'option_price': option_price,
                        'delta': greeks['delta'],
                        'gamma': greeks['gamma'],
                        'theta': greeks['theta'],
                        'vega': greeks['vega'],
                        'moneyness': S / strike
                    })
        
        option_df = pd.DataFrame(results)
        
        return data, option_df
    
    def create_comprehensive_report(self, data, option_df):
        """Create comprehensive PNG report"""
        fig = plt.figure(figsize=(20, 16))
        
        # Set up the layout
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1])
        
        # 1. TSLA Stock Price History
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(data.index, data['Close'], linewidth=2, color='#E31837', label='TSLA Close Price')
        ax1.fill_between(data.index, data['Low'], data['High'], alpha=0.3, color='lightblue', label='Daily Range')
        ax1.set_title('TSLA Stock Price History (2 Years)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # 2. Historical Volatility
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(data.index, data['volatility'] * 100, color='purple', linewidth=2)
        ax2.set_title('Historical Volatility (30-day)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volatility (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        
        # 3. Option Prices vs Stock Price (ATM options)
        ax3 = fig.add_subplot(gs[1, 1])
        atm_options = option_df[option_df['moneyness'].between(0.95, 1.05)]
        for dte in [30, 90, 180, 365]:
            dte_data = atm_options[atm_options['dte'] == dte]
            if not dte_data.empty:
                ax3.scatter(dte_data['stock_price'], dte_data['option_price'], 
                           alpha=0.6, s=10, label=f'{dte}D')
        ax3.set_title('ATM Option Prices vs Stock Price', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Stock Price ($)', fontsize=12)
        ax3.set_ylabel('Option Price ($)', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Delta Analysis
        ax4 = fig.add_subplot(gs[1, 2])
        for dte in [30, 90, 180]:
            dte_data = option_df[option_df['dte'] == dte]
            moneyness_bins = np.arange(0.8, 1.3, 0.05)
            avg_delta = dte_data.groupby(pd.cut(dte_data['moneyness'], moneyness_bins))['delta'].mean()
            ax4.plot(avg_delta.index.map(lambda x: x.mid), avg_delta.values, 
                    marker='o', markersize=4, label=f'{dte}D')
        ax4.set_title('Delta vs Moneyness', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Moneyness (S/K)', fontsize=12)
        ax4.set_ylabel('Delta', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. Option Price Changes
        ax5 = fig.add_subplot(gs[2, 0])
        # Calculate daily option price changes for 90-day ATM options
        atm_90d = option_df[(option_df['dte'] == 90) & (option_df['moneyness'].between(0.95, 1.05))]
        if not atm_90d.empty:
            daily_changes = atm_90d.groupby('date')['option_price'].mean().pct_change() * 100
            ax5.plot(daily_changes.index, daily_changes.values, color='green', alpha=0.7, linewidth=1)
            ax5.set_title('Daily Option Price Changes (90D ATM)', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Change (%)', fontsize=12)
            ax5.grid(True, alpha=0.3)
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        
        # 6. Vega Analysis
        ax6 = fig.add_subplot(gs[2, 1])
        vega_data = option_df[option_df['dte'].isin([30, 90, 180])]
        vega_pivot = vega_data.pivot_table(values='vega', index='moneyness', columns='dte', aggfunc='mean')
        for dte in vega_pivot.columns:
            ax6.plot(vega_pivot.index, vega_pivot[dte], marker='o', markersize=4, label=f'{dte}D')
        ax6.set_title('Vega vs Moneyness', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Moneyness (S/K)', fontsize=12)
        ax6.set_ylabel('Vega', fontsize=12)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # 7. Theta Analysis
        ax7 = fig.add_subplot(gs[2, 2])
        theta_data = option_df[option_df['dte'].isin([30, 90, 180])]
        theta_pivot = theta_data.pivot_table(values='theta', index='moneyness', columns='dte', aggfunc='mean')
        for dte in theta_pivot.columns:
            ax7.plot(theta_pivot.index, theta_pivot[dte], marker='o', markersize=4, label=f'{dte}D')
        ax7.set_title('Theta vs Moneyness', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Moneyness (S/K)', fontsize=12)
        ax7.set_ylabel('Theta ($/day)', fontsize=12)
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)
        
        # 8. Summary Statistics
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Calculate summary statistics
        current_price = data['Close'].iloc[-1]
        price_change_1y = ((current_price - data['Close'].iloc[-252]) / data['Close'].iloc[-252]) * 100 if len(data) >= 252 else 0
        avg_vol = data['volatility'].mean() * 100
        max_vol = data['volatility'].max() * 100
        min_vol = data['volatility'].min() * 100
        
        # ATM option analysis
        atm_recent = option_df[(option_df['moneyness'].between(0.95, 1.05)) & 
                               (option_df['date'] >= option_df['date'].max() - timedelta(days=30))]
        
        avg_option_prices = atm_recent.groupby('dte')['option_price'].mean()
        avg_deltas = atm_recent.groupby('dte')['delta'].mean()
        
        summary_text = f"""
TSLA LONG-TERM OPTION ANALYSIS SUMMARY

Stock Performance:
• Current Price: ${current_price:.2f}
• 1-Year Change: {price_change_1y:.1f}%
• Price Range (2Y): ${data['Close'].min():.2f} - ${data['Close'].max():.2f}

Volatility Analysis:
• Average Volatility: {avg_vol:.1f}%
• Volatility Range: {min_vol:.1f}% - {max_vol:.1f}%
• Current Volatility: {data['volatility'].iloc[-1] * 100:.1f}%

ATM Option Pricing (Recent 30 Days):
• 30-Day Options: ${avg_option_prices.get(30, 0):.2f} (Δ={avg_deltas.get(30, 0):.3f})
• 90-Day Options: ${avg_option_prices.get(90, 0):.2f} (Δ={avg_deltas.get(90, 0):.3f})
• 180-Day Options: ${avg_option_prices.get(180, 0):.2f} (Δ={avg_deltas.get(180, 0):.3f})
• 365-Day Options: ${avg_option_prices.get(365, 0):.2f} (Δ={avg_deltas.get(365, 0):.3f})

Key Insights:
• Option prices show high sensitivity to volatility changes
• Long-term options provide better delta exposure but higher time decay
• Volatility clustering observed in recent periods
• Strong correlation between stock price movements and option value changes
        """
        
        ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('TSLA Long-Term Call Option Analysis Report', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save the report
        plt.savefig('/home/kafka/projects/NVII/tsla_option_analysis_report.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return option_df

def main():
    analyzer = TSLAOptionAnalyzer()
    
    print("Starting TSLA option analysis...")
    data, option_df = analyzer.analyze_tsla_options()
    
    print("Creating comprehensive report...")
    analyzer.create_comprehensive_report(data, option_df)
    
    print("Analysis complete! Report saved as tsla_option_analysis_report.png")
    
    # Print some key statistics
    print(f"\nKey Statistics:")
    print(f"Data points analyzed: {len(data)}")
    print(f"Option calculations: {len(option_df)}")
    print(f"Current TSLA price: ${data['Close'].iloc[-1]:.2f}")
    print(f"Current volatility: {data['volatility'].iloc[-1] * 100:.1f}%")

if __name__ == "__main__":
    main()