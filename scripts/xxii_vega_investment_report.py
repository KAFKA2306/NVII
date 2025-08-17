#!/usr/bin/env python3
"""
XXII (22nd Century Group) Vega-Based Investment Analysis Report
Comprehensive analysis for investment decision based on option vega characteristics.
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

class XXIIVegaInvestmentAnalyzer:
    def __init__(self):
        self.symbol = "XXII"
        self.comparison_symbols = ["MSTR", "COIN", "NVDA", "TSLA", "GME", "AMC"]
        self.risk_free_rate = 0.045  # 4.5%
        self.dividend_yield = 0.0  # XXII typically doesn't pay dividends
        self.market_data = {}
        
    def fetch_market_data(self):
        """
        Fetch comprehensive market data for XXII and comparison stocks
        """
        print("Fetching market data for XXII and comparison stocks...")
        
        all_symbols = [self.symbol] + self.comparison_symbols
        
        if HAS_YFINANCE:
            for symbol in all_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="3mo")
                    info = ticker.info
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        
                        # Calculate multiple volatility measures
                        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                        vol_30d = returns.tail(21).std() * np.sqrt(252) if len(returns) >= 21 else returns.std() * np.sqrt(252)
                        vol_60d = returns.tail(42).std() * np.sqrt(252) if len(returns) >= 42 else returns.std() * np.sqrt(252)
                        vol_90d = returns.std() * np.sqrt(252)
                        
                        # Calculate price momentum and trends
                        price_change_30d = (current_price - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22] if len(hist) >= 22 else 0
                        
                        # Volume analysis
                        avg_volume = hist['Volume'].tail(30).mean()
                        
                        # Market cap estimation
                        market_cap = info.get('marketCap', current_price * 100_000_000)  # Fallback estimate
                        
                        self.market_data[symbol] = {
                            'price': current_price,
                            'volatility_30d': vol_30d,
                            'volatility_60d': vol_60d,
                            'volatility_90d': vol_90d,
                            'price_change_30d': price_change_30d,
                            'avg_volume': avg_volume,
                            'market_cap': market_cap,
                            'dividend_yield': 0.0,  # Assume no dividends for most
                            'sector': info.get('sector', 'Unknown'),
                            'beta': info.get('beta', 1.5)  # Default high beta for volatile stocks
                        }
                        
                        print(f"{symbol}: ${current_price:.2f}, Vol: {vol_30d*100:.1f}%, "
                              f"30d Change: {price_change_30d*100:+.1f}%")
                    
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
                    self.use_fallback_data(symbol)
        else:
            for symbol in all_symbols:
                self.use_fallback_data(symbol)
        
        return self.market_data
    
    def use_fallback_data(self, symbol):
        """
        Use fallback data for stocks that couldn't be fetched
        """
        fallback_data = {
            'XXII': {'price': 1.25, 'vol': 0.85, 'change': -0.15, 'cap': 150_000_000},
            'MSTR': {'price': 366.32, 'vol': 0.53, 'change': 0.05, 'cap': 8_000_000_000},
            'COIN': {'price': 317.55, 'vol': 0.71, 'change': -0.08, 'cap': 25_000_000_000},
            'NVDA': {'price': 180.45, 'vol': 0.24, 'change': 0.12, 'cap': 4_500_000_000_000},
            'TSLA': {'price': 330.56, 'vol': 0.44, 'change': 0.03, 'cap': 1_000_000_000_000},
            'GME': {'price': 45.80, 'vol': 0.95, 'change': -0.20, 'cap': 3_500_000_000},
            'AMC': {'price': 8.75, 'vol': 1.10, 'change': -0.25, 'cap': 2_000_000_000}
        }
        
        if symbol in fallback_data:
            data = fallback_data[symbol]
            self.market_data[symbol] = {
                'price': data['price'],
                'volatility_30d': data['vol'],
                'volatility_60d': data['vol'] * 0.95,
                'volatility_90d': data['vol'] * 0.90,
                'price_change_30d': data['change'],
                'avg_volume': 1_000_000,
                'market_cap': data['cap'],
                'dividend_yield': 0.0,
                'sector': 'Biotechnology' if symbol == 'XXII' else 'Technology',
                'beta': 2.0 if symbol in ['XXII', 'GME', 'AMC'] else 1.5
            }
            print(f"{symbol}: ${data['price']:.2f}, Vol: {data['vol']*100:.1f}% (fallback)")
    
    def calculate_comprehensive_vega_analysis(self):
        """
        Calculate comprehensive vega analysis for all stocks
        """
        print("Calculating comprehensive vega analysis...")
        
        results = []
        
        # Define relative strikes and expirations
        strike_percentages = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
        expirations = [7, 14, 21, 30, 45, 60, 90]
        
        for symbol, data in self.market_data.items():
            S = data['price']
            sigma = data['volatility_30d']
            q = data['dividend_yield']
            
            for days in expirations:
                T = days / 365.0
                
                for strike_pct in strike_percentages:
                    K = S * strike_pct
                    
                    # Calculate Black-Scholes metrics
                    d1 = (np.log(S/K) + (self.risk_free_rate - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    d2 = d1 - sigma*np.sqrt(T)
                    
                    # Vega calculation
                    vega = S * np.sqrt(T) * norm.pdf(d1) * np.exp(-q*T) / 100
                    
                    # Call price
                    call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-self.risk_free_rate*T)*norm.cdf(d2)
                    
                    # Other Greeks
                    delta = np.exp(-q*T) * norm.cdf(d1)
                    gamma = np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
                    theta = (-S*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) 
                            - self.risk_free_rate*K*np.exp(-self.risk_free_rate*T)*norm.cdf(d2) 
                            + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
                    
                    # Investment metrics
                    vega_per_dollar = vega / S
                    vega_yield = vega / call_price if call_price > 0 else 0
                    
                    results.append({
                        'symbol': symbol,
                        'stock_price': S,
                        'strike_price': K,
                        'strike_pct': strike_pct,
                        'moneyness': K / S,
                        'days_to_expiry': days,
                        'time_to_expiry': T,
                        'volatility': sigma,
                        'vega': vega,
                        'call_price': call_price,
                        'delta': delta,
                        'gamma': gamma,
                        'theta': theta,
                        'vega_per_dollar': vega_per_dollar,
                        'vega_yield': vega_yield,
                        'market_cap': data['market_cap'],
                        'beta': data['beta'],
                        'sector': data['sector']
                    })
        
        return pd.DataFrame(results)
    
    def generate_investment_analysis(self, vega_df):
        """
        Generate comprehensive investment analysis based on vega
        """
        print("Generating investment analysis...")
        
        # Focus on XXII analysis
        xxii_data = vega_df[vega_df['symbol'] == 'XXII']
        comparison_data = vega_df[vega_df['symbol'].isin(self.comparison_symbols)]
        
        # ATM options analysis (30-day)
        atm_30d = vega_df[(abs(vega_df['moneyness'] - 1.0) < 0.025) & (vega_df['days_to_expiry'] == 30)]
        
        # Investment scores calculation
        investment_scores = {}
        
        for symbol in [self.symbol] + self.comparison_symbols:
            symbol_atm = atm_30d[atm_30d['symbol'] == symbol]
            if not symbol_atm.empty:
                row = symbol_atm.iloc[0]
                
                # Calculate investment score based on multiple factors
                vega_score = row['vega'] / atm_30d['vega'].max() * 100  # Normalize to 100
                efficiency_score = row['vega_per_dollar'] / atm_30d['vega_per_dollar'].max() * 100
                volatility_score = row['volatility'] / atm_30d['volatility'].max() * 100
                
                # Market cap penalty for very large caps (less upside potential)
                cap_score = 100 - (np.log10(row['market_cap']) - 6) * 10  # Penalty starts at 1M market cap
                cap_score = max(0, min(100, cap_score))
                
                # Beta score (higher beta = higher score for volatility trading)
                beta_score = min(100, row['beta'] * 25)  # Beta of 4 = 100 points
                
                # Composite score
                composite_score = (vega_score * 0.30 + 
                                 efficiency_score * 0.25 + 
                                 volatility_score * 0.20 + 
                                 cap_score * 0.15 + 
                                 beta_score * 0.10)
                
                investment_scores[symbol] = {
                    'vega_score': vega_score,
                    'efficiency_score': efficiency_score,
                    'volatility_score': volatility_score,
                    'cap_score': cap_score,
                    'beta_score': beta_score,
                    'composite_score': composite_score,
                    'vega': row['vega'],
                    'price': row['stock_price'],
                    'market_cap': row['market_cap']
                }
        
        return investment_scores, atm_30d
    
    def create_investment_report_visualization(self, vega_df, investment_scores, atm_30d):
        """
        Create comprehensive visualization for investment report
        """
        print("Creating investment report visualization...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # Color scheme
        colors = {
            'XXII': 'red', 'MSTR': 'orange', 'COIN': 'blue', 
            'NVDA': 'green', 'TSLA': 'purple', 'GME': 'brown', 'AMC': 'pink'
        }
        
        # Plot 1: Investment Score Ranking
        plt.subplot(3, 3, 1)
        scores_df = pd.DataFrame(investment_scores).T.sort_values('composite_score', ascending=True)
        
        bars = plt.barh(range(len(scores_df)), scores_df['composite_score'], 
                       color=[colors.get(symbol, 'gray') for symbol in scores_df.index])
        plt.yticks(range(len(scores_df)), scores_df.index)
        plt.xlabel('Investment Score')
        plt.title('Vega-Based Investment Ranking')
        plt.grid(True, alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores_df['composite_score'])):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{score:.1f}', va='center', fontsize=9)
        
        # Plot 2: XXII Vega Profile by Expiration
        plt.subplot(3, 3, 2)
        xxii_atm = vega_df[(vega_df['symbol'] == 'XXII') & 
                          (abs(vega_df['moneyness'] - 1.0) < 0.025)]
        
        plt.plot(xxii_atm['days_to_expiry'], xxii_atm['vega'], 
                'ro-', linewidth=3, markersize=8, label='XXII Vega')
        plt.xlabel('Days to Expiration')
        plt.ylabel('Vega')
        plt.title('XXII ATM Vega Profile')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Vega vs Market Cap Scatter
        plt.subplot(3, 3, 3)
        for symbol in atm_30d['symbol'].unique():
            symbol_data = atm_30d[atm_30d['symbol'] == symbol]
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                size = 200 if symbol == 'XXII' else 100
                alpha = 1.0 if symbol == 'XXII' else 0.7
                plt.scatter(row['market_cap'], row['vega'], 
                           color=colors.get(symbol, 'gray'), s=size, alpha=alpha, label=symbol)
        
        plt.xscale('log')
        plt.xlabel('Market Cap ($)')
        plt.ylabel('ATM Vega (30-day)')
        plt.title('Vega vs Market Capitalization')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: XXII vs Competitors - Multiple Metrics
        plt.subplot(3, 3, 4)
        metrics = ['vega_score', 'efficiency_score', 'volatility_score', 'cap_score']
        x = np.arange(len(metrics))
        
        if 'XXII' in investment_scores:
            xxii_scores = [investment_scores['XXII'][metric] for metric in metrics]
            avg_scores = [np.mean([investment_scores[s][metric] for s in investment_scores.keys() 
                                 if s != 'XXII']) for metric in metrics]
            
            width = 0.35
            plt.bar(x - width/2, xxii_scores, width, label='XXII', color='red', alpha=0.8)
            plt.bar(x + width/2, avg_scores, width, label='Competitors Avg', color='gray', alpha=0.6)
            
            plt.xlabel('Metrics')
            plt.ylabel('Score (0-100)')
            plt.title('XXII vs Competitors Comparison')
            plt.xticks(x, ['Vega', 'Efficiency', 'Volatility', 'Market Cap'])
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Vega Efficiency Analysis
        plt.subplot(3, 3, 5)
        efficiency_data = atm_30d.copy()
        
        plt.scatter(efficiency_data['vega_per_dollar'] * 1000, efficiency_data['vega'], 
                   c=[colors.get(symbol, 'gray') for symbol in efficiency_data['symbol']], 
                   s=100, alpha=0.8)
        
        # Highlight XXII
        xxii_row = efficiency_data[efficiency_data['symbol'] == 'XXII']
        if not xxii_row.empty:
            plt.scatter(xxii_row['vega_per_dollar'].iloc[0] * 1000, xxii_row['vega'].iloc[0], 
                       color='red', s=300, marker='*', label='XXII', edgecolor='black', linewidth=2)
        
        plt.xlabel('Vega per $1000 Invested')
        plt.ylabel('Absolute Vega')
        plt.title('Vega Efficiency Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Volatility vs Vega Relationship
        plt.subplot(3, 3, 6)
        for symbol in atm_30d['symbol'].unique():
            symbol_data = atm_30d[atm_30d['symbol'] == symbol]
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                size = 200 if symbol == 'XXII' else 100
                plt.scatter(row['volatility'] * 100, row['vega'], 
                           color=colors.get(symbol, 'gray'), s=size, alpha=0.8, label=symbol)
        
        plt.xlabel('Implied Volatility (%)')
        plt.ylabel('ATM Vega')
        plt.title('Volatility vs Vega Relationship')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot 7: XXII Risk-Return Profile
        plt.subplot(3, 3, 7)
        xxii_all_options = vega_df[vega_df['symbol'] == 'XXII']
        
        # Create risk-return scatter for XXII options
        for days in [7, 30, 90]:
            day_data = xxii_all_options[xxii_all_options['days_to_expiry'] == days]
            plt.scatter(day_data['vega'], day_data['call_price'], 
                       alpha=0.6, label=f'{days} days', s=50)
        
        plt.xlabel('Vega')
        plt.ylabel('Option Price ($)')
        plt.title('XXII Risk-Return Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 8: Investment Score Breakdown
        plt.subplot(3, 3, 8)
        if 'XXII' in investment_scores:
            xxii_breakdown = investment_scores['XXII']
            categories = ['Vega\nScore', 'Efficiency\nScore', 'Volatility\nScore', 'Cap\nScore', 'Beta\nScore']
            values = [xxii_breakdown['vega_score'], xxii_breakdown['efficiency_score'], 
                     xxii_breakdown['volatility_score'], xxii_breakdown['cap_score'], 
                     xxii_breakdown['beta_score']]
            
            bars = plt.bar(categories, values, color='red', alpha=0.7)
            plt.ylabel('Score (0-100)')
            plt.title('XXII Investment Score Breakdown')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 9: Recommendation Summary
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        if 'XXII' in investment_scores:
            xxii_score = investment_scores['XXII']['composite_score']
            rank = sorted(investment_scores.values(), key=lambda x: x['composite_score'], reverse=True)
            xxii_rank = [i for i, x in enumerate(rank, 1) if x == investment_scores['XXII']][0]
            
            recommendation = "STRONG BUY" if xxii_score >= 80 else \
                           "BUY" if xxii_score >= 65 else \
                           "HOLD" if xxii_score >= 50 else \
                           "SELL" if xxii_score >= 35 else "STRONG SELL"
            
            color = {'STRONG BUY': 'darkgreen', 'BUY': 'green', 'HOLD': 'orange', 
                    'SELL': 'red', 'STRONG SELL': 'darkred'}[recommendation]
            
            plt.text(0.5, 0.8, 'INVESTMENT RECOMMENDATION', fontsize=16, fontweight='bold', 
                    ha='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.6, recommendation, fontsize=24, fontweight='bold', 
                    ha='center', color=color, transform=plt.gca().transAxes)
            plt.text(0.5, 0.4, f'Score: {xxii_score:.1f}/100', fontsize=14, 
                    ha='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.3, f'Rank: #{xxii_rank} of {len(investment_scores)}', fontsize=12, 
                    ha='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.1, f'Based on Vega Analysis', fontsize=10, 
                    ha='center', style='italic', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig('/home/kafka/projects/NVII/xxii_investment_report.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, vega_df, investment_scores, atm_30d):
        """
        Generate detailed written investment report
        """
        print("\n" + "="*80)
        print("XXII (22ND CENTURY GROUP) INVESTMENT ANALYSIS REPORT")
        print("Based on Option Vega Analysis")
        print("="*80)
        print(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analysis Period: Option chains through 90 days")
        
        if 'XXII' not in investment_scores:
            print("\nERROR: XXII data not available for analysis")
            return
        
        xxii_data = self.market_data['XXII']
        xxii_scores = investment_scores['XXII']
        
        print(f"\nEXECUTIVE SUMMARY:")
        print("-" * 50)
        print(f"Current XXII Price: ${xxii_data['price']:.2f}")
        print(f"30-Day Volatility: {xxii_data['volatility_30d']*100:.1f}%")
        print(f"Market Capitalization: ${xxii_data['market_cap']:,.0f}")
        print(f"Investment Score: {xxii_scores['composite_score']:.1f}/100")
        
        # Determine recommendation
        score = xxii_scores['composite_score']
        if score >= 80:
            recommendation = "STRONG BUY"
            rationale = "Exceptional vega characteristics with high upside potential"
        elif score >= 65:
            recommendation = "BUY"
            rationale = "Strong vega profile with good risk-adjusted returns"
        elif score >= 50:
            recommendation = "HOLD"
            rationale = "Moderate vega appeal, suitable for specific strategies"
        elif score >= 35:
            recommendation = "SELL"
            rationale = "Below-average vega characteristics"
        else:
            recommendation = "STRONG SELL"
            rationale = "Poor vega profile with high risks"
        
        print(f"\nRECOMMENDATION: {recommendation}")
        print(f"Rationale: {rationale}")
        
        print(f"\nVEGA ANALYSIS HIGHLIGHTS:")
        print("-" * 50)
        
        xxii_atm = atm_30d[atm_30d['symbol'] == 'XXII']
        if not xxii_atm.empty:
            xxii_row = xxii_atm.iloc[0]
            max_vega = atm_30d['vega'].max()
            vega_percentile = (atm_30d['vega'] <= xxii_row['vega']).mean() * 100
            
            print(f"• ATM 30-Day Vega: {xxii_row['vega']:.4f}")
            print(f"• Vega Ranking: {vega_percentile:.0f}th percentile")
            print(f"• Vega per Dollar: ${xxii_row['vega_per_dollar']*1000:.2f} per $1,000 invested")
            print(f"• Relative to Max Vega: {(xxii_row['vega']/max_vega)*100:.1f}%")
        
        print(f"\nCOMPETITIVE ANALYSIS:")
        print("-" * 50)
        
        # Rank all stocks by composite score
        ranked_scores = sorted(investment_scores.items(), 
                             key=lambda x: x[1]['composite_score'], reverse=True)
        
        print("Investment Score Rankings:")
        for i, (symbol, scores) in enumerate(ranked_scores, 1):
            status = " ← XXII" if symbol == 'XXII' else ""
            print(f"  {i}. {symbol:4s}: {scores['composite_score']:5.1f}/100 "
                  f"(Vega: {scores['vega']:.4f}){status}")
        
        print(f"\nRISK FACTORS:")
        print("-" * 50)
        print(f"• High Volatility: {xxii_data['volatility_30d']*100:.1f}% (High risk/reward)")
        print(f"• Small Market Cap: ${xxii_data['market_cap']/1_000_000:.0f}M (Liquidity risk)")
        print(f"• Beta Exposure: {xxii_data['beta']:.1f} (Market sensitivity)")
        print(f"• Sector Risk: {xxii_data['sector']} (Regulatory/development risks)")
        
        print(f"\nOPPORTUNITY FACTORS:")
        print("-" * 50)
        print(f"• High Vega Potential: Excellent for volatility strategies")
        print(f"• Small Cap Premium: Higher growth potential")
        print(f"• Option Leverage: Significant multiplier effect")
        print(f"• Market Inefficiency: Potentially undervalued options")
        
        print(f"\nINVESTMENT STRATEGIES:")
        print("-" * 50)
        
        if score >= 65:
            print("RECOMMENDED STRATEGIES:")
            print("• Long Call Options: High vega exposure to volatility expansion")
            print("• Volatility Straddles: Benefit from large price movements")
            print("• Calendar Spreads: Capture time decay while maintaining vega")
            print("• Covered Calls: If holding stock, sell calls for premium income")
        else:
            print("ALTERNATIVE STRATEGIES:")
            print("• Short Volatility Plays: Sell options to capture premium")
            print("• Cash-Secured Puts: Generate income while waiting for entry")
            print("• Avoid Long Options: High vega works against option buyers")
        
        print(f"\nPORTFOLIO ALLOCATION GUIDANCE:")
        print("-" * 50)
        
        if score >= 80:
            allocation = "5-10%"
            risk_level = "Aggressive"
        elif score >= 65:
            allocation = "3-7%"
            risk_level = "Moderate-Aggressive"
        elif score >= 50:
            allocation = "1-3%"
            risk_level = "Conservative-Moderate"
        else:
            allocation = "0-1%"
            risk_level = "Avoid/Minimal"
        
        print(f"• Suggested Allocation: {allocation} of speculative portfolio")
        print(f"• Risk Level: {risk_level}")
        print(f"• Time Horizon: Short to medium term (volatility plays)")
        print(f"• Position Sizing: Small, multiple positions preferred")
        
        print(f"\nMONITORING CRITERIA:")
        print("-" * 50)
        print("• Volatility Changes: Monitor IV expansion/contraction")
        print("• Volume Patterns: Watch for unusual option activity")
        print("• Price Momentum: Track underlying stock movement")
        print("• Market Sentiment: Biotech sector sentiment shifts")
        
        print(f"\nDISCLAIMER:")
        print("-" * 50)
        print("This analysis is based on mathematical option pricing models and")
        print("historical volatility data. Past performance does not guarantee")
        print("future results. Options trading involves substantial risk and is")
        print("not suitable for all investors. Please consult with a qualified")
        print("financial advisor before making investment decisions.")

def main():
    analyzer = XXIIVegaInvestmentAnalyzer()
    
    print("XXII (22nd Century Group) Investment Analysis")
    print("Based on Option Vega Characteristics")
    print("="*50)
    
    # Fetch market data
    analyzer.fetch_market_data()
    
    # Calculate vega analysis
    vega_df = analyzer.calculate_comprehensive_vega_analysis()
    
    # Generate investment analysis
    investment_scores, atm_30d = analyzer.generate_investment_analysis(vega_df)
    
    # Create visualizations
    analyzer.create_investment_report_visualization(vega_df, investment_scores, atm_30d)
    
    # Generate detailed report
    analyzer.generate_detailed_report(vega_df, investment_scores, atm_30d)
    
    print(f"\nAnalysis completed successfully!")
    print(f"Investment report saved as: 'xxii_investment_report.png'")

if __name__ == "__main__":
    main()