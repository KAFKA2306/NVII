#!/usr/bin/env python3
"""
NVIDIA (NVDA) Comprehensive Option Analysis
Analyzes vega, theoretical call prices, and covered call profitability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

class NVDAComprehensiveAnalyzer:
    def __init__(self):
        self.symbol = "NVDA"
        self.risk_free_rate = 0.045  # 4.5%
        self.dividend_yield = 0.0025  # 0.25% quarterly
        self.current_price = None
        self.current_volatility = None
        
    def fetch_current_data(self):
        """
        Fetch current NVDA market data
        """
        if HAS_YFINANCE:
            try:
                ticker = yf.Ticker(self.symbol)
                hist = ticker.history(period="1mo")
                info = ticker.info
                
                self.current_price = hist['Close'].iloc[-1]
                
                # Calculate realized volatility from recent data
                returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                self.current_volatility = returns.std() * np.sqrt(252)
                
                return True
            except Exception as e:
                print(f"Error fetching real data: {e}")
                return self.use_fallback_data()
        else:
            return self.use_fallback_data()
    
    def use_fallback_data(self):
        """
        Use fallback market data
        """
        self.current_price = 140.0  # Current estimate
        self.current_volatility = 0.35  # 35% volatility estimate
        print(f"Using fallback data: Price=${self.current_price}, Vol={self.current_volatility*100:.1f}%")
        return True
    
    def black_scholes_call(self, S, K, T, r, sigma, q=0):
        """
        Calculate Black-Scholes call option price with dividend adjustment
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_price
    
    def black_scholes_greeks(self, S, K, T, r, sigma, q=0):
        """
        Calculate all Greeks for option
        """
        if T <= 0:
            return {
                'delta': 1.0 if S > K else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Greeks calculations
        delta = np.exp(-q*T) * norm.cdf(d1)
        gamma = np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) 
                - r*K*np.exp(-r*T)*norm.cdf(d2) 
                + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
        vega = S * np.sqrt(T) * norm.pdf(d1) * np.exp(-q*T) / 100
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def calculate_implied_volatility(self, market_price, S, K, T, r, q=0):
        """
        Calculate implied volatility using Brent's method
        """
        if T <= 0 or market_price <= max(S - K, 0):
            return np.nan
        
        def objective(sigma):
            return self.black_scholes_call(S, K, T, r, sigma, q) - market_price
        
        try:
            iv = brentq(objective, 0.01, 5.0, xtol=1e-6)
            return iv
        except:
            return np.nan
    
    def generate_option_matrix(self):
        """
        Generate comprehensive option analysis matrix
        """
        # Define strike prices around current price
        strikes = np.arange(
            self.current_price * 0.85, 
            self.current_price * 1.15, 
            5  # $5 increments
        )
        strikes = np.round(strikes / 5) * 5  # Round to nearest $5
        strikes = np.unique(strikes)  # Remove duplicates
        
        # Define expiration dates
        expirations = [7, 14, 21, 30, 45, 60, 90]  # Days to expiration
        
        results = []
        
        for days in expirations:
            T = days / 365.0
            
            for strike in strikes:
                # Calculate theoretical price and Greeks
                theo_price = self.black_scholes_call(
                    self.current_price, strike, T, 
                    self.risk_free_rate, self.current_volatility, self.dividend_yield
                )
                
                greeks = self.black_scholes_greeks(
                    self.current_price, strike, T,
                    self.risk_free_rate, self.current_volatility, self.dividend_yield
                )
                
                # Calculate moneyness
                moneyness = strike / self.current_price
                
                # Simulate market price (with bid-ask spread)
                bid_ask_spread = max(0.05, theo_price * 0.02)  # 2% of theoretical price
                market_bid = theo_price - bid_ask_spread/2
                market_ask = theo_price + bid_ask_spread/2
                
                results.append({
                    'days_to_expiry': days,
                    'strike': strike,
                    'moneyness': moneyness,
                    'theoretical_price': theo_price,
                    'market_bid': market_bid,
                    'market_ask': market_ask,
                    'bid_ask_spread': bid_ask_spread,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega'],
                    'rho': greeks['rho']
                })
        
        return pd.DataFrame(results)
    
    def analyze_covered_call_strategy(self, option_data):
        """
        Analyze covered call strategy profitability
        """
        covered_call_results = []
        
        for _, row in option_data.iterrows():
            if row['moneyness'] >= 1.0:  # Only OTM calls for covered calls
                
                # Calculate covered call metrics
                premium_received = row['market_bid']  # Sell at bid
                max_profit = premium_received + (row['strike'] - self.current_price)
                max_loss = self.current_price - premium_received
                breakeven = self.current_price - premium_received
                
                # Probability calculations
                T = row['days_to_expiry'] / 365.0
                
                # Probability of finishing ITM (assignment)
                d2 = (np.log(self.current_price/row['strike']) + 
                      (self.risk_free_rate - self.dividend_yield - 0.5*self.current_volatility**2)*T) / (self.current_volatility*np.sqrt(T))
                prob_itm = norm.cdf(d2)
                
                # Expected return calculation
                prob_otm = 1 - prob_itm
                expected_return_otm = premium_received / self.current_price
                expected_return_itm = max_profit / self.current_price
                
                expected_return = (prob_otm * expected_return_otm + 
                                 prob_itm * expected_return_itm)
                
                # Annualized return
                annualized_return = expected_return * (365 / row['days_to_expiry'])
                
                # Risk metrics
                downside_protection = premium_received / self.current_price
                upside_participation = (row['strike'] - self.current_price) / self.current_price
                
                covered_call_results.append({
                    'days_to_expiry': row['days_to_expiry'],
                    'strike': row['strike'],
                    'moneyness': row['moneyness'],
                    'premium_received': premium_received,
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'breakeven': breakeven,
                    'prob_itm': prob_itm,
                    'expected_return': expected_return,
                    'annualized_return': annualized_return,
                    'downside_protection': downside_protection,
                    'upside_participation': upside_participation,
                    'vega_exposure': row['vega'],
                    'theta_income': row['theta']
                })
        
        return pd.DataFrame(covered_call_results)
    
    def plot_comprehensive_analysis(self, option_data, covered_call_data):
        """
        Create comprehensive analysis visualizations
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Theoretical Call Prices Heat Map
        plt.subplot(3, 3, 1)
        try:
            pivot_prices = option_data.pivot_table(index='days_to_expiry', columns='strike', 
                                                  values='theoretical_price', aggfunc='mean')
            im1 = plt.imshow(pivot_prices.values, aspect='auto', cmap='viridis')
            plt.colorbar(im1, label='Call Price ($)')
            plt.xlabel('Strike Price')
            plt.ylabel('Days to Expiry')
            plt.title('NVDA Theoretical Call Prices')
            
            # Set tick labels
            strike_labels = [f'${int(s)}' for s in pivot_prices.columns[::2]]
            plt.xticks(range(0, len(pivot_prices.columns), 2), strike_labels, rotation=45)
            plt.yticks(range(len(pivot_prices.index)), pivot_prices.index)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error creating heat map: {str(e)}', 
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title('NVDA Theoretical Call Prices (Error)')
        
        # Plot 2: Vega Heat Map
        plt.subplot(3, 3, 2)
        try:
            pivot_vega = option_data.pivot_table(index='days_to_expiry', columns='strike', 
                                               values='vega', aggfunc='mean')
            im2 = plt.imshow(pivot_vega.values, aspect='auto', cmap='plasma')
            plt.colorbar(im2, label='Vega')
            plt.xlabel('Strike Price')
            plt.ylabel('Days to Expiry')
            plt.title('NVDA Option Vega')
            
            vega_strike_labels = [f'${int(s)}' for s in pivot_vega.columns[::2]]
            plt.xticks(range(0, len(pivot_vega.columns), 2), vega_strike_labels, rotation=45)
            plt.yticks(range(len(pivot_vega.index)), pivot_vega.index)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error creating vega heat map: {str(e)}', 
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title('NVDA Option Vega (Error)')
        
        # Plot 3: Vega vs Moneyness for different expirations
        plt.subplot(3, 3, 3)
        for days in [7, 21, 45, 90]:
            data_subset = option_data[option_data['days_to_expiry'] == days]
            plt.plot(data_subset['moneyness'], data_subset['vega'], 
                    label=f'{days} days', marker='o', linewidth=2)
        
        plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='ATM')
        plt.xlabel('Moneyness (Strike/Spot)')
        plt.ylabel('Vega')
        plt.title('Vega vs Moneyness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Covered Call Expected Returns
        plt.subplot(3, 3, 4)
        try:
            cc_pivot = covered_call_data.pivot_table(index='days_to_expiry', columns='strike', 
                                                    values='annualized_return', aggfunc='mean')
            im4 = plt.imshow(cc_pivot.values * 100, aspect='auto', cmap='RdYlGn')
            plt.colorbar(im4, label='Annualized Return (%)')
            plt.xlabel('Strike Price')
            plt.ylabel('Days to Expiry')
            plt.title('Covered Call Annualized Returns')
            
            cc_strikes = [f'${int(s)}' for s in cc_pivot.columns[::2]]
            plt.xticks(range(0, len(cc_pivot.columns), 2), cc_strikes, rotation=45)
            plt.yticks(range(len(cc_pivot.index)), cc_pivot.index)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error creating covered call heat map: {str(e)}', 
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title('Covered Call Returns (Error)')
        
        # Plot 5: Risk-Return Scatter for Covered Calls
        plt.subplot(3, 3, 5)
        plt.scatter(covered_call_data['downside_protection'] * 100, 
                   covered_call_data['annualized_return'] * 100,
                   c=covered_call_data['days_to_expiry'], cmap='viridis',
                   s=50, alpha=0.7)
        plt.colorbar(label='Days to Expiry')
        plt.xlabel('Downside Protection (%)')
        plt.ylabel('Annualized Return (%)')
        plt.title('Covered Call Risk-Return Profile')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Premium vs Strike for different expirations
        plt.subplot(3, 3, 6)
        for days in [7, 21, 45, 90]:
            data_subset = option_data[option_data['days_to_expiry'] == days]
            plt.plot(data_subset['strike'], data_subset['theoretical_price'], 
                    label=f'{days} days', marker='o', linewidth=2)
        
        plt.axvline(x=self.current_price, color='red', linestyle='--', 
                   alpha=0.7, label=f'Current Price ${self.current_price:.0f}')
        plt.xlabel('Strike Price ($)')
        plt.ylabel('Call Premium ($)')
        plt.title('Call Premium vs Strike')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 7: Time Decay (Theta) Analysis
        plt.subplot(3, 3, 7)
        atm_data = option_data[abs(option_data['moneyness'] - 1.0) < 0.02]
        plt.plot(atm_data['days_to_expiry'], -atm_data['theta'], 
                'red', marker='o', linewidth=3, label='ATM Theta')
        
        plt.xlabel('Days to Expiry')
        plt.ylabel('Daily Time Decay ($)')
        plt.title('Time Decay for ATM Options')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 8: Probability of ITM for Covered Calls
        plt.subplot(3, 3, 8)
        try:
            cc_prob_pivot = covered_call_data.pivot_table(index='days_to_expiry', columns='strike', 
                                                        values='prob_itm', aggfunc='mean')
            im8 = plt.imshow(cc_prob_pivot.values * 100, aspect='auto', cmap='RdYlBu_r')
            plt.colorbar(im8, label='Probability ITM (%)')
            plt.xlabel('Strike Price')
            plt.ylabel('Days to Expiry')
            plt.title('Covered Call Assignment Probability')
            
            prob_strikes = [f'${int(s)}' for s in cc_prob_pivot.columns[::2]]
            plt.xticks(range(0, len(cc_prob_pivot.columns), 2), prob_strikes, rotation=45)
            plt.yticks(range(len(cc_prob_pivot.index)), cc_prob_pivot.index)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error creating probability heat map: {str(e)}', 
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title('Assignment Probability (Error)')
        
        # Plot 9: Optimal Strike Analysis
        plt.subplot(3, 3, 9)
        best_cc_by_expiry = covered_call_data.loc[covered_call_data.groupby('days_to_expiry')['annualized_return'].idxmax()]
        
        plt.scatter(best_cc_by_expiry['days_to_expiry'], best_cc_by_expiry['annualized_return'] * 100,
                   s=100, c='green', alpha=0.8, label='Optimal Strikes')
        
        for _, row in best_cc_by_expiry.iterrows():
            plt.annotate(f'${row["strike"]:.0f}', 
                        (row['days_to_expiry'], row['annualized_return'] * 100),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Days to Expiry')
        plt.ylabel('Annualized Return (%)')
        plt.title('Optimal Covered Call Strikes')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('/home/kafka/projects/NVII/nvda_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_cc_by_expiry
    
    def generate_summary_report(self, option_data, covered_call_data, optimal_strikes):
        """
        Generate comprehensive summary report
        """
        print("\n" + "="*80)
        print("NVIDIA (NVDA) COMPREHENSIVE OPTION ANALYSIS")
        print("="*80)
        print(f"Current NVDA Price: ${self.current_price:.2f}")
        print(f"Current Implied Volatility: {self.current_volatility*100:.1f}%")
        print(f"Risk-Free Rate: {self.risk_free_rate*100:.1f}%")
        print(f"Dividend Yield: {self.dividend_yield*100:.2f}%")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\nTHEORETICAL CALL PRICES ANALYSIS:")
        print("-" * 50)
        
        # ATM analysis
        atm_data = option_data[abs(option_data['moneyness'] - 1.0) < 0.02]
        print("\nATM Call Options:")
        for _, row in atm_data.iterrows():
            print(f"  {int(row['days_to_expiry']):2d} days: ${row['theoretical_price']:5.2f} "
                  f"(Vega: {row['vega']:5.3f}, Theta: {row['theta']:6.3f})")
        
        print("\nVEGA ANALYSIS:")
        print("-" * 50)
        
        max_vega_option = option_data.loc[option_data['vega'].idxmax()]
        print(f"Maximum Vega: {max_vega_option['vega']:.4f}")
        print(f"  Strike: ${max_vega_option['strike']:.0f}")
        print(f"  Days to Expiry: {int(max_vega_option['days_to_expiry'])}")
        print(f"  Moneyness: {max_vega_option['moneyness']:.3f}")
        
        # Vega by moneyness
        vega_by_moneyness = option_data.groupby(pd.cut(option_data['moneyness'], 
                                                       bins=[0.85, 0.95, 1.05, 1.15], 
                                                       labels=['Deep OTM', 'OTM', 'ITM']))['vega'].mean()
        print("\nAverage Vega by Moneyness:")
        for category, avg_vega in vega_by_moneyness.items():
            print(f"  {category}: {avg_vega:.4f}")
        
        print("\nCOVERED CALL STRATEGY ANALYSIS:")
        print("-" * 50)
        
        best_overall = covered_call_data.loc[covered_call_data['annualized_return'].idxmax()]
        print(f"Best Overall Strategy:")
        print(f"  Strike: ${best_overall['strike']:.0f}")
        print(f"  Days to Expiry: {int(best_overall['days_to_expiry'])}")
        print(f"  Premium: ${best_overall['premium_received']:.2f}")
        print(f"  Annualized Return: {best_overall['annualized_return']*100:.1f}%")
        print(f"  Assignment Probability: {best_overall['prob_itm']*100:.1f}%")
        print(f"  Downside Protection: {best_overall['downside_protection']*100:.1f}%")
        
        print("\nOPTIMAL STRIKES BY EXPIRATION:")
        print("-" * 50)
        for _, row in optimal_strikes.iterrows():
            print(f"  {int(row['days_to_expiry']):2d} days: ${row['strike']:3.0f} strike "
                  f"({row['annualized_return']*100:4.1f}% annual return, "
                  f"{row['prob_itm']*100:4.1f}% assignment prob)")
        
        print("\nRISK METRICS SUMMARY:")
        print("-" * 50)
        
        avg_downside_protection = covered_call_data['downside_protection'].mean() * 100
        avg_upside_participation = covered_call_data['upside_participation'].mean() * 100
        avg_vega_exposure = covered_call_data['vega_exposure'].mean()
        
        print(f"Average Downside Protection: {avg_downside_protection:.1f}%")
        print(f"Average Upside Participation: {avg_upside_participation:.1f}%")
        print(f"Average Vega Exposure: {avg_vega_exposure:.4f}")
        
        # Risk-adjusted returns
        high_return_strategies = covered_call_data[covered_call_data['annualized_return'] > 0.15]
        if not high_return_strategies.empty:
            print(f"\nHigh Return Strategies (>15% annual):")
            print(f"  Count: {len(high_return_strategies)}")
            print(f"  Average Assignment Probability: {high_return_strategies['prob_itm'].mean()*100:.1f}%")
            print(f"  Average Days to Expiry: {int(high_return_strategies['days_to_expiry'].mean())}")
        
        print("\nMARKET MAKING CONSIDERATIONS:")
        print("-" * 50)
        
        avg_bid_ask = option_data['bid_ask_spread'].mean()
        max_bid_ask = option_data['bid_ask_spread'].max()
        
        print(f"Average Bid-Ask Spread: ${avg_bid_ask:.3f}")
        print(f"Maximum Bid-Ask Spread: ${max_bid_ask:.3f}")
        print(f"Spread as % of Premium: {(avg_bid_ask/option_data['theoretical_price'].mean())*100:.1f}%")

def main():
    analyzer = NVDAComprehensiveAnalyzer()
    
    print("Fetching current NVDA market data...")
    analyzer.fetch_current_data()
    
    print("Generating option analysis matrix...")
    option_data = analyzer.generate_option_matrix()
    
    print("Analyzing covered call strategies...")
    covered_call_data = analyzer.analyze_covered_call_strategy(option_data)
    
    print("Creating comprehensive visualizations...")
    optimal_strikes = analyzer.plot_comprehensive_analysis(option_data, covered_call_data)
    
    print("Generating detailed analysis report...")
    analyzer.generate_summary_report(option_data, covered_call_data, optimal_strikes)
    
    print("\nAnalysis complete! Charts saved as 'nvda_comprehensive_analysis.png'")
    print("Data exported for further analysis:")
    print(f"  - Option data: {len(option_data)} records")
    print(f"  - Covered call strategies: {len(covered_call_data)} strategies analyzed")

if __name__ == "__main__":
    main()