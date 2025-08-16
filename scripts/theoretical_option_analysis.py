#!/usr/bin/env python3
"""
NVII ETF Theoretical Option Pricing Analysis
===========================================

Comprehensive theoretical analysis of option pricing models for NVII ETF's covered call strategy.
Based on data from NVII.md as of August 15, 2025.

Author: Option Theoretical Value Calculator Agent
Date: August 16, 2025
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NVIIOptionAnalyzer:
    """
    Theoretical option pricing analyzer for NVII ETF covered call strategy
    """
    
    def __init__(self):
        # NVII ETF Parameters (from NVII.md)
        self.current_price = 32.97  # Current stock price (Aug 8, 2025)
        self.target_leverage = 1.25  # Target leverage ratio
        self.leverage_range = (1.05, 1.50)  # Leverage range
        self.dividend_yield = 0.063  # 6.30% annual dividend yield
        self.weekly_dividend_avg = 0.20  # Average weekly dividend
        self.annual_dividend = 2.04  # Annual dividend estimate
        self.aum = 20.34e6  # Assets under management ($20.34M)
        self.shares_outstanding = 650000  # 650K shares
        self.expense_ratio = 0.0099  # 0.99% expense ratio
        
        # Market parameters (estimated)
        self.risk_free_rate = 0.045  # 4.5% risk-free rate (10-year Treasury)
        self.historical_volatility = 0.45  # 45% historical volatility (NVDA-like)
        
        # Strategy parameters
        self.covered_call_ratio = 0.50  # 50% of portfolio in covered calls
        self.unlimited_upside_ratio = 0.50  # 50% unlimited upside
        
        # Recent dividend history for volatility calculation
        self.recent_dividends = [0.2138, 0.29268, 0.22846, 0.24721, 0.1536]
        
    def black_scholes_call(self, S, K, T, r, sigma, q=0):
        """
        Black-Scholes formula for European call options
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield
        """
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_price
    
    def black_scholes_greeks(self, S, K, T, r, sigma, q=0):
        """
        Calculate option Greeks using Black-Scholes model
        """
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Greeks calculations
        delta = np.exp(-q*T) * norm.cdf(d1)
        gamma = np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) 
                - r*K*np.exp(-r*T)*norm.cdf(d2) 
                + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
        vega = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) / 100
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def binomial_american_call(self, S, K, T, r, sigma, n=100, q=0):
        """
        Binomial tree pricing for American call options
        """
        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        asset_prices = np.zeros(n + 1)
        for i in range(n + 1):
            asset_prices[i] = S * (u ** (n - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.maximum(asset_prices - K, 0)
        
        # Work backwards through the tree
        for j in range(n - 1, -1, -1):
            for i in range(j + 1):
                # Calculate option value
                option_values[i] = np.exp(-r * dt) * (
                    p * option_values[i] + (1 - p) * option_values[i + 1]
                )
                # Check for early exercise (American option)
                asset_price = S * (u ** (j - i)) * (d ** i)
                option_values[i] = max(option_values[i], asset_price - K)
        
        return option_values[0]
    
    def monte_carlo_call(self, S, K, T, r, sigma, n_simulations=100000, q=0):
        """
        Monte Carlo simulation for option pricing
        """
        dt = T / 252  # Daily steps
        n_steps = int(T * 252)
        
        # Generate random paths
        Z = np.random.standard_normal((n_simulations, n_steps))
        
        # Initialize price paths
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = S
        
        # Generate price paths
        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
            )
        
        # Calculate payoffs
        payoffs = np.maximum(paths[:, -1] - K, 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        return option_price, np.std(payoffs) / np.sqrt(n_simulations)
    
    def implied_volatility(self, market_price, S, K, T, r, q=0):
        """
        Calculate implied volatility using Newton-Raphson method
        """
        def objective(sigma):
            return abs(self.black_scholes_call(S, K, T, r, sigma, q) - market_price)
        
        result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
        return result.x if result.success else None
    
    def calculate_optimal_strikes(self, otm_levels=[0.02, 0.05, 0.08, 0.10]):
        """
        Calculate optimal strike prices for different OTM levels
        """
        strikes = {}
        for otm in otm_levels:
            strike = self.current_price * (1 + otm)
            strikes[f'{otm*100:.0f}% OTM'] = strike
        return strikes
    
    def analyze_volatility_regimes(self):
        """
        Analyze different volatility regimes and their impact
        """
        regimes = {
            'Low Volatility': 0.25,
            'Normal Volatility': 0.45,
            'High Volatility': 0.65,
            'Extreme Volatility': 0.85
        }
        
        return regimes
    
    def calculate_theoretical_premiums(self, time_to_expiry_weeks=1):
        """
        Calculate theoretical option premiums for various strikes and models
        """
        T = time_to_expiry_weeks / 52  # Convert weeks to years
        strikes = self.calculate_optimal_strikes()
        
        results = []
        
        for strike_label, K in strikes.items():
            # Black-Scholes pricing
            bs_price = self.black_scholes_call(
                self.current_price, K, T, self.risk_free_rate, 
                self.historical_volatility, self.dividend_yield
            )
            
            # Binomial pricing
            bin_price = self.binomial_american_call(
                self.current_price, K, T, self.risk_free_rate, 
                self.historical_volatility, q=self.dividend_yield
            )
            
            # Monte Carlo pricing
            mc_price, mc_std = self.monte_carlo_call(
                self.current_price, K, T, self.risk_free_rate, 
                self.historical_volatility, q=self.dividend_yield
            )
            
            # Greeks calculation
            greeks = self.black_scholes_greeks(
                self.current_price, K, T, self.risk_free_rate, 
                self.historical_volatility, self.dividend_yield
            )
            
            results.append({
                'Strike_Label': strike_label,
                'Strike_Price': K,
                'Moneyness': K / self.current_price,
                'BS_Price': bs_price,
                'Binomial_Price': bin_price,
                'MC_Price': mc_price,
                'MC_StdErr': mc_std,
                'Delta': greeks['delta'],
                'Gamma': greeks['gamma'],
                'Theta': greeks['theta'],
                'Vega': greeks['vega'],
                'Rho': greeks['rho']
            })
        
        return pd.DataFrame(results)
    
    def strategy_optimization_analysis(self):
        """
        Analyze strategy optimization for covered call implementation
        """
        # Calculate weekly option income potential
        premiums_df = self.calculate_theoretical_premiums()
        
        # Estimate weekly income from covered calls (50% of portfolio)
        portfolio_value = self.current_price * self.shares_outstanding
        covered_call_value = portfolio_value * self.covered_call_ratio
        
        # Number of contracts (assuming 100 shares per contract)
        contracts = covered_call_value / (self.current_price * 100)
        
        strategy_analysis = {}
        
        for _, row in premiums_df.iterrows():
            weekly_premium_income = row['BS_Price'] * contracts * 100
            annual_premium_income = weekly_premium_income * 52
            enhanced_yield = annual_premium_income / portfolio_value
            total_yield = self.dividend_yield + enhanced_yield
            
            strategy_analysis[row['Strike_Label']] = {
                'Weekly_Premium_Income': weekly_premium_income,
                'Annual_Premium_Income': annual_premium_income,
                'Enhanced_Yield': enhanced_yield,
                'Total_Yield': total_yield,
                'Assignment_Risk': row['Delta'],
                'Time_Decay_Benefit': abs(row['Theta']) * contracts * 100
            }
        
        return strategy_analysis
    
    def risk_management_analysis(self):
        """
        Comprehensive risk management analysis
        """
        risk_metrics = {}
        
        # Portfolio concentration risk
        risk_metrics['Concentration_Risk'] = {
            'Single_Stock_Exposure': 1.0,  # 100% NVIDIA exposure
            'Sector_Concentration': 'Technology',
            'Leverage_Risk': self.target_leverage,
            'Max_Leverage': self.leverage_range[1]
        }
        
        # Option strategy risks
        risk_metrics['Option_Strategy_Risk'] = {
            'Covered_Call_Ratio': self.covered_call_ratio,
            'Upside_Limitation': 'Partial (50% of portfolio)',
            'Income_Dependency': 'High (6.30% yield target)'
        }
        
        # Volatility sensitivity
        vol_regimes = self.analyze_volatility_regimes()
        risk_metrics['Volatility_Sensitivity'] = {}
        
        for regime, vol in vol_regimes.items():
            # Calculate impact on 5% OTM call
            strike = self.current_price * 1.05
            T = 1/52  # 1 week
            
            option_value = self.black_scholes_call(
                self.current_price, strike, T, self.risk_free_rate, vol, self.dividend_yield
            )
            
            risk_metrics['Volatility_Sensitivity'][regime] = {
                'Volatility': vol,
                'Option_Value': option_value,
                'Weekly_Income_Impact': option_value * (self.shares_outstanding * 0.5 / 100)
            }
        
        return risk_metrics
    
    def performance_attribution(self):
        """
        Analyze performance attribution between leverage and option income
        """
        # Base NVIDIA performance (without leverage)
        base_return = 0.27  # 27% return since inception (from NVII.md)
        
        # Leveraged return contribution
        leveraged_return = base_return * self.target_leverage
        leverage_alpha = leveraged_return - base_return
        
        # Option income contribution
        annual_option_income = self.annual_dividend
        option_yield = annual_option_income / self.current_price
        
        attribution = {
            'Base_NVIDIA_Return': base_return,
            'Leverage_Contribution': leverage_alpha,
            'Option_Income_Yield': option_yield,
            'Total_Expected_Return': leveraged_return + option_yield,
            'Risk_Adjusted_Sharpe': (leveraged_return + option_yield - self.risk_free_rate) / (self.historical_volatility * np.sqrt(self.target_leverage))
        }
        
        return attribution
    
    def stress_testing(self):
        """
        Stress testing under various market scenarios
        """
        scenarios = {
            'Bull_Market': {'price_change': 0.20, 'vol_change': 0.30},
            'Bear_Market': {'price_change': -0.30, 'vol_change': 0.60},
            'High_Vol_Flat': {'price_change': 0.0, 'vol_change': 0.80},
            'Low_Vol_Decline': {'price_change': -0.15, 'vol_change': 0.20},
            'Crash_Scenario': {'price_change': -0.50, 'vol_change': 1.00}
        }
        
        stress_results = {}
        
        for scenario, params in scenarios.items():
            stressed_price = self.current_price * (1 + params['price_change'])
            stressed_vol = self.historical_volatility * (1 + params['vol_change'])
            
            # Calculate option values under stress
            strike = self.current_price * 1.05  # 5% OTM
            T = 1/52  # 1 week
            
            option_value = self.black_scholes_call(
                stressed_price, strike, T, self.risk_free_rate, stressed_vol, self.dividend_yield
            )
            
            # Portfolio impact
            portfolio_change = params['price_change'] * self.target_leverage
            option_income_change = (option_value / (self.current_price * 1.05 - self.current_price)) - 1
            
            stress_results[scenario] = {
                'Price_Change': params['price_change'],
                'Volatility_Change': params['vol_change'],
                'Portfolio_Impact': portfolio_change,
                'Option_Income_Impact': option_income_change,
                'Net_Impact': portfolio_change + option_income_change * 0.1  # Approximate option income weight
            }
        
        return stress_results

def main():
    """
    Main analysis function
    """
    print("NVII ETF Theoretical Option Pricing Analysis")
    print("=" * 50)
    
    analyzer = NVIIOptionAnalyzer()
    
    # 1. Theoretical Option Pricing
    print("\n1. THEORETICAL OPTION PRICING MODELS")
    print("-" * 40)
    
    premiums_df = analyzer.calculate_theoretical_premiums()
    print("\nTheoretical Option Premiums (1-week expiration):")
    print(premiums_df.round(4))
    
    # 2. Greeks Analysis
    print("\n2. OPTION GREEKS ANALYSIS")
    print("-" * 30)
    
    greeks_summary = premiums_df[['Strike_Label', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho']].round(4)
    print(greeks_summary)
    
    # 3. Strategy Optimization
    print("\n3. STRATEGY OPTIMIZATION ANALYSIS")
    print("-" * 35)
    
    strategy_analysis = analyzer.strategy_optimization_analysis()
    for strike, metrics in strategy_analysis.items():
        print(f"\n{strike}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # 4. Risk Management
    print("\n4. RISK MANAGEMENT ANALYSIS")
    print("-" * 30)
    
    risk_analysis = analyzer.risk_management_analysis()
    print("\nVolatility Sensitivity Analysis:")
    for regime, metrics in risk_analysis['Volatility_Sensitivity'].items():
        print(f"{regime}: Vol={metrics['Volatility']:.1%}, "
              f"Option Value=${metrics['Option_Value']:.4f}, "
              f"Weekly Income=${metrics['Weekly_Income_Impact']:.0f}")
    
    # 5. Performance Attribution
    print("\n5. PERFORMANCE ATTRIBUTION")
    print("-" * 28)
    
    attribution = analyzer.performance_attribution()
    for key, value in attribution.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 6. Stress Testing
    print("\n6. STRESS TESTING RESULTS")
    print("-" * 25)
    
    stress_results = analyzer.stress_testing()
    for scenario, results in stress_results.items():
        print(f"\n{scenario}:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    
    # 7. Key Recommendations
    print("\n7. KEY RECOMMENDATIONS")
    print("-" * 22)
    
    print("\nOptimal Strategy Parameters:")
    print(f"• Target Strike: 5% OTM (${analyzer.current_price * 1.05:.2f})")
    print(f"• Expected Weekly Premium: ${premiums_df.loc[1, 'BS_Price']:.4f}")
    print(f"• Risk-Adjusted Return: {attribution['Risk_Adjusted_Sharpe']:.4f}")
    print(f"• Recommended Position Size: 50% of portfolio")
    
    print("\nRisk Management Guidelines:")
    print("• Monitor NVIDIA earnings and guidance closely")
    print("• Adjust strike selection based on realized volatility")
    print("• Consider volatility regime changes for option pricing")
    print("• Maintain diversification outside of NVII for concentration risk")

if __name__ == "__main__":
    main()