#!/usr/bin/env python3
"""
NVII Weekly Option Pricing Analysis - Phase 1
Comprehensive analysis of weekly covered call strategy for NVII ETF
Focuses on NVIDIA volatility patterns and option income generation
"""

import numpy as np
import scipy.stats as stats
from math import log, sqrt, exp
import pandas as pd
from datetime import datetime, timedelta

class BlackScholesCalculator:
    """Black-Scholes option pricing model implementation"""
    
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        """
        Initialize Black-Scholes parameters
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
    
    def d1(self):
        """Calculate d1 parameter"""
        return (log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt(self.T))
    
    def d2(self):
        """Calculate d2 parameter"""
        return self.d1() - self.sigma * sqrt(self.T)
    
    def theoretical_price(self):
        """Calculate theoretical option price"""
        d1_val = self.d1()
        d2_val = self.d2()
        
        if self.option_type == 'call':
            price = (self.S * stats.norm.cdf(d1_val) - 
                    self.K * exp(-self.r * self.T) * stats.norm.cdf(d2_val))
        else:  # put
            price = (self.K * exp(-self.r * self.T) * stats.norm.cdf(-d2_val) - 
                    self.S * stats.norm.cdf(-d1_val))
        
        return price
    
    def delta(self):
        """Calculate Delta (price sensitivity to underlying)"""
        d1_val = self.d1()
        if self.option_type == 'call':
            return stats.norm.cdf(d1_val)
        else:
            return stats.norm.cdf(d1_val) - 1
    
    def gamma(self):
        """Calculate Gamma (Delta sensitivity to underlying)"""
        d1_val = self.d1()
        return stats.norm.pdf(d1_val) / (self.S * self.sigma * sqrt(self.T))
    
    def theta(self):
        """Calculate Theta (time decay)"""
        d1_val = self.d1()
        d2_val = self.d2()
        
        if self.option_type == 'call':
            theta = ((-self.S * stats.norm.pdf(d1_val) * self.sigma) / (2 * sqrt(self.T)) -
                    self.r * self.K * exp(-self.r * self.T) * stats.norm.cdf(d2_val))
        else:
            theta = ((-self.S * stats.norm.pdf(d1_val) * self.sigma) / (2 * sqrt(self.T)) +
                    self.r * self.K * exp(-self.r * self.T) * stats.norm.cdf(-d2_val))
        
        return theta / 365  # Convert to daily theta
    
    def vega(self):
        """Calculate Vega (volatility sensitivity)"""
        d1_val = self.d1()
        return self.S * stats.norm.pdf(d1_val) * sqrt(self.T) / 100  # Per 1% volatility change
    
    def rho(self):
        """Calculate Rho (interest rate sensitivity)"""
        d2_val = self.d2()
        
        if self.option_type == 'call':
            return self.K * self.T * exp(-self.r * self.T) * stats.norm.cdf(d2_val) / 100
        else:
            return -self.K * self.T * exp(-self.r * self.T) * stats.norm.cdf(-d2_val) / 100

def implied_volatility(market_price, S, K, T, r, option_type='call', max_iterations=100, tolerance=0.0001):
    """
    Calculate implied volatility using Newton-Raphson method
    """
    # Initial guess
    sigma = 0.5
    
    for i in range(max_iterations):
        bs = BlackScholesCalculator(S, K, T, r, sigma, option_type)
        price = bs.theoretical_price()
        vega = bs.vega() * 100  # Convert back to per unit volatility
        
        price_diff = price - market_price
        
        if abs(price_diff) < tolerance:
            return sigma
        
        if vega == 0:
            break
            
        sigma = sigma - price_diff / vega
        
        # Ensure sigma stays positive
        if sigma <= 0:
            sigma = 0.01
    
    return sigma

def dividend_analysis(shares, quarterly_dividend, current_price):
    """Analyze dividend metrics"""
    total_quarterly_income = shares * quarterly_dividend
    annual_dividend = quarterly_dividend * 4
    annual_income = total_quarterly_income * 4
    dividend_yield = (annual_dividend / current_price) * 100
    
    return {
        'quarterly_income': total_quarterly_income,
        'annual_dividend_per_share': annual_dividend,
        'annual_income': annual_income,
        'dividend_yield': dividend_yield
    }

class NVIIAnalyzer:
    """NVII-specific analysis for weekly covered call strategy"""
    
    def __init__(self, current_price=32.97, leverage=1.25):
        self.current_price = current_price
        self.leverage = leverage
        self.risk_free_rate = 0.045  # Current 3-month Treasury rate
        self.current_dividend_yield = 0.063  # 6.30% annual
        
        # NVIDIA historical volatility estimates
        self.nvda_volatility_regimes = {
            'low': 0.35,      # 35% - calm market periods
            'medium': 0.55,   # 55% - normal tech volatility
            'high': 0.75,     # 75% - high volatility periods
            'extreme': 1.0    # 100% - crisis/earnings volatility
        }
        
    def calculate_weekly_options(self, volatility_regime='medium'):
        """Calculate weekly option values for various strikes"""
        vol = self.nvda_volatility_regimes[volatility_regime]
        T_weekly = 7/365  # Weekly expiration
        
        # Strike prices around current NVII price
        strikes = np.arange(30.0, 38.0, 0.5)  # $30 to $37.50 in $0.50 increments
        
        results = []
        for strike in strikes:
            # Only analyze out-of-the-money calls for covered call strategy
            if strike > self.current_price:
                bs = BlackScholesCalculator(self.current_price, strike, T_weekly, 
                                          self.risk_free_rate, vol, 'call')
                
                option_price = bs.theoretical_price()
                delta = bs.delta()
                theta = bs.theta()
                
                # Calculate moneyness and leverage impact
                moneyness = strike / self.current_price
                leverage_adjusted_price = option_price * self.leverage
                
                results.append({
                    'strike': strike,
                    'option_price': option_price,
                    'leverage_adjusted_price': leverage_adjusted_price,
                    'delta': delta,
                    'theta': theta,
                    'moneyness': moneyness,
                    'volatility': vol,
                    'regime': volatility_regime
                })
        
        return pd.DataFrame(results)
    
    def calculate_income_scenarios(self):
        """Calculate expected premium income under different scenarios"""
        scenarios = {}
        
        for regime in self.nvda_volatility_regimes.keys():
            weekly_options = self.calculate_weekly_options(regime)
            
            # Assume selling 50% of position as covered calls
            position_coverage = 0.5
            
            # Calculate weekly premium income for different strike selections
            premium_income = {}
            
            for pct_otm in [0.02, 0.05, 0.08, 0.10]:  # 2%, 5%, 8%, 10% out-of-the-money
                target_strike = self.current_price * (1 + pct_otm)
                
                # Find closest strike
                closest_option = weekly_options.iloc[
                    (weekly_options['strike'] - target_strike).abs().argsort()[:1]
                ]
                
                if not closest_option.empty:
                    option_price = closest_option['option_price'].iloc[0]
                    leverage_adjusted = closest_option['leverage_adjusted_price'].iloc[0]
                    
                    # Weekly premium per share (50% of position)
                    weekly_premium = leverage_adjusted * position_coverage
                    annual_premium = weekly_premium * 52
                    yield_on_price = (annual_premium / self.current_price) * 100
                    
                    premium_income[f'{pct_otm*100:.0f}%_OTM'] = {
                        'strike': closest_option['strike'].iloc[0],
                        'weekly_premium': weekly_premium,
                        'annual_premium': annual_premium,
                        'yield_on_price': yield_on_price
                    }
            
            scenarios[regime] = premium_income
        
        return scenarios

def analyze_nvda_volatility():
    """Analyze NVIDIA volatility patterns for NVII strategy"""
    print("="*80)
    print("NVIDIA VOLATILITY ANALYSIS FOR NVII STRATEGY")
    print("="*80)
    
    analyzer = NVIIAnalyzer()
    
    print(f"Current NVII Price: ${analyzer.current_price:.2f}")
    print(f"Target Leverage: {analyzer.leverage:.2f}x")
    print(f"Current Dividend Yield: {analyzer.current_dividend_yield*100:.2f}%")
    print(f"Risk-free Rate: {analyzer.risk_free_rate*100:.2f}%")
    
    print("\nNVIDIA Volatility Regimes:")
    for regime, vol in analyzer.nvda_volatility_regimes.items():
        print(f"  {regime.title()}: {vol*100:.0f}%")
    
    return analyzer

def enhanced_market_analysis():
    """Enhanced analysis with market data and stress testing"""
    print("\n" + "="*80)
    print("7. ENHANCED MARKET ANALYSIS")
    print("="*80)
    
    analyzer = NVIIAnalyzer()
    
    # Simulate different market scenarios
    scenarios = {
        'Bull Market': {'vol': 0.45, 'trend': 0.15},
        'Bear Market': {'vol': 0.85, 'trend': -0.20},
        'Sideways Market': {'vol': 0.35, 'trend': 0.02},
        'Volatile Market': {'vol': 0.90, 'trend': 0.05},
        'Crisis Period': {'vol': 1.20, 'trend': -0.35}
    }
    
    print("\nMarket Scenario Analysis for NVII Strategy:")
    print("Scenario        | Volatility | Expected Trend | Option Yield | Risk Level")
    print("-" * 75)
    
    for scenario, params in scenarios.items():
        # Create temporary analyzer with scenario parameters
        temp_analyzer = NVIIAnalyzer()
        vol = params['vol']
        
        # Calculate option price for 5% OTM
        T_weekly = 7/365
        strike = analyzer.current_price * 1.05
        
        bs = BlackScholesCalculator(analyzer.current_price, strike, T_weekly, 
                                  analyzer.risk_free_rate, vol, 'call')
        option_price = bs.theoretical_price()
        leverage_adjusted = option_price * analyzer.leverage
        
        # Calculate annual yield
        weekly_premium = leverage_adjusted * 0.5  # 50% coverage
        annual_yield = (weekly_premium * 52 / analyzer.current_price) * 100
        
        # Risk assessment
        if vol < 0.4:
            risk = "Low"
        elif vol < 0.7:
            risk = "Medium"
        elif vol < 1.0:
            risk = "High"
        else:
            risk = "Extreme"
        
        print(f"{scenario:15} | {vol*100:9.0f}% | {params['trend']*100:11.0f}% | "
              f"{annual_yield:10.1f}% | {risk:10}")
    
    return scenarios

def main():
    """Execute Phase 1 NVII analysis"""
    print("="*80)
    print("NVII WEEKLY OPTION PRICING ANALYSIS - PHASE 1")
    print("="*80)
    
    # Initialize NVII analyzer
    analyzer = analyze_nvda_volatility()
    
    print("\n" + "="*80)
    print("1. WEEKLY OPTION VALUES ACROSS VOLATILITY REGIMES")
    print("="*80)
    
    # Calculate option values for each volatility regime
    all_regime_data = {}
    for regime in analyzer.nvda_volatility_regimes.keys():
        weekly_options = analyzer.calculate_weekly_options(regime)
        all_regime_data[regime] = weekly_options
        
        print(f"\n{regime.upper()} VOLATILITY REGIME ({analyzer.nvda_volatility_regimes[regime]*100:.0f}%):")
        print("Strike | Option Price | Lev.Adj Price | Delta | Theta | Moneyness")
        print("-" * 65)
        
        for _, row in weekly_options.head(8).iterrows():  # Show first 8 strikes
            print(f"${row['strike']:5.1f} | ${row['option_price']:10.4f} | "
                  f"${row['leverage_adjusted_price']:11.4f} | {row['delta']:5.3f} | "
                  f"${row['theta']:6.4f} | {row['moneyness']:7.4f}")
    
    print("\n" + "="*80)
    print("2. PREMIUM INCOME SCENARIOS")
    print("="*80)
    
    income_scenarios = analyzer.calculate_income_scenarios()
    
    print("\nWeekly Premium Income Analysis (50% position coverage):")
    print("Regime    | Strike Type | Strike  | Weekly $ | Annual $ | Yield %")
    print("-" * 68)
    
    for regime, scenarios in income_scenarios.items():
        for otm_level, data in scenarios.items():
            print(f"{regime:9} | {otm_level:11} | ${data['strike']:6.2f} | "
                  f"${data['weekly_premium']:7.4f} | ${data['annual_premium']:7.2f} | "
                  f"{data['yield_on_price']:6.2f}")
    
    print("\n" + "="*80)
    print("3. LEVERAGE IMPACT ANALYSIS")
    print("="*80)
    
    # Analyze leverage impact on option pricing
    leverages = [1.05, 1.25, 1.50]  # NVII's leverage range
    base_vol = analyzer.nvda_volatility_regimes['medium']
    
    print("\nLeverage Impact on Weekly Option Premiums:")
    print("Leverage | 2% OTM | 5% OTM | 8% OTM | 10% OTM")
    print("-" * 48)
    
    for lev in leverages:
        temp_analyzer = NVIIAnalyzer(current_price=32.97, leverage=lev)
        temp_scenarios = temp_analyzer.calculate_income_scenarios()
        medium_scenario = temp_scenarios['medium']
        
        premiums = [medium_scenario[f'{pct}%_OTM']['weekly_premium'] 
                   for pct in [2, 5, 8, 10]]
        
        print(f"{lev:7.2f}x | ${premiums[0]:5.3f} | ${premiums[1]:5.3f} | "
              f"${premiums[2]:5.3f} | ${premiums[3]:6.3f}")
    
    print("\n" + "="*80)
    print("4. WEEKLY ROLLOVER STRATEGY MODELING")
    print("="*80)
    
    # Model weekly rollover strategy
    def model_rollover_income(weeks=52, volatility_regime='medium'):
        """Model income from weekly option rollovers"""
        weekly_incomes = []
        cumulative_income = 0
        
        # Use 5% OTM as typical strategy
        base_scenario = income_scenarios[volatility_regime]['5%_OTM']
        weekly_premium = base_scenario['weekly_premium']
        
        # Add some variability to weekly premiums (±20%)
        np.random.seed(42)  # For reproducible results
        volatility_multipliers = np.random.normal(1.0, 0.2, weeks)
        
        for week in range(weeks):
            week_premium = weekly_premium * max(0.3, volatility_multipliers[week])
            weekly_incomes.append(week_premium)
            cumulative_income += week_premium
        
        return weekly_incomes, cumulative_income
    
    print("\nAnnual Rollover Income Projections (5% OTM strategy):")
    print("Volatility Regime | Weekly Avg | Annual Total | Effective Yield")
    print("-" * 62)
    
    for regime in analyzer.nvda_volatility_regimes.keys():
        weekly_incomes, annual_total = model_rollover_income(52, regime)
        avg_weekly = np.mean(weekly_incomes)
        effective_yield = (annual_total / analyzer.current_price) * 100
        
        print(f"{regime:16} | ${avg_weekly:9.4f} | ${annual_total:11.2f} | {effective_yield:12.2f}%")
    
    print("\n" + "="*80)
    print("5. RISK-ADJUSTED RETURN ANALYSIS")
    print("="*80)
    
    print("\nComparison with Current NVII Dividend Yield (6.30%):")
    print("Volatility Regime | Option Yield | vs Current | Risk Assessment")
    print("-" * 68)
    
    current_yield = 6.30
    for regime in analyzer.nvda_volatility_regimes.keys():
        regime_scenario = income_scenarios[regime]['5%_OTM']
        option_yield = regime_scenario['yield_on_price']
        vs_current = option_yield - current_yield
        
        if regime == 'low':
            risk = "Conservative"
        elif regime == 'medium':
            risk = "Moderate"
        elif regime == 'high':
            risk = "Aggressive"
        else:
            risk = "Speculative"
        
        print(f"{regime:16} | {option_yield:11.2f}% | {vs_current:+9.2f}% | {risk}")
    
    print("\n" + "="*80)
    print("6. PHASE 1 SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    print("\nKey Findings:")
    print(f"1. NVII's 1.25x leverage amplifies option premiums by 25%")
    print(f"2. Weekly 5% OTM calls generate {income_scenarios['medium']['5%_OTM']['yield_on_price']:.1f}% annual yield in normal volatility")
    print(f"3. Strategy performs best in medium-to-high volatility environments")
    print(f"4. Current 6.30% dividend yield is achievable in low volatility periods")
    
    medium_yield = income_scenarios['medium']['5%_OTM']['yield_on_price']
    high_yield = income_scenarios['high']['5%_OTM']['yield_on_price']
    
    print(f"\nStrategic Recommendations:")
    print(f"- Target 5% OTM strikes for optimal risk/reward balance")
    print(f"- Expect {medium_yield:.1f}%-{high_yield:.1f}% annual yields from option premiums")
    print(f"- Weekly rollover strategy provides consistent income stream")
    print(f"- Leverage effect creates meaningful premium enhancement")
    
    print(f"\nNext Phase Requirements:")
    print(f"- Portfolio construction analysis with risk metrics")
    print(f"- Stress testing under different market scenarios")
    print(f"- Comparison with traditional dividend strategies")
    
    # Run enhanced market analysis
    enhanced_market_analysis()
    
    print("\n" + "="*80)
    print("8. TECHNICAL VALIDATION")
    print("="*80)
    
    # Validate Black-Scholes assumptions for NVII
    print("\nBlack-Scholes Model Validation for NVII:")
    print("Assumption                    | Status      | Notes")
    print("-" * 60)
    print("Constant volatility           | VIOLATED    | NVDA has regime changes")
    print("Constant risk-free rate       | REASONABLE  | Weekly timeframes")
    print("No dividends                  | REASONABLE  | NVDA dividends are small")
    print("European exercise             | REASONABLE  | Most index options")
    print("Efficient markets             | REASONABLE  | NVDA is liquid")
    print("Log-normal price distribution | QUESTIONABLE| Tech stocks can be skewed")
    
    print("\nModel Adjustments Recommended:")
    print("- Use implied volatility when available")
    print("- Consider volatility smile effects")
    print("- Monitor for early exercise premium")
    print("- Adjust for dividend announcements")
    
    print("\n" + "="*80)
    print("9. IMPLEMENTATION CONSIDERATIONS")
    print("="*80)
    
    print("\nWeekly Option Selection Criteria:")
    print("1. Target 5% OTM for balanced risk/reward")
    print("2. Minimum 0.05 delta to ensure meaningful premium")
    print("3. Sufficient open interest (>100 contracts)")
    print("4. Tight bid-ask spreads (<10% of option price)")
    
    print("\nRisk Management Parameters:")
    print("- Maximum 50% portfolio covered call exposure")
    print("- Stop-loss at 50% of premium collected")
    print("- Roll options at 2 DTE if profitable")
    print("- Suspend strategy if IV rank < 20%")
    
    print("\nExpected Performance Ranges:")
    medium_yield = income_scenarios['medium']['5%_OTM']['yield_on_price']
    high_yield = income_scenarios['high']['5%_OTM']['yield_on_price']
    low_yield = income_scenarios['low']['5%_OTM']['yield_on_price']
    
    print(f"Conservative estimate (low vol): {low_yield:.1f}% annual")
    print(f"Base case estimate (med vol):   {medium_yield:.1f}% annual")
    print(f"Aggressive estimate (high vol): {high_yield:.1f}% annual")
    print(f"Current NVII dividend yield:    6.3% annual")
    
    print("\n" + "="*80)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("="*80)
    
    print("\nReady for Phase 2: Portfolio Strategy Analysis")
    print("- Portfolio construction with risk budgets")
    print("- Stress testing and scenario analysis") 
    print("- Benchmark comparison and performance attribution")
    print("- Implementation timeline and monitoring framework")

if __name__ == "__main__":
    main()