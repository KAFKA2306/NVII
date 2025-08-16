#!/usr/bin/env python3
"""
NVII Monte Carlo Simulation Framework
Advanced stochastic modeling for portfolio strategy validation
Incorporates volatility regime changes, path-dependent option outcomes, and realistic market dynamics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

from option_analysis import BlackScholesCalculator, NVIIAnalyzer

class MonteCarloNVIISimulator:
    """Advanced Monte Carlo simulation for NVII strategy performance"""
    
    def __init__(self, initial_capital=100000, nvii_price=32.97, leverage=1.25, time_horizon=1.0):
        self.initial_capital = initial_capital
        self.nvii_price = nvii_price
        self.leverage = leverage
        self.time_horizon = time_horizon
        self.risk_free_rate = 0.045
        
        # Portfolio allocation
        self.covered_call_allocation = 0.5
        self.unlimited_allocation = 0.5
        
        # Volatility regime parameters
        self.volatility_regimes = {
            0: {'vol': 0.35, 'name': 'Low', 'mean_duration': 90},
            1: {'vol': 0.55, 'name': 'Medium', 'mean_duration': 60},
            2: {'vol': 0.75, 'name': 'High', 'mean_duration': 30},
            3: {'vol': 1.00, 'name': 'Extreme', 'mean_duration': 14}
        }
        
        # Volatility regime transition matrix (daily)
        self.transition_matrix = np.array([
            [0.995, 0.004, 0.0008, 0.0002],  # Low to [Low, Med, High, Extreme]
            [0.01, 0.985, 0.004, 0.001],     # Medium to [Low, Med, High, Extreme]
            [0.02, 0.05, 0.92, 0.01],        # High to [Low, Med, High, Extreme]
            [0.05, 0.1, 0.2, 0.65]           # Extreme to [Low, Med, High, Extreme]
        ])
        
        # Market scenario parameters
        self.market_scenarios = {
            'bull': {'drift': 0.25, 'probability': 0.25},
            'moderate_bull': {'drift': 0.15, 'probability': 0.20},
            'sideways': {'drift': 0.03, 'probability': 0.30},
            'moderate_bear': {'drift': -0.10, 'probability': 0.15},
            'bear': {'drift': -0.25, 'probability': 0.10}
        }
        
    def simulate_volatility_regime_path(self, days):
        """Simulate volatility regime changes over time using Markov chain"""
        regimes = np.zeros(days, dtype=int)
        regimes[0] = 1  # Start in medium volatility regime
        
        for day in range(1, days):
            current_regime = regimes[day-1]
            transition_probs = self.transition_matrix[current_regime]
            regimes[day] = np.random.choice(4, p=transition_probs)
        
        return regimes
    
    def simulate_stock_price_path(self, days, annual_drift, volatility_regimes):
        """Simulate stock price path with regime-dependent volatility"""
        dt = 1/252  # Daily time step
        prices = np.zeros(days + 1)
        prices[0] = self.nvii_price
        
        daily_drift = annual_drift * dt
        
        for day in range(days):
            current_regime = volatility_regimes[day]
            current_vol = self.volatility_regimes[current_regime]['vol']
            daily_vol = current_vol * np.sqrt(dt)
            
            # Geometric Brownian Motion with regime-dependent volatility
            shock = np.random.normal(0, 1)
            prices[day + 1] = prices[day] * np.exp(
                (daily_drift - 0.5 * current_vol**2 * dt) + daily_vol * shock
            )
        
        return prices
    
    def calculate_weekly_option_income(self, price_path, volatility_regimes):
        """Calculate weekly option income based on price path and volatility"""
        days = len(price_path) - 1
        weekly_income = []
        weekly_assignments = []
        cumulative_income = 0
        
        # Process in weekly chunks
        for week_start in range(0, days, 7):
            week_end = min(week_start + 7, days)
            
            if week_end - week_start < 7:  # Skip incomplete weeks
                continue
            
            # Starting price for the week
            start_price = price_path[week_start]
            end_price = price_path[week_end]
            
            # Average volatility for the week
            week_regimes = volatility_regimes[week_start:week_end]
            avg_vol = np.mean([self.volatility_regimes[r]['vol'] for r in week_regimes])
            
            # Calculate 5% OTM call option
            strike_price = start_price * 1.05
            time_to_expiry = 7/365
            
            try:
                bs = BlackScholesCalculator(
                    start_price, strike_price, time_to_expiry,
                    self.risk_free_rate, avg_vol, 'call'
                )
                option_price = bs.theoretical_price()
                leverage_adjusted_premium = option_price * self.leverage
                
                # Check for assignment
                assigned = end_price > strike_price
                weekly_assignments.append(assigned)
                
                # Premium income (per share, 50% coverage)
                income_per_share = leverage_adjusted_premium * self.covered_call_allocation
                weekly_income.append(income_per_share)
                cumulative_income += income_per_share
                
            except:
                # Handle edge cases (e.g., negative time value)
                weekly_income.append(0)
                weekly_assignments.append(False)
        
        return weekly_income, weekly_assignments, cumulative_income
    
    def simulate_portfolio_performance(self, annual_drift, num_simulations=1000):
        """Run Monte Carlo simulation for portfolio performance"""
        days = int(252 * self.time_horizon)
        results = []
        
        # Portfolio allocation
        total_shares = self.initial_capital / self.nvii_price
        covered_call_shares = total_shares * self.covered_call_allocation
        unlimited_shares = total_shares * self.unlimited_allocation
        
        for sim in range(num_simulations):
            # Generate regime and price paths
            volatility_regimes = self.simulate_volatility_regime_path(days)
            price_path = self.simulate_stock_price_path(days, annual_drift, volatility_regimes)
            
            # Calculate option income
            weekly_income, assignments, total_option_income = self.calculate_weekly_option_income(
                price_path, volatility_regimes
            )
            
            # Final portfolio values
            final_stock_price = price_path[-1]
            
            # Covered call portion (with assignment effects)
            assignment_rate = np.mean(assignments) if assignments else 0
            
            if assignment_rate > 0.5:  # Frequent assignments limit upside
                avg_assignment_price = self.nvii_price * 1.05  # Simplified
                cc_stock_value = covered_call_shares * avg_assignment_price
            else:
                cc_stock_value = covered_call_shares * final_stock_price
            
            cc_total_value = cc_stock_value + (total_option_income * covered_call_shares)
            
            # Unlimited portion
            ul_stock_value = unlimited_shares * final_stock_price
            
            # Total portfolio
            total_portfolio_value = cc_total_value + ul_stock_value
            portfolio_return = (total_portfolio_value - self.initial_capital) / self.initial_capital
            
            # Pure NVDA comparison
            pure_nvda_value = self.initial_capital * (final_stock_price / self.nvii_price) * self.leverage
            pure_nvda_return = (pure_nvda_value - self.initial_capital) / self.initial_capital
            
            results.append({
                'simulation': sim,
                'portfolio_return': portfolio_return,
                'pure_nvda_return': pure_nvda_return,
                'alpha': portfolio_return - pure_nvda_return,
                'option_income': total_option_income,
                'assignment_rate': assignment_rate,
                'final_stock_price': final_stock_price,
                'max_volatility': max([self.volatility_regimes[r]['vol'] for r in volatility_regimes]),
                'avg_volatility': np.mean([self.volatility_regimes[r]['vol'] for r in volatility_regimes]),
                'portfolio_value': total_portfolio_value,
                'covered_call_value': cc_total_value,
                'unlimited_value': ul_stock_value
            })
        
        return pd.DataFrame(results)
    
    def run_comprehensive_monte_carlo(self, num_simulations=10000):
        """Run comprehensive Monte Carlo analysis across all market scenarios"""
        print("="*80)
        print("NVII MONTE CARLO SIMULATION ANALYSIS")
        print("="*80)
        
        all_results = {}
        
        # Run simulations for each market scenario
        for scenario_name, scenario_data in self.market_scenarios.items():
            print(f"\nRunning {num_simulations:,} simulations for {scenario_name} scenario...")
            
            drift = scenario_data['drift']
            scenario_results = self.simulate_portfolio_performance(drift, num_simulations)
            scenario_results['scenario'] = scenario_name
            scenario_results['scenario_probability'] = scenario_data['probability']
            
            all_results[scenario_name] = scenario_results
        
        # Combine all results
        combined_results = pd.concat(all_results.values(), ignore_index=True)
        
        return all_results, combined_results
    
    def analyze_simulation_results(self, all_results, combined_results):
        """Analyze and present Monte Carlo simulation results"""
        print("\n" + "="*80)
        print("MONTE CARLO SIMULATION RESULTS")
        print("="*80)
        
        # Scenario-specific statistics
        print("\nScenario-Specific Results:")
        print("Scenario      | Mean Return | Std Dev | 5th %tile | 95th %tile | Alpha")
        print("-" * 70)
        
        scenario_stats = {}
        for scenario, results in all_results.items():
            mean_return = results['portfolio_return'].mean()
            std_return = results['portfolio_return'].std()
            p5 = results['portfolio_return'].quantile(0.05)
            p95 = results['portfolio_return'].quantile(0.95)
            mean_alpha = results['alpha'].mean()
            
            scenario_stats[scenario] = {
                'mean_return': mean_return,
                'std_return': std_return,
                'p5': p5,
                'p95': p95,
                'mean_alpha': mean_alpha
            }
            
            print(f"{scenario:13} | {mean_return:10.1%} | {std_return:7.1%} | "
                  f"{p5:8.1%} | {p95:9.1%} | {mean_alpha:+6.1%}")
        
        # Overall probability-weighted statistics
        print("\n" + "="*80)
        print("PROBABILITY-WEIGHTED PORTFOLIO METRICS")
        print("="*80)
        
        # Weight scenarios by probability
        weighted_returns = []
        for _, row in combined_results.iterrows():
            weighted_returns.extend([row['portfolio_return']] * int(row['scenario_probability'] * 1000))
        
        weighted_returns = np.array(weighted_returns)
        
        print(f"\nOverall Portfolio Statistics:")
        print(f"Expected Return:        {np.mean(weighted_returns):7.2%}")
        print(f"Volatility:             {np.std(weighted_returns):7.2%}")
        print(f"Sharpe Ratio:           {(np.mean(weighted_returns) - self.risk_free_rate) / np.std(weighted_returns):7.3f}")
        print(f"5th Percentile:         {np.percentile(weighted_returns, 5):7.1%}")
        print(f"25th Percentile:        {np.percentile(weighted_returns, 25):7.1%}")
        print(f"Median:                 {np.percentile(weighted_returns, 50):7.1%}")
        print(f"75th Percentile:        {np.percentile(weighted_returns, 75):7.1%}")
        print(f"95th Percentile:        {np.percentile(weighted_returns, 95):7.1%}")
        
        # Risk metrics
        var_95 = np.percentile(weighted_returns, 5)
        expected_shortfall = np.mean(weighted_returns[weighted_returns <= var_95])
        
        print(f"\nRisk Metrics:")
        print(f"Value at Risk (95%):    {var_95:7.1%}")
        print(f"Expected Shortfall:     {expected_shortfall:7.1%}")
        print(f"Maximum Drawdown Est:   {np.min(weighted_returns):7.1%}")
        
        # Option income analysis
        print("\n" + "="*80)
        print("OPTION INCOME ANALYSIS")
        print("="*80)
        
        for scenario, results in all_results.items():
            mean_income = results['option_income'].mean()
            annual_yield = (mean_income * 52) / self.nvii_price * 100
            assignment_rate = results['assignment_rate'].mean()
            
            print(f"{scenario:13} | Income: ${mean_income:5.3f}/week | "
                  f"Annual Yield: {annual_yield:5.1f}% | "
                  f"Assignment Rate: {assignment_rate:5.1%}")
        
        return scenario_stats, weighted_returns
    
    def generate_confidence_intervals(self, weighted_returns):
        """Generate confidence intervals for key metrics"""
        print("\n" + "="*80)
        print("CONFIDENCE INTERVALS (95%)")
        print("="*80)
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_means = []
        bootstrap_stds = []
        bootstrap_sharpes = []
        
        for i in range(n_bootstrap):
            bootstrap_sample = np.random.choice(weighted_returns, size=len(weighted_returns), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
            bootstrap_stds.append(np.std(bootstrap_sample))
            bootstrap_sharpes.append((np.mean(bootstrap_sample) - self.risk_free_rate) / np.std(bootstrap_sample))
        
        # Calculate confidence intervals
        mean_ci = np.percentile(bootstrap_means, [2.5, 97.5])
        std_ci = np.percentile(bootstrap_stds, [2.5, 97.5])
        sharpe_ci = np.percentile(bootstrap_sharpes, [2.5, 97.5])
        
        print(f"Expected Return:  [{mean_ci[0]:6.2%}, {mean_ci[1]:6.2%}]")
        print(f"Volatility:       [{std_ci[0]:6.2%}, {std_ci[1]:6.2%}]")
        print(f"Sharpe Ratio:     [{sharpe_ci[0]:6.3f}, {sharpe_ci[1]:6.3f}]")
        
        return {
            'mean_ci': mean_ci,
            'std_ci': std_ci,
            'sharpe_ci': sharpe_ci
        }
    
    def volatility_regime_analysis(self, combined_results):
        """Analyze performance by volatility regime exposure"""
        print("\n" + "="*80)
        print("VOLATILITY REGIME IMPACT ANALYSIS")
        print("="*80)
        
        # Correlate performance with average volatility
        correlation_vol_return = combined_results['avg_volatility'].corr(combined_results['portfolio_return'])
        correlation_vol_alpha = combined_results['avg_volatility'].corr(combined_results['alpha'])
        correlation_vol_income = combined_results['avg_volatility'].corr(combined_results['option_income'])
        
        print(f"Correlation: Volatility vs Portfolio Return: {correlation_vol_return:6.3f}")
        print(f"Correlation: Volatility vs Alpha:            {correlation_vol_alpha:6.3f}")
        print(f"Correlation: Volatility vs Option Income:    {correlation_vol_income:6.3f}")
        
        # Volatility buckets analysis
        vol_buckets = pd.cut(combined_results['avg_volatility'], 
                           bins=[0, 0.4, 0.6, 0.8, 1.2], 
                           labels=['Low', 'Medium', 'High', 'Extreme'])
        
        bucket_analysis = combined_results.groupby(vol_buckets).agg({
            'portfolio_return': ['mean', 'std'],
            'alpha': 'mean',
            'option_income': 'mean',
            'assignment_rate': 'mean'
        }).round(4)
        
        print(f"\nPerformance by Volatility Regime:")
        print(bucket_analysis)
        
        return correlation_vol_return, correlation_vol_alpha, bucket_analysis

def run_comprehensive_analysis():
    """Execute complete Monte Carlo analysis"""
    simulator = MonteCarloNVIISimulator()
    
    # Run simulations
    all_results, combined_results = simulator.run_comprehensive_monte_carlo(num_simulations=10000)
    
    # Analyze results
    scenario_stats, weighted_returns = simulator.analyze_simulation_results(all_results, combined_results)
    
    # Generate confidence intervals
    confidence_intervals = simulator.generate_confidence_intervals(weighted_returns)
    
    # Volatility regime analysis
    vol_correlations, vol_alpha_corr, bucket_analysis = simulator.volatility_regime_analysis(combined_results)
    
    print("\n" + "="*80)
    print("MONTE CARLO ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'scenario_stats': scenario_stats,
        'weighted_returns': weighted_returns,
        'confidence_intervals': confidence_intervals,
        'volatility_analysis': {
            'vol_return_corr': vol_correlations,
            'vol_alpha_corr': vol_alpha_corr,
            'bucket_analysis': bucket_analysis
        },
        'combined_results': combined_results
    }

if __name__ == "__main__":
    results = run_comprehensive_analysis()