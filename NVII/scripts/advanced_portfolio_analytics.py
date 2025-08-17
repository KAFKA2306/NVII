#!/usr/bin/env python3
"""
Advanced Portfolio Analytics for NVII Covered Call Strategy
Implements comprehensive risk management, performance attribution, and strategic analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t
from scipy.optimize import minimize_scalar, minimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedNVIIPortfolioAnalytics:
    """
    Advanced analytics for NVII covered call portfolio strategy
    Comprehensive risk management and performance attribution
    """
    
    def __init__(self, initial_capital=1_000_000, nvii_price=32.97, leverage=1.25):
        self.initial_capital = initial_capital
        self.nvii_price = nvii_price
        self.leverage = leverage
        self.risk_free_rate = 0.045
        
        # Portfolio allocation
        self.cc_allocation = 0.5
        self.ul_allocation = 0.5
        
        # Risk parameters
        self.confidence_levels = [0.95, 0.99]
        self.time_horizons = [1, 7, 30, 252]  # 1 day, 1 week, 1 month, 1 year
        
        # Market scenarios with enhanced probability distributions
        self.detailed_scenarios = self._initialize_detailed_scenarios()
        
    def _initialize_detailed_scenarios(self):
        """Initialize comprehensive market scenarios with full probability distributions"""
        return {
            'bull_strong': {
                'annual_return': 0.40, 'volatility': 0.50, 'probability': 0.15,
                'skewness': 0.5, 'kurtosis': 3.5, 'regime': 'expansion'
            },
            'bull_moderate': {
                'annual_return': 0.25, 'volatility': 0.45, 'probability': 0.20,
                'skewness': 0.3, 'kurtosis': 3.2, 'regime': 'growth'
            },
            'sideways_positive': {
                'annual_return': 0.08, 'volatility': 0.35, 'probability': 0.15,
                'skewness': 0.1, 'kurtosis': 3.0, 'regime': 'stable'
            },
            'sideways_flat': {
                'annual_return': 0.02, 'volatility': 0.30, 'probability': 0.15,
                'skewness': 0.0, 'kurtosis': 2.8, 'regime': 'stable'
            },
            'bear_moderate': {
                'annual_return': -0.15, 'volatility': 0.60, 'probability': 0.20,
                'skewness': -0.5, 'kurtosis': 4.0, 'regime': 'contraction'
            },
            'bear_severe': {
                'annual_return': -0.35, 'volatility': 0.80, 'probability': 0.10,
                'skewness': -0.8, 'kurtosis': 5.0, 'regime': 'crisis'
            },
            'crash': {
                'annual_return': -0.50, 'volatility': 1.20, 'probability': 0.05,
                'skewness': -1.2, 'kurtosis': 8.0, 'regime': 'crisis'
            }
        }
    
    def calculate_portfolio_shares(self):
        """Calculate exact share allocation for portfolio"""
        total_shares = self.initial_capital / self.nvii_price
        cc_shares = total_shares * self.cc_allocation
        ul_shares = total_shares * self.ul_allocation
        
        return {
            'total_shares': total_shares,
            'covered_call_shares': cc_shares,
            'unlimited_shares': ul_shares,
            'weekly_contracts': int(cc_shares / 100),  # Options contracts
            'cc_value': cc_shares * self.nvii_price,
            'ul_value': ul_shares * self.nvii_price
        }
    
    def black_scholes_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate comprehensive Black-Scholes Greeks"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = -(S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + 
                     r*K*np.exp(-r*T)*norm.cdf(d2))
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            theta = -(S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - 
                     r*K*np.exp(-r*T)*norm.cdf(-d2))
        
        gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.sqrt(T)
        rho = K*T*np.exp(-r*T)*norm.cdf(d2) if option_type == 'call' else \
              -K*T*np.exp(-r*T)*norm.cdf(-d2)
        
        return {
            'price': price, 'delta': delta, 'gamma': gamma,
            'theta': theta, 'vega': vega, 'rho': rho
        }
    
    def calculate_var_and_es(self, returns, confidence_level=0.95):
        """Calculate Value at Risk and Expected Shortfall"""
        if len(returns) == 0:
            return {'VaR': 0, 'ES': 0}
        
        sorted_returns = np.sort(returns)
        var_index = int((1 - confidence_level) * len(sorted_returns))
        
        if var_index == 0:
            var = sorted_returns[0]
            es = sorted_returns[0]
        else:
            var = -sorted_returns[var_index]
            es = -np.mean(sorted_returns[:var_index])
        
        return {'VaR': var, 'ES': es}
    
    def monte_carlo_portfolio_simulation(self, num_simulations=10000, time_horizon=1.0):
        """
        Comprehensive Monte Carlo simulation of portfolio performance
        """
        np.random.seed(42)  # For reproducibility
        
        results = {
            'portfolio_returns': [],
            'cc_returns': [],
            'ul_returns': [],
            'option_income': [],
            'assignments': [],
            'drawdowns': []
        }
        
        weeks_in_period = int(52 * time_horizon)
        portfolio_allocation = self.calculate_portfolio_shares()
        
        for sim in range(num_simulations):
            # Randomly select market scenario
            scenario_probs = [s['probability'] for s in self.detailed_scenarios.values()]
            scenario_names = list(self.detailed_scenarios.keys())
            scenario = np.random.choice(scenario_names, p=scenario_probs)
            scenario_data = self.detailed_scenarios[scenario]
            
            # Simulate price path with realistic dynamics
            price_path = self._simulate_realistic_price_path(
                scenario_data, weeks_in_period
            )
            
            # Calculate covered call performance
            cc_result = self._simulate_covered_call_performance(
                price_path, scenario_data, portfolio_allocation
            )
            
            # Calculate unlimited performance
            ul_result = self._simulate_unlimited_performance(
                price_path, portfolio_allocation
            )
            
            # Portfolio totals
            total_return = (cc_result['final_value'] + ul_result['final_value'] - 
                          self.initial_capital) / self.initial_capital
            
            # Store results
            results['portfolio_returns'].append(total_return)
            results['cc_returns'].append(cc_result['return'])
            results['ul_returns'].append(ul_result['return'])
            results['option_income'].append(cc_result['option_income'])
            results['assignments'].append(cc_result['assignments'])
            results['drawdowns'].append(cc_result.get('max_drawdown', 0))
        
        return self._process_simulation_results(results)
    
    def _simulate_realistic_price_path(self, scenario_data, num_weeks):
        """Simulate realistic price path with volatility clustering and jumps"""
        dt = 1/52  # Weekly intervals
        annual_return = scenario_data['annual_return']
        base_vol = scenario_data['volatility']
        
        # Initialize arrays
        prices = [self.nvii_price]
        volatilities = [base_vol]
        
        for week in range(num_weeks):
            # Volatility clustering (GARCH-like behavior)
            vol_persistence = 0.9
            vol_innovation = 0.1 * np.random.normal(0, 0.1)
            current_vol = vol_persistence * volatilities[-1] + vol_innovation
            current_vol = np.clip(current_vol, 0.15, 1.5)  # Reasonable bounds
            
            # Price evolution with potential jumps
            drift = annual_return * dt
            diffusion = current_vol * np.sqrt(dt) * np.random.normal()
            
            # Add jump component (rare large moves)
            jump_prob = 0.02  # 2% chance per week
            if np.random.random() < jump_prob:
                jump_size = np.random.normal(0, 0.15)  # ±15% jump
                diffusion += jump_size
            
            # Update price
            new_price = prices[-1] * np.exp(drift + diffusion)
            prices.append(new_price)
            volatilities.append(current_vol)
        
        return np.array(prices)
    
    def _simulate_covered_call_performance(self, price_path, scenario_data, allocation):
        """Simulate covered call portion with realistic option dynamics"""
        cc_shares = allocation['covered_call_shares']
        initial_value = cc_shares * self.nvii_price
        total_option_income = 0
        assignments = 0
        current_shares = cc_shares
        cash = 0
        
        portfolio_values = [initial_value]
        
        for week in range(1, len(price_path)):
            current_price = price_path[week]
            previous_price = price_path[week-1]
            
            # Weekly option strategy
            strike_price = previous_price * 1.05  # 5% OTM
            time_to_expiry = 7/365
            
            # Calculate option premium using current volatility
            current_vol = scenario_data['volatility'] * (1 + 0.2 * np.random.normal())
            current_vol = np.clip(current_vol, 0.2, 1.2)
            
            greeks = self.black_scholes_greeks(
                previous_price, strike_price, time_to_expiry,
                self.risk_free_rate, current_vol
            )
            
            # Option premium with leverage adjustment
            weekly_premium = greeks['price'] * self.leverage
            contracts_sold = int(current_shares / 100)
            total_premium = weekly_premium * contracts_sold * 100
            total_option_income += total_premium
            cash += total_premium
            
            # Check for assignment
            if current_price > strike_price:
                assignments += 1
                # Shares called away at strike price
                assignment_proceeds = strike_price * current_shares
                cash += assignment_proceeds
                current_shares = 0
                
                # Reinvest in new shares immediately
                current_shares = cash / current_price
                cash = 0
            
            # Calculate current portfolio value
            portfolio_value = current_shares * current_price + cash
            portfolio_values.append(portfolio_value)
        
        final_value = portfolio_values[-1]
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        return {
            'final_value': final_value,
            'return': (final_value - initial_value) / initial_value,
            'option_income': total_option_income,
            'assignments': assignments,
            'max_drawdown': max_drawdown
        }
    
    def _simulate_unlimited_performance(self, price_path, allocation):
        """Simulate unlimited upside portion"""
        ul_shares = allocation['unlimited_shares']
        initial_value = ul_shares * self.nvii_price
        
        # Simple buy and hold with leverage
        final_price = price_path[-1]
        leverage_adjusted_return = ((final_price / self.nvii_price) - 1) * self.leverage
        final_value = initial_value * (1 + leverage_adjusted_return)
        
        return {
            'final_value': final_value,
            'return': (final_value - initial_value) / initial_value
        }
    
    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown from value series"""
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _process_simulation_results(self, results):
        """Process and analyze simulation results"""
        portfolio_returns = np.array(results['portfolio_returns'])
        
        # Basic statistics
        expected_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        
        # Risk metrics
        var_95 = self.calculate_var_and_es(portfolio_returns, 0.95)
        var_99 = self.calculate_var_and_es(portfolio_returns, 0.99)
        
        # Percentiles
        percentiles = np.percentile(portfolio_returns, [5, 10, 25, 50, 75, 90, 95])
        
        # Option income statistics
        option_income = np.array(results['option_income'])
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95['VaR'],
            'es_95': var_95['ES'],
            'var_99': var_99['VaR'],
            'es_99': var_99['ES'],
            'percentiles': {
                '5th': percentiles[0], '10th': percentiles[1],
                '25th': percentiles[2], '50th': percentiles[3],
                '75th': percentiles[4], '90th': percentiles[5],
                '95th': percentiles[6]
            },
            'option_income_stats': {
                'mean': np.mean(option_income),
                'std': np.std(option_income),
                'min': np.min(option_income),
                'max': np.max(option_income)
            },
            'average_assignments': np.mean(results['assignments']),
            'max_drawdown': np.mean(results['drawdowns'])
        }
    
    def dynamic_allocation_optimization(self, market_conditions):
        """
        Optimize allocation based on current market conditions
        """
        def objective_function(cc_allocation):
            """Objective function to maximize risk-adjusted returns"""
            # Simulate portfolio with this allocation
            temp_cc_alloc = cc_allocation[0]
            temp_ul_alloc = 1 - temp_cc_alloc
            
            # Expected return calculation
            cc_expected_return = self._estimate_cc_return(market_conditions)
            ul_expected_return = self._estimate_ul_return(market_conditions)
            
            portfolio_return = (temp_cc_alloc * cc_expected_return + 
                              temp_ul_alloc * ul_expected_return)
            
            # Risk calculation (simplified)
            cc_vol = market_conditions.get('cc_volatility', 0.12)
            ul_vol = market_conditions.get('ul_volatility', 0.45)
            correlation = market_conditions.get('correlation', -0.6)
            
            portfolio_vol = np.sqrt(
                temp_cc_alloc**2 * cc_vol**2 +
                temp_ul_alloc**2 * ul_vol**2 +
                2 * temp_cc_alloc * temp_ul_alloc * correlation * cc_vol * ul_vol
            )
            
            # Sharpe ratio (negative for minimization)
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe
        
        # Optimize allocation
        result = minimize_scalar(
            objective_function,
            bounds=(0.3, 0.7),
            method='bounded'
        )
        
        optimal_cc_allocation = result.x
        optimal_ul_allocation = 1 - optimal_cc_allocation
        
        return {
            'optimal_cc_allocation': optimal_cc_allocation,
            'optimal_ul_allocation': optimal_ul_allocation,
            'expected_sharpe': -result.fun,
            'current_vs_optimal': {
                'current_cc': self.cc_allocation,
                'optimal_cc': optimal_cc_allocation,
                'difference': optimal_cc_allocation - self.cc_allocation
            }
        }
    
    def _estimate_cc_return(self, market_conditions):
        """Estimate covered call return based on market conditions"""
        base_yield = 0.063  # NVII base yield
        option_yield = market_conditions.get('implied_volatility', 0.55) * 0.4
        return base_yield + option_yield * 0.5  # 50% allocation
    
    def _estimate_ul_return(self, market_conditions):
        """Estimate unlimited return based on market conditions"""
        expected_stock_return = market_conditions.get('expected_return', 0.15)
        return expected_stock_return * self.leverage * 0.5  # 50% allocation
    
    def comprehensive_stress_testing(self):
        """
        Comprehensive stress testing across multiple scenarios
        """
        stress_scenarios = {
            'tech_bubble_burst': {
                'nvda_return': -0.70, 'volatility_spike': 2.0,
                'duration_months': 18, 'recovery_probability': 0.8
            },
            'ai_winter': {
                'nvda_return': -0.45, 'volatility_spike': 1.5,
                'duration_months': 12, 'recovery_probability': 0.9
            },
            'market_crash_2008': {
                'nvda_return': -0.55, 'volatility_spike': 1.8,
                'duration_months': 20, 'recovery_probability': 0.95
            },
            'flash_crash': {
                'nvda_return': -0.25, 'volatility_spike': 3.0,
                'duration_months': 1, 'recovery_probability': 0.99
            },
            'regulatory_crackdown': {
                'nvda_return': -0.35, 'volatility_spike': 1.2,
                'duration_months': 24, 'recovery_probability': 0.7
            },
            'hyperinflation': {
                'nvda_return': -0.20, 'volatility_spike': 1.4,
                'duration_months': 36, 'recovery_probability': 0.6
            }
        }
        
        stress_results = {}
        
        for scenario_name, scenario_data in stress_scenarios.items():
            # Calculate portfolio impact
            nvda_return = scenario_data['nvda_return']
            vol_multiplier = scenario_data['volatility_spike']
            
            # Covered call portion
            cc_stock_loss = nvda_return * 0.5 * self.leverage
            enhanced_vol = 0.55 * vol_multiplier  # Base vol * multiplier
            option_income_boost = min(enhanced_vol * 0.6, 1.5)  # Cap at 150%
            cc_total_return = cc_stock_loss + option_income_boost * 0.5
            
            # Unlimited portion
            ul_return = nvda_return * self.leverage * 0.5
            
            # Total portfolio
            portfolio_return = cc_total_return + ul_return
            
            # Recovery analysis
            recovery_time = scenario_data['duration_months']
            recovery_prob = scenario_data['recovery_probability']
            
            stress_results[scenario_name] = {
                'portfolio_return': portfolio_return,
                'nvda_return': nvda_return,
                'alpha': portfolio_return - nvda_return,
                'downside_protection': abs(cc_total_return / cc_stock_loss) if cc_stock_loss < 0 else 0,
                'recovery_time_months': recovery_time,
                'recovery_probability': recovery_prob,
                'option_income_boost': option_income_boost
            }
        
        return stress_results
    
    def performance_attribution_analysis(self):
        """
        Detailed performance attribution across all return sources
        """
        # Base components
        base_dividend = 0.063
        leverage_enhancement = (self.leverage - 1.0) * base_dividend
        
        # Market scenario weighted returns
        scenario_weights = [s['probability'] for s in self.detailed_scenarios.values()]
        scenario_returns = []
        
        for scenario_name, scenario_data in self.detailed_scenarios.items():
            # Option income component
            vol = scenario_data['volatility']
            option_yield = vol * 0.4082  # From theoretical analysis
            option_contribution = option_yield * self.cc_allocation
            
            # Capital appreciation component
            stock_return = scenario_data['annual_return']
            appreciation_contribution = stock_return * self.leverage * self.ul_allocation
            
            scenario_return = (base_dividend + leverage_enhancement + 
                             option_contribution + appreciation_contribution)
            scenario_returns.append(scenario_return)
        
        # Weighted average
        expected_return = np.average(scenario_returns, weights=scenario_weights)
        
        # Component attribution
        avg_option_contribution = np.average([
            s['volatility'] * 0.4082 * self.cc_allocation 
            for s in self.detailed_scenarios.values()
        ], weights=scenario_weights)
        
        avg_appreciation_contribution = np.average([
            s['annual_return'] * self.leverage * self.ul_allocation
            for s in self.detailed_scenarios.values()
        ], weights=scenario_weights)
        
        attribution = {
            'base_dividend': {
                'contribution': base_dividend,
                'percentage': base_dividend / expected_return * 100
            },
            'leverage_enhancement': {
                'contribution': leverage_enhancement,
                'percentage': leverage_enhancement / expected_return * 100
            },
            'option_income': {
                'contribution': avg_option_contribution,
                'percentage': avg_option_contribution / expected_return * 100
            },
            'capital_appreciation': {
                'contribution': avg_appreciation_contribution,
                'percentage': avg_appreciation_contribution / expected_return * 100
            },
            'total_expected_return': expected_return
        }
        
        return attribution
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analytics report"""
        print("=" * 80)
        print("NVII ADVANCED PORTFOLIO ANALYTICS REPORT")
        print("=" * 80)
        
        # Portfolio setup
        allocation = self.calculate_portfolio_shares()
        print(f"\nPortfolio Configuration:")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"NVII Price: ${self.nvii_price:.2f}")
        print(f"Leverage: {self.leverage:.2f}x")
        print(f"Total Shares: {allocation['total_shares']:.0f}")
        print(f"Covered Call Shares: {allocation['covered_call_shares']:.0f}")
        print(f"Unlimited Shares: {allocation['unlimited_shares']:.0f}")
        print(f"Weekly Contracts: {allocation['weekly_contracts']}")
        
        # Monte Carlo simulation
        print(f"\n{'='*20} MONTE CARLO SIMULATION RESULTS {'='*20}")
        simulation_results = self.monte_carlo_portfolio_simulation()
        
        print(f"Expected Annual Return: {simulation_results['expected_return']:8.2%}")
        print(f"Portfolio Volatility:   {simulation_results['volatility']:8.2%}")
        print(f"Sharpe Ratio:          {simulation_results['sharpe_ratio']:8.3f}")
        print(f"95% VaR:               {simulation_results['var_95']:8.2%}")
        print(f"95% Expected Shortfall: {simulation_results['es_95']:8.2%}")
        print(f"99% VaR:               {simulation_results['var_99']:8.2%}")
        print(f"Maximum Drawdown:      {simulation_results['max_drawdown']:8.2%}")
        
        print(f"\nReturn Distribution Percentiles:")
        percentiles = simulation_results['percentiles']
        print(f"5th:  {percentiles['5th']:7.2%}")
        print(f"25th: {percentiles['25th']:7.2%}")
        print(f"50th: {percentiles['50th']:7.2%}")
        print(f"75th: {percentiles['75th']:7.2%}")
        print(f"95th: {percentiles['95th']:7.2%}")
        
        # Performance attribution
        print(f"\n{'='*20} PERFORMANCE ATTRIBUTION ANALYSIS {'='*20}")
        attribution = self.performance_attribution_analysis()
        
        print(f"Component Attribution:")
        for component, data in attribution.items():
            if component != 'total_expected_return':
                print(f"{component.replace('_', ' ').title():20}: {data['contribution']:7.2%} ({data['percentage']:5.1f}%)")
        print(f"{'Total Expected Return':20}: {attribution['total_expected_return']:7.2%} (100.0%)")
        
        # Stress testing
        print(f"\n{'='*20} COMPREHENSIVE STRESS TESTING {'='*20}")
        stress_results = self.comprehensive_stress_testing()
        
        print(f"Scenario              | Portfolio | NVDA    | Alpha   | Protection | Recovery")
        print(f"-" * 70)
        for scenario, data in stress_results.items():
            print(f"{scenario[:20]:20} | {data['portfolio_return']:8.1%} | "
                  f"{data['nvda_return']:7.1%} | {data['alpha']:+7.1%} | "
                  f"{data['downside_protection']:9.1f}x | {data['recovery_time_months']:2.0f} months")
        
        # Dynamic allocation
        print(f"\n{'='*20} DYNAMIC ALLOCATION OPTIMIZATION {'='*20}")
        market_conditions = {
            'implied_volatility': 0.55,
            'expected_return': 0.15,
            'cc_volatility': 0.12,
            'ul_volatility': 0.45,
            'correlation': -0.6
        }
        
        allocation_opt = self.dynamic_allocation_optimization(market_conditions)
        print(f"Current Allocation - CC: {self.cc_allocation:.1%}, UL: {self.ul_allocation:.1%}")
        print(f"Optimal Allocation - CC: {allocation_opt['optimal_cc_allocation']:.1%}, "
              f"UL: {allocation_opt['optimal_ul_allocation']:.1%}")
        print(f"Expected Sharpe Ratio: {allocation_opt['expected_sharpe']:.3f}")
        print(f"Allocation Adjustment: {allocation_opt['current_vs_optimal']['difference']:+.1%}")
        
        return {
            'simulation_results': simulation_results,
            'attribution': attribution,
            'stress_tests': stress_results,
            'allocation_optimization': allocation_opt
        }

def run_advanced_analytics():
    """Execute comprehensive advanced analytics"""
    analyzer = AdvancedNVIIPortfolioAnalytics()
    results = analyzer.generate_comprehensive_report()
    return results

if __name__ == "__main__":
    results = run_advanced_analytics()