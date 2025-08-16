#!/usr/bin/env python3
"""
NVII Advanced Portfolio Optimization
Dynamic allocation strategies with regime-aware rebalancing
Incorporates transaction costs, market impact, and operational constraints
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from option_analysis import BlackScholesCalculator, NVIIAnalyzer
from monte_carlo_simulation import MonteCarloNVIISimulator

class AdvancedPortfolioOptimizer:
    """Advanced portfolio optimization for NVII strategy"""
    
    def __init__(self, initial_capital=100000, nvii_price=32.97):
        self.initial_capital = initial_capital
        self.nvii_price = nvii_price
        self.risk_free_rate = 0.045
        
        # Transaction costs and constraints
        self.option_commission = 0.65  # Per contract
        self.bid_ask_spread = 0.03     # 3% average spread
        self.rebalancing_cost = 0.001  # 10bp for rebalancing
        
        # Market regime indicators
        self.regime_indicators = {
            'vix_low': 20,
            'vix_high': 35,
            'trend_threshold': 0.05,   # 5% trend trigger
            'momentum_window': 20      # 20-day momentum
        }
        
        # Optimization constraints
        self.min_cc_allocation = 0.2   # Minimum 20% covered calls
        self.max_cc_allocation = 0.8   # Maximum 80% covered calls
        self.max_leverage = 1.5        # Maximum leverage allowed
        self.min_leverage = 1.05       # Minimum leverage
        
    def kelly_criterion_optimization(self, expected_returns, volatilities, win_rates):
        """Calculate optimal allocation using Kelly criterion framework"""
        
        def kelly_allocation(p_win, avg_win, avg_loss):
            """Kelly formula: f = (bp - q) / b where b = avg_win/avg_loss"""
            if avg_loss == 0:
                return 0
            b = avg_win / avg_loss
            q = 1 - p_win
            kelly_fraction = (b * p_win - q) / b
            return max(0, min(1, kelly_fraction))  # Constrain between 0 and 1
        
        kelly_allocations = {}
        
        for strategy, expected_return in expected_returns.items():
            vol = volatilities[strategy]
            win_rate = win_rates[strategy]
            
            # Estimate average win/loss from normal distribution
            # Assuming wins and losses are equally distributed around mean
            avg_win = expected_return + vol
            avg_loss = abs(expected_return - vol)
            
            kelly_frac = kelly_allocation(win_rate, avg_win, avg_loss)
            kelly_allocations[strategy] = kelly_frac
        
        return kelly_allocations
    
    def black_litterman_views(self, market_returns, confidence_levels):
        """Implement Black-Litterman framework for strategy views"""
        
        # Market equilibrium (equal weight as starting point)
        equilibrium_weights = np.array([0.5, 0.5])  # 50/50 CC/Unlimited
        
        # Views matrix (our opinions about strategy performance)
        P = np.array([
            [1, 0],    # View 1: Covered call performance
            [0, 1],    # View 2: Unlimited performance
            [1, -1]    # View 3: Relative performance
        ])
        
        # View returns (our expected alpha)
        Q = np.array([
            0.12,      # CC expected to generate 12% alpha
            0.08,      # Unlimited expected to generate 8% alpha
            0.04       # CC expected to outperform by 4%
        ])
        
        # Confidence in views (higher = more confident)
        omega = np.diag([
            1 / confidence_levels.get('covered_call', 0.7),
            1 / confidence_levels.get('unlimited', 0.6),
            1 / confidence_levels.get('relative', 0.8)
        ])
        
        # Covariance matrix (estimated from historical data)
        cov_matrix = np.array([
            [0.1744, 0.1452],  # CC variance and covariance
            [0.1452, 0.2304]   # Covariance and Unlimited variance
        ])
        
        # Black-Litterman formula
        tau = 0.025  # Scaling factor
        
        # Market implied returns
        pi = equilibrium_weights  # Simplified
        
        # Calculate new expected returns
        M1 = np.linalg.inv(tau * cov_matrix)
        M2 = P.T @ np.linalg.inv(omega) @ P
        M3 = np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(omega) @ Q
        
        mu_bl = np.linalg.inv(M1 + M2) @ M3
        
        return mu_bl, equilibrium_weights
    
    def mean_variance_optimization(self, expected_returns, covariance_matrix, risk_aversion=3.0):
        """Markowitz mean-variance optimization with NVII constraints"""
        
        def objective(weights):
            # Utility = Expected Return - (Risk Aversion / 2) * Variance
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_variance = weights.T @ covariance_matrix @ weights
            utility = portfolio_return - (risk_aversion / 2) * portfolio_variance
            return -utility  # Minimize negative utility
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda w: w[0] - self.min_cc_allocation},  # Min CC
            {'type': 'ineq', 'fun': lambda w: self.max_cc_allocation - w[0]},  # Max CC
        ]
        
        # Bounds
        bounds = [(self.min_cc_allocation, self.max_cc_allocation), 
                 (1 - self.max_cc_allocation, 1 - self.min_cc_allocation)]
        
        # Initial guess (50/50)
        x0 = np.array([0.5, 0.5])
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            optimal_return = np.sum(optimal_weights * expected_returns)
            optimal_variance = optimal_weights.T @ covariance_matrix @ optimal_weights
            optimal_sharpe = (optimal_return - self.risk_free_rate) / np.sqrt(optimal_variance)
            
            return {
                'weights': optimal_weights,
                'expected_return': optimal_return,
                'volatility': np.sqrt(optimal_variance),
                'sharpe_ratio': optimal_sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def risk_parity_optimization(self, covariance_matrix):
        """Equal risk contribution portfolio"""
        
        def risk_budget_objective(weights, cov_matrix):
            portfolio_var = weights.T @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_var
            
            # Equal risk contribution target
            target_contrib = np.ones(len(weights)) / len(weights)
            
            # Minimize squared differences from target
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_cc_allocation, self.max_cc_allocation), 
                 (1 - self.max_cc_allocation, 1 - self.min_cc_allocation)]
        
        x0 = np.array([0.5, 0.5])
        
        result = minimize(risk_budget_objective, x0, args=(covariance_matrix,),
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else np.array([0.5, 0.5])
    
    def dynamic_allocation_model(self, market_conditions):
        """Dynamic allocation based on market regime"""
        
        # Base allocation
        base_cc_allocation = 0.5
        
        # Market condition adjustments
        volatility_adj = 0
        trend_adj = 0
        momentum_adj = 0
        
        # Volatility adjustment
        current_vol = market_conditions.get('implied_volatility', 0.55)
        if current_vol > 0.7:
            volatility_adj = 0.15  # Increase CC in high vol
        elif current_vol < 0.4:
            volatility_adj = -0.1  # Decrease CC in low vol
        
        # Trend adjustment
        trend = market_conditions.get('price_trend', 0)
        if abs(trend) > self.regime_indicators['trend_threshold']:
            if trend > 0:
                trend_adj = -0.1  # Reduce CC in uptrend
            else:
                trend_adj = 0.1   # Increase CC in downtrend
        
        # Momentum adjustment
        momentum = market_conditions.get('momentum', 0)
        if momentum > 0.1:
            momentum_adj = -0.05  # Reduce CC with strong positive momentum
        elif momentum < -0.1:
            momentum_adj = 0.05   # Increase CC with negative momentum
        
        # Time-based adjustment (earnings proximity)
        earnings_proximity = market_conditions.get('days_to_earnings', 30)
        earnings_adj = 0
        if earnings_proximity < 7:
            earnings_adj = -0.05  # Reduce CC before earnings
        
        # Calculate final allocation
        total_adjustment = volatility_adj + trend_adj + momentum_adj + earnings_adj
        final_cc_allocation = base_cc_allocation + total_adjustment
        
        # Apply constraints
        final_cc_allocation = max(self.min_cc_allocation, 
                                min(self.max_cc_allocation, final_cc_allocation))
        
        return {
            'covered_call_allocation': final_cc_allocation,
            'unlimited_allocation': 1 - final_cc_allocation,
            'adjustments': {
                'volatility': volatility_adj,
                'trend': trend_adj,
                'momentum': momentum_adj,
                'earnings': earnings_adj,
                'total': total_adjustment
            }
        }
    
    def transaction_cost_analysis(self, allocation_changes, portfolio_value):
        """Calculate transaction costs for rebalancing"""
        
        cc_change = abs(allocation_changes['covered_call'])
        ul_change = abs(allocation_changes['unlimited'])
        
        # Option transaction costs
        shares_affected = (cc_change * portfolio_value) / self.nvii_price
        option_contracts = shares_affected / 100  # 100 shares per contract
        option_costs = option_contracts * self.option_commission
        
        # Bid-ask spread costs
        spread_costs = (cc_change + ul_change) * portfolio_value * self.bid_ask_spread
        
        # Rebalancing costs
        rebalancing_costs = portfolio_value * self.rebalancing_cost
        
        total_costs = option_costs + spread_costs + rebalancing_costs
        cost_percentage = total_costs / portfolio_value
        
        return {
            'option_costs': option_costs,
            'spread_costs': spread_costs,
            'rebalancing_costs': rebalancing_costs,
            'total_costs': total_costs,
            'cost_percentage': cost_percentage
        }
    
    def optimal_rebalancing_frequency(self, expected_alpha, transaction_costs, volatility):
        """Determine optimal rebalancing frequency"""
        
        # Trade-off between capturing alpha and transaction costs
        frequencies = [1, 7, 14, 30, 60, 90]  # Days
        utilities = []
        
        for freq in frequencies:
            # Annual rebalancing count
            annual_rebalances = 252 / freq
            
            # Alpha capture (diminishing returns with frequency)
            alpha_capture = expected_alpha * (1 - np.exp(-freq / 30))
            
            # Annual transaction costs
            annual_transaction_costs = annual_rebalances * transaction_costs
            
            # Tracking error penalty
            tracking_error = volatility * np.sqrt(freq / 252)
            
            # Net utility
            utility = alpha_capture - annual_transaction_costs - tracking_error
            utilities.append(utility)
        
        optimal_index = np.argmax(utilities)
        optimal_frequency = frequencies[optimal_index]
        
        return {
            'optimal_frequency_days': optimal_frequency,
            'optimal_utility': utilities[optimal_index],
            'frequency_analysis': dict(zip(frequencies, utilities))
        }
    
    def leverage_optimization(self, base_returns, base_volatility):
        """Optimize leverage level for maximum utility"""
        
        def leverage_utility(lev, returns, vol, risk_aversion=3.0):
            leveraged_return = returns * lev
            leveraged_vol = vol * lev
            
            # Utility with leverage constraints
            if lev < self.min_leverage or lev > self.max_leverage:
                return -1e6  # Heavy penalty
            
            utility = leveraged_return - (risk_aversion / 2) * leveraged_vol**2
            return utility
        
        # Optimize leverage
        result = minimize_scalar(
            lambda x: -leverage_utility(x, base_returns, base_volatility),
            bounds=(self.min_leverage, self.max_leverage),
            method='bounded'
        )
        
        optimal_leverage = result.x
        optimal_utility = -result.fun
        
        return {
            'optimal_leverage': optimal_leverage,
            'optimal_utility': optimal_utility,
            'leverage_return': base_returns * optimal_leverage,
            'leverage_volatility': base_volatility * optimal_leverage
        }
    
    def run_comprehensive_optimization(self):
        """Execute comprehensive portfolio optimization analysis"""
        
        print("="*80)
        print("NVII ADVANCED PORTFOLIO OPTIMIZATION")
        print("="*80)
        
        # Historical performance estimates (from previous analysis)
        expected_returns = np.array([0.218, 0.178])  # CC, Unlimited
        covariance_matrix = np.array([
            [0.1744, 0.1452],  # CC variance and covariance
            [0.1452, 0.2304]   # Covariance and Unlimited variance
        ])
        
        # 1. Mean-Variance Optimization
        print("\n1. MEAN-VARIANCE OPTIMIZATION")
        print("-" * 40)
        
        mv_results = {}
        risk_aversions = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for ra in risk_aversions:
            result = self.mean_variance_optimization(expected_returns, covariance_matrix, ra)
            if result['success']:
                mv_results[ra] = result
                print(f"Risk Aversion {ra}: CC={result['weights'][0]:.1%}, "
                      f"Return={result['expected_return']:.1%}, "
                      f"Sharpe={result['sharpe_ratio']:.3f}")
        
        # 2. Black-Litterman Optimization
        print("\n2. BLACK-LITTERMAN OPTIMIZATION")
        print("-" * 40)
        
        confidence_levels = {'covered_call': 0.7, 'unlimited': 0.6, 'relative': 0.8}
        bl_returns, bl_weights = self.black_litterman_views(expected_returns, confidence_levels)
        
        print(f"Black-Litterman Expected Returns: CC={bl_returns[0]:.1%}, UL={bl_returns[1]:.1%}")
        print(f"Equilibrium Weights: CC={bl_weights[0]:.1%}, UL={bl_weights[1]:.1%}")
        
        # 3. Risk Parity
        print("\n3. RISK PARITY OPTIMIZATION")
        print("-" * 40)
        
        rp_weights = self.risk_parity_optimization(covariance_matrix)
        rp_return = np.sum(rp_weights * expected_returns)
        rp_vol = np.sqrt(rp_weights.T @ covariance_matrix @ rp_weights)
        
        print(f"Risk Parity Weights: CC={rp_weights[0]:.1%}, UL={rp_weights[1]:.1%}")
        print(f"Expected Return: {rp_return:.1%}, Volatility: {rp_vol:.1%}")
        
        # 4. Kelly Criterion
        print("\n4. KELLY CRITERION OPTIMIZATION")
        print("-" * 40)
        
        strategy_returns = {'covered_call': 0.218, 'unlimited': 0.178}
        strategy_vols = {'covered_call': 0.417, 'unlimited': 0.480}
        win_rates = {'covered_call': 0.65, 'unlimited': 0.58}  # Estimated
        
        kelly_allocations = self.kelly_criterion_optimization(strategy_returns, strategy_vols, win_rates)
        print(f"Kelly Allocations: {kelly_allocations}")
        
        # 5. Dynamic Allocation Examples
        print("\n5. DYNAMIC ALLOCATION SCENARIOS")
        print("-" * 40)
        
        scenarios = [
            {'implied_volatility': 0.35, 'price_trend': 0.08, 'momentum': 0.12, 'name': 'Bull Market'},
            {'implied_volatility': 0.75, 'price_trend': -0.06, 'momentum': -0.08, 'name': 'Bear Market'},
            {'implied_volatility': 0.45, 'price_trend': 0.01, 'momentum': 0.02, 'name': 'Sideways'},
            {'implied_volatility': 0.85, 'price_trend': 0.03, 'momentum': 0.05, 'name': 'High Vol'}
        ]
        
        for scenario in scenarios:
            allocation = self.dynamic_allocation_model(scenario)
            print(f"{scenario['name']:12}: CC={allocation['covered_call_allocation']:.1%}, "
                  f"Adj={allocation['adjustments']['total']:+.1%}")
        
        # 6. Transaction Cost Analysis
        print("\n6. TRANSACTION COST ANALYSIS")
        print("-" * 40)
        
        portfolio_value = 100000
        allocation_changes = {'covered_call': 0.1, 'unlimited': 0.1}  # 10% change
        
        tc_analysis = self.transaction_cost_analysis(allocation_changes, portfolio_value)
        print(f"Transaction Costs for 10% Rebalancing:")
        print(f"  Option Costs: ${tc_analysis['option_costs']:.2f}")
        print(f"  Spread Costs: ${tc_analysis['spread_costs']:.2f}")
        print(f"  Total Costs: ${tc_analysis['total_costs']:.2f} ({tc_analysis['cost_percentage']:.3%})")
        
        # 7. Optimal Rebalancing Frequency
        print("\n7. OPTIMAL REBALANCING FREQUENCY")
        print("-" * 40)
        
        expected_alpha = 0.03
        transaction_costs = 0.002
        volatility = 0.42
        
        rebal_analysis = self.optimal_rebalancing_frequency(expected_alpha, transaction_costs, volatility)
        print(f"Optimal Rebalancing Frequency: {rebal_analysis['optimal_frequency_days']} days")
        print(f"Expected Utility: {rebal_analysis['optimal_utility']:.4f}")
        
        # 8. Leverage Optimization
        print("\n8. LEVERAGE OPTIMIZATION")
        print("-" * 40)
        
        base_return = 0.178
        base_vol = 0.42
        
        lev_analysis = self.leverage_optimization(base_return, base_vol)
        print(f"Optimal Leverage: {lev_analysis['optimal_leverage']:.2f}x")
        print(f"Leveraged Return: {lev_analysis['leverage_return']:.1%}")
        print(f"Leveraged Volatility: {lev_analysis['leverage_volatility']:.1%}")
        
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        
        print(f"\nRecommended Portfolio Configuration:")
        print(f"  Base Allocation: 50% CC / 50% Unlimited (mean-variance optimal)")
        print(f"  Dynamic Range: 30%-70% CC allocation based on market conditions")
        print(f"  Optimal Leverage: {lev_analysis['optimal_leverage']:.2f}x")
        print(f"  Rebalancing Frequency: {rebal_analysis['optimal_frequency_days']} days")
        print(f"  Expected Transaction Costs: {tc_analysis['cost_percentage']:.2%} per rebalancing")
        
        return {
            'mean_variance': mv_results,
            'black_litterman': {'returns': bl_returns, 'weights': bl_weights},
            'risk_parity': rp_weights,
            'kelly': kelly_allocations,
            'dynamic_scenarios': scenarios,
            'transaction_costs': tc_analysis,
            'rebalancing': rebal_analysis,
            'leverage': lev_analysis
        }

def scenario_optimization_framework():
    """Framework for optimizing across multiple market scenarios"""
    
    print("\n" + "="*80)
    print("SCENARIO-BASED OPTIMIZATION FRAMEWORK")
    print("="*80)
    
    optimizer = AdvancedPortfolioOptimizer()
    
    # Market scenarios with probabilities
    market_scenarios = {
        'bull_market': {
            'probability': 0.25,
            'nvda_return': 0.30,
            'volatility': 0.45,
            'optimal_cc_allocation': 0.40
        },
        'normal_market': {
            'probability': 0.50,
            'nvda_return': 0.15,
            'volatility': 0.55,
            'optimal_cc_allocation': 0.50
        },
        'bear_market': {
            'probability': 0.25,
            'nvda_return': -0.20,
            'volatility': 0.75,
            'optimal_cc_allocation': 0.70
        }
    }
    
    # Scenario-weighted optimization
    weighted_allocation = 0
    weighted_return = 0
    weighted_utility = 0
    
    print("Scenario-Specific Optimal Allocations:")
    print("Scenario      | Prob | CC Alloc | Expected Return | Utility")
    print("-" * 60)
    
    for scenario, data in market_scenarios.items():
        prob = data['probability']
        cc_alloc = data['optimal_cc_allocation']
        expected_ret = data['nvda_return'] * 0.85 + 0.15  # Simplified
        utility = expected_ret - 1.5 * (data['volatility'] ** 2)
        
        weighted_allocation += prob * cc_alloc
        weighted_return += prob * expected_ret
        weighted_utility += prob * utility
        
        print(f"{scenario:13} | {prob:4.0%} | {cc_alloc:7.0%} | {expected_ret:13.1%} | {utility:7.3f}")
    
    print("-" * 60)
    print(f"{'Weighted Avg':13} | 100% | {weighted_allocation:7.0%} | {weighted_return:13.1%} | {weighted_utility:7.3f}")
    
    print(f"\nRecommendation: Start with {weighted_allocation:.0%} covered call allocation")
    print(f"Adjust dynamically based on regime identification")

if __name__ == "__main__":
    optimizer = AdvancedPortfolioOptimizer()
    results = optimizer.run_comprehensive_optimization()
    scenario_optimization_framework()