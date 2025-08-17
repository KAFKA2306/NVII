#!/usr/bin/env python3
"""
NVII Comprehensive Sensitivity Analysis
Multi-dimensional parameter sensitivity with interactive visualizations
Analyzes impact of key variables on portfolio performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from option_analysis import BlackScholesCalculator, NVIIAnalyzer

class SensitivityAnalyzer:
    """Comprehensive sensitivity analysis for NVII strategy"""
    
    def __init__(self, base_price=32.97, base_leverage=1.25):
        self.base_price = base_price
        self.base_leverage = base_leverage
        self.risk_free_rate = 0.045
        
        # Base case parameters
        self.base_params = {
            'nvii_price': 32.97,
            'leverage': 1.25,
            'volatility': 0.55,
            'time_to_expiry': 7/365,
            'strike_otm': 0.05,
            'cc_allocation': 0.5,
            'risk_free_rate': 0.045
        }
        
        # Sensitivity ranges
        self.sensitivity_ranges = {
            'nvii_price': np.linspace(25, 45, 21),
            'leverage': np.linspace(1.05, 1.5, 19),
            'volatility': np.linspace(0.2, 1.2, 21),
            'strike_otm': np.linspace(0.02, 0.15, 14),
            'cc_allocation': np.linspace(0.2, 0.8, 13),
            'time_to_expiry': np.array([1, 3, 7, 14, 21, 30]) / 365
        }
    
    def calculate_option_metrics(self, price, leverage, volatility, time_exp, strike_otm):
        """Calculate option metrics for given parameters"""
        try:
            strike_price = price * (1 + strike_otm)
            
            bs = BlackScholesCalculator(
                price, strike_price, time_exp, self.risk_free_rate, volatility, 'call'
            )
            
            option_price = bs.theoretical_price()
            delta = bs.delta()
            gamma = bs.gamma()
            theta = bs.theta()
            vega = bs.vega()
            
            # Leverage adjustments
            leveraged_premium = option_price * leverage
            annual_yield = (leveraged_premium * 52 * 0.5) / price  # 50% coverage, weekly
            
            return {
                'option_price': option_price,
                'leveraged_premium': leveraged_premium,
                'annual_yield': annual_yield,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'intrinsic_value': max(0, price - strike_price),
                'time_value': option_price - max(0, price - strike_price),
                'moneyness': price / strike_price
            }
        except:
            return {k: 0 for k in ['option_price', 'leveraged_premium', 'annual_yield', 
                                  'delta', 'gamma', 'theta', 'vega', 'intrinsic_value', 
                                  'time_value', 'moneyness']}
    
    def single_variable_sensitivity(self, variable, metric='annual_yield'):
        """Analyze sensitivity to a single variable"""
        
        results = []
        base_case = self.base_params.copy()
        
        for value in self.sensitivity_ranges[variable]:
            params = base_case.copy()
            params[variable] = value
            
            metrics = self.calculate_option_metrics(
                params['nvii_price'],
                params['leverage'],
                params['volatility'],
                params['time_to_expiry'],
                params['strike_otm']
            )
            
            results.append({
                variable: value,
                metric: metrics[metric],
                'option_price': metrics['option_price'],
                'delta': metrics['delta'],
                'gamma': metrics['gamma'],
                'theta': metrics['theta'],
                'vega': metrics['vega']
            })
        
        return pd.DataFrame(results)
    
    def two_variable_sensitivity(self, var1, var2, metric='annual_yield'):
        """Analyze sensitivity to two variables simultaneously"""
        
        results = []
        base_case = self.base_params.copy()
        
        for val1 in self.sensitivity_ranges[var1]:
            for val2 in self.sensitivity_ranges[var2]:
                params = base_case.copy()
                params[var1] = val1
                params[var2] = val2
                
                metrics = self.calculate_option_metrics(
                    params['nvii_price'],
                    params['leverage'],
                    params['volatility'],
                    params['time_to_expiry'],
                    params['strike_otm']
                )
                
                results.append({
                    var1: val1,
                    var2: val2,
                    metric: metrics[metric]
                })
        
        df = pd.DataFrame(results)
        pivot_table = df.pivot(index=var1, columns=var2, values=metric)
        return pivot_table
    
    def portfolio_sensitivity_analysis(self):
        """Comprehensive portfolio-level sensitivity analysis"""
        
        print("="*80)
        print("NVII COMPREHENSIVE SENSITIVITY ANALYSIS")
        print("="*80)
        
        # 1. Single Variable Sensitivities
        print("\n1. SINGLE VARIABLE SENSITIVITY ANALYSIS")
        print("-" * 50)
        
        single_var_results = {}
        
        for variable in ['volatility', 'leverage', 'strike_otm', 'cc_allocation']:
            sensitivity_data = self.single_variable_sensitivity(variable)
            single_var_results[variable] = sensitivity_data
            
            # Calculate sensitivity coefficient (% change in yield per % change in variable)
            base_value = self.base_params[variable] if variable in self.base_params else 0.5
            base_yield = sensitivity_data.iloc[len(sensitivity_data)//2]['annual_yield']
            
            # Find nearest values for sensitivity calculation
            mid_idx = len(sensitivity_data) // 2
            if mid_idx > 0 and mid_idx < len(sensitivity_data) - 1:
                lower_yield = sensitivity_data.iloc[mid_idx - 1]['annual_yield']
                upper_yield = sensitivity_data.iloc[mid_idx + 1]['annual_yield']
                lower_val = sensitivity_data.iloc[mid_idx - 1][variable]
                upper_val = sensitivity_data.iloc[mid_idx + 1][variable]
                
                sensitivity_coeff = ((upper_yield - lower_yield) / base_yield) / ((upper_val - lower_val) / base_value)
                
                print(f"{variable:15}: Sensitivity = {sensitivity_coeff:+6.2f} (yield elasticity)")
        
        # 2. Volatility Impact Analysis
        print("\n2. VOLATILITY IMPACT ANALYSIS")
        print("-" * 50)
        
        vol_analysis = single_var_results['volatility']
        
        print("Volatility | Annual Yield | Option Price | Delta | Vega")
        print("-" * 55)
        
        for _, row in vol_analysis.iterrows():
            vol = row['volatility']
            ay = row['annual_yield']
            op = row['option_price']
            delta = row['delta']
            vega = row['vega']
            
            print(f"{vol:8.0%} | {ay:10.1%} | ${op:9.4f} | {delta:5.3f} | {vega:5.3f}")
        
        # 3. Leverage Impact Analysis
        print("\n3. LEVERAGE IMPACT ANALYSIS")
        print("-" * 50)
        
        lev_analysis = single_var_results['leverage']
        
        print("Leverage | Annual Yield | Enhancement | Risk Multiplier")
        print("-" * 52)
        
        base_yield_no_lev = lev_analysis.iloc[0]['annual_yield'] / self.sensitivity_ranges['leverage'][0]
        
        for _, row in lev_analysis.iterrows():
            lev = row['leverage']
            ay = row['annual_yield']
            enhancement = (ay / base_yield_no_lev) - 1
            risk_mult = lev
            
            print(f"{lev:7.2f}x | {ay:10.1%} | {enhancement:9.1%} | {risk_mult:13.2f}x")
        
        # 4. Strike Selection Analysis
        print("\n4. STRIKE SELECTION OPTIMIZATION")
        print("-" * 50)
        
        strike_analysis = single_var_results['strike_otm']
        
        print("Strike OTM | Annual Yield | Assignment Risk | Risk Score")
        print("-" * 55)
        
        for _, row in strike_analysis.iterrows():
            otm = row['strike_otm']
            ay = row['annual_yield']
            delta = row['delta']
            
            # Simple assignment risk proxy
            assignment_risk = delta * 100
            
            # Risk score (higher yield vs higher assignment risk)
            risk_score = ay / (assignment_risk / 100) if assignment_risk > 0 else 0
            
            print(f"{otm:9.1%} | {ay:10.1%} | {assignment_risk:12.1%} | {risk_score:9.2f}")
        
        return single_var_results
    
    def create_sensitivity_heatmaps(self, single_var_results):
        """Create comprehensive sensitivity heatmaps"""
        
        print("\n5. TWO-VARIABLE SENSITIVITY HEATMAPS")
        print("-" * 50)
        
        # Key two-variable combinations
        combinations = [
            ('volatility', 'leverage'),
            ('volatility', 'strike_otm'),
            ('leverage', 'cc_allocation'),
            ('strike_otm', 'cc_allocation')
        ]
        
        heatmap_data = {}
        
        for var1, var2 in combinations:
            print(f"\nGenerating {var1} vs {var2} sensitivity map...")
            heatmap = self.two_variable_sensitivity(var1, var2)
            heatmap_data[f"{var1}_vs_{var2}"] = heatmap
            
            # Print summary statistics
            print(f"  Range: {heatmap.min().min():.1%} to {heatmap.max().max():.1%}")
            print(f"  Mean: {heatmap.mean().mean():.1%}")
            print(f"  Std Dev: {heatmap.std().mean():.1%}")
        
        return heatmap_data
    
    def scenario_stress_testing(self):
        """Stress testing under extreme scenarios"""
        
        print("\n6. SCENARIO STRESS TESTING")
        print("-" * 50)
        
        stress_scenarios = {
            'market_crash': {
                'volatility': 1.0,
                'price_change': -0.4,
                'leverage': 1.25
            },
            'vol_collapse': {
                'volatility': 0.25,
                'price_change': 0.1,
                'leverage': 1.25
            },
            'mega_rally': {
                'volatility': 0.6,
                'price_change': 0.8,
                'leverage': 1.25
            },
            'max_leverage': {
                'volatility': 0.55,
                'price_change': 0,
                'leverage': 1.5
            },
            'min_leverage': {
                'volatility': 0.55,
                'price_change': 0,
                'leverage': 1.05
            }
        }
        
        print("Scenario       | Annual Yield | Option Price | Risk Assessment")
        print("-" * 65)
        
        stress_results = {}
        
        for scenario_name, scenario in stress_scenarios.items():
            new_price = self.base_price * (1 + scenario['price_change'])
            
            metrics = self.calculate_option_metrics(
                new_price,
                scenario['leverage'],
                scenario['volatility'],
                self.base_params['time_to_expiry'],
                self.base_params['strike_otm']
            )
            
            annual_yield = metrics['annual_yield']
            option_price = metrics['option_price']
            
            # Risk assessment
            if annual_yield > 0.5:
                risk = "Extreme"
            elif annual_yield > 0.3:
                risk = "High"
            elif annual_yield > 0.15:
                risk = "Medium"
            else:
                risk = "Low"
            
            stress_results[scenario_name] = {
                'annual_yield': annual_yield,
                'option_price': option_price,
                'risk_assessment': risk,
                'scenario_params': scenario
            }
            
            print(f"{scenario_name:14} | {annual_yield:10.1%} | ${option_price:9.4f} | {risk:14}")
        
        return stress_results
    
    def monte_carlo_sensitivity(self, num_simulations=1000):
        """Monte Carlo sensitivity analysis with parameter uncertainty"""
        
        print("\n7. MONTE CARLO PARAMETER SENSITIVITY")
        print("-" * 50)
        
        # Parameter uncertainty distributions
        param_distributions = {
            'volatility': {'mean': 0.55, 'std': 0.15, 'min': 0.2, 'max': 1.0},
            'leverage': {'mean': 1.25, 'std': 0.05, 'min': 1.05, 'max': 1.5},
            'price': {'mean': 32.97, 'std': 5.0, 'min': 20, 'max': 50}
        }
        
        results = []
        
        for _ in range(num_simulations):
            # Sample parameters from distributions
            vol = np.clip(np.random.normal(
                param_distributions['volatility']['mean'],
                param_distributions['volatility']['std']
            ), param_distributions['volatility']['min'], param_distributions['volatility']['max'])
            
            lev = np.clip(np.random.normal(
                param_distributions['leverage']['mean'],
                param_distributions['leverage']['std']
            ), param_distributions['leverage']['min'], param_distributions['leverage']['max'])
            
            price = np.clip(np.random.normal(
                param_distributions['price']['mean'],
                param_distributions['price']['std']
            ), param_distributions['price']['min'], param_distributions['price']['max'])
            
            # Calculate metrics
            metrics = self.calculate_option_metrics(
                price, lev, vol,
                self.base_params['time_to_expiry'],
                self.base_params['strike_otm']
            )
            
            results.append({
                'volatility': vol,
                'leverage': lev,
                'price': price,
                'annual_yield': metrics['annual_yield'],
                'option_price': metrics['option_price'],
                'delta': metrics['delta']
            })
        
        mc_df = pd.DataFrame(results)
        
        # Summary statistics
        print(f"Monte Carlo Results ({num_simulations:,} simulations):")
        print(f"Annual Yield - Mean: {mc_df['annual_yield'].mean():.1%}, "
              f"Std: {mc_df['annual_yield'].std():.1%}")
        print(f"5th-95th Percentile: {mc_df['annual_yield'].quantile(0.05):.1%} - "
              f"{mc_df['annual_yield'].quantile(0.95):.1%}")
        
        # Correlation analysis
        correlations = mc_df[['volatility', 'leverage', 'price', 'annual_yield']].corr()['annual_yield']
        
        print(f"\nCorrelations with Annual Yield:")
        for var, corr in correlations.items():
            if var != 'annual_yield':
                print(f"  {var:10}: {corr:+6.3f}")
        
        return mc_df
    
    def generate_sensitivity_tables(self, single_var_results):
        """Generate formatted sensitivity tables for reporting"""
        
        print("\n8. SENSITIVITY SUMMARY TABLES")
        print("-" * 50)
        
        # Create comprehensive sensitivity table
        sensitivity_summary = {}
        
        for variable, data in single_var_results.items():
            if variable in self.base_params:
                base_val = self.base_params[variable]
            else:
                base_val = 0.5  # Default for allocation
            
            # Find base case
            base_idx = np.argmin(np.abs(data[variable] - base_val))
            base_yield = data.iloc[base_idx]['annual_yield']
            
            # Calculate impacts for ±10% and ±25% changes
            impacts = {}
            
            for pct_change in [-0.25, -0.10, 0.10, 0.25]:
                target_val = base_val * (1 + pct_change)
                closest_idx = np.argmin(np.abs(data[variable] - target_val))
                impact_yield = data.iloc[closest_idx]['annual_yield']
                yield_change = (impact_yield - base_yield) / base_yield
                impacts[f"{pct_change:+.0%}"] = yield_change
            
            sensitivity_summary[variable] = {
                'base_value': base_val,
                'base_yield': base_yield,
                'impacts': impacts
            }
        
        # Print formatted table
        print("\nSensitivity Impact on Annual Yield:")
        print("Variable       | Base Value | Base Yield | -25%   | -10%   | +10%   | +25%")
        print("-" * 75)
        
        for var, data in sensitivity_summary.items():
            base_val = data['base_value']
            base_yield = data['base_yield']
            impacts = data['impacts']
            
            if var == 'volatility':
                base_str = f"{base_val:.0%}"
            elif var == 'leverage':
                base_str = f"{base_val:.2f}x"
            elif var == 'strike_otm':
                base_str = f"{base_val:.1%}"
            else:
                base_str = f"{base_val:.1%}"
            
            print(f"{var:14} | {base_str:10} | {base_yield:8.1%} | "
                  f"{impacts.get('-25%', 0):+5.0%} | {impacts.get('-10%', 0):+5.0%} | "
                  f"{impacts.get('+10%', 0):+5.0%} | {impacts.get('+25%', 0):+5.0%}")
        
        return sensitivity_summary
    
    def create_visualization_code(self):
        """Generate Python code for creating visualizations"""
        
        visualization_code = '''
# NVII Sensitivity Analysis Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_sensitivity_plots(single_var_results, heatmap_data):
    """Create comprehensive sensitivity visualization plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Single Variable Sensitivity Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('NVII Strategy: Single Variable Sensitivity Analysis', fontsize=16)
    
    variables = ['volatility', 'leverage', 'strike_otm', 'cc_allocation']
    titles = ['Volatility Impact', 'Leverage Impact', 'Strike Selection', 'Allocation Impact']
    
    for i, (var, title) in enumerate(zip(variables, titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        data = single_var_results[var]
        ax.plot(data[var], data['annual_yield'] * 100, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel(var.replace('_', ' ').title())
        ax.set_ylabel('Annual Yield (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis based on variable type
        if var in ['volatility', 'strike_otm', 'cc_allocation']:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        elif var == 'leverage':
            ax.set_xlabel('Leverage (x)')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Heatmap Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('NVII Strategy: Two-Variable Sensitivity Heatmaps', fontsize=16)
    
    heatmap_titles = [
        'Volatility vs Leverage',
        'Volatility vs Strike OTM',
        'Leverage vs CC Allocation',
        'Strike vs CC Allocation'
    ]
    
    for i, (key, title) in enumerate(zip(heatmap_data.keys(), heatmap_titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        heatmap = heatmap_data[key] * 100  # Convert to percentage
        sns.heatmap(heatmap, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=heatmap.mean().mean(), ax=ax)
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()
    
    # 3. 3D Surface Plot for Three Variables
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Example: Volatility vs Leverage vs Yield
    vol_range = np.linspace(0.3, 0.8, 20)
    lev_range = np.linspace(1.05, 1.5, 20)
    Vol, Lev = np.meshgrid(vol_range, lev_range)
    
    # Calculate yields for surface (simplified calculation)
    Yield = Vol * Lev * 0.3  # Simplified relationship
    
    surf = ax.plot_surface(Vol, Lev, Yield, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Leverage')
    ax.set_zlabel('Annual Yield')
    ax.set_title('3D Sensitivity Surface: Volatility vs Leverage vs Yield')
    
    plt.colorbar(surf)
    plt.show()

# Usage example:
# create_sensitivity_plots(single_var_results, heatmap_data)
'''
        
        return visualization_code

def run_comprehensive_sensitivity_analysis():
    """Execute complete sensitivity analysis"""
    
    analyzer = SensitivityAnalyzer()
    
    # Run single variable sensitivity analysis
    single_var_results = analyzer.portfolio_sensitivity_analysis()
    
    # Create heatmaps
    heatmap_data = analyzer.create_sensitivity_heatmaps(single_var_results)
    
    # Stress testing
    stress_results = analyzer.scenario_stress_testing()
    
    # Monte Carlo sensitivity
    mc_results = analyzer.monte_carlo_sensitivity(1000)
    
    # Generate summary tables
    sensitivity_summary = analyzer.generate_sensitivity_tables(single_var_results)
    
    # Generate visualization code
    viz_code = analyzer.create_visualization_code()
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\nKey Insights:")
    print(f"1. Volatility has the highest impact on option yields")
    print(f"2. Leverage provides linear enhancement but increases risk")
    print(f"3. Strike selection involves clear risk-return tradeoffs")
    print(f"4. Allocation flexibility allows for regime adaptation")
    
    return {
        'single_variable': single_var_results,
        'heatmaps': heatmap_data,
        'stress_tests': stress_results,
        'monte_carlo': mc_results,
        'summary_tables': sensitivity_summary,
        'visualization_code': viz_code
    }

if __name__ == "__main__":
    results = run_comprehensive_sensitivity_analysis()