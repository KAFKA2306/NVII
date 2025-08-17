#!/usr/bin/env python3
"""
REX Growth & Income ETF Family Comprehensive Analyzer

Analyzes the entire REX Shares ETF family with common strategy characteristics:
- NVII (NVDA): REX NVDA Growth & Income ETF
- MSII (MSTR): REX MSTR Growth & Income ETF  
- COII (COIN): REX COIN Growth & Income ETF
- TSII (TSLA): REX TSLA Growth & Income ETF

Common Features:
- Leverage: 1.05x-1.50x (target 1.25x)
- Covered Calls: 50% partial coverage
- Distribution: Weekly
- Upside Potential: Unlimited on 50% portion
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class REXFamilyAnalyzer:
    """
    Comprehensive analyzer for REX Growth & Income ETF family
    """
    
    def __init__(self):
        """Initialize REX family analyzer with all ETFs and their underlying assets"""
        
        # REX ETF Family Configuration
        self.rex_etfs = {
            'NVII': {
                'name': 'REX NVDA Growth & Income ETF',
                'underlying': 'NVDA',
                'underlying_name': 'NVIDIA Corporation',
                'launch_date': '2024-05-01',  # Approximate
                'description': 'Tech/AI semiconductor leader'
            },
            'MSII': {
                'name': 'REX MSTR Growth & Income ETF', 
                'underlying': 'MSTR',
                'underlying_name': 'MicroStrategy Inc',
                'launch_date': '2024-06-01',  # Approximate
                'description': 'Bitcoin proxy/business intelligence'
            },
            'COII': {
                'name': 'REX COIN Growth & Income ETF',
                'underlying': 'COIN', 
                'underlying_name': 'Coinbase Global Inc',
                'launch_date': '2024-07-01',  # Approximate
                'description': 'Cryptocurrency exchange platform'
            },
            'TSII': {
                'name': 'REX TSLA Growth & Income ETF',
                'underlying': 'TSLA',
                'underlying_name': 'Tesla Inc',
                'launch_date': '2024-08-01',  # Approximate
                'description': 'Electric vehicles/autonomous driving'
            }
        }
        
        # Common REX Strategy Parameters
        self.strategy_params = {
            'target_leverage': 1.25,
            'leverage_range': (1.05, 1.50),
            'covered_call_coverage': 0.50,  # 50% coverage
            'upside_unlimited_portion': 0.50,  # 50% unlimited upside
            'distribution_frequency': 'weekly',
            'risk_free_rate': 0.045
        }
        
        self.data = {}
        self.analysis_results = {}
        
    def fetch_all_data(self, period='6mo'):
        """
        Fetch data for all REX ETFs and their underlying assets
        
        Args:
            period (str): Data period ('6mo', '1y', '2y', 'max')
        """
        print("📊 Fetching REX Family data...")
        
        for etf_symbol, config in self.rex_etfs.items():
            try:
                underlying_symbol = config['underlying']
                
                print(f"  Fetching {etf_symbol} ({underlying_symbol})...")
                
                # Fetch ETF data
                etf_ticker = yf.Ticker(etf_symbol)
                etf_data = etf_ticker.history(period=period)
                
                # Fetch underlying asset data
                underlying_ticker = yf.Ticker(underlying_symbol) 
                underlying_data = underlying_ticker.history(period=period)
                
                # Store data
                self.data[etf_symbol] = {
                    'etf_data': etf_data,
                    'underlying_data': underlying_data,
                    'etf_info': etf_ticker.info,
                    'underlying_info': underlying_ticker.info,
                    'config': config
                }
                
                print(f"    ✅ {etf_symbol}: {len(etf_data)} trading days")
                
            except Exception as e:
                print(f"    ❌ {etf_symbol}: {str(e)}")
                # Create empty data structure for missing ETFs
                self.data[etf_symbol] = {
                    'etf_data': pd.DataFrame(),
                    'underlying_data': underlying_data if 'underlying_data' in locals() else pd.DataFrame(),
                    'etf_info': {},
                    'underlying_info': underlying_ticker.info if 'underlying_ticker' in locals() else {},
                    'config': config
                }
    
    def calculate_family_metrics(self):
        """
        Calculate comprehensive metrics for all REX ETFs
        """
        print("📈 Calculating REX Family metrics...")
        
        for etf_symbol, data in self.data.items():
            etf_data = data['etf_data']
            underlying_data = data['underlying_data']
            
            if etf_data.empty:
                print(f"  ⚠️ Skipping {etf_symbol} (no data)")
                continue
                
            print(f"  Analyzing {etf_symbol}...")
            
            # Calculate returns
            etf_data['Price_Return'] = etf_data['Close'].pct_change()
            etf_data['Dividend_Yield'] = etf_data['Dividends'] / etf_data['Close'].shift(1)
            etf_data['Total_Return'] = etf_data['Price_Return'] + etf_data['Dividend_Yield']
            etf_data['Cumulative_Total_Return'] = (1 + etf_data['Total_Return']).cumprod() - 1
            
            # Calculate underlying returns for comparison
            if not underlying_data.empty:
                common_dates = etf_data.index.intersection(underlying_data.index)
                if len(common_dates) > 0:
                    underlying_aligned = underlying_data.loc[common_dates]
                    underlying_aligned['Return'] = underlying_aligned['Close'].pct_change()
                    underlying_aligned['Cumulative_Return'] = (1 + underlying_aligned['Return']).cumprod() - 1
                    
                    # Store aligned underlying data
                    data['underlying_aligned'] = underlying_aligned
            
            # Key performance metrics
            total_days = len(etf_data)
            if total_days > 1:
                total_return = etf_data['Cumulative_Total_Return'].iloc[-1]
                annualized_return = ((1 + total_return) ** (252/total_days)) - 1
                volatility = etf_data['Total_Return'].std() * np.sqrt(252)
                sharpe_ratio = (annualized_return - self.strategy_params['risk_free_rate']) / volatility if volatility > 0 else 0
                
                # Dividend analysis
                total_dividends = etf_data['Dividends'].sum()
                dividend_yield_annual = total_dividends / etf_data['Close'].iloc[0] * (252/total_days)
                
                # Weekly dividend consistency (REX specialty)
                dividend_payments = etf_data[etf_data['Dividends'] > 0]['Dividends']
                dividend_consistency = dividend_payments.std() / dividend_payments.mean() if len(dividend_payments) > 1 else 0
                
                # Maximum drawdown
                rolling_max = (1 + etf_data['Total_Return']).cumprod().expanding().max()
                drawdown = (1 + etf_data['Total_Return']).cumprod() / rolling_max - 1
                max_drawdown = drawdown.min()
                
                # Leverage estimation (if possible)
                if not underlying_data.empty and len(common_dates) > 10:
                    etf_returns = etf_data.loc[common_dates, 'Price_Return'].dropna()
                    underlying_returns = underlying_aligned['Return'].dropna()
                    
                    # Estimate beta (proxy for leverage)
                    if len(etf_returns) > 5 and len(underlying_returns) > 5:
                        covariance = np.cov(etf_returns, underlying_returns)[0,1]
                        underlying_variance = np.var(underlying_returns)
                        estimated_leverage = covariance / underlying_variance if underlying_variance > 0 else 1.0
                    else:
                        estimated_leverage = 1.0
                else:
                    estimated_leverage = 1.25  # Default target
                
                # Store results
                self.analysis_results[etf_symbol] = {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'total_dividends': total_dividends,
                    'dividend_yield_annual': dividend_yield_annual,
                    'dividend_consistency': dividend_consistency,
                    'estimated_leverage': estimated_leverage,
                    'trading_days': total_days,
                    'current_price': etf_data['Close'].iloc[-1],
                    'start_price': etf_data['Close'].iloc[0]
                }
    
    def compare_with_underlying(self):
        """
        Compare each REX ETF performance with its underlying asset
        """
        print("🔄 Comparing REX ETFs with underlying assets...")
        
        comparison_data = []
        
        for etf_symbol, data in self.data.items():
            if etf_symbol not in self.analysis_results:
                continue
                
            etf_metrics = self.analysis_results[etf_symbol]
            underlying_symbol = data['config']['underlying']
            
            # Get underlying performance if available
            if 'underlying_aligned' in data and not data['underlying_aligned'].empty:
                underlying_data = data['underlying_aligned']
                underlying_total_return = underlying_data['Cumulative_Return'].iloc[-1]
                underlying_volatility = underlying_data['Return'].std() * np.sqrt(252)
                
                # Performance comparison
                excess_return = etf_metrics['total_return'] - underlying_total_return
                volatility_difference = etf_metrics['volatility'] - underlying_volatility
                
                comparison_data.append({
                    'ETF': etf_symbol,
                    'Underlying': underlying_symbol,
                    'ETF_Return': etf_metrics['total_return'],
                    'Underlying_Return': underlying_total_return,
                    'Excess_Return': excess_return,
                    'ETF_Volatility': etf_metrics['volatility'],
                    'Underlying_Volatility': underlying_volatility,
                    'Volatility_Diff': volatility_difference,
                    'Dividend_Yield': etf_metrics['dividend_yield_annual'],
                    'Estimated_Leverage': etf_metrics['estimated_leverage']
                })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        return self.comparison_df
    
    def generate_family_dashboard(self):
        """
        Generate comprehensive REX family dashboard
        """
        print("🎨 Generating REX Family Dashboard...")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Family Performance Comparison
        ax1 = plt.subplot(3, 3, 1)
        etf_names = []
        returns = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (etf_symbol, metrics) in enumerate(self.analysis_results.items()):
            etf_names.append(etf_symbol)
            returns.append(metrics['total_return'] * 100)
        
        bars = ax1.bar(etf_names, returns, color=colors[:len(etf_names)], alpha=0.8)
        ax1.set_title('REX Family Total Returns', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Total Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, return_val in zip(bars, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{return_val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Dividend Yields Comparison
        ax2 = plt.subplot(3, 3, 2)
        dividend_yields = [metrics['dividend_yield_annual'] * 100 for metrics in self.analysis_results.values()]
        bars2 = ax2.bar(etf_names, dividend_yields, color=colors[:len(etf_names)], alpha=0.8)
        ax2.set_title('Annual Dividend Yields', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Dividend Yield (%)')
        ax2.grid(True, alpha=0.3)
        
        for bar, yield_val in zip(bars2, dividend_yields):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{yield_val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Risk-Adjusted Returns (Sharpe Ratios)
        ax3 = plt.subplot(3, 3, 3)
        sharpe_ratios = [metrics['sharpe_ratio'] for metrics in self.analysis_results.values()]
        bars3 = ax3.bar(etf_names, sharpe_ratios, color=colors[:len(etf_names)], alpha=0.8)
        ax3.set_title('Sharpe Ratios (Risk-Adjusted Returns)', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
        
        for bar, sharpe_val in zip(bars3, sharpe_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{sharpe_val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Volatility Comparison
        ax4 = plt.subplot(3, 3, 4)
        volatilities = [metrics['volatility'] * 100 for metrics in self.analysis_results.values()]
        bars4 = ax4.bar(etf_names, volatilities, color=colors[:len(etf_names)], alpha=0.8)
        ax4.set_title('Annualized Volatility', fontweight='bold', fontsize=14)
        ax4.set_ylabel('Volatility (%)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Maximum Drawdown
        ax5 = plt.subplot(3, 3, 5)
        max_drawdowns = [metrics['max_drawdown'] * 100 for metrics in self.analysis_results.values()]
        bars5 = ax5.bar(etf_names, max_drawdowns, color='red', alpha=0.6)
        ax5.set_title('Maximum Drawdown', fontweight='bold', fontsize=14)
        ax5.set_ylabel('Max Drawdown (%)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Estimated Leverage
        ax6 = plt.subplot(3, 3, 6)
        leverages = [metrics['estimated_leverage'] for metrics in self.analysis_results.values()]
        bars6 = ax6.bar(etf_names, leverages, color='purple', alpha=0.7)
        ax6.axhline(y=1.25, color='red', linestyle='--', alpha=0.8, label='Target (1.25x)')
        ax6.set_title('Estimated Leverage', fontweight='bold', fontsize=14)
        ax6.set_ylabel('Leverage Ratio')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Performance vs Underlying (if comparison data available)
        if hasattr(self, 'comparison_df') and not self.comparison_df.empty:
            ax7 = plt.subplot(3, 3, 7)
            x_pos = np.arange(len(self.comparison_df))
            width = 0.35
            
            ax7.bar(x_pos - width/2, self.comparison_df['ETF_Return'] * 100, 
                   width, label='REX ETF', color='blue', alpha=0.8)
            ax7.bar(x_pos + width/2, self.comparison_df['Underlying_Return'] * 100,
                   width, label='Underlying', color='orange', alpha=0.8)
            
            ax7.set_title('ETF vs Underlying Performance', fontweight='bold', fontsize=14)
            ax7.set_ylabel('Total Return (%)')
            ax7.set_xticks(x_pos)
            ax7.set_xticklabels(self.comparison_df['ETF'])
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Dividend Consistency
        ax8 = plt.subplot(3, 3, 8)
        consistency_scores = [(1 - metrics['dividend_consistency']) * 100 for metrics in self.analysis_results.values()]
        bars8 = ax8.bar(etf_names, consistency_scores, color='green', alpha=0.7)
        ax8.set_title('Dividend Consistency Score', fontweight='bold', fontsize=14)
        ax8.set_ylabel('Consistency Score (%)')
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary Statistics Table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Create summary table
        summary_data = []
        for etf_symbol, metrics in self.analysis_results.items():
            summary_data.append([
                etf_symbol,
                f"${metrics['current_price']:.2f}",
                f"{metrics['total_return']*100:.1f}%",
                f"{metrics['dividend_yield_annual']*100:.1f}%",
                f"{metrics['sharpe_ratio']:.2f}"
            ])
        
        table = ax9.table(cellText=summary_data,
                         colLabels=['ETF', 'Price', 'Return', 'Div Yield', 'Sharpe'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax9.set_title('REX Family Summary', fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        return fig
    
    def export_family_analysis(self, output_dir='/home/kafka/projects/NVII/docs'):
        """
        Export comprehensive REX family analysis to files
        """
        print("💾 Exporting REX Family Analysis...")
        
        # 1. Summary CSV
        summary_data = []
        for etf_symbol, metrics in self.analysis_results.items():
            config = self.data[etf_symbol]['config']
            summary_data.append({
                'ETF_Symbol': etf_symbol,
                'ETF_Name': config['name'],
                'Underlying_Symbol': config['underlying'],
                'Underlying_Name': config['underlying_name'],
                'Description': config['description'],
                'Current_Price': metrics['current_price'],
                'Total_Return': metrics['total_return'],
                'Annualized_Return': metrics['annualized_return'],
                'Dividend_Yield_Annual': metrics['dividend_yield_annual'],
                'Volatility': metrics['volatility'],
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Max_Drawdown': metrics['max_drawdown'],
                'Estimated_Leverage': metrics['estimated_leverage'],
                'Total_Dividends': metrics['total_dividends'],
                'Trading_Days': metrics['trading_days']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{output_dir}/rex_family_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  ✅ Summary saved: {summary_path}")
        
        # 2. Comparison CSV (if available)
        if hasattr(self, 'comparison_df') and not self.comparison_df.empty:
            comparison_path = f"{output_dir}/rex_vs_underlying_comparison.csv"
            self.comparison_df.to_csv(comparison_path, index=False)
            print(f"  ✅ Comparison saved: {comparison_path}")
        
        # 3. Analysis Report
        report_path = f"{output_dir}/rex_family_analysis_report.md"
        self.generate_analysis_report(report_path)
        print(f"  ✅ Report saved: {report_path}")
        
        return summary_df
    
    def generate_analysis_report(self, output_path):
        """
        Generate comprehensive markdown analysis report
        """
        report_content = f"""# REX Growth & Income ETF Family Analysis

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Family Overview

The REX Shares Growth & Income ETF family employs a consistent strategy across multiple high-profile underlying assets:

"""
        
        # Add individual ETF sections
        for etf_symbol, config in self.rex_etfs.items():
            if etf_symbol in self.analysis_results:
                metrics = self.analysis_results[etf_symbol]
                
                report_content += f"""### {etf_symbol} - {config['name']}

**Underlying Asset:** {config['underlying']} ({config['underlying_name']})
**Investment Thesis:** {config['description']}

**Performance Metrics:**
- Current Price: ${metrics['current_price']:.2f}
- Total Return: {metrics['total_return']*100:.2f}%
- Annualized Return: {metrics['annualized_return']*100:.2f}%
- Annual Dividend Yield: {metrics['dividend_yield_annual']*100:.2f}%
- Volatility: {metrics['volatility']*100:.2f}%
- Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
- Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%
- Estimated Leverage: {metrics['estimated_leverage']:.2f}x

"""
        
        # Add family comparison
        if hasattr(self, 'comparison_df') and not self.comparison_df.empty:
            report_content += """## REX vs Underlying Asset Performance

| ETF | Underlying | ETF Return | Underlying Return | Excess Return | Dividend Yield |
|-----|------------|------------|-------------------|---------------|----------------|
"""
            for _, row in self.comparison_df.iterrows():
                report_content += f"| {row['ETF']} | {row['Underlying']} | {row['ETF_Return']*100:.2f}% | {row['Underlying_Return']*100:.2f}% | {row['Excess_Return']*100:.2f}% | {row['Dividend_Yield']*100:.2f}% |\n"
        
        # Add strategy analysis
        report_content += """
## Common Strategy Analysis

All REX Growth & Income ETFs share these strategic characteristics:

### Strategy Parameters
- **Target Leverage:** 1.25x
- **Covered Call Coverage:** 50% of holdings
- **Upside Participation:** 50% unlimited upside potential
- **Distribution Frequency:** Weekly
- **Income Focus:** High dividend yield through covered call premiums

### Risk-Return Profile
The REX strategy aims to:
1. **Generate Income:** Weekly distributions from covered call premiums
2. **Maintain Growth:** 50% uncovered position for unlimited upside
3. **Enhance Returns:** Modest leverage (1.25x target)
4. **Manage Risk:** Covered calls provide some downside protection

### Sector Diversification
The REX family provides exposure to key growth sectors:
- **Technology/AI:** NVII (NVIDIA)
- **Cryptocurrency/FinTech:** MSII (MicroStrategy), COII (Coinbase) 
- **Electric Vehicles/Clean Energy:** TSII (Tesla)

This diversification allows investors to access the REX strategy across different high-growth themes while maintaining consistent income generation.

---
*Analysis generated by REX Family Analyzer*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

def main():
    """
    Main execution function for REX family analysis
    """
    print("🚀 REX Growth & Income ETF Family Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = REXFamilyAnalyzer()
    
    try:
        # Fetch data for all REX ETFs
        analyzer.fetch_all_data(period='6mo')
        
        # Calculate metrics
        analyzer.calculate_family_metrics()
        
        # Compare with underlying assets
        comparison_df = analyzer.compare_with_underlying()
        
        # Generate dashboard
        fig = analyzer.generate_family_dashboard()
        
        # Save chart
        chart_path = '/home/kafka/projects/NVII/docs/rex_family_dashboard.png'
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard saved: {chart_path}")
        plt.close(fig)
        
        # Export analysis
        summary_df = analyzer.export_family_analysis()
        
        # Display summary
        print("\n" + "=" * 60)
        print("📋 REX FAMILY ANALYSIS SUMMARY")
        print("=" * 60)
        
        if not summary_df.empty:
            for _, row in summary_df.iterrows():
                print(f"\n{row['ETF_Symbol']} ({row['Underlying_Symbol']}):")
                print(f"  Price: ${row['Current_Price']:.2f}")
                print(f"  Return: {row['Total_Return']*100:.2f}%")
                print(f"  Dividend Yield: {row['Dividend_Yield_Annual']*100:.2f}%")
                print(f"  Sharpe Ratio: {row['Sharpe_Ratio']:.3f}")
        
        print(f"\n📂 All analysis files saved to: /home/kafka/projects/NVII/docs/")
        print("✅ REX Family Analysis Complete!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()