#!/usr/bin/env python3
"""
NVII vs NVDA Total Returns Comparison

This script fetches real market data for NVII and NVDA, calculates comprehensive
total returns including dividends, and generates detailed comparison analysis.

Based on CLAUDE.md specifications and requirements.txt dependencies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas_datareader as pdr
    import yfinance as yf
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install with: pip install yfinance")
    exit(1)

class TotalReturnsComparator:
    """
    Comprehensive total returns comparison between NVII and NVDA
    """
    
    def __init__(self, start_date='2022-01-01', end_date=None):
        """
        Initialize the comparator
        
        Args:
            start_date (str): Start date for analysis (YYYY-MM-DD)
            end_date (str): End date for analysis (YYYY-MM-DD), defaults to today
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.nvii_data = None
        self.nvda_data = None
        self.comparison_results = {}
        
    def fetch_data(self):
        """
        Fetch historical price data for both NVII and NVDA
        """
        print(f"Fetching data from {self.start_date} to {self.end_date}...")
        
        try:
            # Fetch NVII data
            print("Fetching NVII data...")
            nvii_ticker = yf.Ticker("NVII")
            self.nvii_data = nvii_ticker.history(start=self.start_date, end=self.end_date)
            
            # Fetch NVDA data  
            print("Fetching NVDA data...")
            nvda_ticker = yf.Ticker("NVDA")
            self.nvda_data = nvda_ticker.history(start=self.start_date, end=self.end_date)
            
            if self.nvii_data.empty or self.nvda_data.empty:
                raise ValueError("No data retrieved for one or both tickers")
                
            print(f"NVII: {len(self.nvii_data)} trading days")
            print(f"NVDA: {len(self.nvda_data)} trading days")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise
    
    def calculate_total_returns(self):
        """
        Calculate total returns including dividends for both securities
        """
        print("Calculating total returns...")
        
        results = {}
        
        for ticker, data in [("NVII", self.nvii_data), ("NVDA", self.nvda_data)]:
            if data is None or data.empty:
                continue
                
            # Price returns
            data['Price_Return'] = data['Close'].pct_change()
            
            # Dividend yield (assuming dividends are in the data)
            data['Dividend_Yield'] = data['Dividends'] / data['Close'].shift(1)
            
            # Total return (price + dividends)
            data['Total_Return'] = data['Price_Return'] + data['Dividend_Yield']
            
            # Cumulative returns
            data['Cumulative_Price_Return'] = (1 + data['Price_Return']).cumprod() - 1
            data['Cumulative_Total_Return'] = (1 + data['Total_Return']).cumprod() - 1
            
            # Calculate key metrics
            total_days = len(data)
            annualized_price_return = ((1 + data['Cumulative_Price_Return'].iloc[-1]) ** (252/total_days)) - 1
            annualized_total_return = ((1 + data['Cumulative_Total_Return'].iloc[-1]) ** (252/total_days)) - 1
            
            # Volatility
            annualized_volatility = data['Total_Return'].std() * np.sqrt(252)
            
            # Sharpe ratio (assuming risk-free rate of 4.5% from CLAUDE.md)
            risk_free_rate = 0.045
            sharpe_ratio = (annualized_total_return - risk_free_rate) / annualized_volatility
            
            # Maximum drawdown
            rolling_max = (1 + data['Total_Return']).cumprod().expanding().max()
            drawdown = (1 + data['Total_Return']).cumprod() / rolling_max - 1
            max_drawdown = drawdown.min()
            
            # Dividend statistics
            total_dividends = data['Dividends'].sum()
            dividend_yield_annual = total_dividends / data['Close'].iloc[0] * (252/total_days)
            
            results[ticker] = {
                'start_price': data['Close'].iloc[0],
                'end_price': data['Close'].iloc[-1],
                'total_price_return': data['Cumulative_Price_Return'].iloc[-1],
                'total_dividend_yield': total_dividends / data['Close'].iloc[0],
                'total_return': data['Cumulative_Total_Return'].iloc[-1],
                'annualized_price_return': annualized_price_return,
                'annualized_total_return': annualized_total_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_dividends': total_dividends,
                'annualized_dividend_yield': dividend_yield_annual,
                'trading_days': total_days
            }
        
        self.comparison_results = results
        return results
    
    def generate_comparison_analysis(self):
        """
        Generate detailed comparison analysis
        """
        if not self.comparison_results:
            self.calculate_total_returns()
        
        analysis = []
        analysis.append("# NVII vs NVDA Total Returns Comparison Analysis")
        analysis.append(f"\n**Analysis Period:** {self.start_date} to {self.end_date}")
        analysis.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        analysis.append("\n## Summary Statistics")
        analysis.append("\n| Metric | NVII | NVDA | Difference |")
        analysis.append("|--------|------|------|------------|")
        
        nvii = self.comparison_results.get('NVII', {})
        nvda = self.comparison_results.get('NVDA', {})
        
        metrics = [
            ('Total Return', 'total_return', '%.2%'),
            ('Annualized Total Return', 'annualized_total_return', '%.2%'),
            ('Annualized Price Return', 'annualized_price_return', '%.2%'),
            ('Annualized Dividend Yield', 'annualized_dividend_yield', '%.2%'),
            ('Annualized Volatility', 'annualized_volatility', '%.2%'),
            ('Sharpe Ratio', 'sharpe_ratio', '%.3f'),
            ('Maximum Drawdown', 'max_drawdown', '%.2%'),
        ]
        
        for metric_name, key, fmt in metrics:
            nvii_val = nvii.get(key, 0)
            nvda_val = nvda.get(key, 0)
            diff = nvii_val - nvda_val
            
            if fmt.endswith('%'):
                nvii_str = f"{nvii_val:.2%}"
                nvda_str = f"{nvda_val:.2%}"
                diff_str = f"{diff:+.2%}"
            else:
                nvii_str = f"{nvii_val:.3f}"
                nvda_str = f"{nvda_val:.3f}"
                diff_str = f"{diff:+.3f}"
                
            analysis.append(f"| {metric_name} | {nvii_str} | {nvda_str} | {diff_str} |")
        
        # Price performance
        analysis.append("\n## Price Performance")
        analysis.append(f"- **NVII**: ${nvii.get('start_price', 0):.2f} → ${nvii.get('end_price', 0):.2f} ({nvii.get('total_price_return', 0):.2%})")
        analysis.append(f"- **NVDA**: ${nvda.get('start_price', 0):.2f} → ${nvda.get('end_price', 0):.2f} ({nvda.get('total_price_return', 0):.2%})")
        
        # Dividend analysis
        analysis.append("\n## Dividend Analysis")
        analysis.append(f"- **NVII Total Dividends**: ${nvii.get('total_dividends', 0):.2f} per share")
        analysis.append(f"- **NVDA Total Dividends**: ${nvda.get('total_dividends', 0):.2f} per share")
        analysis.append(f"- **NVII Dividend Contribution**: {nvii.get('total_dividend_yield', 0):.2%} of total return")
        analysis.append(f"- **NVDA Dividend Contribution**: {nvda.get('total_dividend_yield', 0):.2%} of total return")
        
        # Risk-adjusted analysis
        analysis.append("\n## Risk-Adjusted Performance")
        nvii_sharpe = nvii.get('sharpe_ratio', 0)
        nvda_sharpe = nvda.get('sharpe_ratio', 0)
        
        if nvii_sharpe > nvda_sharpe:
            better_performer = "NVII"
            sharpe_diff = nvii_sharpe - nvda_sharpe
        else:
            better_performer = "NVDA"
            sharpe_diff = nvda_sharpe - nvii_sharpe
            
        analysis.append(f"- **Superior Risk-Adjusted Returns**: {better_performer} (Sharpe difference: {sharpe_diff:.3f})")
        analysis.append(f"- **NVII Max Drawdown**: {nvii.get('max_drawdown', 0):.2%}")
        analysis.append(f"- **NVDA Max Drawdown**: {nvda.get('max_drawdown', 0):.2%}")
        
        # Key insights
        analysis.append("\n## Key Insights")
        
        nvii_total = nvii.get('total_return', 0)
        nvda_total = nvda.get('total_return', 0)
        
        if nvii_total > nvda_total:
            analysis.append(f"1. **NVII outperformed NVDA** by {nvii_total - nvda_total:.2%} in total returns")
        else:
            analysis.append(f"1. **NVDA outperformed NVII** by {nvda_total - nvii_total:.2%} in total returns")
            
        nvii_div_contribution = nvii.get('total_dividend_yield', 0) / max(nvii_total, 0.001) if nvii_total > 0 else 0
        analysis.append(f"2. **Dividend Impact**: NVII's dividends contributed {nvii_div_contribution:.1%} of its total return")
        
        analysis.append(f"3. **Volatility**: NVII showed {nvii.get('annualized_volatility', 0):.2%} vs NVDA's {nvda.get('annualized_volatility', 0):.2%} annualized volatility")
        
        return "\n".join(analysis)
    
    def export_to_csv(self):
        """
        Export detailed data to CSV files
        """
        csv_files = []
        
        # Export summary statistics
        if self.comparison_results:
            summary_data = []
            for ticker, results in self.comparison_results.items():
                row = {'Ticker': ticker}
                row.update(results)
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = '/home/kafka/projects/NVII/docs/nvii_nvda_summary_stats.csv'
            summary_df.to_csv(summary_path, index=False)
            csv_files.append(summary_path)
        
        # Export daily returns data
        if self.nvii_data is not None and self.nvda_data is not None:
            # Align dates
            common_dates = self.nvii_data.index.intersection(self.nvda_data.index)
            
            if len(common_dates) > 0:
                combined_data = pd.DataFrame(index=common_dates)
                
                # NVII data
                nvii_aligned = self.nvii_data.loc[common_dates]
                combined_data['NVII_Close'] = nvii_aligned['Close']
                combined_data['NVII_Volume'] = nvii_aligned['Volume']
                combined_data['NVII_Dividends'] = nvii_aligned['Dividends']
                combined_data['NVII_Daily_Return'] = nvii_aligned['Total_Return']
                combined_data['NVII_Cumulative_Return'] = nvii_aligned['Cumulative_Total_Return']
                
                # NVDA data
                nvda_aligned = self.nvda_data.loc[common_dates]
                combined_data['NVDA_Close'] = nvda_aligned['Close']
                combined_data['NVDA_Volume'] = nvda_aligned['Volume']
                combined_data['NVDA_Dividends'] = nvda_aligned['Dividends']
                combined_data['NVDA_Daily_Return'] = nvda_aligned['Total_Return']
                combined_data['NVDA_Cumulative_Return'] = nvda_aligned['Cumulative_Total_Return']
                
                # Calculate relative performance
                combined_data['NVII_vs_NVDA_Excess_Return'] = combined_data['NVII_Daily_Return'] - combined_data['NVDA_Daily_Return']
                
                daily_path = '/home/kafka/projects/NVII/docs/nvii_nvda_daily_data.csv'
                combined_data.to_csv(daily_path)
                csv_files.append(daily_path)
        
        # Export individual ticker data
        for ticker, data in [("NVII", self.nvii_data), ("NVDA", self.nvda_data)]:
            if data is not None and not data.empty:
                ticker_path = f'/home/kafka/projects/NVII/docs/{ticker.lower()}_detailed_data.csv'
                data.to_csv(ticker_path)
                csv_files.append(ticker_path)
        
        return csv_files
    
    def create_visualization(self):
        """
        Create enhanced visualization with dividend information
        """
        if self.nvii_data is None or self.nvda_data is None:
            return None
            
        # Align dates for comparison
        common_dates = self.nvii_data.index.intersection(self.nvda_data.index)
        
        nvii_aligned = self.nvii_data.loc[common_dates]
        nvda_aligned = self.nvda_data.loc[common_dates]
        
        # Calculate normalized performance (base = 100)
        nvii_total_performance = (1 + nvii_aligned['Total_Return']).cumprod() * 100
        nvii_price_performance = (1 + nvii_aligned['Price_Return']).cumprod() * 100
        nvda_performance = (1 + nvda_aligned['Total_Return']).cumprod() * 100
        
        # Create enhanced plot with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # Subplot 1: Cumulative Returns
        ax1.plot(common_dates, nvii_total_performance, label='NVII Total Return (Price + Dividends)', 
                linewidth=2.5, color='#1f77b4', linestyle='-')
        ax1.plot(common_dates, nvii_price_performance, label='NVII Capital Return (Price Only)', 
                linewidth=2.5, color='#1f77b4', linestyle='--', alpha=0.8)
        ax1.plot(common_dates, nvda_performance, label='NVDA Total Return', linewidth=2.5, color='#ff7f0e')
        
        # Add dividend markers for NVII
        nvii_div_dates = nvii_aligned[nvii_aligned['Dividends'] > 0].index
        nvii_div_amounts = nvii_aligned[nvii_aligned['Dividends'] > 0]['Dividends']
        nvii_div_total_performance = nvii_total_performance.loc[nvii_div_dates]
        nvii_div_price_performance = nvii_price_performance.loc[nvii_div_dates]
        
        if len(nvii_div_dates) > 0:
            ax1.scatter(nvii_div_dates, nvii_div_total_performance, 
                       s=100, color='red', marker='v', alpha=0.8, zorder=5,
                       label=f'NVII Dividends (${nvii_div_amounts.sum():.2f} total)')
            
            # Add dividend amount annotations on total return line
            for date, amount, performance in zip(nvii_div_dates, nvii_div_amounts, nvii_div_total_performance):
                if amount > 0.15:  # Only annotate larger dividends to avoid clutter
                    ax1.annotate(f'${amount:.2f}', 
                               xy=(date, performance), 
                               xytext=(5, 10), textcoords='offset points',
                               fontsize=8, color='red', alpha=0.8)
            
            # Show the dividend impact by drawing vertical lines
            for date in nvii_div_dates:
                total_val = nvii_total_performance.loc[date]
                price_val = nvii_price_performance.loc[date]
                ax1.plot([date, date], [price_val, total_val], color='red', alpha=0.3, linewidth=1)
        
        # Add dividend markers for NVDA (if any)
        nvda_div_dates = nvda_aligned[nvda_aligned['Dividends'] > 0].index
        nvda_div_amounts = nvda_aligned[nvda_aligned['Dividends'] > 0]['Dividends']
        nvda_div_performance = nvda_performance.loc[nvda_div_dates]
        
        if len(nvda_div_dates) > 0:
            ax1.scatter(nvda_div_dates, nvda_div_performance, 
                       s=60, color='green', marker='^', alpha=0.8, zorder=5,
                       label=f'NVDA Dividends (${nvda_div_amounts.sum():.2f} total)')
        
        ax1.set_title('NVII Total vs Capital Returns & NVDA Comparison with Dividend Events', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Value (Base = 100)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Dividend Timeline
        ax2.bar(nvii_div_dates, nvii_div_amounts, width=1, alpha=0.7, color='blue', 
                label=f'NVII Weekly Dividends (${nvii_div_amounts.sum():.2f} total)')
        if len(nvda_div_dates) > 0:
            ax2.bar(nvda_div_dates, nvda_div_amounts, width=1, alpha=0.7, color='orange',
                    label=f'NVDA Quarterly Dividends (${nvda_div_amounts.sum():.2f} total)')
        
        ax2.set_title('Dividend Payment Timeline', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Dividend Amount ($)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add dividend frequency info
        if len(nvii_div_dates) > 0:
            avg_nvii_div = nvii_div_amounts.mean()
            ax2.text(0.02, 0.95, f'NVII Avg: ${avg_nvii_div:.2f}/week', 
                    transform=ax2.transAxes, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Subplot 3: NVII Excess Returns and Dividend Impact
        excess_total_return = nvii_total_performance - nvda_performance
        excess_price_return = nvii_price_performance - nvda_performance
        
        ax3.plot(common_dates, excess_total_return, label='NVII Total Return - NVDA', linewidth=2, color='blue')
        ax3.plot(common_dates, excess_price_return, label='NVII Capital Return - NVDA', linewidth=2, color='blue', linestyle='--', alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Fill area between the two lines to show dividend impact
        ax3.fill_between(common_dates, excess_price_return, excess_total_return, alpha=0.3, 
                        color='lightblue', label='Dividend Impact')
        
        ax3.set_title('NVII vs NVDA: Excess Returns (Total vs Capital Only)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Excess Return Points', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Add summary statistics box
        final_total_excess = excess_total_return.iloc[-1]
        final_price_excess = excess_price_return.iloc[-1]
        dividend_contribution = final_total_excess - final_price_excess
        
        stats_text = f'Final Total Excess: {final_total_excess:.1f}\nFinal Price Excess: {final_price_excess:.1f}\nDividend Contribution: {dividend_contribution:.1f}'
        ax3.text(0.02, 0.95, stats_text, transform=ax3.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        return plt

def main():
    """
    Main execution function
    """
    print("=== NVII vs NVDA Total Returns Comparison ===\n")
    
    # Initialize comparator - using a broader date range to capture more data
    comparator = TotalReturnsComparator(start_date='2021-01-01')
    
    try:
        # Fetch data
        comparator.fetch_data()
        
        # Calculate returns
        results = comparator.calculate_total_returns()
        
        # Generate analysis
        analysis_text = comparator.generate_comparison_analysis()
        
        # Save analysis to docs
        output_path = '/home/kafka/projects/NVII/docs/nvii_nvda_returns_comparison.md'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(analysis_text)
        
        print(f"Analysis saved to: {output_path}")
        
        # Create and save visualization
        plt_obj = comparator.create_visualization()
        if plt_obj:
            chart_path = '/home/kafka/projects/NVII/docs/nvii_nvda_performance_chart.png'
            plt_obj.savefig(chart_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {chart_path}")
        
        # Export to CSV
        csv_files = comparator.export_to_csv()
        print("\nCSV files generated:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
        
        # Print summary to console
        print("\n=== SUMMARY ===")
        nvii = results.get('NVII', {})
        nvda = results.get('NVDA', {})
        
        print(f"NVII Total Return: {nvii.get('total_return', 0):.2%}")
        print(f"NVDA Total Return: {nvda.get('total_return', 0):.2%}")
        print(f"NVII Annualized: {nvii.get('annualized_total_return', 0):.2%}")
        print(f"NVDA Annualized: {nvda.get('annualized_total_return', 0):.2%}")
        
        performance_diff = nvii.get('total_return', 0) - nvda.get('total_return', 0)
        if performance_diff > 0:
            print(f"\n🟢 NVII outperformed NVDA by {performance_diff:.2%}")
        else:
            print(f"\n🔴 NVDA outperformed NVII by {-performance_diff:.2%}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()