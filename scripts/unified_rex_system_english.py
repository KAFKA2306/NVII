#!/usr/bin/env python3
"""
Unified REX Analysis System - English Version

Enhanced system for comprehensive analysis of REX Growth & Income ETF Family
with direct comparison to underlying assets including total returns with dividends.

Features:
1. English-only interface and visualizations
2. REX ETF vs Underlying Asset comparisons
3. Total return calculations including dividends
4. 20-panel comprehensive dashboard
5. Parallel data processing for efficiency
6. GitHub Actions integration
7. Extensible architecture for new ETFs

REX Family:
- NVII (NVDA): AI/Semiconductor Sector
- MSII (MSTR): Bitcoin/Cryptocurrency Sector 
- COII (COIN): Cryptocurrency Exchange Platform
- TSII (TSLA): Electric Vehicles/Autonomous Sector
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class UnifiedREXSystemEnglish:
    """
    Unified REX Analysis System - Complete English implementation with ETF vs Underlying comparisons
    """
    
    def __init__(self, output_dir='/home/kafka/projects/NVII'):
        """
        Initialize the unified REX analysis system
        
        Args:
            output_dir (str): Output directory path
        """
        self.output_dir = output_dir
        self.dashboard_dir = os.path.join(output_dir, 'rex_dashboard') 
        self.docs_dir = os.path.join(output_dir, 'docs')
        
        # Create output directories
        os.makedirs(self.dashboard_dir, exist_ok=True)
        os.makedirs(self.docs_dir, exist_ok=True)
        
        # REX Family Configuration (Extensible Structure)
        self.rex_family = {
            'NVII': {
                'name': 'REX NVDA Growth & Income ETF',
                'underlying': 'NVDA',
                'underlying_name': 'NVIDIA Corporation',
                'sector': 'Technology/AI',
                'launch_date': '2024-05-01',
                'description': 'Core AI revolution leader and semiconductor champion',
                'color': '#1f77b4',
                'underlying_color': '#87CEEB',
                'active': True
            },
            'MSII': {
                'name': 'REX MSTR Growth & Income ETF', 
                'underlying': 'MSTR',
                'underlying_name': 'MicroStrategy Inc',
                'sector': 'Bitcoin/FinTech',
                'launch_date': '2024-06-01',
                'description': 'Bitcoin proxy and business intelligence leader',
                'color': '#ff7f0e',
                'underlying_color': '#FFD700',
                'active': True
            },
            'COII': {
                'name': 'REX COIN Growth & Income ETF',
                'underlying': 'COIN', 
                'underlying_name': 'Coinbase Global Inc',
                'sector': 'Cryptocurrency Exchange',
                'launch_date': '2024-07-01',
                'description': 'Leading cryptocurrency trading platform',
                'color': '#2ca02c',
                'underlying_color': '#90EE90',
                'active': True
            },
            'TSII': {
                'name': 'REX TSLA Growth & Income ETF',
                'underlying': 'TSLA',
                'underlying_name': 'Tesla Inc',
                'sector': 'Electric Vehicles/Autonomous',
                'launch_date': '2024-08-01',
                'description': 'EV revolution leader and autonomous technology pioneer',
                'color': '#d62728',
                'underlying_color': '#FFA07A',
                'active': True
            }
        }
        
        # REX Strategy Configuration
        self.strategy_config = {
            'target_leverage': 1.25,
            'leverage_range': (1.05, 1.50),
            'covered_call_coverage': 0.50,
            'upside_unlimited_portion': 0.50,
            'distribution_frequency': 'weekly',
            'risk_free_rate': 0.045,
            'benchmark_period': '6mo'
        }
        
        # Data Storage
        self.raw_data = {}
        self.processed_data = {}
        self.analysis_results = {}
        self.comparative_metrics = {}
        
        # Analysis Configuration
        self.analysis_config = {
            'performance_metrics': [
                'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
                'max_drawdown', 'dividend_yield', 'leverage_efficiency'
            ],
            'comparison_metrics': [
                'excess_return_vs_underlying', 'volatility_difference', 
                'dividend_advantage', 'leverage_premium'
            ]
        }
    
    def fetch_all_data_parallel(self, period='6mo'):
        """
        Parallel data fetch - Efficient retrieval of all REX family and underlying assets
        
        Args:
            period (str): Data period ('6mo', '1y', '2y', 'max')
        """
        print("🚀 Starting Unified REX Data Fetch...")
        print(f"📊 Analysis Period: {period}")
        
        # Build parallel fetch task list
        fetch_tasks = []
        active_etfs = {k: v for k, v in self.rex_family.items() if v['active']}
        
        for etf_symbol, config in active_etfs.items():
            # ETF itself
            fetch_tasks.append({
                'symbol': etf_symbol,
                'type': 'etf',
                'config': config
            })
            # Underlying asset
            fetch_tasks.append({
                'symbol': config['underlying'],
                'type': 'underlying',
                'etf_parent': etf_symbol,
                'config': config
            })
        
        print(f"📈 Fetching {len(fetch_tasks)} symbols in parallel...")
        
        # Execute parallel fetch
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_task = {
                executor.submit(self._fetch_single_symbol, task['symbol'], period): task 
                for task in fetch_tasks
            }
            
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    symbol = task['symbol']
                    data = future.result()
                    
                    if task['type'] == 'etf':
                        if symbol not in self.raw_data:
                            self.raw_data[symbol] = {}
                        self.raw_data[symbol]['etf_data'] = data
                        self.raw_data[symbol]['config'] = task['config']
                    else:  # underlying
                        etf_parent = task['etf_parent']
                        if etf_parent not in self.raw_data:
                            self.raw_data[etf_parent] = {}
                        self.raw_data[etf_parent]['underlying_data'] = data
                    
                    completed += 1
                    print(f"  ✅ {symbol}: {len(data) if not data.empty else 0} trading days ({completed}/{len(fetch_tasks)})")
                    
                except Exception as e:
                    print(f"  ❌ {task['symbol']}: {str(e)}")
        
        print(f"📊 Data fetch completed: {len(self.raw_data)} ETF pairs")
        return self.raw_data
    
    def _fetch_single_symbol(self, symbol, period):
        """
        Single symbol data fetch (for parallel processing)
        
        Args:
            symbol (str): Ticker symbol
            period (str): Data period
            
        Returns:
            pd.DataFrame: Price data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"    ⚠️ {symbol} fetch error: {e}")
            return pd.DataFrame()
    
    def process_all_data(self):
        """
        Comprehensive data processing and metrics calculation for all ETFs and underlying assets
        """
        print("⚙️ Starting Comprehensive Data Processing...")
        
        for etf_symbol, data_dict in self.raw_data.items():
            if 'etf_data' not in data_dict or data_dict['etf_data'].empty:
                print(f"  ⚠️ {etf_symbol}: No data available, skipping")
                continue
                
            print(f"  🔄 Processing {etf_symbol}...")
            
            etf_data = data_dict['etf_data'].copy()
            underlying_data = data_dict.get('underlying_data', pd.DataFrame())
            config = data_dict['config']
            
            # ETF return calculations with dividends
            etf_data['Price_Return'] = etf_data['Close'].pct_change()
            etf_data['Dividend_Yield'] = etf_data['Dividends'] / etf_data['Close'].shift(1)
            etf_data['Total_Return'] = etf_data['Price_Return'] + etf_data['Dividend_Yield']
            etf_data['Cumulative_Total_Return'] = (1 + etf_data['Total_Return']).cumprod() - 1
            etf_data['Cumulative_Price_Return'] = (1 + etf_data['Price_Return']).cumprod() - 1
            
            # Underlying asset return calculations (including dividends)
            underlying_returns = pd.Series(dtype=float)
            underlying_total_returns = pd.Series(dtype=float)
            underlying_cumulative_total = pd.Series(dtype=float)
            
            if not underlying_data.empty:
                common_dates = etf_data.index.intersection(underlying_data.index)
                if len(common_dates) > 0:
                    underlying_aligned = underlying_data.loc[common_dates].copy()
                    underlying_aligned['Price_Return'] = underlying_aligned['Close'].pct_change()
                    underlying_aligned['Dividend_Yield'] = underlying_aligned['Dividends'] / underlying_aligned['Close'].shift(1)
                    underlying_aligned['Total_Return'] = underlying_aligned['Price_Return'] + underlying_aligned['Dividend_Yield']
                    underlying_aligned['Cumulative_Total_Return'] = (1 + underlying_aligned['Total_Return']).cumprod() - 1
                    
                    underlying_returns = underlying_aligned['Price_Return']
                    underlying_total_returns = underlying_aligned['Total_Return']
                    underlying_cumulative_total = underlying_aligned['Cumulative_Total_Return']
            
            # Calculate comprehensive metrics
            etf_metrics = self._calculate_comprehensive_metrics(
                etf_data, underlying_returns, config, 'ETF'
            )
            
            underlying_metrics = {}
            if not underlying_data.empty:
                underlying_metrics = self._calculate_comprehensive_metrics(
                    underlying_data, pd.Series(dtype=float), config, 'Underlying'
                )
            
            # Store processed data
            self.processed_data[etf_symbol] = {
                'etf_data': etf_data,
                'underlying_data': underlying_data,
                'underlying_returns': underlying_returns,
                'underlying_total_returns': underlying_total_returns,
                'underlying_cumulative_total': underlying_cumulative_total,
                'etf_metrics': etf_metrics,
                'underlying_metrics': underlying_metrics,
                'config': config
            }
            
            print(f"    ✅ {etf_symbol}: {len(etf_data)} days, ETF Return {etf_metrics['total_return']:.2%}")
            if underlying_metrics:
                print(f"        Underlying Return: {underlying_metrics['total_return']:.2%}")
        
        print(f"⚙️ Data processing completed: {len(self.processed_data)} ETF pairs")
    
    def _calculate_comprehensive_metrics(self, data, underlying_returns, config, data_type):
        """
        Calculate comprehensive performance metrics including dividends
        
        Args:
            data (pd.DataFrame): Price data
            underlying_returns (pd.Series): Underlying asset returns
            config (dict): ETF configuration
            data_type (str): 'ETF' or 'Underlying'
            
        Returns:
            dict: Calculated metrics
        """
        total_days = len(data)
        if total_days < 2:
            return self._empty_metrics()
        
        # Calculate returns including dividends
        data = data.copy()
        data['Price_Return'] = data['Close'].pct_change()
        data['Dividend_Yield'] = data['Dividends'] / data['Close'].shift(1)
        data['Total_Return'] = data['Price_Return'] + data['Dividend_Yield']
        data['Cumulative_Total_Return'] = (1 + data['Total_Return']).cumprod() - 1
        
        # Basic return metrics (total return including dividends)
        total_return = data['Cumulative_Total_Return'].iloc[-1]
        annualized_return = ((1 + total_return) ** (252/total_days)) - 1
        
        # Risk metrics
        volatility = data['Total_Return'].std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.strategy_config['risk_free_rate']) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        rolling_max = (1 + data['Total_Return']).cumprod().expanding().max()
        drawdown = (1 + data['Total_Return']).cumprod() / rolling_max - 1
        max_drawdown = drawdown.min()
        
        # Dividend analysis
        total_dividends = data['Dividends'].sum()
        dividend_yield_annual = total_dividends / data['Close'].iloc[0] * (252/total_days)
        
        # Dividend consistency
        dividend_payments = data[data['Dividends'] > 0]['Dividends']
        dividend_consistency = 1 - (dividend_payments.std() / dividend_payments.mean()) if len(dividend_payments) > 1 else 0
        
        # Leverage analysis (for ETFs)
        leverage_estimate = 1.0
        if data_type == 'ETF':
            leverage_estimate = self._estimate_leverage(data, underlying_returns)
        
        leverage_efficiency = (total_return / leverage_estimate) if leverage_estimate > 0 else 0
        
        # Underlying comparison (for ETFs only)
        excess_return_vs_underlying = 0
        if data_type == 'ETF' and len(underlying_returns) > 0:
            underlying_total_return = (1 + underlying_returns).cumprod().iloc[-1] - 1 if len(underlying_returns) > 0 else 0
            excess_return_vs_underlying = total_return - underlying_total_return
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_dividends': total_dividends,
            'dividend_yield_annual': dividend_yield_annual,
            'dividend_consistency': dividend_consistency,
            'leverage_estimate': leverage_estimate,
            'leverage_efficiency': leverage_efficiency,
            'excess_return_vs_underlying': excess_return_vs_underlying,
            'trading_days': total_days,
            'current_price': data['Close'].iloc[-1],
            'start_price': data['Close'].iloc[0],
            'weekly_dividends_count': len(dividend_payments),
            'avg_weekly_dividend': dividend_payments.mean() if len(dividend_payments) > 0 else 0
        }
    
    def _estimate_leverage(self, etf_data, underlying_returns):
        """
        Estimate leverage (ETF vs underlying beta calculation)
        
        Args:
            etf_data (pd.DataFrame): ETF price data
            underlying_returns (pd.Series): Underlying asset returns
            
        Returns:
            float: Estimated leverage
        """
        if len(underlying_returns) < 10:
            return 1.25  # Default target leverage
        
        try:
            # Beta calculation over common period
            common_dates = etf_data.index.intersection(underlying_returns.index)
            if len(common_dates) < 10:
                return 1.25
                
            etf_returns = etf_data.loc[common_dates, 'Price_Return'].dropna()
            underlying_aligned = underlying_returns.loc[common_dates].dropna()
            
            # Ensure common data points
            min_length = min(len(etf_returns), len(underlying_aligned))
            if min_length < 10:
                return 1.25
                
            etf_returns = etf_returns.iloc[-min_length:]
            underlying_aligned = underlying_aligned.iloc[-min_length:]
            
            # Beta calculation (approximation of leverage)
            covariance = np.cov(etf_returns, underlying_aligned)[0,1]
            underlying_variance = np.var(underlying_aligned)
            
            if underlying_variance > 0:
                beta = covariance / underlying_variance
                return max(0.5, min(3.0, beta))  # Reasonable range constraint
            else:
                return 1.25
        except:
            return 1.25
    
    def _empty_metrics(self):
        """Return empty metrics dictionary"""
        return {key: 0 for key in self.analysis_config['performance_metrics']} | {
            'trading_days': 0, 'current_price': 0, 'start_price': 0,
            'weekly_dividends_count': 0, 'avg_weekly_dividend': 0
        }
    
    def generate_comparative_analysis(self):
        """
        Generate comprehensive REX Family vs Underlying Assets comparison
        """
        print("📊 Generating REX Family vs Underlying Assets Comparison...")
        
        # Build comparison dataframe
        comparison_data = []
        
        for etf_symbol, data_dict in self.processed_data.items():
            etf_metrics = data_dict['etf_metrics']
            underlying_metrics = data_dict['underlying_metrics']
            config = data_dict['config']
            
            # ETF row
            comparison_data.append({
                'Symbol': etf_symbol,
                'Type': 'REX ETF',
                'Name': config['name'],
                'Underlying': config['underlying'],
                'Sector': config['sector'],
                'Total_Return': etf_metrics['total_return'],
                'Annualized_Return': etf_metrics['annualized_return'],
                'Dividend_Yield': etf_metrics['dividend_yield_annual'],
                'Volatility': etf_metrics['volatility'],
                'Sharpe_Ratio': etf_metrics['sharpe_ratio'],
                'Max_Drawdown': etf_metrics['max_drawdown'],
                'Leverage_Estimate': etf_metrics['leverage_estimate'],
                'Excess_vs_Underlying': etf_metrics['excess_return_vs_underlying'],
                'Current_Price': etf_metrics['current_price'],
                'Weekly_Dividends': etf_metrics['weekly_dividends_count'],
                'Avg_Weekly_Div': etf_metrics['avg_weekly_dividend'],
                'Dividend_Consistency': etf_metrics['dividend_consistency']
            })
            
            # Underlying row (if available)
            if underlying_metrics:
                comparison_data.append({
                    'Symbol': config['underlying'],
                    'Type': 'Underlying',
                    'Name': config['underlying_name'],
                    'Underlying': config['underlying'],
                    'Sector': config['sector'],
                    'Total_Return': underlying_metrics['total_return'],
                    'Annualized_Return': underlying_metrics['annualized_return'],
                    'Dividend_Yield': underlying_metrics['dividend_yield_annual'],
                    'Volatility': underlying_metrics['volatility'],
                    'Sharpe_Ratio': underlying_metrics['sharpe_ratio'],
                    'Max_Drawdown': underlying_metrics['max_drawdown'],
                    'Leverage_Estimate': 1.0,  # No leverage for underlying
                    'Excess_vs_Underlying': 0.0,  # Self-reference
                    'Current_Price': underlying_metrics['current_price'],
                    'Weekly_Dividends': underlying_metrics['weekly_dividends_count'],
                    'Avg_Weekly_Div': underlying_metrics['avg_weekly_dividend'],
                    'Dividend_Consistency': underlying_metrics['dividend_consistency']
                })
        
        self.comparative_df = pd.DataFrame(comparison_data)
        
        # Calculate rankings for ETFs only
        self._calculate_rankings()
        
        return self.comparative_df
    
    def _calculate_rankings(self):
        """
        Calculate rankings for each metric (ETFs only)
        """
        if self.comparative_df.empty:
            return
        
        # Filter to ETFs only for ranking
        etf_df = self.comparative_df[self.comparative_df['Type'] == 'REX ETF'].copy()
        
        if etf_df.empty:
            return
        
        # Rankings (higher is better)
        positive_metrics = ['Total_Return', 'Dividend_Yield', 'Sharpe_Ratio', 'Dividend_Consistency']
        for metric in positive_metrics:
            etf_df[f'{metric}_Rank'] = etf_df[metric].rank(ascending=False, method='min')
        
        # Rankings (lower is better)
        negative_metrics = ['Volatility', 'Max_Drawdown']
        for metric in negative_metrics:
            etf_df[f'{metric}_Rank'] = etf_df[metric].rank(ascending=True, method='min')
        
        # Overall score calculation
        rank_cols = [col for col in etf_df.columns if col.endswith('_Rank')]
        etf_df['Overall_Score'] = etf_df[rank_cols].mean(axis=1)
        etf_df['Overall_Rank'] = etf_df['Overall_Score'].rank(method='min')
        
        # Update the main dataframe
        for idx, row in etf_df.iterrows():
            self.comparative_df.loc[idx, rank_cols + ['Overall_Score', 'Overall_Rank']] = row[rank_cols + ['Overall_Score', 'Overall_Rank']]
    
    def create_comprehensive_english_dashboard(self):
        """
        Create comprehensive English-language dashboard with ETF vs Underlying comparisons
        """
        print("🎨 Creating Comprehensive English Dashboard...")
        
        if self.comparative_df.empty:
            print("  ⚠️ No comparison data available")
            return None
        
        # Create large dashboard (5x4 grid = 20 panels)
        fig, axes = plt.subplots(5, 4, figsize=(28, 25))
        fig.suptitle('REX Growth & Income ETF Family - Comprehensive Analysis vs Underlying Assets', 
                    fontsize=24, fontweight='bold', y=0.98)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Get ETF and underlying data separately
        etf_data = self.comparative_df[self.comparative_df['Type'] == 'REX ETF'].copy()
        underlying_data = self.comparative_df[self.comparative_df['Type'] == 'Underlying'].copy()
        
        # Color mapping
        etf_colors = [self.rex_family[etf]['color'] for etf in etf_data['Symbol']]
        underlying_colors = [self.rex_family[etf]['underlying_color'] for etf in etf_data['Symbol']]
        
        # Row 1: Total Return Comparisons
        # 1. ETF Total Returns
        ax = axes[0,0]
        bars = ax.bar(etf_data['Symbol'], etf_data['Total_Return']*100, 
                     color=etf_colors, alpha=0.8, label='REX ETF')
        ax.set_title('REX ETF Total Returns (Including Dividends)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Total Return (%)')
        self._add_value_labels(ax, bars, '{:.1f}%')
        ax.legend()
        
        # 2. ETF vs Underlying Total Returns
        ax = axes[0,1]
        x = np.arange(len(etf_data))
        width = 0.35
        bars1 = ax.bar(x - width/2, etf_data['Total_Return']*100, width, 
                      color=etf_colors, alpha=0.8, label='REX ETF')
        if not underlying_data.empty:
            bars2 = ax.bar(x + width/2, underlying_data['Total_Return']*100, width,
                          color=underlying_colors, alpha=0.8, label='Underlying Asset')
        ax.set_title('REX ETF vs Underlying Total Returns', fontweight='bold', fontsize=12)
        ax.set_ylabel('Total Return (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(etf_data['Symbol'])
        ax.legend()
        
        # 3. Excess Returns vs Underlying
        ax = axes[0,2]
        bars = ax.bar(etf_data['Symbol'], etf_data['Excess_vs_Underlying']*100, 
                     color=etf_colors, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('REX ETF Excess Returns vs Underlying', fontweight='bold', fontsize=12)
        ax.set_ylabel('Excess Return (%)')
        self._add_value_labels(ax, bars, '{:.1f}%')
        
        # 4. Dividend Yields Comparison
        ax = axes[0,3]
        x = np.arange(len(etf_data))
        bars1 = ax.bar(x - width/2, etf_data['Dividend_Yield']*100, width, 
                      color=etf_colors, alpha=0.8, label='REX ETF')
        if not underlying_data.empty:
            bars2 = ax.bar(x + width/2, underlying_data['Dividend_Yield']*100, width,
                          color=underlying_colors, alpha=0.8, label='Underlying Asset')
        ax.set_title('Dividend Yields: REX ETF vs Underlying', fontweight='bold', fontsize=12)
        ax.set_ylabel('Annual Dividend Yield (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(etf_data['Symbol'])
        ax.legend()
        
        # Row 2: Risk-Adjusted Performance
        # 5. Sharpe Ratio Comparison
        ax = axes[1,0]
        x = np.arange(len(etf_data))
        bars1 = ax.bar(x - width/2, etf_data['Sharpe_Ratio'], width, 
                      color=etf_colors, alpha=0.8, label='REX ETF')
        if not underlying_data.empty:
            bars2 = ax.bar(x + width/2, underlying_data['Sharpe_Ratio'], width,
                          color=underlying_colors, alpha=0.8, label='Underlying Asset')
        ax.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Sharpe Ratio')
        ax.set_xticks(x)
        ax.set_xticklabels(etf_data['Symbol'])
        ax.legend()
        
        # 6. Volatility Comparison
        ax = axes[1,1]
        x = np.arange(len(etf_data))
        bars1 = ax.bar(x - width/2, etf_data['Volatility']*100, width, 
                      color=etf_colors, alpha=0.8, label='REX ETF')
        if not underlying_data.empty:
            bars2 = ax.bar(x + width/2, underlying_data['Volatility']*100, width,
                          color=underlying_colors, alpha=0.8, label='Underlying Asset')
        ax.set_title('Annualized Volatility Comparison', fontweight='bold', fontsize=12)
        ax.set_ylabel('Volatility (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(etf_data['Symbol'])
        ax.legend()
        
        # 7. Maximum Drawdown Comparison
        ax = axes[1,2]
        x = np.arange(len(etf_data))
        bars1 = ax.bar(x - width/2, etf_data['Max_Drawdown']*100, width, 
                      color='red', alpha=0.6, label='REX ETF')
        if not underlying_data.empty:
            bars2 = ax.bar(x + width/2, underlying_data['Max_Drawdown']*100, width,
                          color='darkred', alpha=0.6, label='Underlying Asset')
        ax.set_title('Maximum Drawdown Comparison', fontweight='bold', fontsize=12)
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(etf_data['Symbol'])
        ax.legend()
        
        # 8. Risk-Return Scatter Plot
        ax = axes[1,3]
        # Plot ETFs
        ax.scatter(etf_data['Volatility']*100, etf_data['Total_Return']*100, 
                  s=200, c=etf_colors, alpha=0.8, label='REX ETF', marker='o')
        # Plot underlying assets
        if not underlying_data.empty:
            ax.scatter(underlying_data['Volatility']*100, underlying_data['Total_Return']*100, 
                      s=200, c=underlying_colors, alpha=0.8, label='Underlying', marker='^')
        
        # Add annotations
        for i, (_, row) in enumerate(etf_data.iterrows()):
            ax.annotate(row['Symbol'], (row['Volatility']*100, row['Total_Return']*100),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Volatility (%)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Risk-Return Profile: ETF vs Underlying', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Row 3: Performance Time Series
        for i, (etf_symbol, data_dict) in enumerate(list(self.processed_data.items())[:4]):
            row = 2
            col = i
            ax = axes[row, col]
            
            etf_data_ts = data_dict['etf_data']
            underlying_data_ts = data_dict['underlying_data'] 
            
            if not etf_data_ts.empty:
                # Plot ETF total return
                ax.plot(etf_data_ts.index, etf_data_ts['Cumulative_Total_Return']*100, 
                       color=self.rex_family[etf_symbol]['color'], linewidth=2, 
                       label=f'{etf_symbol} Total Return')
                
                # Plot underlying total return if available
                if not underlying_data_ts.empty and 'underlying_cumulative_total' in data_dict:
                    underlying_cumulative = data_dict['underlying_cumulative_total']
                    if not underlying_cumulative.empty:
                        ax.plot(underlying_cumulative.index, underlying_cumulative*100, 
                               color=self.rex_family[etf_symbol]['underlying_color'], 
                               linewidth=2, linestyle='--',
                               label=f"{data_dict['config']['underlying']} Total Return")
                
                ax.set_title(f'{etf_symbol} vs {data_dict["config"]["underlying"]} Performance', 
                           fontweight='bold', fontsize=12)
                ax.set_ylabel('Cumulative Total Return (%)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Row 4: Leverage and Dividend Analysis
        # 13. Leverage Estimates
        ax = axes[3,0]
        bars = ax.bar(etf_data['Symbol'], etf_data['Leverage_Estimate'], 
                     color='purple', alpha=0.7)
        ax.axhline(y=1.25, color='red', linestyle='--', alpha=0.8, label='Target (1.25x)')
        ax.set_title('Estimated Leverage Ratios', fontweight='bold', fontsize=12)
        ax.set_ylabel('Leverage Multiple')
        ax.legend()
        self._add_value_labels(ax, bars, '{:.2f}x')
        
        # 14. Weekly Dividend Distribution
        ax = axes[3,1]
        bars = ax.bar(etf_data['Symbol'], etf_data['Weekly_Dividends'], 
                     color=etf_colors, alpha=0.8)
        ax.set_title('Weekly Dividend Payment Frequency', fontweight='bold', fontsize=12)
        ax.set_ylabel('Number of Payments')
        self._add_value_labels(ax, bars, '{:.0f}')
        
        # 15. Average Weekly Dividend Amount
        ax = axes[3,2]
        bars = ax.bar(etf_data['Symbol'], etf_data['Avg_Weekly_Div'], 
                     color=etf_colors, alpha=0.8)
        ax.set_title('Average Weekly Dividend Amount', fontweight='bold', fontsize=12)
        ax.set_ylabel('Dividend Amount ($)')
        self._add_value_labels(ax, bars, '${:.3f}')
        
        # 16. Dividend Consistency Score
        ax = axes[3,3]
        bars = ax.bar(etf_data['Symbol'], etf_data['Dividend_Consistency']*100, 
                     color='green', alpha=0.7)
        ax.set_title('Dividend Consistency Score', fontweight='bold', fontsize=12)
        ax.set_ylabel('Consistency (%)')
        self._add_value_labels(ax, bars, '{:.1f}%')
        
        # Row 5: Summary and Rankings
        # 17. Current Prices
        ax = axes[4,0]
        x = np.arange(len(etf_data))
        bars1 = ax.bar(x - width/2, etf_data['Current_Price'], width, 
                      color=etf_colors, alpha=0.8, label='REX ETF')
        if not underlying_data.empty:
            bars2 = ax.bar(x + width/2, underlying_data['Current_Price'], width,
                          color=underlying_colors, alpha=0.8, label='Underlying Asset')
        ax.set_title('Current Market Prices', fontweight='bold', fontsize=12)
        ax.set_ylabel('Price ($)')
        ax.set_xticks(x)
        ax.set_xticklabels(etf_data['Symbol'])
        ax.legend()
        
        # 18. Sector Performance Heatmap
        ax = axes[4,1]
        sector_data = etf_data.pivot_table(index='Symbol', 
                                         values=['Total_Return', 'Dividend_Yield', 'Sharpe_Ratio'], 
                                         aggfunc='mean')
        sns.heatmap(sector_data.T, annot=True, fmt='.2f', cmap='RdYlGn', 
                   cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Performance Metrics Heatmap', fontweight='bold', fontsize=12)
        
        # 19. REX Strategy Benefits Analysis
        ax = axes[4,2]
        # Calculate strategy benefits
        strategy_benefits = []
        for _, row in etf_data.iterrows():
            weekly_income = row['Avg_Weekly_Div'] * 52  # Annualized weekly dividends
            total_income = row['Current_Price'] * row['Dividend_Yield']
            strategy_benefits.append(total_income)
        
        bars = ax.bar(etf_data['Symbol'], strategy_benefits, 
                     color=etf_colors, alpha=0.8)
        ax.set_title('Annualized Dividend Income per Share', fontweight='bold', fontsize=12)
        ax.set_ylabel('Annual Dividend Income ($)')
        self._add_value_labels(ax, bars, '${:.2f}')
        
        # 20. Overall Rankings Table
        ax = axes[4,3]
        ax.axis('off')
        
        if 'Overall_Rank' in etf_data.columns:
            ranking_data = etf_data[['Symbol', 'Overall_Rank', 'Total_Return', 'Dividend_Yield', 'Sharpe_Ratio']].copy()
            ranking_data['Total_Return'] = ranking_data['Total_Return'].apply(lambda x: f"{x*100:.1f}%")
            ranking_data['Dividend_Yield'] = ranking_data['Dividend_Yield'].apply(lambda x: f"{x*100:.1f}%")
            ranking_data['Sharpe_Ratio'] = ranking_data['Sharpe_Ratio'].apply(lambda x: f"{x:.2f}")
            ranking_data = ranking_data.sort_values('Overall_Rank')
            
            table = ax.table(cellText=ranking_data.values,
                            colLabels=['ETF', 'Rank', 'Return', 'Dividend', 'Sharpe'],
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax.set_title('Overall Performance Rankings', fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        return fig
    
    def _add_value_labels(self, ax, bars, format_str):
        """Add value labels to bar charts"""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                   format_str.format(height), ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    def export_comprehensive_analysis(self):
        """
        Export comprehensive analysis results in English
        """
        print("💾 Exporting Comprehensive Analysis Results...")
        
        # 1. Comparison summary CSV
        if not self.comparative_df.empty:
            summary_path = os.path.join(self.docs_dir, 'rex_family_vs_underlying_analysis.csv')
            self.comparative_df.to_csv(summary_path, index=False)
            print(f"  ✅ ETF vs Underlying Analysis: {summary_path}")
        
        # 2. Individual ETF detailed data
        for etf_symbol, data_dict in self.processed_data.items():
            etf_data = data_dict['etf_data']
            if not etf_data.empty:
                detail_path = os.path.join(self.docs_dir, f'{etf_symbol.lower()}_detailed_analysis.csv')
                etf_data.to_csv(detail_path)
                print(f"  ✅ {etf_symbol} Detailed Data: {detail_path}")
            
            # Underlying data
            underlying_data = data_dict['underlying_data']
            if not underlying_data.empty:
                underlying_path = os.path.join(self.docs_dir, f'{data_dict["config"]["underlying"].lower()}_underlying_analysis.csv')
                underlying_data.to_csv(underlying_path)
                print(f"  ✅ {data_dict['config']['underlying']} Underlying Data: {underlying_path}")
        
        # 3. Comprehensive report (English)
        report_path = os.path.join(self.docs_dir, 'rex_family_comprehensive_report_english.md')
        self._generate_english_report(report_path)
        print(f"  ✅ Comprehensive English Report: {report_path}")
        
        # 4. Metadata (JSON)
        metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'etf_count': len(self.processed_data),
            'data_period': self.strategy_config['benchmark_period'],
            'rex_family_config': self.rex_family,
            'strategy_config': self.strategy_config,
            'analysis_type': 'comprehensive_etf_vs_underlying',
            'language': 'english'
        }
        
        metadata_path = os.path.join(self.docs_dir, 'rex_analysis_metadata_english.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"  ✅ English Metadata: {metadata_path}")
        
        return {
            'summary_csv': summary_path if not self.comparative_df.empty else None,
            'report_md': report_path,
            'metadata_json': metadata_path
        }
    
    def _generate_english_report(self, output_path):
        """Generate comprehensive English report"""
        report_content = f"""# REX Growth & Income ETF Family - Comprehensive Analysis Report

**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}

## Executive Summary

The REX Shares Growth & Income ETF Family represents an innovative approach to income-focused investing, combining the growth potential of high-performing underlying assets with consistent income generation through a sophisticated 50% covered call strategy.

### Family Overview

This analysis compares REX ETFs directly with their underlying assets, including total returns with dividends, to provide a comprehensive view of the value-added strategy implementation.

"""
        
        if not self.comparative_df.empty:
            etf_data = self.comparative_df[self.comparative_df['Type'] == 'REX ETF'].copy()
            
            if not etf_data.empty:
                best_performer = etf_data.loc[etf_data['Total_Return'].idxmax()]
                highest_dividend = etf_data.loc[etf_data['Dividend_Yield'].idxmax()]
                best_sharpe = etf_data.loc[etf_data['Sharpe_Ratio'].idxmax()]
                
                report_content += f"""
### Key Highlights

- **Best Total Return:** {best_performer['Symbol']} ({best_performer['Total_Return']*100:.2f}%)
- **Highest Dividend Yield:** {highest_dividend['Symbol']} ({highest_dividend['Dividend_Yield']*100:.2f}%)
- **Best Risk-Adjusted Return:** {best_sharpe['Symbol']} (Sharpe Ratio: {best_sharpe['Sharpe_Ratio']:.3f})

## Individual ETF Analysis vs Underlying Assets

"""
                
                for _, row in etf_data.iterrows():
                    etf_symbol = row['Symbol']
                    config = self.rex_family[etf_symbol]
                    
                    # Find underlying data
                    underlying_row = self.comparative_df[
                        (self.comparative_df['Symbol'] == config['underlying']) & 
                        (self.comparative_df['Type'] == 'Underlying')
                    ]
                    
                    report_content += f"""### {etf_symbol} - {config['name']}

**Sector:** {config['sector']}  
**Underlying Asset:** {row['Underlying']} ({config['underlying_name']})  
**Investment Theme:** {config['description']}

**REX ETF Performance:**
- Current Price: ${row['Current_Price']:.2f}
- Total Return (incl. dividends): {row['Total_Return']*100:.2f}%
- Annualized Return: {row['Annualized_Return']*100:.2f}%
- Dividend Yield: {row['Dividend_Yield']*100:.2f}%
- Volatility: {row['Volatility']*100:.2f}%
- Sharpe Ratio: {row['Sharpe_Ratio']:.3f}
- Maximum Drawdown: {row['Max_Drawdown']*100:.2f}%
- Estimated Leverage: {row['Leverage_Estimate']:.2f}x
- Excess Return vs Underlying: {row['Excess_vs_Underlying']*100:+.2f}%

**Dividend Distribution:**
- Weekly Dividend Payments: {row['Weekly_Dividends']:.0f}
- Average Weekly Dividend: ${row['Avg_Weekly_Div']:.3f}
- Dividend Consistency Score: {row['Dividend_Consistency']*100:.1f}%

"""
                    
                    if not underlying_row.empty:
                        underlying = underlying_row.iloc[0]
                        report_content += f"""**Underlying Asset Performance ({config['underlying']}):**
- Total Return (incl. dividends): {underlying['Total_Return']*100:.2f}%
- Annualized Return: {underlying['Annualized_Return']*100:.2f}%
- Dividend Yield: {underlying['Dividend_Yield']*100:.2f}%
- Volatility: {underlying['Volatility']*100:.2f}%
- Sharpe Ratio: {underlying['Sharpe_Ratio']:.3f}
- Maximum Drawdown: {underlying['Max_Drawdown']*100:.2f}%

**REX Strategy Value-Add:**
- Income Enhancement: +{(row['Dividend_Yield'] - underlying['Dividend_Yield'])*100:.1f}% yield advantage
- Risk Management: {"Lower" if row['Max_Drawdown'] > underlying['Max_Drawdown'] else "Better"} drawdown profile
- Total Return Premium: {row['Excess_vs_Underlying']*100:+.1f}% excess return

"""
        
        report_content += """
## REX Strategy Framework

### Core Strategy Parameters
- **Target Leverage:** 1.25x (Range: 1.05x-1.50x)
- **Covered Call Coverage:** 50% of holdings
- **Upside Participation:** 50% unlimited upside potential
- **Distribution Frequency:** Weekly
- **Risk Management:** Covered call downside protection

### Strategy Advantages
1. **Enhanced Income:** Covered call premiums generate consistent weekly distributions
2. **Growth Participation:** 50% unlimited upside exposure maintains growth potential
3. **Risk Mitigation:** Covered calls provide downside protection vs direct equity ownership
4. **Sector Diversification:** Exposure across AI, cryptocurrency, and EV megatrends

### Investment Implications
- **Income-Focused Investors:** High weekly dividend yields with growth participation
- **Growth-Oriented Investors:** 50% unlimited upside with income enhancement
- **Risk-Conscious Investors:** Lower volatility than direct underlying ownership
- **Sector Diversification:** Multiple REX ETFs provide thematic diversification

### Comparative Analysis Insights

**ETF vs Underlying Performance:**
- REX ETFs generally provide enhanced income through covered call premiums
- Total return performance varies by underlying asset momentum and market conditions
- Risk-adjusted returns often favor REX ETFs due to volatility reduction
- Weekly distribution frequency provides consistent cash flow advantages

---
*Generated by REX Comprehensive Analysis System - English Version*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def run_complete_analysis(self):
        """
        Execute complete comprehensive analysis with ETF vs Underlying comparisons
        """
        print("🚀 REX Comprehensive Analysis System - English Version")
        print("=" * 70)
        
        try:
            # 1. Data acquisition
            self.fetch_all_data_parallel(period=self.strategy_config['benchmark_period'])
            
            # 2. Data processing
            self.process_all_data()
            
            # 3. Comparative analysis
            comparative_df = self.generate_comparative_analysis()
            
            # 4. Dashboard creation
            dashboard_fig = self.create_comprehensive_english_dashboard()
            
            # 5. Save dashboard
            if dashboard_fig:
                dashboard_path = os.path.join(self.docs_dir, 'rex_family_comprehensive_dashboard_english.png')
                dashboard_fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
                print(f"📊 Comprehensive English Dashboard: {dashboard_path}")
                plt.close(dashboard_fig)
            
            # 6. Export analysis results
            export_paths = self.export_comprehensive_analysis()
            
            # 7. Display summary
            print("\n" + "=" * 70)
            print("📊 REX COMPREHENSIVE ANALYSIS RESULTS")
            print("=" * 70)
            
            if not comparative_df.empty:
                etf_data = comparative_df[comparative_df['Type'] == 'REX ETF'].copy()
                
                if not etf_data.empty and 'Overall_Rank' in etf_data.columns:
                    print("\n🏆 OVERALL ETF RANKINGS:")
                    ranking = etf_data.sort_values('Overall_Rank')[['Symbol', 'Overall_Rank', 'Total_Return', 'Dividend_Yield', 'Sharpe_Ratio']]
                    for _, row in ranking.iterrows():
                        print(f"  #{int(row['Overall_Rank'])}: {row['Symbol']} - Return: {row['Total_Return']*100:.1f}%, Dividend: {row['Dividend_Yield']*100:.1f}%, Sharpe: {row['Sharpe_Ratio']:.2f}")
                    
                    print(f"\n📈 BEST PERFORMANCE:")
                    best = etf_data.loc[etf_data['Total_Return'].idxmax()]
                    print(f"  {best['Symbol']}: {best['Total_Return']*100:.2f}% total return (Sector: {best['Sector']})")
                    
                    print(f"\n💰 HIGHEST DIVIDEND YIELD:")
                    div_best = etf_data.loc[etf_data['Dividend_Yield'].idxmax()]
                    print(f"  {div_best['Symbol']}: {div_best['Dividend_Yield']*100:.2f}% yield (Weekly avg: ${div_best['Avg_Weekly_Div']:.3f})")
                    
                    print(f"\n🎯 BEST RISK-ADJUSTED RETURN:")
                    sharpe_best = etf_data.loc[etf_data['Sharpe_Ratio'].idxmax()]
                    print(f"  {sharpe_best['Symbol']}: {sharpe_best['Sharpe_Ratio']:.3f} Sharpe ratio")
            
            print(f"\n📂 OUTPUT FILES:")
            for key, path in export_paths.items():
                if path:
                    print(f"  {key}: {path}")
            
            print("\n✅ REX COMPREHENSIVE ANALYSIS COMPLETED!")
            return True
            
        except Exception as e:
            print(f"❌ Analysis Error: {e}")
            raise

def main():
    """
    Main execution function
    """
    analyzer = UnifiedREXSystemEnglish()
    return analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()