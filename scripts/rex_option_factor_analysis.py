#!/usr/bin/env python3
"""
REX ETF Options Factor Analysis
===============================

Advanced analysis of REX ETF price sensitivity to call option pricing factors,
focusing on vegas, volatilities, and option price changes.

This analysis examines:
1. Call option pricing sensitivity (vega) for each REX ETF
2. Volatility factor relationships
3. Option price change correlations
4. Factor decomposition of REX ETF returns
5. Greeks analysis and sensitivity mapping

Author: REX Options Analysis System
Date: August 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Black-Scholes Option Pricing
from scipy.stats import norm

class REXOptionFactorAnalyzer:
    """
    Advanced factor analysis for REX ETFs focusing on option pricing sensitivities
    """
    
    def __init__(self, output_dir='/home/kafka/projects/NVII'):
        self.output_dir = output_dir
        self.docs_dir = os.path.join(output_dir, 'docs')
        os.makedirs(self.docs_dir, exist_ok=True)
        
        # REX ETF Configuration
        self.rex_etfs = {
            'NVII': {
                'name': 'REX NVDA Growth & Income ETF',
                'underlying': 'NVDA',
                'sector': 'Technology/AI',
                'color': '#1f77b4',
                'launch_date': '2024-05-01'
            },
            'MSII': {
                'name': 'REX MSTR Growth & Income ETF',
                'underlying': 'MSTR',
                'sector': 'Bitcoin/FinTech',
                'color': '#ff7f0e',
                'launch_date': '2024-06-01'
            },
            'COII': {
                'name': 'REX COIN Growth & Income ETF',
                'underlying': 'COIN',
                'sector': 'Cryptocurrency',
                'color': '#2ca02c',
                'launch_date': '2024-07-01'
            },
            'TSII': {
                'name': 'REX TSLA Growth & Income ETF',
                'underlying': 'TSLA',
                'sector': 'Electric Vehicles',
                'color': '#d62728',
                'launch_date': '2024-08-01'
            }
        }
        
        # Options parameters
        self.risk_free_rate = 0.045  # 4.5% risk-free rate
        self.covered_call_ratio = 0.50  # 50% covered call strategy
        
        # Storage for analysis results
        self.price_data = {}
        self.option_data = {}
        self.factor_analysis = {}
        self.sensitivity_results = {}
        
    def fetch_market_data(self, period='6mo'):
        """Fetch price data for REX ETFs and underlying assets"""
        print("📈 Fetching market data for options factor analysis...")
        
        symbols = list(self.rex_etfs.keys()) + [config['underlying'] for config in self.rex_etfs.values()]
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    # Calculate returns and volatility
                    data['Returns'] = data['Close'].pct_change()
                    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
                    
                    # Rolling volatility (20-day)
                    data['Volatility_20d'] = data['Returns'].rolling(20).std() * np.sqrt(252)
                    
                    # Rolling volatility (5-day for short-term)
                    data['Volatility_5d'] = data['Returns'].rolling(5).std() * np.sqrt(252)
                    
                    self.price_data[symbol] = data
                    print(f"  ✅ {symbol}: {len(data)} trading days")
                else:
                    print(f"  ❌ {symbol}: No data available")
                    
            except Exception as e:
                print(f"  ❌ {symbol}: Error - {str(e)}")
        
        print(f"📊 Data collection completed: {len(self.price_data)} symbols")
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """Black-Scholes call option pricing"""
        if T <= 0 or sigma <= 0:
            return 0
            
        d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call_price, 0)
    
    def calculate_vega(self, S, K, T, r, sigma):
        """Calculate vega (sensitivity to volatility)"""
        if T <= 0 or sigma <= 0:
            return 0
            
        d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1) / 100  # Per 1% change in volatility
        return vega
    
    def calculate_delta(self, S, K, T, r, sigma):
        """Calculate delta (sensitivity to underlying price)"""
        if T <= 0 or sigma <= 0:
            return 0
            
        d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1)
        return delta
    
    def calculate_gamma(self, S, K, T, r, sigma):
        """Calculate gamma (rate of change of delta)"""
        if T <= 0 or sigma <= 0:
            return 0
            
        d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    
    def calculate_theta(self, S, K, T, r, sigma):
        """Calculate theta (time decay)"""
        if T <= 0 or sigma <= 0:
            return 0
            
        d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        return theta
    
    def estimate_implied_volatility(self, market_price, S, K, T, r):
        """Estimate implied volatility using numerical methods"""
        def objective(sigma):
            model_price = self.black_scholes_call(S, K, T, r, sigma)
            return abs(model_price - market_price)
        
        try:
            result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
            return result.x if result.success else 0.3  # Default to 30%
        except:
            return 0.3
    
    def calculate_option_metrics(self, etf_symbol):
        """Calculate comprehensive option metrics for a REX ETF"""
        if etf_symbol not in self.price_data:
            return None
            
        etf_data = self.price_data[etf_symbol].copy()
        underlying_symbol = self.rex_etfs[etf_symbol]['underlying']
        
        if underlying_symbol not in self.price_data:
            return None
            
        underlying_data = self.price_data[underlying_symbol]
        
        # Common dates for analysis
        common_dates = etf_data.index.intersection(underlying_data.index)
        if len(common_dates) < 20:
            return None
            
        etf_aligned = etf_data.loc[common_dates].copy()
        underlying_aligned = underlying_data.loc[common_dates].copy()
        
        # Option parameters (monthly options, ATM strikes)
        option_metrics = []
        
        for i, date in enumerate(common_dates):
            if i < 20:  # Skip first 20 days for volatility calculation
                continue
                
            S = underlying_aligned.loc[date, 'Close']  # Underlying price
            K = S  # ATM strike
            T = 30/365  # 30 days to expiration
            sigma = underlying_aligned.loc[date, 'Volatility_20d']
            
            if pd.isna(sigma) or sigma <= 0:
                continue
                
            # Calculate option Greeks
            call_price = self.black_scholes_call(S, K, T, self.risk_free_rate, sigma)
            vega = self.calculate_vega(S, K, T, self.risk_free_rate, sigma)
            delta = self.calculate_delta(S, K, T, self.risk_free_rate, sigma)
            gamma = self.calculate_gamma(S, K, T, self.risk_free_rate, sigma)
            theta = self.calculate_theta(S, K, T, self.risk_free_rate, sigma)
            
            # ETF metrics
            etf_price = etf_aligned.loc[date, 'Close']
            etf_return = etf_aligned.loc[date, 'Returns'] if not pd.isna(etf_aligned.loc[date, 'Returns']) else 0
            etf_volatility = etf_aligned.loc[date, 'Volatility_20d'] if not pd.isna(etf_aligned.loc[date, 'Volatility_20d']) else sigma
            
            # Covered call impact (50% coverage)
            covered_call_premium = call_price * self.covered_call_ratio
            covered_call_vega = vega * self.covered_call_ratio
            
            option_metrics.append({
                'Date': date,
                'ETF_Price': etf_price,
                'ETF_Return': etf_return,
                'ETF_Volatility': etf_volatility,
                'Underlying_Price': S,
                'Underlying_Volatility': sigma,
                'Call_Price': call_price,
                'Vega': vega,
                'Delta': delta,
                'Gamma': gamma,
                'Theta': theta,
                'Covered_Call_Premium': covered_call_premium,
                'Covered_Call_Vega': covered_call_vega,
                'Volatility_Spread': etf_volatility - sigma,
                'Option_Yield': covered_call_premium / S if S > 0 else 0
            })
        
        if not option_metrics:
            return None
            
        option_df = pd.DataFrame(option_metrics)
        option_df.set_index('Date', inplace=True)
        
        # Calculate option price changes and correlations
        option_df['Call_Price_Change'] = option_df['Call_Price'].pct_change()
        option_df['Vega_Change'] = option_df['Vega'].pct_change()
        option_df['Volatility_Change'] = option_df['Underlying_Volatility'].pct_change()
        
        return option_df
    
    def perform_factor_analysis(self):
        """Perform comprehensive factor analysis for all REX ETFs"""
        print("🔬 Performing options factor analysis...")
        
        for etf_symbol in self.rex_etfs.keys():
            print(f"  📊 Analyzing {etf_symbol}...")
            
            option_metrics = self.calculate_option_metrics(etf_symbol)
            if option_metrics is None:
                print(f"    ⚠️ Insufficient data for {etf_symbol}")
                continue
                
            self.option_data[etf_symbol] = option_metrics
            
            # Factor analysis
            factors = self.analyze_sensitivity_factors(etf_symbol, option_metrics)
            self.factor_analysis[etf_symbol] = factors
            
            print(f"    ✅ {etf_symbol}: {len(option_metrics)} data points analyzed")
        
        print(f"🔬 Factor analysis completed for {len(self.factor_analysis)} ETFs")
    
    def analyze_sensitivity_factors(self, etf_symbol, option_data):
        """Analyze sensitivity factors for a specific REX ETF"""
        df = option_data.dropna()
        
        if len(df) < 10:
            return {}
        
        # Calculate correlations
        correlations = {
            'etf_return_vs_call_price_change': df['ETF_Return'].corr(df['Call_Price_Change']),
            'etf_return_vs_vega': df['ETF_Return'].corr(df['Vega']),
            'etf_return_vs_volatility': df['ETF_Return'].corr(df['Underlying_Volatility']),
            'etf_return_vs_option_yield': df['ETF_Return'].corr(df['Option_Yield']),
            'vega_vs_volatility': df['Vega'].corr(df['Underlying_Volatility']),
            'call_price_vs_volatility': df['Call_Price'].corr(df['Underlying_Volatility'])
        }
        
        # Regression analysis: ETF returns vs option factors
        from sklearn.linear_model import LinearRegression
        
        # Prepare features
        X = df[['Vega', 'Underlying_Volatility', 'Option_Yield', 'Delta', 'Gamma']].fillna(0)
        y = df['ETF_Return'].fillna(0)
        
        if len(X) > 5:
            reg = LinearRegression().fit(X, y)
            factor_loadings = {
                'vega_loading': reg.coef_[0],
                'volatility_loading': reg.coef_[1],
                'option_yield_loading': reg.coef_[2],
                'delta_loading': reg.coef_[3],
                'gamma_loading': reg.coef_[4],
                'r_squared': reg.score(X, y)
            }
        else:
            factor_loadings = {}
        
        # Volatility statistics
        vol_stats = {
            'avg_etf_volatility': df['ETF_Volatility'].mean(),
            'avg_underlying_volatility': df['Underlying_Volatility'].mean(),
            'volatility_spread_mean': df['Volatility_Spread'].mean(),
            'volatility_spread_std': df['Volatility_Spread'].std(),
            'avg_vega': df['Vega'].mean(),
            'avg_option_yield': df['Option_Yield'].mean()
        }
        
        return {
            'correlations': correlations,
            'factor_loadings': factor_loadings,
            'volatility_stats': vol_stats,
            'data_points': len(df)
        }
    
    def create_comprehensive_factor_dashboard(self):
        """Create comprehensive dashboard with factor analysis visualizations"""
        print("🎨 Creating options factor analysis dashboard...")
        
        if not self.factor_analysis:
            print("  ⚠️ No factor analysis data available")
            return None
        
        # Create large dashboard (4x5 grid = 20 panels)
        fig = plt.figure(figsize=(28, 32))
        gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('REX ETF Options Factor Analysis - Vegas, Volatilities & Call Option Sensitivities', 
                    fontsize=24, fontweight='bold', y=0.98)
        
        # Color palette
        colors = [self.rex_etfs[etf]['color'] for etf in self.rex_etfs.keys() if etf in self.factor_analysis]
        etf_list = list(self.factor_analysis.keys())
        
        # Row 1: Correlation Analysis
        # 1. ETF Return vs Call Price Change Correlation
        ax1 = fig.add_subplot(gs[0, 0])
        correlations = [self.factor_analysis[etf]['correlations'].get('etf_return_vs_call_price_change', 0) 
                       for etf in etf_list]
        bars = ax1.bar(etf_list, correlations, color=colors, alpha=0.8)
        ax1.set_title('ETF Return vs Call Price Change\nCorrelation', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Correlation Coefficient')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self._add_value_labels(ax1, bars, '{:.3f}')
        
        # 2. ETF Return vs Vega Correlation
        ax2 = fig.add_subplot(gs[0, 1])
        vega_corrs = [self.factor_analysis[etf]['correlations'].get('etf_return_vs_vega', 0) 
                      for etf in etf_list]
        bars = ax2.bar(etf_list, vega_corrs, color=colors, alpha=0.8)
        ax2.set_title('ETF Return vs Vega\nCorrelation', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Correlation Coefficient')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self._add_value_labels(ax2, bars, '{:.3f}')
        
        # 3. ETF Return vs Volatility Correlation
        ax3 = fig.add_subplot(gs[0, 2])
        vol_corrs = [self.factor_analysis[etf]['correlations'].get('etf_return_vs_volatility', 0) 
                     for etf in etf_list]
        bars = ax3.bar(etf_list, vol_corrs, color=colors, alpha=0.8)
        ax3.set_title('ETF Return vs Underlying Volatility\nCorrelation', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Correlation Coefficient')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self._add_value_labels(ax3, bars, '{:.3f}')
        
        # 4. Vega vs Volatility Correlation
        ax4 = fig.add_subplot(gs[0, 3])
        vega_vol_corrs = [self.factor_analysis[etf]['correlations'].get('vega_vs_volatility', 0) 
                          for etf in etf_list]
        bars = ax4.bar(etf_list, vega_vol_corrs, color=colors, alpha=0.8)
        ax4.set_title('Vega vs Volatility\nCorrelation', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Correlation Coefficient')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self._add_value_labels(ax4, bars, '{:.3f}')
        
        # Row 2: Factor Loadings
        # 5. Vega Factor Loadings
        ax5 = fig.add_subplot(gs[1, 0])
        vega_loadings = [self.factor_analysis[etf]['factor_loadings'].get('vega_loading', 0) 
                         for etf in etf_list]
        bars = ax5.bar(etf_list, vega_loadings, color=colors, alpha=0.8)
        ax5.set_title('Vega Factor Loadings\n(ETF Return Sensitivity)', fontweight='bold', fontsize=12)
        ax5.set_ylabel('Factor Loading')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self._add_value_labels(ax5, bars, '{:.4f}')
        
        # 6. Volatility Factor Loadings
        ax6 = fig.add_subplot(gs[1, 1])
        vol_loadings = [self.factor_analysis[etf]['factor_loadings'].get('volatility_loading', 0) 
                        for etf in etf_list]
        bars = ax6.bar(etf_list, vol_loadings, color=colors, alpha=0.8)
        ax6.set_title('Volatility Factor Loadings\n(ETF Return Sensitivity)', fontweight='bold', fontsize=12)
        ax6.set_ylabel('Factor Loading')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self._add_value_labels(ax6, bars, '{:.4f}')
        
        # 7. Option Yield Factor Loadings
        ax7 = fig.add_subplot(gs[1, 2])
        yield_loadings = [self.factor_analysis[etf]['factor_loadings'].get('option_yield_loading', 0) 
                          for etf in etf_list]
        bars = ax7.bar(etf_list, yield_loadings, color=colors, alpha=0.8)
        ax7.set_title('Option Yield Factor Loadings\n(ETF Return Sensitivity)', fontweight='bold', fontsize=12)
        ax7.set_ylabel('Factor Loading')
        ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self._add_value_labels(ax7, bars, '{:.4f}')
        
        # 8. R-Squared (Model Fit)
        ax8 = fig.add_subplot(gs[1, 3])
        r_squared = [self.factor_analysis[etf]['factor_loadings'].get('r_squared', 0) 
                     for etf in etf_list]
        bars = ax8.bar(etf_list, r_squared, color=colors, alpha=0.8)
        ax8.set_title('Factor Model R-Squared\n(Explanatory Power)', fontweight='bold', fontsize=12)
        ax8.set_ylabel('R-Squared')
        ax8.set_ylim(0, 1)
        self._add_value_labels(ax8, bars, '{:.3f}')
        
        # Row 3: Volatility Analysis
        # 9. Average ETF vs Underlying Volatility
        ax9 = fig.add_subplot(gs[2, 0])
        etf_vols = [self.factor_analysis[etf]['volatility_stats'].get('avg_etf_volatility', 0) 
                    for etf in etf_list]
        underlying_vols = [self.factor_analysis[etf]['volatility_stats'].get('avg_underlying_volatility', 0) 
                          for etf in etf_list]
        
        x = np.arange(len(etf_list))
        width = 0.35
        bars1 = ax9.bar(x - width/2, etf_vols, width, label='REX ETF', color=colors, alpha=0.8)
        bars2 = ax9.bar(x + width/2, underlying_vols, width, label='Underlying', color=colors, alpha=0.5)
        ax9.set_title('Average Volatility Comparison\nREX ETF vs Underlying', fontweight='bold', fontsize=12)
        ax9.set_ylabel('Annualized Volatility')
        ax9.set_xticks(x)
        ax9.set_xticklabels(etf_list)
        ax9.legend()
        
        # 10. Volatility Spread (ETF - Underlying)
        ax10 = fig.add_subplot(gs[2, 1])
        vol_spreads = [self.factor_analysis[etf]['volatility_stats'].get('volatility_spread_mean', 0) 
                       for etf in etf_list]
        bars = ax10.bar(etf_list, vol_spreads, color=colors, alpha=0.8)
        ax10.set_title('Volatility Spread\n(ETF - Underlying)', fontweight='bold', fontsize=12)
        ax10.set_ylabel('Volatility Difference')
        ax10.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self._add_value_labels(ax10, bars, '{:.3f}')
        
        # 11. Average Vega Values
        ax11 = fig.add_subplot(gs[2, 2])
        avg_vegas = [self.factor_analysis[etf]['volatility_stats'].get('avg_vega', 0) 
                     for etf in etf_list]
        bars = ax11.bar(etf_list, avg_vegas, color=colors, alpha=0.8)
        ax11.set_title('Average Vega Values\n(Volatility Sensitivity)', fontweight='bold', fontsize=12)
        ax11.set_ylabel('Vega (per 1% vol change)')
        self._add_value_labels(ax11, bars, '{:.2f}')
        
        # 12. Average Option Yield
        ax12 = fig.add_subplot(gs[2, 3])
        option_yields = [self.factor_analysis[etf]['volatility_stats'].get('avg_option_yield', 0) * 100 
                         for etf in etf_list]
        bars = ax12.bar(etf_list, option_yields, color=colors, alpha=0.8)
        ax12.set_title('Average Option Yield\n(Covered Call Premium/Price)', fontweight='bold', fontsize=12)
        ax12.set_ylabel('Option Yield (%)')
        self._add_value_labels(ax12, bars, '{:.2f}%')
        
        # Row 4: Time Series Analysis
        # 13-16. Individual ETF Time Series (4 subplots)
        for i, etf in enumerate(etf_list[:4]):
            ax = fig.add_subplot(gs[3, i])
            
            if etf in self.option_data and not self.option_data[etf].empty:
                data = self.option_data[etf]
                
                # Plot ETF returns vs Vega (normalized)
                ax2 = ax.twinx()
                
                dates = data.index
                etf_returns = data['ETF_Return'].cumsum() * 100
                vega_normalized = (data['Vega'] - data['Vega'].mean()) / data['Vega'].std()
                
                line1 = ax.plot(dates, etf_returns, color=self.rex_etfs[etf]['color'], 
                               linewidth=2, label='Cumulative ETF Return')
                line2 = ax2.plot(dates, vega_normalized, color='red', alpha=0.7, 
                                linestyle='--', linewidth=1.5, label='Normalized Vega')
                
                ax.set_title(f'{etf} Returns vs Vega\nTime Series', fontweight='bold', fontsize=11)
                ax.set_ylabel('Cumulative Return (%)', color=self.rex_etfs[etf]['color'])
                ax2.set_ylabel('Normalized Vega', color='red')
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='upper left', fontsize=8)
                
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        # Row 5: Advanced Analysis
        # 17. Factor Loading Heatmap
        ax17 = fig.add_subplot(gs[4, 0])
        
        # Prepare heatmap data
        factor_names = ['Vega', 'Volatility', 'Option Yield', 'Delta', 'Gamma']
        heatmap_data = []
        
        for etf in etf_list:
            loadings = self.factor_analysis[etf]['factor_loadings']
            row = [
                loadings.get('vega_loading', 0),
                loadings.get('volatility_loading', 0),
                loadings.get('option_yield_loading', 0),
                loadings.get('delta_loading', 0),
                loadings.get('gamma_loading', 0)
            ]
            heatmap_data.append(row)
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, index=etf_list, columns=factor_names)
            sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdBu_r', center=0, 
                       cbar_kws={'shrink': 0.8}, ax=ax17)
            ax17.set_title('Factor Loadings Heatmap\n(ETF Return Sensitivities)', fontweight='bold', fontsize=12)
        
        # 18. Correlation Matrix
        ax18 = fig.add_subplot(gs[4, 1])
        
        corr_names = ['Call Price', 'Vega', 'Volatility', 'Option Yield']
        corr_data = []
        
        for etf in etf_list:
            corrs = self.factor_analysis[etf]['correlations']
            row = [
                corrs.get('etf_return_vs_call_price_change', 0),
                corrs.get('etf_return_vs_vega', 0),
                corrs.get('etf_return_vs_volatility', 0),
                corrs.get('etf_return_vs_option_yield', 0)
            ]
            corr_data.append(row)
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data, index=etf_list, columns=corr_names)
            sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       vmin=-1, vmax=1, cbar_kws={'shrink': 0.8}, ax=ax18)
            ax18.set_title('ETF Return Correlations\nwith Option Factors', fontweight='bold', fontsize=12)
        
        # 19. Volatility Scatter Plot
        ax19 = fig.add_subplot(gs[4, 2])
        
        for i, etf in enumerate(etf_list):
            if etf in self.option_data:
                data = self.option_data[etf].dropna()
                if len(data) > 10:
                    x = data['Underlying_Volatility']
                    y = data['ETF_Return']
                    ax19.scatter(x, y, color=colors[i], alpha=0.6, s=30, label=etf)
        
        ax19.set_xlabel('Underlying Volatility')
        ax19.set_ylabel('ETF Daily Return')
        ax19.set_title('ETF Returns vs\nUnderlying Volatility', fontweight='bold', fontsize=12)
        ax19.legend()
        ax19.grid(True, alpha=0.3)
        
        # 20. Factor Importance Summary
        ax20 = fig.add_subplot(gs[4, 3])
        ax20.axis('off')
        
        # Create summary table
        summary_data = []
        for etf in etf_list:
            analysis = self.factor_analysis[etf]
            
            # Find most important factor (highest absolute loading)
            loadings = analysis['factor_loadings']
            factors = ['vega_loading', 'volatility_loading', 'option_yield_loading']
            factor_values = [abs(loadings.get(f, 0)) for f in factors]
            
            if factor_values:
                max_idx = np.argmax(factor_values)
                dominant_factor = ['Vega', 'Volatility', 'Option Yield'][max_idx]
                factor_strength = factor_values[max_idx]
            else:
                dominant_factor = 'N/A'
                factor_strength = 0
            
            summary_data.append([
                etf,
                f"{analysis['volatility_stats'].get('avg_vega', 0):.2f}",
                f"{analysis['volatility_stats'].get('avg_option_yield', 0)*100:.1f}%",
                dominant_factor,
                f"{factor_strength:.4f}"
            ])
        
        if summary_data:
            table = ax20.table(
                cellText=summary_data,
                colLabels=['ETF', 'Avg Vega', 'Avg Yield', 'Key Factor', 'Strength'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2)
            ax20.set_title('Options Factor Summary\nKey Sensitivities by ETF', 
                          fontweight='bold', fontsize=14, pad=30)
        
        plt.tight_layout()
        return fig
    
    def _add_value_labels(self, ax, bars, format_str):
        """Add value labels to bar charts"""
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and abs(height) > 1e-6:
                ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                       format_str.format(height), ha='center', va='bottom', 
                       fontweight='bold', fontsize=8)
    
    def export_analysis_results(self):
        """Export comprehensive analysis results"""
        print("💾 Exporting options factor analysis results...")
        
        # 1. Factor analysis summary CSV
        summary_data = []
        for etf, analysis in self.factor_analysis.items():
            row = {
                'ETF': etf,
                'Sector': self.rex_etfs[etf]['sector'],
                'Data_Points': analysis['data_points'],
                
                # Correlations
                'ETF_Call_Price_Corr': analysis['correlations'].get('etf_return_vs_call_price_change', 0),
                'ETF_Vega_Corr': analysis['correlations'].get('etf_return_vs_vega', 0),
                'ETF_Volatility_Corr': analysis['correlations'].get('etf_return_vs_volatility', 0),
                'ETF_Option_Yield_Corr': analysis['correlations'].get('etf_return_vs_option_yield', 0),
                'Vega_Volatility_Corr': analysis['correlations'].get('vega_vs_volatility', 0),
                
                # Factor loadings
                'Vega_Loading': analysis['factor_loadings'].get('vega_loading', 0),
                'Volatility_Loading': analysis['factor_loadings'].get('volatility_loading', 0),
                'Option_Yield_Loading': analysis['factor_loadings'].get('option_yield_loading', 0),
                'Delta_Loading': analysis['factor_loadings'].get('delta_loading', 0),
                'Gamma_Loading': analysis['factor_loadings'].get('gamma_loading', 0),
                'R_Squared': analysis['factor_loadings'].get('r_squared', 0),
                
                # Volatility statistics
                'Avg_ETF_Volatility': analysis['volatility_stats'].get('avg_etf_volatility', 0),
                'Avg_Underlying_Volatility': analysis['volatility_stats'].get('avg_underlying_volatility', 0),
                'Volatility_Spread_Mean': analysis['volatility_stats'].get('volatility_spread_mean', 0),
                'Volatility_Spread_Std': analysis['volatility_stats'].get('volatility_spread_std', 0),
                'Avg_Vega': analysis['volatility_stats'].get('avg_vega', 0),
                'Avg_Option_Yield': analysis['volatility_stats'].get('avg_option_yield', 0)
            }
            summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.docs_dir, 'rex_options_factor_analysis.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"  ✅ Factor analysis summary: {summary_path}")
        
        # 2. Individual ETF option data
        for etf, option_data in self.option_data.items():
            if not option_data.empty:
                detail_path = os.path.join(self.docs_dir, f'{etf.lower()}_option_metrics.csv')
                option_data.to_csv(detail_path)
                print(f"  ✅ {etf} option metrics: {detail_path}")
        
        # 3. Analysis metadata
        metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_type': 'options_factor_analysis',
            'etfs_analyzed': list(self.factor_analysis.keys()),
            'rex_family_config': self.rex_etfs,
            'analysis_parameters': {
                'risk_free_rate': self.risk_free_rate,
                'covered_call_ratio': self.covered_call_ratio,
                'option_maturity_days': 30
            }
        }
        
        metadata_path = os.path.join(self.docs_dir, 'rex_options_analysis_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Analysis metadata: {metadata_path}")
        
        return {
            'summary_csv': summary_path if summary_data else None,
            'metadata_json': metadata_path
        }
    
    def run_complete_factor_analysis(self):
        """Execute complete options factor analysis"""
        print("🚀 REX ETF Options Factor Analysis")
        print("=" * 60)
        
        try:
            # 1. Fetch market data
            self.fetch_market_data(period='6mo')
            
            # 2. Perform factor analysis
            self.perform_factor_analysis()
            
            # 3. Create dashboard
            dashboard_fig = self.create_comprehensive_factor_dashboard()
            
            # 4. Save dashboard
            if dashboard_fig:
                dashboard_path = os.path.join(self.docs_dir, 'rex_options_factor_analysis_dashboard.png')
                dashboard_fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
                print(f"📊 Options factor dashboard: {dashboard_path}")
                plt.close(dashboard_fig)
            
            # 5. Export results
            export_paths = self.export_analysis_results()
            
            # 6. Summary
            print("\n" + "=" * 60)
            print("📊 OPTIONS FACTOR ANALYSIS RESULTS")
            print("=" * 60)
            
            for etf, analysis in self.factor_analysis.items():
                print(f"\n🎯 {etf} ({self.rex_etfs[etf]['sector']}):")
                print(f"  Vega Correlation: {analysis['correlations'].get('etf_return_vs_vega', 0):.3f}")
                print(f"  Volatility Correlation: {analysis['correlations'].get('etf_return_vs_volatility', 0):.3f}")
                print(f"  Average Vega: {analysis['volatility_stats'].get('avg_vega', 0):.2f}")
                print(f"  Option Yield: {analysis['volatility_stats'].get('avg_option_yield', 0)*100:.1f}%")
                print(f"  Factor Model R²: {analysis['factor_loadings'].get('r_squared', 0):.3f}")
            
            print(f"\n📂 Output Files:")
            for key, path in export_paths.items():
                if path:
                    print(f"  {key}: {path}")
            
            print("\n✅ OPTIONS FACTOR ANALYSIS COMPLETED!")
            return True
            
        except Exception as e:
            print(f"❌ Analysis Error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    analyzer = REXOptionFactorAnalyzer()
    return analyzer.run_complete_factor_analysis()

if __name__ == "__main__":
    main()