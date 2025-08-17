#!/usr/bin/env python3
"""
NVII Project Dashboard Generator

Weekly-updatable dashboard for comprehensive NVII analysis and tracking.
Integrates all project components into a single, professional HTML dashboard.

Features:
- Automated data fetching and analysis
- Interactive charts and visualizations
- Weekly dividend tracking
- Performance comparisons with NVDA
- Risk metrics and portfolio analysis
- Alert system for significant changes
"""

import json
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

class NVIIDashboardGenerator:
    """
    Comprehensive dashboard generator for NVII project
    """
    
    def __init__(self, output_dir='/home/kafka/projects/NVII/dashboard'):
        """
        Initialize dashboard generator
        
        Args:
            output_dir (str): Directory to output dashboard files
        """
        self.output_dir = output_dir
        self.data = {}
        self.alerts = []
        self.charts = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dashboard configuration
        self.config = {
            'title': 'NVII Investment Dashboard',
            'subtitle': 'REX NVDA Growth & Income ETF Analysis',
            'update_frequency': 'Weekly',
            'risk_free_rate': 0.045,
            'target_leverage': 1.25,
            'dividend_target': 0.30  # 30% annual target
        }
    
    def fetch_current_data(self):
        """
        Fetch current market data for NVII and NVDA
        """
        print("📊 Fetching current market data...")
        
        try:
            # Fetch NVII data (last 6 months for trend analysis)
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            nvii_ticker = yf.Ticker("NVII")
            nvda_ticker = yf.Ticker("NVDA")
            
            # Get historical data
            nvii_hist = nvii_ticker.history(start=start_date, end=end_date)
            nvda_hist = nvda_ticker.history(start=start_date, end=end_date)
            
            # Get current info
            nvii_info = nvii_ticker.info
            nvda_info = nvda_ticker.info
            
            # Calculate returns
            nvii_hist['Price_Return'] = nvii_hist['Close'].pct_change()
            nvii_hist['Dividend_Yield'] = nvii_hist['Dividends'] / nvii_hist['Close'].shift(1)
            nvii_hist['Total_Return'] = nvii_hist['Price_Return'] + nvii_hist['Dividend_Yield']
            nvii_hist['Cumulative_Total_Return'] = (1 + nvii_hist['Total_Return']).cumprod() - 1
            
            nvda_hist['Total_Return'] = nvda_hist['Close'].pct_change() + nvda_hist['Dividends'] / nvda_hist['Close'].shift(1)
            nvda_hist['Cumulative_Total_Return'] = (1 + nvda_hist['Total_Return']).cumprod() - 1
            
            self.data.update({
                'nvii_current': {
                    'price': nvii_hist['Close'].iloc[-1],
                    'volume': nvii_hist['Volume'].iloc[-1],
                    'change_1d': nvii_hist['Price_Return'].iloc[-1],
                    'change_1w': nvii_hist['Price_Return'].tail(5).sum(),
                    'change_1m': nvii_hist['Cumulative_Total_Return'].iloc[-1] - nvii_hist['Cumulative_Total_Return'].iloc[-22],
                    'total_dividends_ytd': nvii_hist['Dividends'].sum(),
                    'dividend_yield_annualized': nvii_hist['Dividends'].sum() / nvii_hist['Close'].iloc[0] * (252/len(nvii_hist)),
                    'volatility_30d': nvii_hist['Total_Return'].tail(30).std() * np.sqrt(252),
                    'max_drawdown': self._calculate_max_drawdown(nvii_hist['Cumulative_Total_Return']),
                    'info': nvii_info
                },
                'nvda_current': {
                    'price': nvda_hist['Close'].iloc[-1],
                    'volume': nvda_hist['Volume'].iloc[-1],
                    'change_1d': nvda_hist['Close'].pct_change().iloc[-1],
                    'change_1w': nvda_hist['Close'].pct_change().tail(5).sum(),
                    'change_1m': nvda_hist['Cumulative_Total_Return'].iloc[-1] - nvda_hist['Cumulative_Total_Return'].iloc[-22],
                    'total_dividends_ytd': nvda_hist['Dividends'].sum(),
                    'volatility_30d': nvda_hist['Total_Return'].tail(30).std() * np.sqrt(252),
                    'info': nvda_info
                },
                'nvii_hist': nvii_hist,
                'nvda_hist': nvda_hist,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            print(f"✅ Data fetched successfully - NVII: ${self.data['nvii_current']['price']:.2f}")
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            raise
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown from cumulative returns"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns / rolling_max - 1
        return drawdown.min()
    
    def analyze_performance(self):
        """
        Analyze current performance and generate insights
        """
        print("📈 Analyzing performance...")
        
        nvii_data = self.data['nvii_current']
        nvda_data = self.data['nvda_current']
        
        # Calculate key metrics
        nvii_sharpe = (nvii_data['dividend_yield_annualized'] - self.config['risk_free_rate']) / nvii_data['volatility_30d']
        
        # Performance comparison
        performance_metrics = {
            'nvii_vs_nvda_1d': nvii_data['change_1d'] - nvda_data['change_1d'],
            'nvii_vs_nvda_1w': nvii_data['change_1w'] - nvda_data['change_1w'],
            'nvii_vs_nvda_1m': nvii_data['change_1m'] - nvda_data['change_1m'],
            'dividend_advantage': nvii_data['dividend_yield_annualized'] - nvda_data.get('dividend_yield_annualized', 0),
            'volatility_advantage': nvda_data['volatility_30d'] - nvii_data['volatility_30d'],
            'sharpe_ratio': nvii_sharpe
        }
        
        # Generate alerts based on performance
        self._generate_alerts(performance_metrics)
        
        self.data['performance_metrics'] = performance_metrics
        
        # Weekly dividend analysis
        self._analyze_weekly_dividends()
        
    def _analyze_weekly_dividends(self):
        """
        Analyze weekly dividend patterns and projections
        """
        nvii_hist = self.data['nvii_hist']
        recent_dividends = nvii_hist[nvii_hist['Dividends'] > 0].tail(10)
        
        if len(recent_dividends) > 0:
            avg_weekly_dividend = recent_dividends['Dividends'].mean()
            total_recent_dividends = recent_dividends['Dividends'].sum()
            
            # Project annual dividend
            weeks_per_year = 52
            projected_annual = avg_weekly_dividend * weeks_per_year
            current_yield_projection = projected_annual / self.data['nvii_current']['price']
            
            dividend_analysis = {
                'recent_avg_weekly': avg_weekly_dividend,
                'last_10_total': total_recent_dividends,
                'projected_annual': projected_annual,
                'projected_yield': current_yield_projection,
                'consistency_score': self._calculate_dividend_consistency(recent_dividends['Dividends']),
                'trend': self._calculate_dividend_trend(recent_dividends['Dividends'])
            }
            
            self.data['dividend_analysis'] = dividend_analysis
    
    def _calculate_dividend_consistency(self, dividends):
        """Calculate dividend consistency score (0-100)"""
        if len(dividends) < 2:
            return 0
        cv = dividends.std() / dividends.mean()  # Coefficient of variation
        return max(0, 100 - (cv * 100))  # Lower CV = higher consistency
    
    def _calculate_dividend_trend(self, dividends):
        """Calculate dividend trend direction"""
        if len(dividends) < 3:
            return "insufficient_data"
        
        recent_avg = dividends.tail(3).mean()
        earlier_avg = dividends.head(3).mean()
        
        if recent_avg > earlier_avg * 1.05:
            return "increasing"
        elif recent_avg < earlier_avg * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_alerts(self, metrics):
        """
        Generate alerts based on performance metrics
        """
        alerts = []
        
        # Price movement alerts
        if abs(metrics['nvii_vs_nvda_1d']) > 0.02:
            direction = "outperformed" if metrics['nvii_vs_nvda_1d'] > 0 else "underperformed"
            alerts.append({
                'type': 'performance',
                'level': 'info',
                'message': f"NVII {direction} NVDA by {abs(metrics['nvii_vs_nvda_1d']):.2%} today"
            })
        
        # Volatility alerts
        if self.data['nvii_current']['volatility_30d'] > 0.35:
            alerts.append({
                'type': 'risk',
                'level': 'warning',
                'message': f"High volatility detected: {self.data['nvii_current']['volatility_30d']:.1%} (30-day)"
            })
        
        # Dividend yield alerts
        target_yield = self.config['dividend_target']
        current_yield = self.data['nvii_current']['dividend_yield_annualized']
        
        if current_yield < target_yield * 0.8:
            alerts.append({
                'type': 'dividend',
                'level': 'warning',
                'message': f"Dividend yield below target: {current_yield:.1%} vs {target_yield:.1%} target"
            })
        elif current_yield > target_yield * 1.2:
            alerts.append({
                'type': 'dividend',
                'level': 'positive',
                'message': f"Strong dividend performance: {current_yield:.1%} vs {target_yield:.1%} target"
            })
        
        # Sharpe ratio alerts
        if metrics['sharpe_ratio'] > 2.0:
            alerts.append({
                'type': 'performance',
                'level': 'positive',
                'message': f"Excellent risk-adjusted returns: Sharpe ratio {metrics['sharpe_ratio']:.2f}"
            })
        
        self.alerts = alerts
    
    def generate_charts(self):
        """
        Generate all charts for the dashboard
        """
        print("📊 Generating charts...")
        
        # Chart 1: NVII vs NVDA Performance
        self._create_performance_chart()
        
        # Chart 2: Dividend Timeline
        self._create_dividend_chart()
        
        # Chart 3: Risk Metrics
        self._create_risk_chart()
        
        # Chart 4: Weekly Performance Heatmap
        self._create_weekly_heatmap()
    
    def _create_performance_chart(self):
        """Create NVII vs NVDA performance comparison chart"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        nvii_hist = self.data['nvii_hist']
        nvda_hist = self.data['nvda_hist']
        
        # Align dates
        common_dates = nvii_hist.index.intersection(nvda_hist.index)
        nvii_aligned = nvii_hist.loc[common_dates]
        nvda_aligned = nvda_hist.loc[common_dates]
        
        # Plot cumulative returns
        ax.plot(common_dates, (nvii_aligned['Cumulative_Total_Return'] * 100), 
                label='NVII Total Return', linewidth=2.5, color='#1f77b4')
        ax.plot(common_dates, (nvda_aligned['Cumulative_Total_Return'] * 100), 
                label='NVDA Total Return', linewidth=2.5, color='#ff7f0e')
        
        # Add dividend markers
        div_dates = nvii_aligned[nvii_aligned['Dividends'] > 0].index
        div_returns = nvii_aligned.loc[div_dates, 'Cumulative_Total_Return'] * 100
        
        ax.scatter(div_dates, div_returns, color='red', s=50, alpha=0.7, zorder=5, label='NVII Dividends')
        
        ax.set_title('NVII vs NVDA: 6-Month Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        self.charts['performance'] = self._fig_to_base64(fig)
        plt.close(fig)
    
    def _create_dividend_chart(self):
        """Create dividend timeline chart"""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        nvii_hist = self.data['nvii_hist']
        dividend_data = nvii_hist[nvii_hist['Dividends'] > 0]
        
        if len(dividend_data) > 0:
            ax.bar(dividend_data.index, dividend_data['Dividends'], 
                   color='steelblue', alpha=0.7, width=1)
            
            # Add trend line
            ax.plot(dividend_data.index, dividend_data['Dividends'].rolling(3).mean(), 
                    color='red', linewidth=2, label='3-Period MA')
            
            ax.set_title('NVII Weekly Dividend Payments', fontsize=14, fontweight='bold')
            ax.set_ylabel('Dividend Amount ($)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        self.charts['dividends'] = self._fig_to_base64(fig)
        plt.close(fig)
    
    def _create_risk_chart(self):
        """Create risk metrics comparison chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Volatility comparison
        volatilities = [
            self.data['nvii_current']['volatility_30d'],
            self.data['nvda_current']['volatility_30d']
        ]
        
        ax1.bar(['NVII', 'NVDA'], [v * 100 for v in volatilities], 
                color=['steelblue', 'orange'], alpha=0.7)
        ax1.set_title('30-Day Volatility Comparison', fontweight='bold')
        ax1.set_ylabel('Volatility (%)')
        ax1.grid(True, alpha=0.3)
        
        # Max drawdown comparison
        drawdowns = [
            self.data['nvii_current']['max_drawdown'],
            self._calculate_max_drawdown(self.data['nvda_hist']['Cumulative_Total_Return'])
        ]
        
        ax2.bar(['NVII', 'NVDA'], [d * 100 for d in drawdowns], 
                color=['steelblue', 'orange'], alpha=0.7)
        ax2.set_title('Maximum Drawdown', fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.charts['risk'] = self._fig_to_base64(fig)
        plt.close(fig)
    
    def _create_weekly_heatmap(self):
        """Create weekly performance heatmap"""
        fig, ax = plt.subplots(figsize=(10, 3))
        
        nvii_hist = self.data['nvii_hist']
        
        # Calculate weekly returns
        weekly_returns = nvii_hist['Total_Return'].resample('W').sum()
        recent_weeks = weekly_returns.tail(10)
        
        # Create heatmap data
        colors = ['red' if x < 0 else 'green' for x in recent_weeks]
        alphas = [min(abs(x) * 10, 1.0) for x in recent_weeks]
        
        bars = ax.bar(range(len(recent_weeks)), recent_weeks * 100, 
                      color=colors, alpha=0.7)
        
        # Color bars based on performance
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
        
        ax.set_title('Weekly Performance (Last 10 Weeks)', fontweight='bold')
        ax.set_ylabel('Weekly Return (%)')
        ax.set_xlabel('Weeks Ago')
        ax.set_xticks(range(len(recent_weeks)))
        ax.set_xticklabels([f'{i}' for i in reversed(range(len(recent_weeks)))])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        self.charts['weekly_heatmap'] = self._fig_to_base64(fig)
        plt.close(fig)
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        buffer.close()
        return f"data:image/png;base64,{image_base64}"
    
    def generate_html_dashboard(self):
        """
        Generate the complete HTML dashboard
        """
        print("🎨 Generating HTML dashboard...")
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config['title']}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>{self.config['title']}</h1>
            <h2>{self.config['subtitle']}</h2>
            <p class="update-time">Last Updated: {self.data['last_update']}</p>
        </header>
        
        {self._generate_alerts_section()}
        {self._generate_summary_section()}
        {self._generate_performance_section()}
        {self._generate_dividend_section()}
        {self._generate_risk_section()}
        {self._generate_market_data_section()}
        
        <footer class="footer">
            <p>Generated by NVII Dashboard Generator | Update Frequency: {self.config['update_frequency']}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        # Save HTML dashboard
        html_path = os.path.join(self.output_dir, 'index.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"✅ Dashboard generated: {html_path}")
        return html_path
    
    def _get_css_styles(self):
        """Get CSS styles for the dashboard"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header h2 {
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 10px;
        }
        
        .update-time {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .section {
            background: white;
            margin: 20px 0;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .section h3 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .card h4 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .metric {
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }
        
        .metric.positive {
            color: #28a745;
        }
        
        .metric.negative {
            color: #dc3545;
        }
        
        .metric.neutral {
            color: #6c757d;
        }
        
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .alerts {
            margin-bottom: 20px;
        }
        
        .alert {
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .alert.info {
            background-color: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }
        
        .alert.warning {
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
        
        .alert.positive {
            background-color: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .comparison-table th,
        .comparison-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .comparison-table th {
            background-color: #667eea;
            color: white;
        }
        
        .comparison-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #6c757d;
            border-top: 1px solid #ddd;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _generate_alerts_section(self):
        """Generate alerts section HTML"""
        if not self.alerts:
            return ""
        
        alerts_html = '<div class="section alerts"><h3>🚨 Alerts & Insights</h3>'
        
        for alert in self.alerts:
            alerts_html += f'''
            <div class="alert {alert['level']}">
                <strong>{alert['type'].title()}:</strong> {alert['message']}
            </div>
            '''
        
        alerts_html += '</div>'
        return alerts_html
    
    def _generate_summary_section(self):
        """Generate summary metrics section"""
        nvii = self.data['nvii_current']
        nvda = self.data['nvda_current']
        metrics = self.data.get('performance_metrics', {})
        
        return f"""
        <div class="section">
            <h3>📊 Current Market Summary</h3>
            <div class="grid">
                <div class="card">
                    <h4>NVII Current Price</h4>
                    <div class="metric">${nvii['price']:.2f}</div>
                    <p>Daily Change: <span class="{'positive' if nvii['change_1d'] >= 0 else 'negative'}">{nvii['change_1d']:+.2%}</span></p>
                </div>
                <div class="card">
                    <h4>NVDA Current Price</h4>
                    <div class="metric">${nvda['price']:.2f}</div>
                    <p>Daily Change: <span class="{'positive' if nvda['change_1d'] >= 0 else 'negative'}">{nvda['change_1d']:+.2%}</span></p>
                </div>
                <div class="card">
                    <h4>NVII Dividend Yield</h4>
                    <div class="metric positive">{nvii['dividend_yield_annualized']:.1%}</div>
                    <p>YTD Dividends: ${nvii['total_dividends_ytd']:.2f}</p>
                </div>
                <div class="card">
                    <h4>Risk-Adjusted Return</h4>
                    <div class="metric positive">{metrics.get('sharpe_ratio', 0):.2f}</div>
                    <p>Sharpe Ratio (30-day)</p>
                </div>
            </div>
        </div>
        """
    
    def _generate_performance_section(self):
        """Generate performance comparison section"""
        return f"""
        <div class="section">
            <h3>📈 Performance Analysis</h3>
            <div class="chart-container">
                <img src="{self.charts.get('performance', '')}" alt="Performance Chart">
            </div>
            <div class="chart-container">
                <img src="{self.charts.get('weekly_heatmap', '')}" alt="Weekly Performance">
            </div>
        </div>
        """
    
    def _generate_dividend_section(self):
        """Generate dividend analysis section"""
        div_analysis = self.data.get('dividend_analysis', {})
        
        return f"""
        <div class="section">
            <h3>💰 Dividend Analysis</h3>
            <div class="grid">
                <div class="card">
                    <h4>Recent Average (Weekly)</h4>
                    <div class="metric">${div_analysis.get('recent_avg_weekly', 0):.3f}</div>
                </div>
                <div class="card">
                    <h4>Projected Annual</h4>
                    <div class="metric">${div_analysis.get('projected_annual', 0):.2f}</div>
                </div>
                <div class="card">
                    <h4>Projected Yield</h4>
                    <div class="metric positive">{div_analysis.get('projected_yield', 0):.1%}</div>
                </div>
                <div class="card">
                    <h4>Consistency Score</h4>
                    <div class="metric neutral">{div_analysis.get('consistency_score', 0):.0f}/100</div>
                </div>
            </div>
            <div class="chart-container">
                <img src="{self.charts.get('dividends', '')}" alt="Dividend Timeline">
            </div>
        </div>
        """
    
    def _generate_risk_section(self):
        """Generate risk analysis section"""
        nvii = self.data['nvii_current']
        nvda = self.data['nvda_current']
        
        return f"""
        <div class="section">
            <h3>⚠️ Risk Analysis</h3>
            <div class="grid">
                <div class="card">
                    <h4>NVII Volatility (30d)</h4>
                    <div class="metric">{nvii['volatility_30d']:.1%}</div>
                </div>
                <div class="card">
                    <h4>NVDA Volatility (30d)</h4>
                    <div class="metric">{nvda['volatility_30d']:.1%}</div>
                </div>
                <div class="card">
                    <h4>NVII Max Drawdown</h4>
                    <div class="metric negative">{nvii['max_drawdown']:.1%}</div>
                </div>
                <div class="card">
                    <h4>Risk Advantage</h4>
                    <div class="metric positive">{nvda['volatility_30d'] - nvii['volatility_30d']:+.1%}</div>
                    <p>NVII vs NVDA Volatility</p>
                </div>
            </div>
            <div class="chart-container">
                <img src="{self.charts.get('risk', '')}" alt="Risk Metrics">
            </div>
        </div>
        """
    
    def _generate_market_data_section(self):
        """Generate detailed market data table"""
        nvii = self.data['nvii_current']
        nvda = self.data['nvda_current']
        metrics = self.data.get('performance_metrics', {})
        
        return f"""
        <div class="section">
            <h3>📋 Detailed Market Data</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>NVII</th>
                        <th>NVDA</th>
                        <th>Difference</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Current Price</td>
                        <td>${nvii['price']:.2f}</td>
                        <td>${nvda['price']:.2f}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>Daily Change</td>
                        <td class="{'positive' if nvii['change_1d'] >= 0 else 'negative'}">{nvii['change_1d']:+.2%}</td>
                        <td class="{'positive' if nvda['change_1d'] >= 0 else 'negative'}">{nvda['change_1d']:+.2%}</td>
                        <td>{metrics.get('nvii_vs_nvda_1d', 0):+.2%}</td>
                    </tr>
                    <tr>
                        <td>Weekly Change</td>
                        <td class="{'positive' if nvii['change_1w'] >= 0 else 'negative'}">{nvii['change_1w']:+.2%}</td>
                        <td class="{'positive' if nvda['change_1w'] >= 0 else 'negative'}">{nvda['change_1w']:+.2%}</td>
                        <td>{metrics.get('nvii_vs_nvda_1w', 0):+.2%}</td>
                    </tr>
                    <tr>
                        <td>Monthly Change</td>
                        <td class="{'positive' if nvii['change_1m'] >= 0 else 'negative'}">{nvii['change_1m']:+.2%}</td>
                        <td class="{'positive' if nvda['change_1m'] >= 0 else 'negative'}">{nvda['change_1m']:+.2%}</td>
                        <td>{metrics.get('nvii_vs_nvda_1m', 0):+.2%}</td>
                    </tr>
                    <tr>
                        <td>Dividend Yield (Ann.)</td>
                        <td class="positive">{nvii['dividend_yield_annualized']:.1%}</td>
                        <td>{nvda.get('dividend_yield_annualized', 0):.1%}</td>
                        <td class="positive">+{metrics.get('dividend_advantage', 0):.1%}</td>
                    </tr>
                    <tr>
                        <td>Volatility (30d)</td>
                        <td>{nvii['volatility_30d']:.1%}</td>
                        <td>{nvda['volatility_30d']:.1%}</td>
                        <td class="positive">{metrics.get('volatility_advantage', 0):+.1%}</td>
                    </tr>
                    <tr>
                        <td>Volume (Latest)</td>
                        <td>{nvii['volume']:,}</td>
                        <td>{nvda['volume']:,}</td>
                        <td>-</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
    
    def save_data_snapshot(self):
        """
        Save current data snapshot for historical tracking
        """
        snapshot_data = {
            'timestamp': self.data['last_update'],
            'nvii_price': self.data['nvii_current']['price'],
            'nvda_price': self.data['nvda_current']['price'],
            'nvii_dividend_yield': self.data['nvii_current']['dividend_yield_annualized'],
            'nvii_volatility': self.data['nvii_current']['volatility_30d'],
            'sharpe_ratio': self.data['performance_metrics'].get('sharpe_ratio', 0),
            'alerts_count': len(self.alerts)
        }
        
        # Append to historical data file
        snapshot_file = os.path.join(self.output_dir, 'historical_snapshots.jsonl')
        with open(snapshot_file, 'a') as f:
            f.write(json.dumps(snapshot_data) + '\n')
        
        print(f"📚 Data snapshot saved to {snapshot_file}")
    
    def generate_dashboard(self):
        """
        Main method to generate complete dashboard
        """
        print("🚀 Starting NVII Dashboard Generation...")
        
        try:
            # Step 1: Fetch current data
            self.fetch_current_data()
            
            # Step 2: Analyze performance
            self.analyze_performance()
            
            # Step 3: Generate charts
            self.generate_charts()
            
            # Step 4: Generate HTML dashboard
            html_path = self.generate_html_dashboard()
            
            # Step 5: Save data snapshot
            self.save_data_snapshot()
            
            print(f"✅ Dashboard generation complete!")
            print(f"📂 Output directory: {self.output_dir}")
            print(f"🌐 Open in browser: file://{os.path.abspath(html_path)}")
            
            return html_path
            
        except Exception as e:
            print(f"❌ Dashboard generation failed: {e}")
            raise

def main():
    """
    Main function to run dashboard generation
    """
    generator = NVIIDashboardGenerator()
    dashboard_path = generator.generate_dashboard()
    return dashboard_path

if __name__ == "__main__":
    main()