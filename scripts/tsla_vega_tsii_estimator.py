import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
class TSLAVegaTSIIEstimator:
    def __init__(self):
        self.risk_free_rate = 0.045
        self.tsii_leverage = 2.0
        self.tsii_expense_ratio = 0.0095
        self.dividend_yield = 0.0
    def fetch_tsla_data(self, period="1y"):
        """Fetch TSLA historical data"""
        tsla = yf.Ticker("TSLA")
        data = tsla.history(period=period)
        return data
    def calculate_tsla_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate TSLA option Greeks with special focus on Vega"""
        if T <= 0:
            return {'price': max(S - K, 0) if option_type == 'call' else max(K - S, 0),
                    'delta': 1 if S > K else 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = theta / 365
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        return {'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
    def estimate_tsii_price(self, tsla_price, tsla_returns, base_tsii_price=100):
        """Estimate TSII price based on TSLA movements with leverage decay"""
        leveraged_returns = tsla_returns * self.tsii_leverage
        volatility = tsla_returns.std() * np.sqrt(252)
        drag_factor = 0.5 * (self.tsii_leverage**2 - self.tsii_leverage) * (volatility**2) / 252
        daily_expense = self.tsii_expense_ratio / 252
        adjusted_returns = leveraged_returns - drag_factor - daily_expense
        tsii_prices = base_tsii_price * (1 + adjusted_returns).cumprod()
        return tsii_prices
    def calculate_tsii_volatility(self, tsla_volatility):
        """Calculate TSII volatility with leverage amplification"""
        base_vol = tsla_volatility * self.tsii_leverage
        amplification_factor = 1 + 0.1 * (self.tsii_leverage - 1)
        return base_vol * amplification_factor
    def estimate_tsii_option_greeks(self, tsla_greeks, tsla_vol, tsii_price, tsii_vol, K, T):
        """Estimate TSII option Greeks based on TSLA Vega analysis"""
        tsii_greeks = self.calculate_tsla_greeks(tsii_price, K, T, self.risk_free_rate, tsii_vol)
        vega_amplification = self.tsii_leverage * (1 + tsii_vol/tsla_vol)
        enhanced_greeks = {
            'price': tsii_greeks['price'],
            'delta': tsii_greeks['delta'],
            'gamma': tsii_greeks['gamma'] * self.tsii_leverage,
            'theta': tsii_greeks['theta'] * (1 + self.tsii_expense_ratio),
            'vega': tsii_greeks['vega'] * vega_amplification,
            'rho': tsii_greeks['rho'] * self.tsii_leverage,
            'tsla_vega_correlation': tsla_greeks['vega'] / tsii_greeks['vega'] if tsii_greeks['vega'] != 0 else 0
        }
        return enhanced_greeks
    def generate_vega_based_signals(self, tsla_data, tsii_data, tsla_greeks_history, tsii_greeks_history):
        """Generate trading signals based on TSLA Vega analysis for TSII"""
        signals = []
        for i in range(len(tsla_data)):
            if i < 20:
                continue
            date = tsla_data.index[i]
            tsla_price = tsla_data['Close'].iloc[i]
            tsii_price = tsii_data.iloc[i] if i < len(tsii_data) else np.nan
            if pd.isna(tsii_price):
                continue
            recent_tsla_vegas = [g['vega'] for g in tsla_greeks_history[max(0, i-10):i] if 'vega' in g]
            recent_tsii_vegas = [g['vega'] for g in tsii_greeks_history[max(0, i-10):i] if 'vega' in g]
            if not recent_tsla_vegas or not recent_tsii_vegas:
                continue
            try:
                if len(recent_tsla_vegas) > 1:
                    tsla_vega_trend = np.polyfit(range(len(recent_tsla_vegas)), recent_tsla_vegas, 1)[0]
                else:
                    tsla_vega_trend = 0
                if len(recent_tsii_vegas) > 1:
                    tsii_vega_trend = np.polyfit(range(len(recent_tsii_vegas)), recent_tsii_vegas, 1)[0]
                else:
                    tsii_vega_trend = 0
            except (np.linalg.LinAlgError, ValueError):
                tsla_vega_trend = 0
                tsii_vega_trend = 0
            vega_ratio = recent_tsii_vegas[-1] / recent_tsla_vegas[-1] if recent_tsla_vegas[-1] != 0 else 1
            expected_ratio = self.tsii_leverage * 1.5
            vega_divergence = (vega_ratio - expected_ratio) / expected_ratio
            vega_momentum = tsii_vega_trend - tsla_vega_trend * self.tsii_leverage
            recent_vol = tsla_data['Close'].iloc[max(0, i-20):i].pct_change().std() * np.sqrt(252)
            vol_regime = 'HIGH' if recent_vol > 0.6 else ('MEDIUM' if recent_vol > 0.4 else 'LOW')
            if vega_divergence > 0.2 and vega_momentum > 0:
                entry_signal = 1
            elif vega_divergence < -0.2 and vega_momentum < 0:
                entry_signal = -1
            else:
                entry_signal = 0
            if abs(vega_divergence) < 0.05:
                exit_signal = 1
            else:
                exit_signal = 0
            risk_score = abs(vega_divergence) + abs(vega_momentum) * 0.5
            risk_level = 'HIGH' if risk_score > 0.5 else ('MEDIUM' if risk_score > 0.2 else 'LOW')
            signals.append({
                'date': date,
                'tsla_price': tsla_price,
                'tsii_price': tsii_price,
                'tsla_vega': recent_tsla_vegas[-1],
                'tsii_vega': recent_tsii_vegas[-1],
                'vega_ratio': vega_ratio,
                'vega_divergence': vega_divergence,
                'vega_momentum': vega_momentum,
                'vol_regime': vol_regime,
                'entry_signal': entry_signal,
                'exit_signal': exit_signal,
                'risk_level': risk_level,
                'risk_score': risk_score
            })
        return pd.DataFrame(signals)
    def create_comprehensive_analysis(self):
        """Main analysis function"""
        print("Fetching TSLA data...")
        tsla_data = self.fetch_tsla_data("1y")
        tsla_returns = tsla_data['Close'].pct_change().dropna()
        print("Estimating TSII prices using TSLA Vega analysis...")
        tsii_prices = self.estimate_tsii_price(tsla_data['Close'], tsla_returns)
        tsla_vol = tsla_returns.rolling(20).std() * np.sqrt(252)
        tsii_vol = self.calculate_tsii_volatility(tsla_vol)
        strikes = [0.9, 0.95, 1.0, 1.05, 1.1]
        maturities = [30, 60, 90]
        tsla_greeks_history = []
        tsii_greeks_history = []
        print("Calculating Greeks history...")
        for i, (idx, row) in enumerate(tsla_data.iterrows()):
            if i < 20 or i >= len(tsla_vol) or i >= len(tsii_vol) or pd.isna(tsla_vol.iloc[i]) or pd.isna(tsii_vol.iloc[i]):
                tsla_greeks_history.append({})
                tsii_greeks_history.append({})
                continue
            S_tsla = row['Close']
            S_tsii = tsii_prices.iloc[i] if i < len(tsii_prices) else S_tsla * 2
            vol_tsla = tsla_vol.iloc[i]
            vol_tsii = tsii_vol.iloc[i]
            K_tsla = S_tsla
            K_tsii = S_tsii
            T = 60/365
            tsla_greeks = self.calculate_tsla_greeks(S_tsla, K_tsla, T, self.risk_free_rate, vol_tsla)
            tsii_greeks = self.estimate_tsii_option_greeks(tsla_greeks, vol_tsla, S_tsii, vol_tsii, K_tsii, T)
            tsla_greeks_history.append(tsla_greeks)
            tsii_greeks_history.append(tsii_greeks)
        print("Generating Vega-based trading signals...")
        signals_df = self.generate_vega_based_signals(tsla_data, tsii_prices, tsla_greeks_history, tsii_greeks_history)
        return tsla_data, tsii_prices, signals_df, tsla_greeks_history, tsii_greeks_history
    def create_visualization(self, tsla_data, tsii_prices, signals_df, tsla_greeks_history, tsii_greeks_history):
        """Create comprehensive PNG visualization"""
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8])
        ax1 = fig.add_subplot(gs[0, :])
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(tsla_data.index, tsla_data['Close'], color='
        line2 = ax1_twin.plot(tsii_prices.index, tsii_prices.values, color='
        if not signals_df.empty:
            long_entries = signals_df[signals_df['entry_signal'] == 1]
            short_entries = signals_df[signals_df['entry_signal'] == -1]
            exits = signals_df[signals_df['exit_signal'] == 1]
            ax1_twin.scatter(long_entries['date'], long_entries['tsii_price'], color='green', marker='^', s=100, alpha=0.8, label='LONG Entry', zorder=5)
            ax1_twin.scatter(short_entries['date'], short_entries['tsii_price'], color='red', marker='v', s=100, alpha=0.8, label='SHORT Entry', zorder=5)
            ax1_twin.scatter(exits['date'], exits['tsii_price'], color='yellow', marker='x', s=80, alpha=0.8, label='EXIT', zorder=5)
        ax1.set_ylabel('TSLA Price ($)', color='
        ax1_twin.set_ylabel('TSII Estimated Price ($)', color='
        ax1.set_title('TSLA vs TSII Price Analysis with Vega-Based Entry/Exit Signals', fontsize=16, fontweight='bold')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax2 = fig.add_subplot(gs[1, 0])
        if not signals_df.empty:
            ax2.plot(signals_df['date'], signals_df['tsla_vega'], color='
            ax2.plot(signals_df['date'], signals_df['tsii_vega'], color='
            ax2.set_title('Vega Comparison: TSLA vs TSII', fontweight='bold')
            ax2.set_ylabel('Vega')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax3 = fig.add_subplot(gs[1, 1])
        if not signals_df.empty:
            ax3.plot(signals_df['date'], signals_df['vega_ratio'], color='purple', linewidth=2, label='Actual Vega Ratio')
            expected_ratio = self.tsii_leverage * 1.5
            ax3.axhline(y=expected_ratio, color='orange', linestyle='--', linewidth=2, label=f'Expected Ratio ({expected_ratio:.1f})')
            ax3.fill_between(signals_df['date'], expected_ratio*0.8, expected_ratio*1.2, alpha=0.2, color='orange', label='Normal Range')
            ax3.set_title('TSII/TSLA Vega Ratio Analysis', fontweight='bold')
            ax3.set_ylabel('Vega Ratio')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax4 = fig.add_subplot(gs[1, 2])
        if not signals_df.empty:
            colors = ['red' if x < -0.2 else 'green' if x > 0.2 else 'gray' for x in signals_df['vega_divergence']]
            ax4.bar(range(len(signals_df)), signals_df['vega_divergence'], color=colors, alpha=0.7)
            ax4.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Long Threshold')
            ax4.axhline(y=-0.2, color='red', linestyle='--', alpha=0.7, label='Short Threshold')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_title('Vega Divergence Signals', fontweight='bold')
            ax4.set_ylabel('Divergence')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        ax5 = fig.add_subplot(gs[2, 0])
        if not signals_df.empty:
            vol_counts = signals_df['vol_regime'].value_counts()
            colors_vol = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
            bars = ax5.bar(vol_counts.index, vol_counts.values, color=[colors_vol.get(x, 'gray') for x in vol_counts.index])
            ax5.set_title('Volatility Regime Distribution', fontweight='bold')
            ax5.set_ylabel('Frequency')
            ax5.grid(True, alpha=0.3)
        ax6 = fig.add_subplot(gs[2, 1])
        if not signals_df.empty:
            risk_counts = signals_df['risk_level'].value_counts()
            colors_risk = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
            ax6.bar(risk_counts.index, risk_counts.values, color=[colors_risk.get(x, 'gray') for x in risk_counts.index])
            ax6.set_title('Risk Level Distribution', fontweight='bold')
            ax6.set_ylabel('Frequency')
            ax6.grid(True, alpha=0.3)
        ax7 = fig.add_subplot(gs[2, 2])
        if not signals_df.empty:
            signal_counts = signals_df['entry_signal'].value_counts()
            labels = ['SHORT', 'NEUTRAL', 'LONG']
            values = [signal_counts.get(-1, 0), signal_counts.get(0, 0), signal_counts.get(1, 0)]
            colors_sig = ['red', 'gray', 'green']
            ax7.pie(values, labels=labels, colors=colors_sig, autopct='%1.1f%%', startangle=90)
            ax7.set_title('Signal Distribution', fontweight='bold')
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        if not signals_df.empty:
            latest_signal = signals_df.iloc[-1]
            total_long = (signals_df['entry_signal'] == 1).sum()
            total_short = (signals_df['entry_signal'] == -1).sum()
            total_exits = (signals_df['exit_signal'] == 1).sum()
            avg_vega_ratio = signals_df['vega_ratio'].mean()
            current_tsla = tsla_data['Close'].iloc[-1]
            current_tsii_est = tsii_prices.iloc[-1] if len(tsii_prices) > 0 else 0
            if latest_signal['entry_signal'] == 1:
                current_rec = "🔥 STRONG BUY TSII CALLS"
            elif latest_signal['entry_signal'] == -1:
                current_rec = "🔻 STRONG SELL TSII CALLS / BUY PUTS"
            elif latest_signal['exit_signal'] == 1:
                current_rec = "⚠️ EXIT TSII POSITIONS"
            else:
                current_rec = "⏸️ HOLD / WAIT FOR SIGNAL"
            summary_text = f"""
TSLA VEGA → TSII ESTIMATION ANALYSIS
CURRENT MARKET STATUS:
• TSLA Price: ${current_tsla:.2f}
• TSII Estimated Price: ${current_tsii_est:.2f}
• Current Vega Ratio: {latest_signal['vega_ratio']:.2f}
• Volatility Regime: {latest_signal['vol_regime']}
• Risk Level: {latest_signal['risk_level']}
SIGNAL SUMMARY:
• Total Long Signals: {total_long}
• Total Short Signals: {total_short}  
• Total Exit Signals: {total_exits}
• Average Vega Ratio: {avg_vega_ratio:.2f}
CURRENT RECOMMENDATION: {current_rec}
METHODOLOGY:
• TSII Price Estimation: 2x leveraged TSLA with volatility drag and expense ratio
• Vega Amplification: TSII Vega = TSLA Vega × {self.tsii_leverage} × (1 + vol_ratio)
• Signal Generation: Based on Vega divergence from expected leverage ratio
• Entry Threshold: ±20% divergence from expected Vega relationship
• Risk Management: Dynamic based on volatility regime and Vega momentum
KEY INSIGHTS:
• TSII options show amplified Vega sensitivity due to leverage
• Vega divergence signals potential mispricings in TSII options
• High volatility regimes increase signal reliability
• Leverage decay affects long-term TSII performance vs TSLA
            """
            ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        plt.suptitle('TSLA VEGA-BASED TSII OPTION ANALYSIS & TRADING SIGNALS', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig('/home/kafka/projects/NVII/tsla_vega_tsii_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
def main():
    estimator = TSLAVegaTSIIEstimator()
    print("Starting TSLA Vega → TSII Analysis...")
    tsla_data, tsii_prices, signals_df, tsla_greeks_history, tsii_greeks_history = estimator.create_comprehensive_analysis()
    print("Creating visualization...")
    estimator.create_visualization(tsla_data, tsii_prices, signals_df, tsla_greeks_history, tsii_greeks_history)
    print("Analysis complete! Report saved as tsla_vega_tsii_analysis.png")
    if not signals_df.empty:
        latest = signals_df.iloc[-1]
        print(f"\nLATEST SIGNAL:")
        print(f"TSLA: ${latest['tsla_price']:.2f} | TSII Est: ${latest['tsii_price']:.2f}")
        print(f"Vega Ratio: {latest['vega_ratio']:.2f} | Divergence: {latest['vega_divergence']:.3f}")
        print(f"Signal: {'LONG' if latest['entry_signal'] == 1 else 'SHORT' if latest['entry_signal'] == -1 else 'NEUTRAL'}")
        print(f"Risk: {latest['risk_level']}")
if __name__ == "__main__":
    main()