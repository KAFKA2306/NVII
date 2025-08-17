#!/usr/bin/env python3
"""
統合REX分析システム (Unified REX Analysis System)

抜本的改善版：効率的・構造的・統合的アプローチ

特徴:
1. 統合データ管理: 全REXファミリーの一元管理
2. 効率的分析: バッチ処理による高速化
3. 構造的比較: 系統的な相対評価
4. 自動化対応: GitHub Actions完全統合
5. 拡張性: 新規ETF自動対応

REXファミリー:
- NVII (NVDA): AI/半導体セクター
- MSII (MSTR): Bitcoin/暗号通貨セクター 
- COII (COIN): 暗号通貨取引プラットフォーム
- TSII (TSLA): EV/自動運転セクター
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

class UnifiedREXSystem:
    """
    統合REX分析システム - 全ファミリーETFの効率的統合分析
    """
    
    def __init__(self, output_dir='/home/kafka/projects/NVII'):
        """
        統合システム初期化
        
        Args:
            output_dir (str): 出力ディレクトリパス
        """
        self.output_dir = output_dir
        self.dashboard_dir = os.path.join(output_dir, 'rex_dashboard') 
        self.docs_dir = os.path.join(output_dir, 'docs')
        
        # 出力ディレクトリ作成
        os.makedirs(self.dashboard_dir, exist_ok=True)
        os.makedirs(self.docs_dir, exist_ok=True)
        
        # REXファミリー定義（拡張可能な構造）
        self.rex_family = {
            'NVII': {
                'name': 'REX NVDA Growth & Income ETF',
                'underlying': 'NVDA',
                'underlying_name': 'NVIDIA Corporation',
                'sector': 'Technology/AI',
                'launch_date': '2024-05-01',
                'description': 'AI革命の中核企業・半導体リーダー',
                'color': '#1f77b4',
                'active': True
            },
            'MSII': {
                'name': 'REX MSTR Growth & Income ETF', 
                'underlying': 'MSTR',
                'underlying_name': 'MicroStrategy Inc',
                'sector': 'Bitcoin/FinTech',
                'launch_date': '2024-06-01',
                'description': 'Bitcoinプロキシ・ビジネスインテリジェンス',
                'color': '#ff7f0e',
                'active': True
            },
            'COII': {
                'name': 'REX COIN Growth & Income ETF',
                'underlying': 'COIN', 
                'underlying_name': 'Coinbase Global Inc',
                'sector': 'Cryptocurrency Exchange',
                'launch_date': '2024-07-01',
                'description': '暗号通貨取引プラットフォーム最大手',
                'color': '#2ca02c',
                'active': True
            },
            'TSII': {
                'name': 'REX TSLA Growth & Income ETF',
                'underlying': 'TSLA',
                'underlying_name': 'Tesla Inc',
                'sector': 'Electric Vehicles/Autonomous',
                'launch_date': '2024-08-01',
                'description': 'EV革命リーダー・自動運転技術',
                'color': '#d62728',
                'active': True
            }
        }
        
        # REX共通戦略パラメータ
        self.strategy_config = {
            'target_leverage': 1.25,
            'leverage_range': (1.05, 1.50),
            'covered_call_coverage': 0.50,
            'upside_unlimited_portion': 0.50,
            'distribution_frequency': 'weekly',
            'risk_free_rate': 0.045,
            'benchmark_period': '6mo'
        }
        
        # データストレージ
        self.raw_data = {}
        self.processed_data = {}
        self.analysis_results = {}
        self.comparative_metrics = {}
        
        # 分析設定
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
        並列データフェッチ - 全REXファミリーと原資産の効率的取得
        
        Args:
            period (str): データ期間 ('6mo', '1y', '2y', 'max')
        """
        print("🚀 統合REXデータフェッチ開始...")
        print(f"📊 対象期間: {period}")
        
        # 並列フェッチ用のシンボルリスト構築
        fetch_tasks = []
        active_etfs = {k: v for k, v in self.rex_family.items() if v['active']}
        
        for etf_symbol, config in active_etfs.items():
            # ETF自体
            fetch_tasks.append({
                'symbol': etf_symbol,
                'type': 'etf',
                'config': config
            })
            # 原資産
            fetch_tasks.append({
                'symbol': config['underlying'],
                'type': 'underlying',
                'etf_parent': etf_symbol,
                'config': config
            })
        
        print(f"📈 フェッチ対象: {len(fetch_tasks)} シンボル")
        
        # 並列実行
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
                    print(f"  ✅ {symbol}: {len(data) if not data.empty else 0} 取引日 ({completed}/{len(fetch_tasks)})")
                    
                except Exception as e:
                    print(f"  ❌ {task['symbol']}: {str(e)}")
        
        print(f"📊 データフェッチ完了: {len(self.raw_data)} ETF")
        return self.raw_data
    
    def _fetch_single_symbol(self, symbol, period):
        """
        単一シンボルデータフェッチ（並列処理用）
        
        Args:
            symbol (str): ティッカーシンボル
            period (str): データ期間
            
        Returns:
            pd.DataFrame: 価格データ
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"    ⚠️ {symbol} フェッチエラー: {e}")
            return pd.DataFrame()
    
    def process_all_data(self):
        """
        全データの統合処理・指標計算
        """
        print("⚙️ 統合データ処理開始...")
        
        for etf_symbol, data_dict in self.raw_data.items():
            if 'etf_data' not in data_dict or data_dict['etf_data'].empty:
                print(f"  ⚠️ {etf_symbol}: データなし、スキップ")
                continue
                
            print(f"  🔄 {etf_symbol} 処理中...")
            
            etf_data = data_dict['etf_data'].copy()
            underlying_data = data_dict.get('underlying_data', pd.DataFrame())
            config = data_dict['config']
            
            # ETFリターン計算
            etf_data['Price_Return'] = etf_data['Close'].pct_change()
            etf_data['Dividend_Yield'] = etf_data['Dividends'] / etf_data['Close'].shift(1)
            etf_data['Total_Return'] = etf_data['Price_Return'] + etf_data['Dividend_Yield']
            etf_data['Cumulative_Total_Return'] = (1 + etf_data['Total_Return']).cumprod() - 1
            
            # 原資産リターン計算（利用可能な場合）
            underlying_returns = pd.Series(dtype=float)
            if not underlying_data.empty:
                common_dates = etf_data.index.intersection(underlying_data.index)
                if len(common_dates) > 0:
                    underlying_aligned = underlying_data.loc[common_dates].copy()
                    underlying_aligned['Return'] = underlying_aligned['Close'].pct_change()
                    underlying_aligned['Cumulative_Return'] = (1 + underlying_aligned['Return']).cumprod() - 1
                    underlying_returns = underlying_aligned['Return']
            
            # パフォーマンス指標計算
            metrics = self._calculate_comprehensive_metrics(
                etf_data, underlying_returns, config
            )
            
            # 処理済みデータ保存
            self.processed_data[etf_symbol] = {
                'etf_data': etf_data,
                'underlying_returns': underlying_returns,
                'metrics': metrics,
                'config': config
            }
            
            print(f"    ✅ {etf_symbol}: {len(etf_data)} 日間, リターン {metrics['total_return']:.2%}")
        
        print(f"⚙️ データ処理完了: {len(self.processed_data)} ETF")
    
    def _calculate_comprehensive_metrics(self, etf_data, underlying_returns, config):
        """
        包括的パフォーマンス指標計算
        
        Args:
            etf_data (pd.DataFrame): ETF価格データ
            underlying_returns (pd.Series): 原資産リターン
            config (dict): ETF設定
            
        Returns:
            dict: 計算済み指標
        """
        total_days = len(etf_data)
        if total_days < 2:
            return self._empty_metrics()
        
        # 基本リターン指標
        total_return = etf_data['Cumulative_Total_Return'].iloc[-1]
        annualized_return = ((1 + total_return) ** (252/total_days)) - 1
        
        # リスク指標
        volatility = etf_data['Total_Return'].std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.strategy_config['risk_free_rate']) / volatility if volatility > 0 else 0
        
        # ドローダウン
        rolling_max = (1 + etf_data['Total_Return']).cumprod().expanding().max()
        drawdown = (1 + etf_data['Total_Return']).cumprod() / rolling_max - 1
        max_drawdown = drawdown.min()
        
        # 配当分析
        total_dividends = etf_data['Dividends'].sum()
        dividend_yield_annual = total_dividends / etf_data['Close'].iloc[0] * (252/total_days)
        
        # 配当の一貫性
        dividend_payments = etf_data[etf_data['Dividends'] > 0]['Dividends']
        dividend_consistency = 1 - (dividend_payments.std() / dividend_payments.mean()) if len(dividend_payments) > 1 else 0
        
        # レバレッジ分析
        leverage_estimate = self._estimate_leverage(etf_data, underlying_returns)
        leverage_efficiency = (total_return / leverage_estimate) if leverage_estimate > 0 else 0
        
        # 原資産比較（利用可能な場合）
        excess_return_vs_underlying = 0
        if len(underlying_returns) > 0:
            underlying_total_return = (1 + underlying_returns).cumprod().iloc[-1] - 1
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
            'current_price': etf_data['Close'].iloc[-1],
            'start_price': etf_data['Close'].iloc[0],
            'weekly_dividends_count': len(dividend_payments),
            'avg_weekly_dividend': dividend_payments.mean() if len(dividend_payments) > 0 else 0
        }
    
    def _estimate_leverage(self, etf_data, underlying_returns):
        """
        レバレッジ推定（ETFと原資産のベータ計算）
        
        Args:
            etf_data (pd.DataFrame): ETF価格データ
            underlying_returns (pd.Series): 原資産リターン
            
        Returns:
            float: 推定レバレッジ
        """
        if len(underlying_returns) < 10:
            return 1.25  # デフォルト目標レバレッジ
        
        try:
            # 共通期間でのベータ計算
            common_dates = etf_data.index.intersection(underlying_returns.index)
            if len(common_dates) < 10:
                return 1.25
                
            etf_returns = etf_data.loc[common_dates, 'Price_Return'].dropna()
            underlying_aligned = underlying_returns.loc[common_dates].dropna()
            
            # 共通のデータポイント確保
            min_length = min(len(etf_returns), len(underlying_aligned))
            if min_length < 10:
                return 1.25
                
            etf_returns = etf_returns.iloc[-min_length:]
            underlying_aligned = underlying_aligned.iloc[-min_length:]
            
            # ベータ計算（レバレッジの近似）
            covariance = np.cov(etf_returns, underlying_aligned)[0,1]
            underlying_variance = np.var(underlying_aligned)
            
            if underlying_variance > 0:
                beta = covariance / underlying_variance
                return max(0.5, min(3.0, beta))  # 合理的範囲に制限
            else:
                return 1.25
        except:
            return 1.25
    
    def _empty_metrics(self):
        """空指標辞書を返す"""
        return {key: 0 for key in self.analysis_config['performance_metrics']} | {
            'trading_days': 0, 'current_price': 0, 'start_price': 0,
            'weekly_dividends_count': 0, 'avg_weekly_dividend': 0
        }
    
    def generate_comparative_analysis(self):
        """
        REXファミリー間の比較分析
        """
        print("📊 REXファミリー比較分析...")
        
        # 比較用データフレーム構築
        comparison_data = []
        
        for etf_symbol, data_dict in self.processed_data.items():
            metrics = data_dict['metrics']
            config = data_dict['config']
            
            comparison_data.append({
                'ETF': etf_symbol,
                'Name': config['name'],
                'Underlying': config['underlying'],
                'Sector': config['sector'],
                'Total_Return': metrics['total_return'],
                'Annualized_Return': metrics['annualized_return'],
                'Dividend_Yield': metrics['dividend_yield_annual'],
                'Volatility': metrics['volatility'],
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Max_Drawdown': metrics['max_drawdown'],
                'Leverage_Estimate': metrics['leverage_estimate'],
                'Excess_vs_Underlying': metrics['excess_return_vs_underlying'],
                'Current_Price': metrics['current_price'],
                'Weekly_Dividends': metrics['weekly_dividends_count'],
                'Avg_Weekly_Div': metrics['avg_weekly_dividend'],
                'Dividend_Consistency': metrics['dividend_consistency']
            })
        
        self.comparative_df = pd.DataFrame(comparison_data)
        
        # ランキング計算
        self._calculate_rankings()
        
        return self.comparative_df
    
    def _calculate_rankings(self):
        """
        各指標でのランキング計算
        """
        if self.comparative_df.empty:
            return
        
        # ランキング計算（高い方が良い指標）
        positive_metrics = ['Total_Return', 'Dividend_Yield', 'Sharpe_Ratio', 'Dividend_Consistency']
        for metric in positive_metrics:
            self.comparative_df[f'{metric}_Rank'] = self.comparative_df[metric].rank(ascending=False, method='min')
        
        # ランキング計算（低い方が良い指標）
        negative_metrics = ['Volatility', 'Max_Drawdown']
        for metric in negative_metrics:
            self.comparative_df[f'{metric}_Rank'] = self.comparative_df[metric].rank(ascending=True, method='min')
        
        # 総合スコア計算（単純平均）
        rank_cols = [col for col in self.comparative_df.columns if col.endswith('_Rank')]
        self.comparative_df['Overall_Score'] = self.comparative_df[rank_cols].mean(axis=1)
        self.comparative_df['Overall_Rank'] = self.comparative_df['Overall_Score'].rank(method='min')
    
    def create_unified_dashboard(self):
        """
        統合REXファミリーダッシュボード作成
        """
        print("🎨 統合ダッシュボード作成...")
        
        if self.comparative_df.empty:
            print("  ⚠️ 比較データなし")
            return None
        
        # 大型ダッシュボード作成（4x4グリッド）
        fig, axes = plt.subplots(4, 4, figsize=(24, 20))
        fig.suptitle('REX Growth & Income ETF Family - 統合分析ダッシュボード', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # カラーパレット
        colors = [self.rex_family[etf]['color'] for etf in self.comparative_df['ETF']]
        
        # 1. 総合リターン比較
        ax = axes[0,0]
        bars = ax.bar(self.comparative_df['ETF'], self.comparative_df['Total_Return']*100, 
                     color=colors, alpha=0.8)
        ax.set_title('総合リターン比較', fontweight='bold')
        ax.set_ylabel('リターン (%)')
        self._add_value_labels(ax, bars, '{:.1f}%')
        
        # 2. 配当利回り比較
        ax = axes[0,1]
        bars = ax.bar(self.comparative_df['ETF'], self.comparative_df['Dividend_Yield']*100, 
                     color=colors, alpha=0.8)
        ax.set_title('年間配当利回り', fontweight='bold')
        ax.set_ylabel('利回り (%)')
        self._add_value_labels(ax, bars, '{:.1f}%')
        
        # 3. シャープレシオ比較
        ax = axes[0,2]
        bars = ax.bar(self.comparative_df['ETF'], self.comparative_df['Sharpe_Ratio'], 
                     color=colors, alpha=0.8)
        ax.set_title('リスク調整後リターン (シャープレシオ)', fontweight='bold')
        ax.set_ylabel('シャープレシオ')
        self._add_value_labels(ax, bars, '{:.2f}')
        
        # 4. ボラティリティ比較
        ax = axes[0,3]
        bars = ax.bar(self.comparative_df['ETF'], self.comparative_df['Volatility']*100, 
                     color=colors, alpha=0.8)
        ax.set_title('年率ボラティリティ', fontweight='bold')
        ax.set_ylabel('ボラティリティ (%)')
        self._add_value_labels(ax, bars, '{:.1f}%')
        
        # 5. 最大ドローダウン
        ax = axes[1,0]
        bars = ax.bar(self.comparative_df['ETF'], self.comparative_df['Max_Drawdown']*100, 
                     color='red', alpha=0.6)
        ax.set_title('最大ドローダウン', fontweight='bold')
        ax.set_ylabel('ドローダウン (%)')
        self._add_value_labels(ax, bars, '{:.1f}%')
        
        # 6. レバレッジ推定値
        ax = axes[1,1]
        bars = ax.bar(self.comparative_df['ETF'], self.comparative_df['Leverage_Estimate'], 
                     color='purple', alpha=0.7)
        ax.axhline(y=1.25, color='red', linestyle='--', alpha=0.8, label='目標 (1.25x)')
        ax.set_title('推定レバレッジ', fontweight='bold')
        ax.set_ylabel('レバレッジ倍率')
        ax.legend()
        self._add_value_labels(ax, bars, '{:.2f}x')
        
        # 7. 原資産対比超過リターン
        ax = axes[1,2]
        bars = ax.bar(self.comparative_df['ETF'], self.comparative_df['Excess_vs_Underlying']*100, 
                     color=colors, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('原資産対比超過リターン', fontweight='bold')
        ax.set_ylabel('超過リターン (%)')
        self._add_value_labels(ax, bars, '{:.1f}%')
        
        # 8. 週次配当回数
        ax = axes[1,3]
        bars = ax.bar(self.comparative_df['ETF'], self.comparative_df['Weekly_Dividends'], 
                     color=colors, alpha=0.8)
        ax.set_title('週次配当支払い回数', fontweight='bold')
        ax.set_ylabel('支払い回数')
        self._add_value_labels(ax, bars, '{:.0f}回')
        
        # 9. セクター別リターン（散布図）
        ax = axes[2,0]
        for i, (_, row) in enumerate(self.comparative_df.iterrows()):
            ax.scatter(row['Volatility']*100, row['Total_Return']*100, 
                      s=200, color=colors[i], alpha=0.8, label=row['ETF'])
            ax.annotate(row['ETF'], (row['Volatility']*100, row['Total_Return']*100),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax.set_xlabel('ボラティリティ (%)')
        ax.set_ylabel('総合リターン (%)')
        ax.set_title('リスク・リターン分布', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 10. 配当一貫性
        ax = axes[2,1]
        bars = ax.bar(self.comparative_df['ETF'], self.comparative_df['Dividend_Consistency']*100, 
                     color='green', alpha=0.7)
        ax.set_title('配当一貫性スコア', fontweight='bold')
        ax.set_ylabel('一貫性 (%)')
        self._add_value_labels(ax, bars, '{:.1f}%')
        
        # 11. 現在価格比較
        ax = axes[2,2]
        bars = ax.bar(self.comparative_df['ETF'], self.comparative_df['Current_Price'], 
                     color=colors, alpha=0.8)
        ax.set_title('現在価格', fontweight='bold')
        ax.set_ylabel('価格 ($)')
        self._add_value_labels(ax, bars, '${:.2f}')
        
        # 12. 平均週次配当
        ax = axes[2,3]
        bars = ax.bar(self.comparative_df['ETF'], self.comparative_df['Avg_Weekly_Div'], 
                     color=colors, alpha=0.8)
        ax.set_title('平均週次配当額', fontweight='bold')
        ax.set_ylabel('配当額 ($)')
        self._add_value_labels(ax, bars, '${:.3f}')
        
        # 13-15. パフォーマンス時系列（利用可能なETF）
        for i, (etf_symbol, data_dict) in enumerate(list(self.processed_data.items())[:3]):
            ax = axes[3,i]
            etf_data = data_dict['etf_data']
            if not etf_data.empty:
                ax.plot(etf_data.index, etf_data['Cumulative_Total_Return']*100, 
                       color=colors[i], linewidth=2, label=f'{etf_symbol} 総合リターン')
                ax.set_title(f'{etf_symbol} パフォーマンス推移', fontweight='bold')
                ax.set_ylabel('累積リターン (%)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 16. 総合ランキング表
        ax = axes[3,3]
        ax.axis('off')
        ranking_data = self.comparative_df[['ETF', 'Overall_Rank', 'Total_Return', 'Dividend_Yield', 'Sharpe_Ratio']].copy()
        ranking_data['Total_Return'] = ranking_data['Total_Return'].apply(lambda x: f"{x*100:.1f}%")
        ranking_data['Dividend_Yield'] = ranking_data['Dividend_Yield'].apply(lambda x: f"{x*100:.1f}%")
        ranking_data['Sharpe_Ratio'] = ranking_data['Sharpe_Ratio'].apply(lambda x: f"{x:.2f}")
        ranking_data = ranking_data.sort_values('Overall_Rank')
        
        table = ax.table(cellText=ranking_data.values,
                        colLabels=['ETF', 'ランク', 'リターン', '配当', 'シャープ'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('総合ランキング', fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        return fig
    
    def _add_value_labels(self, ax, bars, format_str):
        """バーチャートに値ラベル追加"""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                   format_str.format(height), ha='center', va='bottom', fontweight='bold')
    
    def export_comprehensive_analysis(self):
        """
        包括的分析結果エクスポート
        """
        print("💾 統合分析結果エクスポート...")
        
        # 1. 比較サマリーCSV
        if not self.comparative_df.empty:
            summary_path = os.path.join(self.docs_dir, 'rex_family_comprehensive_analysis.csv')
            self.comparative_df.to_csv(summary_path, index=False)
            print(f"  ✅ 比較分析: {summary_path}")
        
        # 2. 個別ETF詳細データ
        for etf_symbol, data_dict in self.processed_data.items():
            etf_data = data_dict['etf_data']
            if not etf_data.empty:
                detail_path = os.path.join(self.docs_dir, f'{etf_symbol.lower()}_detailed_analysis.csv')
                etf_data.to_csv(detail_path)
                print(f"  ✅ {etf_symbol}詳細: {detail_path}")
        
        # 3. 統合レポート（Markdown）
        report_path = os.path.join(self.docs_dir, 'rex_family_integrated_report.md')
        self._generate_integrated_report(report_path)
        print(f"  ✅ 統合レポート: {report_path}")
        
        # 4. メタデータ（JSON）
        metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'etf_count': len(self.processed_data),
            'data_period': self.strategy_config['benchmark_period'],
            'rex_family_config': self.rex_family,
            'strategy_config': self.strategy_config
        }
        
        metadata_path = os.path.join(self.docs_dir, 'rex_analysis_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"  ✅ メタデータ: {metadata_path}")
        
        return {
            'summary_csv': summary_path if not self.comparative_df.empty else None,
            'report_md': report_path,
            'metadata_json': metadata_path
        }
    
    def _generate_integrated_report(self, output_path):
        """統合レポート生成"""
        report_content = f"""# REX Growth & Income ETF Family - 統合分析レポート

**生成日時:** {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## エグゼクティブサマリー

REX Shares社のGrowth & Income ETFファミリーは、革新的な「50%カバードコール戦略」により、
高配当利回りと成長参加の両方を実現する新世代のETF群です。

### ファミリー概要

"""
        
        if not self.comparative_df.empty:
            best_performer = self.comparative_df.loc[self.comparative_df['Total_Return'].idxmax()]
            highest_dividend = self.comparative_df.loc[self.comparative_df['Dividend_Yield'].idxmax()]
            best_sharpe = self.comparative_df.loc[self.comparative_df['Sharpe_Ratio'].idxmax()]
            
            report_content += f"""
### 主要ハイライト

- **最優秀リターン:** {best_performer['ETF']} ({best_performer['Total_Return']*100:.2f}%)
- **最高配当利回り:** {highest_dividend['ETF']} ({highest_dividend['Dividend_Yield']*100:.2f}%)
- **最優秀リスク調整後リターン:** {best_sharpe['ETF']} (シャープレシオ: {best_sharpe['Sharpe_Ratio']:.3f})

## 個別ETF分析

"""
            
            for _, row in self.comparative_df.iterrows():
                etf_symbol = row['ETF']
                config = self.rex_family[etf_symbol]
                
                report_content += f"""### {etf_symbol} - {config['name']}

**セクター:** {config['sector']}  
**原資産:** {row['Underlying']} ({config['underlying_name']})  
**投資テーマ:** {config['description']}

**パフォーマンス指標:**
- 現在価格: ${row['Current_Price']:.2f}
- 総合リターン: {row['Total_Return']*100:.2f}%
- 年率リターン: {row['Annualized_Return']*100:.2f}%
- 配当利回り: {row['Dividend_Yield']*100:.2f}%
- ボラティリティ: {row['Volatility']*100:.2f}%
- シャープレシオ: {row['Sharpe_Ratio']:.3f}
- 最大ドローダウン: {row['Max_Drawdown']*100:.2f}%
- 推定レバレッジ: {row['Leverage_Estimate']:.2f}x
- 原資産対比超過リターン: {row['Excess_vs_Underlying']*100:+.2f}%

**配当実績:**
- 週次配当回数: {row['Weekly_Dividends']:.0f}回
- 平均週次配当: ${row['Avg_Weekly_Div']:.3f}
- 配当一貫性: {row['Dividend_Consistency']*100:.1f}%

"""
        
        report_content += """
## REX戦略の特徴

### 共通戦略パラメータ
- **目標レバレッジ:** 1.25倍 (範囲: 1.05-1.50倍)
- **カバードコール:** 50%部分カバレッジ
- **上昇余地:** 50%の部分で無制限
- **分配頻度:** 週次
- **リスク管理:** カバードコールによる下落保護

### 戦略の優位性
1. **高配当利回り:** カバードコールプレミアムによる週次分配
2. **成長参加:** 50%の無制限上昇余地
3. **リスク緩和:** カバードコールによる下落保護
4. **セクター分散:** AI、暗号通貨、EV等の高成長テーマ

### 投資家への示唆
- **インカム重視投資家:** 高い週次配当利回り
- **成長参加投資家:** 50%の無制限上昇余地
- **リスク管理志向:** 従来の個別株投資より低リスク
- **セクター分散:** 複数のREX ETFでテーマ分散

---
*REX統合分析システムにより生成*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def run_complete_analysis(self):
        """
        完全統合分析実行
        """
        print("🚀 REX統合分析システム開始")
        print("=" * 60)
        
        try:
            # 1. データ取得
            self.fetch_all_data_parallel(period=self.strategy_config['benchmark_period'])
            
            # 2. データ処理
            self.process_all_data()
            
            # 3. 比較分析
            comparative_df = self.generate_comparative_analysis()
            
            # 4. ダッシュボード作成
            dashboard_fig = self.create_unified_dashboard()
            
            # 5. ダッシュボード保存
            if dashboard_fig:
                dashboard_path = os.path.join(self.docs_dir, 'rex_family_integrated_dashboard.png')
                dashboard_fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
                print(f"📊 統合ダッシュボード: {dashboard_path}")
                plt.close(dashboard_fig)
            
            # 6. 分析結果エクスポート
            export_paths = self.export_comprehensive_analysis()
            
            # 7. サマリー表示
            print("\n" + "=" * 60)
            print("📊 REX統合分析結果")
            print("=" * 60)
            
            if not comparative_df.empty:
                print("\n🏆 総合ランキング:")
                ranking = comparative_df.sort_values('Overall_Rank')[['ETF', 'Overall_Rank', 'Total_Return', 'Dividend_Yield', 'Sharpe_Ratio']]
                for _, row in ranking.iterrows():
                    print(f"  {int(row['Overall_Rank'])}位: {row['ETF']} - リターン{row['Total_Return']*100:.1f}%, 配当{row['Dividend_Yield']*100:.1f}%, シャープ{row['Sharpe_Ratio']:.2f}")
                
                print(f"\n📈 最優秀パフォーマンス:")
                best = comparative_df.loc[comparative_df['Total_Return'].idxmax()]
                print(f"  {best['ETF']}: {best['Total_Return']*100:.2f}% (セクター: {best['Sector']})")
                
                print(f"\n💰 最高配当利回り:")
                div_best = comparative_df.loc[comparative_df['Dividend_Yield'].idxmax()]
                print(f"  {div_best['ETF']}: {div_best['Dividend_Yield']*100:.2f}% (週平均: ${div_best['Avg_Weekly_Div']:.3f})")
            
            print(f"\n📂 出力ファイル:")
            for key, path in export_paths.items():
                if path:
                    print(f"  {key}: {path}")
            
            print("\n✅ REX統合分析完了!")
            return True
            
        except Exception as e:
            print(f"❌ 分析エラー: {e}")
            raise

def main():
    """
    メイン実行関数
    """
    analyzer = UnifiedREXSystem()
    return analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()