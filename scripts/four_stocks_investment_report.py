#!/usr/bin/env python3
"""
Four Stocks (NVDA, TSLA, MSTR, COIN) Vega-Based Investment Report
Comprehensive buy recommendation based on vega analysis of these four stocks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

class FourStocksInvestmentAnalyzer:
    def __init__(self):
        self.symbols = ['NVDA', 'TSLA', 'MSTR', 'COIN']
        self.risk_free_rate = 0.045  # 4.5%
        self.dividend_yields = {
            'NVDA': 0.0025,  # 0.25%
            'TSLA': 0.0,     # 0%
            'MSTR': 0.0,     # 0%
            'COIN': 0.0      # 0%
        }
        self.market_data = {}
        
    def fetch_comprehensive_data(self):
        """
        Fetch comprehensive market data for investment analysis
        """
        print("Fetching comprehensive market data for investment analysis...")
        
        if HAS_YFINANCE:
            for symbol in self.symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="6mo")  # 6 months for better analysis
                    info = ticker.info
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        
                        # Multiple volatility calculations
                        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                        vol_30d = returns.tail(21).std() * np.sqrt(252) if len(returns) >= 21 else returns.std() * np.sqrt(252)
                        vol_60d = returns.tail(42).std() * np.sqrt(252) if len(returns) >= 42 else returns.std() * np.sqrt(252)
                        vol_historical = returns.std() * np.sqrt(252)
                        
                        # Price performance metrics
                        price_30d = hist['Close'].iloc[-22] if len(hist) >= 22 else hist['Close'].iloc[0]
                        price_60d = hist['Close'].iloc[-43] if len(hist) >= 43 else hist['Close'].iloc[0]
                        price_ytd = hist['Close'].iloc[0]
                        
                        return_30d = (current_price - price_30d) / price_30d
                        return_60d = (current_price - price_60d) / price_60d
                        return_ytd = (current_price - price_ytd) / price_ytd
                        
                        # Volume and liquidity metrics
                        avg_volume_30d = hist['Volume'].tail(21).mean()
                        volume_trend = hist['Volume'].tail(10).mean() / hist['Volume'].tail(30).mean()
                        
                        # Technical indicators
                        sma_20 = hist['Close'].tail(20).mean()
                        sma_50 = hist['Close'].tail(50).mean() if len(hist) >= 50 else sma_20
                        price_to_sma20 = current_price / sma_20
                        price_to_sma50 = current_price / sma_50
                        
                        # Volatility metrics
                        vol_of_vol = returns.tail(60).rolling(10).std().std() * np.sqrt(252) if len(returns) >= 60 else 0
                        
                        self.market_data[symbol] = {
                            'price': current_price,
                            'volatility_30d': vol_30d,
                            'volatility_60d': vol_60d,
                            'volatility_historical': vol_historical,
                            'vol_of_vol': vol_of_vol,
                            'return_30d': return_30d,
                            'return_60d': return_60d,
                            'return_ytd': return_ytd,
                            'avg_volume_30d': avg_volume_30d,
                            'volume_trend': volume_trend,
                            'price_to_sma20': price_to_sma20,
                            'price_to_sma50': price_to_sma50,
                            'dividend_yield': self.dividend_yields[symbol],
                            'market_cap': info.get('marketCap', 0),
                            'beta': info.get('beta', 1.0),
                            'sector': info.get('sector', 'Technology'),
                            'pe_ratio': info.get('trailingPE', 0),
                            'forward_pe': info.get('forwardPE', 0)
                        }
                        
                        print(f"{symbol}: ${current_price:.2f}, Vol: {vol_30d*100:.1f}%, "
                              f"30d Return: {return_30d*100:+.1f}%")
                    
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
                    self.use_fallback_data(symbol)
        else:
            for symbol in self.symbols:
                self.use_fallback_data(symbol)
        
        return self.market_data
    
    def use_fallback_data(self, symbol):
        """
        Use fallback data based on recent market conditions
        """
        fallback_data = {
            'NVDA': {
                'price': 180.45, 'vol_30d': 0.237, 'return_30d': 0.05, 'market_cap': 4_500_000_000_000,
                'pe': 25, 'beta': 1.2, 'sector': 'Technology'
            },
            'TSLA': {
                'price': 330.56, 'vol_30d': 0.435, 'return_30d': 0.02, 'market_cap': 1_000_000_000_000,
                'pe': 35, 'beta': 1.8, 'sector': 'Consumer Cyclical'
            },
            'MSTR': {
                'price': 366.32, 'vol_30d': 0.533, 'return_30d': -0.03, 'market_cap': 8_000_000_000,
                'pe': 0, 'beta': 2.5, 'sector': 'Technology'
            },
            'COIN': {
                'price': 317.55, 'vol_30d': 0.713, 'return_30d': -0.08, 'market_cap': 25_000_000_000,
                'pe': 15, 'beta': 2.8, 'sector': 'Financial Services'
            }
        }
        
        if symbol in fallback_data:
            data = fallback_data[symbol]
            self.market_data[symbol] = {
                'price': data['price'],
                'volatility_30d': data['vol_30d'],
                'volatility_60d': data['vol_30d'] * 0.95,
                'volatility_historical': data['vol_30d'] * 1.1,
                'vol_of_vol': data['vol_30d'] * 0.1,
                'return_30d': data['return_30d'],
                'return_60d': data['return_30d'] * 1.5,
                'return_ytd': data['return_30d'] * 3,
                'avg_volume_30d': 10_000_000,
                'volume_trend': 1.0,
                'price_to_sma20': 1.02,
                'price_to_sma50': 1.05,
                'dividend_yield': self.dividend_yields[symbol],
                'market_cap': data['market_cap'],
                'beta': data['beta'],
                'sector': data['sector'],
                'pe_ratio': data['pe'],
                'forward_pe': data['pe'] * 0.9
            }
            print(f"{symbol}: ${data['price']:.2f}, Vol: {data['vol_30d']*100:.1f}% (fallback)")
    
    def calculate_investment_scores(self):
        """
        Calculate comprehensive investment scores based on vega and other factors
        """
        print("Calculating comprehensive investment scores...")
        
        scores = {}
        
        # Calculate vega for 30-day ATM options for each stock
        vega_data = {}
        
        for symbol, data in self.market_data.items():
            S = data['price']
            sigma = data['volatility_30d']
            q = data['dividend_yield']
            T = 30 / 365.0  # 30 days
            
            # ATM vega calculation
            d1 = (np.log(1.0) + (self.risk_free_rate - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            vega = S * np.sqrt(T) * norm.pdf(d1) * np.exp(-q*T) / 100
            
            # Additional metrics
            call_price = S * (norm.cdf(d1) * np.exp(-q*T) - norm.cdf(d1 - sigma*np.sqrt(T)) * np.exp(-self.risk_free_rate*T))
            vega_per_dollar = vega / S
            vega_efficiency = vega / (sigma * S)
            
            vega_data[symbol] = {
                'vega': vega,
                'vega_per_dollar': vega_per_dollar,
                'vega_efficiency': vega_efficiency,
                'call_price': call_price
            }
        
        # Calculate normalized scores (0-100)
        max_vega = max(vega_data[s]['vega'] for s in self.symbols)
        max_vol = max(self.market_data[s]['volatility_30d'] for s in self.symbols)
        max_return = max(self.market_data[s]['return_30d'] for s in self.symbols)
        min_return = min(self.market_data[s]['return_30d'] for s in self.symbols)
        
        for symbol in self.symbols:
            data = self.market_data[symbol]
            vega_info = vega_data[symbol]
            
            # Score components (0-100)
            vega_score = (vega_info['vega'] / max_vega) * 100
            volatility_score = (data['volatility_30d'] / max_vol) * 100
            
            # Momentum score (normalized returns)
            momentum_score = ((data['return_30d'] - min_return) / (max_return - min_return)) * 100 if max_return != min_return else 50
            
            # Liquidity score (based on market cap)
            liquidity_score = min(100, np.log10(data['market_cap']) * 10) if data['market_cap'] > 0 else 50
            
            # Technical score (price vs moving averages)
            technical_score = ((data['price_to_sma20'] - 0.95) / 0.1) * 50 + 50
            technical_score = max(0, min(100, technical_score))
            
            # Efficiency score
            max_efficiency = max(vega_data[s]['vega_efficiency'] for s in self.symbols)
            efficiency_score = (vega_info['vega_efficiency'] / max_efficiency) * 100
            
            # Risk-adjusted score (beta consideration)
            risk_adjustment = 100 - min(50, (data['beta'] - 1) * 25)  # Penalty for high beta
            
            # Sector diversification bonus
            sector_bonus = {'Technology': 0, 'Consumer Cyclical': 5, 'Financial Services': 10}.get(data['sector'], 0)
            
            # Composite score calculation
            composite_score = (
                vega_score * 0.25 +           # 25% - Vega potential
                volatility_score * 0.20 +     # 20% - Volatility for option premiums
                momentum_score * 0.15 +       # 15% - Recent performance
                liquidity_score * 0.15 +      # 15% - Market cap/liquidity
                efficiency_score * 0.10 +     # 10% - Vega efficiency
                technical_score * 0.10 +      # 10% - Technical position
                risk_adjustment * 0.05        # 5% - Risk adjustment
            ) + sector_bonus
            
            scores[symbol] = {
                'composite_score': composite_score,
                'vega_score': vega_score,
                'volatility_score': volatility_score,
                'momentum_score': momentum_score,
                'liquidity_score': liquidity_score,
                'efficiency_score': efficiency_score,
                'technical_score': technical_score,
                'risk_adjustment': risk_adjustment,
                'sector_bonus': sector_bonus,
                'vega': vega_info['vega'],
                'vega_per_dollar': vega_info['vega_per_dollar'],
                'recommendation': self.get_recommendation(composite_score)
            }
        
        return scores, vega_data
    
    def get_recommendation(self, score):
        """
        Convert score to recommendation
        """
        if score >= 85:
            return "STRONG BUY"
        elif score >= 70:
            return "BUY"
        elif score >= 55:
            return "MODERATE BUY"
        elif score >= 45:
            return "HOLD"
        elif score >= 30:
            return "SELL"
        else:
            return "STRONG SELL"
    
    def create_investment_report_visual(self, scores, vega_data):
        """
        Create comprehensive visual investment report
        """
        print("Creating investment report visualization...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # Colors for each stock
        colors = {'NVDA': 'green', 'TSLA': 'red', 'MSTR': 'orange', 'COIN': 'blue'}
        
        # Plot 1: Overall Investment Scores
        plt.subplot(3, 3, 1)
        symbols = list(scores.keys())
        composite_scores = [scores[s]['composite_score'] for s in symbols]
        
        bars = plt.bar(symbols, composite_scores, color=[colors[s] for s in symbols], alpha=0.8)
        plt.ylabel('Investment Score (0-100)')
        plt.title('Overall Investment Scores')
        plt.ylim(0, 100)
        
        # Add score labels and recommendation
        for bar, symbol, score in zip(bars, symbols, composite_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                    scores[symbol]['recommendation'], ha='center', va='center', 
                    rotation=90, fontsize=8, fontweight='bold', color='white')
        
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Score Breakdown Comparison
        plt.subplot(3, 3, 2)
        score_categories = ['vega_score', 'volatility_score', 'momentum_score', 'liquidity_score']
        x = np.arange(len(score_categories))
        width = 0.2
        
        for i, symbol in enumerate(symbols):
            values = [scores[symbol][cat] for cat in score_categories]
            plt.bar(x + i*width, values, width, label=symbol, color=colors[symbol], alpha=0.8)
        
        plt.xlabel('Score Categories')
        plt.ylabel('Score (0-100)')
        plt.title('Score Component Breakdown')
        plt.xticks(x + width*1.5, ['Vega', 'Volatility', 'Momentum', 'Liquidity'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Vega vs Price Efficiency
        plt.subplot(3, 3, 3)
        for symbol in symbols:
            vega = vega_data[symbol]['vega']
            vega_per_dollar = vega_data[symbol]['vega_per_dollar']
            plt.scatter(vega_per_dollar * 1000, vega, s=200, color=colors[symbol], 
                       alpha=0.8, label=symbol, edgecolor='black', linewidth=2)
        
        plt.xlabel('Vega per $1000 Invested')
        plt.ylabel('Absolute Vega')
        plt.title('Vega Efficiency Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Risk vs Return Profile
        plt.subplot(3, 3, 4)
        for symbol in symbols:
            data = self.market_data[symbol]
            risk = data['volatility_30d'] * 100
            return_30d = data['return_30d'] * 100
            score = scores[symbol]['composite_score']
            
            plt.scatter(risk, return_30d, s=score*5, color=colors[symbol], 
                       alpha=0.7, label=f"{symbol} (Score: {score:.1f})", 
                       edgecolor='black', linewidth=1)
        
        plt.xlabel('30-Day Volatility (%)')
        plt.ylabel('30-Day Return (%)')
        plt.title('Risk vs Return Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 5: Market Cap vs Vega
        plt.subplot(3, 3, 5)
        for symbol in symbols:
            market_cap = self.market_data[symbol]['market_cap']
            vega = vega_data[symbol]['vega']
            plt.scatter(market_cap, vega, s=200, color=colors[symbol], 
                       alpha=0.8, label=symbol, edgecolor='black', linewidth=2)
        
        plt.xscale('log')
        plt.xlabel('Market Cap ($)')
        plt.ylabel('ATM Vega (30-day)')
        plt.title('Market Cap vs Vega Relationship')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Technical Analysis
        plt.subplot(3, 3, 6)
        technical_scores = [scores[s]['technical_score'] for s in symbols]
        momentum_scores = [scores[s]['momentum_score'] for s in symbols]
        
        plt.scatter(technical_scores, momentum_scores, s=200, 
                   c=[colors[s] for s in symbols], alpha=0.8, edgecolor='black', linewidth=2)
        
        for i, symbol in enumerate(symbols):
            plt.annotate(symbol, (technical_scores[i], momentum_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        plt.xlabel('Technical Score')
        plt.ylabel('Momentum Score')
        plt.title('Technical vs Momentum Analysis')
        plt.grid(True, alpha=0.3)
        
        # Plot 7: Investment Recommendation Summary
        plt.subplot(3, 3, 7)
        plt.axis('off')
        
        # Create recommendation summary
        sorted_symbols = sorted(symbols, key=lambda x: scores[x]['composite_score'], reverse=True)
        
        text_content = "INVESTMENT RANKING\n" + "="*20 + "\n\n"
        for i, symbol in enumerate(sorted_symbols, 1):
            score = scores[symbol]['composite_score']
            rec = scores[symbol]['recommendation']
            price = self.market_data[symbol]['price']
            vega = vega_data[symbol]['vega']
            
            text_content += f"{i}. {symbol}\n"
            text_content += f"   Score: {score:.1f}/100\n"
            text_content += f"   Rec: {rec}\n"
            text_content += f"   Price: ${price:.2f}\n"
            text_content += f"   Vega: {vega:.4f}\n\n"
        
        plt.text(0.05, 0.95, text_content, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        # Plot 8: Sector Allocation Recommendation
        plt.subplot(3, 3, 8)
        top_2_symbols = sorted_symbols[:2]
        allocation_pct = [60, 40] if len(top_2_symbols) >= 2 else [100]
        pie_colors = [colors[s] for s in top_2_symbols]
        
        if len(top_2_symbols) >= 2:
            plt.pie(allocation_pct, labels=top_2_symbols, colors=pie_colors, autopct='%1.1f%%',
                   startangle=90, explode=(0.05, 0.05))
        else:
            plt.pie([100], labels=top_2_symbols, colors=pie_colors, autopct='%1.1f%%')
        
        plt.title('Recommended Portfolio Allocation\n(Top Performers)')
        
        # Plot 9: Risk Metrics Summary
        plt.subplot(3, 3, 9)
        risk_metrics = []
        for symbol in symbols:
            data = self.market_data[symbol]
            risk_score = scores[symbol]['risk_adjustment']
            beta = data['beta']
            vol = data['volatility_30d'] * 100
            
            risk_metrics.append({
                'symbol': symbol,
                'beta': beta,
                'volatility': vol,
                'risk_score': risk_score
            })
        
        # Plot beta vs volatility
        betas = [m['beta'] for m in risk_metrics]
        vols = [m['volatility'] for m in risk_metrics]
        
        plt.scatter(betas, vols, s=200, c=[colors[m['symbol']] for m in risk_metrics], 
                   alpha=0.8, edgecolor='black', linewidth=2)
        
        for metric in risk_metrics:
            plt.annotate(metric['symbol'], (metric['beta'], metric['volatility']), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        plt.xlabel('Beta (Market Sensitivity)')
        plt.ylabel('30-Day Volatility (%)')
        plt.title('Risk Profile Analysis')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/kafka/projects/NVII/four_stocks_investment_report.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_investment_report(self, scores, vega_data):
        """
        Generate detailed written investment report
        """
        print("\n" + "="*80)
        print("FOUR STOCKS VEGA-BASED INVESTMENT REPORT")
        print("NVDA | TSLA | MSTR | COIN")
        print("="*80)
        print(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analysis Focus: Option Vega Characteristics for Investment Decision")
        
        # Sort by investment score
        sorted_symbols = sorted(self.symbols, key=lambda x: scores[x]['composite_score'], reverse=True)
        
        print(f"\nEXECUTIVE SUMMARY:")
        print("-" * 50)
        print(f"Top Recommendation: {sorted_symbols[0]} (Score: {scores[sorted_symbols[0]]['composite_score']:.1f}/100)")
        print(f"Investment Grade: {scores[sorted_symbols[0]]['recommendation']}")
        
        best_performer = sorted_symbols[0]
        best_data = self.market_data[best_performer]
        best_vega = vega_data[best_performer]['vega']
        
        print(f"\nKey Investment Highlights:")
        print(f"• Highest Vega Exposure: {best_vega:.4f}")
        print(f"• Current Price: ${best_data['price']:.2f}")
        print(f"• 30-Day Volatility: {best_data['volatility_30d']*100:.1f}%")
        print(f"• Recent Performance: {best_data['return_30d']*100:+.1f}%")
        
        print(f"\nINVESTMENT RANKINGS:")
        print("-" * 50)
        
        for i, symbol in enumerate(sorted_symbols, 1):
            score_data = scores[symbol]
            market_data = self.market_data[symbol]
            vega_info = vega_data[symbol]
            
            print(f"\n{i}. {symbol} - {score_data['recommendation']}")
            print(f"   Overall Score: {score_data['composite_score']:.1f}/100")
            print(f"   Current Price: ${market_data['price']:.2f}")
            print(f"   ATM Vega (30d): {vega_info['vega']:.4f}")
            print(f"   Volatility: {market_data['volatility_30d']*100:.1f}%")
            print(f"   30-Day Return: {market_data['return_30d']*100:+.1f}%")
            print(f"   Market Cap: ${market_data['market_cap']/1_000_000_000:.1f}B")
            print(f"   Beta: {market_data['beta']:.1f}")
            
            # Score breakdown
            print(f"   Score Breakdown:")
            print(f"     • Vega Score: {score_data['vega_score']:.1f}/100")
            print(f"     • Volatility Score: {score_data['volatility_score']:.1f}/100")
            print(f"     • Momentum Score: {score_data['momentum_score']:.1f}/100")
            print(f"     • Liquidity Score: {score_data['liquidity_score']:.1f}/100")
        
        print(f"\nINVESTMENT STRATEGY RECOMMENDATIONS:")
        print("-" * 50)
        
        top_pick = sorted_symbols[0]
        second_pick = sorted_symbols[1] if len(sorted_symbols) > 1 else None
        
        print(f"PRIMARY INVESTMENT: {top_pick}")
        print(f"• Allocation: 60-70% of options portfolio")
        print(f"• Strategy: Long call options, volatility plays")
        print(f"• Target: ATM or slightly OTM calls, 30-45 day expiration")
        print(f"• Rationale: Highest vega score with strong fundamentals")
        
        if second_pick:
            print(f"\nSECONDARY INVESTMENT: {second_pick}")
            print(f"• Allocation: 30-40% of options portfolio")
            print(f"• Strategy: Diversification play, covered calls if holding stock")
            print(f"• Target: ATM calls or protective strategies")
            print(f"• Rationale: Portfolio diversification and risk management")
        
        print(f"\nVEGA ANALYSIS INSIGHTS:")
        print("-" * 50)
        
        max_vega_symbol = max(self.symbols, key=lambda x: vega_data[x]['vega'])
        max_efficiency_symbol = max(self.symbols, key=lambda x: vega_data[x]['vega_per_dollar'])
        
        print(f"• Highest Absolute Vega: {max_vega_symbol} ({vega_data[max_vega_symbol]['vega']:.4f})")
        print(f"• Best Vega Efficiency: {max_efficiency_symbol} ({vega_data[max_efficiency_symbol]['vega_per_dollar']*1000:.2f} per $1000)")
        print(f"• Volatility Leader: {max(self.symbols, key=lambda x: self.market_data[x]['volatility_30d'])}")
        
        # Calculate portfolio vega exposure
        total_vega = sum(vega_data[s]['vega'] for s in self.symbols)
        print(f"• Combined Portfolio Vega: {total_vega:.4f}")
        print(f"• Avg Vega per Stock: {total_vega/len(self.symbols):.4f}")
        
        print(f"\nRISK ASSESSMENT:")
        print("-" * 50)
        
        avg_vol = np.mean([self.market_data[s]['volatility_30d'] for s in self.symbols])
        high_vol_stocks = [s for s in self.symbols if self.market_data[s]['volatility_30d'] > avg_vol]
        
        print(f"• Average Volatility: {avg_vol*100:.1f}%")
        print(f"• High Volatility Stocks: {', '.join(high_vol_stocks)}")
        print(f"• Portfolio Beta Range: {min(self.market_data[s]['beta'] for s in self.symbols):.1f} - {max(self.market_data[s]['beta'] for s in self.symbols):.1f}")
        
        print(f"\nRisk Factors:")
        print(f"• High volatility exposure requires careful position sizing")
        print(f"• Technology sector concentration risk")
        print(f"• Options time decay risk for long positions")
        print(f"• Market correlation risk during broad selloffs")
        
        print(f"\nPORTFOLIO CONSTRUCTION GUIDELINES:")
        print("-" * 50)
        
        print(f"Recommended Allocation Strategy:")
        print(f"• Core Position (50-60%): {sorted_symbols[0]} - Highest scoring stock")
        if len(sorted_symbols) > 1:
            print(f"• Satellite Position (25-35%): {sorted_symbols[1]} - Diversification")
        if len(sorted_symbols) > 2:
            print(f"• Tactical Position (10-15%): {sorted_symbols[2]} - Opportunistic")
        print(f"• Cash Reserve (10-15%): For volatility spikes and opportunities")
        
        print(f"\nOption Strategy Recommendations:")
        for symbol in sorted_symbols[:2]:  # Top 2 picks
            rec = scores[symbol]['recommendation']
            vol = self.market_data[symbol]['volatility_30d'] * 100
            
            if rec in ['STRONG BUY', 'BUY']:
                print(f"\n{symbol} Option Strategies:")
                print(f"  • Long Calls: ATM or 5% OTM, 30-45 days")
                print(f"  • Call Spreads: If premium is expensive")
                print(f"  • Straddles: If expecting big moves (Vol: {vol:.1f}%)")
                if self.market_data[symbol]['volatility_30d'] > 0.5:
                    print(f"  • Iron Condors: High vol environment, sell premium")
        
        print(f"\nMONITORING AND EXIT CRITERIA:")
        print("-" * 50)
        
        print(f"Position Monitoring:")
        print(f"• Daily: Check implied volatility changes")
        print(f"• Weekly: Review position delta and theta decay")
        print(f"• Monthly: Reassess vega rankings and scores")
        
        print(f"\nExit Triggers:")
        print(f"• Profit target: 50-100% gain on options")
        print(f"• Stop loss: 30-50% loss on position")
        print(f"• Time decay: Exit with <7 days to expiration")
        print(f"• Volatility collapse: Exit if IV drops >20%")
        
        print(f"\nDISCLAIMER:")
        print("-" * 50)
        print("This analysis is based on quantitative models and historical data.")
        print("Options trading involves substantial risk of loss and is not suitable")
        print("for all investors. Past performance does not guarantee future results.")
        print("Please consult with a qualified financial advisor before investing.")

def main():
    analyzer = FourStocksInvestmentAnalyzer()
    
    print("Four Stocks Vega-Based Investment Analysis")
    print("Analyzing: NVDA, TSLA, MSTR, COIN")
    print("="*50)
    
    # Fetch comprehensive data
    analyzer.fetch_comprehensive_data()
    
    # Calculate investment scores
    scores, vega_data = analyzer.calculate_investment_scores()
    
    # Create visual report
    analyzer.create_investment_report_visual(scores, vega_data)
    
    # Generate written report
    analyzer.generate_investment_report(scores, vega_data)
    
    print(f"\nAnalysis completed successfully!")
    print(f"Investment report saved as: 'four_stocks_investment_report.png'")

if __name__ == "__main__":
    main()