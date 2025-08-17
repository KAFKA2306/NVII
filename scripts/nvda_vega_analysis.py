#!/usr/bin/env python3
"""
NVIDIA (NVDA) Vega Analysis Script
Calculates and visualizes the evolution of vega for NVDA options over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NVDAVegaAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.045  # 4.5% risk-free rate
        self.nvda_price = 140.0  # Current NVDA price estimate
        self.dividend_yield = 0.0025  # NVDA dividend yield (quarterly)
        
    def black_scholes_vega(self, S, K, T, r, sigma, q=0):
        """
        Calculate vega using Black-Scholes formula with dividend adjustment
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield
        """
        if T <= 0:
            return 0
            
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        
        # Vega = S * sqrt(T) * N'(d1) * exp(-q*T)
        vega = S * np.sqrt(T) * norm.pdf(d1) * np.exp(-q*T)
        
        return vega / 100  # Convert to percentage points
    
    def calculate_vega_evolution(self, strike_prices, initial_volatility=0.35, days_to_expiration=30):
        """
        Calculate vega evolution over time for multiple strike prices
        """
        time_points = np.linspace(days_to_expiration, 1, days_to_expiration)  # Days to expiration
        results = {}
        
        for strike in strike_prices:
            vegas = []
            for days_left in time_points:
                T = days_left / 365.0  # Convert to years
                vega = self.black_scholes_vega(
                    S=self.nvda_price,
                    K=strike,
                    T=T,
                    r=self.risk_free_rate,
                    sigma=initial_volatility,
                    q=self.dividend_yield
                )
                vegas.append(vega)
            
            results[f'Strike ${strike}'] = {
                'time_points': time_points,
                'vegas': vegas,
                'moneyness': strike / self.nvda_price
            }
        
        return results
    
    def calculate_vega_vs_volatility(self, strike_price, time_to_expiration=30):
        """
        Calculate vega sensitivity to volatility changes
        """
        volatilities = np.linspace(0.15, 0.65, 50)
        T = time_to_expiration / 365.0
        
        vegas = []
        for vol in volatilities:
            vega = self.black_scholes_vega(
                S=self.nvda_price,
                K=strike_price,
                T=T,
                r=self.risk_free_rate,
                sigma=vol,
                q=self.dividend_yield
            )
            vegas.append(vega)
        
        return volatilities, vegas
    
    def plot_vega_evolution(self, vega_data):
        """
        Plot vega evolution over time for different strikes
        """
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Vega Evolution Over Time
        plt.subplot(2, 2, 1)
        for strike_label, data in vega_data.items():
            plt.plot(data['time_points'], data['vegas'], 
                    label=f"{strike_label} (M={data['moneyness']:.2f})", 
                    linewidth=2, marker='o', markersize=3)
        
        plt.xlabel('Days to Expiration')
        plt.ylabel('Vega (per 1% volatility change)')
        plt.title('NVDA Option Vega Evolution Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()  # Invert x-axis so time flows left to right
        
        # Plot 2: Vega vs Moneyness at Different Time Points
        plt.subplot(2, 2, 2)
        time_snapshots = [30, 15, 7, 1]  # Days to expiration
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, days in enumerate(time_snapshots):
            moneyness_values = []
            vega_values = []
            
            for strike_label, data in vega_data.items():
                moneyness_values.append(data['moneyness'])
                # Find closest time point
                closest_idx = np.argmin(np.abs(np.array(data['time_points']) - days))
                vega_values.append(data['vegas'][closest_idx])
            
            plt.plot(moneyness_values, vega_values, 
                    label=f'{days} days to expiration', 
                    color=colors[i], marker='o', linewidth=2)
        
        plt.xlabel('Moneyness (Strike/Spot)')
        plt.ylabel('Vega')
        plt.title('Vega vs Moneyness at Different Times')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Vega vs Volatility
        plt.subplot(2, 2, 3)
        atm_strike = self.nvda_price
        volatilities, vegas = self.calculate_vega_vs_volatility(atm_strike)
        
        plt.plot(volatilities * 100, vegas, 'purple', linewidth=3, marker='o', markersize=4)
        plt.xlabel('Implied Volatility (%)')
        plt.ylabel('Vega')
        plt.title(f'Vega vs Volatility (ATM Strike ${atm_strike})')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Vega Heat Map
        plt.subplot(2, 2, 4)
        strikes = np.linspace(120, 160, 20)
        times = np.linspace(1, 30, 20)
        vega_matrix = np.zeros((len(times), len(strikes)))
        
        for i, t in enumerate(times):
            for j, k in enumerate(strikes):
                T = t / 365.0
                vega_matrix[i, j] = self.black_scholes_vega(
                    S=self.nvda_price, K=k, T=T, 
                    r=self.risk_free_rate, sigma=0.35, q=self.dividend_yield
                )
        
        im = plt.imshow(vega_matrix, aspect='auto', cmap='viridis', 
                       extent=[strikes[0], strikes[-1], times[0], times[-1]])
        plt.colorbar(im, label='Vega')
        plt.xlabel('Strike Price ($)')
        plt.ylabel('Days to Expiration')
        plt.title('NVDA Vega Heat Map')
        
        # Add current price line
        plt.axvline(x=self.nvda_price, color='red', linestyle='--', 
                   label=f'Current Price ${self.nvda_price}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('/home/kafka/projects/NVII/nvda_vega_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_statistics(self, vega_data):
        """
        Generate summary statistics for vega analysis
        """
        print("\n" + "="*60)
        print("NVIDIA (NVDA) VEGA ANALYSIS SUMMARY")
        print("="*60)
        print(f"Current NVDA Price: ${self.nvda_price}")
        print(f"Risk-Free Rate: {self.risk_free_rate*100:.1f}%")
        print(f"Dividend Yield: {self.dividend_yield*100:.2f}%")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\nVEGA EVOLUTION ANALYSIS:")
        print("-" * 40)
        
        for strike_label, data in vega_data.items():
            max_vega = max(data['vegas'])
            min_vega = min(data['vegas'])
            initial_vega = data['vegas'][0]
            final_vega = data['vegas'][-1]
            
            print(f"\n{strike_label}:")
            print(f"  Moneyness: {data['moneyness']:.3f}")
            print(f"  Max Vega: {max_vega:.4f}")
            print(f"  Min Vega: {min_vega:.4f}")
            print(f"  Initial Vega (30 days): {initial_vega:.4f}")
            print(f"  Final Vega (1 day): {final_vega:.4f}")
            print(f"  Vega Decay: {((final_vega - initial_vega) / initial_vega * 100):.1f}%")
        
        # ATM Vega Analysis
        print("\nATM VEGA VOLATILITY SENSITIVITY:")
        print("-" * 40)
        volatilities, vegas = self.calculate_vega_vs_volatility(self.nvda_price)
        max_vega_vol = vegas[np.argmax(vegas)]
        optimal_vol = volatilities[np.argmax(vegas)]
        
        print(f"Maximum Vega: {max_vega_vol:.4f}")
        print(f"Optimal Volatility: {optimal_vol*100:.1f}%")
        print(f"Vega at 20% vol: {vegas[np.argmin(np.abs(volatilities - 0.20))]:0.4f}")
        print(f"Vega at 35% vol: {vegas[np.argmin(np.abs(volatilities - 0.35))]:0.4f}")
        print(f"Vega at 50% vol: {vegas[np.argmin(np.abs(volatilities - 0.50))]:0.4f}")

def main():
    analyzer = NVDAVegaAnalyzer()
    
    # Define strike prices around current NVDA price
    strike_prices = [130, 135, 140, 145, 150]  # Around current price
    
    print("Calculating NVDA vega evolution...")
    vega_data = analyzer.calculate_vega_evolution(strike_prices, days_to_expiration=30)
    
    print("Generating visualizations...")
    analyzer.plot_vega_evolution(vega_data)
    
    print("Generating summary statistics...")
    analyzer.generate_summary_statistics(vega_data)
    
    print("\nAnalysis complete! Charts saved as 'nvda_vega_analysis.png'")

if __name__ == "__main__":
    main()