# Mathematical Formulations for NVII Option Pricing Analysis

## 1. Black-Scholes Model

### Call Option Price Formula
```
C = S₀e⁻ᵍᵀ N(d₁) - Ke⁻ʳᵀ N(d₂)
```

Where:
- **C** = Call option price
- **S₀** = Current stock price ($32.97)
- **K** = Strike price
- **T** = Time to expiration (years)
- **r** = Risk-free rate (4.5%)
- **q** = Dividend yield (6.3%)
- **σ** = Volatility (45%)

### d₁ and d₂ Calculations
```
d₁ = [ln(S₀/K) + (r - q + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Greeks Formulations

#### Delta (Δ)
```
Δ = e⁻ᵍᵀ N(d₁)
```
*Interpretation: Rate of change of option price with respect to underlying price*

#### Gamma (Γ)
```
Γ = (e⁻ᵍᵀ φ(d₁)) / (S₀σ√T)
```
*Where φ(d₁) is the standard normal probability density function*

#### Theta (Θ)
```
Θ = [-S₀φ(d₁)σe⁻ᵍᵀ/(2√T) - rKe⁻ʳᵀN(d₂) + qS₀e⁻ᵍᵀN(d₁)] / 365
```
*Interpretation: Time decay per day*

#### Vega (ν)
```
ν = (S₀e⁻ᵍᵀφ(d₁)√T) / 100
```
*Interpretation: Sensitivity to 1% change in volatility*

#### Rho (ρ)
```
ρ = (KTe⁻ʳᵀN(d₂)) / 100
```
*Interpretation: Sensitivity to 1% change in risk-free rate*

## 2. Binomial Tree Model

### Parameters
```
Δt = T/n
u = e^(σ√Δt)           # Up factor
d = 1/u = e^(-σ√Δt)    # Down factor
p = (e^((r-q)Δt) - d)/(u - d)  # Risk-neutral probability
```

### Asset Price Evolution
```
S(i,j) = S₀ × u^(n-j) × d^j
```
*Where i = time step, j = number of down moves*

### Option Value Recursion
```
C(i,j) = e^(-rΔt) × [p × C(i+1,j) + (1-p) × C(i+1,j+1)]
```

### American Exercise Condition
```
C(i,j) = max(C(i,j), S(i,j) - K)
```

## 3. Monte Carlo Simulation

### Geometric Brownian Motion
```
S(t+Δt) = S(t) × exp[(r - q - σ²/2)Δt + σ√Δt × Z]
```
*Where Z ~ N(0,1) is a standard normal random variable*

### Price Path Generation
```
S₁ = S₀ × exp[(r - q - σ²/2)Δt + σ√Δt × Z₁]
S₂ = S₁ × exp[(r - q - σ²/2)Δt + σ√Δt × Z₂]
...
Sₙ = Sₙ₋₁ × exp[(r - q - σ²/2)Δt + σ√Δt × Zₙ]
```

### Option Value Calculation
```
C = e^(-rT) × (1/N) × Σᵢ₌₁ᴺ max(Sᵢ(T) - K, 0)
```
*Where N = number of simulation paths*

## 4. NVII-Specific Calculations

### Enhanced Yield Calculation
```
Enhanced_Yield = (Annual_Option_Income) / (Portfolio_Value)
Annual_Option_Income = Weekly_Premium × 52 × Number_of_Contracts
Number_of_Contracts = (Portfolio_Value × 0.5) / (S₀ × 100)
```

### Leveraged Return Impact
```
Leveraged_Return = Base_Return × Leverage_Ratio
Leverage_Alpha = Leveraged_Return - Base_Return
```

### Risk-Adjusted Return (Sharpe Ratio)
```
Sharpe_Ratio = (Total_Return - Risk_Free_Rate) / (Volatility × √Leverage_Ratio)
```

### Weekly Income Projection
```
Weekly_Income = Option_Premium × Contracts × 100 × Coverage_Ratio
Coverage_Ratio = 0.5 (50% of portfolio)
Contracts = (AUM × Coverage_Ratio) / (S₀ × 100)
```

## 5. Volatility Models

### Historical Volatility Calculation
```
σ_historical = √(252) × √[(1/(n-1)) × Σᵢ₌₁ⁿ (rᵢ - r̄)²]
```
*Where rᵢ = ln(Sᵢ/Sᵢ₋₁) are daily returns*

### Implied Volatility (Newton-Raphson Method)
```
σₙ₊₁ = σₙ - [BS(σₙ) - Market_Price] / Vega(σₙ)
```
*Iterate until convergence*

## 6. Risk Metrics

### Value at Risk (VaR) - 95% Confidence
```
VaR_95% = Portfolio_Value × [μ - 1.645σ] × √T
```
*Where μ = expected return, σ = portfolio volatility*

### Maximum Drawdown Estimate
```
Max_Drawdown ≈ σ × √(2ln(T×252))
```
*For continuous monitoring over period T*

### Assignment Probability
```
P(Assignment) = N(d₂) = N(d₁ - σ√T)
```
*Probability that option finishes in-the-money*

## 7. Portfolio Construction Formulas

### Optimal Strike Selection
```
Optimal_Strike = S₀ × (1 + k)
```
*Where k optimizes: E[Premium Income] - λ × Var[Assignment Risk]*

### Position Sizing
```
Position_Size = (Target_Income / Option_Premium) × 100
Max_Position = Portfolio_Value × Coverage_Ratio / (S₀ × 100)
```

### Rebalancing Trigger
```
Rebalance_Trigger = |Current_Leverage - Target_Leverage| > Threshold
```

## 8. Performance Attribution

### Total Return Decomposition
```
Total_Return = Dividend_Yield + Option_Income_Yield + Leverage_Alpha + Residual
```

### Information Ratio
```
IR = (Strategy_Return - Benchmark_Return) / Tracking_Error
```

### Alpha Generation
```
α = Strategy_Return - [β × Market_Return + (1-β) × Risk_Free_Rate]
```

## 9. Stress Testing Formulas

### Scenario Analysis
```
Stressed_Portfolio_Value = Initial_Value × (1 + Leverage × Price_Shock)
Stressed_Option_Value = BS(S_stressed, K, T, r, σ_stressed, q)
```

### Correlation Impact
```
Portfolio_Volatility = √(w₁²σ₁² + w₂²σ₂² + 2w₁w₂σ₁σ₂ρ₁₂)
```
*Where w = weights, σ = volatilities, ρ = correlation*

## 10. Implementation Parameters

### Transaction Cost Model
```
Total_Cost = Commission + Bid_Ask_Spread + Market_Impact
Market_Impact = γ × √(Volume / Average_Daily_Volume)
```

### Slippage Estimation
```
Slippage = α × (Order_Size / Market_Depth)^β
```
*Where α and β are market-specific parameters*

---

## Example Calculation: 5% OTM Call Option

### Given Parameters:
- S₀ = $32.97
- K = $34.62 (5% OTM)
- T = 1/52 = 0.0192 years
- r = 0.045
- q = 0.063
- σ = 0.45

### Step-by-Step Calculation:

1. **Calculate d₁:**
```
d₁ = [ln(32.97/34.62) + (0.045 - 0.063 + 0.45²/2) × 0.0192] / (0.45 × √0.0192)
d₁ = [-0.0485 + 0.00176] / 0.0625 = -0.7493
```

2. **Calculate d₂:**
```
d₂ = -0.7493 - 0.45 × √0.0192 = -0.8118
```

3. **Calculate N(d₁) and N(d₂):**
```
N(d₁) = N(-0.7493) = 0.2269
N(d₂) = N(-0.8118) = 0.2084
```

4. **Calculate Call Price:**
```
C = 32.97 × e^(-0.063×0.0192) × 0.2269 - 34.62 × e^(-0.045×0.0192) × 0.2084
C = 32.97 × 0.9988 × 0.2269 - 34.62 × 0.9991 × 0.2084
C = 7.463 - 7.204 = $0.259
```

### Greeks Calculations:

**Delta:**
```
Δ = e^(-0.063×0.0192) × 0.2269 = 0.2245
```

**Gamma:**
```
Γ = (e^(-0.063×0.0192) × φ(-0.7493)) / (32.97 × 0.45 × √0.0192)
Γ = (0.9988 × 0.3011) / (32.97 × 0.0625) = 0.1455
```

This mathematical framework provides the foundation for all theoretical calculations used in the NVII option pricing analysis.