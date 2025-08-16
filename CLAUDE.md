# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a financial analysis project focused on the REX NVDA Growth & Income ETF (NVII). The repository contains Python scripts for sophisticated options pricing analysis and portfolio strategy modeling, specifically analyzing covered call strategies on NVIDIA-leveraged ETF positions.
ポートフォリオの約半分に対してカバードコール戦略を実施し、残り半分は無制限の上昇余地を残します。

## Project Structure

```
NVII/
├── scripts/
│   ├── option_analysis.py      # Phase 1: Black-Scholes option pricing and NVII analysis
│   └── phase2_portfolio_analysis.py  # Phase 2: Portfolio strategy and risk analysis
├── docs/
│   └── NVII.md                 # Detailed ETF information and background
├── analysis/
│   └── phase1_summary.md       # Phase 1 analysis results and findings
└── requirements.txt            # Python dependencies
```

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Analysis Scripts
```bash
# Run Phase 1 option pricing analysis
python3 scripts/option_analysis.py

# Run Phase 2 portfolio strategy analysis  
python3 scripts/phase2_portfolio_analysis.py
```

## Code Architecture

### Core Components

1. **BlackScholesCalculator Class** (`option_analysis.py`)
   - Complete Black-Scholes option pricing implementation
   - Calculates theoretical prices, Greeks (Delta, Gamma, Theta, Vega, Rho)
   - Supports both call and put options
   - Includes implied volatility calculation using Newton-Raphson method

2. **NVIIAnalyzer Class** (`option_analysis.py`)
   - NVII-specific analysis for weekly covered call strategies
   - Models NVIDIA volatility regimes (low: 35%, medium: 55%, high: 75%, extreme: 100%)
   - Calculates leverage-adjusted option premiums (1.05x to 1.50x range)
   - Analyzes income scenarios across different strike selections

3. **Phase2PortfolioAnalyzer Class** (`phase2_portfolio_analysis.py`)
   - Advanced portfolio modeling with 50% covered call / 50% unlimited upside allocation
   - Monte Carlo-style scenario analysis across market conditions
   - Comprehensive risk metrics (Sharpe ratio, Sortino ratio, downside deviation)
   - Stress testing and leverage sensitivity analysis

### Key Financial Modeling Features

- **Weekly Option Strategy**: Models selling 5% out-of-the-money calls weekly
- **Leverage Integration**: Incorporates NVII's 1.25x target leverage into all calculations
- **Multi-Scenario Analysis**: Bull, bear, sideways, and volatile market scenarios
- **Risk Management**: Downside protection analysis and opportunity cost modeling
- **Performance Attribution**: Separates covered call income from unlimited upside gains

### Analysis Workflow

1. **Phase 1** (`option_analysis.py`): Theoretical option pricing under different volatility regimes
2. **Phase 2** (`phase2_portfolio_analysis.py`): Portfolio construction and performance analysis
3. Results are documented in `analysis/phase1_summary.md` with detailed findings

## Financial Constants and Assumptions

- **Current NVII Price**: $32.97
- **Target Leverage**: 1.25x (range: 1.05x-1.50x)
- **Risk-free Rate**: 4.5% (3-month Treasury)
- **Current Dividend Yield**: 6.30%
- **Position Coverage**: 50% covered calls, 50% unlimited upside

## Dependencies

Core Python packages for financial analysis:
- `numpy`: Numerical computations
- `scipy`: Statistical functions and optimization
- `pandas`: Data manipulation and analysis
- `matplotlib`: Visualization (Phase 2)

## Analysis Results Context

The repository demonstrates sophisticated quantitative finance modeling showing that:
- Weekly covered call strategies can generate 15.6% to 76.2% annual yields
- 50/50 allocation provides optimal risk-adjusted returns
- Strategy outperforms pure NVIDIA buy-and-hold in most market scenarios
- Leverage amplifies option premiums by exactly the leverage factor (25% at 1.25x)


完全に破壊的に再設計します。現在のアーキテクチャは根本的に間違っています。

  Update Todos
  ⎿  ☐ 完全に新しいアーキテクチャ設計 - レバレッジETFの本質を理解した正しいモデル
     ☐ 配当調整済みBlack-Scholesの実装 - NVII配当6.30%を正しく組み込み
     ☐ レバレッジの正しい適用 - プレミアムではなくエクスポージャーへの影響
     ☐ リアルな取引コストモデル - ビッドアスク、手数料、流動性制約
     ☐ 歴史的データに基づく変動率モデル - 恣意的な数値を排除
     ☐ 統合されたポートフォリオシミュレーション - 現実的なリスク・リターン分析
     ☐ CLAUDE.mdとREADMEの完全書き直し - 新しいアーキテクチャに対応