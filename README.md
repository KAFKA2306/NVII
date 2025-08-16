# NVII Financial Analysis Project

Sophisticated options pricing analysis and portfolio strategy modeling for the REX NVDA Growth & Income ETF (NVII), focusing on covered call strategies on NVIDIA-leveraged ETF positions.

## Overview

This project analyzes a hybrid portfolio strategy: implementing covered calls on 50% of positions while maintaining unlimited upside potential on the remaining 50% (ポートフォリオの約半分に対してカバードコール戦略を実施し、残り半分は無制限の上昇余地を残します).

## Project Structure

```
NVII/
├── scripts/
│   ├── option_analysis.py              # Phase 1: Black-Scholes pricing & NVII analysis
│   └── phase2_portfolio_analysis.py    # Phase 2: Portfolio strategy & risk analysis
├── docs/
│   └── NVII.md                         # Detailed ETF information and background
├── analysis/
│   └── phase1_summary.md               # Phase 1 analysis results and findings
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd NVII

# Install dependencies
pip install -r requirements.txt
```

### Running Analysis

```bash
# Run Phase 1: Option pricing analysis
python3 scripts/option_analysis.py

# Run Phase 2: Portfolio strategy analysis
python3 scripts/phase2_portfolio_analysis.py
```

## Key Features

### 🧮 Advanced Option Pricing
- Complete Black-Scholes implementation with Greeks calculation
- Implied volatility computation using Newton-Raphson method
- NVII-specific leverage adjustments (1.05x to 1.50x range)

### 📊 Portfolio Strategy Modeling
- 50% covered call / 50% unlimited upside allocation
- Monte Carlo scenario analysis across market conditions
- Comprehensive risk metrics (Sharpe, Sortino ratios)

### 🎯 NVIDIA Volatility Modeling
- Four volatility regimes: Low (35%), Medium (55%), High (75%), Extreme (100%)
- Weekly covered call strategy optimization
- Leverage-adjusted option premium calculations

## Analysis Results

The project demonstrates that:
- Weekly covered call strategies can generate **15.6% to 76.2%** annual yields
- 50/50 allocation provides optimal risk-adjusted returns
- Strategy outperforms pure NVIDIA buy-and-hold in most scenarios
- Leverage amplifies option premiums by exactly the leverage factor

## Financial Assumptions

| Parameter | Value |
|-----------|-------|
| Current NVII Price | $32.97 |
| Target Leverage | 1.25x |
| Risk-free Rate | 4.5% |
| Current Dividend Yield | 6.30% |
| Option Strategy | Weekly 5% OTM calls |

## Dependencies

- `numpy` - Numerical computations
- `scipy` - Statistical functions and optimization
- `pandas` - Data manipulation and analysis
- `matplotlib` - Visualization (Phase 2)

## Documentation

- **[NVII.md](docs/NVII.md)** - Detailed ETF information and background
- **[Phase 1 Summary](analysis/phase1_summary.md)** - Comprehensive analysis results

## License

This project is for educational and research purposes in quantitative finance modeling.


完全に破壊的に再設計します。現在のアーキテクチャは根本的に間違っています。

  Update Todos
  ⎿  ☐ 完全に新しいアーキテクチャ設計 - レバレッジETFの本質を理解した正しいモデル
     ☐ 配当調整済みBlack-Scholesの実装 - NVII配当6.30%を正しく組み込み
     ☐ レバレッジの正しい適用 - プレミアムではなくエクスポージャーへの影響
     ☐ リアルな取引コストモデル - ビッドアスク、手数料、流動性制約
     ☐ 歴史的データに基づく変動率モデル - 恣意的な数値を排除
     ☐ 統合されたポートフォリオシミュレーション - 現実的なリスク・リターン分析
     ☐ CLAUDE.mdとREADMEの完全書き直し - 新しいアーキテクチャに対応# NVII
# NVII
