# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an advanced financial engineering project specialized for REX NVDA Growth & Income ETF (NVII) analysis. The repository contains mathematically rigorous Python scripts for leveraged ETF option pricing and comprehensive portfolio strategy modeling.

## Project Structure

```
NVII/
├── scripts/
│   ├── advanced_option_engine.py       # Advanced option pricing engine (dividend-adjusted Black-Scholes)
│   └── portfolio_simulation_engine.py  # Integrated portfolio simulation system
├── docs/
│   ├── ARCHITECTURE_REDESIGN_REPORT.md # Architecture redesign report
│   ├── COMPREHENSIVE_EDUCATIONAL_ANALYSIS_REPORT.md # Educational analysis
│   ├── EXECUTIVE_SUMMARY.md            # Executive summary
│   ├── NVII論文.md                     # Japanese academic paper
│   ├── STUDENT_VISUAL_GUIDE.md         # Visual guide for students
│   ├── academic_paper.md               # Academic paper (English)
│   ├── mathematical_research_proposal.md # Mathematical research proposal
│   └── records.md                      # Analysis records
├── requirements.txt                    # Python dependencies
├── README.md                          # Project overview (Japanese)
└── CLAUDE.md                          # This file
```

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Analysis Scripts
```bash
# Run advanced option pricing engine
python3 scripts/advanced_option_engine.py

# Run integrated portfolio simulation
python3 scripts/portfolio_simulation_engine.py

# Alternative execution with explicit PYTHONPATH (if import issues occur)
PYTHONPATH=/home/kafka/projects/NVII python3 scripts/portfolio_simulation_engine.py
```

### Testing and Validation
- No formal test suite exists; scripts include validation through historical data analysis
- Both scripts run comprehensive checks and output detailed results to verify mathematical accuracy
- Scripts will output analysis results directly when executed with `python3`

### Script Execution Context
- Scripts are designed to run as standalone modules from the repository root
- Both scripts include `if __name__ == "__main__":` blocks that execute comprehensive analysis
- Expected runtime: 30-60 seconds for full analysis including Monte Carlo simulations
- Output includes detailed financial metrics, risk analysis, and market regime comparisons

## Code Architecture (v2.0 - Complete Redesign)

### Core Components

1. **DividendAdjustedBlackScholes クラス** (`advanced_option_engine.py`)
   - 配当調整を含む数学的に正確なBlack-Scholesオプション価格決定
   - 適切な配当利回り考慮（NVII 6.30%）
   - 完全なGreeksの計算（Delta, Gamma, Theta, Vega）
   - インプライドボラティリティ計算（Brentの方法）

2. **LeveragedETFDynamics クラス** (`advanced_option_engine.py`)
   - レバレッジETFの複雑な動態の正確なモデリング
   - 日次リバランシングによるボラティリティドラッグの考慮
   - 相関減衰と複利効果の適切な計算
   - レバレッジがエクスポージャーに与える影響の正確な理解

3. **AdvancedCoveredCallEngine クラス** (`advanced_option_engine.py`)
   - 包括的なカバードコール戦略分析
   - 現実的な取引コストモデリング（ビッド・アスクスプレッド、手数料、スリッページ）
   - 歴史的データに基づく市場レジーム分析
   - 適切なレバレッジ適用（プレミアムではなくエクスポージャーへ）

4. **AdvancedPortfolioAnalyzer クラス** (`portfolio_simulation_engine.py`)
   - モンテカルロシミュレーションによる統合ポートフォリオ分析
   - 包括的リスクメトリクス（VaR、Expected Shortfall、Sharpe ratio等）
   - ストレステストとテールリスク分析
   - パフォーマンス要因分解と現実的なリターン予測

### 主要な金融モデリング機能

- **数学的精度**: 配当調整済みオプション価格決定と正確なGreeks計算
- **レバレッジの正確性**: ETFの日次リバランシング効果とボラティリティドラッグ
- **現実的コスト**: ビッド・アスクスプレッド、手数料、マーケットインパクトの完全統合
- **リスク管理**: VaR、Expected Shortfall、テール比率を含む包括的リスク分析
- **ストレステスト**: 2008年危機、テックバブル崩壊等の極端シナリオ
- **パフォーマンス分解**: 各戦略要素の寄与度の正確な分離

### 分析ワークフロー（v2.0）

1. **高度オプション分析** (`advanced_option_engine.py`): 配当調整済み理論価格とレバレッジ動態
2. **ポートフォリオシミュレーション** (`portfolio_simulation_engine.py`): 統合リスク・リターン分析
3. **ストレステスト**: 極端な市場条件下での戦略有効性評価

## 金融定数と前提条件

- **現在のNVII価格**: $32.97
- **目標レバレッジ**: 1.25x（範囲: 1.05x-1.50x）
- **リスクフリーレート**: 4.5%（3ヶ月財務省証券）
- **現在の配当利回り**: 6.30%（適切にモデルに組み込み済み）
- **ポジション配分**: 50% カバードコール、50% 無制限上昇余地

## 依存関係

高度な金融分析のためのコアPythonパッケージ:
- `numpy>=1.21.0`: 数値計算
- `scipy>=1.7.0`: 統計関数と最適化
- `pandas>=1.3.0`: データ操作と分析
- `matplotlib>=3.4.0`: 可視化
- `pandas-datareader>=0.10.0`: 金融データ取得
- `investiny>=0.8.0`: 投資データ分析

## 主要な修正点と改善

### 修正された根本的欠陥

1. **レバレッジ適用エラーの修正**:
   - 旧: プレミアムに直接レバレッジを乗算（数学的に誤り）
   - 新: エクスポージャーへの影響を正確にモデリング

2. **配当調整の追加**:
   - 旧: 配当を無視したBlack-Scholes
   - 新: 6.30%配当利回りを含む配当調整済み価格決定

3. **取引コストの統合**:
   - 旧: 取引コスト完全無視
   - 新: ビッド・アスク、手数料、スリッページの包括的モデリング

4. **現実的なボラティリティモデル**:
   - 旧: 恣意的なボラティリティレジーム
   - 新: 歴史的データに基づくレジーム分析

### 新しい分析結果の特徴

- **現実的なリターン予測**: 取引コストとレバレッジドラッグを考慮した実現可能なリターン
- **包括的リスク分析**: VaR、Expected Shortfall、テール比率を含む
- **ストレステスト**: 2008年危機レベルの極端シナリオでの戦略有効性
- **数学的厳密性**: 全ての計算が学術的基準を満たす

## Important Instructions

1. **Current Architecture**: Use only `advanced_option_engine.py` and `portfolio_simulation_engine.py` - these are the authoritative analysis engines

2. **Mathematical Precision**: All calculations must properly account for dividend adjustments and leverage dynamics

3. **Realistic Assumptions**: Base analysis on actual market data and transaction costs, not theoretical values

4. **Dependencies**: Updated dependencies include `pandas-datareader` and `investiny` for market data access

5. **Execution**: Always run scripts from repository root directory to ensure proper imports

6. **Documentation**: When modifying analysis, update both English and Japanese documentation as needed

## Current Market Context (Reference Values)

These values are embedded in the scripts but provided here for reference:
- **NVII Current Price**: $32.97
- **Target Leverage**: 1.25x (range: 1.05x-1.50x)
- **Risk-Free Rate**: 4.5% (3-month Treasury)
- **Dividend Yield**: 6.30% (properly integrated in models)
- **Position Allocation**: 50% covered calls, 50% unlimited upside