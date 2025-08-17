# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際の指針を提供します。

## リポジトリ概要

これは REX NVDA Growth & Income ETF (NVII) に特化した高度な金融分析プロジェクトです。本リポジトリには、レバレッジETFオプション価格決定および包括的なポートフォリオ戦略モデリングのための、数学的に厳密なPythonスクリプトが含まれています。

## プロジェクト構造

```
NVII/
├── scripts/
│   ├── advanced_option_engine.py       # 高度オプション価格決定エンジン（配当調整済みBlack-Scholes）
│   └── portfolio_simulation_engine.py  # 統合ポートフォリオシミュレーションシステム
├── docs/
│   ├── ARCHITECTURE_REDESIGN_REPORT.md # アーキテクチャ再設計レポート
│   ├── academic_paper.md               # 学術論文
│   └── mathematical_research_proposal.md # 数学的研究提案
├── requirements.txt                    # Python依存関係
└── README.md                          # プロジェクト概要（日本語）
```

## 開発コマンド

### 環境設定
```bash
# 依存関係のインストール
pip install -r requirements.txt
```

### 分析スクリプトの実行
```bash
# 高度オプション価格決定エンジンの実行
python3 scripts/advanced_option_engine.py

# 統合ポートフォリオシミュレーションの実行
python3 scripts/portfolio_simulation_engine.py
```

## コードアーキテクチャ（v2.0 - 完全再設計）

### 核心コンポーネント

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

## 重要な指示

1. **現在のアーキテクチャ**: `advanced_option_engine.py` および `portfolio_simulation_engine.py` のみを使用してください

2. **数学的精度**: 全ての計算は配当調整とレバレッジ動態を適切に考慮する必要があります

3. **現実的な前提**: 理論的な数値ではなく、実際の市場データと取引コストに基づく分析を行ってください

4. **依存関係更新**: 新しい依存関係 `pandas-datareader` および `investiny` が追加されています