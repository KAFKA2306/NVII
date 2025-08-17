# NVII カバードコール戦略分析システム

## 概要

REX NVDA Growth & Income ETF (NVII) に特化した高度な金融工学分析プラットフォームです。本システムは、レバレッジETFの複雑な動態を正確にモデリングし、カバードコール戦略の包括的なリスク・リターン分析を提供します。


## 主要機能

### 🎯 数学的精度
- **配当調整済みBlack-Scholes**: NVII配当利回り6.30%を適切に考慮
- **正確なレバレッジモデリング**: ETFの日次リバランシング効果とボラティリティドラッグ
- **包括的Greeks計算**: Delta、Gamma、Theta、Vega の正確な算出

### 💰 現実的な取引コストモデル
- **ビッド・アスクスプレッド**: オプション流動性に基づく実際のスプレッド
- **手数料とスリッページ**: 実際の取引コストの完全統合
- **マーケットインパクト**: 大口取引の価格影響モデリング

### 📊 高度リスク分析
- **VaR & Expected Shortfall**: 95%信頼水準での下方リスク
- **ストレステスト**: 2008年危機、テックバブル崩壊等の極端シナリオ
- **テールリスク分析**: 厚い尾を持つリターン分布の考慮

### 🎲 モンテカルロシミュレーション
- **現実的価格パス**: GARCH風ボラティリティクラスタリング
- **レジーム遷移**: 市場ボラティリティレジームの確率的変化
- **相関破綻**: ストレス時の相関構造変化

## クイックスタート

### 1. 環境設定

```bash
# リポジトリのクローン
git clone [repository_url]
cd NVII

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. 基本分析の実行

```bash
# 高度オプション価格決定エンジン
python3 scripts/advanced_option_engine.py

# 統合ポートフォリオシミュレーション
python3 scripts/portfolio_simulation_engine.py
```

## 使用例

### 基本的なカバードコール分析

```python
from scripts.advanced_option_engine import AdvancedCoveredCallEngine, MarketRegime

# エンジンの初期化
engine = AdvancedCoveredCallEngine()

# 標準的な5% OTMカバードコール分析
result = engine.analyze_covered_call_strategy(
    regime=MarketRegime.NORMAL,
    otm_percentage=0.05,
    time_to_expiry=7/365  # 1週間
)

print(f"週次プレミアム: ${result['net_premium']:.4f}")
print(f"年間利回り: {result['annualized_yield']:.1%}")
print(f"シャープ比率相当: {result['greeks']['delta']:.3f}")
```

### ストレステスト実行

```python
from scripts.portfolio_simulation_engine import AdvancedPortfolioAnalyzer

# アナライザーの初期化
analyzer = AdvancedPortfolioAnalyzer()

# 2008年危機シナリオでのストレステスト
stress_result = analyzer.simulate_portfolio_performance(
    regime=MarketRegime.CRISIS,
    stress_scenario='2008_crisis',
    time_horizon_years=1.0
)

risk_metrics = stress_result['risk_metrics']
print(f"最大ドローダウン: {risk_metrics.maximum_drawdown:.2%}")
print(f"95% VaR: {risk_metrics.value_at_risk_95:.2%}")
```

## アーキテクチャ

### コアモジュール

1. **`advanced_option_engine.py`**
   - 配当調整済みBlack-Scholesエンジン
   - レバレッジETF動態モデリング
   - 高度カバードコール分析

2. **`portfolio_simulation_engine.py`**
   - モンテカルロポートフォリオシミュレーション
   - 包括的リスクメトリクス
   - ストレステストフレームワーク

### 主要クラス

| クラス | 機能 | 特徴 |
|--------|------|------|
| `DividendAdjustedBlackScholes` | オプション価格決定 | 配当調整、正確なGreeks |
| `LeveragedETFDynamics` | レバレッジETFモデリング | 日次リバランシング、相関減衰 |
| `AdvancedCoveredCallEngine` | カバードコール分析 | 取引コスト統合、レジーム分析 |
| `AdvancedPortfolioAnalyzer` | ポートフォリオ最適化 | モンテカルロ、ストレステスト |

## 分析結果の解釈

### リスクメトリクス

- **シャープ比率**: リスク調整後リターンの効率性
- **ソルティーノ比率**: 下方偏差のみを考慮したリスク調整指標
- **95% VaR**: 5%の確率で発生する最大損失
- **Expected Shortfall**: VaRを超える条件付き期待損失

### パフォーマンス要因分解

1. **カバードコールプレミアム収入**: 週次オプション売却からの収益
2. **無制限上昇余地**: カバーされていない50%部分の資本増価
3. **配当収入**: NVII配当利回り6.30%からの収益
4. **取引コスト**: 手数料、スプレッド、スリッページの影響
5. **レバレッジ増幅**: 1.25x レバレッジによる収益増幅効果



## パフォーマンス指標

### 現実的なリターン予測 (v2.0)

| 市場レジーム | 週次プレミアム | 年間利回り期待値 | 取引コスト控除後 |
|-------------|---------------|-----------------|-----------------|
| 低ボラティリティ | $0.08-0.12 | 8-12% | 7-10% |
| 通常 | $0.12-0.18 | 12-18% | 10-15% |
| 高ボラティリティ | $0.18-0.28 | 18-28% | 15-23% |
| 危機時 | $0.25-0.40 | 25-40% | 20-32% |

**注**: v1.0の「15.6%-76.2%」は数学的に不正確でした。

## リスク警告

⚠️ **重要**: このシステムは教育・研究目的です。実際の投資には以下のリスクがあります：

- **レバレッジリスク**: NVII の1.25x レバレッジによる損失増幅
- **ボラティリティリスク**: NVIDIA株の高いボラティリティ
- **流動性リスク**: オプション市場の流動性制約
- **アサインメントリスク**: ITMオプションの早期権利行使
- **相関リスク**: ストレス時の相関破綻


## 📋 
https://github.com/KAFKA2306/NVII/blob/main/docs/academic_paper.md

- **タイトル**: "レバレッジETFにおけるカバードコール戦略の最適化：NVII の包括的分析"
- 数学的導出と理論的基盤（配当調整済みBlack-Scholes）
- 実証分析結果（モンテカルロシミュレーション）
- ストレステスト（2008年危機、テックバブル崩壊等）
- 学術的貢献と実務的示唆

### 📈 実装済み新機能

✅ **配当調整済みBlack-Scholes**（NVII 6.30%対応）  
✅ **正確なレバレッジモデリング**（ETF動態考慮）  
✅ **現実的取引コストモデル**（ビッド・アスク、手数料）  
✅ **包括的リスクメトリクス**（VaR、ES、テール分析）  
✅ **ストレステスト**（2008年危機、テックバブル等）  
✅ **日本語ドキュメント完備**  