# REX Growth & Income 比較データ契約

Issue #1 の比較プロダクトでは、2026年2月26日にREX Sharesが列挙したsingle-stock Growth & Income ETF 9本を履歴上のfund masterとして保持します。ただし、9本をすべて現役fundとして扱いません。

REX Sharesは2026年5月26日、REX COIN Growth & Income ETF、REX CRWV Growth & Income ETF、REX HOOD Growth & Income ETF、REX LLY Growth & Income ETF、REX MSTR Growth & Income ETF、REX PLTR Growth & Income ETFを含む7 fundについて、2026年6月16日の清算を公式に告知しました。7本目はfund-of-fundsのREX Growth & Income Universe ETF (GIF) で、Issue #1の9本single-stock masterには含めません。

そのため `data/rex-growth-income-funds.json` は次の境界を固定します。

- 2026-02-26の公式universe announcementにあるsingle-stock fund 9本を欠落なく保持する。
- NVII / TSII / WMTIは2026-08-11時点のREX公式product pageが現在情報を掲載しているため `ACTIVE` とする。
- MSII / COII / HOII / PLTI / CWII / LLIIは `LIQUIDATED` とし、`liquidation_date=2026-06-16` を必須にする。
- 清算済みfundの履歴は比較・研究用として保持できるが、「現在の9銘柄比較」「現在の9本へalert」と表示してはならない。
- Distribution Rate、30-Day SEC Yield、ROC、NAV、market price、holdingsなどの観測値は今後別snapshotとして保存し、fund masterへ上書きしない。
- repository側model outputは公式観測値とは別contractに置く。

## 一次情報

- 2026-02-26 universe announcement: https://www.rexshares.com/rex-shares-launches-rex-growth-income-universe-etf/
- 2026-05-26 liquidation notice: https://www.rexshares.com/rex-growth-income-etfs-to-liquidate-june-16-2026/
- NVII: https://www.rexshares.com/nvii/
- TSII: https://www.rexshares.com/tsii/
- WMTI: https://www.rexshares.com/wmti/

## 次のデータゲート

1. active 3本について公式product pageから観測snapshotと分配履歴を取り込む。
2. liquidated 6本は清算日以前に存在した公式履歴だけをarchiveとして取り込む。
3. source URL / as-of / observed_atを各snapshotで必須にする。
4. holdings coverageを計算し95%未満を警告する。
5. UIでは `ACTIVE` と `LIQUIDATED` を明示し、historical comparisonとcurrent monitoringを混同しない。

この文書は投資助言ではなく、repository内のデータ品質・表示境界を定義するものです。
