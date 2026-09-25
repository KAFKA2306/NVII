# Option Income ETF dbt layer

Issue #15 generalizes the repository from a REX-only comparison into an issuer-neutral option-income ETF data model.

## Reproduce

```bash
python -m pip install -r requirements-dbt.txt
dbt seed --profiles-dir .
dbt build --profiles-dir .
```

The local warehouse is DuckDB (`nvii.duckdb`). Raw issuer observations remain versioned files under `data/`; dbt does not overwrite them.

## Canonical flow

```
issuer official observation
  -> staging/rex or staging/yieldmax
  -> int_option_income_*
  -> dim_instrument / fct_distribution
  -> mart_option_income_etf_daily
  -> mart_underlying_strategy_comparison
```

`mart_underlying_strategy_comparison` groups spot reference instruments with option-income ETFs by underlying. It currently guarantees the NVDA group (NVDA/NVII/NVDY/NVIT) and TSLA group (TSLA/TSII/TSLY/TEST).

## Provenance rule

Observed issuer values use `provenance_type = observed`. Distribution Rate, SEC Yield, realized distributions and estimated ROC remain separate fields. Total return and up/down capture are calculated from the versioned 2026-08-03 through 2026-09-24 market close/distribution window. NAV decay remains NULL because the repository does not yet hold a comparable daily NAV series. TSII is explicitly marked `OBSERVED_SNAPSHOT_MARKET_DATA_GAP` because the market-data provider resolves the ticker but returned no daily bars for the window; the mart does not substitute market price or another ticker.

## Adding an issuer

Add an issuer-specific staging model that emits the same columns as `stg_rex__observations` and `stg_rex__distributions`, then union it into the intermediate models. Do not add issuer-specific columns to the marts.
