with instruments as (
    select * from {{ ref('dim_instrument') }}
),
latest as (
    select * from {{ ref('mart_option_income_etf_daily') }}
    qualify row_number() over (partition by ticker order by observation_date desc) = 1
)

select
    i.underlying_ticker || '|' || i.ticker as comparison_key,
    i.underlying_ticker,
    i.ticker,
    i.issuer,
    i.instrument_type,
    i.strategy_type,
    i.distribution_frequency,
    i.official_url,
    l.observation_date,
    l.nav_usd,
    l.market_price_usd,
    l.distribution_rate_percent,
    l.sec_yield_percent,
    l.roc_ratio_percent,
    l.expense_ratio_percent,
    l.expense_ratio_basis,
    l.leverage,
    l.performance_window_start,
    l.performance_window_end,
    l.price_observation_count,
    l.total_return_percent,
    l.underlying_total_return_percent,
    l.upside_capture_percent,
    l.downside_capture_percent,
    l.source_url,
    l.observed_at,
    l.provenance_type,
    case
        when i.instrument_type = 'UNDERLYING' then 'REFERENCE_INSTRUMENT'
        when l.ticker is null then 'UNVERIFIED_NO_OBSERVATION'
        when l.metric_status = 'OBSERVED_SNAPSHOT_MARKET_DATA_GAP' then 'OBSERVED_SNAPSHOT_MARKET_DATA_GAP'
        else 'OBSERVED_SNAPSHOT_WITH_MARKET_PERFORMANCE'
    end as comparison_status
from instruments i
left join latest l using (ticker)
