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
    l.source_url,
    l.observed_at,
    l.provenance_type,
    case
        when i.instrument_type = 'UNDERLYING' then 'REFERENCE_INSTRUMENT_NO_OBSERVATION'
        when l.ticker is null then 'UNVERIFIED_NO_OBSERVATION'
        else 'OBSERVED_SNAPSHOT'
    end as comparison_status
from instruments i
left join latest l using (ticker)
