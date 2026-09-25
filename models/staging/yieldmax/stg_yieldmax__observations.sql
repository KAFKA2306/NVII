with source as (
    select observed_at, unnest(funds) as fund
    from read_json_auto('data/yieldmax-option-income-observations.json')
)

select
    fund.ticker::varchar as ticker,
    'YieldMax'::varchar as issuer,
    cast(fund.observation.as_of as date) as observation_date,
    fund.observation.nav_usd::double as nav_usd,
    fund.observation.market_price_usd::double as market_price_usd,
    fund.observation.premium_discount_percent::double as premium_discount_percent,
    fund.observation.fund_assets_usd::double as fund_assets_usd,
    fund.observation.shares_outstanding::bigint as shares_outstanding,
    fund.observation.gross_expense_ratio_percent::double as expense_ratio_percent,
    'GROSS'::varchar as expense_ratio_basis,
    cast(null as double) as leverage,
    cast(fund.distribution.as_of as date) as distribution_date,
    fund.distribution.distribution_rate_percent::double as distribution_rate_percent,
    fund.distribution.sec_yield_percent::double as sec_yield_percent,
    cast(fund.distribution.sec_yield_as_of as date) as sec_yield_date,
    fund.distribution.estimated_roc_percent::double as estimated_roc_percent,
    fund.distribution.frequency::varchar as distribution_frequency,
    fund.source_url::varchar as source_url,
    cast(observed_at as varchar) as observed_at,
    'observed'::varchar as provenance_type
from source
