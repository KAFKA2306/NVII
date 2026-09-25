with observations as (
    select * from {{ ref('int_option_income_observations') }}
),
distribution_stats as (
    select
        ticker,
        count(*) as distribution_observation_count,
        avg(distribution_per_share_usd) as mean_distribution_per_share_usd,
        stddev_pop(distribution_per_share_usd) as stddev_distribution_per_share_usd
    from {{ ref('fct_distribution') }}
    group by ticker
)

select
    o.ticker || '|' || cast(o.observation_date as varchar) as snapshot_key,
    o.observation_date,
    o.ticker,
    d.issuer,
    d.underlying_ticker,
    d.strategy_type,
    d.distribution_frequency,
    o.nav_usd,
    o.market_price_usd,
    o.premium_discount_percent,
    o.fund_assets_usd,
    o.shares_outstanding,
    o.expense_ratio_percent,
    o.expense_ratio_basis,
    o.leverage,
    o.distribution_date,
    o.distribution_rate_percent,
    o.sec_yield_percent,
    o.sec_yield_date,
    o.estimated_roc_percent as roc_ratio_percent,
    s.distribution_observation_count,
    s.mean_distribution_per_share_usd,
    s.stddev_distribution_per_share_usd,
    case
        when s.mean_distribution_per_share_usd = 0 then null
        else s.stddev_distribution_per_share_usd / s.mean_distribution_per_share_usd
    end as distribution_coefficient_of_variation,
    cast(null as double) as nav_decay_percent,
    cast(null as double) as total_return_percent,
    cast(null as double) as underlying_total_return_percent,
    cast(null as double) as upside_capture_percent,
    cast(null as double) as downside_capture_percent,
    cast(null as double) as option_premium_yield_percent,
    o.source_url,
    o.observed_at,
    o.provenance_type,
    'OBSERVED_SNAPSHOT_NO_COMPARABLE_PRICE_SERIES'::varchar as metric_status
from observations o
inner join {{ ref('dim_instrument') }} d using (ticker)
left join distribution_stats s using (ticker)
where d.instrument_type = 'ETF'
