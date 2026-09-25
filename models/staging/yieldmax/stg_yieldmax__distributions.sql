with source as (
    select observed_at, unnest(funds) as fund
    from read_json_auto('data/yieldmax-option-income-observations.json')
),
expanded as (
    select
        observed_at,
        fund.ticker as ticker,
        fund.source_url as source_url,
        unnest(fund.distributions) as distribution
    from source
)

select
    ticker::varchar as ticker,
    'YieldMax'::varchar as issuer,
    cast(distribution.declaration_date as date) as declaration_date,
    cast(distribution.ex_date as date) as ex_date,
    cast(distribution.record_date as date) as record_date,
    cast(distribution.payable_date as date) as payable_date,
    distribution.distribution_per_share_usd::double as distribution_per_share_usd,
    distribution.roc_percent::double as estimated_roc_percent,
    source_url::varchar as source_url,
    cast(observed_at as varchar) as observed_at,
    'observed'::varchar as provenance_type
from expanded
