with tsii as (
    select *
    from {{ ref('mart_underlying_strategy_comparison') }}
    where ticker = 'TSII'
)
select *
from tsii
where comparison_status <> 'OBSERVED_SNAPSHOT_MARKET_DATA_GAP'
