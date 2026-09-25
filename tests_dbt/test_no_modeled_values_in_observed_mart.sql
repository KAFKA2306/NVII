select *
from {{ ref('mart_option_income_etf_daily') }}
where provenance_type <> 'observed'
   or metric_status not in (
        'OBSERVED_MARKET_WINDOW',
        'OBSERVED_SNAPSHOT_MARKET_DATA_GAP'
   )
