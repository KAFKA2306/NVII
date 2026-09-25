select *
from {{ ref('mart_option_income_etf_daily') }}
where provenance_type <> 'observed'
   or metric_status <> 'OBSERVED_SNAPSHOT_NO_COMPARABLE_PRICE_SERIES'
