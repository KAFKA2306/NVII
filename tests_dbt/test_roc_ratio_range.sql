select *
from {{ ref('mart_option_income_etf_daily') }}
where roc_ratio_percent is not null
  and (roc_ratio_percent < 0 or roc_ratio_percent > 100)
