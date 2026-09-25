select p.*
from {{ ref('mart_market_performance_window') }} p
join {{ ref('dim_instrument') }} i using (ticker)
where i.instrument_type = 'ETF'
  and p.ticker <> 'TSII'
  and (
    p.total_return_percent is null
    or p.underlying_total_return_percent is null
    or p.upside_capture_percent is null
    or p.downside_capture_percent is null
  )
