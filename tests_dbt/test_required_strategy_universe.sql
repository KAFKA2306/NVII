with expected(underlying_ticker, ticker) as (
    values
      ('NVDA', 'NVDA'),
      ('NVDA', 'NVII'),
      ('NVDA', 'NVDY'),
      ('NVDA', 'NVIT'),
      ('TSLA', 'TSLA'),
      ('TSLA', 'TSII'),
      ('TSLA', 'TSLY'),
      ('TSLA', 'TEST')
),
actual as (
    select underlying_ticker, ticker
    from {{ ref('mart_underlying_strategy_comparison') }}
)

select e.*
from expected e
left join actual a
  on a.underlying_ticker = e.underlying_ticker
 and a.ticker = e.ticker
where a.ticker is null
