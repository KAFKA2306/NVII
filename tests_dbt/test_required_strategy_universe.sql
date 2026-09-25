with expected(underlying_ticker, ticker) as (
    values
      ('NVDA', 'NVDA'),
      ('NVDA', 'NVII'),
      ('NVDA', 'NVDY'),
      ('NVDA', 'NVIT'),
      ('TSLA', 'TSLA'),
      ('TSLA', 'TSII'),
      ('TSLA', 'TSLY'),
      ('TSLA', 'TEST'),
      ('AMD', 'AMD'),
      ('AMD', 'AMDY'),
      ('TSM', 'TSM'),
      ('TSM', 'TSMY'),
      ('PLTR', 'PLTR'),
      ('PLTR', 'PLTY'),
      ('MSTR', 'MSTR'),
      ('MSTR', 'MSTY'),
      ('COIN', 'COIN'),
      ('COIN', 'CONY'),
      ('HOOD', 'HOOD'),
      ('HOOD', 'HOOY'),
      ('RDDT', 'RDDT'),
      ('RDDT', 'RDYY'),
      ('SMCI', 'SMCI'),
      ('SMCI', 'SMCY')
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
