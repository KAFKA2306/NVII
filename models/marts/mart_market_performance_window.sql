with daily as (
    select
        r.*,
        i.underlying_ticker,
        i.instrument_type
    from {{ ref('int_market_total_returns') }} r
    inner join {{ ref('dim_instrument') }} i using (ticker)
),
summary as (
    select
        ticker,
        underlying_ticker,
        min(date) as window_start,
        max(date) as window_end,
        count(*) as price_observation_count,
        exp(sum(ln(1 + total_return_daily)) filter (where total_return_daily is not null)) - 1 as total_return
    from daily
    group by 1, 2
),
captures as (
    select
        e.ticker,
        e.underlying_ticker,
        exp(sum(ln(1 + e.total_return_daily)) filter (
            where u.total_return_daily > 0 and e.total_return_daily is not null
        )) - 1 as etf_up_return,
        exp(sum(ln(1 + u.total_return_daily)) filter (
            where u.total_return_daily > 0
        )) - 1 as underlying_up_return,
        exp(sum(ln(1 + e.total_return_daily)) filter (
            where u.total_return_daily < 0 and e.total_return_daily is not null
        )) - 1 as etf_down_return,
        exp(sum(ln(1 + u.total_return_daily)) filter (
            where u.total_return_daily < 0
        )) - 1 as underlying_down_return
    from daily e
    inner join daily u
      on u.ticker = e.underlying_ticker
     and u.date = e.date
    where e.instrument_type = 'ETF'
    group by 1, 2
)

select
    s.ticker,
    s.underlying_ticker,
    s.window_start,
    s.window_end,
    s.price_observation_count,
    s.total_return * 100 as total_return_percent,
    u.total_return * 100 as underlying_total_return_percent,
    case
        when c.underlying_up_return is null or c.underlying_up_return = 0 then null
        else c.etf_up_return / c.underlying_up_return * 100
    end as upside_capture_percent,
    case
        when c.underlying_down_return is null or c.underlying_down_return = 0 then null
        else c.etf_down_return / c.underlying_down_return * 100
    end as downside_capture_percent,
    case
        when s.price_observation_count < 2 then 'INSUFFICIENT_PRICE_HISTORY'
        else 'OBSERVED_MARKET_WINDOW'
    end as performance_status
from summary s
left join summary u on u.ticker = s.underlying_ticker
left join captures c using (ticker, underlying_ticker)
