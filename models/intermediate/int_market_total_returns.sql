with prices as (
    select
        ticker,
        date,
        close_usd,
        source_provider,
        source_endpoint,
        observed_at,
        lag(close_usd) over (partition by ticker order by date) as previous_close_usd
    from {{ ref('stg_market__daily_prices') }}
),
dividends as (
    select
        ticker,
        ex_date as date,
        sum(cash_amount_usd) as cash_distribution_usd
    from {{ ref('stg_market__cash_distributions') }}
    group by 1, 2
),
daily as (
    select
        p.ticker,
        p.date,
        p.close_usd,
        coalesce(d.cash_distribution_usd, 0.0) as cash_distribution_usd,
        case
            when p.previous_close_usd is null or p.previous_close_usd = 0 then null
            else (p.close_usd + coalesce(d.cash_distribution_usd, 0.0)) / p.previous_close_usd - 1
        end as total_return_daily,
        p.source_provider,
        p.source_endpoint,
        p.observed_at
    from prices p
    left join dividends d using (ticker, date)
)

select * from daily
