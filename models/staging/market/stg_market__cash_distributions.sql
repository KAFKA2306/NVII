select
    ticker::varchar as ticker,
    cast(ex_date as date) as ex_date,
    cash_amount_usd::double as cash_amount_usd,
    currency::varchar as currency,
    distribution_type::varchar as distribution_type,
    frequency::integer as frequency,
    source_provider::varchar as source_provider,
    source_endpoint::varchar as source_endpoint,
    observed_at::varchar as observed_at,
    'observed'::varchar as provenance_type
from read_csv_auto('data/market-dividends-20260803-20260924.csv', header=true)
