select
    ticker::varchar as ticker,
    cast(date as date) as date,
    close_usd::double as close_usd,
    source_provider::varchar as source_provider,
    source_endpoint::varchar as source_endpoint,
    observed_at::varchar as observed_at,
    'observed'::varchar as provenance_type
from read_csv_auto('data/market-daily-20260803-20260924.csv', header=true)
