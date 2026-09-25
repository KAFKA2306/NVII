select
    ticker::varchar as ticker,
    issuer::varchar as issuer,
    instrument_type::varchar as instrument_type,
    underlying_ticker::varchar as underlying_ticker,
    strategy_type::varchar as strategy_type,
    distribution_frequency::varchar as distribution_frequency,
    official_url::varchar as official_url
from {{ ref('instrument_master') }}
