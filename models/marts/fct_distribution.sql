select
    ticker || '|' || cast(declaration_date as varchar) as distribution_key,
    ticker,
    issuer,
    declaration_date,
    ex_date,
    record_date,
    payable_date,
    distribution_per_share_usd,
    estimated_roc_percent,
    source_url,
    observed_at,
    provenance_type
from {{ ref('int_option_income_distributions') }}
