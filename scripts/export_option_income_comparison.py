#!/usr/bin/env python3
"""Export the dbt comparison mart to the static Pages contract."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

OUTPUT = Path("data/option-income-comparison.json")


def _normalize_numbers(value):
    """Keep the committed JSON stable when SQL DOUBLE values are integral."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_numbers(item) for key, item in value.items()}
    return value


def main() -> int:
    con = duckdb.connect("nvii.duckdb", read_only=True)
    cursor = con.execute(
        """
        select
            underlying_ticker,
            ticker,
            issuer,
            instrument_type,
            strategy_type,
            distribution_frequency,
            official_url,
            cast(observation_date as varchar) as observation_date,
            nav_usd,
            market_price_usd,
            distribution_rate_percent,
            sec_yield_percent,
            roc_ratio_percent,
            expense_ratio_percent,
            expense_ratio_basis,
            leverage,
            cast(performance_window_start as varchar) as performance_window_start,
            cast(performance_window_end as varchar) as performance_window_end,
            price_observation_count,
            round(total_return_percent, 6) as total_return_percent,
            round(underlying_total_return_percent, 6) as underlying_total_return_percent,
            round(upside_capture_percent, 6) as upside_capture_percent,
            round(downside_capture_percent, 6) as downside_capture_percent,
            source_url,
            provenance_type,
            comparison_status
        from main.mart_underlying_strategy_comparison
        order by
            underlying_ticker,
            case when instrument_type = 'UNDERLYING' then 0 else 1 end,
            ticker
        """
    )
    columns = [item[0] for item in cursor.description]
    rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    payload = {
        "schema_version": "option-income-comparison.v1",
        "generated_from": "main.mart_underlying_strategy_comparison",
        "rows": rows,
    }
    payload = _normalize_numbers(payload)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
