#!/usr/bin/env python3
"""Validate current official observations for active REX Growth & Income funds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

ACTIVE = {"NVII", "TSII", "WMTI"}


def validate(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "rex-growth-income-active-observations.v1":
        errors.append("unsupported schema_version")
    if not payload.get("observed_at"):
        errors.append("missing observed_at")
    funds = payload.get("funds")
    if not isinstance(funds, list):
        return errors + ["funds must be a list"]
    tickers = [row.get("ticker") for row in funds if isinstance(row, dict)]
    if set(tickers) != ACTIVE or len(tickers) != len(ACTIVE):
        errors.append("observations must contain exactly NVII, TSII, and WMTI")
    for fund in funds:
        if not isinstance(fund, dict):
            errors.append("each fund must be an object")
            continue
        ticker = fund.get("ticker", "<unknown>")
        source = str(fund.get("source_url", ""))
        if urlparse(source).hostname not in {"www.rexshares.com", "rexshares.com"}:
            errors.append(f"{ticker}: source_url must use the official REX domain")
        observation = fund.get("observation")
        distribution = fund.get("distribution")
        holdings = fund.get("holdings")
        if not isinstance(observation, dict) or not isinstance(distribution, dict) or not isinstance(holdings, dict):
            errors.append(f"{ticker}: observation, distribution, and holdings are required")
            continue
        for key in ("as_of", "nav_usd", "market_price_usd", "premium_discount_percent", "fund_assets_usd", "shares_outstanding", "number_of_holdings", "total_expense_ratio_percent"):
            if key not in observation or observation[key] is None:
                errors.append(f"{ticker}: missing observation.{key}")
        for key in ("as_of", "distribution_rate_percent", "frequency", "sec_yield_percent", "sec_yield_as_of", "estimated_roc_percent", "gross_expense_ratio_percent"):
            if key not in distribution or distribution[key] is None:
                errors.append(f"{ticker}: missing distribution.{key}")
        if distribution.get("distribution_rate_is_total_return") is not False:
            errors.append(f"{ticker}: Distribution Rate must not be represented as total return")
        rows = holdings.get("rows")
        if not isinstance(rows, list):
            errors.append(f"{ticker}: holdings.rows must be a list")
            continue
        expected_count = observation.get("number_of_holdings")
        if holdings.get("published_holding_count") != expected_count or len(rows) != expected_count:
            errors.append(f"{ticker}: holding rows must match the official Number of Holdings")
        if holdings.get("as_of") != observation.get("as_of"):
            errors.append(f"{ticker}: holdings and fund observation must use the same as-of date")
        if any(not row.get("symbol") or not isinstance(row.get("weight_percent"), (int, float)) for row in rows if isinstance(row, dict)):
            errors.append(f"{ticker}: every holding must have symbol and numeric weight_percent")
        calculated = round(sum(float(row["weight_percent"]) for row in rows), 2)
        if calculated != holdings.get("net_weight_sum_percent"):
            errors.append(f"{ticker}: net_weight_sum_percent does not match holding rows")
        if not 99.0 <= calculated <= 101.0:
            errors.append(f"{ticker}: published holdings do not reconcile to approximately 100% net weight")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/rex-growth-income-active-observations.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    errors = validate(payload)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS: validated current official observations for NVII, TSII, and WMTI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
