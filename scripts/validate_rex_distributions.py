#!/usr/bin/env python3
"""Validate realized weekly distribution history for active REX Growth & Income funds."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

ACTIVE = {"NVII", "TSII", "WMTI"}
REQUIRED_FIELDS = ("declaration_date", "ex_date", "record_date", "payable_date", "distribution_per_share_usd")


def parse_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def validate(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "rex-growth-income-distributions.v1":
        errors.append("unsupported schema_version")
    observed_at = payload.get("observed_at")
    try:
        observed_date = datetime.fromisoformat(str(observed_at)).date()
    except ValueError:
        errors.append("observed_at must be an ISO-8601 timestamp")
        observed_date = date.min
    funds = payload.get("funds")
    if not isinstance(funds, list):
        return errors + ["funds must be a list"]
    tickers = [row.get("ticker") for row in funds if isinstance(row, dict)]
    if set(tickers) != ACTIVE or len(tickers) != len(ACTIVE):
        errors.append("distribution history must contain exactly NVII, TSII, and WMTI")
    for fund in funds:
        if not isinstance(fund, dict):
            errors.append("each fund must be an object")
            continue
        ticker = str(fund.get("ticker", "<unknown>"))
        source = str(fund.get("source_url", ""))
        if urlparse(source).hostname not in {"www.rexshares.com", "rexshares.com"}:
            errors.append(f"{ticker}: source_url must use the official REX domain")
        rows = fund.get("distributions")
        if not isinstance(rows, list):
            errors.append(f"{ticker}: distributions must be a list")
            continue
        if len(rows) < 12:
            errors.append(f"{ticker}: at least 12 realized weekly distributions are required")
        declarations: list[date] = []
        identities: set[tuple[object, ...]] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"{ticker}[{index}]: distribution must be an object")
                continue
            missing = [field for field in REQUIRED_FIELDS if row.get(field) is None]
            if missing:
                errors.append(f"{ticker}[{index}]: missing {', '.join(missing)}")
                continue
            dates = {field: parse_date(row[field]) for field in REQUIRED_FIELDS[:4]}
            if any(value is None for value in dates.values()):
                errors.append(f"{ticker}[{index}]: invalid distribution date")
                continue
            declaration = dates["declaration_date"]
            assert declaration is not None
            ex_date = dates["ex_date"]
            record_date = dates["record_date"]
            payable_date = dates["payable_date"]
            assert ex_date is not None and record_date is not None and payable_date is not None
            if not (declaration <= ex_date <= payable_date and declaration <= record_date <= payable_date):
                errors.append(f"{ticker}[{index}]: distribution dates are not chronologically consistent")
            if payable_date > observed_date:
                errors.append(f"{ticker}[{index}]: future/unrealized distribution is not allowed")
            amount = row.get("distribution_per_share_usd")
            if not isinstance(amount, (int, float)) or isinstance(amount, bool) or amount <= 0:
                errors.append(f"{ticker}[{index}]: distribution_per_share_usd must be positive numeric")
            declaration_identity = (row["declaration_date"], row["ex_date"], row["payable_date"])
            if declaration_identity in identities:
                errors.append(f"{ticker}[{index}]: duplicate distribution identity")
            identities.add(declaration_identity)
            declarations.append(declaration)
        if declarations != sorted(declarations, reverse=True):
            errors.append(f"{ticker}: distributions must be newest first")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/rex-growth-income-distributions.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    errors = validate(payload)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS: validated realized weekly distribution history for NVII, TSII, and WMTI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
