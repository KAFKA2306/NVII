#!/usr/bin/env python3
"""Fail-closed validation for the REX Growth & Income fund master."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

EXPECTED = {"NVII", "TSII", "MSII", "COII", "HOII", "PLTI", "CWII", "LLII", "WMTI"}
ACTIVE = {"NVII", "TSII", "WMTI"}
LIQUIDATED = EXPECTED - ACTIVE
LIQUIDATION_DATE = "2026-06-16"
ALLOWED_STATUS = {"ACTIVE", "LIQUIDATED"}


def validate(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "rex-growth-income-fund-master.v1":
        errors.append("unsupported schema_version")
    funds = payload.get("funds")
    if not isinstance(funds, list):
        return errors + ["funds must be a list"]
    tickers = [f.get("ticker") for f in funds if isinstance(f, dict)]
    if set(tickers) != EXPECTED or len(tickers) != len(EXPECTED):
        errors.append("fund master must contain exactly the nine funds in the 2026-02-26 REX universe announcement")
    for fund in funds:
        if not isinstance(fund, dict):
            errors.append("each fund must be an object")
            continue
        ticker = fund.get("ticker")
        for key in ("legal_name", "underlying", "lifecycle_status", "product_url", "lifecycle_as_of"):
            if not fund.get(key):
                errors.append(f"{ticker or '<unknown>'}: missing {key}")
        status = fund.get("lifecycle_status")
        if status not in ALLOWED_STATUS:
            errors.append(f"{ticker}: invalid lifecycle_status")
        host = urlparse(str(fund.get("product_url", ""))).hostname
        if host not in {"www.rexshares.com", "rexshares.com"}:
            errors.append(f"{ticker}: product_url must use the official REX domain")
        if ticker in ACTIVE and status != "ACTIVE":
            errors.append(f"{ticker}: expected ACTIVE as of 2026-08-11")
        if ticker in LIQUIDATED:
            if status != "LIQUIDATED":
                errors.append(f"{ticker}: expected LIQUIDATED")
            if fund.get("liquidation_date") != LIQUIDATION_DATE:
                errors.append(f"{ticker}: liquidation_date must be {LIQUIDATION_DATE}")
        if ticker in ACTIVE and "liquidation_date" in fund:
            errors.append(f"{ticker}: active fund must not carry a liquidation_date")
    for key in ("universe_source", "lifecycle_source"):
        source = payload.get(key)
        if not source or urlparse(str(source)).hostname not in {"www.rexshares.com", "rexshares.com"}:
            errors.append(f"{key} must use the official REX domain")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/rex-growth-income-funds.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    errors = validate(payload)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"PASS: validated {len(payload['funds'])} funds; active={len(ACTIVE)} liquidated={len(LIQUIDATED)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
