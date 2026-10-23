#!/usr/bin/env python3
"""Validate the canonical REX comparison inputs as one deterministic pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PAIR_SCHEMA_VERSION = "rex-compare-pair.v1"
EXPECTED_TICKERS = frozenset({"NVII", "TSII", "WMTI"})


class PairValidationError(ValueError):
    """Raised when two individually valid snapshots cannot be safely joined."""


def _index_funds(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    funds = payload.get("funds")
    if not isinstance(funds, list):
        raise PairValidationError(f"{label}: funds must be a list")

    indexed: dict[str, dict[str, Any]] = {}
    for fund in funds:
        ticker = fund.get("ticker") if isinstance(fund, dict) else None
        if not isinstance(ticker, str) or not ticker:
            raise PairValidationError(f"{label}: every fund requires a ticker")
        if ticker in indexed:
            raise PairValidationError(f"{label}: duplicate ticker {ticker}")
        indexed[ticker] = fund
    return indexed


def validate_pair(
    observations: dict[str, Any], distributions: dict[str, Any]
) -> dict[str, Any]:
    """Validate cross-file identity and temporal compatibility and return pair metadata."""
    observation_funds = _index_funds(observations, "active observations")
    distribution_funds = _index_funds(distributions, "distribution history")

    observation_tickers = set(observation_funds)
    distribution_tickers = set(distribution_funds)
    if observation_tickers != EXPECTED_TICKERS:
        raise PairValidationError(
            "active observations ticker set mismatch: "
            f"expected {sorted(EXPECTED_TICKERS)}, got {sorted(observation_tickers)}"
        )
    if distribution_tickers != EXPECTED_TICKERS:
        raise PairValidationError(
            "distribution history ticker set mismatch: "
            f"expected {sorted(EXPECTED_TICKERS)}, got {sorted(distribution_tickers)}"
        )

    for ticker in sorted(EXPECTED_TICKERS):
        active = observation_funds[ticker]
        history = distribution_funds[ticker]
        active_source = active.get("source_url")
        history_source = history.get("source_url")
        if active_source != history_source or not isinstance(active_source, str):
            raise PairValidationError(
                f"{ticker}: source identity mismatch: {active_source!r} != {history_source!r}"
            )

        active_distribution = active.get("distribution")
        if not isinstance(active_distribution, dict):
            raise PairValidationError(f"{ticker}: active distribution is missing")
        active_as_of = active_distribution.get("as_of")

        rows = history.get("distributions")
        if not isinstance(rows, list) or not rows:
            raise PairValidationError(f"{ticker}: distribution history is empty")
        declaration_dates = [
            row.get("declaration_date") for row in rows if isinstance(row, dict)
        ]
        if len(declaration_dates) != len(rows) or not all(
            isinstance(value, str) and value for value in declaration_dates
        ):
            raise PairValidationError(f"{ticker}: invalid declaration_date in history")
        newest_realized = max(declaration_dates)
        if active_as_of != newest_realized:
            raise PairValidationError(
                f"{ticker}: active distribution.as_of {active_as_of!r} does not match "
                f"newest realized declaration_date {newest_realized!r}"
            )

    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "tickers": sorted(EXPECTED_TICKERS),
        "active_observations": {
            "schema_version": observations.get("schema_version"),
            "observed_at": observations.get("observed_at"),
        },
        "distribution_history": {
            "schema_version": distributions.get("schema_version"),
            "observed_at": distributions.get("observed_at"),
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PairValidationError(f"{path}: top-level JSON value must be an object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("data/rex-growth-income-active-observations.json"),
    )
    parser.add_argument(
        "--distributions",
        type=Path,
        default=Path("data/rex-growth-income-distributions.json"),
    )
    args = parser.parse_args()

    metadata = validate_pair(_load_json(args.observations), _load_json(args.distributions))
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
