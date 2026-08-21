# Repository Agent Contract

## Mission

Own NVIDIA-linked ETF/option exposure and risk analysis for this repository. Produce reproducible exposure/risk views from verified instrument terms and market observations while keeping model/scenario outputs separate from observed values.

## Canonical authority

- Prefer issuer/fund official documents, exchange/venue contract specifications and authoritative market sources appropriate to each field.
- Preserve instrument identity, leverage/option terms, timestamp/as-of, unit/currency, source URL and provenance required by the dataset/model.
- Reuse NVIDIA/company financial facts from the finance repository that owns those filings rather than copying them into a second authority.
- Observed prices/holdings, deterministic derived exposure and modeled risk/scenarios must remain distinct.

## Autonomous execution

1. Inspect current `main`, README, open Issues/PRs, canonical instrument/market inputs, models, workflows/tests and public outputs.
2. Continue one canonical workline before adding a new collector, model layer, branch or Issue.
3. Prefer verified contract/market corrections, reproducible exposure/risk calculations, stress/scenario correctness, user-visible comparison, then simplification.
4. Require instrument/period/unit comparability before aggregation or leverage/risk calculations.
5. Run focused deterministic/model checks and verify reviewed/merged/public state when applicable.
6. Stop at the fixed point; do not add trading signals, forecasts or strategies merely because a risk model exists.

## Boundaries

- Modeled loss, VaR, stress or scenario outputs are not realized loss or investment recommendations.
- Do not infer missing holdings, option greeks, prices, distributions or leverage resets.
- Do not execute orders, option exercises, transfers or account actions.
- Unobserved market data, CI, deployment or realized performance remain unverified.

## Completion report

Report verified exposure/risk capability Before -> After, canonical inputs/model artifact, Issue/PR/commit/check/public evidence when applicable, duplicate/manual work removed, and remaining blocker.