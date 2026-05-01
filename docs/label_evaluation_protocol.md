# Formal Label-Evaluation Protocol

## Purpose

This protocol defines strict, testable anti-leakage rules for setup labels used by the research and training pipeline.

## Time Semantics (Per Label)

For each `asset x setup` label instance at bar index `t`:

- **Trigger timestamp**: bar `t` (the bar where setup presence is evaluated).
- **Earliest tradable timestamp**: bar `t + 1` (strictly after trigger).
- **Fixed outcome horizon**: `max_holding_bars` bars from trigger.
- **Cost model applied**: `round_trip_bps(spread+slippage+commission)`.

The implementation stores explicit metadata columns:

- `..._trigger_bar_index`
- `..._earliest_tradable_bar_index`
- `..._outcome_horizon_bars`
- `..._cost_model`

## Setup-Presence Feature Rules

Setup presence columns (`..._present`) must be computed only from information available at trigger bar `t` or earlier. No feature used for setup presence may depend on bars after `t`.

## Label-Construction Rules

- Post-trigger bars (`t+1` onward) are **prohibited** in setup-presence construction.
- Post-trigger bars are allowed **only** in outcome evaluation (`target/stop/return/MAE/resolution`) over the fixed horizon.

## Enforcement in Code

1. Label generation emits trigger/tradable/horizon/cost metadata per setup.
2. Dataset build runs an automated leakage check over all setups and assets.
3. Any timestamp-rule violation is logged and causes dataset build failure.

## Automated Leakage Checks

The leakage check validates:

- `earliest_tradable_bar_index > trigger_bar_index` for every present setup row.
- `outcome_horizon_bars > 0` for every present setup row.
- required metadata columns exist for each setup.

If violations are found, build aborts with explicit diagnostics.
