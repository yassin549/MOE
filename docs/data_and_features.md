# Data And Features

## Raw Inputs

The current training source files are:

- `data/processed/US100_m1_clean.csv`
- `data/processed/US500_m1_clean.csv`

These are cleaned minute-bar files produced by the existing Dukascopy pipeline.

## Loading And Alignment

Code:

- `src/moe_trading/data/loading.py`

Current behavior:

1. Read each asset CSV with pandas.
2. Validate presence of required candle and calendar columns.
3. Parse `timestamp_utc` as UTC.
4. Sort by time and drop duplicate timestamps within each asset.
5. Inner-join US100 and US500 on common closed-candle timestamps.
6. Add a sequential `bar_index`.
7. Optionally keep only the most recent `data.max_rows` rows.

Why the inner join matters:

- the model always sees synchronized asset state
- no synthetic forward-filling is used across instruments
- one asset cannot contribute future information by having a bar when the other does not

## Time Splits

Code:

- `src/moe_trading/data/splitting.py`

Supported split modes:

- simple chronological split with embargo
- rolling walk-forward splits with embargo

Embargo is used to reduce contamination around split boundaries when labels look forward over a holding horizon.

## Feature Groups

Code:

- `src/moe_trading/features/engineering.py`

The feature pipeline builds three groups:

1. Asset-specific features for US100.
2. Asset-specific features for US500.
3. Cross-asset and regime features shared by experts and manager.

The builder still computes a wider research frame for labeling and diagnostics, but the model now consumes a fixed curated subset of 40 features:

- 14 asset features for US100
- 14 asset features for US500
- 8 cross/session features
- 4 regime features

This avoids training on every numeric column in the frame and keeps the live/training schema compact and stable.

## Asset Features

For each asset prefix `us100_` and `us500_`, the model consumes these 14 curated features:

- `return_1`
- `range`
- `body`
- `upper_wick`
- `lower_wick`
- `volatility_15`
- `atr_15`
- `momentum_5`
- `momentum_15`
- `distance_high_20`
- `distance_low_20`
- `slope_15`
- `swing_position_20`
- `compression_20`

## Session Features

The model consumes these 8 shared cross/session features:

- `minute_sin`
- `minute_cos`
- `is_session_open_window`
- `spread_return_diff`
- `relative_strength_15`
- `corr_30`
- `spread_z_20`
- `divergence_flag`

## Regime Features

The model consumes these 4 regime features:

- `joint_volatility`
- `volatility_ratio`
- `trend_agreement`
- `divergent_regime`

These are simple but structurally correct. They give the manager and experts explicit context about confirmation, divergence, and volatility state across the two indices.

## Scaling

Code:

- `src/moe_trading/data/scaling.py`

Scaling is fit only on the training split and then applied to validation, test, backtest, and live inference.

This is required because the model currently sees price-level and spread-level features that would otherwise be numerically unstable.

## Current Operational Constraint

The feature builder is functionally correct but still expensive:

- it inserts many columns one by one into a pandas frame
- pandas reports fragmentation warnings
- this slows down full-history preprocessing materially

This is a performance issue, not a conceptual issue, and should be treated as the next optimization target before large-scale training.
