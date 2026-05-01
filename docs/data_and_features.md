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

Only numeric columns are passed into the model. Raw string fields such as month names or session labels are excluded.

## Asset Features

For each asset prefix `us100_` and `us500_`, the pipeline creates:

- simple return
- log return
- normalized candle range
- normalized candle body
- upper wick ratio
- lower wick ratio
- true range
- directional sign
- rolling volatility
- ATR-style rolling true range
- rolling mean range
- volume z-scores when volume is enabled
- momentum over multiple windows
- distance from rolling highs and lows
- rolling slope
- swing position inside rolling high/low envelopes
- compression ratio
- simple candle pattern markers:
  - three-bar reversal
  - inside bar
  - outside bar

## Session Features

Built once from the synchronized timestamp context:

- minute-of-day sine
- minute-of-day cosine
- day-of-week sine
- day-of-week cosine
- session-open-window flag

## Cross-Asset Features

Current cross-asset features include:

- close spread ratio
- return spread difference
- body divergence
- range divergence
- relative strength 15
- relative strength 30
- rolling correlation windows
- spread z-scores
- return-difference z-scores
- co-momentum flag
- divergence flag

## Regime Features

Current regime features include:

- joint volatility
- volatility ratio
- trend agreement
- risk-on regime flag
- divergent regime flag
- high-volatility regime flag

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
