# System Overview

## Goal

The system is designed as a setup-detection and trade-selection engine for US100 and US500 under prop-style trading conditions.

It does not try to predict generic next-bar direction. Instead it does the following on every closed candle:

1. Build synchronized multi-asset state from US100 and US500.
2. Evaluate multiple setup families independently with specialized experts.
3. Estimate whether each setup is both present and likely to produce an acceptable trade outcome.
4. Use a manager network to decide whether to trade US100, US500, both, or neither.

## Current Code Entry Points

- Training: `scripts/train.py`
- Backtesting: `scripts/backtest.py`
- Live inference: `scripts/live_infer.py`
- Top-level research-frame builder: `src/moe_trading/pipeline.py`

## Package Map

### Configuration

- `src/moe_trading/config.py`

Contains typed dataclass configs for:

- data loading
- feature engineering
- labeling
- model structure
- training
- backtesting
- experiment output

### Data Layer

- `src/moe_trading/data/loading.py`
- `src/moe_trading/data/splitting.py`
- `src/moe_trading/data/scaling.py`
- `src/moe_trading/data/dataset.py`
- `src/moe_trading/data/schemas.py`

Responsibilities:

- load cleaned US100 and US500 candles
- align them on common timestamps
- build strict time-based train/validation/test partitions
- apply train-only feature scaling
- create rolling sequence tensors for the neural model

### Feature Layer

- `src/moe_trading/features/engineering.py`

Responsibilities:

- compute per-asset features
- compute cross-asset interaction features
- compute regime features
- group numeric feature columns by consumer

### Label Layer

- `src/moe_trading/labels/generation.py`

Responsibilities:

- define setup conditions
- assign setup directions
- simulate ATR-based trade outcomes
- create per-asset expert targets
- create manager trade targets

### Model Layer

- `src/moe_trading/models/tcn.py`
- `src/moe_trading/models/moe.py`

Responsibilities:

- causal TCN blocks
- shared dual-asset context encoder
- one TCN expert per setup family
- cross-asset manager/gating network
- calibration head
- model save/load

### Training Layer

- `src/moe_trading/training/losses.py`
- `src/moe_trading/training/pipeline.py`

Responsibilities:

- multi-task objective
- batching and device transfer
- walk-forward or static split training
- checkpoint persistence
- scaler persistence
- experiment summaries

### Evaluation Layer

- `src/moe_trading/evaluation/metrics.py`
- `src/moe_trading/evaluation/reports.py`

Responsibilities:

- binary metrics for manager decisions
- trade metrics for realized backtests
- markdown report writing

### Execution Layer

- `src/moe_trading/backtesting/engine.py`
- `src/moe_trading/live/pipeline.py`

Responsibilities:

- candle-by-candle backtest decisions
- blocked-trade audit logs
- latest closed-candle live decision object

### Utilities

- `src/moe_trading/utils/reproducibility.py`
- `src/moe_trading/utils/io.py`
- `src/moe_trading/utils/checkpoints.py`
- `src/moe_trading/experiments/tracker.py`

Responsibilities:

- seeding
- artifact directory creation
- JSON saving
- checkpoint resolution
- experiment config/metric logging

## Control Flow

High-level training flow:

1. Load config.
2. Build aligned dual-asset research frame.
3. Engineer features.
4. Generate setup and manager labels.
5. Split chronologically.
6. Fit feature scaler on train split only.
7. Build rolling sequence datasets.
8. Train MoE on walk-forward splits.
9. Save split checkpoints and scaler metadata.

High-level inference flow:

1. Load config.
2. Rebuild current research frame from closed candles only.
3. Load the latest checkpoint and scaler.
4. Scale features with persisted train statistics.
5. Slice the latest sequence window.
6. Run shared encoder, experts, manager, calibrator.
7. Emit structured trade decision.
