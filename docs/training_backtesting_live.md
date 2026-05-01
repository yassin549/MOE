# Training, Backtesting, And Live Inference

## Training

Code:

- `src/moe_trading/training/losses.py`
- `src/moe_trading/training/pipeline.py`

### Current Training Flow

1. Set global seed.
2. Build research frame.
3. Choose walk-forward or static time split.
4. Fit scaler on train split only.
5. Build rolling sequence datasets.
6. Train model on each split.
7. Track validation loss for early stopping.
8. Save split checkpoint and scaler metadata.
9. Save summary JSON.

### Loss Components

The current multi-task loss includes:

- expert classification loss
- manager trade loss
- manager dual-trade loss
- calibration MSE-style loss
- return regression loss
- diversity penalty
- gate entropy regularization

### Current Artifact Layout

Inside `experiment.output_dir`:

- `config.json`
- `training_summary.json`
- `split_XX/model.pt`
- `split_XX/scaler.json`

## Backtesting

Code:

- `src/moe_trading/backtesting/engine.py`

### Behavior

The backtester:

1. Loads the latest checkpoint.
2. Loads the associated scaler.
3. Rebuilds the test research frame.
4. Applies the persisted scaler.
5. Walks candle by candle through test sequences.
6. Evaluates manager probability and context score.
7. Chooses the highest-calibrated expert per asset.
8. Applies threshold logic.
9. Logs both executed trades and blocked decisions.

### Outputs

- `backtest_trades.csv`
- `backtest_decision_log.csv`
- `backtest_summary.json`

## Live Inference

Code:

- `src/moe_trading/live/pipeline.py`

### Behavior

The live pipeline currently:

1. Rebuilds the current research frame from the latest closed candles.
2. Loads the latest saved checkpoint and scaler.
3. Scales features using persisted train statistics.
4. Slices the last `sequence_length` bars.
5. Produces a structured decision object.

### Output Object

`LiveTradeDecision` contains:

- timestamp
- trade flag
- dual-trade flag
- selected assets
- selected experts per asset
- directions per asset
- selected probabilities
- context score

## Scripts

### Train

`python scripts/train.py configs/base.yaml`

### Backtest

`python scripts/backtest.py configs/base.yaml`

### Live

`python scripts/live_infer.py configs/base.yaml`

## Current Operational Reality

The orchestration layer works, but full training is not yet operationally efficient on the current machine.

The dominant cost today is preprocessing:

- alignment is manageable
- feature engineering is slower than it should be
- labeling is the heaviest component

The model-training loop itself is not the main bottleneck yet.
