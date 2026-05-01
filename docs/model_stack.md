# Model Stack

## Code

- `src/moe_trading/models/tcn.py`
- `src/moe_trading/models/moe.py`

## High-Level Architecture

The model is a hybrid multi-input Mixture-of-Experts:

1. Shared multi-asset encoder.
2. One TCN expert per setup type.
3. One cross-asset manager network.
4. One calibration head over expert probabilities.

## TCN Building Block

`src/moe_trading/models/tcn.py` implements:

- `Chomp1d`
- `TemporalBlock`
- `TCNEncoder`

Properties:

- causal convolutions
- exponentially increasing dilation by layer
- residual connection per temporal block
- final representation taken from the last time step

This is appropriate for closed-candle sequence modeling because it preserves temporal ordering and avoids recurrence complexity.

## Shared Multi-Asset Encoder

Class:

- `SharedMultiAssetEncoder`

Inputs:

- one asset sequence
- cross-asset sequence
- regime sequence

Behavior:

- encodes asset-local sequence with a TCN
- encodes concatenated cross/regime sequence with another TCN
- fuses the two last-step states into a shared context vector

This context is intended to tell each expert what the other market is doing, not just what its own asset is doing.

## Setup Experts

Class:

- `SetupExpert`

Each expert takes:

- its asset sequence
- the synchronized cross-asset sequence
- the synchronized regime sequence
- the shared context vector repeated across time

Each expert outputs:

- setup logit
- confidence logit
- expected return head
- direction head

After activation, the main `MultiAssetMoE` object exposes:

- expert probabilities
- expert confidence
- expected returns
- directions

## Manager Network

Class:

- `ManagerNetwork`

Inputs:

- flattened expert probabilities
- flattened expert confidence
- flattened expected returns
- flattened directions
- current manager context vector

Outputs:

- trade probability
- dual-trade probability
- context compatibility score
- gate logits over experts per asset

Conceptually, the manager is the policy layer that decides whether strong expert scores are trustworthy in current cross-asset conditions.

## Calibration Head

Class:

- `ProbabilityCalibrationHead`

Purpose:

- map raw expert probabilities to a calibrated expert-probability surface used downstream by backtest and live decision logic

This is still a learned shallow calibrator, not a post-hoc isotonic or Platt calibration pass.

## Main Forward Output

Dataclass:

- `MoEOutput`

Contains:

- `expert_probabilities`
- `expert_confidence`
- `expected_returns`
- `directions`
- `manager_trade_probability`
- `manager_dual_probability`
- `manager_context_score`
- `manager_gate_weights`
- `calibrated_probabilities`

## Current Strengths

- clean separation between expert scoring and manager selection
- explicit cross-asset context path
- causal temporal blocks
- multi-output heads suitable for richer downstream decision logic

## Current Gaps

- experts are independent modules but there is no explicit expert-specialization supervision beyond label separation and diversity penalty
- manager currently learns coarse binary trade decisions, not full asset/expert action policies
- calibration is integrated but not yet validated against reliability curves
- no attention or explicit sequence cross-talk between US100 and US500 beyond engineered shared context
