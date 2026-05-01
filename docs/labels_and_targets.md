# Labels And Targets

## Objective

The labeling layer does not assign a generic up/down label.

It tries to answer a stricter question:

"Did a specific setup exist here, with a specific trade direction and execution rule set, and did it reach acceptable reward before disqualifying stop or risk conditions?"

## Code

- `src/moe_trading/labels/generation.py`

## Current Setup Families

The configured experts are:

- trend continuation
- pullback continuation
- breakout expansion
- mean reversion
- liquidity sweep reversal
- volatility compression expansion
- session-open momentum
- exhaustion failure

## Labeling Flow

For each asset separately:

1. Compute a boolean setup condition.
2. Compute the trade direction implied by that setup.
3. Use current close as the hypothetical entry.
4. Use ATR-derived stop distance.
5. Use ATR-multiple target distance.
6. Simulate future bars over `max_holding_bars`.
7. Record:
   - target hit or not
   - stop hit or not
   - realized return in R
   - max adverse excursion in R
   - time to resolution
8. Mark the example valid only if MAE remains within configured tolerance.

## Per-Setup Columns Produced

For each `asset x setup`, the pipeline writes:

- `..._valid`
- `..._target`
- `..._return_r`
- `..._mae_r`
- `..._resolution_bars`
- `..._direction`

Interpretation:

- `valid` means the setup pattern existed and passed MAE-based trade validity rules
- `target` means the valid setup also reached its objective under the simulation rules
- `return_r` is the realized return normalized by stop distance
- `direction` is the directional trade idea attached to the setup

## Manager Targets

The manager receives two current targets:

- `manager_trade_target`
- `manager_dual_trade_target`

These are derived from whether any expert on either asset produced a valid and winning candidate, and whether both assets did so simultaneously.

## Leakage Controls

The labeler uses future bars only to assign labels, not to generate features.

Important protection mechanisms already present:

- chronological split
- embargo between splits
- train-only scaling
- live inference reads only the latest closed-candle sequence

## Current Weaknesses

The framework is structurally correct, but it is still a first research version.

Known limitations:

- setup definitions are rule-based heuristics, not yet iterated against actual edge distributions
- execution assumptions are still simplified:
  - entry is current close
  - intrabar tie when target and stop are both touched defaults to stop
- manager labels are binary and coarse
- label generation is computationally expensive because it loops over candidate bars

## What Should Be Improved Before Serious Research

1. Cache labeled datasets to disk per config hash.
2. Refine setup conditions using actual market structure diagnostics.
3. Add richer manager supervision:
   - per-asset select/block targets
   - expert ranking targets
   - rejection-quality targets
4. Add explicit MAE/MFE auxiliary targets if you want tighter risk-aware gating.
