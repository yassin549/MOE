# Label Evaluation Protocol (Formal)

## Scope
This protocol governs setup labeling for all assets/setups in the MoE training dataset. It is the source-of-truth contract for Phase 1 signal recovery, so experiments should fail fast when the protocol is violated.

## Temporal Definitions (per label)
For each `{asset}_{setup}` label row:
- **Trigger timestamp**: the candle close timestamp at index `t` (`{asset}_{setup}_trigger_timestamp`).
- **Earliest tradable timestamp**: strictly after trigger, defined as next bar timestamp `t+1` (`{asset}_{setup}_earliest_tradable_timestamp`).
- **Fixed outcome horizon**: constant number of bars evaluated forward from `t`, stored in `{asset}_{setup}_outcome_horizon_bars` (currently `max_holding_bars`).
- **Cost model**: the versioned execution-cost specification encoded in `{asset}_{setup}_cost_model` and applied to net returns. Cost-adjusted returns are the canonical target for tradable labels and manager supervision.

## Allowed Information at Trigger Time
Setup-presence (`*_present`) may only use features observable at or before trigger close `t`.
- A setup can only be marked present when `*_setup_inputs_available == 1`.
- `*_setup_inputs_available` requires all setup-input features to be non-null at `t`.
- Setup presence and tradable-setup status are separate concepts. `*_present` records heuristic presence; `*_tradable` records positive post-cost edge after outcome simulation.

## Prohibited Construction Patterns
- No post-trigger bars (`t+1...`) may be used when computing setup presence, validity masks, direction, or trigger metadata.
- Post-trigger bars are allowed **only** for outcome simulation/evaluation (`target`, `return_r`, `net_return_r`, `resolution_bars`, MAE).

## Automated Leakage Checks
Dataset build MUST run leakage checks and fail on violations. Checks include:
1. Presence rows with missing trigger-time inputs.
2. Earliest tradable timestamp not strictly later than trigger.
3. Non-positive outcome horizon.
4. Outcome resolution exceeding declared horizon.

Build logs must emit a summary line and per-violation lines under `[dataset-build] leakage ...`.
