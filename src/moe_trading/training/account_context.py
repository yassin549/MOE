"""Deterministic account-context replay used to reduce train/live distribution shift."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from moe_trading.account.rules import PropRuleEngine, TradeIntent
from moe_trading.account.state import AccountState
from moe_trading.config import AppConfig
from moe_trading.policy.decision import ACCOUNT_FEATURE_NAMES, encode_account_state


def build_account_context_array(frame: pd.DataFrame, config: AppConfig) -> np.ndarray:
    """Build per-row account-state features using a deterministic label-driven replay."""
    engine = PropRuleEngine(config.prop)
    state = engine.evaluate_state(AccountState(starting_balance=config.prop.starting_balance))
    setup_names = list(config.model.setup_names)
    asset_names = ("US100", "US500")
    risk_fraction = float(config.backtest.challenge_risk_fraction)
    contexts = np.zeros((len(frame), len(ACCOUNT_FEATURE_NAMES)), dtype=np.float32)
    pending_closures: list[tuple[int, str, float]] = []
    position_counter = 0

    for idx, row in frame.reset_index(drop=True).iterrows():
        timestamp = str(row["timestamp"])
        state = engine.roll_day(state, timestamp)

        matured = [item for item in pending_closures if item[0] <= idx]
        pending_closures = [item for item in pending_closures if item[0] > idx]
        for _, position_id, realized_pnl in matured:
            if any(position.position_id == position_id for position in state.open_positions):
                state = engine.close_position(state, position_id, realized_pnl)

        state = engine.mark_to_market(state, 0.0)
        contexts[idx] = encode_account_state(state, engine)

        for asset in asset_names:
            prefix = asset.lower()
            if float(row.get(f"{prefix}_manager_trade_target", 0.0)) <= 0:
                continue
            best_expert_idx = int(row.get(f"{prefix}_manager_best_expert", -1))
            if best_expert_idx < 0 or best_expert_idx >= len(setup_names):
                continue
            setup_name = setup_names[best_expert_idx]
            entry_price = float(row[f"{prefix}_close"])
            stop_distance = max(float(row[f"{prefix}_atr_15"]) * config.labels.stop_atr_multiple, 1e-6)
            direction = 1 if float(row[f"{prefix}_{setup_name}_direction"]) >= 0 else -1
            stop_price = entry_price - (direction * stop_distance)
            position_id = f"{asset}-{position_counter}"
            position_counter += 1
            trade = TradeIntent(
                asset=asset,
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                risk_fraction=risk_fraction,
                opened_at=timestamp,
            )
            evaluation = engine.validate_trade(state, trade)
            if not evaluation.allowed or evaluation.capped_risk_fraction <= 0.0:
                continue
            if evaluation.capped_risk_fraction != trade.risk_fraction:
                trade = replace(trade, risk_fraction=evaluation.capped_risk_fraction)
            state = engine.open_position(state, trade, position_id)
            realized_r = float(row.get(f"{prefix}_{setup_name}_net_return_r", 0.0))
            resolution_bars = max(1, int(row.get(f"{prefix}_{setup_name}_resolution_bars", 1)))
            realized_pnl = realized_r * trade.risk_fraction * state.starting_balance
            pending_closures.append((idx + resolution_bars, position_id, realized_pnl))

    return contexts
