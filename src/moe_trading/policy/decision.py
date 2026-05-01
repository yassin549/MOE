"""Shared policy-layer input types and account-state encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from moe_trading.account.rules import PropRuleEngine
from moe_trading.account.state import AccountState
from moe_trading.config import BacktestConfig


@dataclass(frozen=True, slots=True)
class PolicyContext:
    account_state: AccountState
    moe_outputs: dict[str, Any] = field(default_factory=dict)
    calendar_context: dict[str, Any] = field(default_factory=dict)
    session_context: dict[str, Any] = field(default_factory=dict)
    volatility_context: dict[str, Any] = field(default_factory=dict)
    correlation_context: dict[str, Any] = field(default_factory=dict)
    existing_position_state: dict[str, Any] = field(default_factory=dict)


ACCOUNT_FEATURE_NAMES = (
    "phase_challenge",
    "phase_funded",
    "phase_sim",
    "remaining_profit_target_fraction",
    "remaining_daily_loss_fraction",
    "remaining_total_loss_fraction",
    "open_risk_fraction",
    "profitable_day_progress",
    "days_elapsed_fraction",
    "challenge_passed",
    "breached",
)


def encode_account_state(state: AccountState, rules_engine: PropRuleEngine) -> np.ndarray:
    rules = rules_engine.current_phase_rules(state)
    profitable_day_denom = max(rules.minimum_profitable_days, 1)
    features = np.array(
        [
            1.0 if state.phase.value == "challenge" else 0.0,
            1.0 if state.phase.value == "funded" else 0.0,
            1.0 if state.phase.value == "sim" else 0.0,
            rules_engine.remaining_profit_target(state) / state.starting_balance,
            rules_engine.remaining_daily_loss_buffer(state) / state.starting_balance,
            rules_engine.remaining_total_loss_buffer(state) / state.starting_balance,
            state.current_open_risk,
            min(state.profitable_day_count / profitable_day_denom, 1.0),
            min(state.days_elapsed / 30.0, 1.0),
            float(state.challenge_passed),
            float(state.breached),
        ],
        dtype=np.float32,
    )
    return features


def expert_trade_threshold(config: BacktestConfig, expert_name: str) -> float:
    """Return the probability threshold for a routed expert candidate."""
    return float(config.expert_min_trade_probability.get(expert_name, config.min_trade_probability))


def routed_expert_scores(
    probabilities: np.ndarray,
    gate_weights: np.ndarray,
    expected_returns: np.ndarray,
    confidence: np.ndarray | None,
    config: BacktestConfig,
) -> np.ndarray:
    scores = probabilities * gate_weights
    if config.use_confidence_in_routing and confidence is not None:
        scores = scores * confidence
    if config.min_expected_return_r > 0.0:
        scores = np.where(expected_returns >= config.min_expected_return_r, scores, 0.0)
    else:
        scores = scores * (1.0 / (1.0 + np.exp(-np.clip(expected_returns, -50.0, 50.0))))
    return scores
