"""Experiment scheduling and status tracking for execution-mechanics tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ExpertCompletionCriteria:
    minimum_trade_count: int = 30
    minimum_post_cost_expectancy_r: float = 0.0
    max_drawdown_floor_r: float = -5.0


@dataclass(slots=True)
class ExpertSchedulerConfig:
    expert_priority: list[str] = field(
        default_factory=lambda: [
            "liquidity_sweep_reversal",
            "trend_continuation",
            "pullback_continuation",
            "breakout_expansion",
            "mean_reversion",
            "volatility_compression_expansion",
            "session_open_momentum",
            "exhaustion_failure",
        ]
    )
    completion: ExpertCompletionCriteria = field(default_factory=ExpertCompletionCriteria)


def expert_status_report(trades: pd.DataFrame, scheduler: ExpertSchedulerConfig) -> list[dict[str, Any]]:
    """Return per-expert status ordered by configured execution priority."""
    report: list[dict[str, Any]] = []
    prior_failed = False
    prior_in_progress = False

    for expert in scheduler.expert_priority:
        expert_trades = trades[trades["expert"] == expert] if not trades.empty and "expert" in trades.columns else pd.DataFrame()
        num_trades = int(len(expert_trades))

        if expert_trades.empty:
            expectancy_r = 0.0
            max_drawdown_r = 0.0
        else:
            returns = expert_trades["net_return_r"].to_numpy()
            expectancy_r = float(returns.mean())
            equity = returns.cumsum()
            running_peak = np.maximum.accumulate(equity)
            max_drawdown_r = float((equity - running_peak).min())

        meets_trade_count = num_trades >= scheduler.completion.minimum_trade_count
        meets_expectancy = expectancy_r >= scheduler.completion.minimum_post_cost_expectancy_r
        meets_drawdown = max_drawdown_r >= scheduler.completion.max_drawdown_floor_r
        passed = meets_trade_count and meets_expectancy and meets_drawdown

        if prior_failed:
            status = "pending"
        elif prior_in_progress:
            status = "pending"
        elif passed:
            status = "passed"
        elif num_trades == 0:
            status = "pending"
            prior_in_progress = True
        elif not meets_trade_count:
            status = "in-progress"
            prior_in_progress = True
        else:
            status = "failed"
            prior_failed = True

        report.append(
            {
                "expert": expert,
                "status": status,
                "num_trades": num_trades,
                "expectancy_r": expectancy_r,
                "max_drawdown_r": max_drawdown_r,
                "meets_minimum_trade_count": meets_trade_count,
                "meets_non_negative_post_cost_expectancy": meets_expectancy,
                "meets_stable_drawdown_bounds": meets_drawdown,
            }
        )

    return report
