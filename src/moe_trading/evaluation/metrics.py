"""Metric computation for model outputs and backtest results."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from moe_trading.config import AppConfig


def binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    brier = float(np.mean((y_prob - y_true) ** 2))
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "brier": brier,
        "base_rate": float(y_true.mean()) if len(y_true) else 0.0,
    }


def _safe_float(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


def _streak_lengths(mask: np.ndarray) -> list[int]:
    streaks: list[int] = []
    current = 0
    for hit in mask.astype(bool):
        if hit:
            current += 1
        elif current:
            streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    return streaks


def _equity_metrics(returns: np.ndarray) -> dict[str, float]:
    equity_curve_r = np.cumsum(returns)
    if len(equity_curve_r) == 0:
        return {
            "ending_equity_r": 0.0,
            "peak_equity_r": 0.0,
            "recovery_factor": 0.0,
            "ulcer_index": 0.0,
        }

    running_peak_r = np.maximum.accumulate(np.concatenate(([0.0], equity_curve_r)))
    drawdown_r = np.concatenate(([0.0], equity_curve_r)) - running_peak_r
    max_drawdown_r = float(-drawdown_r.min())
    recovery_factor = 0.0 if max_drawdown_r == 0.0 else float(equity_curve_r[-1] / max_drawdown_r)
    ulcer_index = float(np.sqrt(np.mean(np.square(drawdown_r))))
    return {
        "ending_equity_r": float(equity_curve_r[-1]),
        "peak_equity_r": float(running_peak_r.max()),
        "recovery_factor": recovery_factor,
        "ulcer_index": ulcer_index,
    }


def _period_return_summary(trades: pd.DataFrame, freq: str) -> dict[str, Any]:
    if trades.empty:
        return {
            "periods": 0,
            "win_rate": 0.0,
            "average_return_r": 0.0,
            "average_win_return_r": 0.0,
            "average_loss_return_r": 0.0,
            "best_return_r": 0.0,
            "worst_return_r": 0.0,
        }

    timestamps = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    if freq == "D":
        keys = timestamps.dt.floor("D")
    elif freq == "W":
        keys = timestamps.dt.to_period("W-MON").dt.start_time
    elif freq == "M":
        keys = timestamps.dt.to_period("M").dt.start_time
    else:
        raise ValueError(f"Unsupported period frequency: {freq}")
    grouped = trades.assign(_period=keys).groupby("_period")["net_return_r"].sum()
    values = grouped.to_numpy(dtype=np.float64)
    wins = values[values > 0]
    losses = values[values < 0]
    return {
        "periods": int(values.size),
        "win_rate": float((values > 0).mean()) if values.size else 0.0,
        "average_return_r": float(values.mean()) if values.size else 0.0,
        "average_win_return_r": float(wins.mean()) if wins.size else 0.0,
        "average_loss_return_r": float(losses.mean()) if losses.size else 0.0,
        "best_return_r": float(values.max()) if values.size else 0.0,
        "worst_return_r": float(values.min()) if values.size else 0.0,
    }


def _challenge_window_result(
    trades: pd.DataFrame,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    config: AppConfig,
) -> dict[str, Any]:
    start_day = pd.Timestamp(start_day).tz_localize(None)
    end_day = pd.Timestamp(end_day).tz_localize(None)
    rules = config.prop.challenge
    starting_balance = float(config.prop.starting_balance)
    profit_target_amount = float((rules.profit_target or 0.0) * starting_balance)
    daily_loss_limit = float(rules.daily_loss_limit * starting_balance)
    overall_loss_limit = float(rules.overall_loss_limit * starting_balance)

    balance = starting_balance
    profitable_day_count = 0
    date_range = pd.date_range(start_day, end_day, freq="D")
    if trades.empty:
        trades = trades.copy()
        trades["_day"] = pd.Series(dtype="datetime64[ns]")
    else:
        timestamps = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
        trades = trades.assign(_day=timestamps.dt.floor("D"))

    for day_offset, day in enumerate(date_range, start=1):
        day_trades = trades.loc[trades["_day"] == day]
        day_peak = balance
        realized_today = 0.0
        for pnl_r in day_trades["net_return_r"].to_numpy(dtype=np.float64):
            pnl = float(pnl_r * starting_balance)
            balance += pnl
            realized_today += pnl
            day_peak = max(day_peak, balance)
            daily_drawdown = max(day_peak - balance, 0.0)
            overall_drawdown = max(starting_balance - balance, 0.0)
            if daily_drawdown >= daily_loss_limit or overall_drawdown >= overall_loss_limit:
                return {
                    "passed": False,
                    "breached": True,
                    "days_to_outcome": day_offset,
                    "ending_balance": balance,
                }
        if realized_today > 0.0:
            profitable_day_count += 1
        profit_progress = balance - starting_balance
        target_hit = profit_progress >= profit_target_amount
        profitable_days_hit = profitable_day_count >= rules.minimum_profitable_days
        if target_hit and profitable_days_hit:
            return {
                "passed": True,
                "breached": False,
                "days_to_outcome": day_offset,
                "ending_balance": balance,
            }
    return {
        "passed": False,
        "breached": False,
        "days_to_outcome": None,
        "ending_balance": balance,
    }


def challenge_pass_metrics(
    trades: pd.DataFrame,
    config: AppConfig,
    evaluation_start: str | pd.Timestamp | None,
    evaluation_end: str | pd.Timestamp | None,
) -> dict[str, Any]:
    if evaluation_start is None or evaluation_end is None:
        return {
            "days_to_pass_from_start": None,
            "pass_within_10d": None,
            "pass_within_20d": None,
            "pass_within_30d": None,
            "pass_10d_eligible_starts": 0,
            "pass_20d_eligible_starts": 0,
            "pass_30d_eligible_starts": 0,
            "rolling_pass_days_min": None,
            "rolling_pass_days_max": None,
        }

    start_day = pd.Timestamp(evaluation_start).tz_convert("UTC").tz_localize(None).floor("D")
    end_day = pd.Timestamp(evaluation_end).tz_convert("UTC").tz_localize(None).floor("D")
    if end_day < start_day:
        start_day, end_day = end_day, start_day

    overall = _challenge_window_result(trades, start_day, end_day, config)
    days = pd.date_range(start_day, end_day, freq="D")
    horizon_results: dict[int, list[bool]] = {10: [], 20: [], 30: []}
    pass_day_values: list[int] = []

    for start_idx, rolling_start in enumerate(days):
        full_end = days[-1]
        full_result = _challenge_window_result(trades, rolling_start, full_end, config)
        if full_result["passed"] and full_result["days_to_outcome"] is not None:
            pass_day_values.append(int(full_result["days_to_outcome"]))
        for horizon in (10, 20, 30):
            horizon_end_idx = start_idx + horizon - 1
            if horizon_end_idx >= len(days):
                continue
            horizon_end = days[horizon_end_idx]
            horizon_results[horizon].append(bool(_challenge_window_result(trades, rolling_start, horizon_end, config)["passed"]))

    return {
        "days_to_pass_from_start": int(overall["days_to_outcome"]) if overall["passed"] and overall["days_to_outcome"] is not None else None,
        "pass_within_10d": float(np.mean(horizon_results[10])) if horizon_results[10] else None,
        "pass_within_20d": float(np.mean(horizon_results[20])) if horizon_results[20] else None,
        "pass_within_30d": float(np.mean(horizon_results[30])) if horizon_results[30] else None,
        "pass_10d_eligible_starts": int(len(horizon_results[10])),
        "pass_20d_eligible_starts": int(len(horizon_results[20])),
        "pass_30d_eligible_starts": int(len(horizon_results[30])),
        "rolling_pass_days_min": min(pass_day_values) if pass_day_values else None,
        "rolling_pass_days_max": max(pass_day_values) if pass_day_values else None,
    }


def backtest_diagnostics(
    trades: pd.DataFrame,
    config: AppConfig,
    evaluation_start: str | pd.Timestamp | None,
    evaluation_end: str | pd.Timestamp | None,
) -> dict[str, Any]:
    daily = _period_return_summary(trades, "D")
    weekly = _period_return_summary(trades, "W")
    monthly = _period_return_summary(trades, "M")
    pass_metrics = challenge_pass_metrics(trades, config, evaluation_start, evaluation_end)
    risk_by_expert: dict[str, float] = {}
    if not trades.empty:
        risk_series = trades["risk_fraction"] if "risk_fraction" in trades.columns else pd.Series(0.0, index=trades.index)
        risk_by_expert = (
            trades.assign(_risk_fraction=risk_series)
            .groupby("expert")["_risk_fraction"]
            .mean()
            .astype(float)
            .to_dict()
        )
    return {
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "average_winning_day_profit_r": float(daily["average_win_return_r"]),
        "average_losing_day_loss_r": float(daily["average_loss_return_r"]),
        "average_risk_per_trade_by_expert": risk_by_expert,
        **pass_metrics,
    }


def trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "breakeven_rate": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "average_r": 0.0,
            "median_r": 0.0,
            "average_win_r": 0.0,
            "average_loss_r": 0.0,
            "payoff_ratio": 0.0,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "net_profit_r": 0.0,
            "best_trade_r": 0.0,
            "worst_trade_r": 0.0,
            "max_drawdown_r": 0.0,
            "ending_equity_r": 0.0,
            "peak_equity_r": 0.0,
            "recovery_factor": 0.0,
            "sharpe_like": 0.0,
            "sortino_like": 0.0,
            "ulcer_index": 0.0,
            "return_std_r": 0.0,
            "sqn": 0.0,
            "longest_win_streak": 0,
            "longest_losing_streak": 0,
            "longest_flat_streak": 0,
            "winning_streaks": 0,
            "losing_streaks": 0,
            "trades_per_day": 0.0,
            "daily_trade_count_min": 0,
            "daily_trade_count_max": 0,
            "daily_trade_count_median": 0.0,
            "daily_trade_count_p90": 0.0,
            "average_risk_fraction": 0.0,
            "median_risk_fraction": 0.0,
            "min_risk_fraction": 0.0,
            "max_risk_fraction": 0.0,
            "active_days": 0,
            "first_trade_timestamp": None,
            "last_trade_timestamp": None,
        }

    returns = trades["net_return_r"].to_numpy()
    equity = returns.cumsum()
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    breakeven = returns[returns == 0]
    std_r = float(returns.std())
    downside = returns[returns < 0]
    sharpe_like = 0.0 if std_r == 0 else math.sqrt(252) * returns.mean() / std_r
    downside_std = float(downside.std()) if len(downside) else 0.0
    sortino_like = 0.0 if downside_std == 0.0 else math.sqrt(252) * returns.mean() / downside_std
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff_ratio = 0.0 if avg_loss == 0.0 else avg_win / abs(avg_loss)
    profit_factor = math.inf if len(losses) == 0 and len(wins) else 0.0 if gross_profit == 0.0 else gross_profit / abs(gross_loss)
    sqn = 0.0 if std_r == 0.0 else math.sqrt(len(returns)) * returns.mean() / std_r

    timestamps = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    active_days = int(timestamps.dt.floor("D").nunique()) if timestamps.notna().any() else 0
    trades_per_day = float(len(trades) / active_days) if active_days else 0.0
    daily_counts = timestamps.dt.floor("D").value_counts().to_numpy() if timestamps.notna().any() else np.array([])
    risk_values = trades["risk_fraction"].to_numpy() if "risk_fraction" in trades else np.array([])

    win_streaks = _streak_lengths(returns > 0)
    losing_streaks = _streak_lengths(returns < 0)
    flat_streaks = _streak_lengths(returns == 0)
    equity_summary = _equity_metrics(returns)

    return {
        "num_trades": int(len(trades)),
        "win_rate": float((returns > 0).mean()),
        "loss_rate": float((returns < 0).mean()),
        "breakeven_rate": float((returns == 0).mean()),
        "expectancy_r": float(returns.mean()),
        "profit_factor": _safe_float(float(profit_factor)) if profit_factor != math.inf else None,
        "average_r": float(returns.mean()),
        "median_r": float(np.median(returns)),
        "average_win_r": avg_win,
        "average_loss_r": avg_loss,
        "payoff_ratio": float(payoff_ratio),
        "gross_profit_r": gross_profit,
        "gross_loss_r": gross_loss,
        "net_profit_r": float(returns.sum()),
        "best_trade_r": float(returns.max()),
        "worst_trade_r": float(returns.min()),
        "max_drawdown_r": float(drawdown.min()),
        "ending_equity_r": float(equity_summary["ending_equity_r"]),
        "peak_equity_r": float(equity_summary["peak_equity_r"]),
        "recovery_factor": float(equity_summary["recovery_factor"]),
        "sharpe_like": float(sharpe_like),
        "sortino_like": float(sortino_like),
        "ulcer_index": float(equity_summary["ulcer_index"]),
        "return_std_r": std_r,
        "sqn": float(sqn),
        "longest_win_streak": max(win_streaks, default=0),
        "longest_losing_streak": max(losing_streaks, default=0),
        "longest_flat_streak": max(flat_streaks, default=0),
        "winning_streaks": len(win_streaks),
        "losing_streaks": len(losing_streaks),
        "trades_per_day": trades_per_day,
        "daily_trade_count_min": int(daily_counts.min()) if len(daily_counts) else 0,
        "daily_trade_count_max": int(daily_counts.max()) if len(daily_counts) else 0,
        "daily_trade_count_median": float(np.median(daily_counts)) if len(daily_counts) else 0.0,
        "daily_trade_count_p90": float(np.percentile(daily_counts, 90)) if len(daily_counts) else 0.0,
        "average_risk_fraction": float(risk_values.mean()) if len(risk_values) else 0.0,
        "median_risk_fraction": float(np.median(risk_values)) if len(risk_values) else 0.0,
        "min_risk_fraction": float(risk_values.min()) if len(risk_values) else 0.0,
        "max_risk_fraction": float(risk_values.max()) if len(risk_values) else 0.0,
        "active_days": active_days,
        "first_trade_timestamp": str(timestamps.min()) if timestamps.notna().any() else None,
        "last_trade_timestamp": str(timestamps.max()) if timestamps.notna().any() else None,
    }
