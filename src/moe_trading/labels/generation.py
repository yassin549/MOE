"""Setup-specific labeling for MoE experts and the manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from moe_trading.config import BacktestConfig, LabelConfig
from moe_trading.cost_model import CostModelSpec, cost_model_version, total_round_trip_bps


SETUP_DIRECTIONAL = {
    "trend_continuation": True,
    "pullback_continuation": True,
    "breakout_expansion": True,
    "mean_reversion": True,
    "liquidity_sweep_reversal": True,
    "volatility_compression_expansion": True,
    "session_open_momentum": True,
    "exhaustion_failure": True,
}


@dataclass(slots=True)
class TradeOutcome:
    hit_target: int
    hit_stop: int
    return_r: float
    max_adverse_r: float
    resolution_bars: int


def _simulate_outcome(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start_idx: int,
    direction: int,
    entry: float,
    stop_distance: float,
    target_distance: float,
    max_holding_bars: int,
) -> TradeOutcome:
    stop_price = entry - direction * stop_distance
    target_price = entry + direction * target_distance
    max_adverse = 0.0
    for offset in range(1, max_holding_bars + 1):
        idx = start_idx + offset
        if idx >= len(close):
            break
        bar_high = high[idx]
        bar_low = low[idx]
        if direction > 0:
            adverse = max(0.0, (entry - bar_low) / stop_distance)
            target_hit = bar_high >= target_price
            stop_hit = bar_low <= stop_price
        else:
            adverse = max(0.0, (bar_high - entry) / stop_distance)
            target_hit = bar_low <= target_price
            stop_hit = bar_high >= stop_price
        max_adverse = max(max_adverse, adverse)

        if target_hit and stop_hit:
            return TradeOutcome(0, 1, -1.0, max_adverse, offset)
        if target_hit:
            return TradeOutcome(1, 0, target_distance / stop_distance, max_adverse, offset)
        if stop_hit:
            return TradeOutcome(0, 1, -1.0, max_adverse, offset)

    final_idx = min(start_idx + max_holding_bars, len(close) - 1)
    pnl = direction * (close[final_idx] - entry) / stop_distance
    return TradeOutcome(0, 0, float(pnl), max_adverse, final_idx - start_idx)


def _setup_condition(frame: pd.DataFrame, asset: str, label_config: LabelConfig) -> dict[str, pd.Series]:
    prefix = asset.lower()
    wick_strength = frame[[f"{prefix}_lower_wick", f"{prefix}_upper_wick"]].max(axis=1)
    wick_threshold = wick_strength.rolling(240, min_periods=30).quantile(0.9)
    compression_threshold = frame[f"{prefix}_compression_20"].rolling(240, min_periods=30).quantile(0.2)
    trend = (
        (frame[f"{prefix}_slope_15"] > label_config.min_trend_strength)
        & (frame[f"{prefix}_momentum_10"] > 0)
        & (frame["trend_agreement"] > 0)
    )
    pullback = (
        trend
        & (frame[f"{prefix}_distance_high_20"] < 0)
        & (frame[f"{prefix}_atr_15"] > 0)
        & (_abs(frame[f"{prefix}_body"]) < frame[f"{prefix}_atr_15"] * label_config.pullback_depth_atr / frame[f"{prefix}_close"])
    )
    breakout = (
        (frame[f"{prefix}_distance_high_20"] > -0.0005)
        & (frame[f"{prefix}_volume_z_15"] > 0)
        & (frame["spread_return_diff"].abs() < frame["spread_return_diff"].rolling(30).std())
    )
    mean_rev = (
        (frame["spread_z_20"].abs() > label_config.mean_reversion_extension_zscore)
        & (frame["divergent_regime"] > 0)
    )
    sweep = (
        (wick_strength > wick_threshold.fillna(wick_strength.quantile(0.9)))
        & (wick_strength > (frame[f"{prefix}_range"] * 0.35).fillna(0.0))
    )
    compression = (
        (frame[f"{prefix}_compression_20"] <= compression_threshold.fillna(label_config.compression_threshold))
        & (frame["joint_volatility"] < frame["joint_volatility"].rolling(120).median())
    )
    session_open = frame["is_session_open_window"] > 0
    exhaustion = (
        (frame[f"{prefix}_outside_bar"] > 0)
        & (_abs(frame[f"{prefix}_body"]) < label_config.exhaustion_reversal_body_threshold / 100.0)
    )
    return {
        "trend_continuation": trend,
        "pullback_continuation": pullback,
        "breakout_expansion": breakout,
        "mean_reversion": mean_rev,
        "liquidity_sweep_reversal": sweep,
        "volatility_compression_expansion": compression,
        "session_open_momentum": session_open,
        "exhaustion_failure": exhaustion,
    }


def _abs(series: pd.Series) -> pd.Series:
    return series.abs()


def _direction_for_setup(frame: pd.DataFrame, asset: str) -> dict[str, pd.Series]:
    """Compute the intended trade direction for each setup without mutating labels."""
    prefix = asset.lower()
    return {
        "trend_continuation": np.sign(frame[f"{prefix}_momentum_10"]).replace(0, 1),
        "pullback_continuation": np.sign(frame[f"{prefix}_slope_15"]).replace(0, 1),
        "breakout_expansion": np.sign(frame[f"{prefix}_momentum_5"]).replace(0, 1),
        "mean_reversion": -np.sign(frame["spread_z_20"]).replace(0, 1),
        "liquidity_sweep_reversal": np.where(frame[f"{prefix}_lower_wick"] > frame[f"{prefix}_upper_wick"], 1, -1),
        "volatility_compression_expansion": np.sign(frame[f"{prefix}_momentum_3"]).replace(0, 1),
        "session_open_momentum": np.sign(frame[f"{prefix}_momentum_3"]).replace(0, 1),
        "exhaustion_failure": -np.sign(frame[f"{prefix}_momentum_5"]).replace(0, 1),
    }




def _required_setup_columns(asset: str) -> list[str]:
    prefix = asset.lower()
    return [
        f"{prefix}_lower_wick",
        f"{prefix}_upper_wick",
        f"{prefix}_compression_20",
        f"{prefix}_slope_15",
        f"{prefix}_momentum_10",
        f"{prefix}_distance_high_20",
        f"{prefix}_atr_15",
        f"{prefix}_body",
        f"{prefix}_volume_z_15",
        f"{prefix}_range",
        f"{prefix}_outside_bar",
        "trend_agreement",
        "spread_return_diff",
        "spread_z_20",
        "divergent_regime",
        "joint_volatility",
        "is_session_open_window",
    ]


def _setup_input_available_mask(frame: pd.DataFrame, asset: str) -> np.ndarray:
    required = [c for c in _required_setup_columns(asset) if c in frame.columns]
    if not required:
        return np.zeros(len(frame), dtype=bool)
    return frame[required].notna().all(axis=1).to_numpy()


def run_label_leakage_checks(frame: pd.DataFrame, setup_names: list[str], max_holding_bars: int) -> dict[str, Any]:
    violations: list[str] = []
    for asset in ("us100", "us500"):
        for setup in setup_names:
            present_col = f"{asset}_{setup}_present"
            trigger_col = f"{asset}_{setup}_trigger_timestamp"
            tradable_col = f"{asset}_{setup}_earliest_tradable_timestamp"
            horizon_col = f"{asset}_{setup}_outcome_horizon_bars"
            availability_col = f"{asset}_{setup}_setup_inputs_available"
            if present_col not in frame.columns:
                continue
            present = frame[present_col].astype(bool)
            if availability_col in frame.columns and (present & ~frame[availability_col].astype(bool)).any():
                violations.append(f"{asset}:{setup}:present_without_trigger_time_inputs")
            if trigger_col in frame.columns and tradable_col in frame.columns:
                trigger_ts = pd.to_datetime(frame[trigger_col], utc=True, errors="coerce")
                tradable_ts = pd.to_datetime(frame[tradable_col], utc=True, errors="coerce")
                bad_order = present & (tradable_ts <= trigger_ts)
                if bad_order.any():
                    violations.append(f"{asset}:{setup}:earliest_tradable_not_after_trigger")
            if horizon_col in frame.columns and present.any():
                invalid_horizon = present & (frame[horizon_col].astype(float) <= 0)
                if invalid_horizon.any():
                    violations.append(f"{asset}:{setup}:non_positive_outcome_horizon")
            resolution_col = f"{asset}_{setup}_resolution_bars"
            if resolution_col in frame.columns and present.any():
                overflow = present & (frame[resolution_col].astype(float) > float(max_holding_bars))
                if overflow.any():
                    violations.append(f"{asset}:{setup}:resolution_exceeds_horizon")
    return {"passed": not violations, "violations": violations}

def _net_return_after_costs(
    return_r: float,
    entry_price: float,
    stop_distance: float,
    backtest_config: BacktestConfig,
) -> float:
    total_bps = total_round_trip_bps(backtest_config)
    total_price_cost = entry_price * (total_bps / 10_000.0)
    return float(return_r - (total_price_cost / max(stop_distance, 1e-6)))


def generate_labels(
    frame: pd.DataFrame,
    label_config: LabelConfig,
    setup_names: list[str],
    backtest_config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Create setup-validity and trade-outcome labels for both assets and the manager."""
    labeled = frame.copy()
    backtest_config = backtest_config or BacktestConfig()
    output_columns: dict[str, np.ndarray | pd.Series] = {}
    manager_targets: dict[str, np.ndarray] = {}
    model_version = cost_model_version(CostModelSpec.from_backtest_config(backtest_config))

    timestamp_series = labeled["timestamp"].astype(str) if "timestamp" in labeled.columns else pd.Series(labeled.index.astype(str), index=labeled.index)

    for asset in ("US100", "US500"):
        prefix = asset.lower()
        setup_conditions = _setup_condition(labeled, asset, label_config)
        directions = _direction_for_setup(labeled, asset)
        per_setup_valids: list[np.ndarray] = []
        per_setup_returns: list[np.ndarray] = []
        per_setup_wins: list[np.ndarray] = []

        high = labeled[f"{prefix}_high"].to_numpy()
        low = labeled[f"{prefix}_low"].to_numpy()
        close = labeled[f"{prefix}_close"].to_numpy()
        atr = labeled[f"{prefix}_atr_15"].to_numpy()

        for setup in setup_names:
            input_available = _setup_input_available_mask(labeled, asset)
            condition = setup_conditions[setup].fillna(False).to_numpy() & input_available
            condition[-1] = False
            direction = pd.Series(directions[setup], index=labeled.index).fillna(1).astype(int).to_numpy()
            present_values = condition.astype(np.int64)
            valid_values = np.zeros(len(labeled), dtype=np.int64)
            win_values = np.zeros(len(labeled), dtype=np.int64)
            tradable_values = np.zeros(len(labeled), dtype=np.int64)
            return_r_values = np.zeros(len(labeled), dtype=np.float32)
            net_return_r_values = np.zeros(len(labeled), dtype=np.float32)
            mae_values = np.zeros(len(labeled), dtype=np.float32)
            resolution_values = np.zeros(len(labeled), dtype=np.float32)

            for idx in np.flatnonzero(condition):
                stop_distance = max(float(atr[idx]) * label_config.stop_atr_multiple, 1e-6)
                target_distance = stop_distance * label_config.target_atr_multiple
                outcome = _simulate_outcome(
                    high=high,
                    low=low,
                    close=close,
                    start_idx=idx,
                    direction=int(direction[idx]),
                    entry=float(close[idx]),
                    stop_distance=stop_distance,
                    target_distance=target_distance,
                    max_holding_bars=label_config.max_holding_bars,
                )
                if outcome.max_adverse_r <= label_config.max_adverse_excursion_atr:
                    valid_values[idx] = 1
                if valid_values[idx] and outcome.hit_target:
                    win_values[idx] = 1
                return_r_values[idx] = outcome.return_r
                net_return_r_values[idx] = _net_return_after_costs(
                    return_r=outcome.return_r,
                    entry_price=float(close[idx]),
                    stop_distance=stop_distance,
                    backtest_config=backtest_config,
                )
                if valid_values[idx] and net_return_r_values[idx] > label_config.min_manager_edge_r:
                    tradable_values[idx] = 1
                mae_values[idx] = outcome.max_adverse_r
                resolution_values[idx] = outcome.resolution_bars

            present_col = f"{prefix}_{setup}_present"
            valid_col = f"{prefix}_{setup}_valid"
            target_col = f"{prefix}_{setup}_target"
            tradable_col = f"{prefix}_{setup}_tradable"
            return_col = f"{prefix}_{setup}_return_r"
            net_return_col = f"{prefix}_{setup}_net_return_r"
            mae_col = f"{prefix}_{setup}_mae_r"
            resolution_col = f"{prefix}_{setup}_resolution_bars"
            direction_col = f"{prefix}_{setup}_direction"
            trigger_col = f"{prefix}_{setup}_trigger_timestamp"
            earliest_tradable_col = f"{prefix}_{setup}_earliest_tradable_timestamp"
            horizon_col = f"{prefix}_{setup}_outcome_horizon_bars"
            cost_col = f"{prefix}_{setup}_cost_model"
            setup_input_col = f"{prefix}_{setup}_setup_inputs_available"

            output_columns[present_col] = present_values
            output_columns[valid_col] = valid_values
            output_columns[target_col] = win_values
            output_columns[tradable_col] = tradable_values
            output_columns[return_col] = return_r_values
            output_columns[net_return_col] = net_return_r_values
            output_columns[mae_col] = mae_values
            output_columns[resolution_col] = resolution_values
            output_columns[direction_col] = direction
            output_columns[trigger_col] = timestamp_series.to_numpy()
            output_columns[earliest_tradable_col] = timestamp_series.shift(-1).fillna(timestamp_series).to_numpy()
            output_columns[horizon_col] = np.full(len(labeled), label_config.max_holding_bars, dtype=np.int64)
            output_columns[cost_col] = np.full(len(labeled), model_version, dtype=object)
            output_columns[setup_input_col] = input_available.astype(np.int64)
            manager_targets[f"{prefix}_{setup}"] = tradable_values.astype(np.int64)
            per_setup_valids.append(valid_values)
            per_setup_returns.append(net_return_r_values)
            per_setup_wins.append(win_values)

        valid_matrix = np.column_stack(per_setup_valids)
        return_matrix = np.column_stack(per_setup_returns)
        win_matrix = np.column_stack(per_setup_wins)
        masked_returns = np.where(valid_matrix > 0, return_matrix, -np.inf)
        best_expert_idx = np.argmax(masked_returns, axis=1).astype(np.int64)
        has_valid = valid_matrix.sum(axis=1) > 0
        best_expert_idx = np.where(has_valid, best_expert_idx, -1)
        best_returns = np.where(has_valid, masked_returns[np.arange(len(labeled)), np.maximum(best_expert_idx, 0)], 0.0)
        asset_trade_target = (best_returns > label_config.min_manager_edge_r).astype(np.int64)
        output_columns[f"{prefix}_manager_trade_target"] = asset_trade_target
        output_columns[f"{prefix}_manager_best_expert"] = best_expert_idx
        output_columns[f"{prefix}_manager_best_net_return_r"] = best_returns.astype(np.float32)
        output_columns[f"{prefix}_manager_has_valid_setup"] = has_valid.astype(np.int64)

    manager_us100 = np.column_stack([values for key, values in manager_targets.items() if key.startswith("us100_")])
    manager_us500 = np.column_stack([values for key, values in manager_targets.items() if key.startswith("us500_")])
    output_columns["manager_trade_target"] = (
        (output_columns["us100_manager_trade_target"] + output_columns["us500_manager_trade_target"]) > 0
    ).astype(np.int64)
    output_columns["manager_dual_trade_target"] = (
        (output_columns["us100_manager_trade_target"] > 0) & (output_columns["us500_manager_trade_target"] > 0)
    ).astype(np.int64)
    return pd.concat([labeled, pd.DataFrame(output_columns, index=labeled.index)], axis=1)
