"""Heuristic baseline backtest for each expert.

This script loads the same pre‑processed OHLC CSV used for label generation, applies the
raw setup conditions (without any learned model) and computes basic performance metrics
per expert. The results are written to `heuristic_baseline_report.csv` in the same
directory.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

from moe_trading.labels.generation import _setup_condition, _direction_for_setup
from moe_trading.config import BacktestConfig, LabelConfig


def backtest_expert(df: pd.DataFrame, asset: str, setup: str, config: BacktestConfig, label_cfg: LabelConfig):
    """Run a simple backtest for a single expert on a single asset.

    Returns a dict with win_rate, expectancy, total_trades, long_pct, short_pct.
    """
    prefix = asset.lower()
    # Compute setup condition and direction using the same functions as the label generator
    conditions = _setup_condition(df, asset, label_cfg)
    directions = _direction_for_setup(df, asset)
    condition = conditions[setup].fillna(False)
    direction = pd.Series(directions[setup], index=df.index).fillna(1).astype(int)

    # Use the same simulation logic from generate_labels to compute outcomes
    high = df[f"{prefix}_high"].to_numpy()
    low = df[f"{prefix}_low"].to_numpy()
    close = df[f"{prefix}_close"].to_numpy()
    atr = df[f"{prefix}_atr_15"].to_numpy()

    wins = 0
    total = 0
    net_returns = []
    longs = 0
    shorts = 0
    for idx in np.flatnonzero(condition.to_numpy()):
        # compute stop/target based on ATR as in label generation
        stop_dist = max(float(atr[idx]) * label_cfg.stop_atr_multiple, 1e-6)
        target_dist = stop_dist * label_cfg.target_atr_multiple
        # simple outcome: use the same _simulate_outcome function (imported locally)
        from moe_trading.labels.generation import _simulate_outcome, _net_return_after_costs
        outcome = _simulate_outcome(
            high=high,
            low=low,
            close=close,
            start_idx=idx,
            direction=int(direction[idx]),
            entry=float(close[idx]),
            stop_distance=stop_dist,
            target_distance=target_dist,
            max_holding_bars=label_cfg.max_holding_bars,
        )
        # filter by max adverse excursion
        if outcome.max_adverse_r > label_cfg.max_adverse_excursion_atr:
            continue
        total += 1
        if int(direction[idx]) == 1:
            longs += 1
        else:
            shorts += 1
        if outcome.hit_target:
            wins += 1
        net_ret = _net_return_after_costs(
            return_r=outcome.return_r,
            entry_price=float(close[idx]),
            stop_distance=stop_dist,
            backtest_config=config,
        )
        net_returns.append(net_ret)

    if total == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "long_pct": 0.0,
            "short_pct": 0.0,
        }
    return {
        "trades": int(total),
        "win_rate": wins / total,
        "expectancy": float(np.mean(net_returns)) if net_returns else 0.0,
        "long_pct": longs / total,
        "short_pct": shorts / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Heuristic baseline backtest per expert")
    parser.add_argument("data_path", type=pathlib.Path, help="CSV with generated labels")
    parser.add_argument("output_path", type=pathlib.Path, default=pathlib.Path("heuristic_baseline_report.csv"), nargs="?",
                        help="Where to write the markdown report CSV")
    args = parser.parse_args()
    if not args.data_path.is_file():
        sys.exit(f"File not found: {args.data_path}")
    df = pd.read_csv(args.data_path)
    cfg = BacktestConfig()
    lbl_cfg = LabelConfig()
    assets = ["US100", "US500"]
    setups = [
        "trend_continuation",
        "pullback_continuation",
        "breakout_expansion",
        "mean_reversion",
        "liquidity_sweep_reversal",
        "volatility_compression_expansion",
        "session_open_momentum",
        "exhaustion_failure",
    ]
    rows = []
    for asset in assets:
        for setup in setups:
            metrics = backtest_expert(df, asset, setup, cfg, lbl_cfg)
            rows.append({
                "asset": asset,
                "setup": setup,
                **metrics,
            })
    report_df = pd.DataFrame(rows)
    report_df.to_csv(args.output_path, index=False)
    print(f"Heuristic baseline report written to {args.output_path}")


if __name__ == "__main__":
    main()
