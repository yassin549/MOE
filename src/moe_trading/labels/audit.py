"""Phase 1 label and heuristic audit helpers."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any

import numpy as np
import pandas as pd

from moe_trading.config import AppConfig
from moe_trading.labels.generation import SETUP_DIRECTIONAL
from moe_trading.utils.io import ensure_dir, save_json


ASSETS = ("US100", "US500")


def _normal_quantile(confidence_level: float) -> float:
    """Approximate two-tailed z-score for a confidence level."""
    return 1.959963984540054 if confidence_level >= 0.95 else 1.6448536269514722


def _mean_confidence_interval_from_summary(mean: float, sample_std: float, n: int, confidence_level: float) -> tuple[float, float]:
    if n <= 1:
        return mean, mean
    stderr = sample_std / math.sqrt(n)
    margin = _normal_quantile(confidence_level) * stderr
    return mean - margin, mean + margin


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def _direction_balance(direction: pd.Series) -> dict[str, float]:
    if direction.empty:
        return {
            "long_share": 0.0,
            "short_share": 0.0,
            "direction_dominant_share": 0.0,
        }
    long_share = float((direction == 1).mean())
    short_share = float((direction == -1).mean())
    return {
        "long_share": long_share,
        "short_share": short_share,
        "direction_dominant_share": max(long_share, short_share),
    }


def compute_label_audit_table(
    frame: pd.DataFrame,
    setup_names: list[str],
    config: AppConfig,
    assets: tuple[str, ...] = ASSETS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    threshold = float(config.train.direction_symmetry_min_share)
    min_edge_r = float(config.labels.min_manager_edge_r)

    for asset in assets:
        prefix = asset.lower()
        for setup in setup_names:
            present_col = f"{prefix}_{setup}_present"
            valid_col = f"{prefix}_{setup}_valid"
            target_col = f"{prefix}_{setup}_target"
            tradable_col = f"{prefix}_{setup}_tradable"
            direction_col = f"{prefix}_{setup}_direction"
            return_col = f"{prefix}_{setup}_return_r"
            net_return_col = f"{prefix}_{setup}_net_return_r"

            if present_col not in frame.columns:
                continue

            present_mask = frame[present_col].astype(bool)
            valid_mask = frame[valid_col].astype(bool)
            tradable_mask = frame[tradable_col].astype(bool)
            direction = frame.loc[present_mask, direction_col]
            raw_returns = frame.loc[present_mask, return_col]
            net_returns = frame.loc[present_mask, net_return_col]
            valid_net_returns = frame.loc[valid_mask, net_return_col]
            positive_net_rate = float((valid_net_returns > 0).mean()) if valid_mask.any() else 0.0
            balance = _direction_balance(direction)
            min_side_share = min(balance["long_share"], balance["short_share"])

            rows.append(
                {
                    "asset": asset,
                    "setup": setup,
                    "is_directional": bool(SETUP_DIRECTIONAL.get(setup, True)),
                    "setup_present_count": int(present_mask.sum()),
                    "valid_label_count": int(valid_mask.sum()),
                    "tradable_label_count": int(tradable_mask.sum()),
                    "winning_label_count": int(frame.loc[valid_mask, target_col].sum()) if valid_mask.any() else 0,
                    "label_win_rate": float(frame.loc[valid_mask, target_col].mean()) if valid_mask.any() else 0.0,
                    "positive_net_rate": positive_net_rate,
                    "label_precision": float(frame.loc[valid_mask, target_col].mean()) if valid_mask.any() else 0.0,
                    "direction_long_share": balance["long_share"],
                    "direction_short_share": balance["short_share"],
                    "direction_dominant_share": balance["direction_dominant_share"],
                    "direction_minority_share": min_side_share,
                    "direction_symmetry_threshold": threshold,
                    "direction_symmetry_pass": (not SETUP_DIRECTIONAL.get(setup, True)) or (min_side_share >= threshold),
                    "raw_heuristic_expectancy_r": _safe_mean(raw_returns),
                    "post_cost_expectancy_r": _safe_mean(net_returns),
                    "valid_post_cost_expectancy_r": _safe_mean(valid_net_returns),
                    "valid_post_cost_std_r": float(valid_net_returns.std(ddof=1)) if len(valid_net_returns) > 1 else 0.0,
                    "tradable_share_of_present": float(tradable_mask.sum() / max(present_mask.sum(), 1)),
                    "mean_manager_edge_threshold_r": min_edge_r,
                }
            )
    return pd.DataFrame(rows)


def compute_heuristic_baseline_table(label_audit: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    if label_audit.empty:
        return pd.DataFrame(
            columns=[
                "asset",
                "setup",
                "trades",
                "win_rate",
                "expectancy_r",
                "expectancy_ci_lower_r",
                "expectancy_ci_upper_r",
                "expectancy_confidence_level",
                "expectancy_evaluable",
                "expectancy_significance_flag",
                "expectancy_status",
                "profitability_gate_pass",
                "long_share",
                "short_share",
            ]
        )
    baseline = label_audit[
        [
            "asset",
            "setup",
            "setup_present_count",
            "positive_net_rate",
            "post_cost_expectancy_r",
            "valid_post_cost_std_r",
            "direction_long_share",
            "direction_short_share",
        ]
    ].copy()
    baseline = baseline.rename(
        columns={
            "setup_present_count": "trades",
            "positive_net_rate": "win_rate",
            "post_cost_expectancy_r": "expectancy_r",
            "direction_long_share": "long_share",
            "direction_short_share": "short_share",
        }
    )
    min_trades = max(int(config.backtest.expectancy_min_trades), 1)
    confidence_level = float(config.backtest.expectancy_confidence_level)
    baseline["expectancy_confidence_level"] = confidence_level
    baseline["expectancy_ci_lower_r"] = baseline["expectancy_r"]
    baseline["expectancy_ci_upper_r"] = baseline["expectancy_r"]
    baseline["expectancy_evaluable"] = baseline["trades"] >= min_trades
    baseline["expectancy_status"] = np.where(baseline["expectancy_evaluable"], "evaluable", "not_enough_data")
    baseline["expectancy_significance_flag"] = False
    baseline["profitability_gate_pass"] = False

    evaluable_mask = baseline["expectancy_evaluable"]
    for idx in baseline.index[evaluable_mask]:
        lower, upper = _mean_confidence_interval_from_summary(
            mean=float(baseline.at[idx, "expectancy_r"]),
            sample_std=float(baseline.at[idx, "valid_post_cost_std_r"]),
            n=int(baseline.at[idx, "trades"]),
            confidence_level=confidence_level,
        )
        baseline.at[idx, "expectancy_ci_lower_r"] = lower
        baseline.at[idx, "expectancy_ci_upper_r"] = upper
    baseline.loc[evaluable_mask, "expectancy_significance_flag"] = baseline.loc[evaluable_mask, "expectancy_r"] > 0.0
    baseline.loc[evaluable_mask, "profitability_gate_pass"] = (
        (baseline.loc[evaluable_mask, "expectancy_r"] > 0.0)
        & (baseline.loc[evaluable_mask, "expectancy_ci_lower_r"] >= 0.0)
    )
    baseline = baseline.drop(columns=["valid_post_cost_std_r"])
    return baseline


def generate_phase1_audit_artifacts(frame: pd.DataFrame, config: AppConfig, output_dir: str | Path) -> dict[str, Any]:
    output_path = ensure_dir(output_dir)
    label_audit = compute_label_audit_table(frame, config.model.setup_names, config)
    heuristic_baseline = compute_heuristic_baseline_table(label_audit, config)

    label_audit.to_csv(output_path / "label_audit.csv", index=False)
    heuristic_baseline.to_csv(output_path / "heuristic_baseline.csv", index=False)
    save_json({"rows": label_audit.to_dict(orient="records")}, output_path / "label_audit.json")
    save_json({"rows": heuristic_baseline.to_dict(orient="records")}, output_path / "heuristic_baseline.json")

    return {
        "label_audit_path": str(output_path / "label_audit.csv"),
        "heuristic_baseline_path": str(output_path / "heuristic_baseline.csv"),
        "label_failures": int((~label_audit["direction_symmetry_pass"]).sum()) if not label_audit.empty else 0,
        "negative_expectancy_labels": int((label_audit["post_cost_expectancy_r"] <= 0.0).sum()) if not label_audit.empty else 0,
        "negative_expectancy_heuristics": int((heuristic_baseline["expectancy_r"] <= 0.0).sum()) if not heuristic_baseline.empty else 0,
    }
