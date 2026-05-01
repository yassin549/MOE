"""Experiment reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from moe_trading.utils.io import ensure_dir


def write_report(payload: dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    lines = ["# Experiment Report", ""]
    for key, value in payload.items():
        lines.append(f"- **{key}**: {value}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def flatten_for_sheet(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        nested_key = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
        if isinstance(value, dict):
            flat.update(flatten_for_sheet(value, nested_key))
        else:
            flat[nested_key] = value
    return flat


def append_run_sheet(
    row: dict[str, Any],
    path: str | Path,
    allow_cost_model_mismatch: bool = False,
) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    frame = pd.DataFrame([row])
    if output_path.exists():
        existing = pd.read_csv(output_path)
        incoming_version = row.get("cost_model_version")
        incoming_baseline = row.get("baseline_tag")
        if not allow_cost_model_mismatch and incoming_version and incoming_baseline:
            compare_subset = existing
            if "baseline_tag" in existing.columns:
                compare_subset = existing.loc[existing["baseline_tag"] == incoming_baseline]
            if not compare_subset.empty and "cost_model_version" in compare_subset.columns:
                existing_versions = set(compare_subset["cost_model_version"].dropna().astype(str).tolist())
                if existing_versions and existing_versions != {str(incoming_version)}:
                    raise ValueError(
                        "Cross-run comparison blocked: baseline_tag contains mixed cost_model_version values. "
                        "Set allow_cost_model_mismatch=True to override."
                    )
        frame = pd.concat([existing, frame], ignore_index=True, sort=False)
    frame.to_csv(output_path, index=False)


def make_run_metadata(
    config_name: str,
    experiment_name: str,
    output_dir: str,
    model_path: str | None,
    evaluation_start: str | None,
    evaluation_end: str | None,
    baseline_tag: str | None = None,
    cost_model_version: str | None = None,
    cost_model_hash: str | None = None,
    cost_model_parameters: dict[str, Any] | None = None,
    data_period_start: str | None = None,
    data_period_end: str | None = None,
    asset_universe: str | None = None,
) -> dict[str, Any]:
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_name": config_name,
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "model_path": model_path,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "baseline_tag": baseline_tag,
        "cost_model_version": cost_model_version,
        "cost_model_hash": cost_model_hash,
        "cost_model_parameters": cost_model_parameters or {},
        "data_period_start": data_period_start or evaluation_start,
        "data_period_end": data_period_end or evaluation_end,
        "asset_universe": asset_universe,
    }
