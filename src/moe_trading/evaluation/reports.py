"""Experiment reporting."""

from __future__ import annotations

from datetime import datetime, timezone
import json
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


def append_run_sheet(row: dict[str, Any], path: str | Path, allow_cost_model_mismatch: bool = False) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    frame = pd.DataFrame([row])
    if output_path.exists():
        existing = pd.read_csv(output_path)
        baseline_tag = row.get("baseline_tag")
        cost_model_version = row.get("cost_model_version")
        if (
            not allow_cost_model_mismatch
            and baseline_tag is not None
            and str(baseline_tag) != ""
            and cost_model_version is not None
            and str(cost_model_version) != ""
            and "baseline_tag" in existing.columns
            and "cost_model_version" in existing.columns
        ):
            matching = existing.loc[existing["baseline_tag"].fillna("").astype(str) == str(baseline_tag)]
            mismatched = matching.loc[
                matching["cost_model_version"].fillna("").astype(str).ne(str(cost_model_version))
                & matching["cost_model_version"].fillna("").astype(str).ne("")
            ]
            if not mismatched.empty:
                raise ValueError(
                    f"Run sheet already contains baseline_tag={baseline_tag!r} with a different cost_model_version."
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
        "cost_model_parameters": json.dumps(cost_model_parameters, sort_keys=True) if cost_model_parameters is not None else None,
    }
