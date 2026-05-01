"""Experiment reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from moe_trading.cost_model import cost_model_stamp
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


def append_run_sheet(row: dict[str, Any], path: str | Path, allow_mixed_cost_model_versions: bool = False) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    frame = pd.DataFrame([row])
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if not allow_mixed_cost_model_versions and "cost_model_version" in existing.columns and "cost_model_version" in frame.columns:
            existing_versions = {str(v) for v in existing["cost_model_version"].dropna().astype(str).unique()}
            new_versions = {str(v) for v in frame["cost_model_version"].dropna().astype(str).unique()}
            if existing_versions and new_versions and existing_versions != new_versions:
                raise ValueError(
                    f"Cross-run comparison blocked: run sheet has cost model versions {sorted(existing_versions)} but new row has {sorted(new_versions)}. "
                    "Use allow_mixed_cost_model_versions=True (or CLI override) to append anyway."
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
    asset_universe: list[str],
    config: Any,
    baseline_tag: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_name": config_name,
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "model_path": model_path,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "baseline_tag": baseline_tag,
        "asset_universe": sorted(asset_universe),
    }
    metadata.update(cost_model_stamp(config))
    return metadata
