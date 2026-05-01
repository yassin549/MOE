"""Minimal experiment tracker for configs, metrics, and artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from moe_trading.utils.io import ensure_dir, save_json


class ExperimentTracker:
    """Persist experiment configs, metrics, and artifact references."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = ensure_dir(output_dir)

    def log_config(self, config: Any) -> None:
        payload = asdict(config) if is_dataclass(config) else config
        save_json(payload, self.output_dir / "config.json")

    def log_metrics(self, name: str, payload: dict[str, Any]) -> None:
        save_json(payload, self.output_dir / f"{name}.json")

    def artifact_path(self, name: str) -> Path:
        return self.output_dir / name
