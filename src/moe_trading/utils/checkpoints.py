"""Checkpoint discovery helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_model_checkpoint(experiment_dir: str | Path) -> Path:
    """Resolve the most recent split checkpoint or a direct model path."""
    base = Path(experiment_dir)
    direct = base / "model.pt"
    if direct.exists():
        return direct
    split_paths = sorted(base.glob("split_*/model.pt"))
    if not split_paths:
        raise FileNotFoundError(f"No model checkpoint found under {base}")
    return split_paths[-1]
