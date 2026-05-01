"""Scaffold for the next backtesting engine rewrite.

The previous array-only engine has been intentionally removed. The next
implementation must simulate the model as if it were trading live:

- synchronized closed-candle replay
- incremental feature state
- per-step model inference
- order/fill lifecycle
- deterministic market-state transitions

See ``docs/realtime_backtest_plan.md`` for the implementation plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REMOVAL_MESSAGE = (
    "The previous backtesting engine has been removed. "
    "Use the realtime simulator scaffold and implementation plan in "
    "docs/realtime_backtest_plan.md to build the new live-like engine."
)


@dataclass(slots=True)
class BacktestArtifact:
    name: str
    model_path: Path
    scaler_path: Path
    split_index: int


def _discover_backtest_artifacts(experiment_dir: str | Path) -> list[BacktestArtifact]:
    base = Path(experiment_dir)
    direct_model = base / "model.pt"
    direct_scaler = base / "scaler.json"
    if direct_model.exists():
        if not direct_scaler.exists():
            raise FileNotFoundError(f"Expected scaler.json next to {direct_model}")
        return [BacktestArtifact(name=base.name, model_path=direct_model, scaler_path=direct_scaler, split_index=1)]

    split_dirs = sorted(path for path in base.glob("split_*") if path.is_dir())
    artifacts: list[BacktestArtifact] = []
    for split_dir in split_dirs:
        model_path = split_dir / "model.pt"
        scaler_path = split_dir / "scaler.json"
        if model_path.exists() and scaler_path.exists():
            split_index = int(split_dir.name.split("_")[-1])
            artifacts.append(
                BacktestArtifact(
                    name=split_dir.name,
                    model_path=model_path,
                    scaler_path=scaler_path,
                    split_index=split_index,
                )
            )
    if not artifacts:
        raise FileNotFoundError(f"No backtestable model/scaler artifacts found under {base}")
    return artifacts


def load_backtest_inputs(*args, **kwargs):
    raise RuntimeError(REMOVAL_MESSAGE)


def run_backtest(*args, **kwargs):
    raise RuntimeError(REMOVAL_MESSAGE)


def run_backtest_npz(*args, **kwargs):
    raise RuntimeError(REMOVAL_MESSAGE)


def trade_frame(*args, **kwargs):
    raise RuntimeError(REMOVAL_MESSAGE)


def summarize_result(*args, **kwargs):
    raise RuntimeError(REMOVAL_MESSAGE)


def save_backtest_outputs(*args, **kwargs):
    raise RuntimeError(REMOVAL_MESSAGE)


__all__ = [
    "BacktestArtifact",
    "REMOVAL_MESSAGE",
    "_discover_backtest_artifacts",
]
