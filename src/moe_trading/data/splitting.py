"""Time-based split logic with embargo support."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def make_time_split(
    frame: pd.DataFrame,
    validation_ratio: float,
    test_ratio: float,
    embargo_bars: int,
) -> TimeSplit:
    """Build a strict chronological split and apply an embargo between partitions."""
    if not 0 < validation_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("validation_ratio and test_ratio must be in (0, 1)")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be < 1")

    total = len(frame)
    train_end = int(total * (1 - validation_ratio - test_ratio))
    val_end = int(total * (1 - test_ratio))

    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end + embargo_bars : val_end].copy()
    test = frame.iloc[val_end + embargo_bars :].copy()
    return TimeSplit(train=train, validation=validation, test=test)


def generate_walk_forward_splits(
    frame: pd.DataFrame,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int,
    embargo_bars: int,
) -> list[TimeSplit]:
    """Create rolling walk-forward splits."""
    splits: list[TimeSplit] = []
    start = 0
    total = len(frame)
    while start + train_size + validation_size + test_size + (2 * embargo_bars) <= total:
        train_end = start + train_size
        val_start = train_end + embargo_bars
        val_end = val_start + validation_size
        test_start = val_end + embargo_bars
        test_end = test_start + test_size
        splits.append(
            TimeSplit(
                train=frame.iloc[start:train_end].copy(),
                validation=frame.iloc[val_start:val_end].copy(),
                test=frame.iloc[test_start:test_end].copy(),
            )
        )
        start += step_size
    return splits
