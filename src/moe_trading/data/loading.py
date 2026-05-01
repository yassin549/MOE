"""Data loading and synchronization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from moe_trading.config import DataConfig


REQUIRED_COLUMNS = [
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "hour_utc",
    "minute_utc",
    "day_of_week",
    "session_utc",
]


def load_asset_frame(path: str | Path, asset_name: str, data_config: DataConfig) -> pd.DataFrame:
    """Load a single asset file and enforce timestamp ordering."""
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{asset_name} is missing required columns: {missing}")

    frame = frame.copy()
    frame[data_config.timestamp_col] = pd.to_datetime(frame[data_config.timestamp_col], utc=True)
    frame = frame.sort_values(data_config.timestamp_col).drop_duplicates(data_config.timestamp_col)
    frame["asset"] = asset_name
    return frame.reset_index(drop=True)


def align_assets(us100: pd.DataFrame, us500: pd.DataFrame, data_config: DataConfig) -> pd.DataFrame:
    """Inner-join assets on closed-candle timestamps and preserve bar integrity metadata."""
    left = us100.add_prefix("us100_")
    right = us500.add_prefix("us500_")
    aligned = left.merge(
        right,
        left_on=f"us100_{data_config.timestamp_col}",
        right_on=f"us500_{data_config.timestamp_col}",
        how="inner",
        validate="one_to_one",
    )
    aligned = aligned.rename(columns={f"us100_{data_config.timestamp_col}": "timestamp"})
    aligned = aligned.drop(columns=[f"us500_{data_config.timestamp_col}"])
    aligned = aligned.sort_values("timestamp").reset_index(drop=True)
    aligned["bar_index"] = range(len(aligned))
    return aligned


def load_multi_asset_frame(data_config: DataConfig) -> pd.DataFrame:
    """Load and align the configured US100 and US500 files."""
    us100 = load_asset_frame(data_config.us100_file, "US100", data_config)
    us500 = load_asset_frame(data_config.us500_file, "US500", data_config)
    aligned = align_assets(us100, us500, data_config)
    if data_config.max_rows is not None:
        aligned = aligned.tail(data_config.max_rows).reset_index(drop=True)
    return aligned
