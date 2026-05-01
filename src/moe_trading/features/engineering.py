"""Feature generation for aligned multi-asset minute bars."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from moe_trading.config import FeatureConfig


ASSET_PREFIXES = ("us100", "us500")


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=np.float64)
    x_centered = x - x.mean()
    weights = x_centered / float((x_centered**2).sum())
    return series.rolling(window).apply(lambda values: float(np.dot(values, weights)), raw=True)


def _session_features(frame: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    minute_of_day = frame["us100_hour_utc"] * 60 + frame["us100_minute_utc"]
    return pd.DataFrame(
        {
            "minute_sin": np.sin(2 * math.pi * minute_of_day / 1440.0),
            "minute_cos": np.cos(2 * math.pi * minute_of_day / 1440.0),
            "dow_sin": np.sin(2 * math.pi * frame["us100_day_of_week"] / 7.0),
            "dow_cos": np.cos(2 * math.pi * frame["us100_day_of_week"] / 7.0),
            "is_session_open_window": frame["us100_hour_utc"].isin(feature_config.session_open_hours_utc).astype(float),
        },
        index=frame.index,
    )


def _price_action_features(frame: pd.DataFrame, prefix: str, feature_config: FeatureConfig) -> pd.DataFrame:
    open_ = frame[f"{prefix}_open"]
    high = frame[f"{prefix}_high"]
    low = frame[f"{prefix}_low"]
    close = frame[f"{prefix}_close"]
    volume = frame[f"{prefix}_volume"]
    features: dict[str, pd.Series] = {}

    features[f"{prefix}_return_1"] = close.pct_change()
    features[f"{prefix}_log_return_1"] = np.log(close).diff()
    features[f"{prefix}_range"] = _safe_div(high - low, close)
    features[f"{prefix}_body"] = _safe_div(close - open_, open_)
    features[f"{prefix}_upper_wick"] = _safe_div(high - np.maximum(open_, close), close)
    features[f"{prefix}_lower_wick"] = _safe_div(np.minimum(open_, close) - low, close)
    true_range = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    features[f"{prefix}_true_range"] = true_range
    features[f"{prefix}_direction"] = np.sign(close.diff()).fillna(0.0)

    for window in feature_config.volatility_windows:
        features[f"{prefix}_volatility_{window}"] = features[f"{prefix}_log_return_1"].rolling(window).std()
        features[f"{prefix}_atr_{window}"] = true_range.rolling(window).mean()
        features[f"{prefix}_range_mean_{window}"] = features[f"{prefix}_range"].rolling(window).mean()
        if feature_config.use_volume_features:
            features[f"{prefix}_volume_z_{window}"] = (
                volume - volume.rolling(window).mean()
            ) / volume.rolling(window).std()

    compression_atr = features[
        f"{prefix}_atr_{feature_config.volatility_windows[min(len(feature_config.volatility_windows) - 1, 1)]}"
    ]

    for window in feature_config.momentum_windows:
        features[f"{prefix}_momentum_{window}"] = _safe_div(close - close.shift(window), close.shift(window))
        features[f"{prefix}_distance_high_{window}"] = _safe_div(close - high.rolling(window).max(), close)
        features[f"{prefix}_distance_low_{window}"] = _safe_div(close - low.rolling(window).min(), close)

    for window in feature_config.slope_windows:
        features[f"{prefix}_slope_{window}"] = _rolling_slope(close, window)

    for window in feature_config.swing_windows:
        rolling_high = high.rolling(window).max()
        rolling_low = low.rolling(window).min()
        features[f"{prefix}_swing_position_{window}"] = _safe_div(close - rolling_low, rolling_high - rolling_low)

    for window in feature_config.compression_windows:
        features[f"{prefix}_compression_{window}"] = _safe_div(true_range.rolling(window).mean(), compression_atr)

    features[f"{prefix}_three_bar_reversal"] = (
        (close.shift(2) < open_.shift(2))
        & (close.shift(1) < open_.shift(1))
        & (close > open_)
        & (close > high.shift(1))
    ).astype(float)
    features[f"{prefix}_inside_bar"] = ((high < high.shift(1)) & (low > low.shift(1))).astype(float)
    features[f"{prefix}_outside_bar"] = ((high > high.shift(1)) & (low < low.shift(1))).astype(float)
    return pd.DataFrame(features, index=frame.index)


def _cross_asset_features(frame: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    close_100 = frame["us100_close"]
    close_500 = frame["us500_close"]
    ret_100 = frame["us100_log_return_1"]
    ret_500 = frame["us500_log_return_1"]
    spread = np.log(close_100) - np.log(close_500)
    features: dict[str, pd.Series] = {
        "spread_close_ratio": close_100 / close_500,
        "spread_return_diff": ret_100 - ret_500,
        "body_divergence": frame["us100_body"] - frame["us500_body"],
        "range_divergence": frame["us100_range"] - frame["us500_range"],
        "relative_strength_15": frame["us100_momentum_15"] - frame["us500_momentum_15"],
        "relative_strength_30": frame["us100_momentum_30"] - frame["us500_momentum_30"],
    }

    for window in feature_config.correlation_windows:
        features[f"corr_{window}"] = ret_100.rolling(window).corr(ret_500)

    for window in feature_config.zscore_windows:
        rolling_mean = spread.rolling(window).mean()
        rolling_std = spread.rolling(window).std()
        features[f"spread_z_{window}"] = (spread - rolling_mean) / rolling_std
        features[f"return_diff_z_{window}"] = (
            features["spread_return_diff"] - features["spread_return_diff"].rolling(window).mean()
        ) / features["spread_return_diff"].rolling(window).std()

    features["co_momentum"] = ((frame["us100_momentum_5"] > 0) & (frame["us500_momentum_5"] > 0)).astype(float)
    features["divergence_flag"] = (
        np.sign(frame["us100_momentum_5"]).fillna(0.0) != np.sign(frame["us500_momentum_5"]).fillna(0.0)
    ).astype(float)
    return pd.DataFrame(features, index=frame.index)


def _regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    vol_100 = frame["us100_volatility_30"]
    vol_500 = frame["us500_volatility_30"]
    corr = frame["corr_30"]
    joint_volatility = (vol_100 + vol_500) / 2.0
    return pd.DataFrame(
        {
            "joint_volatility": joint_volatility,
            "volatility_ratio": _safe_div(vol_100, vol_500),
            "trend_agreement": (
                np.sign(frame["us100_slope_15"]).fillna(0.0) == np.sign(frame["us500_slope_15"]).fillna(0.0)
            ).astype(float),
            "risk_on_regime": ((corr > 0.5) & (joint_volatility < joint_volatility.rolling(120).median())).astype(float),
            "divergent_regime": ((corr < 0.2) | (frame["divergence_flag"] > 0)).astype(float),
            "high_vol_regime": (joint_volatility > joint_volatility.rolling(240).quantile(0.7)).astype(float),
        },
        index=frame.index,
    )


def build_feature_frame(aligned: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    """Create single-table aligned features for both assets and their interaction."""
    frame_parts: list[pd.DataFrame] = [aligned.reset_index(drop=True)]
    session_frame = _session_features(frame_parts[0], feature_config)
    frame_parts.append(session_frame)
    working_frame = pd.concat(frame_parts, axis=1)
    for prefix in ASSET_PREFIXES:
        frame_parts.append(_price_action_features(working_frame, prefix, feature_config))
        working_frame = pd.concat(frame_parts, axis=1)
    cross_frame = _cross_asset_features(working_frame, feature_config)
    frame_parts.append(cross_frame)
    working_frame = pd.concat(frame_parts, axis=1)
    frame_parts.append(_regime_features(working_frame))
    frame = pd.concat(frame_parts, axis=1)
    frame = frame.copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna().reset_index(drop=True)
    return frame


def collect_feature_columns(frame: pd.DataFrame) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Split engineered feature columns into asset, cross-asset, and regime groups."""
    asset_columns: dict[str, list[str]] = {"US100": [], "US500": []}
    cross_columns: list[str] = []
    regime_columns: list[str] = []

    for column in frame.columns:
        if not is_numeric_dtype(frame[column]):
            continue
        if (
            "_target" in column
            or "_valid" in column
            or "_return_r" in column
            or "_net_return_r" in column
            or "_mae_r" in column
            or "_resolution_bars" in column
            or "_direction" in column
            or "_manager_" in column
            or column.startswith("manager_")
        ):
            continue
        if column.startswith("us100_") and column not in {
            "us100_timestamp_utc",
            "us100_asset",
            "us100_symbol",
            "us100_instrument_id",
        }:
            asset_columns["US100"].append(column)
        elif column.startswith("us500_") and column not in {
            "us500_timestamp_utc",
            "us500_asset",
            "us500_symbol",
            "us500_instrument_id",
        }:
            asset_columns["US500"].append(column)
        elif column in {"joint_volatility", "volatility_ratio", "trend_agreement", "risk_on_regime", "divergent_regime", "high_vol_regime"}:
            regime_columns.append(column)
        elif column not in {
            "timestamp",
            "bar_index",
            "us100_symbol",
            "us500_symbol",
            "us100_instrument_id",
            "us500_instrument_id",
            "us100_session_utc",
            "us500_session_utc",
            "us100_asset",
            "us500_asset",
        }:
            cross_columns.append(column)
    return asset_columns, cross_columns, regime_columns
