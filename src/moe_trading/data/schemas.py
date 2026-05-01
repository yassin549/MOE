"""Typed data containers used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MultiAssetFrame:
    frame: pd.DataFrame
    asset_feature_columns: dict[str, list[str]]
    cross_asset_feature_columns: list[str]
    regime_feature_columns: list[str]
    label_columns: list[str]


@dataclass(slots=True)
class SequenceBatch:
    asset_sequences: dict[str, np.ndarray]
    cross_sequences: np.ndarray
    regime_sequences: np.ndarray
    manager_context: np.ndarray
    account_context: np.ndarray
    expert_labels: np.ndarray
    manager_labels: np.ndarray
    asset_manager_labels: np.ndarray
    gate_targets: np.ndarray
    returns: np.ndarray
    timestamps: np.ndarray
