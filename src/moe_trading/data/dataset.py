"""Sequence dataset creation for MoE training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from moe_trading.data.schemas import MultiAssetFrame
from moe_trading.policy.decision import ACCOUNT_FEATURE_NAMES


@dataclass(slots=True)
class SequenceSample:
    asset_sequences: dict[str, torch.Tensor]
    cross_sequence: torch.Tensor
    regime_sequence: torch.Tensor
    manager_context: torch.Tensor
    account_context: torch.Tensor
    expert_valids: torch.Tensor
    expert_labels: torch.Tensor
    directions: torch.Tensor
    manager_label: torch.Tensor
    asset_manager_labels: torch.Tensor
    gate_supervision_mask: torch.Tensor
    gate_targets: torch.Tensor
    returns: torch.Tensor
    timestamp: str


class MultiAssetSequenceDataset(Dataset[SequenceSample]):
    """Construct rolling closed-candle sequences from the labeled feature frame."""

    def __init__(
        self,
        frame_bundle: MultiAssetFrame,
        sequence_length: int,
        setup_names: list[str],
        default_account_context: np.ndarray | None = None,
        account_context_array: np.ndarray | None = None,
    ) -> None:
        self.bundle = frame_bundle
        self.sequence_length = sequence_length
        self.setup_names = setup_names
        self.frame = frame_bundle.frame.reset_index(drop=True)
        self.indices = np.arange(sequence_length - 1, len(self.frame))
        self.asset_arrays = {
            asset: np.array(self.frame[columns].to_numpy(dtype=np.float32), copy=True)
            for asset, columns in self.bundle.asset_feature_columns.items()
        }
        self.cross_array = np.array(
            self.frame[self.bundle.cross_asset_feature_columns].to_numpy(dtype=np.float32),
            copy=True,
        )
        self.regime_array = np.array(
            self.frame[self.bundle.regime_feature_columns].to_numpy(dtype=np.float32),
            copy=True,
        )
        manager_context_columns = self.bundle.cross_asset_feature_columns + self.bundle.regime_feature_columns
        self.manager_context_array = np.array(
            self.frame[manager_context_columns].to_numpy(dtype=np.float32),
            copy=True,
        )
        if account_context_array is not None:
            self.account_context_array = np.array(account_context_array, dtype=np.float32, copy=True)
        else:
            account_context = default_account_context if default_account_context is not None else np.zeros(len(ACCOUNT_FEATURE_NAMES), dtype=np.float32)
            self.account_context_array = np.repeat(account_context[None, :], len(self.frame), axis=0)
        self.timestamp_array = self.frame["timestamp"].astype(str).to_numpy()

        expert_valids: list[np.ndarray] = []
        expert_labels: list[np.ndarray] = []
        directions: list[np.ndarray] = []
        expert_returns: list[np.ndarray] = []
        for asset in ("US100", "US500"):
            prefix = asset.lower()
            valid_columns = [f"{prefix}_{setup}_valid" for setup in self.setup_names]
            target_columns = [f"{prefix}_{setup}_target" for setup in self.setup_names]
            direction_columns = [f"{prefix}_{setup}_direction" for setup in self.setup_names]
            net_return_columns = [f"{prefix}_{setup}_net_return_r" for setup in self.setup_names]
            return_columns = [f"{prefix}_{setup}_return_r" for setup in self.setup_names]
            expert_valids.append(
                self._columns_or_default(valid_columns, dtype=np.float32, default=0.0)
            )
            expert_labels.append(
                self._columns_or_default(target_columns, dtype=np.float32, default=0.0)
            )
            directions.append(
                self._columns_or_default(direction_columns, dtype=np.float32, default=1.0)
            )
            expert_returns.append(
                self._columns_or_default(net_return_columns, dtype=np.float32, default=None)
                if all(column in self.frame.columns for column in net_return_columns)
                else self._columns_or_default(return_columns, dtype=np.float32, default=0.0)
            )
        self.expert_valid_array = np.stack(expert_valids, axis=1)
        self.expert_label_array = np.stack(expert_labels, axis=1)
        self.direction_array = np.stack(directions, axis=1)
        self.return_array = np.stack(expert_returns, axis=1)
        self.manager_label_array = self._columns_or_default(
            ["manager_trade_target", "manager_dual_trade_target"],
            dtype=np.float32,
            default=0.0,
        )
        self.asset_manager_label_array = self._columns_or_default(
            ["us100_manager_trade_target", "us500_manager_trade_target"],
            dtype=np.float32,
            default=0.0,
        )
        self.gate_supervision_mask_array = self._columns_or_default(
            ["us100_manager_has_valid_setup", "us500_manager_has_valid_setup"],
            dtype=np.float32,
            default=0.0,
        )
        self.gate_target_array = self._columns_or_default(
            ["us100_manager_best_expert", "us500_manager_best_expert"],
            dtype=np.int64,
            default=0,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> SequenceSample:
        end_idx = int(self.indices[index])
        start_idx = end_idx - self.sequence_length + 1

        asset_sequences = {
            asset: torch.from_numpy(array[start_idx : end_idx + 1])
            for asset, array in self.asset_arrays.items()
        }
        cross_sequence = torch.from_numpy(self.cross_array[start_idx : end_idx + 1])
        regime_sequence = torch.from_numpy(self.regime_array[start_idx : end_idx + 1])
        manager_label = torch.from_numpy(self.manager_label_array[end_idx])
        manager_context = torch.from_numpy(self.manager_context_array[end_idx])
        account_context = torch.from_numpy(self.account_context_array[end_idx])

        return SequenceSample(
            asset_sequences=asset_sequences,
            cross_sequence=cross_sequence,
            regime_sequence=regime_sequence,
            manager_context=manager_context,
            account_context=account_context,
            expert_valids=torch.from_numpy(self.expert_valid_array[end_idx]),
            expert_labels=torch.from_numpy(self.expert_label_array[end_idx]),
            directions=torch.from_numpy(self.direction_array[end_idx]),
            manager_label=manager_label,
            asset_manager_labels=torch.from_numpy(self.asset_manager_label_array[end_idx]),
            gate_supervision_mask=torch.from_numpy(self.gate_supervision_mask_array[end_idx]),
            gate_targets=torch.from_numpy(self.gate_target_array[end_idx]),
            returns=torch.from_numpy(self.return_array[end_idx]),
            timestamp=self.timestamp_array[end_idx],
        )

    def _columns_or_default(self, columns: list[str], dtype, default: float | int | None) -> np.ndarray:
        if all(column in self.frame.columns for column in columns):
            return np.array(self.frame[columns].to_numpy(dtype=dtype), copy=True)
        if default is None:
            raise KeyError(f"Missing required columns: {columns}")
        return np.full((len(self.frame), len(columns)), default, dtype=dtype)


def collate_sequence_samples(samples: list[SequenceSample]) -> dict[str, torch.Tensor | list[str] | dict[str, torch.Tensor]]:
    """Collate structured samples for DataLoader use."""
    asset_batches: dict[str, list[torch.Tensor]] = {"US100": [], "US500": []}
    for sample in samples:
        for asset in asset_batches:
            asset_batches[asset].append(sample.asset_sequences[asset])

    return {
        "asset_sequences": {asset: torch.stack(tensors) for asset, tensors in asset_batches.items()},
        "cross_sequence": torch.stack([sample.cross_sequence for sample in samples]),
        "regime_sequence": torch.stack([sample.regime_sequence for sample in samples]),
        "manager_context": torch.stack([sample.manager_context for sample in samples]),
        "account_context": torch.stack([sample.account_context for sample in samples]),
        "expert_valids": torch.stack([sample.expert_valids for sample in samples]),
        "expert_labels": torch.stack([sample.expert_labels for sample in samples]),
        "directions": torch.stack([sample.directions for sample in samples]),
        "manager_labels": torch.stack([sample.manager_label for sample in samples]),
        "asset_manager_labels": torch.stack([sample.asset_manager_labels for sample in samples]),
        "gate_supervision_mask": torch.stack([sample.gate_supervision_mask for sample in samples]),
        "gate_targets": torch.stack([sample.gate_targets for sample in samples]),
        "returns": torch.stack([sample.returns for sample in samples]),
        "timestamps": [sample.timestamp for sample in samples],
    }
