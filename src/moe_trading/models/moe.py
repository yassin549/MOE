"""Shared encoder, TCN experts, manager, and calibration heads."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict

import torch
from torch import nn

from moe_trading.config import ModelConfig
from moe_trading.models.tcn import TCNEncoder


@dataclass(slots=True)
class MoEOutput:
    expert_setup_logits: torch.Tensor
    expert_confidence_logits: torch.Tensor
    expert_probabilities: torch.Tensor
    expert_confidence: torch.Tensor
    expected_returns: torch.Tensor
    direction_logits: torch.Tensor
    directions: torch.Tensor
    manager_trade_logits: torch.Tensor
    manager_dual_logits: torch.Tensor
    manager_trade_probability: torch.Tensor
    manager_dual_probability: torch.Tensor
    manager_context_score: torch.Tensor
    manager_gate_weights: torch.Tensor
    calibrated_probabilities: torch.Tensor


class SharedMultiAssetEncoder(nn.Module):
    """Encode per-asset and shared cross-asset sequences into contextual states."""

    def __init__(self, asset_input_dim: int, cross_input_dim: int, regime_input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.asset_encoder = TCNEncoder(asset_input_dim, config.shared_dim, config.num_tcn_layers, config.kernel_size, config.dropout)
        self.cross_encoder = TCNEncoder(cross_input_dim + regime_input_dim, config.shared_dim, config.num_tcn_layers, config.kernel_size, config.dropout)
        self.fusion = nn.Sequential(
            nn.Linear(config.shared_dim * 2, config.shared_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        asset_sequence: torch.Tensor,
        cross_sequence: torch.Tensor,
        regime_sequence: torch.Tensor,
    ) -> torch.Tensor:
        _, asset_last = self.asset_encoder(asset_sequence)
        _, cross_last = self.cross_encoder(torch.cat([cross_sequence, regime_sequence], dim=-1))
        return self.fusion(torch.cat([asset_last, cross_last], dim=-1))


class SetupExpert(nn.Module):
    """Single setup expert that predicts setup validity, confidence, return, and direction."""

    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = TCNEncoder(input_dim, config.expert_dim, config.num_tcn_layers, config.kernel_size, config.dropout)
        self.head = nn.Sequential(
            nn.Linear(config.expert_dim, config.expert_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.setup_logit = nn.Linear(config.expert_dim, 1)
        self.confidence_logit = nn.Linear(config.expert_dim, 1)
        self.return_head = nn.Linear(config.expert_dim, 1)
        self.direction_head = nn.Linear(config.expert_dim, 1)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, last_state = self.encoder(sequence)
        hidden = self.head(last_state)
        return (
            self.setup_logit(hidden),
            self.confidence_logit(hidden),
            self.return_head(hidden),
            self.direction_head(hidden),
        )


class ManagerNetwork(nn.Module):
    """Cross-asset meta-controller selecting whether expert outputs are tradable."""

    def __init__(self, num_assets: int, num_experts: int, context_dim: int, config: ModelConfig) -> None:
        super().__init__()
        expert_feature_dim = num_assets * num_experts * 5
        self.network = nn.Sequential(
            nn.Linear(expert_feature_dim + context_dim, config.manager_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.manager_dim, config.manager_dim),
            nn.ReLU(),
        )
        self.trade_logit = nn.Linear(config.manager_dim, 1)
        self.dual_trade_logit = nn.Linear(config.manager_dim, 1)
        self.context_logit = nn.Linear(config.manager_dim, 1)
        self.gate_head = nn.Linear(config.manager_dim, num_assets * num_experts)

    def forward(self, expert_features: torch.Tensor, manager_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.network(torch.cat([expert_features, manager_context], dim=-1))
        gate_logits = self.gate_head(hidden)
        return self.trade_logit(hidden), self.dual_trade_logit(hidden), self.context_logit(hidden), gate_logits


class ProbabilityCalibrationHead(nn.Module):
    """Monotone-ish shallow calibrator over raw expert probabilities."""

    def __init__(self, num_assets: int, num_experts: int, config: ModelConfig) -> None:
        super().__init__()
        total = num_assets * num_experts
        self.network = nn.Sequential(
            nn.Linear(total, config.calibration_dim),
            nn.ReLU(),
            nn.Linear(config.calibration_dim, total),
        )

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(probabilities))


class MultiAssetMoE(nn.Module):
    """End-to-end US100/US500 setup-detection and trade-selection model."""

    def __init__(
        self,
        asset_input_dim: int,
        cross_input_dim: int,
        regime_input_dim: int,
        manager_context_dim: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        num_experts = len(config.setup_names)
        self.shared_encoder = SharedMultiAssetEncoder(asset_input_dim, cross_input_dim, regime_input_dim, config)
        expert_input_dim = asset_input_dim + cross_input_dim + regime_input_dim + config.shared_dim
        self.experts = nn.ModuleList([SetupExpert(expert_input_dim, config) for _ in config.setup_names])
        self.manager = ManagerNetwork(2, num_experts, manager_context_dim, config)

    def forward(
        self,
        asset_sequences: dict[str, torch.Tensor],
        cross_sequence: torch.Tensor,
        regime_sequence: torch.Tensor,
        manager_context: torch.Tensor,
        account_context: torch.Tensor | None = None,
    ) -> MoEOutput:
        expert_outputs = self.infer_expert_outputs(asset_sequences, cross_sequence, regime_sequence)
        manager_outputs = self.forward_manager_only(
            expert_outputs["manager_expert_features"],
            manager_context,
            account_context=account_context,
        )
        return MoEOutput(
            expert_setup_logits=expert_outputs["expert_setup_logits"],
            expert_confidence_logits=expert_outputs["expert_confidence_logits"],
            expert_probabilities=expert_outputs["expert_probabilities"],
            expert_confidence=expert_outputs["expert_confidence"],
            expected_returns=expert_outputs["expected_returns"],
            direction_logits=expert_outputs["direction_logits"],
            directions=expert_outputs["directions"],
            manager_trade_logits=manager_outputs["manager_trade_logits"],
            manager_dual_logits=manager_outputs["manager_dual_logits"],
            manager_trade_probability=manager_outputs["manager_trade_probability"],
            manager_dual_probability=manager_outputs["manager_dual_probability"],
            manager_context_score=manager_outputs["manager_context_score"],
            manager_gate_weights=manager_outputs["manager_gate_weights"],
            calibrated_probabilities=expert_outputs["calibrated_probabilities"],
        )

    def infer_expert_outputs(
        self,
        asset_sequences: dict[str, torch.Tensor],
        cross_sequence: torch.Tensor,
        regime_sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        per_asset_setup_logits = []
        per_asset_confidence_logits = []
        per_asset_probabilities = []
        per_asset_confidence = []
        per_asset_returns = []
        per_asset_direction_logits = []
        per_asset_directions = []

        for asset_name in ("US100", "US500"):
            asset_sequence = asset_sequences[asset_name]
            shared_state = self.shared_encoder(asset_sequence, cross_sequence, regime_sequence)
            shared_expanded = shared_state.unsqueeze(1).expand(-1, asset_sequence.size(1), -1)
            expert_input = torch.cat([asset_sequence, cross_sequence, regime_sequence, shared_expanded], dim=-1)

            setup_logits = []
            confidence_logits = []
            return_outputs = []
            direction_logits = []
            for expert in self.experts:
                setup_logit, confidence_logit, return_output, direction_logit = expert(expert_input)
                setup_logits.append(setup_logit)
                confidence_logits.append(confidence_logit)
                return_outputs.append(return_output)
                direction_logits.append(direction_logit)

            setup_logits_cat = torch.cat(setup_logits, dim=-1)
            confidence_logits_cat = torch.cat(confidence_logits, dim=-1)
            return_outputs_cat = torch.cat(return_outputs, dim=-1)
            direction_logits_cat = torch.cat(direction_logits, dim=-1)

            per_asset_setup_logits.append(setup_logits_cat)
            per_asset_confidence_logits.append(confidence_logits_cat)
            per_asset_probabilities.append(torch.sigmoid(setup_logits_cat))
            per_asset_confidence.append(torch.sigmoid(confidence_logits_cat))
            per_asset_returns.append(return_outputs_cat)
            per_asset_direction_logits.append(direction_logits_cat)
            per_asset_directions.append(torch.tanh(direction_logits_cat))

        expert_setup_logits = torch.stack(per_asset_setup_logits, dim=1)
        expert_confidence_logits = torch.stack(per_asset_confidence_logits, dim=1)
        expert_probabilities = torch.stack(per_asset_probabilities, dim=1)
        expert_confidence = torch.stack(per_asset_confidence, dim=1)
        expected_returns = torch.stack(per_asset_returns, dim=1)
        direction_logits = torch.stack(per_asset_direction_logits, dim=1)
        directions = torch.tanh(torch.stack(per_asset_directions, dim=1))

        manager_expert_features = torch.cat(
            [
                expert_setup_logits.reshape(expert_setup_logits.size(0), -1),
                expert_probabilities.reshape(expert_probabilities.size(0), -1),
                expert_confidence.reshape(expert_confidence.size(0), -1),
                expected_returns.reshape(expected_returns.size(0), -1),
                direction_logits.reshape(direction_logits.size(0), -1),
            ],
            dim=-1,
        )
        return {
            "expert_setup_logits": expert_setup_logits,
            "expert_confidence_logits": expert_confidence_logits,
            "expert_probabilities": expert_probabilities,
            "expert_confidence": expert_confidence,
            "expected_returns": expected_returns,
            "direction_logits": direction_logits,
            "directions": directions,
            "manager_expert_features": manager_expert_features,
            "calibrated_probabilities": expert_probabilities,
        }

    def forward_manager_only(
        self,
        manager_expert_features: torch.Tensor,
        manager_context: torch.Tensor,
        account_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        combined_manager_context = manager_context if account_context is None else torch.cat([manager_context, account_context], dim=-1)
        trade_logit, dual_logit, context_logit, gate_logits = self.manager(manager_expert_features, combined_manager_context)
        num_assets = 2
        num_experts = len(self.config.setup_names)
        return {
            "manager_trade_logits": trade_logit,
            "manager_dual_logits": dual_logit,
            "manager_trade_probability": torch.sigmoid(trade_logit),
            "manager_dual_probability": torch.sigmoid(dual_logit),
            "manager_context_score": torch.sigmoid(context_logit),
            "manager_gate_weights": torch.softmax(gate_logits.reshape(manager_expert_features.size(0), num_assets, num_experts), dim=-1),
        }


def save_model(model: MultiAssetMoE, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "config": asdict(model.config)}, path)


def load_model(
    path: str,
    asset_input_dim: int,
    cross_input_dim: int,
    regime_input_dim: int,
    manager_context_dim: int,
    config: ModelConfig,
) -> MultiAssetMoE:
    checkpoint = torch.load(path, map_location="cpu")
    model = MultiAssetMoE(asset_input_dim, cross_input_dim, regime_input_dim, manager_context_dim, config)
    state_dict = checkpoint["state_dict"]
    target_state = model.state_dict()
    for key, value in state_dict.items():
        if key not in target_state:
            continue
        target_value = target_state[key]
        if value.shape == target_value.shape:
            target_state[key] = value
            continue
        if key == "manager.network.0.weight" and value.ndim == 2 and target_value.ndim == 2:
            rows = min(value.size(0), target_value.size(0))
            cols = min(value.size(1), target_value.size(1))
            target_value[:rows, :cols] = value[:rows, :cols]
            target_state[key] = target_value
    model.load_state_dict(target_state, strict=False)
    return model
