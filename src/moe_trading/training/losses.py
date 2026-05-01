"""Loss functions for expert, manager, calibration, and diversity objectives."""

from __future__ import annotations

import torch
from torch import nn

from moe_trading.config import TrainConfig
from moe_trading.models.moe import MoEOutput


class MultiTaskMoELoss(nn.Module):
    def __init__(self, config: TrainConfig, setup_names: list[str] | None = None) -> None:
        super().__init__()
        self.config = config
        self.setup_names = setup_names or []
        self.regression_loss = nn.SmoothL1Loss(reduction="none")
        self.active_expert_mask: torch.Tensor | None = None

    def set_active_expert_mask(self, active_expert_mask: torch.Tensor | None) -> None:
        self.active_expert_mask = active_expert_mask

    def forward(
        self,
        output: MoEOutput,
        expert_valids: torch.Tensor,
        expert_targets: torch.Tensor,
        direction_targets: torch.Tensor,
        manager_targets: torch.Tensor,
        asset_manager_targets: torch.Tensor,
        gate_supervision_mask: torch.Tensor,
        gate_targets: torch.Tensor,
        return_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        eps = 1e-6
        pos_weight = torch.tensor(float(self.config.class_positive_weight), device=expert_targets.device)
        manager_pos_weight = torch.tensor(float(self.config.manager_positive_weight), device=expert_targets.device)
        active_mask = self.active_expert_mask
        if active_mask is None:
            active_mask = torch.ones_like(expert_targets)
        else:
            active_mask = active_mask.to(expert_targets.device)

        expert_loss_raw = nn.functional.binary_cross_entropy_with_logits(
            output.expert_setup_logits,
            expert_targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        confidence_loss_raw = nn.functional.binary_cross_entropy_with_logits(
            output.expert_confidence_logits,
            expert_valids,
            pos_weight=pos_weight,
            reduction="none",
        )
        expanded_active_mask = active_mask.expand_as(expert_targets)
        expert_weight = self._expert_loss_weight(expert_targets)
        active_weight_sum = expanded_active_mask.sum().clamp_min(1.0)
        expert_loss = (expert_loss_raw * expanded_active_mask * expert_weight).sum() / active_weight_sum
        confidence_loss = (confidence_loss_raw * expanded_active_mask * expert_weight).sum() / active_weight_sum
        manager_trade_raw = nn.functional.binary_cross_entropy_with_logits(
            output.manager_trade_logits.squeeze(-1),
            manager_targets[:, 0],
            pos_weight=manager_pos_weight,
            reduction="none",
        )
        manager_trade_weights = torch.where(
            manager_targets[:, 0] > 0,
            torch.ones_like(manager_targets[:, 0]),
            torch.full_like(manager_targets[:, 0], float(self.config.manager_false_positive_weight)),
        )
        manager_trade = (manager_trade_raw * manager_trade_weights).mean()
        manager_dual_raw = nn.functional.binary_cross_entropy_with_logits(
            output.manager_dual_logits.squeeze(-1),
            manager_targets[:, 1],
            pos_weight=manager_pos_weight,
            reduction="none",
        )
        manager_dual_weights = torch.where(
            manager_targets[:, 1] > 0,
            torch.ones_like(manager_targets[:, 1]),
            torch.full_like(manager_targets[:, 1], float(self.config.manager_false_positive_weight)),
        )
        manager_dual = (manager_dual_raw * manager_dual_weights).mean()
        gate_losses = []
        for asset_idx in range(output.manager_gate_weights.size(1)):
            asset_mask = gate_supervision_mask[:, asset_idx] > 0
            if asset_mask.any():
                target_weights = torch.where(
                    asset_manager_targets[asset_mask, asset_idx] > 0,
                    torch.ones_like(asset_manager_targets[asset_mask, asset_idx]),
                    torch.full_like(
                        asset_manager_targets[asset_mask, asset_idx],
                        float(self.config.gate_negative_weight),
                    ),
                )
                per_sample_loss = nn.functional.nll_loss(
                    torch.log(output.manager_gate_weights[asset_mask, asset_idx, :].clamp(eps, 1.0)),
                    gate_targets[asset_mask, asset_idx].long(),
                    reduction="none",
                )
                gate_losses.append(
                    (per_sample_loss * target_weights).sum() / target_weights.sum().clamp_min(1.0)
                )
        gate_loss = torch.stack(gate_losses).mean() if gate_losses else manager_trade * 0.0
        calibration_loss = output.expert_probabilities.sum() * 0.0
        valid_mask = (expert_valids > 0) & (expanded_active_mask > 0)
        clipped_returns = return_targets.clamp(-3.0, 3.0)
        regression_values = self.regression_loss(output.expected_returns, clipped_returns)
        regression_loss = regression_values[valid_mask].mean() if valid_mask.any() else regression_values.mean() * 0.0
        direction_margin = nn.functional.softplus(-direction_targets * output.direction_logits)
        direction_weights = self._direction_weights(direction_targets, valid_mask)
        direction_weighted = direction_margin * direction_weights
        direction_loss = direction_weighted[valid_mask].mean() if valid_mask.any() else direction_weighted.mean() * 0.0

        flat_probs = output.expert_probabilities.reshape(output.expert_probabilities.size(0), -1)
        diversity_matrix = torch.corrcoef(flat_probs.T).nan_to_num(0.0)
        identity = torch.eye(diversity_matrix.size(0), device=diversity_matrix.device)
        diversity_penalty = ((diversity_matrix - identity) ** 2).mean()

        entropy = -(output.manager_gate_weights.clamp(eps, 1 - eps) * torch.log(output.manager_gate_weights.clamp(eps, 1 - eps))).sum(dim=-1).mean()
        gate_balance_penalty = self._gate_balance_penalty(output.manager_gate_weights)

        total = (
            expert_loss
            + confidence_loss
            + manager_trade
            + manager_dual
            + self.config.gate_supervision_weight * gate_loss
            + self.config.calibration_weight * calibration_loss
            + self.config.regression_weight * regression_loss
            + self.config.direction_loss_weight * direction_loss
            + self.config.gate_balance_weight * gate_balance_penalty
            + self.config.diversity_weight * diversity_penalty
            - self.config.entropy_weight * entropy
        )
        return total, {
            "loss": float(total.detach().cpu()),
            "expert_loss": float(expert_loss.detach().cpu()),
            "confidence_loss": float(confidence_loss.detach().cpu()),
            "manager_trade_loss": float(manager_trade.detach().cpu()),
            "manager_dual_loss": float(manager_dual.detach().cpu()),
            "gate_loss": float(gate_loss.detach().cpu()),
            "calibration_loss": float(calibration_loss.detach().cpu()),
            "regression_loss": float(regression_loss.detach().cpu()),
            "direction_loss": float(direction_loss.detach().cpu()),
            "gate_balance_penalty": float(gate_balance_penalty.detach().cpu()),
            "diversity_penalty": float(diversity_penalty.detach().cpu()),
        }

    def _expert_loss_weight(self, expert_targets: torch.Tensor) -> torch.Tensor:
        if not self.setup_names or not self.config.expert_loss_weights:
            return torch.ones_like(expert_targets)
        weights = [
            float(self.config.expert_loss_weights.get(name, 1.0))
            for name in self.setup_names
        ]
        tensor = torch.tensor(weights, dtype=expert_targets.dtype, device=expert_targets.device)
        return tensor.view(1, 1, -1).expand_as(expert_targets)

    def _direction_weights(self, direction_targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(direction_targets)
        long_weight = self.config.direction_long_weight
        short_weight = self.config.direction_short_weight
        if self.config.direction_auto_balance and valid_mask.any():
            valid_directions = direction_targets[valid_mask]
            long_count = (valid_directions > 0).sum().float()
            short_count = (valid_directions < 0).sum().float()
            total = (long_count + short_count).clamp_min(1.0)
            if long_weight is None and long_count > 0:
                long_weight = float((total / (2.0 * long_count)).detach().cpu())
            if short_weight is None and short_count > 0:
                short_weight = float((total / (2.0 * short_count)).detach().cpu())
        if long_weight is not None:
            weights = torch.where(direction_targets > 0, torch.full_like(weights, float(long_weight)), weights)
        if short_weight is not None:
            weights = torch.where(direction_targets < 0, torch.full_like(weights, float(short_weight)), weights)
        return weights

    def _gate_balance_penalty(self, gate_weights: torch.Tensor) -> torch.Tensor:
        usage = gate_weights.mean(dim=(0, 1))
        if self.setup_names and self.config.gate_target_usage:
            target_values = [
                float(self.config.gate_target_usage.get(name, 1.0 / len(self.setup_names)))
                for name in self.setup_names
            ]
            target = torch.tensor(target_values, dtype=usage.dtype, device=usage.device)
            target = target / target.sum().clamp_min(1e-6)
        else:
            target = torch.full_like(usage, 1.0 / usage.numel())
        return ((usage - target) ** 2).mean()
