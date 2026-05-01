"""Closed-candle live inference pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import torch

from moe_trading.account.rules import PropRuleEngine
from moe_trading.account.state import AccountState
from moe_trading.config import AppConfig
from moe_trading.data.scaling import FeatureScaler
from moe_trading.models.moe import load_model
from moe_trading.pipeline import build_feature_bundle
from moe_trading.policy.decision import ACCOUNT_FEATURE_NAMES, encode_account_state, expert_trade_threshold, routed_expert_scores
from moe_trading.utils.calibration import apply_calibration, load_calibration_artifact
from moe_trading.utils.checkpoints import resolve_model_checkpoint


@dataclass(slots=True)
class LiveTradeDecision:
    timestamp: str
    trade: bool
    dual_trade: bool
    assets: list[str]
    experts: dict[str, str]
    directions: dict[str, int]
    probabilities: dict[str, float]
    context_score: float


def infer_latest_decision(config: AppConfig) -> LiveTradeDecision:
    """Generate the latest structured decision from the most recent closed candle."""
    bundle = build_feature_bundle(config)
    model_path = resolve_model_checkpoint(config.experiment.output_dir)
    calibration_artifact = load_calibration_artifact(model_path.parent / "calibration.json")
    scaler_payload = json.loads((model_path.parent / "scaler.json").read_text(encoding="utf-8"))
    scaler = FeatureScaler.from_dict({"means": scaler_payload["means"], "stds": scaler_payload["stds"]})
    feature_columns = scaler_payload["feature_columns"]
    scaled_frame = scaler.transform(bundle.frame, feature_columns)
    latest = scaled_frame.iloc[-config.data.sequence_length :].reset_index(drop=True)

    asset_input_dim = len(bundle.asset_feature_columns["US100"])
    cross_input_dim = len(bundle.cross_asset_feature_columns)
    regime_input_dim = len(bundle.regime_feature_columns)
    manager_context_dim = cross_input_dim + regime_input_dim + len(ACCOUNT_FEATURE_NAMES)
    model = load_model(
        str(model_path),
        asset_input_dim,
        cross_input_dim,
        regime_input_dim,
        manager_context_dim,
        config.model,
    )
    model.eval()

    with torch.no_grad():
        asset_sequences = {
            asset: torch.tensor(
                latest[columns].to_numpy(dtype=np.float32)[None, :, :],
                dtype=torch.float32,
            )
            for asset, columns in bundle.asset_feature_columns.items()
        }
        cross_seq = torch.tensor(
            latest[bundle.cross_asset_feature_columns].to_numpy(dtype=np.float32)[None, :, :],
            dtype=torch.float32,
        )
        regime_seq = torch.tensor(
            latest[bundle.regime_feature_columns].to_numpy(dtype=np.float32)[None, :, :],
            dtype=torch.float32,
        )
        manager_context_cols = bundle.cross_asset_feature_columns + bundle.regime_feature_columns
        manager_context = torch.tensor(
            latest.iloc[-1][manager_context_cols].to_numpy(dtype=np.float32)[None, :],
            dtype=torch.float32,
        )
        account_context = torch.tensor(
            encode_account_state(AccountState(), PropRuleEngine(config.prop))[None, :],
            dtype=torch.float32,
        )
        output = model(asset_sequences, cross_seq, regime_seq, manager_context, account_context=account_context)

    manager_trade = float(output.manager_trade_probability.item())
    dual_trade = float(output.manager_dual_probability.item()) >= 0.5
    context_score = float(output.manager_context_score.item())
    calibrated = apply_calibration(output.expert_setup_logits, calibration_artifact)[0].cpu().numpy()
    directions = output.directions[0].cpu().numpy()

    chosen_assets: list[str] = []
    chosen_experts: dict[str, str] = {}
    chosen_directions: dict[str, int] = {}
    chosen_probs: dict[str, float] = {}

    if manager_trade >= config.backtest.min_trade_probability and context_score >= config.backtest.min_context_score:
        for asset_idx, asset in enumerate(("US100", "US500")):
            routed_scores = routed_expert_scores(
                calibrated[asset_idx],
                output.manager_gate_weights[0, asset_idx].cpu().numpy(),
                output.expected_returns[0, asset_idx].cpu().numpy(),
                output.expert_confidence[0, asset_idx].cpu().numpy(),
                config.backtest,
            )
            expert_idx = int(np.argmax(routed_scores))
            expert_name = config.model.setup_names[expert_idx]
            expert_prob = float(calibrated[asset_idx, expert_idx])
            if expert_prob >= expert_trade_threshold(config.backtest, expert_name):
                chosen_assets.append(asset)
                chosen_experts[asset] = expert_name
                chosen_directions[asset] = int(np.sign(directions[asset_idx, expert_idx]) or 1)
                chosen_probs[asset] = expert_prob

    return LiveTradeDecision(
        timestamp=str(latest.iloc[-1]["timestamp"]),
        trade=bool(chosen_assets),
        dual_trade=dual_trade and len(chosen_assets) == 2,
        assets=chosen_assets,
        experts=chosen_experts,
        directions=chosen_directions,
        probabilities=chosen_probs,
        context_score=context_score,
    )


def run_live_inference(config: AppConfig) -> dict[str, Any]:
    decision = infer_latest_decision(config)
    return asdict(decision)
