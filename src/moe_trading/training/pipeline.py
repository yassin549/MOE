"""Training orchestration and walk-forward evaluation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from moe_trading.config import AppConfig
from moe_trading.data.dataset import MultiAssetSequenceDataset, collate_sequence_samples
from moe_trading.data.scaling import FeatureScaler
from moe_trading.data.splitting import TimeSplit, generate_walk_forward_splits, make_time_split
from moe_trading.evaluation.metrics import binary_classification_metrics
from moe_trading.experiments.tracker import ExperimentTracker
from moe_trading.labels.audit import generate_phase1_audit_artifacts
from moe_trading.models.moe import MultiAssetMoE, save_model
from moe_trading.pipeline import build_research_frame
from moe_trading.policy.decision import ACCOUNT_FEATURE_NAMES, encode_account_state
from moe_trading.account.rules import PropRuleEngine
from moe_trading.account.state import AccountState
from moe_trading.training.losses import MultiTaskMoELoss
from moe_trading.training.account_context import build_account_context_array
from moe_trading.utils.io import save_json
from moe_trading.utils.reproducibility import set_global_seed
from moe_trading.utils.calibration import calibration_error, fit_platt_scaler, save_calibration_artifact


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "asset_sequences": {k: v.to(device) for k, v in batch["asset_sequences"].items()},
        "cross_sequence": batch["cross_sequence"].to(device),
        "regime_sequence": batch["regime_sequence"].to(device),
        "manager_context": batch["manager_context"].to(device),
        "account_context": batch["account_context"].to(device),
        "expert_valids": batch["expert_valids"].to(device),
        "expert_labels": batch["expert_labels"].to(device),
        "directions": batch["directions"].to(device),
        "manager_labels": batch["manager_labels"].to(device),
        "asset_manager_labels": batch["asset_manager_labels"].to(device),
        "gate_supervision_mask": batch["gate_supervision_mask"].to(device),
        "gate_targets": batch["gate_targets"].to(device),
        "returns": batch["returns"].to(device),
        "timestamps": batch["timestamps"],
    }


def _run_epoch(
    model: MultiAssetMoE,
    loader: DataLoader,
    criterion: MultiTaskMoELoss,
    device: torch.device,
    optimizer: AdamW | None = None,
    grad_clip_norm: float = 1.0,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    metrics_log: list[dict[str, float]] = []
    trade_probabilities: list[np.ndarray] = []
    trade_targets: list[np.ndarray] = []
    direction_predictions: list[np.ndarray] = []
    direction_targets: list[np.ndarray] = []
    gate_usage: list[np.ndarray] = []

    for batch in loader:
        batch = _to_device(batch, device)
        output = model(
            asset_sequences=batch["asset_sequences"],
            cross_sequence=batch["cross_sequence"],
            regime_sequence=batch["regime_sequence"],
            manager_context=batch["manager_context"],
            account_context=batch["account_context"],
        )
        loss, metric_dict = criterion(
            output,
            batch["expert_valids"],
            batch["expert_labels"],
            batch["directions"],
            batch["manager_labels"],
            batch["asset_manager_labels"],
            batch["gate_supervision_mask"],
            batch["gate_targets"],
            batch["returns"],
        )

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        metrics_log.append(metric_dict)
        trade_probabilities.append(output.manager_trade_probability.detach().cpu().numpy().reshape(-1))
        trade_targets.append(batch["manager_labels"][:, 0].detach().cpu().numpy().reshape(-1))
        valid_mask = (batch["expert_valids"] > 0).detach().cpu().numpy().reshape(-1)
        if valid_mask.any():
            predicted_direction = np.sign(output.direction_logits.detach().cpu().numpy().reshape(-1))
            predicted_direction[predicted_direction == 0] = 1
            direction_predictions.append(predicted_direction[valid_mask])
            direction_targets.append(batch["directions"].detach().cpu().numpy().reshape(-1)[valid_mask])
        gate_usage.append(output.manager_gate_weights.detach().cpu().numpy().mean(axis=(0, 1)))

    aggregate = {key: float(np.mean([item[key] for item in metrics_log])) for key in metrics_log[0]}
    y_prob = np.concatenate(trade_probabilities)
    y_true = np.concatenate(trade_targets)
    aggregate.update({f"manager_{k}": v for k, v in binary_classification_metrics(y_true, y_prob).items()})
    if direction_predictions:
        pred = np.concatenate(direction_predictions)
        target = np.concatenate(direction_targets)
        aggregate["direction_accuracy"] = float((pred == target).mean())
        aggregate["direction_pred_long_rate"] = float((pred > 0).mean())
        aggregate["direction_target_long_rate"] = float((target > 0).mean())
    if gate_usage:
        usage = np.stack(gate_usage).mean(axis=0)
        for idx, value in enumerate(usage):
            aggregate[f"gate_expert_{idx}_usage"] = float(value)
    return aggregate, {"manager_prob": y_prob, "manager_true": y_true}


def _make_bundle(bundle, frame):
    return type(bundle)(
        frame=frame,
        asset_feature_columns=bundle.asset_feature_columns,
        cross_asset_feature_columns=bundle.cross_asset_feature_columns,
        regime_feature_columns=bundle.regime_feature_columns,
        label_columns=bundle.label_columns,
    )


def _scale_split(bundle, split: TimeSplit) -> tuple[TimeSplit, FeatureScaler, list[str]]:
    feature_columns = (
        bundle.asset_feature_columns["US100"]
        + bundle.asset_feature_columns["US500"]
        + bundle.cross_asset_feature_columns
        + bundle.regime_feature_columns
    )
    feature_columns = list(dict.fromkeys(feature_columns))
    scaler = FeatureScaler.fit(split.train, feature_columns)
    return (
        TimeSplit(
            train=scaler.transform(split.train, feature_columns),
            validation=scaler.transform(split.validation, feature_columns),
            test=scaler.transform(split.test, feature_columns),
        ),
        scaler,
        feature_columns,
    )


def _active_expert_mask(frame, setup_names: list[str]) -> torch.Tensor:
    active_mask = np.zeros((2, len(setup_names)), dtype=np.float32)
    for asset_idx, asset in enumerate(("us100", "us500")):
        for setup_idx, setup in enumerate(setup_names):
            valid_count = float(frame[f"{asset}_{setup}_valid"].sum())
            target_count = float(frame[f"{asset}_{setup}_target"].sum())
            active_mask[asset_idx, setup_idx] = 1.0 if (valid_count > 0 or target_count > 0) else 0.0
    return torch.tensor(active_mask, dtype=torch.float32)


def _fit_single_split(config: AppConfig, bundle, split: TimeSplit, output_dir: Path, split_name: str) -> dict[str, Any]:
    default_account_context = encode_account_state(AccountState(), PropRuleEngine(config.prop))
    train_account_context = build_account_context_array(split.train, config) if config.train.use_dynamic_account_context else None
    val_account_context = build_account_context_array(split.validation, config) if config.train.use_dynamic_account_context else None
    test_account_context = build_account_context_array(split.test, config) if config.train.use_dynamic_account_context else None
    split, scaler, feature_columns = _scale_split(bundle, split)
    train_dataset = MultiAssetSequenceDataset(
        _make_bundle(bundle, split.train),
        config.data.sequence_length,
        config.model.setup_names,
        default_account_context,
        account_context_array=train_account_context,
    )
    val_dataset = MultiAssetSequenceDataset(
        _make_bundle(bundle, split.validation),
        config.data.sequence_length,
        config.model.setup_names,
        default_account_context,
        account_context_array=val_account_context,
    )
    test_dataset = MultiAssetSequenceDataset(
        _make_bundle(bundle, split.test),
        config.data.sequence_length,
        config.model.setup_names,
        default_account_context,
        account_context_array=test_account_context,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_sequence_samples)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_sequence_samples)
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_sequence_samples)

    asset_input_dim = len(bundle.asset_feature_columns["US100"])
    cross_input_dim = len(bundle.cross_asset_feature_columns)
    regime_input_dim = len(bundle.regime_feature_columns)
    manager_context_dim = cross_input_dim + regime_input_dim + len(ACCOUNT_FEATURE_NAMES)
    model = MultiAssetMoE(asset_input_dim, cross_input_dim, regime_input_dim, manager_context_dim, config.model)

    device = torch.device(config.train.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    criterion = MultiTaskMoELoss(config.train, config.model.setup_names)
    criterion.set_active_expert_mask(_active_expert_mask(split.train, config.model.setup_names))

    best_val = float("inf")
    best_state = None
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in range(config.train.max_epochs):
        train_metrics, _ = _run_epoch(model, train_loader, criterion, device, optimizer, config.train.gradient_clip_norm)
        val_metrics, _ = _run_epoch(model, val_loader, criterion, device)
        epoch_metrics = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(epoch_metrics)

        selection_score = _selection_objective(val_metrics)
        if selection_score < best_val:
            best_val = selection_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.train.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    calibration_artifact = _fit_calibration_artifact(model, val_loader, device, config)
    manager_calibration = _fit_manager_calibration_and_threshold(model, val_loader, device, config)
    calibration_artifact["manager"] = manager_calibration
    test_metrics, _ = _run_epoch(model, test_loader, criterion, device)
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    model_path = split_dir / "model.pt"
    save_model(model, str(model_path))
    scaler_path = split_dir / "scaler.json"
    save_json({"feature_columns": feature_columns, **scaler.to_dict()}, scaler_path)
    calibration_path = split_dir / "calibration.json"
    save_calibration_artifact(calibration_artifact, calibration_path)
    tracker = ExperimentTracker(split_dir)
    tracker.log_metrics("scaler", {"feature_columns": feature_columns, **scaler.to_dict()})
    return {
        "history": history,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "calibration_path": str(calibration_path),
        "selection_score": best_val,
    }


def _selection_objective(metrics: dict[str, float]) -> float:
    brier = float(metrics.get("manager_brier", 1.0))
    f1 = float(metrics.get("manager_f1", 0.0))
    gate_usage = [value for key, value in metrics.items() if key.startswith("gate_expert_")]
    concentration = max(gate_usage) if gate_usage else 1.0
    collapse_penalty = max(concentration - 0.55, 0.0)
    return brier + (1.0 - f1) + collapse_penalty


def _calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    if y_true.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (y_prob >= edges[idx]) & (y_prob <= edges[idx + 1])
        else:
            mask = (y_prob >= edges[idx]) & (y_prob < edges[idx + 1])
        if not mask.any():
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(confidence - accuracy)
    return float(ece)


def _fit_manager_calibration_from_arrays(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    realized_returns: np.ndarray,
    config: AppConfig,
) -> tuple[dict[str, float], float, dict[str, float]]:
    y_prob_t = torch.tensor(y_prob, dtype=torch.float32)
    y_true_t = torch.tensor(y_true, dtype=torch.float32)
    scale, bias = fit_platt_scaler(y_prob_t.logit(eps=1e-6), y_true_t, min_samples=config.train.calibration_min_samples)
    calibrated_prob = 1.0 / (1.0 + np.exp(-((y_prob_t.numpy() * 0 + y_prob_t.logit(eps=1e-6).numpy()) * scale + bias)))

    candidate_thresholds = np.linspace(0.05, 0.95, 37)
    best_threshold = float(config.backtest.min_trade_probability)
    best_expectancy = float("-inf")
    for threshold in candidate_thresholds:
        routed = calibrated_prob >= threshold
        expectancy = float(realized_returns[routed].mean()) if routed.any() else float("-inf")
        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_threshold = float(threshold)

    def _metric_block(probs: np.ndarray) -> dict[str, float]:
        classification = binary_classification_metrics(y_true, probs, threshold=best_threshold)
        routed = probs >= best_threshold
        return {
            "brier": float(classification["brier"]),
            "calibration_error": _calibration_error(y_true, probs),
            "precision_at_threshold": float(classification["precision"]),
            "routed_expectancy": float(realized_returns[routed].mean()) if routed.any() else 0.0,
            "route_rate": float(routed.mean()) if routed.size else 0.0,
        }

    return (
        {"scale": float(scale), "bias": float(bias)},
        best_threshold,
        {
            "pre": _metric_block(y_prob),
            "post": _metric_block(calibrated_prob),
            "selection_post_cost_expectancy": best_expectancy if np.isfinite(best_expectancy) else 0.0,
        },
    )


def _fit_calibration_artifact(model: MultiAssetMoE, loader: DataLoader, device: torch.device, config: AppConfig) -> dict[str, object]:
    logits_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    manager_prob_batches: list[np.ndarray] = []
    manager_target_batches: list[np.ndarray] = []
    manager_return_batches: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = _to_device(batch, device)
            output = model(
                asset_sequences=batch["asset_sequences"],
                cross_sequence=batch["cross_sequence"],
                regime_sequence=batch["regime_sequence"],
                manager_context=batch["manager_context"],
                account_context=batch["account_context"],
            )
            logits_batches.append(output.expert_setup_logits.detach().cpu())
            target_batches.append(batch["expert_labels"].detach().cpu())
            manager_prob_batches.append(output.manager_trade_probability.detach().cpu().numpy().reshape(-1))
            manager_target_batches.append(batch["manager_labels"][:, 0].detach().cpu().numpy().reshape(-1))
            manager_return_batches.append(batch["returns"].amax(dim=(1, 2)).detach().cpu().numpy().reshape(-1))
    if not logits_batches:
        num_assets = 2
        num_experts = len(config.model.setup_names)
        return {
            "method": config.train.calibration_method,
            "asset_names": ["US100", "US500"],
            "setup_names": list(config.model.setup_names),
            "scales": [[1.0] * num_experts for _ in range(num_assets)],
            "biases": [[0.0] * num_experts for _ in range(num_assets)],
        }

    logits = torch.cat(logits_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    scales: list[list[float]] = []
    biases: list[list[float]] = []
    for asset_idx in range(logits.size(1)):
        asset_scales: list[float] = []
        asset_biases: list[float] = []
        for expert_idx in range(logits.size(2)):
            scale, bias = fit_platt_scaler(
                logits[:, asset_idx, expert_idx],
                targets[:, asset_idx, expert_idx],
                min_samples=config.train.calibration_min_samples,
            )
            asset_scales.append(scale)
            asset_biases.append(bias)
        scales.append(asset_scales)
        biases.append(asset_biases)
    manager_calibration, threshold, manager_metrics = _fit_manager_calibration_from_arrays(
        np.concatenate(manager_prob_batches) if manager_prob_batches else np.array([], dtype=np.float32),
        np.concatenate(manager_target_batches) if manager_target_batches else np.array([], dtype=np.float32),
        np.concatenate(manager_return_batches) if manager_return_batches else np.array([], dtype=np.float32),
        config,
    )
    return {
        "method": config.train.calibration_method,
        "asset_names": ["US100", "US500"],
        "setup_names": list(config.model.setup_names),
        "scales": scales,
        "biases": biases,
        "manager_calibration": manager_calibration,
        "manager_routing_threshold": threshold,
        "manager_calibration_metrics": manager_metrics,
    }


def _fit_manager_calibration_and_threshold(
    model: MultiAssetMoE,
    loader: DataLoader,
    device: torch.device,
    config: AppConfig,
) -> dict[str, object]:
    logits_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    return_batches: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = _to_device(batch, device)
            output = model(
                asset_sequences=batch["asset_sequences"],
                cross_sequence=batch["cross_sequence"],
                regime_sequence=batch["regime_sequence"],
                manager_context=batch["manager_context"],
                account_context=batch["account_context"],
            )
            logits_batches.append(output.manager_trade_logits.detach().cpu().reshape(-1))
            target_batches.append(batch["manager_labels"][:, 0].detach().cpu().reshape(-1))
            # Collapse per-asset/per-expert returns to one realized-return proxy per sample
            # so threshold selection is aligned with one manager trade probability per sample.
            return_batches.append(batch["returns"].amax(dim=(1, 2)).detach().cpu().reshape(-1))
    if not logits_batches:
        return {"method": config.train.calibration_method, "scale": 1.0, "bias": 0.0, "threshold": config.backtest.min_trade_probability}

    logits = torch.cat(logits_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    realized_returns = torch.cat(return_batches, dim=0)
    raw_prob = torch.sigmoid(logits)
    scale, bias = fit_platt_scaler(logits, targets, min_samples=config.train.calibration_min_samples)
    cal_prob = torch.sigmoid((logits * scale) + bias)

    threshold_grid = torch.linspace(0.05, 0.95, steps=37)
    cost = (config.backtest.spread_bps + config.backtest.slippage_bps + config.backtest.commission_bps) / 10_000.0
    best_threshold = float(config.backtest.min_trade_probability)
    best_expectancy = float("-inf")
    best_precision = 0.0
    for threshold in threshold_grid:
        mask = cal_prob >= threshold
        if not torch.any(mask):
            continue
        precision = float(targets[mask].mean().item())
        expectancy = float((realized_returns[mask] - cost).mean().item())
        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_threshold = float(threshold.item())
            best_precision = precision

    def _metrics(prob: torch.Tensor, threshold: float) -> dict[str, float]:
        mask = prob >= threshold
        precision = float(targets[mask].mean().item()) if torch.any(mask) else 0.0
        expectancy = float((realized_returns[mask] - cost).mean().item()) if torch.any(mask) else 0.0
        brier = float(torch.mean((prob - targets) ** 2).item())
        return {
            "brier": brier,
            "calibration_error": calibration_error(targets, prob),
            "precision_at_threshold": precision,
            "routed_expectancy_post_cost": expectancy,
        }

    return {
        "method": config.train.calibration_method,
        "scale": float(scale),
        "bias": float(bias),
        "threshold": best_threshold,
        "selection_objective": "max_routed_expectancy_post_cost",
        "metrics_pre": _metrics(raw_prob, float(config.backtest.min_trade_probability)),
        "metrics_post": _metrics(cal_prob, best_threshold),
        "selected_precision": best_precision,
        "selected_expectancy_post_cost": best_expectancy,
    }


def run_training_pipeline(config: AppConfig) -> dict[str, Any]:
    """Run the end-to-end training pipeline on the configured data."""
    set_global_seed(config.train.seed)
    bundle = build_research_frame(config)
    tracker = ExperimentTracker(config.experiment.output_dir)
    tracker.log_config(config)
    phase1_audit = generate_phase1_audit_artifacts(bundle.frame, config, Path(config.experiment.output_dir) / "phase1_audit")
    tracker.log_metrics("phase1_audit_summary", phase1_audit)

    if config.train.use_walk_forward:
        splits = generate_walk_forward_splits(
            bundle.frame,
            train_size=config.train.walk_forward_train_bars,
            validation_size=config.train.walk_forward_validation_bars,
            test_size=config.train.walk_forward_test_bars,
            step_size=config.train.walk_forward_step_bars,
            embargo_bars=config.data.embargo_bars,
        )
        if not splits:
            raise ValueError("No walk-forward splits could be created with the configured sizes.")
    else:
        splits = [make_time_split(bundle.frame, config.data.validation_ratio, config.data.test_ratio, config.data.embargo_bars)]

    split_results = []
    for split_idx, split in enumerate(splits, start=1):
        split_results.append(_fit_single_split(config, bundle, split, Path(config.experiment.output_dir), f"split_{split_idx:02d}"))

    summary = {
        "config": asdict(config),
        "num_splits": len(split_results),
        "split_test_metrics": [result["test_metrics"] for result in split_results],
        "model_paths": [result["model_path"] for result in split_results],
        "scaler_paths": [result["scaler_path"] for result in split_results],
        "calibration_paths": [result["calibration_path"] for result in split_results],
        "selection_scores": [result["selection_score"] for result in split_results],
        "phase1_audit": phase1_audit,
    }
    tracker.log_metrics("training_summary", summary)
    return summary
