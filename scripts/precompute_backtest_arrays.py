import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_trading.config import load_config
from moe_trading.data.dataset import MultiAssetSequenceDataset, collate_sequence_samples
from moe_trading.data.scaling import FeatureScaler
from moe_trading.models.moe import load_model
from moe_trading.pipeline import build_feature_bundle
from moe_trading.policy.decision import ACCOUNT_FEATURE_NAMES, expert_trade_threshold, routed_expert_scores
from moe_trading.utils.calibration import apply_calibration, load_calibration_artifact


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline precompute for the NumPy backtest engine.")
    parser.add_argument("--config", required=True, help="Path to the base config used to rebuild the feature frame.")
    parser.add_argument("--model-path", required=True, help="Checkpoint path.")
    parser.add_argument("--scaler-path", required=True, help="Scaler JSON path for the checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory where asset .npz files will be written.")
    parser.add_argument("--start", default=None, help="Inclusive UTC start timestamp.")
    parser.add_argument("--end", default=None, help="Inclusive UTC end timestamp.")
    parser.add_argument("--days", type=int, default=None, help="If provided, keep only the last N calendar days.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional tail-row cap applied before feature generation.")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size.")
    parser.add_argument(
        "--selection-mode",
        choices=("threshold", "daily_topk"),
        default="daily_topk",
        help="How to turn model candidates into final entry signals.",
    )
    parser.add_argument("--save-diagnostics", action="store_true", help="Write signal funnel diagnostics JSON.")
    return parser


def _filter_frame(frame: pd.DataFrame, start: str | None, end: str | None, days: int | None) -> pd.DataFrame:
    filtered = frame
    if start is not None:
        filtered = filtered.loc[filtered["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        filtered = filtered.loc[filtered["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    if days is not None:
        cutoff = filtered["timestamp"].max() - pd.Timedelta(days=days)
        filtered = filtered.loc[filtered["timestamp"] >= cutoff]
    return filtered.reset_index(drop=True)


def _feature_columns(bundle) -> list[str]:
    return list(
        dict.fromkeys(
            bundle.asset_feature_columns["US100"]
            + bundle.asset_feature_columns["US500"]
            + bundle.cross_asset_feature_columns
            + bundle.regime_feature_columns
        )
    )


def _make_bundle(bundle, frame: pd.DataFrame):
    return type(bundle)(
        frame=frame,
        asset_feature_columns=bundle.asset_feature_columns,
        cross_asset_feature_columns=bundle.cross_asset_feature_columns,
        regime_feature_columns=bundle.regime_feature_columns,
        label_columns=bundle.label_columns,
    )


def _load_arrays(args: argparse.Namespace):
    config = load_config(args.config)
    config.data.max_rows = args.max_rows
    bundle = build_feature_bundle(config)
    filtered = _filter_frame(bundle.frame, args.start, args.end, args.days)
    if filtered.empty:
        raise ValueError("No rows remain after applying the requested date filter.")

    scaler_payload = json.loads(Path(args.scaler_path).read_text(encoding="utf-8"))
    scaler = FeatureScaler.from_dict({"means": scaler_payload["means"], "stds": scaler_payload["stds"]})
    feature_columns = scaler_payload.get("feature_columns") or _feature_columns(bundle)
    raw_frame = filtered.reset_index(drop=True)
    scaled_frame = scaler.transform(raw_frame, feature_columns)
    dataset = MultiAssetSequenceDataset(_make_bundle(bundle, scaled_frame), config.data.sequence_length, config.model.setup_names)

    asset_input_dim = len(bundle.asset_feature_columns["US100"])
    cross_input_dim = len(bundle.cross_asset_feature_columns)
    regime_input_dim = len(bundle.regime_feature_columns)
    manager_context_dim = cross_input_dim + regime_input_dim + len(ACCOUNT_FEATURE_NAMES)
    model = load_model(args.model_path, asset_input_dim, cross_input_dim, regime_input_dim, manager_context_dim, config.model)
    model.eval()
    calibration = load_calibration_artifact(Path(args.model_path).parent / "calibration.json")
    return config, raw_frame, dataset, model, calibration


def _precompute_signals(
    config,
    raw_frame: pd.DataFrame,
    dataset: MultiAssetSequenceDataset,
    model,
    calibration,
    batch_size: int,
    selection_mode: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequence_samples)
    zero_account = np.zeros((1, len(ACCOUNT_FEATURE_NAMES)), dtype=np.float32)

    prices = {
        "US100": raw_frame["us100_close"].to_numpy(dtype=np.float64),
        "US500": raw_frame["us500_close"].to_numpy(dtype=np.float64),
    }
    atr15 = {
        "US100": raw_frame["us100_atr_15"].to_numpy(dtype=np.float64),
        "US500": raw_frame["us500_atr_15"].to_numpy(dtype=np.float64),
    }
    timestamps = raw_frame["timestamp"].astype(str).to_numpy(dtype="U32")

    n = len(raw_frame)
    sequence_offset = config.data.sequence_length - 1
    outputs = {
        "US100": {
            "prices": prices["US100"],
            "signals": np.zeros(n, dtype=np.int8),
            "sl": np.full(n, np.nan, dtype=np.float64),
            "tp": np.full(n, np.nan, dtype=np.float64),
            "position_size": np.zeros(n, dtype=np.float64),
            "timestamps": timestamps,
        },
        "US500": {
            "prices": prices["US500"],
            "signals": np.zeros(n, dtype=np.int8),
            "sl": np.full(n, np.nan, dtype=np.float64),
            "tp": np.full(n, np.nan, dtype=np.float64),
            "position_size": np.zeros(n, dtype=np.float64),
            "timestamps": timestamps,
        },
    }

    asset_names = ("US100", "US500")
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
    setup_names = tuple(config.model.setup_names)
    diagnostics = {
        "rows": int(n),
        "sequence_rows": int(len(dataset)),
        "manager_pass_trade": 0,
        "manager_pass_context": 0,
        "manager_pass_both": 0,
        "asset_candidates_seen": {asset: 0 for asset in asset_names},
        "asset_pass_expert_threshold": {asset: 0 for asset in asset_names},
        "asset_emitted_signals": {asset: 0 for asset in asset_names},
        "expert_wins": {asset: {name: 0 for name in setup_names} for asset in asset_names},
        "expert_threshold_pass": {asset: {name: 0 for name in setup_names} for asset in asset_names},
        "manager_probability": [],
        "context_score": [],
        "routed_score": {asset: [] for asset in asset_names},
        "expert_probability": {asset: [] for asset in asset_names},
        "selected_candidates": {asset: 0 for asset in asset_names},
        "selection_mode": selection_mode,
    }
    candidate_records: list[dict[str, object]] = []

    cursor = 0
    with torch.no_grad():
        for batch in loader:
            expert_outputs = model.infer_expert_outputs(
                asset_sequences={asset: tensor for asset, tensor in batch["asset_sequences"].items()},
                cross_sequence=batch["cross_sequence"],
                regime_sequence=batch["regime_sequence"],
            )
            batch_size_local = expert_outputs["manager_expert_features"].shape[0]
            account_context = torch.from_numpy(np.repeat(zero_account, batch_size_local, axis=0))
            manager_outputs = model.forward_manager_only(
                expert_outputs["manager_expert_features"],
                batch["manager_context"],
                account_context=account_context,
            )

            calibrated = apply_calibration(expert_outputs["expert_setup_logits"], calibration).cpu().numpy()
            confidence = expert_outputs["expert_confidence"].cpu().numpy()
            expected_returns = expert_outputs["expected_returns"].cpu().numpy()
            directions = expert_outputs["directions"].cpu().numpy()
            manager_prob = manager_outputs["manager_trade_probability"].cpu().numpy().reshape(-1)
            context_score = manager_outputs["manager_context_score"].cpu().numpy().reshape(-1)
            gate_weights = manager_outputs["manager_gate_weights"].cpu().numpy()

            for batch_index in range(batch_size_local):
                raw_index = cursor + batch_index + sequence_offset
                manager_trade_ok = manager_prob[batch_index] >= config.backtest.min_trade_probability
                context_ok = context_score[batch_index] >= config.backtest.min_context_score
                if manager_trade_ok:
                    diagnostics["manager_pass_trade"] += 1
                if context_ok:
                    diagnostics["manager_pass_context"] += 1
                if manager_trade_ok and context_ok:
                    diagnostics["manager_pass_both"] += 1
                diagnostics["manager_probability"].append(float(manager_prob[batch_index]))
                diagnostics["context_score"].append(float(context_score[batch_index]))

                for asset_index, asset in enumerate(asset_names):
                    diagnostics["asset_candidates_seen"][asset] += 1
                    routed_scores = routed_expert_scores(
                        calibrated[batch_index, asset_index],
                        gate_weights[batch_index, asset_index],
                        expected_returns[batch_index, asset_index],
                        confidence[batch_index, asset_index],
                        config.backtest,
                    )
                    expert_index = int(np.argmax(routed_scores))
                    expert_name = config.model.setup_names[expert_index]
                    expert_prob = float(calibrated[batch_index, asset_index, expert_index])
                    expert_threshold = expert_trade_threshold(config.backtest, expert_name)
                    diagnostics["expert_wins"][asset][expert_name] += 1
                    diagnostics["routed_score"][asset].append(float(routed_scores[expert_index]))
                    diagnostics["expert_probability"][asset].append(expert_prob)
                    direction = int(np.sign(directions[batch_index, asset_index, expert_index]))
                    if direction == 0:
                        direction = 1

                    if expert_prob >= expert_threshold:
                        diagnostics["asset_pass_expert_threshold"][asset] += 1
                        diagnostics["expert_threshold_pass"][asset][expert_name] += 1

                    if expert_prob >= expert_threshold:
                        stop_distance = max(float(atr15[asset][raw_index]) * config.labels.stop_atr_multiple, 1e-6)
                        entry_price = float(prices[asset][raw_index])
                        candidate_records.append(
                            {
                                "day": str(timestamps[raw_index])[:10],
                                "raw_index": raw_index,
                                "asset": asset,
                                "direction": direction,
                                "entry_price": entry_price,
                                "stop_distance": stop_distance,
                                "manager_trade_ok": manager_trade_ok,
                                "context_ok": context_ok,
                                "manager_prob": float(manager_prob[batch_index]),
                                "context_score": float(context_score[batch_index]),
                                "routed_score": float(routed_scores[expert_index]),
                                "combined_score": float(
                                    manager_prob[batch_index] * context_score[batch_index] * routed_scores[expert_index]
                                ),
                            }
                        )

            cursor += batch_size_local

    if selection_mode == "threshold":
        selected_records = [
            record
            for record in candidate_records
            if bool(record["manager_trade_ok"]) and bool(record["context_ok"])
        ]
    else:
        selected_records = []
        candidates_by_day: dict[str, list[dict[str, object]]] = {}
        for record in candidate_records:
            candidates_by_day.setdefault(str(record["day"]), []).append(record)

        daily_target = max(1, int(config.backtest.target_trades_per_day_max))
        hold_bars = max(1, int(config.labels.max_holding_bars))
        for day in sorted(candidates_by_day):
            chosen_count = 0
            last_asset_index: dict[str, int] = {}
            ranked = sorted(candidates_by_day[day], key=lambda item: float(item["combined_score"]), reverse=True)
            for record in ranked:
                if chosen_count >= daily_target:
                    break
                asset = str(record["asset"])
                raw_index = int(record["raw_index"])
                if asset in last_asset_index and raw_index - last_asset_index[asset] < hold_bars:
                    continue
                selected_records.append(record)
                last_asset_index[asset] = raw_index
                chosen_count += 1

    for record in selected_records:
        asset = str(record["asset"])
        raw_index = int(record["raw_index"])
        direction = int(record["direction"])
        entry_price = float(record["entry_price"])
        stop_distance = float(record["stop_distance"])
        outputs[asset]["signals"][raw_index] = np.int8(direction)
        outputs[asset]["sl"][raw_index] = entry_price - (direction * stop_distance)
        outputs[asset]["tp"][raw_index] = entry_price + (direction * stop_distance * config.labels.target_atr_multiple)
        outputs[asset]["position_size"][raw_index] = float(config.backtest.challenge_risk_fraction)
        diagnostics["asset_emitted_signals"][asset] += 1
        diagnostics["selected_candidates"][asset] += 1

    for key in ("manager_probability", "context_score"):
        values = np.asarray(diagnostics[key], dtype=np.float64)
        diagnostics[key] = {
            "count": int(values.size),
            "mean": float(values.mean()) if values.size else 0.0,
            "p50": float(np.percentile(values, 50)) if values.size else 0.0,
            "p90": float(np.percentile(values, 90)) if values.size else 0.0,
            "p99": float(np.percentile(values, 99)) if values.size else 0.0,
            "max": float(values.max()) if values.size else 0.0,
        }
    for asset in asset_names:
        for bucket in ("routed_score", "expert_probability"):
            values = np.asarray(diagnostics[bucket][asset], dtype=np.float64)
            diagnostics[bucket][asset] = {
                "count": int(values.size),
                "mean": float(values.mean()) if values.size else 0.0,
                "p50": float(np.percentile(values, 50)) if values.size else 0.0,
                "p90": float(np.percentile(values, 90)) if values.size else 0.0,
                "p99": float(np.percentile(values, 99)) if values.size else 0.0,
                "max": float(values.max()) if values.size else 0.0,
            }

    return outputs, diagnostics


if __name__ == "__main__":
    args = _build_parser().parse_args()
    config, raw_frame, dataset, model, calibration = _load_arrays(args)
    payloads, diagnostics = _precompute_signals(
        config,
        raw_frame,
        dataset,
        model,
        calibration,
        batch_size=args.batch_size,
        selection_mode=args.selection_mode,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for asset, payload in payloads.items():
        np.savez_compressed(output_dir / f"{asset.lower()}_backtest_arrays.npz", **payload)
    if args.save_diagnostics:
        (output_dir / "signal_funnel_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(
        "Precompute complete: "
        f"rows={len(raw_frame)}, "
        f"start={raw_frame['timestamp'].iloc[0]}, "
        f"end={raw_frame['timestamp'].iloc[-1]}, "
        f"us100_signals={(payloads['US100']['signals'] != 0).sum()}, "
        f"us500_signals={(payloads['US500']['signals'] != 0).sum()}"
    )
