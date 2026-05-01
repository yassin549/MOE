"""Realtime, live-like backtest simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from moe_trading.account.rules import PropRuleEngine
from moe_trading.account.state import AccountState
from moe_trading.config import AppConfig
from moe_trading.data.dataset import MultiAssetSequenceDataset, collate_sequence_samples
from moe_trading.data.scaling import FeatureScaler
from moe_trading.data.schemas import MultiAssetFrame
from moe_trading.evaluation.metrics import trade_metrics
from moe_trading.models.moe import load_model
from moe_trading.pipeline import build_feature_bundle
from moe_trading.policy.decision import ACCOUNT_FEATURE_NAMES, encode_account_state, expert_trade_threshold, routed_expert_scores
from moe_trading.utils.calibration import apply_calibration, load_calibration_artifact
from moe_trading.utils.checkpoints import resolve_model_checkpoint


@dataclass(slots=True)
class ReplayConfig:
    sequence_length: int
    max_open_positions: int
    per_trade_risk_fraction: float
    spread_bps: float
    slippage_bps: float
    commission_bps: float
    max_holding_bars: int
    allow_long: bool = True
    allow_short: bool = True
    one_trade_per_asset: bool = True


@dataclass(slots=True)
class CandleBatch:
    timestamp_ns: np.ndarray
    timestamp_str: np.ndarray
    us100_open: np.ndarray
    us100_high: np.ndarray
    us100_low: np.ndarray
    us100_close: np.ndarray
    us500_open: np.ndarray
    us500_high: np.ndarray
    us500_low: np.ndarray
    us500_close: np.ndarray


@dataclass(slots=True)
class ModelSignal:
    asset: str
    direction: int
    score: float
    stop_price: float
    target_price: float
    size_fraction: float
    expert: str
    probability: float


@dataclass(slots=True)
class FillEvent:
    timestamp_ns: int
    asset: str
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    reason: str
    opened_at: str
    closed_at: str
    expert: str
    probability: float
    score: float
    risk_fraction: float


@dataclass(slots=True)
class PortfolioState:
    balance: float
    equity: float
    open_positions: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    current_open_risk: float = 0.0


@dataclass(slots=True)
class SimulationResult:
    final_balance: float
    final_equity: float
    equity_curve: np.ndarray
    fills: list[FillEvent] = field(default_factory=list)
    trades_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PendingOrder:
    asset: str
    direction: int
    score: float
    stop_price: float
    target_price: float
    size_fraction: float
    expert: str
    probability: float
    created_index: int


@dataclass(slots=True)
class OpenPosition:
    asset: str
    direction: int
    quantity: float
    entry_price: float
    stop_price: float
    target_price: float
    risk_fraction: float
    size_fraction: float
    opened_index: int
    opened_at: str
    expert: str
    probability: float
    score: float


class ModelAdapter(Protocol):
    """Live-like contract for stepwise model inference."""

    def on_bar(self, bar_index: int, portfolio: PortfolioState) -> list[ModelSignal]:
        ...


class MoELiveModelAdapter:
    """Stepwise adapter that calls the MoE exactly once per replay bar."""

    def __init__(self, config: AppConfig, bundle: MultiAssetFrame, model_path: str | Path, scaler_path: str | Path) -> None:
        self.config = config
        self.frame = bundle.frame.reset_index(drop=True)
        scaler_payload = json.loads(Path(scaler_path).read_text(encoding="utf-8"))
        scaler = FeatureScaler.from_dict({"means": scaler_payload["means"], "stds": scaler_payload["stds"]})
        feature_columns = scaler_payload.get("feature_columns") or self._feature_columns(bundle)
        scaled_frame = scaler.transform(self.frame, feature_columns)
        scaled_bundle = MultiAssetFrame(
            frame=scaled_frame,
            asset_feature_columns=bundle.asset_feature_columns,
            cross_asset_feature_columns=bundle.cross_asset_feature_columns,
            regime_feature_columns=bundle.regime_feature_columns,
            label_columns=bundle.label_columns,
        )
        self.asset_columns = bundle.asset_feature_columns
        self.cross_columns = bundle.cross_asset_feature_columns
        self.regime_columns = bundle.regime_feature_columns
        manager_context_cols = self.cross_columns + self.regime_columns
        self.close_arrays = {
            "US100": self.frame["us100_close"].to_numpy(dtype=np.float64, copy=True),
            "US500": self.frame["us500_close"].to_numpy(dtype=np.float64, copy=True),
        }
        self.atr_arrays = {
            "US100": self.frame["us100_atr_15"].to_numpy(dtype=np.float64, copy=True),
            "US500": self.frame["us500_atr_15"].to_numpy(dtype=np.float64, copy=True),
        }

        self.asset_tensors = {
            asset: torch.from_numpy(np.ascontiguousarray(scaled_frame[columns].to_numpy(dtype=np.float32)))
            for asset, columns in self.asset_columns.items()
        }
        self.cross_tensor = torch.from_numpy(
            np.ascontiguousarray(scaled_frame[self.cross_columns].to_numpy(dtype=np.float32))
        )
        self.regime_tensor = torch.from_numpy(
            np.ascontiguousarray(scaled_frame[self.regime_columns].to_numpy(dtype=np.float32))
        )
        self.manager_context_tensor = torch.from_numpy(
            np.ascontiguousarray(scaled_frame[manager_context_cols].to_numpy(dtype=np.float32))
        )
        self.rule_engine = PropRuleEngine(config.prop)
        self.calibration_artifact = load_calibration_artifact(Path(model_path).parent / "calibration.json")

        asset_input_dim = len(bundle.asset_feature_columns["US100"])
        cross_input_dim = len(bundle.cross_asset_feature_columns)
        regime_input_dim = len(bundle.regime_feature_columns)
        manager_context_dim = cross_input_dim + regime_input_dim + len(ACCOUNT_FEATURE_NAMES)
        self.model = load_model(str(model_path), asset_input_dim, cross_input_dim, regime_input_dim, manager_context_dim, config.model)
        self.model.eval()
        self.sequence_offset = config.data.sequence_length - 1
        self.precompute_seconds = 0.0
        self._precompute_expert_cache(scaled_bundle)

    @staticmethod
    def _feature_columns(bundle) -> list[str]:
        return list(
            dict.fromkeys(
                bundle.asset_feature_columns["US100"]
                + bundle.asset_feature_columns["US500"]
                + bundle.cross_asset_feature_columns
                + bundle.regime_feature_columns
            )
        )

    def _account_context(self, portfolio: PortfolioState) -> torch.Tensor:
        state = AccountState(
            current_equity=float(portfolio.equity),
            current_balance=float(portfolio.balance),
            current_open_risk=float(portfolio.current_open_risk),
            open_unrealized_pnl=float(portfolio.unrealized_pnl),
            realized_pnl_overall=float(portfolio.realized_pnl),
            daily_peak_equity=max(float(portfolio.balance), float(portfolio.equity)),
            overall_peak_equity=max(float(portfolio.balance), float(portfolio.equity)),
        )
        encoded = encode_account_state(state, self.rule_engine)
        return torch.from_numpy(encoded[None, :])

    def _precompute_expert_cache(self, bundle: MultiAssetFrame) -> None:
        start_time = time.perf_counter()
        dataset = MultiAssetSequenceDataset(bundle, self.config.data.sequence_length, self.config.model.setup_names)
        batch_size = max(self.config.train.batch_size, 256)
        manager_features: list[np.ndarray] = []
        setup_logits: list[np.ndarray] = []
        confidence: list[np.ndarray] = []
        expected_returns: list[np.ndarray] = []
        directions: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(dataset), batch_size):
                samples = [dataset[idx] for idx in range(start, min(start + batch_size, len(dataset)))]
                batch = collate_sequence_samples(samples)
                expert_outputs = self.model.infer_expert_outputs(
                    asset_sequences={asset: tensor for asset, tensor in batch["asset_sequences"].items()},
                    cross_sequence=batch["cross_sequence"],
                    regime_sequence=batch["regime_sequence"],
                )
                manager_features.append(expert_outputs["manager_expert_features"].cpu().numpy())
                setup_logits.append(expert_outputs["expert_setup_logits"].cpu().numpy())
                confidence.append(expert_outputs["expert_confidence"].cpu().numpy())
                expected_returns.append(expert_outputs["expected_returns"].cpu().numpy())
                directions.append(expert_outputs["directions"].cpu().numpy())

        n = len(self.frame)
        num_assets = 2
        num_experts = len(self.config.model.setup_names)
        manager_feature_dim = manager_features[0].shape[1] if manager_features else num_assets * num_experts * 5
        self.cached_manager_features = np.zeros((n, manager_feature_dim), dtype=np.float32)
        self.cached_setup_logits = np.zeros((n, num_assets, num_experts), dtype=np.float32)
        self.cached_confidence = np.zeros((n, num_assets, num_experts), dtype=np.float32)
        self.cached_expected_returns = np.zeros((n, num_assets, num_experts), dtype=np.float32)
        self.cached_directions = np.zeros((n, num_assets, num_experts), dtype=np.float32)
        if manager_features:
            features_concat = np.concatenate(manager_features, axis=0)
            setup_logits_concat = np.concatenate(setup_logits, axis=0)
            confidence_concat = np.concatenate(confidence, axis=0)
            expected_returns_concat = np.concatenate(expected_returns, axis=0)
            directions_concat = np.concatenate(directions, axis=0)
            self.cached_manager_features[self.sequence_offset :] = features_concat
            self.cached_setup_logits[self.sequence_offset :] = setup_logits_concat
            self.cached_confidence[self.sequence_offset :] = confidence_concat
            self.cached_expected_returns[self.sequence_offset :] = expected_returns_concat
            self.cached_directions[self.sequence_offset :] = directions_concat
        self.precompute_seconds = time.perf_counter() - start_time

    def on_bar(self, bar_index: int, portfolio: PortfolioState) -> list[ModelSignal]:
        seq_len = self.config.data.sequence_length
        if bar_index < seq_len - 1:
            return []
        with torch.inference_mode():
            manager_context = self.manager_context_tensor[bar_index : bar_index + 1]
            account_context = self._account_context(portfolio)
            manager_features = torch.from_numpy(self.cached_manager_features[bar_index : bar_index + 1])
            manager_output = self.model.forward_manager_only(
                manager_features,
                manager_context,
                account_context=account_context,
            )

        manager_trade = float(manager_output["manager_trade_probability"].item())
        context_score = float(manager_output["manager_context_score"].item())
        if manager_trade < self.config.backtest.min_trade_probability or context_score < self.config.backtest.min_context_score:
            return []

        calibrated = apply_calibration(
            torch.from_numpy(self.cached_setup_logits[bar_index : bar_index + 1]),
            self.calibration_artifact,
        )[0].cpu().numpy()
        confidence = self.cached_confidence[bar_index]
        expected_returns = self.cached_expected_returns[bar_index]
        directions = self.cached_directions[bar_index]
        gate_weights = manager_output["manager_gate_weights"][0].cpu().numpy()

        signals: list[ModelSignal] = []
        for asset_idx, asset in enumerate(("US100", "US500")):
            routed_scores = routed_expert_scores(
                calibrated[asset_idx],
                gate_weights[asset_idx],
                expected_returns[asset_idx],
                confidence[asset_idx],
                self.config.backtest,
            )
            expert_idx = int(np.argmax(routed_scores))
            expert_name = self.config.model.setup_names[expert_idx]
            expert_prob = float(calibrated[asset_idx, expert_idx])
            if expert_prob < expert_trade_threshold(self.config.backtest, expert_name):
                continue
            direction = int(np.sign(directions[asset_idx, expert_idx]) or 1)
            if direction == 0:
                continue
            entry_price = float(self.close_arrays[asset][bar_index])
            stop_distance = max(float(self.atr_arrays[asset][bar_index]) * self.config.labels.stop_atr_multiple, 1e-6)
            signals.append(
                ModelSignal(
                    asset=asset,
                    direction=direction,
                    score=float(routed_scores[expert_idx]),
                    stop_price=entry_price - (direction * stop_distance),
                    target_price=entry_price + (direction * stop_distance * self.config.labels.target_atr_multiple),
                    size_fraction=float(self.config.backtest.challenge_risk_fraction),
                    expert=expert_name,
                    probability=expert_prob,
                )
            )
        signals.sort(key=lambda item: item.score, reverse=True)
        if not self.config.backtest.allow_dual_asset_trades and signals:
            return signals[:1]
        return signals


class RealtimeBacktestSimulator:
    """Sequential live-like simulator with next-bar entries and intra-bar exits."""

    def __init__(self, config: ReplayConfig) -> None:
        self.config = config

    def run(self, candles: CandleBatch, model: ModelAdapter) -> SimulationResult:
        n = candles.timestamp_ns.shape[0]
        equity_curve = np.empty(n, dtype=np.float64)
        balance = 100_000.0
        portfolio = PortfolioState(balance=balance, equity=balance)
        pending_orders: dict[str, PendingOrder] = {}
        positions: dict[str, OpenPosition] = {}
        fills: list[FillEvent] = []

        spread_rate = self.config.spread_bps / 10000.0
        slippage_rate = self.config.slippage_bps / 10000.0
        commission_rate = self.config.commission_bps / 10000.0
        inference_seconds = 0.0
        loop_start = time.perf_counter()

        for bar_index in range(n):
            for asset in ("US100", "US500"):
                order = pending_orders.pop(asset, None)
                if order is not None and asset not in positions:
                    open_price = float(getattr(candles, f"{asset.lower()}_open")[bar_index])
                    entry_price = self._apply_entry_costs(open_price, order.direction, spread_rate, slippage_rate)
                    notional = portfolio.balance * order.size_fraction
                    if notional > 0.0:
                        commission = notional * commission_rate
                        portfolio.balance -= commission
                        quantity = notional / max(entry_price, 1e-9)
                        positions[asset] = OpenPosition(
                            asset=asset,
                            direction=order.direction,
                            quantity=quantity,
                            entry_price=entry_price,
                            stop_price=order.stop_price,
                            target_price=order.target_price,
                            size_fraction=order.size_fraction,
                            opened_index=bar_index,
                            opened_at=candles.timestamp_str[bar_index],
                            expert=order.expert,
                            probability=order.probability,
                            score=order.score,
                            risk_fraction=order.size_fraction,
                        )

            for asset in list(positions):
                position = positions[asset]
                high = float(getattr(candles, f"{asset.lower()}_high")[bar_index])
                low = float(getattr(candles, f"{asset.lower()}_low")[bar_index])
                close = float(getattr(candles, f"{asset.lower()}_close")[bar_index])
                exit_reason = ""
                raw_exit_price = close

                if position.direction > 0:
                    if low <= position.stop_price:
                        raw_exit_price = position.stop_price
                        exit_reason = "stop_hit"
                    elif high >= position.target_price:
                        raw_exit_price = position.target_price
                        exit_reason = "target_hit"
                else:
                    if high >= position.stop_price:
                        raw_exit_price = position.stop_price
                        exit_reason = "stop_hit"
                    elif low <= position.target_price:
                        raw_exit_price = position.target_price
                        exit_reason = "target_hit"

                if not exit_reason and (bar_index - position.opened_index) >= self.config.max_holding_bars:
                    exit_reason = "max_holding_exit"

                if exit_reason:
                    exit_price = self._apply_exit_costs(raw_exit_price, position.direction, spread_rate, slippage_rate)
                    pnl = position.direction * (exit_price - position.entry_price) * position.quantity
                    pnl -= abs(position.quantity * exit_price) * commission_rate
                    portfolio.balance += pnl
                    portfolio.realized_pnl += pnl
                    fills.append(
                        FillEvent(
                            timestamp_ns=int(candles.timestamp_ns[bar_index]),
                            asset=asset,
                            direction=position.direction,
                            entry_price=position.entry_price,
                            exit_price=exit_price,
                            pnl=pnl,
                            reason=exit_reason,
                            opened_at=position.opened_at,
                            closed_at=str(candles.timestamp_str[bar_index]),
                            expert=position.expert,
                            probability=position.probability,
                            score=position.score,
                            risk_fraction=position.size_fraction,
                        )
                    )
                    del positions[asset]

            unrealized = 0.0
            open_risk = 0.0
            for asset, position in positions.items():
                close = float(getattr(candles, f"{asset.lower()}_close")[bar_index])
                unrealized += position.direction * (close - position.entry_price) * position.quantity
                open_risk += position.size_fraction
            portfolio.unrealized_pnl = unrealized
            portfolio.current_open_risk = open_risk
            portfolio.open_positions = len(positions)
            portfolio.equity = portfolio.balance + unrealized
            equity_curve[bar_index] = portfolio.equity

            if bar_index >= n - 1:
                continue

            t0 = time.perf_counter()
            signals = model.on_bar(bar_index, portfolio)
            inference_seconds += time.perf_counter() - t0
            for signal in signals:
                if signal.asset in positions or signal.asset in pending_orders:
                    continue
                if self.config.one_trade_per_asset and signal.asset in positions:
                    continue
                if len(positions) + len(pending_orders) >= self.config.max_open_positions:
                    break
                if signal.direction > 0 and not self.config.allow_long:
                    continue
                if signal.direction < 0 and not self.config.allow_short:
                    continue
                pending_orders[signal.asset] = PendingOrder(
                    asset=signal.asset,
                    direction=signal.direction,
                    score=signal.score,
                    stop_price=signal.stop_price,
                    target_price=signal.target_price,
                    size_fraction=signal.size_fraction,
                    expert=signal.expert,
                    probability=signal.probability,
                    created_index=bar_index,
                )

        final_index = n - 1
        for asset in list(positions):
            position = positions.pop(asset)
            final_close = float(getattr(candles, f"{asset.lower()}_close")[final_index])
            exit_price = self._apply_exit_costs(final_close, position.direction, spread_rate, slippage_rate)
            pnl = position.direction * (exit_price - position.entry_price) * position.quantity
            pnl -= abs(position.quantity * exit_price) * commission_rate
            portfolio.balance += pnl
            portfolio.realized_pnl += pnl
            fills.append(
                FillEvent(
                    timestamp_ns=int(candles.timestamp_ns[final_index]),
                    asset=asset,
                    direction=position.direction,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    reason="end_of_window",
                    opened_at=position.opened_at,
                    closed_at=str(candles.timestamp_str[final_index]),
                    expert=position.expert,
                    probability=position.probability,
                    score=position.score,
                    risk_fraction=position.size_fraction,
                )
            )
        portfolio.equity = portfolio.balance
        equity_curve[final_index] = portfolio.equity

        trades_frame = pd.DataFrame(
            [
                {
                    "timestamp": item.closed_at,
                    "opened_at": item.opened_at,
                    "closed_at": item.closed_at,
                    "asset": item.asset,
                    "expert": item.expert,
                    "direction": item.direction,
                    "probability": item.probability,
                    "score": item.score,
                    "entry_price": item.entry_price,
                    "exit_price": item.exit_price,
                    "realized_pnl": item.pnl,
                    "net_return_r": item.pnl / 100_000.0,
                    "risk_fraction": item.risk_fraction,
                    "exit_reason": item.reason,
                }
                for item in fills
            ]
        )
        summary = trade_metrics(trades_frame) if not trades_frame.empty else trade_metrics(pd.DataFrame())
        performance = {
            "bars": float(n),
            "loop_seconds": time.perf_counter() - loop_start,
            "inference_seconds": inference_seconds,
            "expert_precompute_seconds": float(getattr(model, "precompute_seconds", 0.0)),
            "bars_per_second": float(n / max(time.perf_counter() - loop_start, 1e-9)),
        }
        summary["final_balance"] = float(portfolio.balance)
        return SimulationResult(
            final_balance=float(portfolio.balance),
            final_equity=float(portfolio.equity),
            equity_curve=equity_curve,
            fills=fills,
            trades_frame=trades_frame,
            summary=summary,
            performance=performance,
        )

    @staticmethod
    def _apply_entry_costs(price: float, direction: int, spread_rate: float, slippage_rate: float) -> float:
        total = spread_rate + slippage_rate
        return price * (1.0 + total) if direction > 0 else price * (1.0 - total)

    @staticmethod
    def _apply_exit_costs(price: float, direction: int, spread_rate: float, slippage_rate: float) -> float:
        total = spread_rate + slippage_rate
        return price * (1.0 - total) if direction > 0 else price * (1.0 + total)


def load_realtime_candles(frame: pd.DataFrame) -> CandleBatch:
    return CandleBatch(
        timestamp_ns=frame["timestamp"].astype("int64").to_numpy(copy=True),
        timestamp_str=frame["timestamp"].astype(str).to_numpy(copy=True),
        us100_open=frame["us100_open"].to_numpy(dtype=np.float64, copy=True),
        us100_high=frame["us100_high"].to_numpy(dtype=np.float64, copy=True),
        us100_low=frame["us100_low"].to_numpy(dtype=np.float64, copy=True),
        us100_close=frame["us100_close"].to_numpy(dtype=np.float64, copy=True),
        us500_open=frame["us500_open"].to_numpy(dtype=np.float64, copy=True),
        us500_high=frame["us500_high"].to_numpy(dtype=np.float64, copy=True),
        us500_low=frame["us500_low"].to_numpy(dtype=np.float64, copy=True),
        us500_close=frame["us500_close"].to_numpy(dtype=np.float64, copy=True),
    )


def build_realtime_components(
    config: AppConfig,
    experiment_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> tuple[CandleBatch, MoELiveModelAdapter, ReplayConfig]:
    bundle = build_feature_bundle(config)
    frame = bundle.frame.reset_index(drop=True)
    resolved_model_path = Path(model_path) if model_path is not None else resolve_model_checkpoint(experiment_dir or config.experiment.output_dir)
    resolved_scaler_path = Path(scaler_path) if scaler_path is not None else resolved_model_path.parent / "scaler.json"
    candles = load_realtime_candles(frame)
    adapter = MoELiveModelAdapter(config, bundle, resolved_model_path, resolved_scaler_path)
    replay = ReplayConfig(
        sequence_length=config.data.sequence_length,
        max_open_positions=2 if config.backtest.allow_dual_asset_trades else 1,
        per_trade_risk_fraction=config.backtest.challenge_risk_fraction,
        spread_bps=config.backtest.spread_bps,
        slippage_bps=config.backtest.slippage_bps,
        commission_bps=config.backtest.commission_bps,
        max_holding_bars=config.labels.max_holding_bars,
        allow_long=True,
        allow_short=True,
        one_trade_per_asset=config.backtest.one_trade_per_asset,
    )
    return candles, adapter, replay


def run_realtime_backtest(
    config: AppConfig,
    experiment_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> SimulationResult:
    candles, adapter, replay = build_realtime_components(
        config,
        experiment_dir=experiment_dir,
        model_path=model_path,
        scaler_path=scaler_path,
    )
    simulator = RealtimeBacktestSimulator(replay)
    return simulator.run(candles, adapter)


__all__ = [
    "CandleBatch",
    "FillEvent",
    "ModelAdapter",
    "ModelSignal",
    "MoELiveModelAdapter",
    "PortfolioState",
    "RealtimeBacktestSimulator",
    "ReplayConfig",
    "SimulationResult",
    "build_realtime_components",
    "load_realtime_candles",
    "run_realtime_backtest",
]
