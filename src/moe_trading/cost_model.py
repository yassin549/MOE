"""Versioned cost-model helpers for backtesting and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from moe_trading.config import AppConfig


@dataclass(frozen=True, slots=True)
class CostModelConfig:
    spread_bps: float
    slippage_bps: float
    commission_bps: float
    one_trade_per_asset: bool
    allow_dual_asset_trades: bool
    min_trade_probability: float
    min_context_score: float


def build_cost_model_config(config: AppConfig) -> CostModelConfig:
    backtest = config.backtest
    return CostModelConfig(
        spread_bps=float(backtest.spread_bps),
        slippage_bps=float(backtest.slippage_bps),
        commission_bps=float(backtest.commission_bps),
        one_trade_per_asset=bool(backtest.one_trade_per_asset),
        allow_dual_asset_trades=bool(backtest.allow_dual_asset_trades),
        min_trade_probability=float(backtest.min_trade_probability),
        min_context_score=float(backtest.min_context_score),
    )


def cost_model_version(config: AppConfig) -> str:
    payload = asdict(build_cost_model_config(config))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def cost_model_stamp(config: AppConfig) -> dict[str, Any]:
    model = build_cost_model_config(config)
    return {
        "cost_model_version": cost_model_version(config),
        "cost_model_params": asdict(model),
    }


def total_round_trip_bps(config: AppConfig) -> float:
    model = build_cost_model_config(config)
    return 2.0 * (model.spread_bps + model.slippage_bps + model.commission_bps)
