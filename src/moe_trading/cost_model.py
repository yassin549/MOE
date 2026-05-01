"""Versioned cost model helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from moe_trading.config import AppConfig, BacktestConfig


@dataclass(frozen=True, slots=True)
class CostModelSpec:
    spread_bps: float
    slippage_bps: float
    commission_bps: float
    one_trade_per_asset: bool
    allow_dual_asset_trades: bool
    min_expected_return_r: float

    @classmethod
    def from_backtest_config(cls, config: BacktestConfig) -> "CostModelSpec":
        return cls(
            spread_bps=float(config.spread_bps),
            slippage_bps=float(config.slippage_bps),
            commission_bps=float(config.commission_bps),
            one_trade_per_asset=bool(config.one_trade_per_asset),
            allow_dual_asset_trades=bool(config.allow_dual_asset_trades),
            min_expected_return_r=float(config.min_expected_return_r),
        )

    def stable_payload(self) -> dict[str, Any]:
        return asdict(self)


def cost_model_fingerprint(spec: CostModelSpec) -> str:
    encoded = json.dumps(spec.stable_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def cost_model_version(spec: CostModelSpec) -> str:
    return f"cm-{cost_model_fingerprint(spec)}"


def cost_model_metadata(config: AppConfig) -> dict[str, Any]:
    spec = CostModelSpec.from_backtest_config(config.backtest)
    return {
        "cost_model_version": cost_model_version(spec),
        "cost_model_hash": cost_model_fingerprint(spec),
        "cost_model_parameters": spec.stable_payload(),
    }
