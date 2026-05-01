"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    aligned_dir: str = "data/aligned"
    artifact_dir: str = "artifacts"
    us100_file: str = "data/processed/US100_m1_clean.csv"
    us500_file: str = "data/processed/US500_m1_clean.csv"
    timestamp_col: str = "timestamp_utc"
    asset_col: str = "symbol"
    sequence_length: int = 128
    prediction_horizon: int = 30
    bar_minutes: int = 1
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    embargo_bars: int = 60
    max_rows: int | None = None
    use_cache: bool = True
    cache_dir: str = "artifacts/cache/research_frames"


@dataclass(slots=True)
class FeatureConfig:
    volatility_windows: list[int] = field(default_factory=lambda: [5, 15, 30, 60])
    slope_windows: list[int] = field(default_factory=lambda: [5, 15, 30])
    momentum_windows: list[int] = field(default_factory=lambda: [3, 5, 10, 15, 20, 30])
    swing_windows: list[int] = field(default_factory=lambda: [10, 20, 50])
    compression_windows: list[int] = field(default_factory=lambda: [10, 20, 40])
    correlation_windows: list[int] = field(default_factory=lambda: [15, 30, 60])
    zscore_windows: list[int] = field(default_factory=lambda: [20, 60])
    session_open_hours_utc: list[int] = field(default_factory=lambda: [7, 8, 13, 14])
    use_volume_features: bool = True


@dataclass(slots=True)
class LabelConfig:
    stop_atr_multiple: float = 1.25
    target_atr_multiple: float = 2.0
    max_holding_bars: int = 45
    max_adverse_excursion_atr: float = 1.15
    min_manager_edge_r: float = 0.0
    min_breakout_zscore: float = 1.2
    min_trend_strength: float = 0.2
    mean_reversion_extension_zscore: float = 1.0
    compression_threshold: float = 0.75
    sweep_wick_threshold: float = 0.45
    pullback_depth_atr: float = 0.8
    exhaustion_reversal_body_threshold: float = 0.2
    expectancy_min_trades: int = 30
    expectancy_confidence_level: float = 0.95


@dataclass(slots=True)
class ModelConfig:
    setup_names: list[str] = field(
        default_factory=lambda: [
            "trend_continuation",
            "pullback_continuation",
            "breakout_expansion",
            "mean_reversion",
            "liquidity_sweep_reversal",
            "volatility_compression_expansion",
            "session_open_momentum",
            "exhaustion_failure",
        ]
    )
    hidden_dim: int = 64
    shared_dim: int = 64
    expert_dim: int = 64
    manager_dim: int = 64
    num_tcn_layers: int = 4
    kernel_size: int = 3
    dropout: float = 0.1
    calibration_dim: int = 32


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    early_stopping_patience: int = 5
    gradient_clip_norm: float = 1.0
    class_positive_weight: float = 3.0
    manager_positive_weight: float = 2.0
    manager_false_positive_weight: float = 1.0
    calibration_weight: float = 0.2
    regression_weight: float = 0.2
    direction_loss_weight: float = 0.1
    direction_long_weight: float | None = None
    direction_short_weight: float | None = None
    direction_auto_balance: bool = True
    gate_supervision_weight: float = 0.3
    gate_balance_weight: float = 0.0
    gate_target_usage: dict[str, float] = field(default_factory=dict)
    expert_loss_weights: dict[str, float] = field(default_factory=dict)
    diversity_weight: float = 0.05
    entropy_weight: float = 0.01
    gate_negative_weight: float = 0.25
    direction_symmetry_min_share: float = 0.3
    use_dynamic_account_context: bool = True
    calibration_method: str = "platt"
    calibration_min_samples: int = 25
    seed: int = 42
    num_workers: int = 0
    device: str = "cpu"
    use_walk_forward: bool = True
    walk_forward_train_bars: int = 250_000
    walk_forward_validation_bars: int = 50_000
    walk_forward_test_bars: int = 50_000
    walk_forward_step_bars: int = 50_000


@dataclass(slots=True)
class BacktestConfig:
    spread_bps: float = 0.8
    slippage_bps: float = 0.5
    commission_bps: float = 0.2
    one_trade_per_asset: bool = True
    min_trade_probability: float = 0.55
    min_context_score: float = 0.5
    expert_min_trade_probability: dict[str, float] = field(default_factory=dict)
    allow_dual_asset_trades: bool = True
    min_expected_return_r: float = 0.0
    use_confidence_in_routing: bool = False
    challenge_risk_fraction: float = 0.009
    funded_risk_fraction: float = 0.005
    min_risk_fraction: float = 0.001
    max_risk_multiplier: float = 1.15
    target_trades_per_day_min: float = 2.0
    target_trades_per_day_max: float = 3.0
    exceptional_day_trade_cap: int = 4
    min_routed_share_per_active_expert: float = 0.05
    min_executed_trades_per_active_expert: int = 5
    min_active_experts: int = 4


@dataclass(slots=True)
class PropPhaseConfig:
    profit_target: float | None = None
    daily_loss_limit: float = 0.05
    overall_loss_limit: float = 0.10
    per_trade_loss_limit: float = 0.03
    minimum_profitable_days: int = 0
    allow_weekend_holding: bool = False
    allow_news_trading: bool = True
    max_concurrent_positions: int = 2
    max_aggregate_open_risk: float = 0.05


@dataclass(slots=True)
class PropConfig:
    starting_balance: float = 100_000.0
    challenge: PropPhaseConfig = field(
        default_factory=lambda: PropPhaseConfig(
            profit_target=0.10,
            daily_loss_limit=0.05,
            overall_loss_limit=0.10,
            per_trade_loss_limit=0.03,
            minimum_profitable_days=3,
            allow_weekend_holding=True,
            allow_news_trading=True,
            max_concurrent_positions=2,
            max_aggregate_open_risk=0.05,
        )
    )
    funded: PropPhaseConfig = field(
        default_factory=lambda: PropPhaseConfig(
            profit_target=None,
            daily_loss_limit=0.05,
            overall_loss_limit=0.10,
            per_trade_loss_limit=0.03,
            minimum_profitable_days=0,
            allow_weekend_holding=False,
            allow_news_trading=True,
            max_concurrent_positions=2,
            max_aggregate_open_risk=0.05,
        )
    )
    profit_split: float = 0.80
    challenge_fee_refund_multiple: float = 2.0




@dataclass(slots=True)
class ExpertCompletionCriteriaConfig:
    minimum_trade_count: int = 30
    minimum_post_cost_expectancy_r: float = 0.0
    max_drawdown_floor_r: float = -5.0


@dataclass(slots=True)
class ExpertSchedulerConfig:
    expert_priority: list[str] = field(
        default_factory=lambda: [
            "liquidity_sweep_reversal",
            "trend_continuation",
            "pullback_continuation",
            "breakout_expansion",
            "mean_reversion",
            "volatility_compression_expansion",
            "session_open_momentum",
            "exhaustion_failure",
        ]
    )
    completion: ExpertCompletionCriteriaConfig = field(default_factory=ExpertCompletionCriteriaConfig)


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "us100_us500_moe"
    output_dir: str = "artifacts/experiments/default"
    notes: str = "Baseline research-grade multi-asset MoE."
    scheduler: ExpertSchedulerConfig = field(default_factory=ExpertSchedulerConfig)


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    prop: PropConfig = field(default_factory=PropConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load application configuration from YAML and merge with defaults."""
    config = AppConfig()
    if path is None:
        return config

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _merge_dataclass(config, raw)
