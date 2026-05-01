import pandas as pd

from moe_trading.config import BacktestConfig, LabelConfig
from moe_trading.labels.generation import generate_labels


def test_generate_labels_adds_manager_targets():
    size = 300
    frame = pd.DataFrame(
        {
            "us100_high": [101.0 + i * 0.1 for i in range(size)],
            "us100_low": [99.0 + i * 0.1 for i in range(size)],
            "us100_close": [100.0 + i * 0.1 for i in range(size)],
            "us100_atr_15": [1.0] * size,
            "us100_slope_15": [0.3] * size,
            "us100_momentum_10": [0.2] * size,
            "us100_momentum_5": [0.1] * size,
            "us100_momentum_3": [0.1] * size,
            "us100_distance_high_20": [0.0] * size,
            "us100_volume_z_15": [1.0] * size,
            "us100_lower_wick": [0.01] * size,
            "us100_upper_wick": [0.01] * size,
            "us100_range": [0.02] * size,
            "us100_compression_20": [0.5] * size,
            "us100_outside_bar": [0.0] * size,
            "us100_body": [0.001] * size,
            "us500_high": [101.0 + i * 0.1 for i in range(size)],
            "us500_low": [99.0 + i * 0.1 for i in range(size)],
            "us500_close": [100.0 + i * 0.1 for i in range(size)],
            "us500_atr_15": [1.0] * size,
            "us500_slope_15": [0.3] * size,
            "us500_momentum_10": [0.2] * size,
            "us500_momentum_5": [0.1] * size,
            "us500_momentum_3": [0.1] * size,
            "us500_distance_high_20": [0.0] * size,
            "us500_volume_z_15": [1.0] * size,
            "us500_lower_wick": [0.01] * size,
            "us500_upper_wick": [0.01] * size,
            "us500_range": [0.02] * size,
            "us500_compression_20": [0.5] * size,
            "us500_outside_bar": [0.0] * size,
            "us500_body": [0.001] * size,
            "trend_agreement": [1.0] * size,
            "spread_return_diff": [0.0] * size,
            "spread_z_20": [0.0] * size,
            "divergent_regime": [0.0] * size,
            "joint_volatility": [0.1] * size,
            "is_session_open_window": [1.0] * size,
        }
    )
    labeled = generate_labels(
        frame,
        LabelConfig(),
        [
            "trend_continuation",
            "pullback_continuation",
            "breakout_expansion",
            "mean_reversion",
            "liquidity_sweep_reversal",
            "volatility_compression_expansion",
            "session_open_momentum",
            "exhaustion_failure",
        ],
    )
    assert "manager_trade_target" in labeled.columns
    assert "manager_dual_trade_target" in labeled.columns
    assert "us100_manager_trade_target" in labeled.columns
    assert "us500_manager_trade_target" in labeled.columns
    assert "us100_manager_best_expert" in labeled.columns
    assert "us500_manager_best_expert" in labeled.columns
    assert "us100_trend_continuation_net_return_r" in labeled.columns
    assert "us500_trend_continuation_net_return_r" in labeled.columns


def test_generate_labels_uses_net_return_for_manager_targets():
    size = 260
    frame = pd.DataFrame(
        {
            "us100_high": [100.01] * size,
            "us100_low": [99.99] * size,
            "us100_close": [100.0] * size,
            "us100_atr_15": [0.0005] * size,
            "us100_slope_15": [0.3] * size,
            "us100_momentum_10": [0.2] * size,
            "us100_momentum_5": [0.1] * size,
            "us100_momentum_3": [0.1] * size,
            "us100_distance_high_20": [0.0] * size,
            "us100_volume_z_15": [1.0] * size,
            "us100_lower_wick": [0.01] * size,
            "us100_upper_wick": [0.01] * size,
            "us100_range": [0.02] * size,
            "us100_compression_20": [0.5] * size,
            "us100_outside_bar": [0.0] * size,
            "us100_body": [0.001] * size,
            "us500_high": [100.01] * size,
            "us500_low": [99.99] * size,
            "us500_close": [100.0] * size,
            "us500_atr_15": [0.0005] * size,
            "us500_slope_15": [0.3] * size,
            "us500_momentum_10": [0.2] * size,
            "us500_momentum_5": [0.1] * size,
            "us500_momentum_3": [0.1] * size,
            "us500_distance_high_20": [0.0] * size,
            "us500_volume_z_15": [1.0] * size,
            "us500_lower_wick": [0.01] * size,
            "us500_upper_wick": [0.01] * size,
            "us500_range": [0.02] * size,
            "us500_compression_20": [0.5] * size,
            "us500_outside_bar": [0.0] * size,
            "us500_body": [0.001] * size,
            "trend_agreement": [1.0] * size,
            "spread_return_diff": [0.0] * size,
            "spread_z_20": [0.0] * size,
            "divergent_regime": [0.0] * size,
            "joint_volatility": [0.1] * size,
            "is_session_open_window": [1.0] * size,
        }
    )

    labeled = generate_labels(
        frame,
        LabelConfig(min_manager_edge_r=0.0),
        ["trend_continuation"],
        BacktestConfig(spread_bps=30.0, slippage_bps=30.0, commission_bps=30.0),
    )

    assert (labeled["us100_trend_continuation_net_return_r"] <= labeled["us100_trend_continuation_return_r"]).all()
    assert labeled["us100_manager_trade_target"].sum() == 0
