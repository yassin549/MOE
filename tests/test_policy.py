import numpy as np
import pandas as pd
import torch

from moe_trading.account.rules import PropRuleEngine
from moe_trading.account.state import AccountPhase, AccountState
from moe_trading.config import AppConfig, BacktestConfig, PropConfig
from moe_trading.policy.decision import ACCOUNT_FEATURE_NAMES, encode_account_state, expert_trade_threshold, routed_expert_scores
from moe_trading.training.account_context import build_account_context_array
from moe_trading.utils.calibration import apply_calibration


def test_encode_account_state_exposes_stable_feature_vector():
    engine = PropRuleEngine(PropConfig())
    state = AccountState(
        phase=AccountPhase.FUNDED,
        current_equity=103_500.0,
        current_balance=103_000.0,
        daily_peak_equity=104_000.0,
        overall_peak_equity=105_000.0,
        profitable_day_count=2,
        days_elapsed=8,
        current_open_risk=0.015,
    )

    features = encode_account_state(state, engine)

    assert features.shape == (len(ACCOUNT_FEATURE_NAMES),)
    assert features.dtype == np.float32
    assert features[0] == 0.0
    assert features[1] == 1.0
    assert features[2] == 0.0
    assert features[6] == 0.015


def test_expert_trade_threshold_uses_expert_override_before_global_default():
    config = BacktestConfig(
        min_trade_probability=0.55,
        expert_min_trade_probability={"trend_continuation": 0.62},
    )

    assert expert_trade_threshold(config, "trend_continuation") == 0.62
    assert expert_trade_threshold(config, "pullback_continuation") == 0.55


def test_routed_expert_scores_can_ignore_confidence():
    config = BacktestConfig(use_confidence_in_routing=False)
    scores = routed_expert_scores(
        probabilities=np.array([0.6, 0.4], dtype=np.float32),
        gate_weights=np.array([0.5, 0.5], dtype=np.float32),
        expected_returns=np.array([1.0, 1.0], dtype=np.float32),
        confidence=np.array([0.1, 0.9], dtype=np.float32),
        config=config,
    )
    assert scores[0] > scores[1]


def test_apply_calibration_respects_saved_scale_and_bias():
    logits = np.array([[[0.0, 1.0]]], dtype=np.float32)
    calibrated = apply_calibration(
        torch.from_numpy(logits),
        {"scales": [[2.0, 1.0]], "biases": [[0.0, -1.0]]},
    ).numpy()
    assert calibrated.shape == logits.shape
    assert calibrated[0, 0, 0] == 0.5
    assert calibrated[0, 0, 1] < 0.7311


def test_build_account_context_array_is_dynamic_when_targets_exist():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC"),
            "us100_close": [100.0, 101.0, 102.0, 103.0],
            "us100_atr_15": [1.0, 1.0, 1.0, 1.0],
            "us100_manager_trade_target": [1, 0, 0, 0],
            "us100_manager_best_expert": [0, -1, -1, -1],
            "us100_trend_continuation_direction": [1, 1, 1, 1],
            "us100_trend_continuation_net_return_r": [1.0, 0.0, 0.0, 0.0],
            "us100_trend_continuation_resolution_bars": [1, 1, 1, 1],
            "us500_close": [100.0, 100.0, 100.0, 100.0],
            "us500_atr_15": [1.0, 1.0, 1.0, 1.0],
            "us500_manager_trade_target": [0, 0, 0, 0],
            "us500_manager_best_expert": [-1, -1, -1, -1],
        }
    )
    config = AppConfig()
    config.model.setup_names = ["trend_continuation"]
    contexts = build_account_context_array(frame, config)
    assert contexts.shape == (4, len(ACCOUNT_FEATURE_NAMES))
    assert not np.allclose(contexts[0], contexts[-1])
