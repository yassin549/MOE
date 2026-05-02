import numpy as np
import torch

from moe_trading.config import AppConfig
from moe_trading.training.pipeline import _fit_manager_calibration_and_threshold, _fit_manager_calibration_from_arrays


def test_fit_manager_calibration_from_arrays_returns_serializable_metrics():
    config = AppConfig()
    y_prob = np.array([0.2, 0.35, 0.6, 0.8], dtype=np.float32)
    y_true = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    realized_returns = np.array([-0.002, -0.001, 0.003, 0.006], dtype=np.float32)

    calibration, threshold, metrics = _fit_manager_calibration_from_arrays(y_prob, y_true, realized_returns, config)

    assert set(calibration) == {"scale", "bias"}
    assert 0.05 <= threshold <= 0.95
    assert set(metrics) == {"pre", "post", "selection_post_cost_expectancy"}
    assert set(metrics["pre"]) == {
        "brier",
        "calibration_error",
        "precision_at_threshold",
        "routed_expectancy",
        "route_rate",
    }


class _DummyOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.manager_trade_logits = logits


class _DummyModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch_size = kwargs["manager_context"].shape[0]
        return _DummyOutput(torch.linspace(-0.5, 0.5, steps=batch_size, dtype=torch.float32).unsqueeze(-1))


def test_fit_manager_calibration_and_threshold_aligns_returns_per_sample():
    config = AppConfig()
    batch_size = 4
    loader = [
        {
            "asset_sequences": {
                "US100": torch.zeros(batch_size, 2, 1),
                "US500": torch.zeros(batch_size, 2, 1),
            },
            "cross_sequence": torch.zeros(batch_size, 2, 1),
            "regime_sequence": torch.zeros(batch_size, 2, 1),
            "manager_context": torch.zeros(batch_size, 1),
            "account_context": torch.zeros(batch_size, 11),
            "expert_valids": torch.zeros(batch_size, 2, 8),
            "expert_labels": torch.zeros(batch_size, 2, 8),
            "directions": torch.ones(batch_size, 2, 8),
            "manager_labels": torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            "asset_manager_labels": torch.zeros(batch_size, 2),
            "gate_supervision_mask": torch.zeros(batch_size, 2),
            "gate_targets": torch.zeros(batch_size, 2, dtype=torch.int64),
            "returns": torch.tensor(
                [
                    [[-0.3] * 8, [-0.2] * 8],
                    [[0.4] * 8, [0.1] * 8],
                    [[-0.1] * 8, [-0.4] * 8],
                    [[0.2] * 8, [0.5] * 8],
                ],
                dtype=torch.float32,
            ),
            "timestamps": ["t0", "t1", "t2", "t3"],
        }
    ]

    calibration = _fit_manager_calibration_and_threshold(_DummyModel(), loader, torch.device("cpu"), config)

    assert set(calibration) >= {"scale", "bias", "threshold", "metrics_pre", "metrics_post"}
