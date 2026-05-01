import torch
import pandas as pd

from moe_trading.config import ModelConfig
from moe_trading.data.dataset import MultiAssetSequenceDataset
from moe_trading.data.schemas import MultiAssetFrame
from moe_trading.features.engineering import collect_feature_columns
from moe_trading.models.moe import MultiAssetMoE


def test_moe_forward_shapes():
    config = ModelConfig()
    model = MultiAssetMoE(asset_input_dim=10, cross_input_dim=6, regime_input_dim=4, manager_context_dim=21, config=config)
    batch_size = 3
    seq_len = 32
    output = model(
        asset_sequences={
            "US100": torch.randn(batch_size, seq_len, 10),
            "US500": torch.randn(batch_size, seq_len, 10),
        },
        cross_sequence=torch.randn(batch_size, seq_len, 6),
        regime_sequence=torch.randn(batch_size, seq_len, 4),
        manager_context=torch.randn(batch_size, 10),
        account_context=torch.randn(batch_size, 11),
    )
    assert output.expert_probabilities.shape == (batch_size, 2, len(config.setup_names))
    assert output.manager_trade_probability.shape == (batch_size, 1)
    assert output.manager_gate_weights.shape == (batch_size, 2, len(config.setup_names))


def test_sequence_dataset_supports_feature_only_inference_bundle():
    config = ModelConfig()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC"),
            "us100_feat": [1.0, 2.0, 3.0, 4.0],
            "us500_feat": [1.5, 2.5, 3.5, 4.5],
            "cross_feat": [0.1, 0.2, 0.3, 0.4],
            "regime_feat": [0.0, 1.0, 0.0, 1.0],
        }
    )
    bundle = MultiAssetFrame(
        frame=frame,
        asset_feature_columns={"US100": ["us100_feat"], "US500": ["us500_feat"]},
        cross_asset_feature_columns=["cross_feat"],
        regime_feature_columns=["regime_feat"],
        label_columns=[],
    )

    dataset = MultiAssetSequenceDataset(bundle, sequence_length=2, setup_names=config.setup_names)
    sample = dataset[0]

    assert len(dataset) == 3
    assert sample.expert_valids.shape == (2, len(config.setup_names))
    assert sample.manager_label.shape == (2,)
    assert float(sample.directions[0, 0]) == 1.0


def test_collect_feature_columns_excludes_manager_label_leakage():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="min", tz="UTC"),
            "us100_feat": [1.0, 2.0],
            "us500_feat": [1.5, 2.5],
            "joint_volatility": [0.1, 0.2],
            "spread_close_ratio": [1.0, 1.1],
            "us100_manager_best_expert": [1, 0],
            "manager_trade_target": [1, 0],
        }
    )

    asset_columns, cross_columns, regime_columns = collect_feature_columns(frame)

    assert "us100_feat" in asset_columns["US100"]
    assert "us500_feat" in asset_columns["US500"]
    assert "joint_volatility" in regime_columns
    assert "spread_close_ratio" in cross_columns
    assert "us100_manager_best_expert" not in asset_columns["US100"]
