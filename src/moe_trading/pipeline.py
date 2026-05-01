"""Top-level data preparation pipeline."""

from __future__ import annotations

from moe_trading.config import AppConfig
from moe_trading.data.loading import load_multi_asset_frame
from moe_trading.data.schemas import MultiAssetFrame
from moe_trading.features.engineering import build_feature_frame, collect_feature_columns
from moe_trading.labels.generation import generate_labels
from moe_trading.utils.cache import load_research_frame_cache, save_research_frame_cache


def _bundle_from_frame(frame, label_columns: list[str]) -> MultiAssetFrame:
    asset_columns, cross_columns, regime_columns = collect_feature_columns(frame)
    return MultiAssetFrame(
        frame=frame,
        asset_feature_columns=asset_columns,
        cross_asset_feature_columns=cross_columns,
        regime_feature_columns=regime_columns,
        label_columns=label_columns,
    )


def build_feature_bundle(config: AppConfig) -> MultiAssetFrame:
    """Load, align, and featurize the current data without label generation."""
    aligned = load_multi_asset_frame(config.data)
    feature_frame = build_feature_frame(aligned, config.features)
    return _bundle_from_frame(feature_frame, [])


def build_research_frame(config: AppConfig) -> MultiAssetFrame:
    """Load, align, featurize, and label the research frame."""
    if config.data.use_cache:
        cached_bundle = load_research_frame_cache(config)
        if cached_bundle is not None:
            return cached_bundle

    aligned = load_multi_asset_frame(config.data)
    feature_frame = build_feature_frame(aligned, config.features)
    labeled = generate_labels(feature_frame, config.labels, config.model.setup_names, config.backtest)
    label_columns = [
        column
        for column in labeled.columns
        if "_target" in column or "_direction" in column or "_return_r" in column or "_net_return_r" in column
    ]
    bundle = _bundle_from_frame(labeled, label_columns)
    if config.data.use_cache:
        save_research_frame_cache(bundle, config)
    return bundle
