from pathlib import Path

import pandas as pd

from moe_trading.config import AppConfig
from moe_trading.data.schemas import MultiAssetFrame
from moe_trading.utils.cache import (
    build_research_frame_cache_key,
    load_research_frame_cache,
    save_research_frame_cache,
)


def test_research_frame_cache_round_trip(tmp_path: Path):
    config = AppConfig()
    config.data.cache_dir = str(tmp_path / "cache")
    config.data.us100_file = str(tmp_path / "us100.csv")
    config.data.us500_file = str(tmp_path / "us500.csv")

    source = "timestamp_utc,open,high,low,close,volume,hour_utc,minute_utc,day_of_week,session_utc\n"
    row = "2026-01-01T00:00:00Z,1,1,1,1,1,0,0,3,asia\n"
    Path(config.data.us100_file).write_text(source + row, encoding="utf-8")
    Path(config.data.us500_file).write_text(source + row, encoding="utf-8")

    bundle = MultiAssetFrame(
        frame=pd.DataFrame({"timestamp": ["2026-01-01T00:00:00Z"], "x": [1.0]}),
        asset_feature_columns={"US100": ["x"], "US500": ["x"]},
        cross_asset_feature_columns=["x"],
        regime_feature_columns=[],
        label_columns=[],
    )

    save_research_frame_cache(bundle, config)
    loaded = load_research_frame_cache(config)

    assert loaded is not None
    assert loaded.frame.equals(bundle.frame)
    assert build_research_frame_cache_key(config)
