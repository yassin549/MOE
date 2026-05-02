"""Cache helpers for expensive pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from moe_trading.config import AppConfig
from moe_trading.data.schemas import MultiAssetFrame
from moe_trading.utils.io import ensure_dir


PIPELINE_CACHE_VERSION = "v6"


def _source_metadata(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_research_frame_cache_key(config: AppConfig) -> str:
    """Build a stable cache key for the research-frame artifact."""
    payload = {
        "version": PIPELINE_CACHE_VERSION,
        "data": {
            "us100_file": config.data.us100_file,
            "us500_file": config.data.us500_file,
            "timestamp_col": config.data.timestamp_col,
            "asset_col": config.data.asset_col,
            "bar_minutes": config.data.bar_minutes,
            "max_rows": config.data.max_rows,
        },
        "features": asdict(config.features),
        "labels": asdict(config.labels),
        "setup_names": config.model.setup_names,
        "sources": {
            "us100": _source_metadata(config.data.us100_file),
            "us500": _source_metadata(config.data.us500_file),
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def research_frame_cache_paths(config: AppConfig) -> tuple[Path, Path]:
    """Return data and metadata paths for the current research-frame cache key."""
    cache_dir = ensure_dir(config.data.cache_dir)
    key = build_research_frame_cache_key(config)
    return cache_dir / f"{key}.pkl", cache_dir / f"{key}.json"


def save_research_frame_cache(bundle: MultiAssetFrame, config: AppConfig) -> None:
    """Persist the built research frame and column groups."""
    data_path, meta_path = research_frame_cache_paths(config)
    payload = {
        "frame": bundle.frame,
        "asset_feature_columns": bundle.asset_feature_columns,
        "cross_asset_feature_columns": bundle.cross_asset_feature_columns,
        "regime_feature_columns": bundle.regime_feature_columns,
        "label_columns": bundle.label_columns,
    }
    pd.to_pickle(payload, data_path)
    meta_path.write_text(
        json.dumps(
            {
                "cache_key": build_research_frame_cache_key(config),
                "data_path": str(data_path),
                "version": PIPELINE_CACHE_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_research_frame_cache(config: AppConfig) -> MultiAssetFrame | None:
    """Load a cached research frame when available."""
    data_path, _ = research_frame_cache_paths(config)
    if not data_path.exists():
        return None
    payload = pd.read_pickle(data_path)
    return MultiAssetFrame(
        frame=payload["frame"],
        asset_feature_columns=payload["asset_feature_columns"],
        cross_asset_feature_columns=payload["cross_asset_feature_columns"],
        regime_feature_columns=payload["regime_feature_columns"],
        label_columns=payload["label_columns"],
    )
