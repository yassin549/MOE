from pathlib import Path

import pandas as pd
import pytest

from moe_trading.config import AppConfig
from moe_trading.cost_model import cost_model_metadata
from moe_trading.evaluation.reports import append_run_sheet, make_run_metadata


def test_cost_model_metadata_stable_for_same_config():
    config = AppConfig()
    left = cost_model_metadata(config)
    right = cost_model_metadata(config)
    assert left["cost_model_version"] == right["cost_model_version"]
    assert left["cost_model_hash"] == right["cost_model_hash"]


def test_append_run_sheet_blocks_mixed_cost_models_for_same_baseline_tag(tmp_path: Path):
    sheet = tmp_path / "run_sheet.csv"
    first = {
        **make_run_metadata(
            config_name="a.yaml",
            experiment_name="exp",
            output_dir="out/a",
            model_path=None,
            evaluation_start="2026-01-01T00:00:00+00:00",
            evaluation_end="2026-01-02T00:00:00+00:00",
            baseline_tag="baseline_1",
            cost_model_version="cm-aaaa",
        ),
        "summary_num_trades": 5,
    }
    append_run_sheet(first, sheet)

    second = {
        **make_run_metadata(
            config_name="b.yaml",
            experiment_name="exp",
            output_dir="out/b",
            model_path=None,
            evaluation_start="2026-01-01T00:00:00+00:00",
            evaluation_end="2026-01-02T00:00:00+00:00",
            baseline_tag="baseline_1",
            cost_model_version="cm-bbbb",
        ),
        "summary_num_trades": 7,
    }

    with pytest.raises(ValueError):
        append_run_sheet(second, sheet)

    append_run_sheet(second, sheet, allow_cost_model_mismatch=True)
    frame = pd.read_csv(sheet)
    assert len(frame) == 2
