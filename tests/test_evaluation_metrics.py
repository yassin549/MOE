from pathlib import Path
import shutil

import pandas as pd

from moe_trading.backtesting.engine import _discover_backtest_artifacts
from moe_trading.config import AppConfig
from moe_trading.evaluation.metrics import backtest_diagnostics, expert_trade_metrics, trade_metrics


def test_trade_metrics_exposes_quant_summary_fields():
    trades = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T10:00:00+00:00", "net_return_r": 1.0, "asset": "US100", "expert": "trend", "direction": 1},
            {"timestamp": "2026-01-01T11:00:00+00:00", "net_return_r": -0.5, "asset": "US100", "expert": "trend", "direction": 1, "risk_fraction": 0.009},
            {"timestamp": "2026-01-02T10:00:00+00:00", "net_return_r": -0.25, "asset": "US500", "expert": "mean_reversion", "direction": -1, "risk_fraction": 0.01},
            {"timestamp": "2026-01-02T11:00:00+00:00", "net_return_r": 0.75, "asset": "US500", "expert": "mean_reversion", "direction": -1, "risk_fraction": 0.008},
            {"timestamp": "2026-01-02T12:00:00+00:00", "net_return_r": 0.0, "asset": "US500", "expert": "trend", "direction": 1, "risk_fraction": 0.009},
        ]
    )
    trades.loc[0, "risk_fraction"] = 0.008

    summary = trade_metrics(trades)

    assert summary["num_trades"] == 5
    assert summary["win_rate"] == 0.4
    assert summary["loss_rate"] == 0.4
    assert summary["breakeven_rate"] == 0.2
    assert summary["gross_profit_r"] == 1.75
    assert summary["gross_loss_r"] == -0.75
    assert round(summary["profit_factor"], 6) == round(1.75 / 0.75, 6)
    assert summary["max_drawdown_r"] == -0.75
    assert summary["ending_equity_r"] == 1.0
    assert summary["longest_win_streak"] == 1
    assert summary["longest_losing_streak"] == 2
    assert summary["active_days"] == 2
    assert summary["trades_per_day"] == 2.5
    assert summary["daily_trade_count_min"] == 2
    assert summary["daily_trade_count_max"] == 3
    assert summary["median_risk_fraction"] == 0.009


def test_trade_metrics_handles_all_wins_without_infinite_json_values():
    trades = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T10:00:00+00:00", "net_return_r": 0.5, "asset": "US100", "expert": "trend", "direction": 1},
            {"timestamp": "2026-01-01T11:00:00+00:00", "net_return_r": 0.25, "asset": "US500", "expert": "trend", "direction": 1},
        ]
    )

    summary = trade_metrics(trades)

    assert summary["profit_factor"] is None
    assert summary["longest_losing_streak"] == 0
    assert summary["max_drawdown_r"] == 0.0


def test_backtest_diagnostics_reports_period_and_pass_fields():
    trades = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T10:00:00+00:00", "net_return_r": 0.06, "asset": "US100", "expert": "trend", "direction": 1, "risk_fraction": 0.009},
            {"timestamp": "2026-01-02T10:00:00+00:00", "net_return_r": 0.02, "asset": "US100", "expert": "trend", "direction": 1, "risk_fraction": 0.009},
            {"timestamp": "2026-01-03T10:00:00+00:00", "net_return_r": 0.03, "asset": "US500", "expert": "trend", "direction": 1, "risk_fraction": 0.009},
        ]
    )
    config = AppConfig()

    diagnostics = backtest_diagnostics(trades, config, "2026-01-01T00:00:00+00:00", "2026-01-03T23:59:00+00:00")

    assert diagnostics["daily"]["periods"] == 3
    assert diagnostics["daily"]["win_rate"] == 1.0
    assert diagnostics["average_risk_per_trade_by_expert"]["trend"] == 0.009
    assert diagnostics["active_expert_count"] == 1
    assert diagnostics["experts_with_positive_expectancy"] == 1
    assert diagnostics["experts_with_confident_positive_expectancy"] == 1
    assert diagnostics["days_to_pass_from_start"] == 3


def test_expert_trade_metrics_reports_usage_direction_and_asset_balance():
    trades = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T10:00:00+00:00", "net_return_r": 0.5, "asset": "US100", "expert": "trend", "direction": 1, "risk_fraction": 0.009},
            {"timestamp": "2026-01-01T11:00:00+00:00", "net_return_r": -0.25, "asset": "US500", "expert": "trend", "direction": -1, "risk_fraction": 0.007},
            {"timestamp": "2026-01-02T10:00:00+00:00", "net_return_r": 0.25, "asset": "US500", "expert": "mean_reversion", "direction": -1, "risk_fraction": 0.008},
        ]
    )

    metrics = expert_trade_metrics(trades)

    assert [row["expert"] for row in metrics] == ["trend", "mean_reversion"]
    assert metrics[0]["executed_trade_count"] == 2
    assert round(metrics[0]["routed_usage_share"], 6) == round(2 / 3, 6)
    assert metrics[0]["direction_long_share"] == 0.5
    assert metrics[0]["direction_short_share"] == 0.5
    assert metrics[0]["asset_us100_share"] == 0.5
    assert metrics[0]["asset_us500_share"] == 0.5
    assert "expectancy_ci_lower_r" in metrics[0]


def test_discover_backtest_artifacts_prefers_split_directories():
    scratch = Path("artifacts/test_scratch/backtest_artifacts")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    try:
        split_1 = scratch / "split_01"
        split_1.mkdir()
        (split_1 / "model.pt").write_text("model", encoding="utf-8")
        (split_1 / "scaler.json").write_text("{}", encoding="utf-8")

        split_2 = scratch / "split_02"
        split_2.mkdir()
        (split_2 / "model.pt").write_text("model", encoding="utf-8")
        (split_2 / "scaler.json").write_text("{}", encoding="utf-8")

        artifacts = _discover_backtest_artifacts(scratch)

        assert [artifact.name for artifact in artifacts] == ["split_01", "split_02"]
        assert [artifact.split_index for artifact in artifacts] == [1, 2]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
