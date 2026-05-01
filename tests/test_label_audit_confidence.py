import pandas as pd

from moe_trading.config import AppConfig
from moe_trading.labels.audit import compute_heuristic_baseline_table


def test_heuristic_baseline_marks_not_enough_data_when_under_threshold():
    config = AppConfig()
    config.backtest.expectancy_min_trades = 3
    label_audit = pd.DataFrame(
        [
            {
                "asset": "US100",
                "setup": "trend",
                "setup_present_count": 2,
                "positive_net_rate": 1.0,
                "post_cost_expectancy_r": 0.4,
                "valid_post_cost_std_r": 0.1,
                "direction_long_share": 1.0,
                "direction_short_share": 0.0,
            }
        ]
    )

    baseline = compute_heuristic_baseline_table(label_audit, config)

    assert baseline.loc[0, "expectancy_status"] == "not_enough_data"
    assert not bool(baseline.loc[0, "expectancy_evaluable"])
    assert not bool(baseline.loc[0, "profitability_gate_pass"])


def test_heuristic_baseline_uses_ci_lower_bound_for_non_negative_expectancy_gate():
    config = AppConfig()
    config.backtest.expectancy_min_trades = 3
    config.backtest.expectancy_confidence_level = 0.95
    label_audit = pd.DataFrame(
        [
            {
                "asset": "US100",
                "setup": "trend",
                "setup_present_count": 3,
                "positive_net_rate": 2 / 3,
                "post_cost_expectancy_r": 0.2,
                "valid_post_cost_std_r": 0.6,
                "direction_long_share": 0.7,
                "direction_short_share": 0.3,
            }
        ]
    )

    baseline = compute_heuristic_baseline_table(label_audit, config)

    assert baseline.loc[0, "expectancy_status"] == "evaluable"
    assert bool(baseline.loc[0, "expectancy_evaluable"])
    assert baseline.loc[0, "expectancy_r"] > 0.0
    assert baseline.loc[0, "expectancy_ci_lower_r"] < 0.0
    assert not bool(baseline.loc[0, "profitability_gate_pass"])
