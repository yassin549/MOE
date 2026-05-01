import pandas as pd

from moe_trading.config import AppConfig
from moe_trading.labels.audit import compute_heuristic_baseline_table, compute_label_audit_table


def test_heuristic_baseline_reports_not_enough_data_when_below_threshold():
    frame = pd.DataFrame(
        {
            "us100_trend_continuation_present": [1, 1],
            "us100_trend_continuation_valid": [1, 1],
            "us100_trend_continuation_target": [1, 0],
            "us100_trend_continuation_tradable": [1, 1],
            "us100_trend_continuation_direction": [1, -1],
            "us100_trend_continuation_return_r": [0.4, 0.1],
            "us100_trend_continuation_net_return_r": [0.3, 0.05],
        }
    )
    config = AppConfig()
    config.model.setup_names = ["trend_continuation"]
    config.labels.expectancy_min_trades = 5

    label_audit = compute_label_audit_table(frame, config.model.setup_names, config, assets=("US100",))
    baseline = compute_heuristic_baseline_table(label_audit)

    row = baseline.iloc[0]
    assert row["trades"] == 2
    assert not bool(row["expectancy_evaluable"])
    assert row["expectancy_data_state"] == "not_enough_data"
    assert not bool(row["profitability_gate_pass"])


def test_heuristic_baseline_requires_positive_point_and_non_negative_ci_bound():
    label_audit = pd.DataFrame(
        [
            {
                "asset": "US100",
                "setup": "trend_continuation",
                "setup_present_count": 50,
                "positive_net_rate": 0.55,
                "post_cost_expectancy_r": 0.01,
                "expectancy_ci_lower_r": -0.02,
                "expectancy_ci_upper_r": 0.03,
                "expectancy_min_trades": 30,
                "direction_long_share": 0.5,
                "direction_short_share": 0.5,
            },
            {
                "asset": "US500",
                "setup": "trend_continuation",
                "setup_present_count": 50,
                "positive_net_rate": 0.62,
                "post_cost_expectancy_r": 0.02,
                "expectancy_ci_lower_r": 0.001,
                "expectancy_ci_upper_r": 0.04,
                "expectancy_min_trades": 30,
                "direction_long_share": 0.6,
                "direction_short_share": 0.4,
            },
        ]
    )

    baseline = compute_heuristic_baseline_table(label_audit)

    assert bool(baseline.loc[0, "expectancy_evaluable"])
    assert not bool(baseline.loc[0, "profitability_gate_pass"])
    assert not bool(baseline.loc[0, "expectancy_significant"])

    assert bool(baseline.loc[1, "expectancy_evaluable"])
    assert bool(baseline.loc[1, "profitability_gate_pass"])
    assert bool(baseline.loc[1, "expectancy_significant"])
