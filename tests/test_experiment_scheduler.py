import pandas as pd

from moe_trading.experiments.scheduler import ExpertCompletionCriteria, ExpertSchedulerConfig, expert_status_report


def test_expert_scheduler_priority_and_statuses():
    trades = pd.DataFrame(
        [
            {"expert": "liquidity_sweep_reversal", "net_return_r": 0.1},
            {"expert": "liquidity_sweep_reversal", "net_return_r": 0.2},
            {"expert": "trend_continuation", "net_return_r": -0.2},
            {"expert": "trend_continuation", "net_return_r": 0.1},
        ]
    )
    cfg = ExpertSchedulerConfig(
        expert_priority=["liquidity_sweep_reversal", "trend_continuation", "pullback_continuation"],
        completion=ExpertCompletionCriteria(minimum_trade_count=2, minimum_post_cost_expectancy_r=0.0, max_drawdown_floor_r=-0.5),
    )

    report = expert_status_report(trades, cfg)

    assert [item["expert"] for item in report] == ["liquidity_sweep_reversal", "trend_continuation", "pullback_continuation"]
    assert report[0]["status"] == "passed"
    assert report[1]["status"] == "failed"
    assert report[2]["status"] == "pending"
