import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_trading.config import load_config
from moe_trading.cost_model import cost_model_metadata
from moe_trading.evaluation.metrics import backtest_diagnostics, expert_trade_metrics, trade_metrics
from moe_trading.evaluation.reports import append_run_sheet, flatten_for_sheet, make_run_metadata
from moe_trading.experiments.scheduler import ExpertCompletionCriteria, ExpertSchedulerConfig, expert_status_report
from moe_trading.utils.io import save_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record an existing backtest run into the shared metrics sheet.")
    parser.add_argument("--config", required=True, help="Config path used for the run.")
    parser.add_argument("--trades", required=True, help="Trades CSV path.")
    parser.add_argument("--sheet-path", default=str(ROOT / "reports" / "backtest_run_sheet.csv"), help="Destination CSV sheet.")
    parser.add_argument("--evaluation-start", default=None, help="Optional evaluation window start timestamp.")
    parser.add_argument("--evaluation-end", default=None, help="Optional evaluation window end timestamp.")
    parser.add_argument("--baseline-tag", default=None, help="Optional tag like 'baseline_pre_fix'.")
    parser.add_argument("--output-dir", default=None, help="Optional original output directory.")
    parser.add_argument("--model-path", default=None, help="Optional original model path.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    config = load_config(args.config)
    run_cost_model = cost_model_metadata(config)
    trades = pd.read_csv(args.trades)
    if "risk_fraction" not in trades.columns:
        trades["risk_fraction"] = float(config.backtest.challenge_risk_fraction)
    summary = trade_metrics(trades)
    diagnostics = backtest_diagnostics(trades, config, args.evaluation_start, args.evaluation_end)
    scheduler_cfg = ExpertSchedulerConfig(
        expert_priority=list(config.experiment.scheduler.expert_priority),
        completion=ExpertCompletionCriteria(
            minimum_trade_count=int(config.experiment.scheduler.completion.minimum_trade_count),
            minimum_post_cost_expectancy_r=float(config.experiment.scheduler.completion.minimum_post_cost_expectancy_r),
            max_drawdown_floor_r=float(config.experiment.scheduler.completion.max_drawdown_floor_r),
        ),
    )
    expert_status = expert_status_report(trades, scheduler_cfg)
    expert_metrics = expert_trade_metrics(trades)
    output_dir = Path(args.output_dir or Path(args.trades).parent)
    row = {
        **make_run_metadata(
            config_name=str(args.config),
            experiment_name=config.experiment.name,
            output_dir=str(output_dir),
            model_path=args.model_path,
            evaluation_start=args.evaluation_start,
            evaluation_end=args.evaluation_end,
            baseline_tag=args.baseline_tag,
            **run_cost_model,
        ),
        **flatten_for_sheet(summary, "summary"),
        **flatten_for_sheet(diagnostics, "diag"),
    }
    for idx, item in enumerate(expert_status):
        row.update(flatten_for_sheet(item, f"expert_status_{idx:02d}"))
    append_run_sheet(row, args.sheet_path)
    artifact_path = output_dir / "expert_experiment_status.json"
    save_json({"scheduler": asdict(scheduler_cfg), "experts": expert_status}, artifact_path)
    if expert_metrics:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(expert_metrics).to_csv(output_dir / "expert_trade_metrics.csv", index=False)
    print(
        json.dumps(
            {
                "recorded": True,
                "sheet_path": args.sheet_path,
                "num_trades": int(summary["num_trades"]),
                "expert_status_path": str(artifact_path),
            },
            indent=2,
        )
    )
