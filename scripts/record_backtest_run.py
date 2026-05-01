import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_trading.config import load_config
from moe_trading.experiments.scheduler import ExpertCompletionCriteria, ExpertSchedulerConfig, expert_status_report
from moe_trading.utils.io import save_json
from moe_trading.evaluation.metrics import backtest_diagnostics, trade_metrics
from moe_trading.evaluation.reports import append_run_sheet, flatten_for_sheet, make_run_metadata


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

    row = {
        **make_run_metadata(
            config_name=str(args.config),
            experiment_name=config.experiment.name,
            output_dir=str(args.output_dir or Path(args.trades).parent),
            model_path=args.model_path,
            evaluation_start=args.evaluation_start,
            evaluation_end=args.evaluation_end,
            baseline_tag=args.baseline_tag,
        ),
        **flatten_for_sheet(summary, "summary"),
        **flatten_for_sheet(diagnostics, "diag"),
    }
    for idx, item in enumerate(expert_status):
        row.update(flatten_for_sheet(item, f"expert_status_{idx:02d}"))
    append_run_sheet(row, args.sheet_path)
    artifact_path = Path(args.output_dir or Path(args.trades).parent) / "expert_experiment_status.json"
    save_json({"scheduler": asdict(scheduler_cfg), "experts": expert_status}, artifact_path)
    print(json.dumps({"recorded": True, "sheet_path": args.sheet_path, "num_trades": int(summary["num_trades"]), "expert_status_path": str(artifact_path)}, indent=2))
