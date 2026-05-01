import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_trading.config import load_config
from moe_trading.cost_model import cost_model_metadata
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
    parser.add_argument("--allow-cost-model-mismatch", action="store_true", help="Override cost-model guard when appending comparable runs.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    config = load_config(args.config)
    trades = pd.read_csv(args.trades)
    if "risk_fraction" not in trades.columns:
        trades["risk_fraction"] = float(config.backtest.challenge_risk_fraction)
    summary = trade_metrics(trades)
    diagnostics = backtest_diagnostics(trades, config, args.evaluation_start, args.evaluation_end)
    cost_meta = cost_model_metadata(config)
    asset_universe = ",".join(sorted({str(a) for a in trades["asset"].dropna().unique()})) if "asset" in trades.columns else None
    data_period_start = str(trades["timestamp"].min()) if "timestamp" in trades.columns and not trades.empty else args.evaluation_start
    data_period_end = str(trades["timestamp"].max()) if "timestamp" in trades.columns and not trades.empty else args.evaluation_end
    row = {
        **make_run_metadata(
            config_name=str(args.config),
            experiment_name=config.experiment.name,
            output_dir=str(args.output_dir or Path(args.trades).parent),
            model_path=args.model_path,
            evaluation_start=args.evaluation_start,
            evaluation_end=args.evaluation_end,
            baseline_tag=args.baseline_tag,
            **cost_meta,
            data_period_start=data_period_start,
            data_period_end=data_period_end,
            asset_universe=asset_universe,
        ),
        **flatten_for_sheet(summary, "summary"),
        **flatten_for_sheet(diagnostics, "diag"),
    }
    append_run_sheet(row, args.sheet_path, allow_cost_model_mismatch=args.allow_cost_model_mismatch)
    print(json.dumps({"recorded": True, "sheet_path": args.sheet_path, "num_trades": int(summary["num_trades"])}, indent=2))
