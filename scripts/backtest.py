import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_trading.backtesting.realtime import RealtimeBacktestSimulator, build_realtime_components
from moe_trading.config import load_config
from moe_trading.evaluation.metrics import backtest_diagnostics
from moe_trading.evaluation.reports import append_run_sheet, flatten_for_sheet, make_run_metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the realtime MoE backtest simulator.")
    parser.add_argument("config", help="Path to the config file.")
    parser.add_argument("--experiment-dir", default=None, help="Experiment directory containing the model checkpoint.")
    parser.add_argument("--model-path", default=None, help="Optional direct model path.")
    parser.add_argument("--scaler-path", default=None, help="Optional direct scaler path.")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "realtime_backtest"), help="Output directory.")
    parser.add_argument("--sheet-path", default=str(ROOT / "reports" / "backtest_run_sheet.csv"), help="CSV sheet updated after each run.")
    parser.add_argument("--allow-mixed-cost-model-versions", action="store_true", help="Allow appending to run sheet when cost-model versions differ.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    config = load_config(args.config)
    candles, adapter, replay = build_realtime_components(
        config,
        experiment_dir=args.experiment_dir,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
    )
    result = RealtimeBacktestSimulator(replay).run(candles, adapter)
    evaluation_start = str(candles.timestamp_str[0]) if len(candles.timestamp_str) else None
    evaluation_end = str(candles.timestamp_str[-1]) if len(candles.timestamp_str) else None
    diagnostics = backtest_diagnostics(result.trades_frame, config, evaluation_start, evaluation_end)
    detailed_metrics = {
        "summary": result.summary,
        "diagnostics": diagnostics,
        "performance": result.performance,
        "metadata": make_run_metadata(
            config_name=str(args.config),
            experiment_name=config.experiment.name,
            output_dir=str(args.output_dir),
            model_path=str(args.model_path) if args.model_path is not None else None,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            asset_universe=["US100", "US500"],
            config=config,
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "backtest_summary.json").write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    (output_dir / "backtest_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (output_dir / "backtest_detailed_metrics.json").write_text(json.dumps(detailed_metrics, indent=2), encoding="utf-8")
    (output_dir / "backtest_performance.json").write_text(json.dumps(result.performance, indent=2), encoding="utf-8")
    np.save(output_dir / "equity_curve.npy", result.equity_curve)
    result.trades_frame.to_csv(output_dir / "trades.csv", index=False)
    row = {
        **make_run_metadata(
            config_name=str(args.config),
            experiment_name=config.experiment.name,
            output_dir=str(output_dir),
            model_path=str(args.model_path) if args.model_path is not None else None,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            asset_universe=["US100", "US500"],
            config=config,
        ),
        **flatten_for_sheet(result.summary, "summary"),
        **flatten_for_sheet(diagnostics, "diag"),
        **flatten_for_sheet(result.performance, "perf"),
    }
    append_run_sheet(row, args.sheet_path, allow_mixed_cost_model_versions=args.allow_mixed_cost_model_versions)

    print(
        "Realtime backtest complete: "
        f"trades={result.summary.get('num_trades', 0)}, "
        f"win_rate={result.summary.get('win_rate', 0.0):.4f}, "
        f"ending_equity_r={result.summary.get('ending_equity_r', 0.0):.4f}, "
        f"bars_per_second={result.performance.get('bars_per_second', 0.0):.2f}"
    )
