import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_trading.backtesting.realtime import RealtimeBacktestSimulator, build_realtime_components
from moe_trading.config import load_config
from moe_trading.cost_model import cost_model_metadata
from moe_trading.data.splitting import make_time_split
from moe_trading.evaluation.metrics import backtest_diagnostics, expert_trade_metrics, routed_usage_gate, trade_metrics
from moe_trading.evaluation.reports import append_run_sheet, flatten_for_sheet, make_run_metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the realtime MoE backtest simulator.")
    parser.add_argument("config", help="Path to the config file.")
    parser.add_argument("--experiment-dir", default=None, help="Experiment directory containing the model checkpoint.")
    parser.add_argument("--model-path", default=None, help="Optional direct model path.")
    parser.add_argument("--scaler-path", default=None, help="Optional direct scaler path.")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "realtime_backtest"), help="Output directory.")
    parser.add_argument("--sheet-path", default=str(ROOT / "reports" / "backtest_run_sheet.csv"), help="CSV sheet updated after each run.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    config = load_config(args.config)
    run_cost_model = cost_model_metadata(config)
    candles, adapter, replay, frame = build_realtime_components(
        config,
        experiment_dir=args.experiment_dir,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
    )
    result = RealtimeBacktestSimulator(replay).run(candles, adapter)
    evaluation_start = str(candles.timestamp_str[0]) if len(candles.timestamp_str) else None
    evaluation_end = str(candles.timestamp_str[-1]) if len(candles.timestamp_str) else None
    diagnostics = backtest_diagnostics(result.trades_frame, config, evaluation_start, evaluation_end)
    expert_metrics = expert_trade_metrics(result.trades_frame)
    split = make_time_split(frame, config.data.validation_ratio, config.data.test_ratio, config.data.embargo_bars)

    def _split_metrics(partition_frame) -> dict:
        if result.trades_frame.empty:
            part_trades = result.trades_frame.copy()
        else:
            start_ts = pd.Timestamp(partition_frame["timestamp"].iloc[0]).tz_convert("UTC")
            end_ts = pd.Timestamp(partition_frame["timestamp"].iloc[-1]).tz_convert("UTC")
            trade_ts = pd.to_datetime(result.trades_frame["timestamp"], utc=True, errors="coerce")
            mask = (trade_ts >= start_ts) & (trade_ts <= end_ts)
            part_trades = result.trades_frame.loc[mask]
        return {
            "trade_metrics": trade_metrics(part_trades),
            "routed_usage_gate": routed_usage_gate(part_trades, config),
        }

    split_breakdown = {
        "train": _split_metrics(split.train),
        "validation": _split_metrics(split.validation),
        "test": _split_metrics(split.test),
    }
    routed_gate = routed_usage_gate(result.trades_frame, config)
    detailed_metrics = {
        "summary": result.summary,
        "diagnostics": diagnostics,
        "routed_usage_gate": routed_gate,
        "split_breakdown": split_breakdown,
        "expert_metrics": expert_metrics,
        "performance": result.performance,
        "metadata": make_run_metadata(
            config_name=str(args.config),
            experiment_name=config.experiment.name,
            output_dir=str(args.output_dir),
            model_path=str(args.model_path) if args.model_path is not None else None,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            **run_cost_model,
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "backtest_summary.json").write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    (output_dir / "backtest_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (output_dir / "expert_trade_metrics.json").write_text(json.dumps(expert_metrics, indent=2), encoding="utf-8")
    (output_dir / "backtest_detailed_metrics.json").write_text(json.dumps(detailed_metrics, indent=2), encoding="utf-8")
    (output_dir / "backtest_performance.json").write_text(json.dumps(result.performance, indent=2), encoding="utf-8")
    np.save(output_dir / "equity_curve.npy", result.equity_curve)
    result.trades_frame.to_csv(output_dir / "trades.csv", index=False)
    if expert_metrics:
        pd.DataFrame(expert_metrics).to_csv(output_dir / "expert_trade_metrics.csv", index=False)
    row = {
        **make_run_metadata(
            config_name=str(args.config),
            experiment_name=config.experiment.name,
            output_dir=str(output_dir),
            model_path=str(args.model_path) if args.model_path is not None else None,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            **run_cost_model,
        ),
        **flatten_for_sheet(result.summary, "summary"),
        **flatten_for_sheet(diagnostics, "diag"),
        **flatten_for_sheet(routed_gate, "gate"),
        **flatten_for_sheet(result.performance, "perf"),
    }
    append_run_sheet(row, args.sheet_path)

    print(
        "Realtime backtest complete: "
        f"trades={result.summary.get('num_trades', 0)}, "
        f"win_rate={result.summary.get('win_rate', 0.0):.4f}, "
        f"ending_equity_r={result.summary.get('ending_equity_r', 0.0):.4f}, "
        f"bars_per_second={result.performance.get('bars_per_second', 0.0):.2f}"
    )
