"""Label audit script
Generates per‑expert label quality metrics:
- precision (win / valid)
- win rate (wins / valid)
- expectancy (net_return_r mean on valid rows)
- direction balance (long vs short counts)
The script reads a CSV of pre‑processed OHLC data with the generated labels
(`generate_labels` output) and prints a markdown table. It is intended to be
run before any model training to decide which setups have stand‑alone edge.
"""

import argparse
import pathlib
import sys

import pandas as pd


def compute_metrics(df: pd.DataFrame, asset: str, setup: str):
    prefix = asset.lower()
    valid_col = f"{prefix}_{setup}_valid"
    win_col = f"{prefix}_{setup}_target"
    net_ret_col = f"{prefix}_{setup}_net_return_r"
    direction_col = f"{prefix}_{setup}_direction"
    if valid_col not in df.columns:
        return None
    valid = df[valid_col].astype(bool)
    if valid.sum() == 0:
        return {
            "valid": 0,
            "win_rate": 0.0,
            "precision": 0.0,
            "expectancy": 0.0,
            "long_pct": 0.0,
            "short_pct": 0.0,
        }
    wins = df.loc[valid, win_col].sum()
    precision = wins / valid.sum()
    expectancy = df.loc[valid, net_ret_col].mean()
    # direction balance
    direction = df.loc[valid, direction_col]
    long_pct = (direction == 1).mean()
    short_pct = (direction == -1).mean()
    return {
        "valid": int(valid.sum()),
        "win_rate": precision,
        "precision": precision,
        "expectancy": expectancy,
        "long_pct": long_pct,
        "short_pct": short_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit labeling quality")
    parser.add_argument("data_path", type=pathlib.Path, help="Path to CSV with generated labels")
    args = parser.parse_args()
    if not args.data_path.is_file():
        sys.exit(f"File not found: {args.data_path}")
    df = pd.read_csv(args.data_path)
    assets = ["US100", "US500"]
    setups = [
        "trend_continuation",
        "pullback_continuation",
        "breakout_expansion",
        "mean_reversion",
        "liquidity_sweep_reversal",
        "volatility_compression_expansion",
        "session_open_momentum",
        "exhaustion_failure",
    ]
    rows = []
    for asset in assets:
        for setup in setups:
            metrics = compute_metrics(df, asset, setup)
            if metrics is None:
                continue
            rows.append({
                "asset": asset,
                "setup": setup,
                **metrics,
            })
    report_df = pd.DataFrame(rows)
    # output markdown table
    print("| Asset | Setup | Valid | Win Rate | Expectancy | Long % | Short % |")
    print("|---|---|---|---|---|---|---|")
    for _, row in report_df.iterrows():
        print(f"| {row['asset']} | {row['setup']} | {row['valid']} | {row['win_rate']:.2%} | {row['expectancy']:.4f} | {row['long_pct']:.2%} | {row['short_pct']:.2%} |")


if __name__ == "__main__":
    main()
