# Latest Backtest Comparison

Baseline: configs/dev_quality.yaml (baseline_pre_fix)
Fresh run: configs/dev_quality_postfix.yaml

| metric | baseline | fresh_run | delta |
|---|---:|---:|---:|
| summary_num_trades | 23 | 295 | 272 |
| summary_win_rate | 0.2608695652173913 | 0.3864406779661017 | 0.1255711127487104 |
| summary_expectancy_r | 0.4781108695652174 | -1.778013977057614e-06 | -0.47811264757919447 |
| summary_profit_factor | 11.968580120692232 | 0.5956149826712435 | -11.372965138020989 |
| summary_net_profit_r | 10.99655 | -0.0005245141232319 | -10.99707451412323 |
| summary_max_drawdown_r | -1.00015 | -0.0005657877423118 | 0.9995842122576883 |
| summary_longest_win_streak | 3 | 5 | 2 |
| summary_longest_losing_streak | 14 | 9 | -5 |
| diag_daily_win_rate | 1.0 | 0.0 | -1.0 |
| diag_weekly_win_rate | 1.0 | 0.0 | -1.0 |
| diag_monthly_win_rate | 1.0 | 0.0 | -1.0 |
| diag_average_winning_day_profit_r | 10.99655 | 0.0 | -10.99655 |
| diag_average_losing_day_loss_r | 0.0 | -0.0001049028246463 | -0.0001049028246463 |