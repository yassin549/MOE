# Dukascopy US100 / US500 Data Report

Source: Dukascopy minute bars downloaded for 2021-04-27 through 2026-04-28 (exclusive end date).
Instruments: US100 (usatechidxusd), US500 (usa500idxusd)

## Cleaning policy
- Removed malformed OHLCV rows.
- Removed zero-volume rows.
- Removed duplicate timestamps.
- Preserved genuine market-session gaps and exposed them with `minutes_since_prev_bar` and `is_gap_after_prev_bar`.
- Enriched each bar with UTC calendar fields for direct model feature engineering.

## US100
- Clean rows: 1709909
- Raw rows read: 1709909
- Duplicates dropped: 0
- Invalid rows dropped: 0
- Zero-volume rows dropped: 0
- Trading days: 1557
- Bars per trading day: avg=1098.21, median=1330.00, min=15, max=1335
- Gaps: count=1445, avg_minutes=637.60, median_minutes=106.00, max_minutes=4432

## US100 Hourly Profile (UTC)
- 10: bars=77280, avg_abs_return_pct=0.015852, avg_volume=0.055911
- 11: bars=77317, avg_abs_return_pct=0.017034, avg_volume=0.058650
- 12: bars=77293, avg_abs_return_pct=0.021676, avg_volume=0.074334
- 13: bars=77263, avg_abs_return_pct=0.035114, avg_volume=0.118631
- 14: bars=77268, avg_abs_return_pct=0.044293, avg_volume=0.153416
- 15: bars=77228, avg_abs_return_pct=0.039249, avg_volume=0.148328
- 16: bars=77236, avg_abs_return_pct=0.031783, avg_volume=0.132240
- 17: bars=75921, avg_abs_return_pct=0.029637, avg_volume=0.124733
- 18: bars=74779, avg_abs_return_pct=0.030528, avg_volume=0.126314
- 19: bars=74689, avg_abs_return_pct=0.032852, avg_volume=0.133002
- 20: bars=37476, avg_abs_return_pct=0.030954, avg_volume=0.114394
- 21: bars=6263, avg_abs_return_pct=0.024997, avg_volume=0.075546
- 22: bars=49102, avg_abs_return_pct=0.015600, avg_volume=0.026844
- 23: bars=77260, avg_abs_return_pct=0.013199, avg_volume=0.030439
- 00: bars=77422, avg_abs_return_pct=0.014291, avg_volume=0.049455
- 01: bars=77452, avg_abs_return_pct=0.014099, avg_volume=0.056130
- 02: bars=77444, avg_abs_return_pct=0.011369, avg_volume=0.045391
- 03: bars=77331, avg_abs_return_pct=0.010071, avg_volume=0.036053
- 04: bars=77377, avg_abs_return_pct=0.008841, avg_volume=0.027880
- 05: bars=77339, avg_abs_return_pct=0.011516, avg_volume=0.040977
- 06: bars=77269, avg_abs_return_pct=0.013677, avg_volume=0.044966
- 07: bars=77274, avg_abs_return_pct=0.018644, avg_volume=0.071253
- 08: bars=77334, avg_abs_return_pct=0.019426, avg_volume=0.071835
- 09: bars=77292, avg_abs_return_pct=0.016612, avg_volume=0.060664

## US100 Weekday Profile
- Tuesday: bars=346503, avg_abs_return_pct=0.020890, avg_volume=0.078772
- Wednesday: bars=344772, avg_abs_return_pct=0.021299, avg_volume=0.079209
- Thursday: bars=342897, avg_abs_return_pct=0.022045, avg_volume=0.079302
- Friday: bars=313800, avg_abs_return_pct=0.022812, avg_volume=0.083419
- Sunday: bars=25201, avg_abs_return_pct=0.020802, avg_volume=0.036255
- Monday: bars=336736, avg_abs_return_pct=0.020254, avg_volume=0.073388

## US100 Monthly Profile
- April: bars=138335, avg_abs_return_pct=0.026308, avg_volume=0.062228
- May: bars=146867, avg_abs_return_pct=0.022259, avg_volume=0.071090
- June: bars=141889, avg_abs_return_pct=0.019597, avg_volume=0.080653
- July: bars=143636, avg_abs_return_pct=0.017911, avg_volume=0.080670
- August: bars=147862, avg_abs_return_pct=0.019389, avg_volume=0.084096
- September: bars=142920, avg_abs_return_pct=0.019669, avg_volume=0.094380
- October: bars=146758, avg_abs_return_pct=0.021911, avg_volume=0.105326
- November: bars=139940, avg_abs_return_pct=0.019639, avg_volume=0.084897
- December: bars=140829, avg_abs_return_pct=0.018476, avg_volume=0.067664
- January: bars=141484, avg_abs_return_pct=0.022604, avg_volume=0.069061
- February: bars=133844, avg_abs_return_pct=0.023260, avg_volume=0.063096
- March: bars=145545, avg_abs_return_pct=0.026294, avg_volume=0.072232

## US500
- Clean rows: 1683883
- Raw rows read: 1683883
- Duplicates dropped: 0
- Invalid rows dropped: 0
- Zero-volume rows dropped: 0
- Trading days: 1557
- Bars per trading day: avg=1081.49, median=1296.00, min=15, max=1335
- Gaps: count=23534, avg_minutes=41.20, median_minutes=2.00, max_minutes=4427

## US500 Hourly Profile (UTC)
- 10: bars=76662, avg_abs_return_pct=0.012530, avg_volume=0.070162
- 11: bars=76716, avg_abs_return_pct=0.013330, avg_volume=0.071608
- 12: bars=76986, avg_abs_return_pct=0.016828, avg_volume=0.093164
- 13: bars=77170, avg_abs_return_pct=0.024753, avg_volume=0.214801
- 14: bars=77208, avg_abs_return_pct=0.030795, avg_volume=0.333111
- 15: bars=77088, avg_abs_return_pct=0.028231, avg_volume=0.308487
- 16: bars=77104, avg_abs_return_pct=0.023588, avg_volume=0.243223
- 17: bars=75836, avg_abs_return_pct=0.022666, avg_volume=0.223085
- 18: bars=74725, avg_abs_return_pct=0.023859, avg_volume=0.233245
- 19: bars=74688, avg_abs_return_pct=0.026641, avg_volume=0.256546
- 20: bars=37386, avg_abs_return_pct=0.024738, avg_volume=0.199594
- 21: bars=6244, avg_abs_return_pct=0.018222, avg_volume=0.086991
- 22: bars=47625, avg_abs_return_pct=0.013179, avg_volume=0.026414
- 23: bars=73900, avg_abs_return_pct=0.010711, avg_volume=0.029258
- 00: bars=76552, avg_abs_return_pct=0.011392, avg_volume=0.045070
- 01: bars=76778, avg_abs_return_pct=0.011036, avg_volume=0.048587
- 02: bars=75320, avg_abs_return_pct=0.008930, avg_volume=0.036422
- 03: bars=73318, avg_abs_return_pct=0.008071, avg_volume=0.029007
- 04: bars=70539, avg_abs_return_pct=0.007268, avg_volume=0.022155
- 05: bars=74842, avg_abs_return_pct=0.009037, avg_volume=0.032987
- 06: bars=76197, avg_abs_return_pct=0.010798, avg_volume=0.048370
- 07: bars=77009, avg_abs_return_pct=0.014880, avg_volume=0.095851
- 08: bars=77140, avg_abs_return_pct=0.015268, avg_volume=0.102189
- 09: bars=76850, avg_abs_return_pct=0.013029, avg_volume=0.078704

## US500 Weekday Profile
- Tuesday: bars=341097, avg_abs_return_pct=0.016088, avg_volume=0.120607
- Wednesday: bars=339153, avg_abs_return_pct=0.016351, avg_volume=0.122975
- Thursday: bars=337988, avg_abs_return_pct=0.016939, avg_volume=0.128975
- Friday: bars=309721, avg_abs_return_pct=0.017707, avg_volume=0.138120
- Sunday: bars=24992, avg_abs_return_pct=0.016680, avg_volume=0.042311
- Monday: bars=330932, avg_abs_return_pct=0.015628, avg_volume=0.116680

## US500 Monthly Profile
- April: bars=136603, avg_abs_return_pct=0.021044, avg_volume=0.174406
- May: bars=144314, avg_abs_return_pct=0.017404, avg_volume=0.137023
- June: bars=138808, avg_abs_return_pct=0.015233, avg_volume=0.096562
- July: bars=140261, avg_abs_return_pct=0.013390, avg_volume=0.096726
- August: bars=145367, avg_abs_return_pct=0.014436, avg_volume=0.110679
- September: bars=140582, avg_abs_return_pct=0.015647, avg_volume=0.124469
- October: bars=145614, avg_abs_return_pct=0.017324, avg_volume=0.107766
- November: bars=137617, avg_abs_return_pct=0.014998, avg_volume=0.099736
- December: bars=138044, avg_abs_return_pct=0.014011, avg_volume=0.092208
- January: bars=140096, avg_abs_return_pct=0.016542, avg_volume=0.128257
- February: bars=132003, avg_abs_return_pct=0.017263, avg_volume=0.140026
- March: bars=144574, avg_abs_return_pct=0.021009, avg_volume=0.180757
