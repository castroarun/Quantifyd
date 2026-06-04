# research/57 G3a (FIXED) — intraday move-stop frequency (recipe move1.5+PT40, 26 trades)

Stop+PT checked intraday at each tf (global-minute throttle). Exit = straddle premium at trigger minute. Net Rs80/leg. **30d SIGNAL.**

| check tf | mean | median | win% | worst | std |
|---|---|---|---|---|---|
| 1-min (continuous) | +2956 | +100 | +50 | -4736 | 6647 |
| 5-min | +2929 | -165 | +50 | -4632 | 6606 |
| 10-min | +3413 | +32 | +50 | -4632 | 7116 |
| 15-min | +3754 | +1187 | +62 | -4856 | 6966 |
| EOD 15:20 only | +7133 | +3586 | +69 | -4995 | 9234 |

## Read
- Finer tf fires nearer the 1.5% line (less overshoot) -> smaller worst; too fine may whipsaw.
- vs EOD-only: shows how much the intraday stop improves the tail. 30d SIGNAL.