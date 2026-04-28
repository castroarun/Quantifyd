# Path C on Stocks — Compression Breakout Backtest
Same logic as NIFTY Path C: post-12:00 day-extreme break + range compression filter. Applied to the 10 stocks with 5-min intraday data.

- Period: 2024-03-01 to 2026-03-25
- Universe: RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, BHARTIARTL, ITC, KOTAKBANK, HINDUNILVR
- Timeframes: 5/10/15-min
- Variant grid: range_threshold ∈ {0.4%, 0.6%, 0.8%, 1.0%, off} × RSI ∈ {on, off}
- Total signals fired: **20730**
- Ranked configurations (n>=5): **430**

## Top 10 by Sharpe-style score (n>=20, mean>0)

| symbol | tf | variant | policy | n | mean | std | WR% | Sharpe |
|---|---|---|---|---|---|---|---|---|
| ITC | 10min | rng0.008_norsi | T0 | 43 | 0.453 | 1.641 | 60.5 | 0.1670 |
| ITC | 10min | rng0.008_norsi | T1_SL_1xATR | 43 | 0.453 | 1.641 | 60.5 | 0.1670 |
| ITC | 10min | rng0.008_rsi40_60 | T0 | 40 | 0.285 | 1.309 | 60.0 | 0.1307 |
| ITC | 10min | rng0.008_rsi40_60 | T1_SL_1xATR | 40 | 0.285 | 1.309 | 60.0 | 0.1307 |
| ITC | 5min | rng0.008_norsi | T0 | 49 | 0.351 | 1.539 | 55.1 | 0.1257 |
| ITC | 5min | rng0.008_norsi | T1_SL_1xATR | 49 | 0.351 | 1.539 | 55.1 | 0.1257 |
| ITC | 5min | rng0.008_rsi40_60 | T0 | 49 | 0.351 | 1.539 | 55.1 | 0.1257 |
| ITC | 5min | rng0.008_rsi40_60 | T1_SL_1xATR | 49 | 0.351 | 1.539 | 55.1 | 0.1257 |
| ITC | 15min | rng0.008_norsi | T0 | 40 | 0.217 | 1.330 | 55.0 | 0.0900 |
| ITC | 15min | rng0.008_norsi | T1_SL_1xATR | 40 | 0.217 | 1.330 | 55.0 | 0.0900 |

## Best variant per stock (T0, n>=10)

| symbol | tf | variant | n | mean | std | WR% | Sharpe |
|---|---|---|---|---|---|---|---|
| RELIANCE | 10min | rng0.010_norsi | 91 | 0.856 | 7.000 | 47.3 | 0.0578 |
| TCS | 10min | rngoff_norsi | 225 | 0.140 | 22.806 | 48.4 | 0.0030 |
| HDFCBANK | 15min | rng0.006_norsi | 11 | 0.818 | 3.514 | 45.5 | 0.1058 |
| INFY | 5min | rng0.008_rsi40_60 | 16 | 1.303 | 6.016 | 56.2 | 0.1218 |
| ICICIBANK | 15min | rng0.010_rsi40_60 | 68 | 0.427 | 5.078 | 57.4 | 0.0483 |
| SBIN | 10min | rng0.010_rsi40_60 | 68 | 0.526 | 4.074 | 50.0 | 0.0645 |
| BHARTIARTL | 5min | rng0.008_norsi | 19 | -0.308 | 7.711 | 42.1 | -0.0168 |
| ITC | 10min | rng0.008_norsi | 43 | 0.453 | 1.641 | 60.5 | 0.1670 |
| KOTAKBANK | 10min | rngoff_norsi | 239 | -0.238 | 8.741 | 48.1 | -0.0131 |
| HINDUNILVR | 10min | rng0.008_rsi40_60 | 34 | 0.063 | 8.442 | 47.1 | 0.0035 |
