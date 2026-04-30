# Volume-Breakout Sweep (EXPANDED, 79 stocks) — Status

## Goal

Run the volume-confirmed first-candle breakout signal generator (research/30) across the full 79-stock universe with 5-min intraday data, including a NEW 10-min timeframe, to find the universe-wide volume leaders.

## Universe

- Cohort A (10 stocks since 2018-01-01): RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, BHARTIARTL, ITC, KOTAKBANK, HINDUNILVR
- Cohort B (69 stocks since 2024-03-18): ADANIENT, ADANIPORTS, AMBUJACEM, APOLLOHOSP, ASIANPAINT, AXISBANK, BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BANKBARODA, BEL, BPCL, BRITANNIA, CHOLAFIN, CIPLA, COALINDIA, COFORGE, COLPAL, CUMMINSIND, DABUR, DELHIVERY, DIVISLAB, DLF, DRREDDY, EICHERMOT, FEDERALBNK, GAIL, GODREJPROP, GRASIM, HAL, HAVELLS, HCLTECH, HDFCLIFE, HEROMOTOCO, HINDALCO, IDFCFIRSTB, INDUSINDBK, IOC, IRCTC, JINDALSTEL, JSWSTEEL, LT, M&M, MARICO, MARUTI, MCX, MUTHOOTFIN, NESTLEIND, NTPC, ONGC, PAYTM, PERSISTENT, PIDILITIND, PNB, POWERGRID, SBILIFE, SHREECEM, SIEMENS, SUNPHARMA, TATACONSUM, TATAPOWER, TATASTEEL, TECHM, TITAN, TRENT, ULTRACEMCO, VEDL, VOLTAS, WIPRO

## Variant Grid

- timeframes: ['5min', '10min', '15min']
- vol_mult: [1.5, 2.0, 3.0]
- gap_pct: [0.0, 0.003, 0.005, None]
- rsi_modes: [False, True]
- directions: ['long', 'short']
- Cells per stock: 144
- Total cells: 11376

## Status

- State: **RUNNING**
- Started: 2026-04-30 12:10:59
- Last completed stock: BANKBARODA
- Stocks completed: 10 / 79
- Signals logged: 15,695

## Crash Recovery

Script is **resumable**. To resume:

```bash
python research/30b_volume_breakout_expanded/scripts/run_volbreakout_expanded.py
```

It reads `results/volbreakout_signals.csv`, builds a (symbol, tf, variant, direction, date) skip-set, and only computes signals not yet logged.

If aggregation/markdown-write was the only step that failed, run with `--aggregate-only`.

## Outputs

- `results/volbreakout_signals.csv` (per-signal x exit-policy, gitignored)
- `results/volbreakout_ranking.csv` (per-cell summary)
- `results/volume_leaders.csv` (per-stock leaderboard)
- `results/RESULTS.md` (final report)

