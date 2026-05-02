# Volume-Confirmed First-Candle Breakout Sweep — Status

## Goal

Test whether the first 5-min or 15-min candle closing above (below) the prior day's high (low) with a clear volume spike is exploitable as a stand-alone intraday momentum strategy on the 10 large-caps with 5-min intraday data.

## Universe & Period

- Stocks: RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, BHARTIARTL, ITC, KOTAKBANK, HINDUNILVR
- Period: 2024-03-01 to 2026-03-25 (locked, same as research/29)
- Data: market_data_unified, 5minute timeframe; 15-min built from 09:15+09:20+09:25 5-min bars

## Variant Grid

| Axis | Values |
|---|---|
| timeframe | ['5min', '15min'] |
| vol_mult | [1.5, 2.0, 3.0] (multiplier of 20-session first-bar avg) |
| gap_pct | [0.0, 0.003, 0.005, None] (None = filter off; positive = min |gap| ratio) |
| use_rsi | [False, True] (5-min RSI(14) at signal_time; long>=60, short<=40) |
| direction | ['long', 'short'] |

Total cells: 10 stocks × 2 tf × 3 vm × 4 gap × 2 rsi × 2 dir = 960

## Exit Policies (13)

All policies are evaluated in parallel for every signal:

1. **T_NO** — hold to 15:25 IST
2. **T_HARD_SL** — fixed SL at first-bar opposite extreme
3. **T_ATR_SL_{0.3, 0.5, 1.0}** — fixed SL = entry − k × daily ATR(14)
4. **T_CHANDELIER_{1.0, 1.5, 2.0}** — trail SL = running extreme − k × ATR
5. **T_R_TARGET_{1, 1.5, 2, 3}R** — target = x × hard_SL_distance, else hard SL
6. **T_STEP_TRAIL** — at +0.5R move SL to entry; at +1.5R to +0.5R; at +3R to +1.5R

## Status

- COMPLETED. 11412 signals across all cells.
- Outputs:
  - `results/volbreakout_signals.csv`
  - `results/volbreakout_ranking.csv`
  - `results/RESULTS.md`

## Crash Recovery

This script is **resumable**. To resume:

1. Inspect `results/volbreakout_signals.csv` — count rows by (symbol, timeframe, variant, direction, date) tuples.
2. Re-run `python research/30_volume_breakout/scripts/run_volbreakout.py`. Already-logged tuples are skipped automatically.
3. After all signals are logged, the same script aggregates and writes `volbreakout_ranking.csv` + `RESULTS.md`.

## Final Aggregation

Per-cell ranking is on the cross of (symbol × timeframe × variant × direction × exit_policy). Primary rank = `sharpe_score` = `(mean_net_pct / std_net_pct) × win_rate_fraction`. Secondary = expectancy_pct.
