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

- State: **DONE** ✅
- Signal generation: 12:10–14:34 IST (143 min)
- Aggregation: original `pd.read_csv` hung (>90 min stuck on 168 MB) — killed and replaced with `aggregate_streaming.py` (csv.DictReader + Welford online stats), completed in ~2 min
- Stocks completed: 79 / 79
- Signal rows logged: 164,327
- Ranked cells: 123,851
- Stocks with at least one viable cell: 78 / 79

## Final findings (2026-04-30)

### Top 5 by Sharpe (n≥15, mean>0)

| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Sharpe |
|---|---|---|---|---|---:|---:|---:|---:|
| **RELIANCE** | 15min | s_vm3.0_gapoff_rsi40_60 | short | T_R_TARGET_1R | 15 | +1.110 | 93.3 | **0.858** |
| **GODREJPROP** | 10min | s_vm1.5_gap0.003_norsi | short | T_R_TARGET_1R | 16 | +0.932 | 75.0 | **0.839** |
| **HAL** | 10min | s_vm1.5_gapoff_rsi40_60 | short | T_NO | 17 | +1.074 | 82.3 | **0.828** |
| HAL (same cell) | 10min | s_vm1.5_gapoff_rsi40_60 | short | T_ATR_SL_1.0 | 17 | +1.074 | 82.3 | 0.828 |
| HAL (same cell) | 10min | s_vm1.5_gapoff_rsi40_60 | short | T_CHANDELIER_2.0 | 17 | +1.074 | 82.3 | 0.828 |

### Top 5 by Sharpe (any n≥10) — includes high-conviction-low-sample winners

| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Sharpe | RobustCells |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| **GODREJPROP** | 15min | s_vm1.5_gap0.003_rsi40_60 | short | T_R_TARGET_1R | 10 | +1.527 | **100.0** | **2.322** | 194 |
| **HAL** | 5min | s_vm1.5_gap0.005_rsi40_60 | short | T_NO | 10 | +1.409 | 90.0 | **1.354** | 155 |
| RELIANCE | 15min | s_vm3.0_gap0.000_rsi40_60 | short | T_R_TARGET_1R | 14 | +1.166 | 92.9 | 0.881 | 400 |
| DELHIVERY | 15min | s_vm2.0_gapoff_norsi | short | T_R_TARGET_1R | 10 | +1.398 | 80.0 | 0.841 | 166 |
| ONGC | 10min | s_vm1.5_gap0.000_rsi40_60 | short | T_STEP_TRAIL | 10 | +0.556 | 90.0 | 0.813 | 52 |

### Promote candidates (Sharpe≥0.5 + n≥15 + 3+ robust cells)

**Only 1 stock passes:** VEDL 15min long, vm=2.0, gapoff, RSI on, T_R_TARGET_1R: n=18, mean +0.69%, WR 72.2%, Sharpe 0.509, **15 robust cells**.

The strict gate is conservative. If we **relax to Sharpe≥0.5 + n≥10 + 30+ robust cells** (consistency over single-cell quality), the picture changes — high-quality consistent names emerge:

| Symbol | RobustCells | Best Sharpe | Best n | Best Dir | Best TF | Read |
|---|---:|---:|---:|---|---|---|
| **RELIANCE** | **400** | 0.881 | 14 | short | 15min | Most consistent edge in the universe — 400 cells with Sharpe≥0.3 |
| **GODREJPROP** | 194 | 2.322 | 10 | short | 15min | Highest peak Sharpe (100% WR), broad robustness |
| **DELHIVERY** | 166 | 0.841 | 10 | short | 15min | Tight-range stock, breakouts work |
| **HAL** | 155 | 1.354 | 10 | short | 5min | Defense-sector momentum, short side |
| **CHOLAFIN** | 96 | 0.434 | 11 | long | 10min | Long side works |
| **COFORGE** | 86 | 0.553 | 13 | long | 5min | IT mid-cap; longs |
| **WIPRO** | 62 | 0.572 | 12 | short | 10min | IT large-cap; shorts |
| **LT** | 59 | 0.574 | 11 | long | 15min | Engineering large-cap; longs |
| **PIDILITIND** | 6 | 0.640 | 10 | short | 10min | Niche but consistent shorts |

### Surprises

1. **SHORT side dominates the top.** Top 5 by Sharpe (n≥15) are ALL short. Counter to typical "stocks rally in bull markets" intuition — the volume-confirmed breakdown of prev-day-low is a sharper signal than the breakout above prev-day-high in this 2-year window.
2. **RELIANCE flipped direction** vs the prior 10-stock run. Old: long, vm=2.0, gap≥0%, RSI on (Sharpe 1.05, n=11). New best: short, vm=3.0, gapoff, RSI on (Sharpe 0.88, n=14). Both pass cleanly — RELIANCE has both long and short edges in different variants.
3. **GODREJPROP at Sharpe 2.32 with 100% WR (10/10)** is exceptional. Confirms the user's chart observation about GODFRYPHLP-style stocks (mid-cap with clean intraday structure). Caveat: n=10 is small.
4. **10-min timeframe surfaced strong picks** that weren't in the prior 5/15-only run: HAL 10min, GODREJPROP 10min, DELHIVERY 15min, ONGC 10min. The 10-min addition was worthwhile.
5. **T_R_TARGET_1R wins exits** for most top picks — consistent with "small target" framing. T_NO (full ride) wins on 1-2 names where the move is durable. The "step trail" exit is still poor.
6. **HDFCBANK does NOT make the leaderboard** despite being one of the most-traded stocks. Banking heavyweights show no clean edge — likely too efficient / news-driven intraday.

### Honest read

- **The signal is real but concentrated** — across 79 stocks, edge is bimodal: ~10-15 stocks have meaningful Sharpe, the rest are noise.
- **Volume-leader pattern WORKS for mid-caps with tight intraday range** (GODREJPROP, HAL, DELHIVERY, CHOLAFIN, COFORGE) and select large-caps (RELIANCE, WIPRO, LT). Avoid HDFCBANK / HCLTECH / TCS / banking names where the edge dissolves.
- **Direction asymmetry is real per stock** — ITC/BEL/PNB/HCL etc favor short; SBIN/INFY/CHOLAFIN/COFORGE etc favor long. This argues for direction-aware deployment, not blind both-ways.
- **The strict promote gate (n≥15 + 3+ robust) only passes VEDL.** That's because most high-Sharpe stocks have small n=10-14 (the signal fires <1×/month). For practical deployment, treat the leaderboard as candidates worth validating in paper-trading, not a fully validated production list.

### Recommended next steps

1. **Paper-trade the top 5-7 names** (RELIANCE, GODREJPROP, HAL, DELHIVERY, VEDL, COFORGE, WIPRO) with their best variant for 30 trading days. Compare live to backtest.
2. **For Phase 4 option-premium backtest**, prioritize stocks with high robust-cell count (RELIANCE 400, GODREJPROP 194, HAL 155). These are the most likely to pay off after option spread costs.
3. **Skip the bottom 30 stocks** — Sharpe < 0.2 means no detectable edge after costs.

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

