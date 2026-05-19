# Carry-Forward 75% WR Quest — Patterns 1, 2, 6 Results

**Date:** 2026-05-07
**Universe:** F&O 80 stocks (78 with usable daily history)
**Train:** 2018-01-01 to 2023-12-31
**Test:**  2024-01-01 to 2025-11-27 (effective end — daily data ends ~Nov 2025)
**Cost:** F&O 0.06% round-trip default; CNC stress 0.20% round-trip

## TL;DR — No walk-forward winner cleared the gates

Across **74,880 BTST + 116,832 swing + 22,400 weekly = 214,112 ranking rows** swept,
**zero (symbol, direction, config, params) tuples** cleared the hard gates of:
- train WR ≥ 75% AND test WR ≥ 70%
- test PF ≥ 1.8
- test MaxDD ≤ 15%
- test n ≥ 30 (15 for weekly)

The structural ceiling found in research/37 (intraday) extends to overnight and
multi-day daily-bar systems on the F&O cohort. The 75% WR + favorable RR target
appears non-achievable with traditional momentum / breakout / trend-continuation
patterns on liquid Indian equity at the daily timeframe.

## Top 3 candidates per pattern (best train+test combined, ignoring 75% gate)

### BTST (1-night hold, close-to-close)

| Symbol | Direction | Config | Train WR | Test WR | Test PF | Test ret |
|---|---|---|---|---|---|---|
| RELIANCE | long | NOTPSL cp=0.75 vr=1.0 retmin=0.0 up=True | 62.8% (n=137) | 59.4% (n=32) | 0.90 | -1.8% |
| TATAPOWER | long | NOTPSL cp=0.75 vr=1.0 retmin=0.0 up=False | 60.6% (n=160) | 62.0% (n=50) | 1.84 | +21.4% |
| ADANIENT | long | NOTPSL cp=0.75 vr=1.0 retmin=0.0 up=False | 62.2% (n=180) | 55.0% (n=40) | 1.78 | +20.3% |

Best test-WR (but train was much weaker): SIEMENS long 68.2% test (n=44), train 49.6%.

### Swing breakout (Donchian + 50-EMA-rising + volume)

| Symbol | Direction | Config | Train WR | Test WR | Test PF | Test ret |
|---|---|---|---|---|---|---|
| M&M | long | D20 TP20_SL15 H10 vr=1.2 sma200=True | 52.0% (n=50) | 60.0% (n=20) | 1.87 | +10.8% |
| BPCL | long | D20 TP30_SL15 H3 vr=1.2 sma200=False | 49.0% (n=49) | 60.0% (n=20) | 1.91 | +11.4% |
| BEL | long | D20 TP40_SL20 H5 vr=1.2 sma200=True | 46.7% (n=45) | 55.0% (n=20) | 2.07 | +18.0% |

Best swing breakout WR ceiling: ~60% test, ~52% train. Far below the 75% gate.

### Weekly continuation (8-week Donchian on weekly close)

| Symbol | Direction | Config | Train WR | Test WR | Test PF | Test ret |
|---|---|---|---|---|---|---|
| MCX | long | W8 TP20_SL15 H5 | 58.1% (n=31) | 65.2% (n=23) | 2.33 | +16.6% |
| MCX | long | W12 TP20_SL15 H10 | 60.7% (n=28) | 61.9% (n=21) | 2.02 | +12.7% |
| RELIANCE | short | W12 TP30_SL15 H5 | 66.7% (n=12) | 60.0% (n=10) | 2.59 | +9.9% |

MCX weekly is the most consistent — train ~58-61%, test 62-65%. Still below 75%.

## Hard-gate report

| Pattern | Strong (train≥70 AND test≥65, n≥20-30) | Walk-forward winners |
|---|---|---|
| BTST | 0 | 0 |
| Swing breakout | 0 | 0 |
| Weekly continuation | 0 | 0 |

## Cost-stress test (0.20% round-trip CNC delivery)

For top-20 setups by AWS at default 0.06% cost, count of those still positive at 0.20%:

| Pattern | Top-20 still positive at stress | Note |
|---|---|---|
| BTST | **20/20** | All MARUTI long variants — BTST is cost-resilient |
| Swing | 0/20 | Multi-day holds with TP/SL get bled by 0.20% per trade |
| Weekly | 0/20 | Same — long holds + larger spreads kill it |

BTST is the only pattern where top setups survive the CNC cost level. Net mean return:
- BTST: +1.03% default → -4.41% stress (top setups still +15-22% net)
- Swing: -0.60% default → -3.53% stress (no winners survive)
- Weekly: +3.74% default → +0.65% stress (positive on average but no winners survive top-20)

## Surviving stock cohorts

`results/01_btst_diamonds.txt` (30 stocks, top by AWS, fallback): includes RELIANCE,
TCS, HDFCBANK, ICICIBANK, KOTAKBANK, LT, AXISBANK, MARUTI, BAJFINANCE, HCLTECH,
SUNPHARMA, BHARTIARTL, INFY, SBIN, ITC, HINDUNILVR, TITAN, ASIANPAINT, INDUSINDBK,
NESTLEIND, ULTRACEMCO, WIPRO, BAJAJFINSV, HINDALCO, COALINDIA, GRASIM, TATACONSUM,
DRREDDY, M&M, TECHM.

`results/02_swing_breakout_diamonds.txt` (3 stocks, fallback): M&M, BPCL, BEL.

`results/06_weekly_trend_diamonds.txt` (24 stocks, fallback): includes MCX,
SIEMENS, RELIANCE, INDUSINDBK, AXISBANK, etc.

These are the cohorts where the patterns showed *some* edge in test, even if
below the 75% gate.

## Honest caveats

1. **Test window is shorter than expected** — most F&O daily data ends ~Nov 2025
   (not 2026-03 as the brief assumed). Test n is constrained.
2. **Daily-bar BTST is too crude** — true BTST entry needs intraday last-hour
   strength (5-min close > VWAP, near day high) which we approximated only via
   `close_pos`. A re-test using 5-min last-hour signal generation on the 10
   stocks with full 5-min history (RELIANCE, TCS, HDFCBANK, etc., 2018+) is the
   logical next step.
3. **Donchian-20 swing breakout fires too rarely on liquid F&O names** — most
   stocks in the cohort had 30-50 train signals, 10-20 test signals. Either the
   universe is wrong (try mid/small caps) or the signal is too restrictive.
4. **Weekly 8-week breakout is too rare** — most stocks have <30 train trades
   total. Expanding to weekly RSI / MACD / momentum patterns would give more
   sample.
5. **No regime overlay tested in detail** — index uptrend gating gave a small
   bump (BTST `up=True` setups slightly better) but didn't crack 70% test WR.

## Recommended next steps

1. **5-min driven BTST on the 10 long-history stocks** — use research/37
   _engine.py's last-hour 5-min features for entry timing. Could reach 75% on
   that narrow universe.
2. **Wider universe for swing breakout** — Nifty 500 mid-caps with weak liquidity
   may have more consolidation breakouts; the F&O cohort is too well-arbitraged.
3. **Combine pattern stacking** — BTST signal + weekly trend gate + sector
   leadership. Singletons cap at 60-65% test WR.
4. **Look at SHORT side specifically** — shorts on weekly D-low breaks at 12-week
   lookback showed train 66-68% WR (RELIANCE, INDUSINDBK) but n is tiny.
5. **Earnings post-drift** (pattern 4 in STATUS doc) — high-WR results in academic
   literature; needs an earnings calendar dataset.

## Files

| File | Purpose |
|---|---|
| `scripts/_engine_daily.py` | Daily-bar engine: loaders + indicators + simulator |
| `scripts/01_btst.py` | BTST sweep (resumable) |
| `scripts/02_swing_breakout.py` | Donchian swing sweep (resumable) |
| `scripts/06_weekly_trend.py` | Weekly continuation sweep |
| `results/01_btst_perstock.csv` | Per-stock BTST baseline |
| `results/01_btst_ranking.csv` | 74,880-row BTST sweep |
| `results/01_btst_walk_forward.csv` | Walk-forward filter (empty — no winners) |
| `results/01_btst_diamonds.txt` | 30 stocks fallback cohort |
| `results/02_swing_breakout_*` | Swing equivalents (116,832 rows) |
| `results/06_weekly_trend_*` | Weekly equivalents (22,400 rows) |
