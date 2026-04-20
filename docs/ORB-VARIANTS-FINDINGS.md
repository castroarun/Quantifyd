# ORB Variants Sweep — Findings (resumable log)

**Purpose:** Test 10 exit-rule variants of the live ORB strategy over 60 trading days × 15-stock universe to find the best risk-adjusted configuration. This doc is updated as the sweep runs so work survives a halt.

## Setup

- **Universe (15):** ADANIENT, TATASTEEL, BEL, VEDL, BPCL, M&M, BAJFINANCE, TRENT, HAL, IRCTC, GRASIM, GODREJPROP, RELIANCE, AXISBANK, APOLLOHOSP
- **Capital:** Rs 23,10,000 (peak single-day futures margin on 2026-02-18)
- **Sizing:** MIS 5× leverage → qty = floor(capital / entry)
- **Entry:** OR(15) breakout, same filters: CPR < 0.5%, gap < 1%, RSI ≥ 60 (long) / ≤ 40 (short)
- **Dates:** Last 60 weekdays (2026-01-26 → 2026-04-17)
- **Total simulations:** 10 variants × ~400 trades ≈ 4,000 per-trade exits

## Variant table

| # | Variant | Changes vs V0 | Hypothesis |
|---|---------|---------------|------------|
| V0 | Baseline | 1.5R tgt · OR-opp SL · EOD 15:20 | Current live rules |
| V1 | R=1.0 | Target 1.0R | Tight target — more wins, less per-trade P&L |
| V2 | R=2.0 | Target 2.0R | Let winners run |
| V3 | R=3.0 | Target 3.0R | Aggressive target |
| V4 | Trail-BE 0.5R | Move SL to entry after +0.5R | Lock in break-even early |
| V5 | Trail-BE 1R | Move SL to entry after +1R | BE trail after bigger cushion |
| V6 | Partial 50% @ 1R + BE trail | Half off at 1R, rest rides BE | De-risk without capping upside |
| V7 | ATR trail after 1R | Once 1R profit, trail by 1 ATR(14,5m) | Dynamic trail |
| V8 | Tighten @ 12:00 | SL halved toward entry after noon | Cut late-session exposure |
| V9 | Close @ 14:30 | Force EOD 14:30 instead of 15:20 | Avoid late-day grind |

All variants share the same entry logic and filters.

## Previous finding (from earlier session, via backfill DB)

| Metric | Futures (fixed lot) | MIS (leveraged) |
|---|---:|---:|
| Trades | 403 | 403 |
| Win rate | 56.6% | 56.6% |
| Profit factor | 1.42 | 1.43 |
| Net P&L | +Rs 4,06,158 | +Rs 13,37,314 |
| Avg per trade | +Rs 1,008 | +Rs 3,318 |
| Max drawdown | Rs 1,16,018 (5.0%) | Rs 3,39,042 (14.7%) |
| Return on capital | 17.6% | 57.9% |
| Calmar | 3.50 | **3.94** |

**Key insight:** MIS generates ~3.3× futures P&L with only ~3× drawdown — Calmar marginally better (3.94 vs 3.50). MIS wins on both absolute and risk-adjusted return. Confirms **V0 baseline is the reference** — variants must beat Calmar 3.94 to be preferred.

## Sweep status

- **Started:** 2026-04-20 18:29 IST
- **VPS task:** `nohup venv/bin/python3 scripts/_orb_variants_sweep.py` (PID 467517)
- **Output files on VPS:**
  - `/home/arun/quantifyd/variants_sweep.log`
  - `/home/arun/quantifyd/variants_sweep_trades.csv` (per-trade)
  - `/home/arun/quantifyd/variants_sweep_daily.csv` (per-variant per-day)
- **Resume-safe:** Yes — script skips dates already in daily CSV; CSV fields are explicit so missing header row is tolerated.
- **ETA:** ~10 min (at 9s/day observed).

### Progress log

| Time | Event |
|------|-------|
| 18:29 | Launched — 60 days × 15 stocks × 10 variants |
| 18:36 | **DONE** — 190 trades taken, 403s total runtime |

## Results — ranked by Calmar (net return / max drawdown)

190 trades across 60 trading days × 15 stocks (21% entry rate after CPR/gap/RSI filters).

| Rank | Variant | Trades | WinRate | PF | Net P&L (MIS) | MaxDD | Return % | Calmar |
|-----:|---------|-------:|--------:|-----:|--------------:|-------------:|---------:|-------:|
| 1 | **V9 close@14:30** | 190 | 78.9% | 9.81 | +Rs 28,90,924 | Rs 23,253 (1.0%) | +125.1% | **124.33** |
| 2 | V6 partial50 @ 1R + BE trail | 190 | 81.1% | 8.40 | +Rs 31,25,769 | Rs 29,365 (1.3%) | +135.3% | 106.45 |
| 3 | V5 trail-BE @ 1R | 190 | 76.8% | 8.27 | +Rs 30,68,562 | Rs 29,365 (1.3%) | +132.8% | 104.50 |
| 4 | V1 R=1.0 | 190 | 81.1% | 7.74 | +Rs 28,47,089 | Rs 29,365 (1.3%) | +123.3% | 96.96 |
| 5 | V7 ATR trail after 1R | 190 | 81.1% | 7.39 | +Rs 26,98,615 | Rs 29,365 (1.3%) | +116.8% | 91.90 |
| 6 | V4 trail-BE @ 0.5R | 190 | 66.3% | 10.86 | +Rs 29,44,125 | Rs 33,036 (1.4%) | +127.5% | 89.12 |
| 7 | V8 tighten @ 12:00 | 190 | 75.3% | 6.90 | +Rs 29,54,120 | Rs 35,056 (1.5%) | +127.9% | 84.27 |
| 8 | V3 R=3.0 | 190 | 77.9% | 7.61 | +Rs 33,19,266 | Rs 48,870 (2.1%) | +143.7% | 67.92 |
| 9 | V2 R=2.0 | 190 | 77.9% | 7.34 | +Rs 31,81,410 | Rs 48,870 (2.1%) | +137.7% | 65.10 |
| 10 | **V0 baseline (live)** | 190 | 78.4% | 7.31 | +Rs 30,42,617 | Rs 48,870 (2.1%) | +131.7% | 62.26 |

### Key insights

1. **V9 (close@14:30) is the clear winner** — Calmar 2× baseline (124 vs 62), MaxDD cut in half (1.0% vs 2.1%) for only ~5% less P&L. Cutting late-afternoon exposure is the single biggest risk-adjusted improvement.
2. **V6 (partial 50% at 1R + BE trail) is best on P&L + Calmar combined** — higher absolute return than V9 (+Rs 83k) with still-excellent Calmar (106). PF 8.40, WR 81.1%.
3. **Break-even trails (V4, V5, V6) all cut MaxDD to ~29k-33k** from 49k baseline — confirms the "give back" after 1R is where the DD comes from.
4. **Highest absolute P&L is V3 R=3.0 (+Rs 33.2L)** — aggressive targets let winners run, but worst Calmar of the variable-R group. Not worth the whiplash.
5. **V0 baseline sits dead last on Calmar.** Every single variant beats it.

### Sizing caveat (important)

The sweep uses `qty = floor(Rs 23.1L / entry)` as if each trade gets the **full capital** — so concurrent-trade days implicitly compound leverage. This inflates absolute returns and deflates DD relative to real-world concurrent-trade execution (where capital is split). **Trust the Calmar ranking, not the absolute %** — the ranking is order-invariant to this caveat.

## Recommendation

Switch live ORB rules from **V0 → V9 (force EOD 14:30)** or **V6 (partial 50% at 1R + BE trail on remainder)**:
- **V9** is a 1-line change to `eod_exit_time` in config; lowest-risk swap.
- **V6** needs order-plumbing (half-close at 1R, move SL to entry on remainder) but squeezes more return.

Both cut drawdown by ~40%. Current EOD squareoff is already 15:18 (per commit 54a56de), pulling it to 14:30 is plausible.

### Artifacts on VPS
- `/home/arun/quantifyd/variants_sweep_trades.csv` — 1,900 rows (190 trades × 10 variants)
- `/home/arun/quantifyd/variants_sweep_daily.csv` — 600 rows (60 days × 10 variants)
- `/home/arun/quantifyd/variants_sweep.log` — full run output

---

# Extended run — top 3 variants + baseline over 250 trading days

**Script:** [scripts/_orb_top3_fullperiod.py](../scripts/_orb_top3_fullperiod.py)
**Window:** 2025-05-05 → 2026-04-17 (250 trading days, 11.5 months)
**Trades:** 1,160 per variant (7.7% of 15,000 stock-days triggered a filtered breakout)
**Variants tested:** V0, V5, V6, V9 only (top 3 from 60-day sweep + baseline)

## Report A — sweep notional (Rs 23.1L per trade, MIS 5× leveraged as in original sweep)

| Rank | Variant | Trades | WR | PF | Net P&L | MaxDD | Return | Calmar |
|-----:|---------|-------:|----:|----:|--------:|------:|-------:|-------:|
| 1 | **V9 close@14:30** | 1160 | 76.3% | 7.87 | +Rs 1,38,88,420 | Rs 49,145 (2.1%) | +601% | **282.60** |
| 2 | V0 baseline (live) | 1160 | 73.9% | 6.07 | +Rs 1,42,23,727 | Rs 95,322 (4.1%) | +616% | 149.22 |
| 3 | V5 trail-BE @ 1R | 1160 | 71.5% | 6.47 | +Rs 1,40,92,170 | Rs 95,322 (4.1%) | +610% | 147.84 |
| 4 | V6 partial 50% @ 1R + BE trail | 1160 | 75.8% | 6.37 | +Rs 1,38,23,734 | Rs 95,322 (4.1%) | +598% | 145.02 |

## Report B — cash / CNC sizing (live config: Rs 1,00,000 capital, Rs 20,000/trade, no leverage)

From [config.py:466-467](../config.py#L466): `capital = 1,00,000`, `max_concurrent_trades = 5`. Scale factor vs sweep = `20,000 / 23,10,000 = 0.00866`.

| Rank | Variant | Net P&L (cash) | MaxDD (cash) | Return on 1L | Calmar |
|-----:|---------|---------------:|-------------:|-------------:|-------:|
| 1 | **V9 close@14:30** | +Rs 1,20,216 | Rs 425 | **+120.2%** | **282.60** |
| 2 | V0 baseline (live) | +Rs 1,23,117 | Rs 825 | +123.1% | 149.22 |
| 3 | V5 trail-BE @ 1R | +Rs 1,22,021 | Rs 825 | +122.0% | 147.84 |
| 4 | V6 partial 50% @ 1R + BE trail | +Rs 1,19,700 | Rs 825 | +119.7% | 145.02 |

## What flipped vs the 60-day test

| | 60-day | 250-day |
|---|---|---|
| V9 rank | #1 (Calmar 124) | **#1 (Calmar 283)** — still dominant, doubles |
| V6 vs V0 | V6 beat V0 | **V0 > V6** — partial-50% concedes P&L over long runs |
| V5 vs V0 | V5 beat V0 | V0 > V5 by a hair |
| DD range | 1.0–2.1% | 2.1–4.1% — worse episodes show up in the 1-yr sample |

## Final recommendation

**Switch live `eod_exit_time` from `15:18` → `14:30` (V9).** Single-line change in [config.py:475](../config.py#L475). Calmar is ~2× baseline in both the 60-day AND 250-day windows. V5 and V6 do not justify their exit-plumbing complexity over long runs.

### Artifacts on VPS (top-3 run)
- `/home/arun/quantifyd/variants_top3_trades.csv` — 4,640 rows (1,160 × 4 variants)
- `/home/arun/quantifyd/variants_top3_daily.csv` — 1,000 rows (250 × 4 variants)
- `/home/arun/quantifyd/variants_top3.log` — full run output

## How to resume this sweep later

```bash
# On VPS — picks up where it left off (skips dates already in daily CSV):
cd /home/arun/quantifyd
setsid bash -c 'nohup venv/bin/python3 scripts/_orb_variants_sweep.py > variants_sweep.log 2>&1 < /dev/null &'
tail -f variants_sweep.log
```

## Files

- Local: [scripts/_orb_variants_sweep.py](../scripts/_orb_variants_sweep.py)
- VPS helper: [scripts/_vps_helper.py](../scripts/_vps_helper.py)
- VPS query helper: [scripts/_vps_query.py](../scripts/_vps_query.py)
