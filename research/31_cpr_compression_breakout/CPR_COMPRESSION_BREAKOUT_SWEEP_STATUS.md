# CPR-Compression Range Breakout — 79 Stocks Across 5/10/15-min

**STATUS: PLANNING** (sections 1-4 locked, runner not yet launched)
**Created:** 2026-04-30
**Owner:** Claude (driving) / Arun (deciding)
**This file is the SOLE source of truth for crash recovery.**

---

## 1. The Ask

**What you asked (verbatim):**
> "narrow cpr, break of yesterday's range (yesterday's to be a wider
> cpr than today's or narrow range yesterday) and then the price moves
> well... pls can u assess"

**What we're actually testing:**
Is the **CPR-compression-into-breakout** a tradable intraday edge?
Specifically: when today's CPR is narrow AND yesterday was either a
wide-CPR day or a narrow-range day, does an intraday break of yesterday's
high or low produce a clean directional follow-through that an option
seller (or scalper) can ride?

This is **distinct from research/30b's volume-breakout** — that one
uses the FIRST candle close past prev_day_high with volume confirmation.
CCRB uses **CPR-geometry as the entry FILTER**, then watches the
session for an **any-time** range-break trigger.

---

## 2. The Base — what's being tested

### Daily-bar setup (computed pre-session from prev day's daily candle + today's CPR)

A trading day qualifies as a **CCRB setup day** if BOTH:

- **Today's CPR is narrow:** `today_cpr_width / today_open ≤ today_narrow_threshold`
  - Sweep: `today_narrow_threshold ∈ {0.30%, 0.40%, 0.50%}`
- **Yesterday was a compression context** — at least one of:
  - **Variant W** (yesterday wide-CPR): `yesterday_cpr_width / yesterday_open ≥ yesterday_wide_threshold`
    - Sweep: `yesterday_wide_threshold ∈ {0.50%, 0.65%, 0.80%}`
  - **Variant N** (yesterday narrow-range): `(yesterday_high - yesterday_low) / yesterday_open ≤ yesterday_narrow_range_threshold`
    - Sweep: `yesterday_narrow_range_threshold ∈ {0.50%, 0.70%, 0.90%}`
  - **Variant W_OR_N**: either of the above qualifies (loosest)
  - **Variant W_AND_N**: both required (tightest, fewest signals)

Use Standard CPR formulas:
```
pivot      = (PDH + PDL + PDC) / 3
top_central     (TC) = (PDH + PDL) / 2
bottom_central  (BC) = (2 × pivot) - TC
cpr_width = abs(TC - BC)
```

### Intraday signal (entry trigger, any time during session)

LONG entry — first 5/10/15-min candle close where:
- `close > yesterday_high`
- `prev_close ≤ yesterday_high` (fresh transition, not sustained)
- After 09:20 IST (give CPR a moment to print)
- Before 14:00 IST (last-entry cutoff)
- Optional: volume confirmation — `bar_volume ≥ vol_mult × 20-day avg same-bar-position volume`

SHORT entry — mirror (close < yesterday_low).

**One signal per day per (stock × tf × variant × direction).** First fresh transition wins.

### Direction handling

Long and short tested independently — every (stock × variant) combo evaluated for both sides.

### Variant axes

| Axis | Values | Count |
|---|---|---|
| Timeframe | 5min, 10min, 15min | 3 |
| Today narrow threshold (`today_narrow`) | 0.30%, 0.40%, 0.50% | 3 |
| Yesterday context (`yesterday_ctx`) | W (wide CPR), N (narrow range), W_OR_N, W_AND_N | 4 |
| Yesterday wide threshold (used by W and W_AND_N) | 0.50%, 0.65%, 0.80% | 3 |
| Yesterday narrow-range threshold (used by N and W_AND_N) | 0.50%, 0.70%, 0.90% | 3 |
| Volume confirmation | off, vm=1.5, vm=2.0 | 3 |
| Direction | long, short | 2 |

Note: not every combination of yesterday_ctx × wide_threshold × narrow_range_threshold makes sense:
- `W` only uses `yesterday_wide_threshold` (3 effective settings)
- `N` only uses `yesterday_narrow_range_threshold` (3 effective settings)
- `W_OR_N` and `W_AND_N` use both (3 × 3 = 9 settings each)

So per (stock × tf × today_narrow × volume × direction):
- W variants: 3
- N variants: 3
- W_OR_N: 9
- W_AND_N: 9
- = 24 yesterday-context variants

Per stock × tf × today_narrow: 24 × 3 (volume) × 2 (direction) = 144 cells
Per stock × tf: 3 (today_narrow) × 144 = **432 cells**
Per stock: 3 (tf) × 432 = **1,296 cells**
**Total cells: 79 × 1,296 = 102,384 cells**

That's an order of magnitude bigger than research/30b. Many will be empty.

### Exit policies (13 tested in parallel per signal — same as research/30b)

`T_NO`, `T_HARD_SL` (yesterday's opposite extreme), `T_ATR_SL_{0.3, 0.5, 1.0}`,
`T_CHANDELIER_{1.0, 1.5, 2.0}`, `T_R_TARGET_{1R, 1.5R, 2R, 3R}`, `T_STEP_TRAIL`.

R defined as: `R = abs(entry - yesterday_opposite_extreme)`.

### Universe — 79 stocks (same as research/30b)

**Cohort A (10 stocks since 2018-01-01):**
RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, BHARTIARTL, ITC, KOTAKBANK, HINDUNILVR

**Cohort B (69 stocks since 2024-03-18):**
ADANIENT, ADANIPORTS, AMBUJACEM, APOLLOHOSP, ASIANPAINT, AXISBANK, BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BANKBARODA, BEL, BPCL, BRITANNIA, CHOLAFIN, CIPLA, COALINDIA, COFORGE, COLPAL, CUMMINSIND, DABUR, DELHIVERY, DIVISLAB, DLF, DRREDDY, EICHERMOT, FEDERALBNK, GAIL, GODREJPROP, GRASIM, HAL, HAVELLS, HCLTECH, HDFCLIFE, HEROMOTOCO, HINDALCO, IDFCFIRSTB, INDUSINDBK, IOC, IRCTC, JINDALSTEL, JSWSTEEL, LT, M&M, MARICO, MARUTI, MCX, MUTHOOTFIN, NESTLEIND, NTPC, ONGC, PAYTM, PERSISTENT, PIDILITIND, PNB, POWERGRID, SBILIFE, SHREECEM, SIEMENS, SUNPHARMA, TATACONSUM, TATAPOWER, TATASTEEL, TECHM, TITAN, TRENT, ULTRACEMCO, VEDL, VOLTAS, WIPRO

### Period

Per-stock available range, capped at 2026-03-25 (data end).
Cohort A: 2018-2026 (~2,065 days). Cohort B: 2024-2026 (~485 days).

### Success criterion (ranking metric)

Same Sharpe-style score as research/30b for direct comparability:
`sharpe_score = (mean_net_pct / std_net_pct) × win_rate_fraction`

### Promote gate

A stock makes the promote list if it passes ALL three:
1. Best-cell Sharpe ≥ 0.5
2. n ≥ 15 in that cell
3. Robust across ≥ 3 variants (Sharpe ≥ 0.3 in 3+ cells with mean > 0)

### Comparison to research/30b (the volume-breakout sweep)

After both runs complete, will compare:
- **Same stock, same direction, same timeframe** — does CCRB beat first-bar-volume?
- **Are CCRB winners DIFFERENT stocks** than the volume-breakout winners?
- **Can the two signals be combined** (CCRB-day + first-bar-volume confirmation) for higher conviction with rare firings?

---

## 3. Plan — Variant grid + cell count

```
79 stocks × 3 timeframes × 3 today_narrow_thresholds × 24 yesterday_ctx
       × 3 volume_modes × 2 directions = 102,384 cells
```

Vast majority will be empty (n<5) — narrow-CPR-today + compression-yesterday + breakout is a rare daily setup.

Estimated computational cost (extrapolating from research/30b pace):
- Cohort A: ~600s/stock (more cells per stock × more days)
- Cohort B: ~120s/stock
- Total: 10 × 600 + 69 × 120 = 14,280s = **~4 hours wall**

Aggregation will use the streaming approach from research/30b
(no `pd.read_csv` on the heavy CSV).

---

## 4. Status (live running log)

**State:** AGGREGATING
**Started:** 2026-05-02 14:41:15
**Last completed stock:** WIPRO
**Stocks completed:** 79 / 79  (100.0%)
**Signals logged:** 447,528
**Elapsed:** 268.2 min
**Last update:** 2026-05-02 19:09:26 IST

### Files

- `results/ccrb_signals.csv` — per-signal x exit-policy rows
- `results/run.log` — per-stock progress

---

## 5. Crash Recovery

### A) Check what finished
```bash
tail -5 research/31_cpr_compression_breakout/results/run.log
wc -l research/31_cpr_compression_breakout/results/ccrb_signals.csv
```

### B) Resume signal generation (resumable)
```bash
cd /c/Users/Castro/Documents/Projects/Covered_Calls
python research/31_cpr_compression_breakout/scripts/run_ccrb.py
```

### C) Aggregate only
```bash
python research/31_cpr_compression_breakout/scripts/aggregate_ccrb.py
```

Runner reads `ccrb_signals.csv`, builds a (symbol,tf,variant,dir,date) skip-set, only computes unfinished cells.

### D) Files NOT to touch
- `results/ccrb_signals.csv`
- `results/run.log`
- This file (auto-updated)

---

## 6. Files (output map)

| File | Purpose | Committable? |
|---|---|---|
| SWEEP-STATUS.md | This file | yes |
| scripts/signals_ccrb.py | Signal generator | yes |
| scripts/run_ccrb.py | Sweep runner | yes |
| scripts/aggregate_ccrb.py | Streaming aggregator | yes |
| results/ccrb_signals.csv | Per-signal rows (large) | gitignored |
| results/ccrb_ranking.csv | Per-cell aggregate | gitignored if >5MB |
| results/ccrb_leaders.csv | Per-stock leaderboard | yes |
| results/RESULTS.md | Final report | yes |

---

## 7. Findings (during + final)

_Will populate after aggregation._

---

## 8. Comparison to research/30b

_Will populate after aggregation._

**Last updated:** 2026-05-02 19:09:26 IST
