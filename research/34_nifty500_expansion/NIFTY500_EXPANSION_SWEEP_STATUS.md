# Nifty 500 Expansion — Volume-BO + CCRB Re-tested on Full Universe

**STATUS: PLANNING** (sections 1-4 locked, no scripts run yet)
**Created:** 2026-05-02
**Owner:** Claude (driving) / Arun (deciding)
**This file is the SOLE source of truth for crash recovery.**

---

## 1. The Ask

**What you asked (verbatim):**
> "phase 1 yes... but assess if we have sufficient storage 1st and report"
> [storage assessment given — 53 GB free, 10 GB needed]
> "C" [option C — full Nifty 500, both sweeps, CCRB grid trimmed]

Earlier framing:
> "im thinking of applying this across nifty stocks including the microcap"

**What we're actually testing:**
Re-run both Volume-Breakout and CPR-Compression-Range-Breakout (CCRB)
strategies on the **full Nifty 500 universe** (currently we only have
backtest data for 79 stocks — 75 from N500 + 4 FNO extras). Identify
the comprehensive deployable list of stocks for each strategy.

Scope decisions:
- **No microcaps below Nifty 500** in this phase. Per research/32's
  warning, the universe-wide top setup-frequency stocks are illiquid
  penny names (SHIVAUM 94%, UEL 90%, etc.) where CPR is mechanically
  narrow because the bar barely moves. Need a separate microcap-
  specific strategy (Phase 2, later). Nifty 500 is the cleanest large/
  mid/small-cap pool with reasonable liquidity.
- **CCRB variant grid trimmed** — drop `ctxN` (yesterday narrow-range
  alone) and `ctxW_AND_N`. Research/32 proved both are dead in the
  liquid universe (ctxN fires 0.04-0.32 stocks/day; ctxW_AND_N fires
  zero in research/31). This halves the CCRB cell count (1,296 → 648
  per stock) without losing real signals.

---

## 2. The Base — what's being tested

### Phase A — Backfill 5-min data for 320 missing Nifty 500 stocks

Source: Kite historical_data API
- Period: 2024-03-18 → 2026-03-25 (matches Cohort B coverage in current 79)
- Chunk size: 7 days per call (per existing `services/data_manager.py`)
- Rate limit: 0.35s between calls (3 req/sec target)
- Per stock: ~106 chunks, ~37s
- Total: 320 × 37s = ~3.3 hours, allow 4h with buffer

Storage projection:
- Per stock × 2-yr period: ~32K rows × ~250 bytes = ~8 MB
- Total DB growth: ~2.6 GB
- Will be appended to `backtest_data/market_data.db` (currently 2.25 GB → 4.85 GB after)

### Phase B — Volume-BO sweep on Nifty 500 universe

Same signal as research/30b (volume-confirmed first-candle breakout):
- LONG: first 5/10/15-min candle close > prev_day_high + volume spike
- SHORT: mirror

Variant grid (unchanged from /30b):
- TF: 5min, 10min, 15min
- vol_mult: 1.5, 2.0, 3.0
- gap: 0%, 0.3%, 0.5%, off
- RSI: off, on (≥60 long / ≤40 short)
- Direction: long, short

Per stock: 3 × 3 × 4 × 2 × 2 = 144 cells
Total: 395 × 144 = **56,880 cells**

### Phase C — CCRB sweep on Nifty 500 universe (trimmed grid)

Same signal as research/31 (today narrow CPR + yesterday compression + range break):

Trimmed variant grid:
- TF: 5min, 10min, 15min
- today_narrow: 0.30%, 0.40%, 0.50%
- yesterday_ctx: **W (3 variants), W_OR_N (9 variants)** — DROP `N` and `W_AND_N` per research/32
- vol: off, vm=1.5, vm=2.0
- Direction: long, short

Per stock: 3 × 3 × 12 × 3 × 2 = **648 cells per stock** (was 1,296 in /31)
Total: 395 × 648 = **256,000 cells**

### Universe (after backfill)

**Cohort A** (10 stocks, 8-yr history): RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, BHARTIARTL, ITC, KOTAKBANK, HINDUNILVR

**Cohort B-original** (69 stocks, ~2-yr): the existing 69 in research/30b/31 (ADANIENT through WIPRO)

**Cohort C-new** (320 stocks, ~2-yr from backfill): all other Nifty 500 names

Total: **399 unique stocks** (= 75 N500-overlap + 4 FNO-extras + 320 new).

### Period

- Cohort A: 2018-01-01 → 2026-03-25
- Cohort B + C: 2024-03-18 → 2026-03-25

### Liquidity gate (built into the runner)

Before running each cell, apply: `min_price ≥ Rs 50` AND `20-day median rupee turnover ≥ Rs 5 crore`. Stocks failing this on the day are skipped.

This guards against penny-stock pollution (per research/32). For Nifty 500 the gate is largely a no-op (nearly all qualify), but harmless to include.

### Success criterion (ranking metric)

Same Sharpe-style score as research/30b/31 for direct comparability:
`sharpe_score = (mean_net_pct / std_net_pct) × win_rate_fraction`

### Promote gate

A stock makes the promote list if it passes ALL three:
1. Best-cell Sharpe ≥ 0.5
2. n ≥ 15 in that cell
3. Robust across ≥ 3 variants (Sharpe ≥ 0.3 in 3+ cells with mean > 0)

### Comparison to research/30b and /31

After both sweeps complete:
- Re-confirm prior 79-stock leaderboard rankings
- Identify NEW stocks from the 320-backfill that crack the top tier
- Highlight stocks robust on BOTH signals (the high-conviction set)
- Build a "deployable Nifty 500 watchlist" for the live paper scanner

---

## 3. Plan — phases + cell count + ETA

```
Phase A — Backfill 320 stocks               ~4h wall
Phase B — Vol-BO sweep on 395 stocks        ~3h wall (56,880 cells)
Phase C — CCRB sweep on 395 stocks (trimmed) ~16h wall (256,000 cells)
Phase D — Aggregation + comparison           ~10 min wall
                                            ≈ 23 hours total wall
```

Phases are **sequential** (B and C both need the data from A; D needs B and C).

Storage projection (verified 2026-05-02): 53 GB free → 10 GB needed → **43 GB safety margin**, no constraint.

---

## 4. Status (live running log)

**State:** PLANNING (about to launch Phase A)
**Last update:** 2026-05-02 (sections 1-4 locked, scripts being built)

### Event log

| Date/time | Event | Notes |
|---|---|---|
| 2026-05-02 | Folder + STATUS doc created (sections 1-4) | Per LIVE-STATUS-MD convention: STATUS first, run later |
| | (TBD) Phase A — backfill script built and launched | |
| | (TBD) Phase A — backfill complete | 320 new symbols × ~32K rows = ~10M rows added to market_data.db |
| | (TBD) Phase B — vol-BO sweep launched | |
| | (TBD) Phase B — vol-BO sweep complete | |
| | (TBD) Phase C — CCRB sweep launched | |
| | (TBD) Phase C — CCRB sweep complete | |
| | (TBD) Phase D — aggregation done; findings written | |

---

## 5. Crash Recovery — how to resume without Claude

### A) Check what finished

```bash
# Phase A — backfill complete?
python -c "import sqlite3; c=sqlite3.connect('backtest_data/market_data.db'); print('Stocks with 5-min:', c.execute(\"SELECT COUNT(DISTINCT symbol) FROM market_data_unified WHERE timeframe='5minute'\").fetchone()[0])"
# Expected after Phase A: 399 (currently 79)

# Phase B — vol-BO progress
tail -5 research/34_nifty500_expansion/results/volbo_run.log 2>/dev/null
wc -l research/34_nifty500_expansion/results/volbo_signals.csv 2>/dev/null

# Phase C — CCRB progress
tail -5 research/34_nifty500_expansion/results/ccrb_run.log 2>/dev/null
wc -l research/34_nifty500_expansion/results/ccrb_signals.csv 2>/dev/null
```

### B) Resume Phase A (backfill)

The backfill script reads existing `market_data_unified` rows and skips
stocks/chunks already downloaded. Safe to re-run:

```bash
cd /c/Users/Castro/Documents/Projects/Covered_Calls
python research/34_nifty500_expansion/scripts/backfill_nifty500.py
```

### C) Resume Phase B (vol-BO sweep)

Append-only CSV with skip-set on `(symbol, tf, variant, direction, date)`:

```bash
python research/34_nifty500_expansion/scripts/run_volbo_500.py
```

### D) Resume Phase C (CCRB sweep)

Same resumability pattern:

```bash
python research/34_nifty500_expansion/scripts/run_ccrb_500.py
```

### E) Run aggregation only (Phase D)

Streaming aggregator (modeled on research/30b — DO NOT use pd.read_csv on the heavy CSVs; hangs on Windows):

```bash
python research/34_nifty500_expansion/scripts/aggregate_volbo.py
python research/34_nifty500_expansion/scripts/aggregate_ccrb.py
```

### F) Files NOT to touch during runs

- `backtest_data/market_data.db` (being appended during Phase A)
- `results/volbo_signals.csv`, `results/ccrb_signals.csv` (being appended)
- `results/*.log` (live progress)

---

## 6. Files (output map)

| File | Purpose | Committable? |
|---|---|---|
| `NIFTY500_EXPANSION_SWEEP_STATUS.md` | This file | ✅ |
| `scripts/backfill_nifty500.py` | Phase A — Kite download | ✅ |
| `scripts/run_volbo_500.py` | Phase B — vol-BO runner | ✅ |
| `scripts/run_ccrb_500.py` | Phase C — CCRB runner | ✅ |
| `scripts/aggregate_*.py` | Phase D — streaming aggregators | ✅ |
| `results/volbo_signals.csv` | Per-signal × per-policy (will be ~5 GB) | ❌ gitignore |
| `results/volbo_ranking.csv` | Per-cell aggregate (~150 MB) | ❌ gitignore if too big |
| `results/volbo_leaders.csv` | Per-stock leaderboard (~30 KB) | ✅ |
| `results/ccrb_signals.csv` | (will be ~3 GB after trim) | ❌ gitignore |
| `results/ccrb_ranking.csv` | (~250 MB) | ❌ gitignore if too big |
| `results/ccrb_leaders.csv` | (~30 KB) | ✅ |
| `results/RESULTS.md` | Final findings | ✅ |
| `results/*.log` | Per-stock progress | ❌ gitignore |

---

## 7. Findings (during + final)

_Will populate as phases complete._

---

## 8. Comparison to research/30b and /31 (filled at end)

_Will compare full-N500 leaderboards vs prior 79-stock results,
identify NEW high-Sharpe stocks from the 320-backfill, and recommend
a deployable Nifty 500 watchlist._

---

**Last updated:** 2026-05-02
**Update cadence:** at every phase transition + significant progress milestones
