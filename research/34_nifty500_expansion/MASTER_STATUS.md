# Nifty 500 Expansion — Master Status Doc

**THIS IS THE PRIMARY STATUS DOC.** The auto-generated `NIFTY500_EXPANSION_SWEEP_STATUS.md` and `CCRB_RUN_PROGRESS.md` are runner-progress files; this one tracks the full project narrative.

**Created:** 2026-05-02
**Last updated:** 2026-05-05 (current session)
**Owner:** Claude (driving) / Arun (deciding)

---

## 1. The Ask (full chain so far)

1. **Initial:** "im seeing only 15 mins, did u test 5 and 10 mins also? also how ab not sticking to these 10 stocks but scanning and narrowing down to those special volume loaders?"
2. **Phase 1 plan:** test full Nifty 500 universe with both vol-BO and CCRB strategies
3. **Storage check:** confirmed 53 GB free, 10 GB needed → comfortable
4. **Path C chosen:** trim CCRB grid (drop ctxN + ctxW_AND_N) for compute efficiency
5. **Trade frequency target:** 2-3 trades/day across the deployable portfolio
6. **Per-stock tuning approved** over single-rule or bucket-generic — accepted as the higher-quality approach
7. **Liquidity-gated wider universe:** include beyond Nifty 500 names that pass Rs 50 Cr/day turnover gate. Currently approved.

---

## 2. The Base — what's being tested

### Two strategies in parallel

| Strategy | Folder | Trigger | Filter |
|---|---|---|---|
| Vol-BO | research/30b → /34 | First 5/10/15-min candle close past prev_day_high/low | Volume ≥ vm × 20-day avg, optional gap, optional RSI |
| CCRB | research/31 → /34 | First fresh transition past prev_day_high/low between 09:20-14:00 | Today narrow CPR + yesterday wide CPR or narrow range |

### Universe evolution

| Step | Universe | Stocks | Notes |
|---|---|---|---|
| Original | Cohort A + B | 79 | research/30b, /31 baselines |
| Phase A backfill | + 320 N500 stocks | 380 | Done; 15 N500 names lost to delisting/renames |
| Trim to compute-able | Top-100 by daily turnover (Rs 172+ Cr/day) | 100 | Phase B + C run on this |
| **Phase E (planned)** | **Rs 50+ Cr/day turnover** | **~353** | **Add 135 new mid/small-cap names + re-test** |

### Variant grids

**Vol-BO (per stock):** TF × vm × gap × RSI × dir = 3 × 3 × 4 × 2 × 2 = 144 cells per stock

**CCRB (per stock, trimmed grid):** TF × today_narrow × yctx (W only + W_OR_N) × vol × dir = 3 × 3 × 12 × 3 × 2 = 648 cells per stock

### Period

- Cohort A: 2018-01-01 → 2026-03-25
- Cohort B (original 69): 2024-03-18 → 2026-03-25
- Cohort C (new from backfill): 2024-03-18 → 2026-03-25

### Liquidity gate (in runners)

- Min 20-day median price ≥ Rs 50
- Min 20-day median rupee turnover (close × volume) ≥ Rs 5 cr (current; will lower to Rs 25 cr in Phase E to admit Rs 50+ Cr/day stocks correctly)

### Success criterion

`sharpe_score = (mean_net_pct / std_net_pct) × win_rate_fraction`

### Promote gate

- Best-cell Sharpe ≥ 0.5
- n ≥ 15 trades in that cell
- Robust across ≥ 3 variants (Sharpe ≥ 0.3 in 3+ cells)

### Deployment approach (LOCKED 2026-05-05)

**Per-stock tuning** — each stock gets its own variant config (TF, vm, gap, RSI, exit). No bucket-generic or single rule. Implemented as a `STOCK_CONFIGS` dict in the live scanner.

---

## 3. Plan — phases

```
Phase A: Backfill 320 missing Nifty 500 stocks       ✅ DONE  (380/395 succeeded)
Phase B: Vol-BO sweep on top-100 universe            ✅ DONE  (230K signals, RESULTS.md written)
Phase C: CCRB sweep on top-100 (trimmed grid)        🔄 RUNNING (bash bzec0c3zn, ~5h remaining)
Phase D: Phase B+C aggregation + comparison           ⏳ PENDING
Phase E: Backfill Rs 50+ Cr/day expansion (135 new)   ⏳ PENDING
Phase F: Re-run vol-BO on 353-stock universe          ⏳ PENDING (~6h)
Phase G: Re-run CCRB on 353-stock universe            ⏳ PENDING (~12h)
Phase H: Final per-stock STOCK_CONFIGS list          ⏳ PENDING
```

---

## 4. Status (live event log)

| Date/time | Event | Notes |
|---|---|---|
| 2026-05-02 | Folder + STATUS doc created | Per LIVE-STATUS-MD convention |
| 2026-05-02 22:08 | Phase A backfill launched | bash bm1bsea6w, 320 missing stocks |
| 2026-05-04 21:30 | Resumed after laptop suspend + token refresh | 237/395 → continued |
| 2026-05-05 (early) | Phase A DONE | 380/395 stocks (15 lost to delisting) |
| 2026-05-05 14:30 | Phase B vol-BO launched on full 378-stock universe | bash bbvu1owyi |
| 2026-05-05 (mid) | Phase B exited at 28/378 | Trimmed to top-100 by turnover, restarted |
| 2026-05-05 16:30 | Phase B relaunched on top-100 | bash bjr3p9rtz |
| 2026-05-05 19:00 | Phase B DONE | 100/100 stocks, 230K signals, RESULTS.md auto-aggregated |
| 2026-05-05 19:06 | Phase C CCRB launched on top-100 | bash bzec0c3zn |
| 2026-05-05 19:30 | Patched runners to use separate progress files | Master STATUS protected for future runs |
| 2026-05-05 20:42 | Phase C progress check — 28/100 stocks, 261K signals | Last: COCHINSHIP, elapsed 94.6 min, pace 3.4 min/stock, ETA ~4h more |

### Top-15 vol-BO findings (Phase B done)

| # | Stock | TF | Dir | Sharpe | WR% | Mean% | n |
|---|---|---|---|---|---|---|---|
| 1 | HAL | 5m | short | 1.354 | 90.0 | +1.41 | 10 |
| 2 | HDFCAMC | 10m | long | 0.996 | 90.9 | +1.46 | 11 |
| 3 | RBLBANK | 10m | long | 0.919 | 92.3 | +1.08 | 13 |
| 4 | RELIANCE | 15m | short | 0.881 | 92.9 | +1.17 | 14 |
| 5 | PERSISTENT | 5m | long | 0.796 | 81.8 | +1.42 | 11 |
| 6 | LAURUSLABS | 10m | long | 0.772 | 84.6 | +1.42 | 13 |
| 7 | COCHINSHIP | 15m | short | 0.693 | 78.6 | +0.70 | 14 |
| 8 | EICHERMOT | 5m | long | 0.653 | 80.0 | +0.50 | 10 |
| 9 | GODFRYPHLP | 10m | long | 0.611 | 80.0 | +1.88 | 10 |
| 10 | AARTIIND | 10m | short | 0.603 | 80.0 | +1.94 | 10 |
| 11 | 3MINDIA | 5m | short | 0.592 | 84.6 | +0.60 | 13 |
| 12 | ACC | 5m | short | 0.574 | 86.7 | +0.85 | 15 |
| 13 | LT | 15m | long | 0.574 | 81.8 | +0.68 | 11 |
| 14 | WIPRO | 10m | short | 0.572 | 83.3 | +0.68 | 12 |
| 15 | ADANIGREEN | 15m | short | 0.557 | 81.8 | +2.95 | 11 |

Of 119 stocks tested: 51 with Sharpe ≥ 0.3, 19 with Sharpe ≥ 0.5.

---

## 5. Crash Recovery — how to resume without Claude

### Where to look

| File | Purpose |
|---|---|
| `MASTER_STATUS.md` (this file) | Project narrative, decisions, plan |
| `VOLBO_RUN_PROGRESS.md` | Auto-generated vol-BO live progress |
| `CCRB_RUN_PROGRESS.md` | Auto-generated CCRB live progress (will be created once new code path runs) |
| `NIFTY500_EXPANSION_SWEEP_STATUS.md` | DEPRECATED — was being clobbered by runners; keep for legacy compatibility |

### Check what finished

```bash
# Phase A — backfill complete?
python -c "import sqlite3; c=sqlite3.connect('backtest_data/market_data.db'); print(c.execute(\"SELECT COUNT(DISTINCT symbol) FROM market_data_unified WHERE timeframe='5minute'\").fetchone()[0], 'stocks have 5-min data')"
# Expected: 380

# Phase B — vol-BO complete?
ls research/34_nifty500_expansion/results/RESULTS.md  # exists if done
ls research/34_nifty500_expansion/results/volbo_leaders.csv

# Phase C — CCRB progress
tail -5 research/34_nifty500_expansion/results/ccrb_run.log
wc -l research/34_nifty500_expansion/results/ccrb_signals.csv
```

### Resume each phase

```bash
# Phase A backfill (resumable, skips done stocks):
python research/34_nifty500_expansion/scripts/backfill_nifty500.py
# (requires fresh Kite token in backtest_data/access_token.json — copy from VPS via paramiko)

# Phase B vol-BO (resumable):
python research/34_nifty500_expansion/scripts/run_volbo_500.py

# Phase C CCRB (resumable):
python research/34_nifty500_expansion/scripts/run_ccrb_500.py

# Aggregate vol-BO (streaming, fast):
# already auto-aggregated by runner; or run aggregate_volbo.py if needed

# Aggregate CCRB:
python research/34_nifty500_expansion/scripts/aggregate_ccrb.py
```

### Phase E backfill (planned, not yet started)

When Phase C finishes, lower the liquidity threshold and add 135 new stocks (Rs 50-200 Cr/day turnover, no 5-min data yet):

```bash
# (Need to write phase_E_backfill.py — derives new stock list from
#  research/34/results/top_by_turnover.csv where cr_per_day >= 50
#  AND stock not already in 5-min DB.)
```

### Files NOT to touch during runs

- `backtest_data/market_data.db` (active during Phase A, E)
- `results/volbo_signals.csv`, `results/ccrb_signals.csv` (active)
- `results/*_run.log` (live)
- `VOLBO_RUN_PROGRESS.md`, `CCRB_RUN_PROGRESS.md` (auto-overwritten by runners)

### Files safe to inspect

- This `MASTER_STATUS.md`
- `scripts/*.py`
- `results/RESULTS.md`, `results/*_leaders.csv` (between phases)
- `results/top_by_turnover.csv` (the universe ranking)

---

## 6. Files (output map)

| File | Purpose | Committable? |
|---|---|---|
| `MASTER_STATUS.md` | Master narrative (this file) | ✅ |
| `NIFTY500_EXPANSION_SWEEP_STATUS.md` | Legacy / clobbered by runner | partial |
| `VOLBO_RUN_PROGRESS.md` | Vol-BO auto-progress | ✅ once stable |
| `CCRB_RUN_PROGRESS.md` | CCRB auto-progress | ✅ once stable |
| `scripts/backfill_nifty500.py` | Phase A backfill | ✅ |
| `scripts/run_volbo_500.py` | Phase B vol-BO runner | ✅ |
| `scripts/run_ccrb_500.py` | Phase C CCRB runner | ✅ |
| `scripts/aggregate_volbo.py` | Streaming vol-BO aggregator | ✅ |
| `scripts/aggregate_ccrb.py` | Streaming CCRB aggregator | ✅ |
| `results/RESULTS.md` | Vol-BO findings | ✅ |
| `results/volbo_signals.csv` | Heavy (300 MB+) | ❌ gitignore |
| `results/volbo_ranking.csv` | Per-cell aggregate | ❌ gitignore (60 MB+) |
| `results/volbo_leaders.csv` | Per-stock leaderboard | ✅ (small) |
| `results/ccrb_signals.csv` | Heavy | ❌ gitignore |
| `results/ccrb_ranking.csv` | Per-cell | ❌ gitignore |
| `results/ccrb_leaders.csv` | Per-stock | ✅ |
| `results/top_by_turnover.csv` | Universe ranking | ✅ |
| `results/*.log` | Run logs | ❌ gitignore |

---

## 7. Findings (during + final)

### Phase B (Vol-BO on top-100) — DONE

- 230,180 signal rows
- Top-15 leaderboard: HAL #1 (Sharpe 1.35); 8 NEW stocks emerged from the universe expansion (HDFCAMC, RBLBANK, LAURUSLABS, COCHINSHIP, AARTIIND, 3MINDIA, ACC, ADANIGREEN)
- **GODFRYPHLP** (your original chart observation) ranked #9 with Sharpe 0.61, mean +1.88%, 80% WR — confirmed
- Promote-gate (Sharpe ≥ 0.5 + n ≥ 15 + 3+ robust): only ACC and VEDL
- Per-stock-best avg Sharpe: 0.74 across top 15
- Best broad rule across all 119 stocks: 15-min SHORT vm 3.0 gap-off T_NO — works on 14/71 stocks at Sharpe ≥ 0.3, avg Sharpe only 0.12. **Confirms per-stock tuning is the right approach.**

### Phase C (CCRB on top-100) — RUNNING

Will populate after completion.

---

## 8. Per-stock STOCK_CONFIGS deployment list

_Will be populated after Phase C finishes — combined vol-BO + CCRB top picks with their full per-stock variant configs ready for the live paper scanner._

---

## 9. Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-02 | Universe = full Nifty 500, not microcaps | Research/32 showed universe-wide top setup-frequency stocks were penny names |
| 2026-05-02 | CCRB grid trimmed (drop ctxN, ctxW_AND_N) | Research/32: ctxN dead in liquid universe; research/31: ctxW_AND_N never fires |
| 2026-05-05 | Trim Phase B/C universe to top-100 by turnover | Compute-time concern (32h → 10h) without losing tradable names |
| 2026-05-05 | Per-stock tuning, NOT bucket-generic | Avg Sharpe 0.74 (per-stock-best) vs ~0.5 (bucket-generic) vs 0.12 (single rule) |
| 2026-05-05 | Phase E expansion to Rs 50 Cr/day liquidity gate | User wants 2-3 trades/day; need wider universe |
| 2026-05-05 | Auto-progress writes split from master STATUS | Runner kept clobbering master doc — now uses separate VOLBO_RUN_PROGRESS.md / CCRB_RUN_PROGRESS.md |

---

**This file is the SOLE source of truth for project resume.**
