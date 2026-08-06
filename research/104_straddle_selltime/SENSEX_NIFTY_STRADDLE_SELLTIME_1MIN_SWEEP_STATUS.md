# ATM Straddle Sell-Timing — Best Time-of-Day to Short, by DTE, on 75 Days of REAL 1-min Chain

**STATUS: DONE** — hunch OVERTURNED: driver is DTE/day-of-week, not intraday time; keep 09:16; signal refuted. Research/104.

> **Real-data study (not modeled).** Uses `options_data.db` `option_chain` — real 1-min option LTPs
> with spot, 2026-04-20→2026-08-06, 75 days each for SENSEX & NIFTY. This is the higher-confidence
> counterpart to research/103 (which was modeled). Gate target: **G2** (does a tradeable
> entry-timing rule capture the intraday decay, net of cost, on a meaningful sample).

## 1. The Ask (restated precisely)

**Trigger:** Arun observed on 08-05/08-06 that ATM premiums **rise through the morning and only
decay late** — and the NAS systems sell at **09:16**, i.e. straight into the morning expansion, then
get stopped out before the decay arrives.

**What we're testing:** Across 75 real days, **what entry time-of-day maximises the net P&L of a
sell-ATM-straddle / square-at-15:25 (MIS same-day) trade, split by DTE** — and does a **"premium has
peaked" signal** beat a fixed clock time? Single metric: **mean net P&L per lot** by (symbol, entry
time, DTE), with the **p05 tail** and win-rate. This informs the NAS live entry-time decision
(no live change without sign-off).

## 2. Economic hypothesis (G0)

Morning **IV expansion (vega)** + **gamma-chop (realized vol)** inflate the ATM straddle over the
first ~3 hours, while **theta is back-loaded** (esp. DTE0, where decay concentrates in the last
1.5–2h). So selling **after the AM vol peak** should capture more decay with less morning drawdown.
Counterparty: morning hedging/positioning demand + expiry gamma. Decay risk: intraday seasonality
can shift with regime — 75 days is a G2 sample, not a decade; treat as directional evidence.

**Falsification:** if the 09:20 morning entry is **not** worse than midday across DTE buckets, the
08-05/06 observation was small-sample noise and the NAS 09:16 timing stands.

## 3. Base mechanics (locked)

- **Data:** `option_chain` (real 1-min LTP + underlying_spot), front weekly expiry, strike nearest
  spot ("running ATM"). SENSEX expiry Thu, NIFTY Tue (both read from `expiry_date`).
- **Trade:** SELL ATM straddle at entry time T; **square off = buy back the SAME strike at 15:25**
  (matches NAS MIS same-day exit). P&L/unit = prem(T) − prem_sameK(15:25).
- **Cost:** ₹160 brokerage + 0.5 premium-pt/leg slippage → net = pnl_pts×lot − (160 + 2×lot).
  SENSEX lot 20 (≈₹200/lot), NIFTY lot 65 (≈₹290/lot). Report **gross AND net**.
- **DTE** = (front_expiry − day).days, bucketed 0 / 1 / 2+.
- **Metric:** mean net P&L per lot; also median, win%, p05 (tail), n.

## 4. Plan (grid)

- **Entry time (10):** 09:20, 09:45, 10:15, 10:45, 11:15, 11:45, 12:15, 12:45, 13:15, 13:45.
- **+ SIGNAL entry:** first minute after 09:45 where running-ATM straddle ≤ 0.92 × its running
  intraday high ("premium rolled over") — tests clock-free peak-detection.
- **Symbol (2):** SENSEX, NIFTY. **DTE (3):** 0, 1, 2+.
- Cells = 11 entries × 2 symbols × 3 DTE, each over ≤75 days. Fixed: exit 15:25, front expiry,
  running-ATM, cost model. The DTE0 row (SENSEX Thu / NIFTY Tue) is the headline.

## 5. Status (live log)

| Date/time IST | Event | Notes |
|---|---|---|
| 2026-08-06 ~16:00 | Framed + STATUS written | Real option_chain: 75 days SENSEX+NIFTY |
| 2026-08-06 ~16:05 | Sweep launched (VPS bg) | detail CSV appended per day; summary at end |
| 2026-08-06 ~16:20 | DONE — RESULTS written | Hunch overturned; driver is DTE/day-of-week not time |

## 6. Crash recovery

- Rerun `research/104_straddle_selltime/scripts/run_selltime_sweep.py` — it appends a per-day-trade
  detail CSV (`results/selltime_detail.csv`) and rebuilds the summary from it; safe to re-run
  (idempotent per symbol-day via a done-set).
- Read-only on `options_data.db`. Progress: `tail results/run.log`.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `SENSEX_NIFTY_STRADDLE_SELLTIME_1MIN_SWEEP_STATUS.md` | This doc | yes |
| `scripts/run_selltime_sweep.py` | Sweep runner | yes |
| `results/selltime_detail.csv` | Per day-trade (small) | yes |
| `results/selltime_summary.csv` | Aggregate table (deliverable) | yes |
| `results/RESULTS.md` | Verdict | yes |

## 8. Findings

See `results/RESULTS.md`. Headline (75 days, real chain, no-stop hold-to-15:25):

1. **Hunch OVERTURNED** — 09:16 is NOT the worst sell-time. **SENSEX DTE0 (Thu) 09:20 early entry =
   +2,575/lot, 93% win, p05 −139** (best cell). 08-06 was a bad-morning outlier; held it still won —
   the loss came from the STOP firing, not the entry time.
2. **Driver is DTE / day-of-week, not the clock:** SENSEX **Wed (DTE1) = dangerous** (mean-neg every
   entry, p05 ≈ −₹17k/lot tail — 08-05's −11.5k live loss); NIFTY **Mon (DTE1) = sweet spot** (+737,
   75% win, p05 −904 at 13:15). SENSEX Thu good-if-held; NIFTY Tue tail-heavy early.
3. **"Premium-peaked" SIGNAL REFUTED** — catches a falling knife, worse than best clock w/ fat tails.
4. **Decision:** keep 09:16 (don't move entry time). Real lever = **stop/sizing on SENSEX-Wednesday**
   (the fat-tail day) — ties to research/103. n=15/DTE cell + one regime → modest confidence.
