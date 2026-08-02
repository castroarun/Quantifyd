# All-Time-High Breakout — Entry at ATH-Day Close, Donchian/SuperTrend Trail (MidSmallcap-400)

STATUS: RUNNING (G1 probe)

---

## 1. The Ask

**What you asked:** "A new strategy — stocks hitting all-time high (minimum liquidity),
entry by that all-time-high-day close price, trailing with the Donchian chain, SuperTrend
etc. Test and see if we have any edge. Use only stocks we already have data for; define a
universe like MidSmallcap-400."

**What we're testing:** Across the liquid MidSmallcap-400 pool (PIT, no survivorship), does
entering a stock on the day it closes at a **new all-time high** and trailing out with a
Donchian / SuperTrend stop produce a **per-trade edge that beats a random-entry control** —
and then, does that survive into an investable portfolio net of cost?

**Success metric:** per-trade net expectancy (G1); then portfolio net Calmar vs NIFTYBEES
and vs a random-entry book (G4).

**Falsification (decided now):** if ATH-entry per-trade net return does NOT beat a
date/stock-matched RANDOM-entry control with the SAME exit rule (t < 2), the "breakout" is
just being-long-a-strong-stock (drift/survivorship) → verdict NO EDGE. (Same discipline
that killed research/87/88 and the 20/200-SMA study.)

## 2. Economic hypothesis

New all-time highs = **no overhead supply** (no trapped buyers-turned-sellers above),
plus a momentum/late-comer-flow effect → price tends to continue. Counterparty: profit-
takers and mean-reversion sellers. Decay risk: crowded/known breakout trade; false
breakouts in choppy regimes; small-caps gap through trailing stops. **This must be shown
to add value OVER simply holding the same strong stock (random-entry control).**

## 3. The Base (G1 probe mechanics — close-only first, kill-cheap)

- **Universe:** MidSmallcap-400 = `rs2.pit_universe` band `combo`=(100,500) by trailing
  traded value → built-in minimum-liquidity + point-in-time (no survivorship). Eligibility
  checked at the most recent month-end.
- **ATH signal:** close = expanding max of the stock's own close history (data from ~2012).
  A **fresh breakout** = today is a new all-time high AND yesterday was not (first day of a
  new-high streak). No re-entry while already holding that name.
- **Entry:** at the ATH-day close.
- **Exit (G1):** close-based **Donchian-N** — exit at the first close below the lowest close
  of the prior N days. N ∈ {10, 20, 30}. (SuperTrend / ATR-Donchian on real OHLC deferred to
  G2 — only built if G1 shows a signal.)
- **Cost:** 0.4% round-trip on each trade.
- **Controls (MANDATORY):**
  - **RANDOM-ENTRY:** same stocks, same N-Donchian exit, entry on random dates (matched
    count) → isolates whether the ATH timing adds anything over just holding the name.
  - **DRIFT / always-in:** buy-hold return of the same names over the matched horizon.
- **Period:** 2014-01 → 2026 (VPS canonical DB).

## 4. Plan (grid + counts)

| Axis | Values |
|---|---|
| Exit (Donchian N) | 10, 20, 30 |
| Cohort | ATH-entry vs RANDOM-entry vs DRIFT |
| ATH freshness | fresh-breakout (base); (later: require prior ATH ≥ K days ago) |

G1 output per (N, cohort): trade count, mean net per-trade return, win%, expectancy,
median hold days, **t-stat of ATH-vs-random difference**. Gate: ATH beats random with t≳2
on ≥2 of 3 N values → proceed to G2 (SuperTrend, OHLC gaps, portfolio construction,
concurrency/sizing, regime gate). Else NO EDGE / CONCLUDED.

## 5. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-08-02 | STATUS 1-4 written; G1 probe building | close-based Donchian, random control |
| 2026-08-02 | G1 done (plain ATH vs random) | ATH beats random on WIN-RATE (+12pp) but NOT expectancy (diff-t 1.46/1.80/1.50 < 2) — fails gate |
| 2026-08-02 | G1b done (ATH + volume) | ATH+vol vs random+SAME-vol-spike = t 1.06/1.09/0.65 — the ATH adds ~nothing once you condition on volume; edge is volume+drift |
| 2026-08-02 | G1c running (consolidation breakout) | tight base + wall-break to new ATH on vol; deciding variant |

**Live findings:**
- **Plain ATH (G1): NO robust edge.** Higher win-rate than random (42-47% vs 30-35%) but per-trade expectancy does not beat a matched random-entry control (diff-t 1.5-1.8 < 2). Both cohorts carried by 2014-26 small-cap bull drift.
- **ATH + volume (G1b): volume EXPLAINS the signal, doesn't rescue it.** vs random-day-with-same-volume-spike, the all-time-high adds t≈1 (1.06/1.09/0.65) — and *weaker* at higher volume thresholds. The apparent edge is "high-volume day in a trending small-cap," not the ATH itself.
- **Verdict so far: the all-time-high per se is NOT an edge** — same drift/survivorship trap as research/87/88/91. Consolidation-breakout (G1c) is the last, strongest test.

## 6. Crash Recovery

- Host VPS `arun@94.136.185.54:/home/arun/quantifyd`, `venv/bin/python`.
- Probe: `venv/bin/python research/100_ath_breakout/scripts/g1_probe.py`
  (writes `results/g1_trades.csv` + prints the cohort comparison). Resumable per-stock.
- Reuses `research/41_.../scripts/02_rs_sweep.py` (rs2: load, pit_universe, month_ends).

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/g1_probe.py` | ATH-entry vs random per-trade probe | yes |
| `results/g1_trades.csv` | per-trade log | yes if small |
| `results/RESULTS.md` | verdict | yes |

## 8. Findings

_(to be filled)_
