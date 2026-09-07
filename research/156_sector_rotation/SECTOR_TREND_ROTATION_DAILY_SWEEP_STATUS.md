# Sector Trend Detection — Rotation Across Sectors AND Sector-Gated Stock Selection
**STATUS: DONE - VERDICT NO EDGE (branch A) / NO ADDED VALUE (branch B)** · research/156_sector_rotation · opened 07-Sep-2026 19:45 IST · host VPS 94.136.185.54

---

## 1. The Ask

**What Arun asked (verbatim):**

> "we have many sector indexes like nifty real estate, nifty it, nifty auto, bank nifty so and
> so.... can we do a completely fresh study - if there is a way to figure out the trending(s),
> to-be trending sector(s), either ride on them in some prporotaios or further drill down into
> those sector leaders stocks and get some curated portfolio than can make execllent returns
> (above 20% cagr), even up to or better than our current oa and/or tn. We can use RS, or any
> other way of comparing sectors, be leoborate, do not hv any bias to any existing systems or
> its filters or cirteria or so. Our idea is to also see if this study can complemenet our
> portfolio if not for a standalong king maker"

**What we are actually testing** — two separate questions, both in scope:

- **Branch A — allocation ACROSS sectors.** Is sector leadership predictable enough that
  holding the top-N sectors (in some weighting, rebalanced on some clock) beats (i) holding all
  sectors equally, (ii) NIFTY500 buy-and-hold, and (iii) a random-N-sector null distribution?
- **Branch B — sector as a UNIVERSE FILTER for stocks.** Does identifying leading sectors and
  then holding the leaders *within* them beat the same stock-selection rule run over the whole
  universe with no sector filter, and beat random stocks drawn from the same leading sectors?

**"To-be trending"** is treated as a first-class hypothesis: is there any measurable signal that
*anticipates* sector leadership (acceleration, breadth, oversold-reversal, dispersion regime)
rather than merely confirming it after the fact? Tested explicitly at G1 as forward-IC at leads
of 1 and 3 months.

**"No bias to existing systems"** is honoured as an instruction about *inherited design*: nothing
is copied from Open Alpha (ATH-close trigger, RS>=70, 16 slots, -8% stop, 15-SMA trail) or True
North (Nifty-200 top-8, 100-SMA gate, 15-day Donchian). Signals are derived from first principles
and the data chooses. It is **not** a licence to drop measurement discipline: costs, taxes,
offset ensembles, paired comparison, null controls and plateau checks all still apply — they are
how we avoid fooling ourselves, not house style.

**Success bar (pre-registered, §4.4).** Standalone: **>20% CAGR after tax and costs**, ideally at
or above Open Alpha (34.90% / -26.43% / Calmar 1.33) and True North (19.91% / -25.11% / 0.81).
Complement: the standing blend bar against the current TN 40 / OA 40 / IPO 20 book.

---

## 2. Prior art — what this must differentiate itself from

| Study | What it found | How r/156 differs |
|---|---|---|
| **r/147** SECROT | **KILLED.** 10.6% CAGR, -53.2% DD, corr 0.37/0.39 to TN/OA, 2020 crash -31.6% | r/147 tested **one cell**: 8 real sector indices, 126-day momentum, **top-2**, monthly, **always invested, no gate**, 10 bps, no tax. A single point in a large space, screened only as a *third-sleeve diversifier*. r/156 sweeps the space (signal family x lookback x N x weighting x clock x gate), adds a **20-industry synthetic cross-section back to 2007** (more assets, longer window, includes 2008), and adds **branch B**, which r/147 never touched. If the sweep merely reproduces SECROT's profile, that is the answer and we say so. |
| **r/63** GTAA | Equal-weight Nifty/Gold/Nasdaq **beat** momentum top-1 (Calmar 1.73) — *diversification beat selection* | Therefore the **equal-weight-all-sectors null is a required benchmark**, not a formality. A rotation rule that does not beat EW has found nothing. |
| **r/64** factor-index rotation | Best clean result Nifty->VALUE + Gold + Nasdaq, Calmar 1.83; **Kite's Quality/LowVol/Commodities index series are CORRUPT** | Every sector series is integrity-checked before use (§3). Result below: the nine sector indices are clean. |
| **r/145** full-universe TN | Rejected: +2.2pp standalone but **re-imported OA's smallcap beta** (blend Calmar 1.65 -> 1.47) | Sector baskets are long Indian equity beta by construction. A high standalone CAGR that simply re-buys beta OA already owns is a **kill**, not a win. Correlation and blend value are decisive, not decorative. |

---

## 3. Data reality (probed 07-Sep-2026, `scripts/p0_probe.py`, before any modelling)

### 3.1 Real sector indices — usable, clean, but SHORT

| Series | Bars | From | To | Integrity |
|---|---|---|---|---|
| NIFTYAUTO, NIFTYIT, NIFTYENERGY, NIFTYFINSRV, NIFTYFMCG, NIFTYMETAL, NIFTYPHARMA, NIFTYPSUBANK, NIFTYREALTY | 2,894-2,895 | **2015-01-01** | 2026-09-07 | 0 extra (phantom) days, <=1 missing day vs the NIFTY50 calendar, **no split-scale steps**, <=4 days with abs(ret)>12% (all real: Mar-2020, PSU-bank events) |
| BANKNIFTY | 3,887 | 2011-01-03 | 2026-09-07 | clean — but it is a **subset of NIFTYFINSRV**, so it is excluded from the rotation cross-section and kept only as a variant |
| NIFTY50 / NIFTY500 / NIFTYNEXT50 | 2,895-3,887 | 2011 / 2015 | 2026-09-07 | clean |
| NIFTYMIDCAP150 / NIFTYSMLCAP250 | 3,887 | 2011-01-03 | 2026-09-07 | **1,990 rows are O=H=L=C** — the pre-2015 history is close-only. Use **close only** for these two. |
| NIFTYMEDIA, NIFTYINFRA, NIFTYPVTBANK, NIFTYCONSUMPTION, NIFTYCOMMODITIES | — | — | — | **NOT IN THE DB.** Not used. |

Index series carry `volume = 0` throughout (expected — they are indices, not instruments). No
volume-based filter can be applied at index level.

**The binding constraint: 9 sectors x 11.7 years.** ~140 monthly rebalances and a nine-name
cross-section is a small sample for a rotation rule. One crash (2020), the 2018 and 2022H1
grinds, no 2008. This is stated in every headline number and drives the design below.

### 3.2 Synthetic industry baskets — longer window, at a survivorship price

`backtest_data/{nifty200,niftymidcap150,niftysmallcap250}_official.csv` carry an **Industry**
column (NSE macro-industry). Union = **500 symbols across 20 industries**, all 500 present in
`market_data_unified` (timeframe 'day'), 497 current to 2026-09-07.

Depth: 241 of 500 have data from <=2008-01-01; 296 from <=2015. Per-industry counts from <=2008
range 3 (Diversified) to 37 (Financial Services).

**Construction:** equal-weight daily-rebalanced basket per industry, a stock entering its basket
the day its data starts, industries formed only while they hold **>=5 constituents**. Window
**2007-01-01 ->**, giving the 2008 crash.

**Survivorship is severe and is stated with every synthetic number:** membership is *today's*
Nifty-500 union applied backwards. Names that were in an industry in 2008 and later delisted or
fell out of the index are absent. The synthetic long-window results are therefore a **regime and
robustness check, never a headline**, and every synthetic-derived figure carries the label.

**Validation (mandatory before use, per playbook §3 external-series rule):** each synthetic
basket is correlated against its matching real sector index over the 2015-2026 overlap (daily and
monthly return correlation + annualised drift difference), and the table is published in
RESULTS.md. A basket that does not track its real index is not used as that sector's proxy.

### 3.3 Known defects that touch this study

- **Split adjustment is not retroactive** (MCX, HEG, NAZARA, CUPID ...). Branch B uses
  distance-from-252-day-high as one candidate stock signal -> the affected symbols are re-checked
  for single-day abs(ret) > 40% steps and excluded from that signal's universe if flagged.
- **Phantom holiday rows** NaN-poison `rolling()`. All rolling statistics are computed on the
  `dropna()`'d per-symbol series and re-aligned, never on a union-index frame.
- **Partial candles**: the study window ends 2026-08-29 (last full week) to avoid a live-day
  candle contaminating the final bar.

---

## 4. The Base — what is being tested

### 4.1 Assets

- **SECT9** (headline, tradeable): the nine NSE sector indices, 2015-01-01 -> 2026-08-29.
- **IND20** (long-window robustness, survivorship-flagged): 20 synthetic equal-weight industry
  baskets, 2007-01-01 -> 2026-08-29.
- **Benchmarks**: NIFTY500 (headline), NIFTY50, NIFTYMIDCAP150, NIFTYSMLCAP250, NIFTYBEES.

### 4.2 Signal families (branch A / the sector-trend detector)

Every signal is computed on data <= the rebalance close and traded on the **next day's open**.

| # | Family | Definition | Lookbacks L (trading days) |
|---|---|---|---|
| 1 | ABSMOM | total return over L | 21, 42, 63, 126, 189, 252 |
| 2 | MOM_SKIP | return over L skipping the last 21d (classic 12-1) | 126, 189, 252 |
| 3 | RISKADJ | return over L / realised vol over L | 63, 126, 252 |
| 4 | TSVOTE | count of positive returns over {63,126,252} | — |
| 5 | DISTHIGH | -(1 - close / rolling max close over L) | 126, 252 |
| 6 | MADIST | close / SMA(L) - 1 | 50, 100, 200 |
| 7 | ACCEL | mom(63) - mom(252)*(63/252) — short vs long trend, the **"to-be trending" candidate** | — |
| 8 | REVERSAL | -return over L (cross-sectional mean reversion) | 21, 63, 252 |
| 9 | VOLSCMOM | mom(L) / vol(21) | 126, 252 |
| 10 | BREADTH | share of the industry's constituents above their own SMA(L) — **internal**, IND20 only | 50, 200 |
| 11 | BREADTH_CHG | 21-day change in BREADTH(50) — early-detection candidate, IND20 only | — |
| 12 | LOWVOL | -realised vol over L (a deliberate non-momentum control) | 63, 252 |

**Note on RS:** ranking by "return minus the NIFTY500 return over the same L" is **rank-identical**
to ranking by absolute return, because the benchmark term is common to all sectors. RS therefore
enters the design as a **gate** (hold only sectors whose return beats the benchmark / is positive),
never as a separate ranking axis. This is stated because Arun named RS explicitly.

### 4.3 Books

**Branch A (rotation across sectors).** At each rebalance: rank the cross-section, hold the top N,
weight them, hold to the next rebalance. Costs on turnover.

| Axis | Values |
|---|---|
| signal spec | the G1 survivors (capped at 8) |
| N held | 1, 2, 3, 4, 5 |
| weighting | equal / rank-weighted / inverse-vol / signal-proportional |
| rebalance clock | monthly / quarterly / fortnightly |
| rebalance-day offset | 0, 1, 2, 3 (the deterministic-book analogue of seeds — §6.1) |
| gate | none / sector-own-absolute-momentum > 0 else cash / NIFTY500 > 200-SMA else cash |

**Branch B (sector-gated stock selection).** At each rebalance: pick the top-K sectors by the best
branch-A signal, then hold the top stocks *within* those sectors by a within-sector stock signal,
equal-weighted across slots.

| Axis | Values |
|---|---|
| K sectors | 2, 3, 4, 5 |
| slots | 10, 15, 20 |
| within-sector stock signal | mom126 / mom252-skip21 / riskadj126 / distance-from-252d-high |
| rebalance clock | monthly / quarterly |
| offset | 0, 1, 2, 3 |

### 4.4 Costs, taxes, cash — and the pre-registered bars

- **25 bps per side** headline, with a **cost ladder 25 / 40 / 60** on every finalist.
- **After tax**: 20% STCG / 12.5% LTCG (>365 days) with Indian FY loss-netting settled 1 April.
  Rotation books are almost entirely short-term; this is a material haircut and is the number
  quoted in the verdict.
- **Idle cash 5% p.a.** whenever a gate is in cash.
- **Drawdown for any sub-window is measured from the running peak of the FULL curve**, never the
  window's first bar (the r/154 correction).

**PRE-REGISTERED GATES — written before any result is seen.**

- **G1 (does sector leadership exist at all?)** A signal family proceeds only if its monthly rank
  IC against the forward 1-month return has **abs(t) >= 2.0 on both asset sets, or >= 2.5 on SECT9**,
  **and** the top-minus-bottom tercile spread is monotone across terciles. If **no** family clears
  this, the study stops at G1 and reports **NO EDGE** — that is the cheap kill and it is the
  expected outcome given r/147.
- **G2 standalone bar:** after-tax, net-of-cost, offset-ensemble **median CAGR >= 20%**, **worst
  offset >= 18%**, **Calmar >= 1.0**, and it must beat **all four** of: equal-weight-all-sectors,
  NIFTY500 buy-and-hold, the **95th percentile of a 500-draw random-N-sector null**, and (for
  gated variants) a cash-null at the same average exposure.
- **G3 branch-B bar (the decomposition):** the sector-gated book must beat, on **paired** offsets,
  (a) the identical stock rule over the **full universe with no sector filter**, (b) **random
  stocks** drawn from the same top-K sectors, and (c) the same stock rule over **random sectors**.
  Failing (a) means the sector layer adds nothing; failing (b) means the stock layer is decoration.
- **Complement bar (unchanged house standard):** vs the current **TN 40 / OA 40 / IPO 20** book,
  **+0.10 Calmar or -2pp drawdown at >= equal CAGR after tax**, robust across TN offsets and OA
  seeds, beating the cash-null at the same weight, **correlation < 0.40** to both TN and OA.
- **Falsification:** if the best surviving configuration's advantage over the equal-weight null is
  smaller than the spread across rebalance-day offsets, the finding is noise and is reported as
  such regardless of headline CAGR.

---

## 5. Plan and cell counts (disclosed for multiple-testing discount)

| Phase | What | Cells |
|---|---|---|
| **P0** | Data probe + integrity + synthetic-basket construction and validation vs real indices | — (done) |
| **P1 / G1** | Forward-IC of every signal spec: 12 families x lookbacks = **~28 specs** x forward horizon {1m, 3m} x asset set {SECT9, IND20} x window {full, first half, second half} | **~336** |
| **P2 / G2** | Branch A rotation sweep: 8 signals x 5 N x 4 weightings x 3 clocks x 3 gates = **1,440 configs** x 4 offsets | **5,760 runs** |
| **P3** | Nulls: equal-weight, NIFTY500 B&H, 500-draw random-N-sector null per N, cash-null | ~2,000 draws |
| **P4 / G3** | Branch B: 4 K x 3 slots x 4 stock signals x 2 clocks = **96 configs** x 4 offsets, plus the three paired controls at 100 draws each on finalists | **384 + ~1,200** |
| **P5** | Robustness on finalists: two windows, plateau neighbourhood, cost ladder 25/40/60, per-year, outlier-deletion | ~200 |
| **P6** | Portfolio fit: correlation to TN/OA/IPO, blend weight sweep 5-30% across 12 TN offsets x 10 OA seeds | ~700 |
| **P7** | Report package: YoY house table, curves + drawdown panel, tearsheet, publish, roster refresh | — |

**Total ~= 10,600 evaluated cells.** Any single "winner" must therefore be discounted heavily; only
**plateaus** are reported as findings, and the ranking metric (after-tax Calmar, with the CAGR bar
as a hard filter) is fixed above before the first run.

---

## 6. Status log

**FINAL STATE: DONE.** All phases complete 07-Sep-2026 20:51 IST. Verdict **NO EDGE (branch A) /
NO ADDED VALUE (branch B)**. Published at `/app/backtest/sector-trend-rotation-research156`.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 07-Sep-2026 19:37 | Study opened, VPS reachable, clock checked | 19:37 IST; this study touches no backend |
| 07-Sep-2026 19:42 | P0 data probe run | 9 sector indices clean 2015+; 5 more absent from the DB; 500-symbol / 20-industry map found; 241 symbols have <=2008 history |
| 07-Sep-2026 19:50 | STATUS doc written, sections 1-5 locked BEFORE any modelling | Gates pre-registered |
| 07-Sep-2026 19:49 | **P1 (G1) done** - 388 IC cells | **Real sector indices show NOTHING**: max abs(t) 1.46 at 1m, 2.33 across all 168 SECT9 tests. Synthetic 20-industry panel shows t 3.6-5.6 -> immediately suspect |
| 07-Sep-2026 19:51 | **Basket validation FAILS** | Every synthetic basket out-drifts its real sector index by +4 to +14pp of CAGR/yr (r/154 accepted gold at +0.5pp). Panel barred from headlines |
| 07-Sep-2026 20:14 | **P1b falsification done** | (A) same-8: real t 0.9-1.4 vs synthetic 2.6-2.9. (B) drift-stripped: t only falls 5.2->4.3. (C) **shuffled industry labels still give t 1.0-1.9, 95th pct 3.0-3.5** -> most of the "sector momentum" is stock momentum among survivors |
| 07-Sep-2026 20:17 | **P4 branch-B controls done** | Sector-gated book 32.5% CAGR; **no-sector-filter control 32.9%** - book wins 5 of 16 paired offsets. Sector layer adds nothing |
| 07-Sep-2026 20:36 | **P2 branch-A sweep done** - 11,520 runs + 2,536 nulls | **0 of 1,440 SECT9 configs clear the bar.** Best 16.3%/-36.9%/0.44. EW-all 14.0%/0.32. Midcap150 B&H 18.0%/0.41 beats every cell |
| 07-Sep-2026 20:39 | P5 finalists, cost ladder, stress windows done | r/147 SECROT reproduced after tax at 8.9%/-56.0%/0.16 |
| 07-Sep-2026 20:43 | **P6 blend done** (window-matched after a first run mixed windows) | Every candidate loses CAGR, gains at most +0.04 Calmar, and **loses to a plain cash sleeve** at the same weight. Correlations 0.41-0.54 to the live legs, above the 0.40 ceiling |
| 07-Sep-2026 20:43 | P7 YoY table + growth-of-100 chart with drawdown panel written | House format, DD from the full-curve peak |
| 07-Sep-2026 20:51 | RESULTS.md written; published to `backtests.ts`; frontend built on the VPS | Page and chart both serve 200 |

## 7. Crash recovery - how to resume without Claude

All work lives on the VPS at `/home/arun/quantifyd/research/156_sector_rotation/`.

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd/research/156_sector_rotation
ls -la results/            # every phase writes its CSV here, incrementally
tail -50 results/*.log     # per-phase progress logs
pgrep -af "156_sector"     # is anything still running?
```

Re-run any phase - every script is resume-safe and skips cells already present in its CSV:

```bash
cd /home/arun/quantifyd
flock /tmp/qf_sweep.lock venv/bin/python -u research/156_sector_rotation/scripts/p1_ic.py
flock /tmp/qf_sweep.lock venv/bin/python -u research/156_sector_rotation/scripts/p1b_falsify.py
flock /tmp/qf_sweep.lock venv/bin/python -u research/156_sector_rotation/scripts/p2_rotation.py
flock /tmp/qf_sweep.lock venv/bin/python -u research/156_sector_rotation/scripts/p4_stocks.py
venv/bin/python -u research/156_sector_rotation/scripts/p5_finalists.py
venv/bin/python -u research/156_sector_rotation/scripts/p6_blend.py
venv/bin/python -u research/156_sector_rotation/scripts/p7_report.py
venv/bin/python    research/156_sector_rotation/scripts/publish.py
```

Safe to inspect: everything in `results/`. Do NOT hand-edit the CSVs - delete a row to force its
cell to be recomputed. Nothing in this study writes to `market_data.db` or touches any live or
paper trading service. A backup of the pre-publish `backtests.ts` sits at `/tmp/backtests.ts.bak`.

## 8. Files

| File | Purpose | Committable |
|---|---|---|
| `SECTOR_TREND_ROTATION_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/p0_probe.py` | data reality + integrity probe | yes |
| `scripts/common.py` | data loading, basket construction, book simulator, tax/cost model | yes |
| `scripts/p1_ic.py` | G1 forward-IC of every signal spec + basket validation | yes |
| `scripts/p1b_falsify.py` | same-8 head-to-head, drift-stripped, shuffled-label null | yes |
| `scripts/p2_rotation.py` | branch A sweep + all four nulls | yes |
| `scripts/p4_stocks.py` | branch B sweep + the three paired controls | yes |
| `scripts/p5_finalists.py` | finalist NAVs, cost ladder, stress windows, per-year | yes |
| `scripts/p6_blend.py` | correlation + blend value vs TN/OA/IPO, window-matched | yes |
| `scripts/p7_report.py` | YoY house table + growth-of-100 chart with drawdown panel | yes |
| `scripts/publish.py` | appends the study to `frontend/src/data/backtests.ts` | yes |
| `results/*.csv`, `results/*.log` | per-cell output and progress logs | yes (small) |
| `results/sector_rotation_research156.png` | the factsheet chart | yes |
| `results/RESULTS.md` | final findings + verdict | yes |

## 9. Findings

**VERDICT: NO EDGE (branch A - rotation across sectors) / NO ADDED VALUE (branch B - sector as a
universe filter). Neither is adoptable, standalone or as a complement.** Full write-up in
`results/RESULTS.md`; published report at `/app/backtest/sector-trend-rotation-research156`.

1. **0 of 1,440 rotation configurations** clear the pre-registered bar. Best 16.3% CAGR /
   -36.9% DD / Calmar 0.44 after tax. Equal-weighting all nine sectors gives 14.0% / 0.32;
   NIFTY 500 buy-and-hold 13.2% / 0.35; **Midcap 150 buy-and-hold 18.0% / 0.41 beats every cell**.
2. **Momentum ranks sectors better than chance and it is not worth the concentration.** Against a
   500-draw random-sector null the best configuration sits at the 94th-100th percentile - yet
   rotating at all loses to holding everything. The r/63 "diversification beat selection" lesson,
   on a new asset class.
3. **"To-be trending" found nothing.** Acceleration t = 1.37 (1m) / 1.79 (3m); breadth change
   t = 0.36. Nothing anticipates leadership; momentum only weakly confirms it.
4. **Branch B: the sector layer is a round trip to nowhere.** Sector-gated stock book 32.5% CAGR /
   Calmar 0.85; the identical stock rule with NO sector filter 32.9% / 0.84, winning 11 of 16
   paired offsets. Random sectors cost ~8pp; momentum sectors recover exactly that ~8pp.
5. **A reusable data warning.** Synthetic sector proxies built from today's index membership
   out-drift the real indices by +4 to +14pp of CAGR a year, and **shuffled industry labels
   reproduce most of the apparent "sector momentum"**. Any future sector study must run the
   same-universe head-to-head and the shuffled-label null before believing a wide-panel result.
6. **No complement value.** Correlation 0.41-0.54 to the live legs (ceiling was 0.40); best blend
   improvement +0.04 Calmar at slightly lower CAGR, and a plain cash sleeve at the same weight
   beats every candidate on Calmar.
7. **r/147 is confirmed and extended.** Its single SECROT cell reproduces after tax at 8.9% /
   -56.0% / 0.16 - the worst book here. The sweep shows the kill was not a one-cell artefact.

**The one follow-up worth having** (registered in the Ops & Review Centre, due 07-Mar-2027):
back-fill the nine real sector indices to their 2005 NSE inception. That adds 2008 and roughly
doubles the sample. It is a data-acquisition task, not a modelling one, and it is the only thing
that could reopen this line honestly.
