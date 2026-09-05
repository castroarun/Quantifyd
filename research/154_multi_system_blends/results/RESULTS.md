# research/154 — Six-Sleeve Correlation & Blend Matrix — RESULTS

**Data snapshot:** `backtest_data/market_data.db`, VPS, max date 2026-09-04 · **Run date:** 2026-09-05
**Host:** VPS 94.136.185.54 · **Scripts:** `scripts/{build_sleeves,rebuild_gold,blend_matrix,analyze,overlap,frontier,report}.py`
All figures **after Indian tax** (20% STCG / 12.5% LTCG, FY-netted), **net of 25 bps/side**, idle
cash 5% p.a., **paired across 360 paths** (30 Open-Alpha seeds × 12 True-North rebalance-day
offsets), monthly rebalanced. **Cells disclosed: 789 (subsets × weights × panels) + 5,301 +
1,992 (frontier enumeration) + 90 (correlations) = 8,172**, each over up to 360 paired paths.

---

## VERDICT: **STRATEGY (candidate)** — the deployed pair is under-diversified, and the fix is
## two satellites (IPO-Base and GOLD), not a third breakout sleeve

Three findings in order of how much they should change what we do:

1. **The deployed TN+OA pair's worst drawdown in twenty years is the 2008 crash: −16.5%, not
   −2.4%.** Research/146 and /151 measured the 2008 window starting **2008-01-01**, which is
   *after* the December-2007 peak, so the drawdown from that peak was invisible. Measured from
   the running peak of the actual curve, the pair loses −15.9% peak-to-trough (Dec-2007 →
   Jun-2008) and −17.15% on daily marks. **The standing conclusion that "the TN gate plus OA's
   stops have already stripped the crash tail" is withdrawn.** The pair has a crash tail.
2. **Correlation says the book owns one factor.** OA↔VCP is **0.74 daily / 0.75 monthly**;
   at position level **87% of Open Alpha's signals are also VCP signals** and the two books
   hold the same stock on the same day 42–49% of the time. VCP (r/151 NO EDGE) is Open Alpha
   wearing a different screen. MYB shares 90% of its signals with VCP. The only two sleeves
   that are genuinely different things are **IPO** (0.21–0.25 daily to OA, **0.0% signal and
   0.0% holding-day overlap — literally not one shared symbol-day in 16 years**) and **GOLD**
   (≈0 to everything, negative monthly).
3. **197 of 1,767 enumerated weight vectors beat the deployed pair on all three windows** at
   equal-or-better CAGR, on ≥288/360 paired paths, against a cash null *and* against an
   IPO-beta-matched null. Every one of them holds gold; almost all hold IPO; and the ones that
   win most cut True North's weight. This is a **broad plateau, not a peak** — which is the
   robustness evidence, and also the honest warning that many neighbouring answers are equally
   good, so the exact weights are not the finding. The direction is.

**Recommended portfolio — the honest one:** the deployed pair plus **two** satellites, not one.
**OA 40% / TN 25% / IPO 20% / GOLD 15%** is the best vector that keeps both live books, caps the
unproven sleeve at 20% and gold at 15%: **28.2% CAGR / −10.8% MaxDD / Calmar 2.61** vs the pair's
**27.7% / −17.0% / 1.68** on 2006-04→2026-08, winning on 360/360 paths against the pair, 360/360
against the cash null and 358/360 against the beta-matched null. **But IPO has never traded, not
even on paper.** The deployable-today step is therefore the gold-only leg, with IPO going to a
paper soak first (§9).

---

## 1. The six sleeves and where their curves came from

| Sleeve | What | Prior verdict | Curve | Standalone on 2006-04→2026-08 (CAGR / MaxDD / Calmar) |
|---|---|---|---|---|
| **OA** Open Alpha | LIVE ₹10L. Close above prior all-time-high close; 16 slots @6.25%; −8% close stop; SMA-15 trail; no gate | LIVE | generated here, `bluesky_replay.py`, **30 seeds** | 34.58 / −25.58 / 1.36 |
| **TN** True North | LIVE momentum. Nifty-200 top-8, monthly rebalance, NIFTYBEES-100SMA weekly gate, 15-day Donchian stop | LIVE | generated here, `tn_sweep.py`, **12 offsets** | 19.98 / −18.89 / 1.08 |
| **VCP** r/151 | 30-day-high pivot breakout | **NO EDGE** | `vcp_equity_seeds.csv` | 36.23 / −35.56 / 1.02 |
| **MYB** r/152 | 3-year-high close that is not an all-time high | **SIGNAL**, not adopted | `myb_equity_seeds.csv` | (2010+ only) 23.4 / −17.7 / 1.29 |
| **IPO** r/153 | IPO-Base MID: listed ≤6 months, 25-day base, 8 slots @18.75%, −8% stop, SMA-20 trail, +25% TP | **STRATEGY candidate** | `ipo_equity_seeds.csv` | 31.74 / −17.47 / **1.84** |
| **GOLD** r/147 | GOLDBEES buy-and-hold | candidate | GOLDBEES 2015+ · **rebuilt** daily reconstruction before that | 13.54 / −26.67 / 0.51 |

**A path** = (OA seed *s*, TN offset *o*) → 360 paths; within a path VCP/MYB/IPO use the same
seed index *s*. Every A-vs-B figure below is the **distribution of paired differences**, never
unpaired medians.

**True North's offset dispersion is large and was previously under-sampled.** Across the full
12 rebalance-day offsets TN's CAGR runs **14.9%–25.0%**; r/146 cached only offsets 0/4/8
(20.9 / 22.0 / 16.6), which happens to miss both tails.

---

## 2. Data defect found and fixed: the gold reference series was missing 40 of 274 months

Research/147's cached `gold_inr_ref.csv` (the reconstruction that extends gold before
GOLDBEES's 2015 start) is a **monthly** series with **40 of its 274 months absent** — 14.6%.
Two causes, both in the Yahoo *monthly* candles it was built from:

- Yahoo's monthly `GC=F` and `INR=X` series themselves drop months (43 and 22 missing);
- their epoch stamps carry a US/UTC offset, so bars like `2004-03-31 23:00` land in the wrong
  month, collide with the real March bar, and `duplicated(keep='last')` deletes one.

A sparse monthly series makes `pct_change()` silently span two months, mis-stating every
pre-2015 gold return. **Fixed** (`scripts/rebuild_gold.py`): pull the **daily** series instead
(`GC=F` from 2000-08, `INR=X` from 2003-12), stamp months with a +12h offset so a timezone shift
cannot cross a month boundary, and align onto the NSE trading calendar. Result: **0 missing
months**, 2005-01-03 → 2026-09-04.

Validation against the real instrument over the 2015+ overlap (2,889 days / 140 months):

| | value | usable for |
|---|---|---|
| **monthly** return correlation | **0.878** (r/147's sparse series scored 0.788) | ✅ monthly-rebalanced blends |
| **daily** return correlation | **0.390** | ❌ — COMEX close vs NSE close is a timing mismatch, not a data error |
| annualised drift (GOLDBEES − reconstruction) | **−1.00 pp/yr** | acceptable for yearly cells |

The reconstruction is written to `results/gold_nav.csv` only — **never into `market_data.db`** —
and every figure that touches pre-2015 gold is labelled below.

---

## 3. The correlation matrix (the direct answer to the ask)

**Panel B — 2006-04 → 2026-08, the longest common window (MYB cannot exist here).**
Seed-median; the OA↔TN reference point every new pair is judged against is in bold.

**Daily returns**

| | OA | TN | VCP | IPO | GOLD |
|---|---|---|---|---|---|
| **OA** | 1.000 | **0.421** | 0.749 | **0.211** | 0.076 |
| **TN** | **0.421** | 1.000 | 0.473 | **0.220** | −0.037 |
| **VCP** | 0.749 | 0.473 | 1.000 | 0.269 | 0.041 |
| **IPO** | **0.211** | **0.220** | 0.269 | 1.000 | −0.003 |
| **GOLD** | 0.076 | −0.037 | 0.041 | −0.003 | 1.000 |

**Monthly returns**

| | OA | TN | VCP | IPO | GOLD |
|---|---|---|---|---|---|
| **OA** | 1.000 | 0.512 | 0.767 | 0.344 | −0.049 |
| **TN** | 0.512 | 1.000 | 0.525 | 0.268 | −0.071 |
| **VCP** | 0.767 | 0.525 | 1.000 | 0.395 | −0.075 |
| **IPO** | 0.344 | 0.268 | 0.395 | 1.000 | −0.053 |
| **GOLD** | −0.049 | −0.071 | −0.075 | −0.053 | 1.000 |

**Panel A — 2010-01 → 2026-08, all six sleeves** (gold 2010-14 = labelled reconstruction).

**Daily** / *monthly* below the names:

| | OA | TN | VCP | MYB | IPO | GOLD |
|---|---|---|---|---|---|---|
| **OA** | — | 0.357 / *0.432* | 0.738 / *0.750* | 0.412 / *0.502* | **0.226** / *0.353* | 0.076 / *−0.080* |
| **TN** | | — | 0.434 / *0.466* | 0.371 / *0.409* | **0.210** / *0.271* | −0.037 / *−0.126* |
| **VCP** | | | — | 0.583 / *0.612* | 0.289 / *0.377* | 0.041 / *−0.096* |
| **MYB** | | | | — | 0.290 / *0.408* | −0.027 / *−0.067* |
| **IPO** | | | | | — | −0.003 / *−0.067* |

Seed ranges are tight (e.g. OA↔VCP daily 0.704–0.771 across 900 seed pairs; OA↔TN daily
0.314–0.391 across 360). Full detail incl. Panel C in `results/p1_correlations.csv`.
**Daily correlations involving GOLD are computed on real GOLDBEES data only (2015+)** because
the reconstruction's daily correlation is a timing artefact; monthly uses the full window.

**Reading:** everything in the breakout family sits at 0.21–0.75 to Open Alpha. Only IPO
(0.21–0.25) is meaningfully below the OA↔TN reference of 0.42, and only gold is at zero.

---

## 4. Position-level overlap — two sleeves can correlate modestly and still be the same trades

Signal overlap is computed on the raw screens (before slot competition, so it is seed-free);
holding-day overlap is the median across 5 seeds. Window 2010-01 → 2026-09.
IPO's trading calendar was reconstructed from the DB and **validated: 100.00% of the 20,244
recorded holding periods reproduced exactly.**

| Pair | shared signals | % of A's signals | % of B's signals | holding-day overlap (% of A / % of B) |
|---|---|---|---|---|
| **OA ~ VCP** | 14,893 | **87.0%** | 51.1% | **48.6% / 41.5%** |
| **VCP ~ MYB** | 1,740 | 6.0% | **90.2%** | 4.0% / 22.0% |
| OA ~ MYB | 0 | 0.0% | 0.0% | 0.6% / 2.9% |
| **OA ~ IPO** | 0 | **0.0%** | **0.0%** | **0.0% / 0.0%** |
| VCP ~ IPO | 0 | 0.0% | 0.0% | 0.0% / 0.0% |
| MYB ~ IPO | 0 | 0.0% | 0.0% | 0.0% / 0.0% |

- **VCP is Open Alpha.** 87% of OA's signals are VCP signals and they hold the same name on
  the same day nearly half the time. r/151's NO EDGE verdict is confirmed from a second angle,
  and VCP subtracts in every blend (§5).
- **A correction to the brief:** the "MYB shares 75–93% of its signals with OA" figure belongs
  to the *raw* multi-year-high family that r/152 killed. The **adopted** MYB residual (a 3-year
  high that is *not* an all-time high) shares **0%** of its signals with OA by construction —
  its 90% overlap is with **VCP**, not OA. Same conclusion, different partner.
- **OA and IPO have never once held the same stock on the same day.** That is what a genuine
  diversifier looks like at position level, and it is why IPO's blend value survives the
  controls that kill everything else.

---

## 5. All 57 subsets, equal weight (the "all combinations" answer)

Full table: `results/p2_subsets.csv` (140 equal-weight cells across three panels).
Panel B, 2006-04 → 2026-08, sorted by Calmar, baseline in bold:

| Subset (equal weight) | CAGR | MaxDD | Calmar | ΔCAGR | ΔDD | ΔCalmar | Calmar wins | vs cash-null |
|---|---|---|---|---|---|---|---|---|
| OA+TN+IPO+GOLD | 26.06 | −8.03 | **3.224** | −1.62 | +8.66 | +1.552 | 360/360 | +0.977 (360/360) |
| OA+IPO+GOLD | 27.89 | −8.99 | 3.103 | **+0.02** | +7.94 | +1.458 | 360/360 | +1.173 (360/360) |
| OA+VCP+IPO+GOLD | 30.13 | −10.11 | 2.976 | +2.37 | +6.81 | +1.338 | 360/360 | +0.806 (360/360) |
| OA+TN+VCP+IPO+GOLD | 28.19 | −10.11 | 2.829 | +0.56 | +6.75 | +1.162 | 360/360 | +0.284 (357/360) |
| TN+IPO+GOLD | 22.65 | −8.92 | 2.489 | −4.99 | +8.34 | +0.895 | 360/360 | +0.664 (320/360) |
| OA+TN+GOLD | 23.71 | −10.64 | 2.207 | −4.04 | +6.57 | +0.648 | 360/360 | +0.352 (360/360) |
| OA+TN+IPO | 29.59 | −13.69 | 2.152 | +1.95 | +3.31 | +0.537 | 360/360 | +0.230 (330/360) |
| OA+GOLD | 24.92 | −12.71 | 1.957 | −2.81 | +4.28 | +0.294 | 360/360 | +0.289 (360/360) |
| **TN+OA (deployed)** | **27.74** | **−17.01** | **1.678** | — | — | — | — | — |
| OA+TN+VCP | 30.60 | −22.00 | 1.398 | +3.01 | −5.24 | −0.269 | 13/360 | **−0.545 (0/360)** |
| OA+VCP | 35.77 | −26.61 | 1.347 | +7.99 | −9.76 | −0.328 | 33/360 | **−0.350 (0/360)** |
| TN+VCP | 28.09 | −24.54 | 1.152 | +0.46 | −7.74 | −0.504 | 0/360 | −0.437 (14/360) |

Every subset containing **VCP** is beaten by its own cash-null. Every subset containing
**GOLD** improves the drawdown; every subset containing **IPO** improves both. On Panel A
(2010+, all six) the ordering is the same, topped by OA+TN+MYB+IPO+GOLD at 20% each
(24.95 / −7.84 / **3.158**).

---

## 6. The control that changes the answer: IPO is 80% cash

Research/153's own G3 print records that the IPO sleeve is **invested only 19.6% of NAV on
average**, and the yearly table makes it visible — the sleeve returns exactly the 5% idle-cash
yield with ~zero drawdown in whole years:

| Year | 2008 | 2009 | 2012 | 2013 | 2014 |
|---|---|---|---|---|---|
| IPO sleeve return | **+4.2%** | **+3.1%** | +2.5% | **+5.1%** | **+5.0%** |

2013 and 2014 the book took **no trades at all**. A plain cash-null at the same weight does not
catch this, so a second control was built: the **IPO beta-matched null** — replace IPO with
**19.6% OA + 80.4% cash** at the same weight, which reproduces its average market exposure but
none of its selection or timing.

| Book | Panel A (2010+) | Panel B (2006+) | Panel C (2015+) |
|---|---|---|---|
| OA+TN+IPO+GOLD **25/25/25/25** | +0.239 Calmar, **232/360** ❌ | +0.812, 360/360 ✅ | +0.083, **224/360** ❌ |
| OA+TN+IPO+GOLD **30/30/20/20** | +0.100, **237/360** ❌ | +0.556, 356/360 ✅ | +0.148, 296/360 ✅ |
| OA+TN+IPO+GOLD **40/40/10/10** | +0.198, 346/360 ✅ | +0.247, 359/360 ✅ | +0.273, 341/360 ✅ |
| OA+IPO+GOLD **33/33/33** | +0.301, 336/360 ✅ | +0.950, 360/360 ✅ | +0.592, 360/360 ✅ |

(✅ = beats the beta-matched null on ≥288/360 = 80% of paired paths.)

**Read it plainly:** beyond roughly 20% IPO weight, the extra Calmar on the post-2010 windows is
**not distinguishable from de-levering**. Below that it is a genuine gain on all three panels.
IPO's crash protection is *structural* — the IPO window closes in a bear market, so the sleeve
is automatically flat — which is a real, repeatable mechanism, but it is "having nothing to
buy", not alpha. Both statements belong in any pitch of this book.

---

## 7. The weight frontier — enumerated, not cherry-picked

Every weight vector on a 5% grid over {OA, TN, IPO, GOLD} (**1,767 vectors × 3 panels = 5,301
cells**) plus a 10% grid including MYB (**996 × 2 = 1,992**). A vector is **ADMITTED** only if,
on **all three panels**: median CAGR ≥ the deployed pair's, and it beats the pair, the cash null
**and** the beta-matched null on **≥288/360 paired paths** each.

**197 of 1,767 admitted.** Weight distribution among the admitted set:

| GOLD weight | 0% | 5% | 10% | 15% | 20% | 25% | 30% | 35% |
|---|---|---|---|---|---|---|---|---|
| admitted vectors | 15 | 35 | 40 | 37 | 27 | 20 | 18 | 5 |

| TN weight | 0% | 5% | 10% | 15% | 20% | 25% | 30% | 35% | 40% | 45% |
|---|---|---|---|---|---|---|---|---|---|---|
| admitted vectors | **43** | 33 | 25 | 23 | 20 | 15 | 13 | 12 | 8 | 5 |

**The frontier's unconstrained optimum drops True North entirely** (OA 20 / IPO 50 / GOLD 30 →
Calmar 3.16 / 3.27 / 3.96 on A/B/C). We do **not** recommend that; it is reported because
hiding it would be dishonest. Applying operational constraints — keep both live books
(TN ≥ 15%), cap the never-traded sleeve (IPO ≤ 20%) and cap gold (≤ 20%, r/147's caveat that
2015-26 was a strong gold decade) — leaves **73 admitted vectors**, topped by:

| Weights | A CAGR / DD / Calmar | B CAGR / DD / Calmar | C CAGR / DD / Calmar |
|---|---|---|---|
| **OA 40 / TN 25 / IPO 20 / GOLD 15** | 27.10 / −10.77 / **2.506** | 28.21 / −10.77 / **2.612** | 32.10 / −10.77 / **2.966** |
| OA 45 / TN 20 / IPO 15 / GOLD 20 | 27.03 / −10.78 / 2.489 | 28.08 / −10.78 / 2.590 | 31.73 / −10.78 / 2.927 |
| OA 45 / TN 25 / IPO 15 / GOLD 15 | 27.26 / −11.37 / 2.382 | 28.31 / −11.37 / 2.474 | 32.01 / −11.37 / 2.802 |
| OA 40 / TN 30 / IPO 20 / GOLD 10 | 27.29 / −11.47 / 2.380 | 28.44 / −11.48 / 2.476 | 32.38 / −11.47 / 2.817 |
| **OA 45 / TN 35 / IPO 10 / GOLD 10** (minimum change) | 26.82 / −12.18 / 2.204 | 27.90 / −12.34 / 2.277 | 31.31 / −12.18 / 2.554 |

The neighbourhood is **contiguous and monotone** — OA 0.40–0.50, TN 0.15–0.35, IPO 0.10–0.20,
GOLD 0.10–0.20 is admitted throughout. That is plateau behaviour, and it is the reason this
survives an 8,172-cell multiple-testing discount: the winner is not a lone cell.

**Gold-only (the actionable-today subset, no unproven sleeve): 14 of the 197 admitted vectors
contain no IPO**, and every one of them **raises OA's weight and cuts TN's**:

| Weights | B CAGR / DD / Calmar | vs pair |
|---|---|---|
| **OA 60 / TN 15 / GOLD 25** | 28.02 / −13.31 / **2.095** | +0.27pp CAGR, +3.70pp DD, 360/360 |
| OA 65 / TN 5 / GOLD 30 | 28.35 / −13.79 / 2.045 | +0.66pp, +3.22pp, 360/360 |
| OA 55 / TN 30 / GOLD 15 | 27.84 / −13.78 / 2.004 | +0.09pp, +3.23pp, 360/360 |
| OA 55 / TN 40 / GOLD 5 | 28.28 / −15.72 / 1.795 | +0.55pp, +1.29pp, 358/360 |

Note **r/147's recommended 45/45/10 is NOT admitted** — it fails the strict "CAGR ≥ the pair"
condition by −1.13pp on the 2006+ window. Gold pays for its drawdown reduction with return
unless OA's weight rises to fund it.

---

## 8. Per-window behaviour — and the 2008 correction

Median across 360 paths, **drawdown measured from the running peak of the full curve**, not
from the window's own first bar. MYB cannot be evaluated in 2008 at all (history starts
2010-01); gold before 2015 is the labelled reconstruction.

| Book (Panel B) | 2008 crash ret / dd | 2020 crash ret / dd | 2018 grind ret / dd | 2022H1 grind ret / dd |
|---|---|---|---|---|
| **TN+OA 50-50 (deployed)** | +0.8 / **−16.5** | −1.4 / −8.3 | −10.2 / −12.7 | −5.3 / −11.0 |
| OA 45 / TN 35 / IPO 10 / GOLD 10 | +4.0 / −11.5 | −0.3 / −1.1 | −7.3 / −10.2 | −3.9 / −9.3 |
| OA 40 / TN 25 / IPO 20 / GOLD 15 | +7.3 / −7.5 | +0.8 / −0.4 | −4.4 / −8.7 | −2.5 / −7.5 |
| OA+IPO+GOLD 33/33/33 | **+12.2 / −4.1** | +1.5 / −0.3 | −0.7 / −8.3 | −1.1 / −7.4 |
| OA+TN+VCP 33 each | −9.1 / −22.0 | −1.3 / −6.8 | −12.7 / −13.8 | −8.5 / −11.3 |
| *OA alone* | +1.5 / −17.7 | −3.7 / −9.7 | −10.8 / −18.3 | −7.2 / −17.4 |
| *TN alone* | −0.5 / −16.4 | +1.0 / −13.7 | −10.7 / −14.7 | −4.7 / −10.9 |
| *VCP alone* | −26.5 / −35.6 | −1.1 / −8.1 | −18.4 / −18.6 | −14.3 / −16.8 |
| *IPO alone* | +3.8 / **−2.7** | −5.5 / −5.9 | +3.2 / −11.9 | −3.1 / −11.9 |
| *GOLD alone* | **+29.2** / −13.1 | +13.9 / 0.0 | +4.6 / −14.4 | +6.1 / −12.5 |

**Prior claim withdrawn.** The brief carried a structural finding that "the TN gate plus OA's
per-stock stops have already stripped the crash tail (blend drawdown inside the 2008 window is
just −2.4%), so crash-alpha candidates solve a problem the pair does not have." That number was
an artefact of starting the window on 2008-01-01. The pair's true 2008 drawdown is **−16.5%**
(monthly marks) / **−17.15%** (daily marks) and it is the **single deepest drawdown the pair has
taken in twenty years**. Crash protection is therefore *not* a solved problem, and the second
half of the finding — that the pair also bleeds in grinds — still stands (2018 −12.7%,
2022H1 −11.0%).

The 2008 improvement decomposes honestly into two mechanisms, both real, neither alpha:
**gold's crisis rally in rupee terms** (+29.2%, on *reconstructed* data — this is the single
biggest reason to treat the 2008 column as directional) and **IPO being structurally in cash
because no company lists into a crash**.

**Daily-marked robustness** (`results/p8_daily_marked.csv`, 120 paths, honest intra-month
drawdown; the 2015+ panel uses real GOLDBEES only):

| Book | 2006+ CAGR / daily MaxDD / Calmar | 2015+ CAGR / daily MaxDD / Calmar |
|---|---|---|
| TN+OA 50-50 (deployed) | 26.52 / −17.15 / 1.536 | 30.00 / −16.84 / 1.797 |
| OA 45 / TN 35 / IPO 10 / GOLD 10 | 26.97 / −13.78 / 1.994 | 30.75 / −12.57 / 2.417 |
| OA 40 / TN 25 / IPO 20 / GOLD 15 | 27.56 / −12.90 / 2.188 | 31.70 / −10.98 / 2.862 |
| OA+IPO+GOLD 33/33/33 | 27.56 / −11.79 / 2.318 | 32.07 / −10.50 / 3.073 |

Daily marking deepens every drawdown as expected but does not change a single ranking.

---

## 9. The two registered open questions

### Q1 — r/152: "MYB+OA (28.71 / −14.5 / 1.98) scored above the deployed TN+OA (26.16 / −16.1 / 1.56) on 2010-2026, but that window excludes 2008."

**Reproduced, and it is not evidence to act on.** On Panel A (2010-01→2026-08) OA+MYB 50-50
gives 29.13 / −14.50 / **2.017** vs the pair's 26.63 / −15.69 / 1.685 — +2.60pp CAGR, +0.316
Calmar, winning **314/360** paired paths. On Panel C (2015+) it is stronger still (+0.613,
349/360). Both confirm r/152's arithmetic.

**What evidence would settle it, and why we still cannot get it.** The claim being tested is
"a second breakout sleeve is a better partner for OA than True North's regime gate." Settling it
requires observing the candidate through a systemic equity crash, and **MYB's construction makes
that impossible**: its pivot is a three-year high, so it cannot produce a signal before 2010 no
matter how much data we add. No re-run fixes this; only a future crash will.

**What the common-window work says in the meantime — it weakens the case.** Panel B lets us ask
the same question of the sleeves that *do* reach 2008:

- Removing True North entirely (**OA alone**) loses to the pair on **334 of 360 paths**
  (−0.296 Calmar), and takes a −17.7% drawdown in 2008 and −18.3% in the 2018 grind.
- The pair's own worst moment is 2008 (§8), so the window MYB cannot be tested in is precisely
  the window that decides the question.
- Every 2006-testable substitute for TN that *does* win — IPO, gold — wins because it is
  **uncorrelated** (0.21 and ≈0), not because it is a better breakout. MYB's correlation to OA
  is **0.412 daily / 0.502 monthly** and it shares 90% of its signals with VCP, which is Open
  Alpha. MYB is the same factor with a different label.
- Directly: at equal satellite budget, **adding MYB is worse than adding more gold** (Q2 below).

**Verdict on Q1: the MYB+OA result is real on its window and is not a reason to change the
book.** It is the third time this project has found that a second smallcap-breakout sleeve looks
good post-2010 (r/62, r/145, r/152) and the third time the reason has been the missing crash.

### Q2 — r/152 exploratory: "80% TN+OA / 10% GOLD / 10% MYB scored Calmar 2.43 vs 2.08 gold-only."

**REFUTED.** The comparison used gold at **half the weight** (10%) as the "gold-only" reference,
which is a smaller allocation, not a null. Against the correct null — **gold-only at the same
total satellite weight** (OA 40 / TN 40 / GOLD 20) — the mix **loses**:

| Panel | 80/10/10 (Calmar) | gold-only 20% null | paired ΔCalmar | 80/10/10 wins |
|---|---|---|---|---|
| A (2010+) | 2.212 | **2.326** | **−0.094** | **91/360** |
| C (2015+) | 2.551 | **2.663** | **−0.092** | **60/360** |

For reference, against the *mis-specified* null (gold 10%, Calmar 1.968) the mix does "win"
+0.257 — which is how the original 2.43-vs-2.08 arose. It beats a cash null (+0.302, 360/360)
and a MYB-only null (+0.135, 355/360), so the *mix* is better than either extreme; it is simply
**not better than spending the whole satellite budget on gold**. **Do not label this a finding.**

---

## 10. YoY house-format table

Panel B, 2006-04 → 2026-08, after tax, net of 25 bps/side, median of 360 paired paths, monthly
rebalanced. Each cell = **annual return with the intra-year max drawdown (from the running peak)
in brackets beneath**. Benchmarks are excluded from the best-of picks.
Machine-readable: `results/p8_yoy.csv`.

| Year | TN+OA 50-50 (deployed) | OA45 TN35 IPO10 GOLD10 | OA40 TN25 IPO20 GOLD15 | OA+IPO+GOLD 33/33/33 | NIFTY 50 (NIFTYBEES) | BEST CAGR | LEAST DD | BEST OVERALL |
|---|---|---|---|---|---|---|---|---|
| 2006 | +15.0 (−4.8) | +17.6 (−3.8) | +20.4 (−3.8) | **+23.6 (−3.2)** | +15.9 (−11.8) | OA+IPO+GOLD | OA+IPO+GOLD | OA+IPO+GOLD |
| 2007 | **+97.4 (−5.9)** | +91.7 (−4.8) | +88.7 (−3.9) | +77.7 (−1.8) | +53.0 (−7.8) | deployed | OA+IPO+GOLD | deployed |
| 2008 | −13.9 (−15.9) | −8.3 (−11.0) | −4.5 (−8.3) | **+5.6 (−4.1)** | −52.1 (−55.2) | OA+IPO+GOLD | OA+IPO+GOLD | OA+IPO+GOLD |
| 2009 | **+62.5 (−14.3)** | +51.0 (−7.9) | +42.5 (−3.6) | +26.1 (−4.1) | +75.6 (−54.9) | deployed | OA40 TN25 | deployed |
| 2010 | +10.5 (−15.0) | +15.1 (−10.5) | +18.6 (−7.9) | **+25.3 (−5.7)** | +18.6 (−20.6) | OA+IPO+GOLD | OA+IPO+GOLD | OA+IPO+GOLD |
| 2011 | −6.9 (−9.0) | −1.3 (−7.2) | +2.3 (−6.1) | **+12.3 (−4.4)** | −24.0 (−24.3) | OA+IPO+GOLD | OA+IPO+GOLD | OA+IPO+GOLD |
| 2012 | **+14.9 (−7.8)** | +13.2 (−3.2) | +11.9 (−2.5) | +8.9 (−1.2) | +26.5 (−19.7) | deployed | OA+IPO+GOLD | OA45 TN35 |
| 2013 | **+9.0 (−4.3)** | +6.2 (−4.4) | +4.9 (−4.0) | −0.0 (−6.0) | +7.2 (−11.3) | deployed | OA40 TN25 | deployed |
| 2014 | **+66.4 (−4.5)** | +53.0 (−3.7) | +43.4 (−2.9) | +24.9 (−3.4) | +31.6 (−3.5) | deployed | OA40 TN25 | deployed |
| 2015 | +2.4 (−9.2) | +3.6 (−7.2) | +4.6 (−5.8) | **+5.7 (−4.1)** | −4.3 (−10.4) | OA+IPO+GOLD | OA+IPO+GOLD | OA+IPO+GOLD |
| 2016 | +21.6 (−6.8) | +22.5 (−5.0) | +24.2 (−4.1) | **+25.3 (−3.0)** | +4.0 (−20.8) | OA+IPO+GOLD | OA+IPO+GOLD | OA+IPO+GOLD |
| 2017 | **+67.9 (−1.9)** | +63.1 (−1.2) | +60.0 (−0.3) | +51.8 (0.0) | +29.9 (−2.7) | deployed | OA+IPO+GOLD | deployed |
| 2018 | −12.9 (−12.9) | −10.3 (−10.7) | −8.1 (−10.0) | **−4.4 (−8.3)** | +4.8 (−11.0) | OA+IPO+GOLD | OA+IPO+GOLD | OA+IPO+GOLD |
| 2019 | +4.1 (−15.2) | +7.3 (−11.8) | +9.5 (−9.1) | **+15.0 (−4.4)** | +13.6 (−7.0) | OA+IPO+GOLD | OA+IPO+GOLD | OA+IPO+GOLD |
| 2020 | **+84.4 (−8.3)** | +82.7 (−1.2) | +81.9 (−0.8) | +74.3 (−0.5) | +15.4 (−28.8) | deployed | OA+IPO+GOLD | OA45 TN35 |
| 2021 | **+103.9 (−2.0)** | +87.3 (−3.5) | +75.5 (−4.9) | +53.3 (−7.1) | +26.0 (−3.9) | deployed | deployed | deployed |
| 2022 | +6.5 (−11.7) | +7.3 (−9.9) | +9.5 (−8.7) | **+10.9 (−6.9)** | +5.5 (−10.2) | OA+IPO+GOLD | OA+IPO+GOLD | OA+IPO+GOLD |
| 2023 | +48.0 (−9.0) | **+48.1 (−7.2)** | +47.8 (−6.1) | +45.2 (−3.9) | +21.0 (−7.3) | OA45 TN35 | OA+IPO+GOLD | OA40 TN25 |
| 2024 | +46.1 (−8.0) | +48.9 (−5.6) | +51.6 (−4.2) | **+52.9 (−1.1)** | +10.4 (−8.3) | OA+IPO+GOLD | OA+IPO+GOLD | OA+IPO+GOLD |
| 2025 | +8.0 (−12.2) | +13.0 (−10.8) | +14.9 (−10.4) | **+26.7 (−8.5)** | +11.7 (−13.8) | OA+IPO+GOLD | OA+IPO+GOLD | OA+IPO+GOLD |
| 2026¹ | +20.3 (−5.3) | +25.3 (−5.4) | +31.4 (−4.5) | **+44.2 (−4.5)** | −6.9 (−14.5) | OA+IPO+GOLD | OA40 TN25 | OA+IPO+GOLD |
| **FULL** | **27.74 / −17.01 / 1.68** | **27.90 / −12.34 / 2.28** | **28.21 / −10.77 / 2.61** | **27.89 / −8.99 / 3.10** | 10.67 / −55.16 / 0.19 | | | |

¹ 2026 is eight months (Jan–Aug). All columns share the identical window 2006-04 → 2026-08.

**The shape of the trade-off is the whole story.** All four books finish within 0.5pp of each
other on CAGR over twenty years, and the diversified ones do it with **half the drawdown**. The
deployed pair wins the biggest up years (2007, 2009, 2014, 2020, 2021 — it is the most levered
to one factor); the diversified books win **14 of 21 years on return-net-of-drawdown** and every
single bad year.

Figure: `results/multi_system_blends_research154.png` (growth of ₹100, log, with drawdown panel).

---

## 11. Honest caveats — read these before acting

1. **IPO has never traded.** Not live, not on paper. Its listing-date table was built in r/153
   from a heuristic that rejects bulk data-onboarding waves; that table is validated but new.
   Everything IPO contributes here rests on a 30-seed backtest of a screen decoded three hours
   ago. **A 20% allocation to it is a research recommendation, not a deployment instruction.**
2. **IPO is 80% cash and has multi-year dead zones** (2013 and 2014: zero trades). Its
   diversification is partly structural cash. Beyond ~20% weight, the beta-matched null shows
   the extra Calmar is indistinguishable from de-levering on two of three panels.
3. **IPO capacity is the binding constraint at scale.** Young listings are small and thin. At
   20% of a ₹10L book (₹2L across 8 slots) it is fine; at ₹1cr it is not obviously executable,
   and r/153's capacity note should be re-derived before any size increase.
4. **Gold pre-2015 is reconstructed** (XAU × USDINR, daily, monthly correlation 0.878 to the
   real instrument, −1.0pp/yr drift). The 2008 column — the single most persuasive row in this
   study — rests on it. Treat 2008 as directional, not decision-grade.
5. **2015-2026 was a strong gold decade.** Gold's −19% year (2013) is inside our window and is
   visible in the sleeve table, but gold has no carry and a 1980-2000-style two-decade dead
   period is outside any data we have.
6. **Survivorship.** All equity sleeves run on Kite's current instrument list; delisted names
   are absent. 2006 has ~528 priced symbols, so the early window is survivorship-flattered for
   OA, VCP and IPO alike. This inflates absolute CAGR across the board; it should bias the
   *relative* comparison less, since all four books draw from the same universe.
7. **Multiple testing.** 8,172 cells were run. The defence is the plateau (197 admitted vectors
   forming one contiguous region, §7), the paired win counts (≥288/360 required, most are
   360/360), and three independent nulls — not the size of any single number.
8. **Monthly rebalancing is assumed frictionless.** Rebalancing four sleeves monthly means real
   turnover between books, real costs and real tax events that are **not** modelled at the
   blend level (they are modelled inside each sleeve). A quarterly-rebalance sensitivity is
   owed before deployment.
9. **Not tested, and why:** (a) blend-level rebalancing cost/tax, per (8); (b) whether IPO's
   edge survives a live paper soak — that is the next gate, not a backtest question; (c) any
   sleeve outside the six named; (d) leverage or a cash-drag-financed overlay.

---

## 12. Next levers

1. **Take gold now, at the frontier's weights, not r/147's.** The gold-only admitted set says
   the pair should become roughly **OA 55-60 / TN 15-30 / GOLD 15-25**, which both adds the
   diversifier and acts on r/144's standing conclusion that more return comes from weighting OA
   up. r/147's 45/45/10 is *not* admitted (CAGR shortfall).
2. **Paper-soak IPO before it gets a rupee.** Pre-registered pass criterion, dated review, ops
   entry — see the STATUS doc §5. Only then revisit the 20% weight.
3. **Retire VCP and MYB from consideration, permanently.** VCP is Open Alpha at 87% signal
   overlap; MYB is VCP at 90%. Both are logged in the dead-ends table.
4. **Re-open the crash question.** With the 2008 measurement corrected, "the pair has no crash
   tail" is false, and crash-alpha candidates that r/146 rejected on that basis deserve one
   re-look — specifically the ones killed for solving a problem we thought we did not have.
5. **Re-audit every prior window-drawdown figure in r/146 through r/153** for the same
   window-start artefact.

---

**Reproducibility stamp.** Data snapshot `market_data.db` max date 2026-09-04, VPS. Costs 25 bps
per side; tax 20% STCG / 12.5% LTCG with FY netting; idle cash 5% p.a. Paths = 30 OA seeds ×
12 TN offsets. Scripts and every result CSV are committed under
`research/154_multi_system_blends/`.
