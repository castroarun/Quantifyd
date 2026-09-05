# research/152 — Multi-Year Breakout (bananapatterns.com) — RESULTS

**Data snapshot:** `backtest_data/market_data.db`, VPS, max date 2026-09-04 · **Run date:** 2026-09-05
**Engine:** `scripts/myb_replay.py` — an extension of the r/142 decoded engine, asserted
**bit-identical** to `bluesky_replay.simulate` with its extras switched off (`--selftest`).
All figures **after Indian tax** (20% STCG / 12.5% LTCG, FY-netted), **net of 25 bps/side**,
idle cash at 5% p.a., **seed-ensemble medians** with the range and the worst seed stated.

---

## VERDICT — three separate answers, because the screen is three different things

### 1. The screen as the site presents it (any multi-year high) — **KILL: it is Open Alpha in disguise**

| Family (highest-close level, no age filter), W2 2010→2026 | signals | % of its signals that are ALSO Open Alpha signals | % of ALL Open Alpha signals it captures |
|---|---|---|---|
| N = 2 years, inclusive | 19,963 | **75.6%** | 91.6% |
| N = 3 years, inclusive | 17,716 | **80.2%** | 86.3% |
| N = 5 years, inclusive | 14,929 | **86.7%** | 78.6% |
| N = 10 years, inclusive | 9,943 | **93.3%** | 62.1% |

Open Alpha (live, ₹10L) buys a close above the prior **all-time-high** close. A multi-year
high that is also an all-time high *is* an OA signal, and that is what this screen mostly
fires on. The pre-registered kill condition (≥ 60% overlap) is met by a wide margin. Running
it would be running OA twice.

### 2. The "multi-year" quality itself — **NO EDGE: requiring an OLD ceiling is subtractive**

The thing that makes the screen "multi-year" rather than "a high" is that the resistance has
*stood for years*. Tested directly (medians across all cells, W2, after tax):

| Resistance must have held for | CAGR | MaxDD | Calmar | trades/yr |
|---|---|---|---|---|
| 0 months (any N-year high) | **22.8%** | −34.8% | 0.67 | 70-80 |
| ≥ 6 months | 12.9% | −19.1% | 0.68 | 20-40 |
| ≥ 12 months | 11.1% | −18.1% | 0.61 | 6-32 |

Return roughly **halves**; drawdown falls in step, so Calmar is flat. The age filter is
**de-levering, not alpha** — a cash-null wearing a chart pattern. The premise of the screen
is the one part of it that does not pay.

### 3. The distinctive residual (a 3-year high that is NOT an all-time high) — **SIGNAL, real, but NOT ADOPTED**

This is the only genuinely new signal set in the family, and it survives on its own:

**Adopted-candidate spec** `N3_close_excl_age0 + trail-15` — close above the highest close of
the previous 3 years, where that level is **below** the stock's all-time high (and the
breakout close stays below it); RS ≥ 70; 20-day median traded value ≥ ₹5 cr; ETFs excluded;
buy-stop **at the pivot** filled `max(pivot, open)`; −8% close stop; exit on a close below
SMA-15; 16 slots @ 6.25% of NAV; no market gate.

| Metric (W2 2010-01-04 → 2026-09-03, **30 seeds**, after tax, 25 bps) | Value |
|---|---|
| CAGR (median) | **23.45%** [21.74 .. 25.37] — worst seed 21.74% |
| MaxDD (median / worst seed) | −25.3% / −28.2% |
| Calmar | **0.93** |
| Trades | 1,929 (115.9/yr) · win 45.1% · avg win +13.7% / avg loss −4.5% |
| Mean per trade, net + after tax | +3.78% · max losing streak 16 · avg hold 17.5 days |
| NIFTYBEES buy-and-hold, same window | 10.42% / −38%-class |
| W1 2020-2025 (the site's window) | 47.71% CAGR / −24.4% DD / Calmar 1.92 |

**It clears the pre-registered standalone bar** (≥ 20% CAGR, Calmar ≥ 0.80, worst seed ≥ 15%).
**It is genuinely not Open Alpha at the position level:** 0% signal-date overlap by
construction and only **3.8-4.4% holding-day overlap** across five seeds — it owns different
names on different days.

**But it fails the pre-registered COMPLEMENT bar on correlation**, which is the bar that
matters for a book that already runs two long-equity breakout systems:

| Pre-registered condition | Result | Pass? |
|---|---|---|
| +0.10 Calmar vs the TN+OA 50-50 baseline | +0.174 at 10% weight, **30/30 paired paths** | ✅ |
| or −2pp MaxDD at ≥ equal CAGR | −1.6pp at 10%, **−2.4pp at 15%**, CAGR −0.13pp | ✅ |
| robust across OA seeds × TN offsets | 30/30 paths at every weight tested | ✅ |
| beats the cash-null at the same weight | +0.086 Calmar and **+2.00pp CAGR** at 10% (30/30) | ✅ |
| **correlation < 0.4 to BOTH legs** | **OA 0.426 daily / 0.535 monthly; TN 0.371-0.379 daily / 0.406-0.438 monthly** | ❌ |

It holds different stocks and still rides the same factor. That is the honest description:
**a third long Indian smallcap-breakout sleeve.**

---

## The comparison that settles it — MYB vs r/147's gold sleeve, same window, same paths

GOLDBEES history starts 2015-01, so the head-to-head runs 2015-01-01 → 2026-09-03 (11.7y),
10 OA seeds × 3 TN offsets, each blend paired against the **same-path** TN+OA 50-50 baseline.

| Third sleeve @ weight | CAGR | MaxDD | Calmar | paired ΔCalmar | paths won |
|---|---|---|---|---|---|
| — (TN+OA 50-50 baseline) | 29.63% | −16.10% | 1.79 | — | — |
| **MYB 10%** | **30.09%** | −14.45% | 2.03 | +0.240 | 30/30 |
| **GOLD 10%** | 28.36% | **−13.37%** | **2.08** | **+0.282** | 30/30 |
| cash-null 10% | 27.10% | −14.07% | 1.87 | +0.095 | 30/30 |
| MYB 15% | 30.31% | −13.59% | 2.17 | +0.376 | 30/30 |
| GOLD 15% | 27.70% | −11.92% | **2.26** | **+0.474** | 30/30 |
| MYB 20% | 30.52% | −12.73% | 2.31 | +0.525 | 30/30 |
| GOLD 20% | 27.03% | −10.54% | **2.49** | **+0.712** | 30/30 |

Correlations on this window: **MYB** +0.368 daily / +0.557 monthly to OA, +0.387..+0.407 daily
to TN. **GOLD** +0.081 / −0.077 to OA, −0.033..−0.040 daily to TN.

**Gold wins the risk-adjusted contest at every weight and at essentially zero correlation.
MYB wins on raw CAGR at every weight** (+1.7 to +3.5pp) because it is equity beta, not a
diversifier. For the question r/146 and r/147 were asked — *what should the third sleeve be?*
— the answer remains **gold**.

### EXPLORATORY, NOT PRE-REGISTERED — they are not mutually exclusive

Because the two sleeves fail in different places, a four-sleeve book beats either
(2015+ window, same 30 paths, paired vs TN+OA 50-50):

| TN+OA | GOLD | MYB | CAGR | MaxDD | Calmar | ΔCalmar | paths won |
|---|---|---|---|---|---|---|---|
| 100% | — | — | 29.63% | −16.10% | 1.79 | — | — |
| 90% | 10% | — | 28.36% | −13.37% | 2.08 | +0.282 | 30/30 |
| 90% | — | 10% | 30.09% | −14.45% | 2.03 | +0.240 | 30/30 |
| **80%** | **10%** | **10%** | **28.81%** | **−11.54%** | **2.43** | **+0.628** | 30/30 |
| 75% | 10% | 15% | 29.02% | −10.79% | 2.58 | +0.811 | 30/30 |
| 70% | 15% | 15% | 28.35% | −9.49% | 2.89 | +1.060 | 30/30 |

**This is a post-hoc weight search on a favourable 11.7-year window and must not be adopted
off this study.** It is flagged as the highest-value next lever, and it needs its own
pre-registered study (r/155-class) with the weight grid and the bar fixed before running.

---

## Robustness battery (adopted-candidate spec, W2)

| Test | Result | Read |
|---|---|---|
| 30-seed ensemble | 23.45% [21.74 .. 25.37], DD −25.3 / worst −28.2 | Path risk is small: 1,929 trades leaves little room for selection luck |
| **Cost ladder** 25 / 40 / 60 bps | **23.45% / 20.39% / 16.90%** CAGR; Calmar 0.93 / 0.78 / 0.62 | ≈ **1.7pp of CAGR per +10 bps per side** — steep; 115.9 trades/yr ≈ 7.2× book turnover |
| Drop the top-10 trades | mean/trade 3.777% → **3.161%** | Not lottery-ticket driven |
| Cap winners at +50% / +100% | 3.164% / 3.652% | Same conclusion — the edge is broad |
| 2020 crash window | **+2.5%** (in-window DD −1.5%) | Flat-to-positive in the crash |
| 2018 grind | **+1.8%** (DD −16.3%) — baseline blend −9.9% | The one window where it genuinely helps the pair |
| 2022H1 grind | **−11.9%** (DD −18.1%) — baseline blend −5.9% | It **loses** in the other grind; no clean "earns in grinds" story |
| 2015-16 | −6.0% | Loses |
| 2011-12 | −0.3% vs baseline +17.0% | Long dead patch |

**Parameter plateau (trail SMA, `N3_close_excl_age0`, W2 Calmar):** 10 → 1.07, **15 → 0.93**,
20 → 0.77, 25 → 0.79, 30 → 0.69, 50 → 0.60, 150 → 0.22. A monotone gradient, not a lone peak.
The identical gradient appears on the **Open-Alpha control** (trail-10 1.31, 15 0.92, 20 0.86,
25 0.82, 30 0.78, 50 0.71), so *short trails are a property of the whole breakout family in
this window, not a discovery about multi-year highs*. Trail-15 was adopted rather than the
better-scoring trail-10 because 10 sits at the edge of the tested range and pushes turnover
to 140 trades/yr — the plateau's honest interior point is 15.

---

## What was swept, and how much of it (multiple-testing disclosure)

| Phase | Grid | Cells | Simulations |
|---|---|---|---|
| A — signal inventory + OA overlap | N {2,3,5,10} × level {close, high} × ATH {incl, excl, athonly} × age {0, 6, 12 mo} | 72 matrices × 2 windows | 0 (counting only) |
| B — G1 | the same 72 on a fixed book, 10 seeds, 2 windows | 144 | 1,440 |
| C — G2 mechanics (OFAT + plateau) | 7 surviving families × 27 arms (stop, 7 exits incl. plateau neighbours, 8 sizings, gate, fill, base quality, cost) × 2 windows | 364 | 3,640 |
| D — adoption | 30 seeds + cost ladder + outlier tests + holding overlap | — | ~70 |
| E — blend | 6 weights × 3 TN offsets × 10 OA seeds × {MYB, cash, gold} + 4-sleeve probe | — | NAV arithmetic |
| | | **~580 configurations** | **~5,150 book simulations** |

Any single winner in this study should be discounted as best-of-~580. That is precisely why
the adoption bar was written down **before** the first run and why the correlation leg was
not relaxed after the blend numbers came in looking good.

---

## Data integrity — what was checked and what was done about it

- **Phantom holiday rows: CLEAN.** Every trading day 2010→2026 scanned for the signature
  (row count < 50% of the local 21-day median **and** > 85% zero volume): 2 days only
  (2014-04-24, 2014-10-15). The 2026-01-15 purge is intact.
- **Split-scale defect: PRESENT and handled.** 131 unadjusted price-scale steps detected on
  the traded universe (1-day close ratios within 12% of 2 / 2.5 / 3 / 4 / 5 / 10×). Because a
  multi-year lookback straddles such a break, each affected symbol was **blacked out for
  entries from `d − N years` to `d + 20 days`**. Cost of the blackout: **0.1-2.4% of signals**
  — reported per cell in `results/phaseA_signals.csv` (`blackout_cost_pct`). The mitigation
  can only remove signals, never manufacture them.
- **A NaN bug was found and fixed before any result was used.** `rolling(W, min_periods=W)`
  on a union-index wide frame demands zero missing rows in W trading days; N = 10 collapsed to
  **1 symbol / 32 signals**. Fixed to `min_periods=1` plus a separate per-symbol
  `prior non-NaN rows ≥ N×252` history mask. Phase A was re-run from scratch. This is the
  same failure class that disabled r/142's SMA-200 gate.
- **Engine equivalence proven, not assumed:** `simulate_ext` reproduces
  `bluesky_replay.simulate` to the rupee (₹2,259,279 both, 304 trades both) when the new
  mechanics are off.

---

## Honest caveats

1. **Survivorship.** The universe is today's symbol list applied to the past; delisted names
   are absent. Every number here is flattered, the pre-2015 ones most.
2. **Window depth is the binding constraint, and it is the screen's own fault.** The database
   holds 2 symbols in 2000-2002 and only 527 from 2005, so an N-year-high screen cannot be
   tested before 2010 (N ≤ 5) or 2015 (N = 10). **2008 is untestable for this family** — the
   one crash that would discriminate a breakout book is out of reach.
3. **N = 10 results run on a different (2015+) window** than N ≤ 5 and are therefore not
   directly comparable; their apparent strength is partly a friendlier sample. On the common
   2020-25 window N = 3 / N = 5 beat N = 10.
4. **"All-time high" means all-time *within our data*.** For a 2005-listed stock the ATH in
   2020 is a 15-year high. Pre-2000 peaks (1992, 2000 bubbles) are invisible, so the
   exclusive/inclusive split is approximate for the oldest names.
5. **Turnover is high**: 115.9 trades/yr on 16 slots ≈ 7.2× book/yr, all STCG. The cost slope
   (−1.7pp CAGR per +10 bps/side) means execution quality, not the signal, decides whether
   this is a 23% book or a 17% book.
6. **Capacity not separately measured** beyond the ₹5cr traded-value floor inherited from OA.
   At a ₹10L book this is irrelevant; at ₹1cr+ it needs its own check.
7. **The 4-sleeve result is exploratory** and post-hoc. It is a hypothesis, not a finding.
8. **No replication gate was run**: the site's exact dials and its published headline numbers
   for this screen were never legible. Every setting here is a swept axis, so nothing in this
   study either confirms or refutes what bananapatterns.com claims for it.

---

## Next levers

1. **Pre-registered 4-sleeve study (highest value).** TN / OA / GOLD / MYB weight grid, bar
   and metric fixed in advance, both windows, seeds and offsets, cash-null and gold-only
   nulls. The exploratory probe says +0.63 Calmar at 80/10/10; that number must be re-earned
   under pre-registration before it means anything.
2. **If the dials arrive, run the replication gate** — encode their settings verbatim,
   reproduce their trade list and headline numbers, report the match, then stop.
3. **Do not re-test the age filter.** Its dose-response is monotone and negative across 24
   cells and two windows; treat "the level must be old" as a closed question.
4. **The r/154 hand-off is written**: `results/myb_equity_seeds.csv` (30 seeds, daily,
   after-tax, cash 5%) and `results/myb_adopted_spec.json`.

---

## Files

| File | Contents |
|---|---|
| `results/phaseA_signals.csv` | 72 signal matrices × 2 windows: counts, blackout cost, OA overlap |
| `results/phaseB_g1.csv` | 72-cell G1 sweep, both windows, 10 seeds |
| `results/phaseC_g2.csv` | 364-cell mechanics sweep (OFAT + trail plateau) |
| `results/myb_equity_seeds.csv` | **30-seed daily equity, adopted spec — for r/154** |
| `results/myb_adopted_spec.json` | **the spec — for r/154** |
| `results/final_robustness.csv` | cost ladder, outlier deletion, per-window rows |
| `results/final_yoy.csv` | per-year median return + median intra-year drawdown |
| `results/oa_overlap.csv` | signal-date and holding-day overlap with Open Alpha |
| `results/blend152_corrected.csv` · `blend152_paired.csv` · `blend152_windows.csv` | same-window blend, paired deltas, per-window |
| `results/myb_vs_gold.csv` | head-to-head against the r/147 gold sleeve |
| `results/four_sleeve_exploratory.csv` | **exploratory, not pre-registered** |
| `results/yoy_table.html` · `yoy_table.csv` | house-format YoY table |
| `results/myb_curve_vs_indices.png` | growth of ₹100 (log) + drawdown panel vs NIFTY 50 / Midcap 150 / Smallcap 250 |
