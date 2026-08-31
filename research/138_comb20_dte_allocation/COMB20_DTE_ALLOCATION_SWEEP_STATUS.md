# COMB20 Day Allocation — Which DTE Cells the 30% Stop Actually Earns On

STATUS: **SIGNAL — a change to a LIVE book is implied. Nothing deployed.**
research/138 · 2026-09-01

---

## 1. How this question arose

It started as "should we add the V1 + 30% CSL variant to the portfolio". Arun
corrected that twice, and both corrections were right:

- I measured its correlation against a portfolio that **excluded the entire
  CSL/COMB family** (those records live in `csl_paper_state.json`, not the
  `nas_*_trading.db` stores), and reported **0.09**.
- He then pointed out the live Monday book already runs a 30% combined-premium
  stop. It does: `NAS_COMB20 = {DTE0: sl 25, DTE1: sl 30}` at 09:16→15:20, and
  for a Tuesday NIFTY expiry **DTE1 is Monday**.

On the three Mondays where both traded, the correlation is **0.96**. The
candidate is not a new system — it is a replay of a book already running. So the
question became: **is COMB20's day allocation wrong?**

## 2. The 30% stop by DTE — 92 replay days at 10 lots

| DTE | weekday | n | net | mean/day | **t** | worst | maxDD | stops fired |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Tue | 18 | ₹2,78,180 | ₹15,454 | 1.85 | −₹43,385 | −₹45,165 | 5 |
| 1 | Mon | 20 | ₹1,44,805 | ₹7,240 | 1.23 | −₹45,140 | −₹59,630 | 2 |
| 2 | Fri | 18 | ₹1,06,255 | ₹5,903 | 0.94 | −₹63,990 | −₹77,765 | 1 |
| **3** | **Thu** | 18 | **₹1,55,265** | ₹8,626 | **3.85** | **−₹13,875** | **−₹15,790** | **0** |
| 4 | Wed | 18 | ₹39,630 | ₹2,202 | 0.35 | −₹88,300 | −₹88,300 | 1 |
| | **all** | 92 | ₹7,24,135 | ₹7,871 | 2.89 | −₹88,300 | −₹1,21,930 | 9 |

**DTE3 (Thursday) is the only cell that clears significance**, and it does so
with by far the smallest drawdown — −₹15,790 against −₹45k to −₹88k elsewhere —
having **never fired its stop in 18 days**. DTE4 (Wednesday) is the worst on
every measure and owns the book's single worst day.

## 3. The live book, on the same axis

| DTE | day | n | net (10 lots) | mean/day |
|---:|---|---:|---:|---:|
| 0 | Tue | 2 | −₹900 | −₹450 |
| **1** | **Mon** | 3 | **−₹56,630** | **−₹18,877** |
| 2 | Fri | 3 | +₹18,695 | +₹6,232 |
| 3 | Thu | 2 | −₹4,644 | −₹2,322 |

COMB20 is configured to run **DTE0 and DTE1**. DTE1 is where it bleeds.

## 4. Drop-one and keep-only

| variant | net | maxDD | ret/DD |
|---|---:|---:|---:|
| keep all | ₹7,24,135 | −₹1,21,930 | 5.94 |
| **drop DTE1** | ₹5,79,330 | −₹88,555 | **6.54** ← only improvement |
| drop DTE0 | ₹4,45,955 | −₹97,525 | 4.57 |
| drop DTE2 | ₹6,17,880 | −₹1,06,690 | 5.79 |
| drop DTE3 | ₹5,68,870 | −₹1,21,770 | 4.67 |
| drop DTE4 | ₹6,84,505 | −₹1,21,930 | 5.61 |

| single cell | net | maxDD | ret/DD |
|---|---:|---:|---:|
| **only DTE3** | ₹1,55,265 | **−₹15,790** | **9.83** |
| only DTE0 | ₹2,78,180 | −₹45,165 | 6.16 |
| only DTE1 | ₹1,44,805 | −₹59,630 | 2.43 |
| only DTE2 | ₹1,06,255 | −₹77,765 | 1.37 |
| only DTE4 | ₹39,630 | −₹88,300 | 0.45 |

## 5. What two independent sources agree on

**DTE1 (Monday) is the weakest cell COMB20 runs.** The replay says dropping it is
the *only* removal that improves return-over-drawdown; the live book says it has
lost ₹56,630 over three Mondays, worse than every other day combined. Two
different data sources, same conclusion.

**DTE4 (Wednesday) being removed was correct.** THE STACK already dropped it; the
replay independently ranks it last at t = 0.35 with the worst single day.

**COMB20 does not run its best cell.** DTE3 (Thursday) is the strongest by a
distance and is not in COMB20's configured pair.

## 6. What would have to be true before changing anything

- **Multiple testing.** Five cells were examined; one clearing t = 3.85 is still
  significant after a crude five-way correction (p ≈ 0.001 → 0.005), but it is
  one cell of five and should be treated as a hypothesis, not a fact.
- **The live sample is 2–3 days per cell.** It cannot confirm anything on its
  own; its value here is agreeing with the replay on DTE1.
- **One regime, four months, modelled fills.** The same caveat as everything else
  on that page.
- A Thursday COMB variant already exists (`NAS_COMB20_THU`, DTE3 at **SL 20**,
  5 lots), so the Thursday cell is not untried — but at a different stop.

## 7. Recommended next step — NOT a deploy

Run the DTE3 cell at **30%** against the existing `NAS_COMB20_THU` at 20% over
the recorded chain, to separate "Thursday is good" from "30% is good on
Thursday". If Thursday survives that, the change to propose is a reallocation —
drop DTE1, add DTE3 — as a **strategy change with its own evidence and an
after-15:40 deploy**, not a parameter tweak.

---

## PHASE 2 — 2026-08-31 20:1x IST · CONCLUDED

| time | event | notes |
|---|---|---|
| 20:05 | Built the real-money-only book | Roster taken from `nas_day_matrix.json` (`live:true` + DTEs) and `csl_paper_exec.py` `BOOKS` (`"mode":"live"`). **7 live sleeves, not 20.** |
| 20:08 | Pooled-vs-real correction | Real book: 43d, +Rs 2,38,557, maxDD -Rs 36,082, **ret/DD 6.61, t 2.32**. The -Rs 5.99L DD I reported earlier is paper, not Arun's money. |
| 20:12 | Confound test run (`confound.py`) | 5 DTE x 6 stops = 30 cells, re-priced from the untruncated `options_study.json` path. |
| 20:13 | **Thursday resolved: it is the DAY, not the stop** | DTE3 identical from SL20 to no-stop (Rs 1,55,265, t 3.85, DD -Rs 15,790, **0/18 stops**). Live paper twin already runs SL20 = same thing. No stop change to make. |
| 20:14 | Monday: no parameter fix exists | Positive at every stop but t 1.0-1.6 throughout, ~4x Thursday's DD. Stability split shows Rs 1,31,845 -> Rs 12,960 across halves. |
| 20:16 | **CLAIM RETRACTED** | An intermediate "Friday = +Rs 2,86,628, t 2.18, nobody live" was a **lot-sizing artefact** (raw sum over a shadow running 1-10 lots). Normalised: +Rs 83,407, **t 0.41**, top-3 = 272% of total. Not an opportunity. |
| 20:18 | The real finding | The 9:16 suite and the held-straddle/CSL mechanic have near-**opposite** weekday edges (suite t: Mon 2.89, Tue -0.24, Thu -0.37; CSL t: Mon 1.23, Tue 1.85, Thu 3.85) — and the live config runs both on Mon+Tue. |
| 20:20 | Verdict written | `results/RESULTS.md`. **SIGNAL. Nothing deployed.** |

**Outcome:** `NAS_COMB20` is the only live sleeve losing money (-Rs 10,089, t -1.00),
and the cause is day allocation, not parameters: it stacks correlated size on Monday
(the suite's best day, t 2.89) and sits out Thursday (its own best cell, t 3.85, and
a day the suite loses on).

**Recommended action: the cheap one.** Do NOT move real money to Thursday yet —
NIFTY was pulled off Thursdays on 2026-08-27 for SENSEX-expiry margin, and the live
Thursday record is 2 days and negative. Let `NAS_COMB20_THU` accumulate ~8 paper
Thursdays, then re-read. The open live question is whether COMB20's Monday cell
should keep running at all (3 live Mondays, all losing, -Rs 56,630 at 10-lot equiv).

**Flagged, not touched:** `is_live_book()` requires `"mode":"live"`; only `NAS_COMB20`
has it, yet comments assert `CSL_TIMEB_SENSEX` and `CSL_TIMEB2_LIVE` are REAL. Confirm
against the broker — live-trading code, not edited here.
