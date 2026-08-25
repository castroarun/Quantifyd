# research/126 — Four ways to defend a short-premium book — RESULTS

**Verdict: NO EDGE for the portfolio profit trail (Arm A) and for entry-time wings
(Arm B). SIGNAL — not yet investable — for the two shapes that only pay when the book is
already winning: profit-triggered portfolio wings (Arm B2) and symmetric strike
spreading (Arm C). Neither clears its full bar, both have a coherent mechanism and a
real tail effect, and both are proposed as paper twins, not live changes.**

**Plus one finding that is not about trading at all: an 8-lot real-money book was
invisible to every automated view of the account (§1). That is a risk-control defect and
it is reported first because it is the most important thing in this document.**

Stage: **G2/G3 (mechanics + robustness, real data, measured cost model).**
Data: real 1-minute option chain (`options_data.db`, 2026-04-20 → 2026-08-24) + the live
9:16 suite's real per-minute MTM. Cost model throughout: forced mid-session exit
+6.548 pt/leg-side vs +0.178 for a time exit (443 real live leg-sides), plus the exact
Zerodha rate card. READ-ONLY on every database. **No live rule was changed.**

---

## 1. GOVERNANCE GAP — an 8-lot real-money book that nothing was watching

On 2026-08-25 `CSL_TIMEB2_NIFTY` sold **520 qty (8 lots)** of NIFTY 24150 CE+PE at 13:15
and bought them back at 14:30, booking **−₹2,990**. Broker order tags: `TIMEB2_NIFTY`.
That position was **larger than COMB (520 vs 130 qty)** and it did not appear in any
automated view of the book. This study's first harness missed it for exactly the reason
every other view misses it.

| defect | detail |
|---|---|
| **Name collision on real money** | `BOOKS["CSL_TIMEB2_NIFTY"]` in `csl_paper_exec.py` is a **PAPER, 2-lot (qty 130)** book, and it recorded its own separate paper trade the same day (+₹867). The **REAL 8-lot** book has the *same name* and is run by a standalone one-shot, `research/125_expiry_afternoon_straddle/scripts/timeb2_oneshot.py`. Two different things, one name, one of them real. |
| **Not in the daemon's state** | The one-shot publishes only `static/app/timeb2_live.json` + its own `timeb2_live_days.json`. It never writes a `source: REAL` record into `csl_paper_state.json`, so **any harness that derives "live books" from the daemon's `BOOKS` dict or its state file silently drops it.** |
| **Not monitored** | The NAS integrity watchdog scans the NAS variant APIs only; a standalone one-shot is outside its field of view. |
| **The reconciliation table is EMPTY** | `journal.db :: journal_kite_reconciliation` has **0 rows**. The one table designed to catch broker-vs-DB divergence has never been populated. |
| **The journal's `mode` field is wrong** | `journal_trades` flags the force-paper squeeze variants (`NAS-ATM/ATM2/ATM4`) as `LIVE` across 76 days. Anything trusting that field over-counts real money badly. (The `mode` column in the *positions* tables **is** reliable — 287 live legs, every one carrying a real Kite order id, zero live-without-id.) |
| **No broker orderbook is persisted** | Kite `orders()` is today-only and nothing snapshots it daily, so the account's true fill history **cannot be reconstructed for any past date**. Every historical "what was live" claim in this repo is an inference from per-strategy DBs, not from the broker. |

**Scope consequence for this study, stated precisely.** TimeB2's one-shot was written and
first run on 2026-08-25 (commit `1b78873`, same day), and its ledger contains exactly one
day. **The 62–84-day historical samples are therefore not understated by TimeB2** — it did
not exist for them. What the miss did corrupt was the *worked example*, which is corrected
in §2, and the *method*: a book universe must be derived from broker fills, not from a
daemon's dictionary.

**A second scope fact this exposed**, worth Arun's attention independently: the real-money
footprint is far patchier than the book list suggests, by design —
`NAS_ATM_DEFAULTS['live_weekdays'] = (0, 1, 4)` means **real Kite orders only on
Mon/Tue/Fri**. Measured by real order ids: the 9:16 suite has real broker orders on **29**
days, the SENSEX suite on **6** (2026-07-22 → 08-20), the squeeze variants on 10–11 (May–Jul),
916_OTM on 5. This study follows the r/90 convention — *replay the currently-deployed book
over history* — rather than "what was real money that day", because the latter is not
recoverable. That choice is stated everywhere it matters.

**Recommendations (risk control, not trading):** give the one-shot a distinct name; have it
write a `source: REAL` record into the shared state; add a daily broker-orderbook snapshot
so history becomes reconstructable; populate `journal_kite_reconciliation`; and make the
watchdog enumerate positions from the broker rather than from a book list.

---

## 2. The worked example — 2026-08-25, corrected

Rebuilt independently from the live MTM tables, the CSL state, the live JSON, and TimeB2's
own ledger, with TimeB2's curve reconstructed from the 1-minute chain:

| time | portfolio | 916_ATM | 916_ATM2 | 916_ATM4 | **TIMEB2** | TimeB | COMB |
|---|---|---|---|---|---|---|---|
| 13:15 | +11,498 | +1,306 | +3,835 | −1,008 | −26 | +3,640 | +3,750 |
| **14:01 (peak)** | **+18,817** | +1,306 | +4,888 | −1,008 | +4,342 | +3,640 | +4,914 |
| 14:30 | +7,650 | +1,306 | +2,698 | −1,008 | −2,080 | +3,640 | +3,094 |
| 14:33 | +890 | +1,306 | −708 | −1,008 | −2,080 | +3,640 | −260 |
| 15:27 | **−7,240** | +1,306 | −5,388 | −1,008 | −2,080 | +3,640 | −3,711 |

**Peak +₹18,817 at 14:01 → −₹7,240. Give-back ₹26,058.** Against the coordinator's booked
figures (peak +₹19,201 @14:00, booked −₹8,402, give-back ₹27,603) this reconciles: the
difference is **gross MTM vs booked net** — TimeB2 marks −2,080 on the chain but booked
−2,990 once the +6.548 pt/leg-side exit slippage and charges are paid, and TimeB marks
+3,640 vs +3,402 booked. The mechanism, the peak minute, and the shape agree exactly.

The structural point from the original commission is unchanged and confirmed: **the 9:16
suite peaked at +₹5,187 against its +₹12,000 arm threshold — the deployed venue trail could
never have armed.**

What each Arm-B2 trigger would have done **on this specific day** (reported as one day,
not as evidence):

| trigger | arms? | arm time | spot | wings (100-wide) |
|---|---|---|---|---|
| ABS ₹12,000 | YES | 13:00 | 24,157.0 | 24250 CE / 24050 PE |
| ABS ₹15,000 | YES | 13:38 | 24,132.8 | 24250 CE / 24050 PE |
| **ABS ₹20,000** | **no** | — | — | peak 18,817 never reached 20k |

The cell that survives this study's robustness tests (ABS ₹20,000) **would not have fired
today.** The cells that would have fired are the ones that fail the super-winner guard.
That tension is the honest state of Arm B2.

---

## 3. Arm A — the portfolio profit trail. **NO EDGE.** (unchanged by the scope fix)

62 full-book sessions, 2026-05-20 → 2026-08-24. Baseline **₹626,965 total, mean ₹10,112/day,
worst −₹29,950**; give-back median ₹4,137.

- **0 of 132 trail cells beat doing nothing.** Best (`A5000_G2500_ONLYLOSERS`) = **−₹84,427**,
  13.5% of the book's P&L, and improves the worst day by **₹0**.
- The ARM × GIVEBACK plateau map is **monotone to the boundary "no trail"** — the r/116
  signature reproduced on a portfolio construction.
- The deployed suite trail is itself **−₹21,373** (10 fires, 7 needless, ₹35,661 on those).
- Fixed-TP ladder independently reproduces r/90: 30k −85,721 → 5k −730,276.
- **Placebo:** the real trail beats random-minute exiting decisively (arm 10k: 508,510 vs
  placebo p95 314,747) and still loses to naked 626,965 — **skilful early exiting, and skill
  is not the problem.**
- **Why it cannot work on the tail:** a *profit* trail can only arm on a day that reached
  profit. The worst days never do. Structurally incapable of fixing the left tail.

---

## 4. Arm B — wings bought at entry. **NO EDGE.**

The staleness audit **passed** (0% one-sided quotes, median spread 0.3–1.5% of mid,
identical-print runs 1.0–1.2 min, zero sleeve-days excluded by the r/89 volume rule; bought
at ASK, sold at BID), so the numbers are trustworthy — and they are bad. Wings cost
**28–100% of the book's entire P&L**. NIFTY 500-wide costs ₹105,345 to improve the worst day
by **₹396**; SENSEX ≥600-wide leaves the worst day essentially unchanged.

**The decomposition is the finding:** TB_NIFTY 100-wide costs ₹168,103 = **₹4,341 spread +
₹157,794 decay**. Not an execution problem. The wing hands back the theta the book earns.

---

## 5. Arm B2 — wings bought only once the PORTFOLIO is up. **SIGNAL, not established.**

This is the shape Arun actually asked for. 84 days (2026-04-20 → 08-24). Baseline here
**includes TimeB2 replayed on every day as a deployed book** (a modelling choice — it really
ran once): naked total **₹829,145**, mean ₹9,870, worst **−₹72,351**.

**504 cells swept** (12 triggers × 7 distances × 3 coverage × 2 unwind).
**71 beat naked; 118 improve the worst day.**

| cell | armed | total | Δ vs naked | worst | Δ worst | t | wing paid |
|---|---|---|---|---|---|---|---|
| ABS_15000 / 100 / ALL / EOD | 48/84 | 985,207 | **+156,062** | −63,092 | +9,259 | 0.53 | 14 |
| ABS_20000 / 100 / ALL / EOD | 39/84 | 983,062 | **+153,917** | −63,092 | +9,259 | 0.90 | 14 |
| ABS_12000 / 100 / ALL / EOD | 59/84 | 928,183 | +99,038 | **−38,721** | **+33,630** | 0.32 | 14 |
| ABS_20000 / 300 / ALL / EOD | 39/84 | 933,338 | +104,193 | −63,092 | +9,259 | 1.01 | 14 |

**The plateau is broad and interpretable** (Δ total, coverage ALL, unwind EOD):

| trigger | 100 | 150 | 200 | 250 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|---|
| ABS 5,000 | −210,063 | −170,158 | −142,492 | −138,432 | −119,013 | −72,498 | −60,790 |
| ABS 8,000 | −5,944 | −2,410 | −14,951 | −39,933 | −45,265 | −26,275 | −35,573 |
| ABS 10,000 | −15,568 | −38,309 | −35,169 | −10,317 | +4,396 | +1,083 | −13,920 |
| **ABS 12,000** | **+99,038** | +57,394 | +47,052 | +60,868 | +61,560 | +39,706 | +10,533 |
| **ABS 15,000** | **+156,062** | +108,614 | +105,480 | +111,456 | +102,965 | +66,221 | +31,044 |
| **ABS 20,000** | **+153,917** | +135,760 | +121,178 | +119,030 | +104,193 | +66,628 | +35,079 |
| T1400_8000 | −248,598 | −178,038 | −132,820 | −105,636 | −88,816 | −70,465 | −66,323 |

Everything ≥ ₹12,000 is positive at **every distance**; everything ≤ ₹10,000 is negative.
The dividing line is **how often you arm**: ABS 5,000 arms on 92% of days, 10,000 on 75%,
15,000 on 57%, **20,000 on 46%**. Arm too readily and you buy insurance on days that never
needed it. Coverage `ALL` > `BIGGEST` > `ADVERSE`; unwind `EOD` > `RECOVER` (selling the
wings back when the book recovers throws away exactly the protection you bought).

### 5.1 The mechanism — verified, and it is the right one

The overlay is designed to catch **peak-then-collapse** days. There were three such days,
and it caught the two biggest:

| day | portfolio peak | armed | naked close | wing P&L | hedged close |
|---|---|---|---|---|---|
| 2026-07-08 | +21,425 | 13:03 | **−72,351** | +111,819 | **+39,468** |
| 2026-06-12 | +21,161 | 13:27 | **−54,614** | +95,555 | **+40,940** |

And on the bad days that were **never up**, it correctly never armed and cost nothing:
2026-05-06 (peak +14,445, closed −63,092), 2026-04-30 (peak +6,029), 2026-06-03 (peak 0).
That asymmetry — insure only when there is profit to protect — is precisely why this shape
behaves differently from Arm A's trail and Arm B's entry-time wing.

### 5.2 And the reason it is NOT yet investable

- **Super-winner guard fails for the best cells.** 2026-07-08 alone contributes **+₹254,453**
  of ABS_15000/100's +₹156,062 total — **163%**. Remove that one day and the cell is
  **−₹98,391 (t −0.67)**. ABS_12000/100 is worse: top-day 244% of total, ex-top-1 −₹142,197.
- **Only the ABS_20000 family survives it**: top day 73% of total, **ex-top-1 +₹42,098
  (t +0.32)**; ABS_20000/300 ex-top-1 +₹29,192 (t +0.41). Positive, but nowhere near
  significant.
- **OOS:** ABS_20000/100 IS +113,222 → OOS +40,695 and ABS_20000/300 IS +40,215 → OOS +63,978
  (same sign both halves — good). But ABS_12000/100 flips to −4,649 and ABS_15000/250 flips
  the other way (−7,840 → +119,296). Only the high-trigger family is stable.
- **No cell reaches t = 2**, let alone a family-wise bar over 504 cells.
- The whole case rests on **2–3 events in 84 days**. The expectation may well be real; the
  estimate is not precise enough to act on.

**Verdict: SIGNAL.** Coherent mechanism, broad plateau, correct behaviour on the days that
matter, stable sign OOS for the high-trigger family — but one-day-dominated and statistically
indistinguishable from zero. **Paper twin, not a live change.**

---

## 6. Arm C — strike / entry diversification. **SIGNAL on the tail; fails its bar.**

### 6.1 A replay was built, failed reconciliation, and was discarded — read this first

The first Arm-C engine implemented `config.py`'s documented 9:16 rules faithfully: per-leg
30% SL, trail-to-cost, re-enter up to 5×. It produced **−₹437,588** on NIFTY against the live
book's **+₹164,988**, because it cascaded **4.04 cycles/day** against the live book's **1.04
trades/day**. The live trade table says why: real `916_ATM` exit reasons are **58
eod_squareoff, 10 ST_EXIT, and ZERO SL_HIT** — the documented per-leg 30% SL is **dormant in
the live system** (a SuperTrend trail exits first). Those numbers were discarded.

**This is itself worth recording:** the config documents a stop that the live book never
hits, so any study reasoning from config alone — including the exit-rule × offset
interaction this arm was commissioned to measure — would have been wrong. Arm C was
re-based onto three constructions that *do* reconcile: **HOLD** (09:16→15:15, no stop — what
the live suite actually does on 84% of days), **COMB** (the r/116-validated per-DTE combined
SL), and **RUPEE2500** (the ATM2 rupee stop, isolated).

### 6.2 The interaction — the stated prior is REFUTED

The prediction was that a fixed ₹2,500/lot stop is a *larger %-of-credit* move off-ATM and
would therefore fire **less**. It fires **more** (NIFTY stop rate by offset):

| offset | −4 | −2 | 0 | +2 | +4 |
|---|---|---|---|---|---|
| RUPEE2500 stop rate | 53.0% | 43.4% | **28.9%** | 36.1% | 47.0% |
| COMB stop rate | 27.7% | 22.9% | **12.0%** | 25.3% | 30.1% |
| credit (pts) | 310.1 | 261.9 | **240.8** | 252.7 | 293.5 |

The reason is in the credit row: a "straddle" struck away from spot carries **intrinsic
value and net delta**, so it moves *more* in rupees. Both stops are minimised **at the
money**, and for the COMB construction ATM is also the best cell on mean (+₹1,496/day at
offset 0, worse at every offset). **For stopped constructions, spreading strikes hurts.**

### 6.3 Where it does work: symmetric spreading of an unstopped book

Equal-notional, 3 clones × 2 lots, 83 days per venue:

| | NIFTY mean | NIFTY worst | NIFTY credit | SENSEX mean | SENSEX worst | SENSEX credit |
|---|---|---|---|---|---|---|
| ALL_ATM (deployed) | 2,745 | −50,146 | 722 | 1,939 | −48,122 | 2,335 |
| SYM ±1 | 2,316 | −48,775 | 731 | 1,748 | −46,962 | 2,345 |
| SYM ±2 | 2,695 | −39,987 | 755 | 1,547 | −38,747 | 2,376 |
| SYM ±3 | 2,368 | −35,416 | 794 | 1,349 | −36,416 | 2,428 |
| **SYM ±4** | 2,117 | **−31,747** | 844 | 1,254 | **−31,428** | 2,496 |

- **The worst day improves monotonically in k on both venues** — a ~37% tail cut. That is a
  plateau, not a spike.
- **Credit RISES with k** (722→844, 2,335→2,496). So this is **not** "selling less premium".
  The mechanism column reads *structural*, not *downsizing*, for SYM ±2/±3/±4 on both venues.
- **Only SYMMETRIC works.** ALL-UP, ALL-DOWN and LADDER families are erratic and mostly worse
  (NIFTY ALLUP_4 mean −444, worst −64,129), because a one-sided offset adds **directional
  delta**. Symmetric spreading is the only configuration that decorrelates the clones without
  taking a view.

### 6.4 The three nulls — two cleared, the decisive ones not

| null | result |
|---|---|
| (a) deployed ALL_ATM | **cleared on the tail** (−50,146 → −31,747), mean cost not significant in-sample (t −0.06 to −0.51) |
| (b) random-leg placebo (500 draws) | **CLEARED** — NIFTY SYM_4 worst is at the **99.8th percentile**, SYM_3 97.4th; SENSEX SYM_4 99.6th, SYM_3 93.0th |
| (c) just trade smaller at ATM | **SPLIT** — NIFTY SYM_2/3/4 beat downsizing; **SENSEX SYM_k all lose to it** |
| family-wise haircut | **FAILS both venues** (NIFTY max&#124;t&#124; 2.09 vs null-95th 2.97; SENSEX 1.28 vs 2.98) |
| out-of-sample | **FAILS on the mean** — NIFTY SYM_2 IS +989/day → **OOS −1,064**; SYM_3 +790 → −1,516; SYM_4 +978 → −2,196 |

**Verdict: SIGNAL, tail-only.** The tail reduction is real, monotone, placebo-clearing and
mechanistically explained. The claim that it is *free* is **not** established: the mean cost
is unmeasurable in-sample and turns clearly negative out-of-sample. Margin is unchanged
(same lot count; an OTM short is marginally cheaper on SPAN). **Not deployable as measured.**

---

## 7. The four defences on one axis

| defence | cost over sample | worst-day improvement | ₹ per ₹1 of tail cut | verdict |
|---|---|---|---|---|
| Portfolio profit trail (best of 132) | 84,427 | **₹0** | ∞ — buys nothing | NO EDGE |
| Existing suite trail (live today) | 21,373 | ₹0 | ∞ | negative, n too small to act |
| Entry-time wings (NIFTY 200) | 261,108 | 9,804 | 26.6 | NO EDGE |
| **Profit-triggered wings (ABS 20k/100)** | **pays +153,917** | **9,259** | **free** | SIGNAL, one-day dominated |
| **Symmetric strike spread ±4** | 52,144 (NIFTY mean cost) | **18,399** | 2.8 | SIGNAL, fails FWER + OOS |

---

## 8. Honest caveats

- **One regime, ~84 days.** No VIX shock, no gap-down cluster. Arm B2's entire case rests on
  2–3 peak-then-collapse events; Arm C's tail case on a handful of large days.
- **Arm B2's baseline replays TimeB2 (8 lots) on all 84 days.** It really traded once. This
  inflates the book's size and its tail (worst −72,351) relative to what was actually at risk
  on most of those days.
- **The 9:16 suite is never modelled** — it is real per-minute MTM, **rescaled** from its
  5/1/10/2/3-lot eras to the deployed 2 lots/system. Exact in P&L (linear in lots); it cannot
  rescale the market impact of a bigger clip.
- **Arm C tests HOLD/COMB/RUPEE2500 proxies, not the live suite** (whose SuperTrend trail is
  not replayable — see §6.1). Treat the sign as the finding and the magnitudes as indicative.
- **Config documents a dormant stop** (§6.1). Anything in this repo reasoning from
  `leg_sl_pct` should be re-checked against live exit reasons.
- **1-minute granularity understates intrabar breaches**; the bias runs in the trail's favour
  and it loses anyway.
- **Multiple testing:** 139 Arm-A cells, 28 Arm-B, 504 Arm-B2, 12 Arm-C portfolio×venue. No
  haircut is needed for Arm A (nothing beat the null). Arm B2 and Arm C are both explicitly
  declared **not** to survive a family-wise haircut.
- **Margin relief from defined-risk structures is not modelled** — a real argument for wings
  that this study does not price.
- **The broker orderbook cannot be reconstructed historically** (§1), so "which books were
  real money on day X" is an inference, not a fact, for every day before today.

---

## 9. Recommendation — for Arun's sign-off. No live change is proposed.

1. **Do not deploy a portfolio profit trail.** Fourth independent reproduction of "tightening
   defence manufactures losses" (r/114, r/116, r/121–122, now r/126).
2. **Do not buy wings at entry.** The bill is theta, which is the book's own edge.
3. **Fix the governance gap (§1) — this is the highest-priority item in the document** and it
   is independent of every trading question here.
4. **Consider a paper twin of Arm B2** at **trigger ₹20,000, 100–300 wide, coverage ALL,
   unwind EOD**. It is the only cell that survives the super-winner guard with a stable OOS
   sign, its mechanism is verified on the two days that matter, and it costs nothing on days
   the book was never up. Paper first — it would not have armed today.
5. **Consider a paper twin of Arm C** at **symmetric ±2 to ±4 steps**, NIFTY only (SENSEX
   loses to plain downsizing). Expect ~35% less tail for a mean cost of roughly ₹1–2k/day
   that this sample cannot measure precisely.
6. **The incumbent champion remains the clock.** The only book that kept its gain on
   2026-08-25 did so by being flat at 11:00. Before adding any overlay, the higher-EV question
   is whether the full-day books (COMB, the 9:16 suite) should have scheduled exits at all —
   an r/122 window question, priced at a time exit (+0.178 pt) rather than a forced one
   (+6.548 pt).
7. **Dated re-checks for the Ops & Review Center:** Arm B2 and Arm C re-run at ~40 more
   sessions (**2026-11**); the deployed suite trail (−₹21,373 here, 7 of 10 firings needless)
   at the same time.

---

## 10. Files

| file | purpose | committable |
|---|---|---|
| `scripts/stage0_live_recon.py` | live portfolio curve + reconciliation | yes |
| `scripts/stage1_build_book.py` | chain pass → sleeve minute paths + wing price paths | yes |
| `scripts/stage2_trail_sweep.py` | Arm A: 139 variants + nulls + random-minute placebo | yes |
| `scripts/stage3_concentration.py` | strike overlap, cross-book correlation, exit clustering | yes |
| `scripts/stage4_diversify.py` / `stage6_analyse_c.py` | Arm C first pass (COMB shape) | yes |
| `scripts/stage5_wings.py` | Arm B: staleness+liquidity audit, then economics | yes |
| `scripts/stage7_armc_engine.py` | Arm C v2 engine (HOLD / COMB / RUPEE2500 × 9 offsets) | yes |
| `scripts/stage8_b2_wings.py` | Arm B2 engine (504 cells, TimeB2 included) | yes |
| `scripts/stage9_analyse_c.py` | Arm C analysis: interaction, plateau, 3 nulls, OOS, FWER | yes |
| `scripts/stage10_analyse_b2.py`, `stage11_b2_robust.py` | Arm B2 grid + super-winner/OOS | yes |
| `scripts/stage12_worked_0825.py` | the corrected 2026-08-25 worked example | yes |
| `results/*.txt`, `results/*_grid.csv`, `results/sleeve_days.csv` | summaries | yes |
| `results/book_minute.csv.gz`, `wing_minute.csv.gz`, `armc_cells.csv`, `b2_cells.csv`, `diversify_cells.csv`, `trail_daily.csv`, `stage0_live_portfolio.csv` | heavy intermediates | **NO — gitignored** |

**Regeneration** (VPS, `/home/arun/quantifyd`, READ-ONLY, ~12 min total):

```bash
cd /home/arun/quantifyd
for s in stage1_build_book stage4_diversify stage7_armc_engine stage8_b2_wings; do
  nice -n 15 python3 research/126_portfolio_profit_trail/scripts/$s.py
done
for s in stage0_live_recon stage2_trail_sweep stage3_concentration stage5_wings \
         stage6_analyse_c stage9_analyse_c stage10_analyse_b2 stage11_b2_robust \
         stage12_worked_0825; do
  nice -n 15 python3 research/126_portfolio_profit_trail/scripts/$s.py
done
```

**Reproducibility stamp:** `options_data.db` snapshot 2026-08-25 (88 recorded days from
2026-04-20); `nas_916_atm{,2,4}_trading.db` MTM through 2026-08-25; live rules from
`csl_paper_config.json` frozen 2026-08-13T14:17; TimeB2 from
`research/125_expiry_afternoon_straddle/results/timeb2_live_days.json`. Cost model:
SLIP_ENTRY 0.0, SLIP_TIME 0.178, SLIP_STOP 6.548 pt/leg-side + exact Zerodha rate card.
Suite rescaled to 2 lots/system. Seeds: placebo 20260825, FWER 7.
