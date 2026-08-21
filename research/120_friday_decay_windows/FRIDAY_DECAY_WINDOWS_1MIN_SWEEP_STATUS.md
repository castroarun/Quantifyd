# Friday Decay Windows — Where Does Premium Bleed Safest, and Is a Second TimeB Slot Worth It?

STATUS: **DONE** (2026-08-21) — verdict: **SIGNAL (already captured) on part 1, NO EDGE on part 2. Do not add a second Friday slot.**

## 2. The Ask

**What Arun asked (2026-08-21):** "can you study if there is any time window on a Friday where
SX and/or NIFTY decays better with least possibility of volatile moves? Is it good to have a
second slot for TimeB if there is any advantage to consider?"

**What we are actually testing.** Two questions, in order — the second only matters if the
first has an answer:

1. **Is there a Friday window that decays well with low move-risk?** For each venue, across a
   grid of start times and durations, what is the net P&L per lot AND the adverse-move
   exposure? A window that earns by sitting through danger is not the same as one that earns
   because nothing happens in it.
2. **Does a SECOND slot add anything?** Not "is slot B profitable in isolation" but **does it
   add value on top of slot A** — marginal P&L, correlation with slot A's outcome, extra
   round-trip cost, and the margin it locks up while slot A may still be open.

**Current Friday state (the baseline to beat):** NIFTY TimeB runs **DTE2 10:00–12:00 SL20**;
SENSEX has **no live Friday cell** (its DTE4 window was removed on 19-Aug as one of the grid's
weakest). NIFTY COMB also runs full-day 09:16→15:20, so any new slot must be judged as an
addition to a book that is already in the market.

## 3. The Base

- **Stage A — options truth:** `options_data.db :: option_chain`, 1-minute, 2026-04-20 →
  2026-08-21, READ-ONLY. Fridays only. NIFTY (DTE2) and SENSEX (DTE4). ~17 Fridays.
- **Stage B — the volatility clock, long history:** `market_data.db :: market_data_unified`,
  `SENSEX`/`NIFTY 50`, timeframe `minute`, 2021-01-01 → 2026-08-21 (~280 Fridays). Options do
  not exist that far back, so measure the *underlying*: realised vol by minute-of-day, mean
  absolute move per 15-min bucket, frequency of large adverse excursions by time of day.
  **This is what answers "least possibility of volatile moves" with a real sample.**
- **Grid:** start times 09:20 → 14:30 in 15-min steps; durations 45 / 60 / 90 / 120 min and
  hold-to-15:20. Combined-SL 20% (the deployed TimeB shape) plus a no-stop arm for contrast.
- **Costs:** 0.5 pt/leg-side NIFTY, 1.0 SENSEX, plus Rs30/leg-side/lot. A second slot pays a
  full extra round trip — that cost must be charged, not waved away.

## 4. The traps that decide whether this study is worth anything

1. **Multiple testing is the main threat.** ~22 starts x 5 durations x 2 venues is 220 cells on
   17 Fridays. Something will look brilliant by chance. Requirements: pre-register the grid,
   report how many cells were tried, and **only accept a winner whose NEIGHBOURS also work** —
   a real window is a plateau, an artefact is an isolated cell. Report the surface, not the max.
2. **Separate "decays well" from "avoided trouble".** Report both net P&L *and* the
   distribution of maximum adverse excursion inside the window. The ideal window has low MAE,
   not merely a good mean.
3. **Stage B must corroborate Stage A.** If the best 17-Friday window does not sit in a
   genuinely calm part of the day across 280 Fridays, treat it as noise and say so.
4. **The second slot must be judged as a marginal addition**, jointly with slot A and with
   COMB already running — including whether the two slots' bad days coincide (they will if both
   are short gamma on the same index; measure it).
5. **Margin reality:** a second slot that only works while COMB and slot A are open may not be
   fundable. Note peak concurrent margin for any recommended pairing.

## 5. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-21 ~13:0x IST | Question raised after the Fri NIFTY-vs-SENSEX stop analysis | brief written, agent launched |
| 2026-08-21 13:0x IST | Agent picked up; coverage probe done | option_chain: 17 Fridays 2026-05-01..08-21 both venues, 1-min, ~0.8s/day query. **2026-08-21 EXCLUDED (today, market still open, partial day) -> n=16 complete Fridays.** |
| 2026-08-21 13:1x IST | Stage B data reality found | `market_data_unified` has **SENSEX 1-minute 2021-01-01..2026-08-20 (508,378 rows)** but **NO NIFTY 1-minute at all** - NIFTY intraday exists only as `NIFTY50` **5-minute** 2015-02-02..2026-07-17. Plan adapted: SENSEX 1-min is the primary volatility clock; NIFTY 5-min is a shape-only cross-check; a SENSEX 1-min-vs-5-min control quantifies how much 5-min understates excursions. |
| 2026-08-21 13:1x IST | Live baseline confirmed from the frozen config | `backtest_data/csl_paper_config.json`: `CSL_TIMEB_NIFTY` DTE2 = **10:00-12:00 SL20** (Friday); `CSL_TIMEB_SENSEX` has DTE0+DTE1 only (**no Friday cell**); `CSL_TIMEB2_NIFTY` (the existing 2L paper "second slots" book) has DTE0+DTE1 only (**no Friday cell**). So the question is exactly: add a DTE2 cell to TIMEB2, and/or a DTE4 cell to TB-SENSEX. |
| 2026-08-21 13:0x IST | Stage A launched (grid + all-start control) | 22 starts x 5 durations x 2 arms x 2 venues = 440 cells, plus the every-minute all-start baseline. Ran niced -10 alongside research/116; ~4 min. |
| 2026-08-21 13:0x IST | Stage B launched (volatility clock) | SENSEX 1-min 274 Fridays, SENSEX 5-min resample (resolution control), NIFTY50 5-min 542 Fridays. Done in ~6 min. |
| 2026-08-21 13:1x IST | **DATA TRAP FOUND — 2 of the 16 "Fridays" are exchange holidays** | 2026-05-01 and 2026-06-26: the recorder polls on holidays and captures a FROZEN chain (1 distinct underlying print all "session"; `market_data_unified` has 0 rows for both). Every window on those days books exactly minus the round trip and reads as a real losing Friday. Added a holiday guard (reject a day with <50 distinct spot prints) and re-ran the whole pipeline. **n = 14 clean Fridays, 2026-05-08 -> 08-14.** |
| 2026-08-21 13:1x IST | Surface + Westfall-Young haircut | **No individual cell survives** the family-wise correction on any of the 4 venue-arm families (best observed max\|t\| 6.25 vs null-95 7.43; family-wise p 0.10-0.42). The accepted claim is therefore the 3-BLOCK shape, not a cell. |
| 2026-08-21 13:1x IST | Block test passes on both venues and both arms | MORNING(09:35-11:05) minus MIDDAY(11:20-13:20), paired by Friday: NIFTY +550 (t 2.56, p 0.024, 11/14 Fridays), SENSEX +611 (t 2.48, p 0.028, 12/14). Same sign on the NOSTOP arm -> the shape is in the premium path, not in the stop. |
| 2026-08-21 13:1x IST | **Stage B REFUSES to corroborate the "calm" half of the question** | The calmest Friday window over 274/542 Fridays is 11:20-12:00 (20.2 bp mean 45-min excursion, 17% of Fridays >30bp); the morning that PAYS is the second-most dangerous (32.0 bp, 44%). Spearman(long-run risk, sample net) = **+0.31 p=0.0011** (NIFTY) / +0.17 p=0.07 (SENSEX). The 5 calmest windows all LOSE; the 5 most dangerous all EARN. |
| 2026-08-21 13:1x IST | Part 2 marginal analysis done | Every candidate second slot fails: 12:00-14:00 negative (12:00-13:00 collects **zero gross**), 14:00-15:20 positive but **r(COMB,B) 0.58-0.62** with a book already in the market, pre-A 09:20-10:00 is the worst cell in the study (-535/lot, wins 2/14). SENSEX Friday morning looks best in isolation (+660, t 5.15) but is **r 0.71 with the NIFTY slot A that runs the same hours** and pushes peak Friday margin to 46-51% of capital. |
| 2026-08-21 13:2x IST | RESULTS.md written, STATUS -> DONE | Verdict: keep 10:00-12:00 SL20, add no second slot, open no SENSEX Friday cell. Optional non-urgent lever (move A to 09:35-11:35, +173/Fri at t 1.99) explicitly NOT recommended now - it buys P&L by taking more move-risk on 14 days of evidence. |

## 6. Crash Recovery

Read-only on both DBs; no live state touched. `market_data.db` is 30 GB — always filter by
symbol AND timeframe. Scripts in `scripts/`, outputs in `results/`.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `FRIDAY_DECAY_WINDOWS_1MIN_SWEEP_STATUS.md` | this file | yes |
| `scripts/*.py` | window grid + volatility clock + marginal-slot test | yes |
| `results/*.csv` | the full surface, not just the winner | yes |
| `results/RESULTS.md` | verdict + recommendation | yes |

## 8. Findings

**Full write-up: `results/RESULTS.md`. Verdict: SIGNAL (already captured) on part 1, NO EDGE on
part 2.**

### The headline

The question contains a false premise, and that is the finding. On a Friday the time of day that
**decays best** and the time of day with the **least possibility of volatile moves** are opposite
ends of the session — and they are opposite for a reason: the premium available in a window is
the price of the risk carried through it. Rank-correlation between a window's long-run risk and
its sample P&L is **+0.31 (p = 0.0011)** on NIFTY and **+0.17 (p = 0.07)** on SENSEX. Positive.
There is no calm-and-profitable corner to move into.

### Part 1 — is there a good Friday window?

Yes, and **the live book is already in it.** NIFTY TimeB's Friday DTE2 cell **10:00-12:00 SL20**
returns **+400 Rs/lot/Friday, 13 wins in 14, worst -344, t = 4.69, mean MAE 353 Rs/lot** - the
best-behaved thing on a Friday in this data, and it survives leave-one-out without a sign flip.

The surface (all 440 cells in `results/surface_cells.csv`, printed in `surface_report.txt`) has a
clean three-block shape that repeats on **both venues and both stop arms**:

| block | NIFTY SL20 | SENSEX SL20 |
|---|---|---|
| MORNING 09:35-11:05 | **+223** (86% win, t 2.54) | **+261** (86%, t 3.12) |
| MIDDAY 11:20-13:20 | -326 (36%, t -1.72) | -350 (43%, t -1.47) |
| LATE 13:35-14:30 | +226 (71%, t 1.24) | -65 (57%, t -0.25) |

MORNING minus MIDDAY, paired by Friday: **NIFTY +550 (t 2.56, p 0.024, 11/14)**, **SENSEX +611
(t 2.48, p 0.028, 12/14)**, same sign on the NOSTOP arm.

**But no individual cell survives the multiple-testing haircut** (Westfall-Young max-t over each
110-cell family: observed max|t| 3.54-6.25 vs null-95 6.56-7.43, family-wise p 0.10-0.42). Anyone
quoting a single best window is quoting the maximum of a 110-cell search on 14 days.

### The "least volatile" half - answered from 274 / 542 Fridays, and it overturns Stage A

The calmest Friday window is **11:20-12:00** (20.2 bp mean 45-min excursion, 17% of Fridays move
>30 bp). That is exactly where a straddle earns nothing: the five calmest windows in the long
sample returned **-34, -90, -141, -159, -394 Rs/lot**, and a NIFTY 12:00-13:00 straddle collects
**zero gross** - the whole loss is the round trip. The five most dangerous windows all earned
(+143 to +756). The 14-Friday option sample happened to show LOW MAE in the morning; the
274/542-Friday sample says that is a small-sample accident. **The morning window is not safe; it
is well paid.**

### Part 2 - is a second slot worth it? NO.

| candidate B | B mean | r(COMB,B) | why it fails |
|---|---|---|---|
| 09:20-10:00 (pre-A) | **-535**, wins 2/14 | -0.24 | worst cell in the entire study |
| 12:00-13:00 | -250 | +0.26 | **zero gross premium** - nothing to harvest |
| 13:00-14:00 SL25 (the TIMEB2 Mon/Tue shape) | **-521**, worst -3,786 | +0.49 | loses badly on Friday |
| 12:00-15:20 | -76 | **+0.81** | is COMB |
| 14:00-15:20 / 14:05-15:20 | +269 / +369 | **+0.58 / +0.62** | the only positive ones, and they are leverage on a book already in the market |

**SENSEX Friday (DTE4) is the seductive one and must also be declined:** 09:35-11:35 returns
**+660/lot, 13/14, t = 5.15** - the best single line in the study - but **r = 0.71 with the NIFTY
slot A that runs the same hours** on a 0.98-correlated index, and at 2-3 lots it pushes peak
Friday margin from 37% to **46-51% of Rs 44.7L capital**. It is size dressed up as diversification.

### Recommendation

**Change nothing on Friday.** Keep NIFTY TimeB at 10:00-12:00 SL20; add no Friday cell to
`CSL_TIMEB2_NIFTY`; re-open no Friday cell on TB-SENSEX. The only lever worth remembering -
moving A to 09:35-11:35 for +173/Friday - is **t = 1.99 on 14 days** and buys P&L by accepting
**more** move-risk (44% vs 35% of Fridays moving >30 bp), which is the opposite of what was asked.
Re-run the surface after 2026-11 with ~28 Fridays before touching a real-money window.

### Two things worth carrying elsewhere

1. **The holiday guard is a reusable data rule.** Any study reading `option_chain` day-by-day must
   reject days with a frozen chain, or it silently books a cost-only loss on every exchange holiday.
2. **COMB's Friday is the weakest Friday component of the stack** (+191/lot, worst -5,853, +656
   without one day). A dedicated COMB-on-Friday review is a cheaper win than any new slot.

