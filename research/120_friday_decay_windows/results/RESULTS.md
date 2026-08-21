# research/120 — Friday decay windows: where does premium bleed safest, and is a second TimeB slot worth it?

## VERDICT: **SIGNAL (already captured) on part 1 — NO EDGE on part 2. Do not add a second Friday slot.**

**The question contains a false premise, and that is the finding.** On a Friday the time of day
that *decays best* and the time of day with the *least possibility of volatile moves* are
**opposite ends of the session**, and they are opposite ends for a reason: the premium you can
earn in a window is the price of the risk you carry through it.

- Over **274 SENSEX Fridays of 1-minute index data (2021→2026)** and **542 NIFTY Fridays of
  5-minute data (2015→2026)**, the genuinely calmest part of a Friday is **11:20–12:00**
  (mean 45-min excursion 20.2 bp, only 17% of Fridays move more than 30 bp).
- Over **14 clean Fridays of real 1-minute option prices**, that is exactly where a short
  straddle makes **nothing**: the five calmest windows in the long sample earn **−34, −90,
  −141, −159 and −394 Rs/lot**. A 12:00–13:00 NIFTY straddle collects **zero gross** — you pay
  a round trip to hold a position that does not decay.
- The windows that pay are the **09:35–10:20 morning**, and the long sample says those are the
  **second-most dangerous** part of the Friday (mean excursion 32.0 bp, 44% of Fridays >30 bp
  — *worse* than the Mon–Thu morning at 29.5 bp).
- Formally: rank-correlation between a window's **long-run risk** and its **sample P&L** is
  **+0.31 (p = 0.0011)** for NIFTY and **+0.17 (p = 0.07)** for SENSEX. Positive. There is no
  calm-and-profitable corner to move into.

**The good news is that the live book is already sitting in the right pocket.** NIFTY TimeB's
Friday DTE2 cell — **10:00–12:00 SL20** — returns **+400 Rs/lot/Friday, 13 wins in 14, worst
−344, t = 4.69, mean MAE 353 Rs/lot.** It is the best-behaved thing on a Friday in this data.

**And every candidate second slot fails**, for three different reasons depending on where you
put it: the midday slots have no premium to harvest, the late slots are 0.58–0.62 correlated
with COMB which is already in the market, and the pre-open slot (09:20–10:00) is the single
worst cell in the entire study (**−535 Rs/lot, wins 2 of 14**).

> **Recommendation: change nothing on Friday.** Keep NIFTY TimeB at 10:00–12:00 SL20, do not
> add a Friday cell to `CSL_TIMEB2_NIFTY`, and do not re-open a Friday cell on TB-SENSEX. The
> one optional, small, *non-urgent* tweak is discussed in §7 — and it buys P&L by taking **more**
> move-risk, not less, which is the opposite of what was asked for.

---

## 1. What was asked, and what was actually tested

> **Arun (2026-08-21):** *"can you study if there is any time window on a Friday where SX and/or
> NIFTY decays better with least possibility of volatile moves? Is it good to have a second slot
> for TimeB if there is any advantage to consider?"*

Restated as two ordered questions:

1. For each venue, across a pre-registered grid of Friday start times and durations, what is the
   **net P&L per lot** and what is the **adverse-move exposure**? A window that earns by sitting
   through danger is not the same as one that earns because nothing happens in it.
2. Does a **second slot add anything on top of slot A** — marginal P&L, correlation of the two
   slots' bad days, the extra round trip, and the margin locked while slot A is still open?

**The live Friday state, read from the frozen config, not from memory**
(`backtest_data/csl_paper_config.json`, frozen 2026-08-13):

| Book | Friday cell | Status |
|---|---|---|
| `CSL_TIMEB_NIFTY` (8 lots, REAL) | DTE2 = **10:00–12:00 SL20** | the slot A of this study |
| `NAS_COMB20` (2 lots, REAL) | 09:16–15:20 SL20, every day | already in the market all Friday |
| `CSL_TIMEB_SENSEX` (8 lots, REAL) | DTE0 + DTE1 only — **no Friday cell** | cut 19-Aug |
| `CSL_TIMEB2_NIFTY` (2 lots, PAPER) | DTE0 + DTE1 only — **no Friday cell** | the existing "second slots" book |

So the concrete decision is: *should TIMEB2 get a DTE2 (Friday) cell, and/or should TB-SENSEX
get a DTE4 cell?*

---

## 2. Data, and the trap that cost two Fridays

| Stage | Source | Coverage | Role |
|---|---|---|---|
| **A** | `options_data.db :: option_chain`, 1-minute | **14 clean Fridays**, 2026-05-08 → 08-14 | rupee truth |
| **B** | `market_data.db :: market_data_unified`, SENSEX **1-minute** | **274 Fridays**, 2021-01-01 → 2026-08-20 | the volatility clock |
| **B2** | same, `NIFTY50` **5-minute** | **542 Fridays**, 2015-02-02 → 2026-07-17 | NIFTY clock (coarse) |

**Data reality found on the way in.** `market_data_unified` has **no NIFTY 1-minute series at
all** — NIFTY intraday exists only as `NIFTY50` at 5-minute resolution. Under the project's
"no 5-min in options backtests" rule, the NIFTY clock is therefore used for **shape only**; the
absolute tail frequencies are read off the SENSEX 1-minute series, and a SENSEX-resampled-to-5-min
control is written to `volclock_buckets.csv` so the understatement is measured rather than assumed.
The two indices give the same clock shape, which is the useful part.

**The trap.** The recorder polls on exchange holidays too. **2026-05-01 and 2026-06-26 are market
holidays** — the chain is frozen (1 distinct underlying value for the whole "session";
`market_data_unified` has 0 rows for both days). Every window on those days books exactly minus
the round-trip cost and looks like a real losing Friday. They were in the first run and dragged
every book's mean down and every win rate. `stage_a_windows.py` now carries a **holiday guard**
(reject any day with fewer than 50 distinct underlying prints) and the study runs on **14 Fridays**.

Other provenance notes: expiry labelling is derived per day, not assumed — every NIFTY Friday is
**DTE2** (Tuesday expiry, 4 calendar days) and every SENSEX Friday is **DTE4** (Thursday expiry,
6 calendar days) except 2026-05-22 whose front expiry was a Wednesday. Strikes are chosen from
the **spot at the window's own start minute**, never from a later print. **2026-08-21 (today) is
excluded** — the session is still open.

**Costs, charged on every trade including the second slot's extra round trip:**
NIFTY 0.5 pt/leg-side × 65 + Rs30/leg-side = **Rs 250/lot**; SENSEX 1.0 pt/leg-side × 20 + Rs30
= **Rs 200/lot**. Cost is a constant per trade, so cost sensitivity is exact arithmetic: at
**2× cost** every number below falls by another Rs 250 (NIFTY) / Rs 200 (SENSEX).

---

## 3. The surface — reported whole, not just its maximum

Pre-registered grid: **22 start times (09:20 → 14:30, 15-min steps) × 5 durations (45 / 60 / 90 /
120 / hold-to-15:20) × 2 stop arms (combined-SL 20%, no stop) × 2 venues = 440 cells**, of which
110 per venue-arm. Every cell was tried; all 440 are in `surface_cells.csv`, and the full
printed surface is in `surface_report.txt`.

### NIFTY, combined-SL 20% — mean net Rs/lot per Friday | mean MAE Rs/lot (n = 14)

| start | 45 | 60 | 90 | 120 | HOLD |
|---|---|---|---|---|---|
| 09:20 | −487 \| 774 | −442 \| 926 | −247 \| 926 | +146 \| 926 | +543 \| 1849 |
| **09:35** | −50 \| 344 | +98 \| 344 | **+333** \| 345 | **+572** \| 345 | +756 \| 1297 |
| **09:50** | −22 \| 272 | +89 \| 295 | **+332** \| 300 | **+434** \| 300 | +582 \| 1404 |
| 10:05 | −58 \| 311 | +47 \| 323 | +245 \| 341 | +280 \| 353 | +162 \| 1937 |
| 10:20 | +28 \| 252 | +155 \| 262 | +209 \| 314 | +299 \| 367 | +143 \| 1957 |
| 10:35 | +17 \| 207 | +74 \| 234 | +187 \| 261 | +117 \| 486 | −79 \| 1785 |
| 10:50 | −101 \| 332 | −49 \| 349 | +72 \| 366 | +6 \| 585 | −23 \| 1868 |
| 11:05 | −34 \| 166 | +53 \| 170 | +4 \| 381 | +33 \| 462 | −59 \| 1902 |
| 11:20 | −90 \| 185 | −19 \| 223 | −263 \| 454 | −193 \| 592 | +0 \| 1852 |
| 11:35 | −141 \| 232 | −166 \| 408 | −186 \| 481 | −184 \| 627 | −53 \| 1838 |
| 11:50 | −159 \| 279 | −305 \| 351 | −248 \| 505 | −377 \| 853 | −47 \| 1656 |
| 12:05 | −394 \| 355 | −217 \| 400 | −186 \| 566 | −554 \| 932 | −93 \| 1653 |
| 12:20 | −258 \| 373 | −326 \| 488 | **−595** \| 945 | −501 \| 1077 | −73 \| 1730 |
| 12:35 | −227 \| 299 | −197 \| 372 | −541 \| 802 | −183 \| 974 | +8 \| 1537 |
| 12:50 | −233 \| 301 | −519 \| 750 | −336 \| 924 | −361 \| 1265 | −76 \| 1654 |
| 13:05 | −594 \| 647 | **−690** \| 747 | −394 \| 982 | −278 \| 1512 | −88 \| 1556 |
| 13:20 | −542 \| 638 | −421 \| 820 | −496 \| 1187 | −1 \| 1501 | −1 \| 1501 |
| 13:35 | −340 \| 654 | −335 \| 732 | −66 \| 1111 | +80 \| 1146 | +80 \| 1146 |
| 13:50 | −164 \| 440 | −327 \| 637 | +250 \| 818 | +250 \| 818 | +250 \| 818 |
| 14:05 | −240 \| 459 | +71 \| 620 | +369 \| 653 | +369 \| 653 | +369 \| 653 |
| 14:20 | +1 \| 546 | +272 \| 584 | +272 \| 584 | +272 \| 584 | +272 \| 584 |
| 14:30 | +142 \| 486 | +232 \| 486 | +232 \| 486 | +232 \| 486 | +232 \| 486 |

### SENSEX, combined-SL 20% — the same shape, independently

| start | 45 | 60 | 90 | 120 | HOLD |
|---|---|---|---|---|---|
| 09:20 | −153 \| 584 | −188 \| 742 | +35 \| 763 | +391 \| 763 | +256 \| 1904 |
| **09:35** | +76 \| 295 | +226 \| 300 | **+409** \| 360 | **+660** \| 360 | +519 \| 1477 |
| **09:50** | +47 \| 255 | +134 \| 292 | **+423** \| 304 | **+449** \| 304 | +159 \| 1657 |
| 10:05 | +0 \| 338 | +39 \| 358 | +270 \| 360 | +351 \| 360 | +34 \| 1887 |
| 10:20 | +43 \| 267 | +261 \| 276 | +270 \| 282 | +347 \| 294 | +44 \| 1881 |
| 10:35 | +120 \| 202 | +123 \| 242 | +221 \| 256 | +94 \| 477 | −258 \| 1967 |
| 11:05 | +28 \| 144 | +123 \| 162 | +35 \| 377 | +15 \| 474 | −250 \| 1894 |
| 11:20 | −93 \| 251 | −97 \| 322 | −321 \| 590 | −326 \| 723 | −373 \| 1972 |
| 12:05 | −373 \| 380 | −262 \| 445 | −317 \| 674 | **−621** \| 1082 | −442 \| 1845 |
| 12:20 | −260 \| 384 | −369 \| 517 | −486 \| 962 | −578 \| 1184 | −225 \| 1796 |
| 13:05 | −450 \| 670 | −599 \| 825 | −309 \| 1064 | −290 \| 1627 | −242 \| 1673 |
| 13:35 | −365 \| 710 | −315 \| 780 | −181 \| 1198 | −166 \| 1259 | −166 \| 1259 |

*(full 22-row SENSEX table in `surface_report.txt`)*

**The plateau is the point.** This is not one lucky cell. The **09:35–10:35 × 90–120 min block is
positive in 20 of 20 cells across both venues and both stop arms**, and the **11:20–13:20 block is
negative in 34 of 36**. The two venues were fitted independently and agree; the SL20 and NOSTOP
arms agree, which means the shape is in the **premium path**, not in the stop.

---

## 4. The multiple-testing haircut — and why the headline is a block, not a cell

440 cells on 14 Fridays. Something will look brilliant by chance, so the family was tested with a
**Westfall–Young max-t bootstrap** (5,000 day-resamples) on each cell's *excess over the
same-day, same-duration all-start average* — the control that removes "short straddles decay all
day, so every window looks positive".

| family | cells | observed max \|t\| | null 95% max \|t\| | family-wise p | |
|---|---|---|---|---|---|
| NIFTY SL20 | 110 | 6.25 (14:30 / 60) | 7.43 | 0.100 | **not significant** |
| NIFTY NOSTOP | 110 | 6.20 (14:30 / 60) | 7.47 | 0.099 | **not significant** |
| SENSEX SL20 | 110 | 3.54 (09:35 / 120) | 6.98 | 0.423 | **not significant** |
| SENSEX NOSTOP | 110 | 3.69 (09:35 / HOLD) | 6.56 | 0.352 | **not significant** |

**No individual cell survives.** Anyone quoting "09:35–11:35 makes +572/lot" as a discovered
window is quoting the maximum of a 110-cell search on 14 days.

The **shape**, however, is a three-parameter claim rather than a 110-cell search, and it holds:

| venue | arm | MORNING (09:35–11:05) | MIDDAY (11:20–13:20) | LATE (13:35–14:30) |
|---|---|---|---|---|
| NIFTY | SL20 | **+223** (86% win, t 2.54) | −326 (36% win, t −1.72) | +226 (71% win, t 1.24) |
| NIFTY | NOSTOP | **+223** (86%, t 2.54) | −420 (36%, t −1.56) | +203 (71%, t 1.01) |
| SENSEX | SL20 | **+261** (86%, t 3.12) | −350 (43%, t −1.47) | −65 (57%, t −0.25) |
| SENSEX | NOSTOP | **+261** (86%, t 3.12) | −426 (43%, t −1.41) | −67 (57%, t −0.26) |

**MORNING minus MIDDAY, paired by Friday:**

| | difference | t | p | Fridays positive |
|---|---|---|---|---|
| NIFTY SL20 | **+550 Rs/lot** | 2.56 | 0.024 | 11 / 14 |
| NIFTY NOSTOP | +643 | 2.18 | 0.048 | 11 / 14 |
| SENSEX SL20 | **+611 Rs/lot** | 2.48 | 0.028 | 12 / 14 |
| SENSEX NOSTOP | +687 | 2.23 | 0.044 | 12 / 14 |

Four tests, four passes, on two venues that were not fitted together. That is the real result of
part 1 — and it says the morning, which is where slot A already lives.

**Concentration check** (research/118's lesson — is one Friday carrying it?). Leave-one-out on
every headline book: **no sign flips anywhere.**

| book | mean | worst day | mean ex-worst | mean ex-best |
|---|---|---|---|---|
| A — NIFTY 10:00–12:00 SL20 | +400 | −344 | +457 | +359 |
| NIFTY 09:35–11:35 SL20 | +572 | −448 | +651 | +517 |
| SENSEX 09:35–11:35 SL20 | +660 | −631 | +759 | +609 |
| NIFTY 14:05–15:20 SL20 | +369 | −1,579 | +519 | +292 |
| NIFTY 13:00–14:00 SL25 | −521 | −3,786 | −270 | −588 |
| COMB 09:16–15:20 SL20 | +191 | −5,853 | +656 | +22 |

Note the last row: **COMB's Friday is one bad day away from zero** (+191, and +656 without
2026-05-29). The morning window books are not.

---

## 5. The volatility clock — and why it refuses to corroborate the "calm" half of the question

274 SENSEX Fridays of 1-minute data. Maximum excursion from the entry price inside the window,
in basis points:

| start | 45-min mean / p90 / % of Fridays >30bp | 90-min mean / p90 / %>30bp |
|---|---|---|
| 09:20 | 34.8 / 61.2 / **49.3%** | 44.4 / 79.0 / 66.8% |
| **09:35** | **32.0** / 54.8 / **44.2%** | 41.2 / 76.8 / 56.6% |
| 09:50 | 28.5 / 48.6 / 36.1% | 37.3 / 61.7 / 54.7% |
| 10:05 | 27.1 / 47.4 / 34.7% | 35.5 / 62.0 / 48.9% |
| 10:20 | 26.2 / 49.4 / 27.0% | 34.0 / 58.4 / 44.2% |
| 10:50 | 22.5 / 42.5 / 19.7% | 31.3 / 55.3 / 40.9% |
| **11:20** | **20.8** / 38.6 / **17.2%** | 30.3 / 58.0 / 36.1% |
| **11:35** | **20.2** / 36.4 / **17.6%** | 30.3 / 55.7 / 37.0% |
| 12:20 | 22.0 / 39.5 / 19.3% | 31.9 / 59.4 / 39.4% |
| 13:20 | 25.2 / 45.0 / 23.0% | 36.1 / 66.8 / 47.4% |
| 14:05 | 25.8 / 44.0 / 30.7% | 35.8 / 63.3 / 51.8% |
| 14:30 | 29.6 / 52.7 / 38.0% | 30.8 / 54.0 / 40.1% |

The NIFTY 542-Friday clock has the same shape (09:35 = 31.7 bp / 36.7% >30bp; 11:35 = 21.7 bp /
17.3%). The per-minute view agrees: SENSEX mean absolute 1-minute move falls from 3.83 bp in the
09:15 bucket to a **trough of 1.89 bp at 11:45–12:00**, then rises again to 3.38 bp at 15:00.

**Friday is also, mildly, the more dangerous day**: Friday morning excursions run above the
Mon–Thu equivalents at every morning start (09:35: 32.0 vs 29.5 bp; 09:50: 28.5 vs 27.3), and the
Friday full-day hold from 09:20 travels 77.8 bp against Mon–Thu's 74.9.

### The money exhibit — join Stage A to Stage B

Rank-correlate each grid cell's **14-Friday net P&L** against the **same cell's long-run risk**:

| | Spearman(long-run mean excursion, sample net) | Spearman(long-run % days >30bp, sample net) |
|---|---|---|
| NIFTY (542 Fridays) | **+0.307** (p = 0.0011) | **+0.333** (p = 0.0004) |
| SENSEX (274 Fridays) | +0.136 (p = 0.156) | +0.173 (p = 0.071) |

Positive on both venues. Concretely, for NIFTY:

| the 5 CALMEST windows in 542 Fridays | long-run risk | what they earned |
|---|---|---|
| 11:35 / 45 | 21.7 bp, 17.3% >30bp | **−141** |
| 11:20 / 45 | 22.2 bp, 17.9% | **−90** |
| 11:05 / 45 | 22.5 bp, 20.3% | **−34** |
| 12:05 / 45 | 22.7 bp, 19.9% | **−394** |
| 11:50 / 45 | 23.1 bp, 20.5% | **−159** |

| the 5 MOST DANGEROUS windows | long-run risk | what they earned |
|---|---|---|
| 10:20 / HOLD | 71.3 bp, 84.7% >30bp | **+143** |
| 10:05 / HOLD | 75.1 bp, 87.6% | **+162** |
| 09:50 / HOLD | 78.6 bp, 89.7% | **+582** |
| 09:35 / HOLD | 80.7 bp, 92.4% | **+756** |
| 09:20 / HOLD | 83.3 bp, 92.1% | **+543** |

**Every calm window loses. Every dangerous window earns.** The 14-Friday option sample happened
to show *low* MAE in the morning (mean 345 Rs/lot at 09:35/120 versus 945 at 12:20/90) — that is
a 14-day accident on the risk side, and the 274/542-Friday sample overturns it. **The morning
window is not safe; it is well paid.**

**Trap #3 from the brief, applied honestly:** the best 14-Friday window does **not** sit in a calm
part of the day, so its low measured MAE is treated as noise and is not part of the recommendation.

---

## 6. Part 2 — the second slot, judged marginally

Slot A = NIFTY TimeB Friday DTE2 **10:00–12:00 SL20**. COMB = NIFTY **09:16–15:20 SL20**, 2 lots,
in the market all Friday. Every candidate B pays a **full extra round trip**.

### Standalone, per lot per Friday (14 Fridays)

| book | mean | median | win% | worst | t |
|---|---|---|---|---|---|
| **A — 10:00–12:00 SL20 (live)** | **+400** | +358 | **93** | **−344** | **4.69** |
| COMB 09:16–15:20 SL20 (live) | +191 | +1,619 | 79 | −5,853 | 0.26 |
| B 09:20–10:00 (pre-A) | **−535** | −476 | **14** | −1,345 | −4.18 |
| B 09:35–10:00 | −80 | −3 | 50 | −556 | −0.96 |
| B 12:00–13:00 | −250 | −193 | 21 | −1,680 | −1.76 |
| B 12:30–13:30 | −260 | −9 | 50 | −2,333 | −1.38 |
| **B 13:00–14:00 SL25** *(the TIMEB2 Mon/Tue shape)* | **−521** | −8 | 50 | **−3,786** | −1.75 |
| B 13:00–15:20 | −215 | +641 | 64 | −4,062 | −0.44 |
| B 12:00–15:20 | −76 | +408 | 71 | −5,141 | −0.14 |
| B 14:00–15:20 | +269 | +421 | 71 | −1,638 | 1.37 |
| B 14:05–15:20 | +369 | +415 | 86 | −1,579 | 1.82 |

### On top of A, and against COMB

| candidate B | B mean | A+B mean | A+B worst | r(A,B) | **r(COMB,B)** | B loses on A's bad days |
|---|---|---|---|---|---|---|
| 12:00–13:00 | −250 | +150 | −1,394 | +0.06 | +0.26 | yes |
| 12:30–13:30 | −260 | +140 | −1,501 | −0.40 | +0.01 | no |
| 13:00–14:00 SL25 | −521 | −122 | −3,282 | −0.21 | +0.49 | no |
| 13:00–15:20 | −215 | +185 | −3,428 | −0.20 | **+0.64** | no |
| 12:00–15:20 | −76 | +324 | −4,507 | −0.14 | **+0.81** | no |
| **14:00–15:20** | +269 | +669 | −1,134 | +0.21 | **+0.58** | **yes** |
| **14:05–15:20** | +369 | +769 | −1,075 | +0.15 | **+0.62** | **yes** |
| 09:20–10:00 | −535 | −136 | −1,056 | +0.09 | −0.24 | yes |

*(A has only one losing Friday in 14 — 2026-07-17 — so the "bad-day overlap" column is a single
observation and should be read as colour, not evidence.)*

**Reading it:**

- **Everything from 12:00 to 14:00 is negative.** 12:00–13:00 collects **zero gross premium** —
  the whole −250 is the round trip. That is the direct rupee statement of §5: the calmest hour of
  the Friday has nothing to sell.
- **The only positive second slot is the last 75–80 minutes**, and it is **0.58–0.62 correlated
  with COMB**, which is already short the same index over exactly those minutes at 2 lots. Adding
  it is **leverage on an existing position, not a new source of return** — and it is the arm whose
  leave-one-out gap is widest (+369 → +519 without its worst day).
- The **pre-A morning slot is the worst thing in the study**: 09:20–10:00 loses 535/lot and wins
  2 of 14. The first 40 minutes are where the index is most violent and the theta is smallest.
- **Break-even costs**: A survives up to **Rs 650/lot** round trip (2.6× current). 14:05–15:20
  survives to Rs 619 (2.5×). 12:00–13:00 breaks even at **Rs 0** — it never earns a rupee gross.

### Margin — capital Rs 44.7L, NIFTY Rs 1.65L/lot, SENSEX Rs 2.04L/lot

| configuration | peak concurrent | % of capital |
|---|---|---|
| **Today, 10:00–12:00**: COMB 2L + TB-N 8L | 10 NIFTY lots = **Rs 16.50L** | 37% |
| + NIFTY second slot 2L after 12:00 | 4 lots = Rs 6.60L in that window | 15% |
| + NIFTY second slot 3L after 12:00 | 5 lots = Rs 8.25L | 18% |
| + SENSEX Friday 2L at 09:35–11:35 (**overlaps A**) | 10 NIFTY + 2 SENSEX = **Rs 20.58L** | 46% |
| + SENSEX Friday 3L at 09:35–11:35 (**overlaps A**) | 10 NIFTY + 3 SENSEX = **Rs 22.62L** | 51% |

A **post-12:00** NIFTY second slot is cheap in margin terms (slot A is closed by then; only COMB
overlaps). A **SENSEX morning** cell is expensive because it runs *simultaneously* with slot A —
and that is the configuration whose P&L looks best, which is exactly the trap.

---

## 7. What the evidence supports, plainly

**1 — Do not add a second Friday slot. (The direct answer to the question.)**
No candidate clears the bar. The profitable ones are correlated with a book that is already open;
the uncorrelated ones lose money; the biggest single loser in the study is the pre-A slot. The
existing `CSL_TIMEB2_NIFTY` shape (13:00–14:00 SL25), transplanted to Friday, would have returned
**−521/lot with a worst day of −3,786**. Its Mon/Tue cells are a different question and are not
touched by this study.

**2 — Do not open a Friday cell on TB-SENSEX either — for a reason that is easy to miss.**
SENSEX 09:35–11:35 is the single best line in the whole study: **+660/lot, 13 of 14, worst −631,
t = 5.15**. But its correlation with the NIFTY slot A that already runs at the same hours is
**+0.71**. It is the same short-gamma bet on a 0.98-correlated index over the same minutes. At
2–3 lots it pushes peak Friday margin to **46–51% of capital** for a return stream that mostly
already exists. If Arun wants more Friday morning exposure, **the honest instrument is size on
the existing NIFTY cell, not a second venue dressed up as diversification.** (Whether that is
wise is a sizing decision, not a finding of this study.)

**3 — Optional, small, non-urgent: the existing window could start ~25 minutes earlier.**
NIFTY **09:35–11:35 SL20** returns **+572/lot** versus slot A's **+400**, a paired **+173/Friday
at t = 1.99** — right at the edge, on 14 days, for one parameter. 09:50–11:50 adds only +35
(t = 0.81), so the gain is concentrated in the 09:35–09:50 quarter-hour, which is also the most
volatile quarter-hour of the tradeable morning. **It buys P&L by accepting more move-risk**
(long-run 45-min excursion 32.0 bp at 09:35 vs 27.1 bp at 10:05; 44% vs 35% of Fridays moving
>30 bp). Given the book is real money at 8 lots and the evidence is a single marginal t on 14
Fridays, the defensible action is **leave it and re-check after another quarter of Fridays**, not
move it now.

**4 — The framing to keep.** "Decays well" and "least volatile" are the same axis with opposite
signs on a Friday. There is no window that gives both. The book's current Friday shape — a
morning window on NIFTY plus a small all-day COMB — is already the sane corner of that trade-off.

---

## 8. Sins accounting

| Sin | How it was controlled |
|---|---|
| **Look-ahead** | Strike chosen from the spot at the window's own start minute; stop evaluated minute-forward only; expiry derived per day from the chain, never assumed |
| **Survivorship / sample selection** | Every recorded Friday used; the two exchange **holidays** removed by a data rule (frozen chain), not by looking at their P&L; today's partial session excluded |
| **Overfitting / multiple testing** | Full 440-cell grid pre-registered and **reported whole**; Westfall–Young max-t bootstrap over each 110-cell family — **nothing survives**; the accepted claim is a 3-block shape agreeing across 2 venues × 2 stop arms; plateau (neighbour means) reported per cell in `surface_cells.csv` |
| **Cost neglect** | Net everywhere; a second slot charged a full extra round trip; cost is a per-trade constant so 2× sensitivity is exact; break-even cost stated per book |
| **Regime dependence** | 14 Fridays is **one regime** — this is the study's biggest weakness, and it is precisely why the risk half is answered from 274 / 542 Fridays instead |
| **Correlation / single factor** | The decisive part-2 test: r(COMB, B) 0.58–0.81 for the profitable slots; r(NIFTY-A, SENSEX-morning) = 0.71 |
| **Capacity / margin** | Peak concurrent margin tabulated for every proposed pairing against Rs 44.7L |
| **Placebo / control** *(research/115's lesson)* | Every cell measured as **excess over the same-day, same-duration all-start average** — without it the whole surface reads positive simply because straddles decay; and the Stage-A↔Stage-B join is itself the placebo that kills the "calm window" story |

---

## 9. Honest caveats

1. **14 Fridays.** One regime (May–August 2026), one volatility environment, low realised vol
   throughout. Slot A's *93% win rate* and *worst −344* are 14-day statistics and will not hold;
   research/118 made exactly this mistake reading "worst −127" off 12 Thursdays when the true
   DTE0 worst was −21,500/lot. **Size for a much worse Friday than anything in this sample.**
2. **No individual window is statistically significant** after the multiple-testing haircut. The
   only claim being made is the block shape, and even that rests on t ≈ 2.2–3.1.
3. **The NIFTY volatility clock is 5-minute**, which understates within-bucket extremes. It is
   used for shape; absolute tail frequencies come from the SENSEX 1-minute series. A SENSEX
   5-min resample is in `volclock_buckets.csv` for anyone who wants the size of that understatement.
4. **Single-entry model.** Each window is one straddle sold at the start minute and covered at the
   end or on a combined-20% stop, evaluated on 1-minute LTP with a fixed slippage. It does not model
   the live books' 5-second polling, two-poll dwell, or 50% disaster backstop. Fills are optimistic
   by construction on the stop arm.
5. **Stage A and Stage B are different eras.** The option truth is 2026; the volatility clock is
   2021–2026 and 2015–2026. The clock's *shape* is stable across the whole span, but the two
   stages are not a like-for-like comparison and are not treated as one.
6. **The 09:35 alternative is one t-test away from nothing** (t = 1.99, p ≈ 0.07 two-sided). It is
   reported because it was asked for, not because it is established.

---

## 10. Next levers

- **Re-run this exact surface after 2026-11** with ~28 Fridays. The scripts are resumable and the
  holiday guard is now in place; a second quarter roughly halves the standard errors, and the
  09:35-vs-10:00 question then has enough power to settle.
- **The `CSL_TIMEB2_NIFTY` review already scheduled for 2026-09-05** ("merge into TB-CSL or drop")
  should carry this study's answer for the Friday leg: **no Friday cell**.
- **If more Friday morning exposure is wanted**, test it as *size on the existing NIFTY cell*
  against the same margin budget, and compare it head-to-head with the SENSEX-morning alternative
  on a joint-drawdown basis, not on standalone mean.
- **Open question this study raises but does not answer:** COMB's Friday (+191/lot, worst −5,853,
  +656 without one day) looks like the weakest Friday component of the stack. A dedicated
  COMB-on-Friday review is a cheaper win than any new slot.

---

**Reproducibility stamp.** Data snapshot 2026-08-21 (`options_data.db` 12.1 GB,
`market_data.db` 30.3 GB, both opened read-only). Scripts:
`scripts/stage_a_windows.py` (grid + all-start control), `scripts/stage_b_volclock.py`
(volatility clock), `scripts/analyze_surface.py` (surface + Westfall–Young),
`scripts/marginal_slot.py` (part 2 + margin), `scripts/robustness.py` (blocks, leave-one-out,
Stage-A↔Stage-B join). Costs NIFTY Rs 250/lot, SENSEX Rs 200/lot round trip. Bootstrap seed 20260821.
