# RESULTS — SENSEX per-DTE allocation

**Verdict: CONCLUDED. Unlike NIFTY, there is NO strong day-allocation signal on SENSEX.
One large and real finding: stops are very expensive on expiry day, which independently
reproduces research/114 on a fresh sample. Nothing deployed.**

research/139 · 2026-08-31

---

## 0. What was built

There was no SENSEX equivalent of the NIFTY recorded-chain study, so research/138's
method could not simply be pointed at it. `scripts/sensex_study.py` builds one from
`options_data.db` (`symbol='SENSEX'`, 92 recorded days from 2026-04-20), in the same
shape as the NIFTY file: per day, the 09:16 ATM straddle's 5-minute combined-premium
path. Output: `static/app/sensex_options_study.json`.

Then the identical DTE × stop grid, so the two venues read side by side.

**Three things that are not the same as NIFTY and govern how this is read:**

| | NIFTY | SENSEX |
|---|---|---|
| expiry | Tuesday | **Thursday** |
| weekday → DTE | Mon 1 · Tue 0 · Wed 4 · Thu 3 · Fri 2 | **Mon 3 · Tue 2 · Wed 1 · Thu 0 · Fri 4** |
| lot | 65 (10 lots = qty 650) | **20 (10 lots = qty 200)** |

10 SENSEX lots is **not** the same notional as 10 NIFTY lots. The per-DTE comparison
*inside* SENSEX is exact; against NIFTY it is indicative only.

Also: the live SENSEX CSL books run **windows** (TimeB DTE0 13:00–15:20, DTE1
10:30–12:00), not the full day. This grid is a full-day 09:16→15:20 construction, so it
speaks directly to the **9:16 suite** (`sensex_atm`/`atm2`/`atm4`) and only indirectly to
the TimeB windows.

**On the stopless column:** it is a **measurement control** to detect when a stop is
inert or actively harmful — never a recommendation (Arun, 2026-08-31: *"having no stop
loss cannot be a recommendation"*). Every table below labels it `nostop*`.

---

## 1. The grid — 10 lots (qty 200), 91 days

**Net P&L**

| DTE | day | n | SL15 | SL20 | SL25 | SL30 | SL40 | nostop* |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | **Thu** | 18 | ₹1,64,338 | ₹1,45,118 | ₹84,748 | ₹1,56,058 | ₹1,59,058 | **₹5,10,488** |
| 1 | Wed | 18 | ₹218 | ₹11,968 | −₹18,102 | **−₹64,222** | −₹81,332 | −₹1,05,902 |
| 2 | Tue | 18 | ₹47,008 | ₹51,198 | ₹8,978 | ₹8,998 | ₹14,578 | ₹14,578 |
| 3 | Mon | 20 | ₹55,200 | ₹55,840 | ₹65,400 | ₹56,340 | ₹63,480 | ₹63,480 |
| 4 | Fri | 17 | ₹76,747 | ₹31,447 | ₹44,487 | ₹44,487 | ₹76,027 | ₹76,027 |

**t** (2.0 is the 1-in-20 bar; **30 cells tested**, so expect ~1.5 by luck)

| DTE | day | SL15 | SL20 | SL25 | SL30 | SL40 | nostop* |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Thu | 1.21 | 1.04 | 0.55 | 0.88 | 0.88 | **4.19** |
| 1 | Wed | 0.00 | 0.12 | −0.16 | **−0.51** | −0.58 | −0.55 |
| 2 | Tue | 0.52 | 0.51 | 0.08 | 0.08 | 0.13 | 0.13 |
| 3 | Mon | 0.57 | 0.58 | 0.70 | 0.57 | 0.67 | 0.67 |
| 4 | Fri | 0.91 | 0.28 | 0.42 | 0.42 | 0.89 | 0.89 |

**Stops fired**

| DTE | day | SL15 | SL20 | SL25 | SL30 | SL40 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Thu | 9/18 | 9/18 | 9/18 | 8/18 | **7/18** |
| 1 | Wed | 7/18 | 6/18 | 6/18 | 6/18 | 4/18 |
| 2 | Tue | 6/18 | 5/18 | 5/18 | 3/18 | 0/18 |
| 3 | Mon | 4/20 | 2/20 | 1/20 | 1/20 | 0/20 |
| 4 | Fri | 2/17 | 2/17 | 1/17 | 1/17 | 0/17 |

**The headline is the DTE0 row.** Every stop level from 15% to 40% costs roughly
**₹3.5 lakh of a ₹5.1 lakh cell**, and even at SL40 the stop still fires on **7 of 18**
expiry days. This is not a case of a stop being *inert* (NIFTY Thursday) — it is a stop
being **actively destructive**: expiry gamma trips it, and then the premium decays
without us.

That independently reproduces **research/114** ("on 12 clean expiry days the stop turned
+₹2,630/lot/day at a 92% win rate into −₹227 at 25%") on a different and larger sample.
The live book already acts on it: SENSEX DTE0 runs `sl: "none"` with a **50% disaster
backstop** in the executor, i.e. a stop deliberately too wide for gamma to trip — not a
stopless book.

---

## 2. Stability — where SENSEX differs most from NIFTY

| DTE | day | stop | 1st half | 2nd half | verdict |
|---:|---|---:|---:|---:|---|
| 3 | **Mon** | 20 | ₹51,970 | ₹3,870 | **both +** |
| 3 | Mon | 30 | ₹51,970 | ₹4,370 | **both +** |
| 0 | Thu | 20 | ₹3,989 | ₹1,41,129 | both + |
| 0 | Thu | 30 | −₹77,801 | ₹2,33,859 | flips |
| 1 | Wed | 20 | −₹23,751 | ₹35,719 | flips |
| 1 | Wed | 30 | −₹93,531 | ₹29,309 | flips |
| 2 | Tue | 20/30 | −₹8,421 / −₹41,601 | ₹59,619 / ₹50,599 | flips |
| 4 | Fri | 20/30 | −₹47,682 / −₹34,642 | ₹79,129 | flips |

**Only Monday (DTE3) survives in both halves**, and even it decays hard (₹51,970 →
₹3,870). Everything else flips sign between halves.

**This is the crucial contrast with NIFTY.** On NIFTY, DTE3/Thursday was ₹72,595 →
₹82,670 — level across halves with t 3.85, a genuinely differentiated cell. SENSEX has
nothing like it: the best non-control t in the whole grid is **1.21**, below the noise
bar for 30 cells tested. **SENSEX days are not meaningfully different from one another.**

---

## 3. What SENSEX runs today vs what the grid says

| day | DTE | 9:16 suite | CSL sleeves | grid @SL30 | live record |
|---|---:|---|---|---|---:|
| Mon | 3 | dark | CSL30F_SENSEX (paper) | ₹56,340 · t 0.57 · DD −₹55,369 | shadow **+₹33,913** |
| Tue | 2 | dark | CSL30F_SENSEX (paper) | ₹8,998 · t 0.08 · DD −₹91,563 | shadow +₹874 |
| **Wed** | 1 | **LIVE** ×3 | TimeB, 30F, 30F_WED | **−₹64,222 · t −0.51** · DD −₹1,17,939 | live **−₹593** |
| **Thu** | 0 | **LIVE** ×3 | TimeB, 30F | ₹1,56,058 · t 0.88 · DD −₹1,34,503 | live **+₹30,894** |
| Fri | 4 | dark | CSL30F_SENSEX (paper) | ₹44,487 · t 0.42 · DD −₹88,397 | shadow +₹41,602 |

Live SENSEX total (9:16 suite, real money): **+₹30,301** — essentially all of it Thursday.

---

## 4. Recommendations

**(a) DTE0 / Thursday — keep the stop WIDE. Do not tighten. Already correct.**
This is the one large effect in the study. Any stop ≤40% costs ~70% of the cell, and
SL40 still fires 7/18. The live 50% backstop is the right shape and should stay. The
open question worth one cheap run is **where the damage actually stops** — test 50%,
60%, 75% to find the level at which gamma stops tripping it, so the backstop is chosen
on evidence rather than convention. That is a genuine follow-up, not a stopless proposal.

**(b) DTE1 / Wednesday — the weakest cell, and it is live.**
Negative at four of six levels, the **only negative t in the grid** (−0.51 at SL30), and
it flips across halves. It is the SENSEX analogue of NIFTY's Monday. Two things temper
acting now: the live record is only **−₹593 over 5 sessions** (flat, not bleeding), and
Arun already made this call knowingly — `CSL30F_SENSEX_WED` carries the note *"USER
OVERRIDE vs study: Wed full-day cell is −571/day 64% (n=11) and verdict Q4 said
windows-only — Arun chose live anyway after seeing the table. Review after 4 live
Wednesdays."* **Recommendation: let that existing 4-Wednesday review be the gate.** This
study is a third independent vote against the cell, and should be attached to it.

**(c) DTE3 / Monday and DTE4 / Friday — do NOT promote.**
They carry the best shadow records (+₹33,913 and +₹41,602) and Monday is the only cell
stable in both halves. But t is 0.57 and 0.42 against a 30-cell test, and Monday's second
half is 7% of its first. That is not enough to put money on, and it is exactly the shape
of the NIFTY Monday mistake research/138 just unwound.

**(d) The structural conclusion.** NIFTY rewarded a day-allocation change because its
days are genuinely differentiated (t 3.85 vs 1.23 across cells). **SENSEX does not.**
The SENSEX lever is not *which day* — it is **the stop on expiry day**, which is already
pulled. Do not go looking for a SENSEX day-allocation edge; the data says it is not there.

---

## 5. Guards

| deadly sin | control |
|---|---|
| multiple testing | 30 cells stated up front; the noise bar (~1.5) named; nothing below it acted on |
| overfitting | first/second-half split on every cell; only Monday survives, and it is *not* recommended |
| look-ahead | entry is the first snapshot at/after 09:16; the stop is evaluated forward bar by bar |
| cost neglect | round-trip cost scaled from the NIFTY study by qty (₹49 at qty 200) |
| regime dependence | 92 days, one regime — stated as the binding limit on all of this |
| cross-venue error | NIFTY vs SENSEX lot sizes and expiry weekdays kept separate throughout |

**The binding limitation: 92 recorded days is one market regime.** Every t here is small,
and the honest reading of this study is mostly *negative* — it tells us where NOT to look
on SENSEX.

## 6. Files

| file | purpose |
|---|---|
| `scripts/sensex_study.py` | builds `static/app/sensex_options_study.json` from the recorded chain |
| `scripts/sensex_grid.py` | the DTE × stop grid, stability split, live comparison |
| `results/RESULTS.md` | this file |

---

# PHASE 2 — the stop-width walk, and a correction to phase 1's explanation

**2026-08-31. One recommendation, and one conclusion of mine that the cross-check
overturns.**

## 1. What I got wrong in phase 1

Phase 1 said the SENSEX expiry-day result "independently reproduces research/114"
and explained it as **expiry gamma tripping the stop**. Stated that way it implies
a general law about expiry days. It is not one. Running the same stop-width walk on
**NIFTY's** expiry day gives the **opposite** answer on every dimension.

**SENSEX expiry day (DTE0 / Thursday) — 18 days, 10 lots (qty 200)**

| stop | net | t | maxDD | worst day | fires |
|---:|---:|---:|---:|---:|---:|
| 20% | ₹1,45,118 | 1.04 | −₹72,221 | −₹25,469 | 9/18 |
| 25% | ₹84,748 | 0.55 | −₹1,14,411 | −₹41,309 | 9/18 |
| 30% | ₹1,56,058 | 0.88 | −₹1,34,503 | −₹41,309 | 8/18 |
| 40% | ₹1,59,058 | 0.88 | −₹1,11,333 | −₹42,879 | 7/18 |
| **50% (live)** | **₹2,82,278** | **1.74** | −₹74,987 | −₹42,519 | 4/18 |
| **75%** | **₹4,13,978** | **2.71** | **−₹64,399** | −₹64,399 | **1/18** |
| 100% | ₹5,10,488 | 4.19 | −₹31,019 | −₹31,019 | 0/18 |
| *nostop\** | *₹5,10,488* | *4.19* | *−₹31,019* | *−₹31,019* | *0/18* |

**NIFTY expiry day (DTE0 / Tuesday) — 18 days, 10 lots (qty 650)**

| stop | net | t | maxDD | worst day | fires |
|---:|---:|---:|---:|---:|---:|
| **20%** | **₹2,69,860** | **1.88** | **−₹46,405** | −₹31,750 | 6/18 |
| 25% (live) | ₹2,60,175 | 1.78 | −₹46,405 | −₹31,750 | 6/18 |
| 30% | ₹2,17,340 | 1.37 | −₹67,855 | −₹43,385 | 6/18 |
| 40% | ₹1,98,815 | 1.22 | −₹67,855 | −₹43,385 | 6/18 |
| 50% | ₹1,34,595 | 0.73 | −₹1,07,100 | −₹63,470 | 6/18 |
| 75% | ₹1,16,915 | 0.60 | −₹1,28,290 | −₹70,035 | 3/18 |
| *nostop\** | *₹30,725* | *0.13* | *−₹2,14,480* | *−₹1,55,900* | *0/18* |

\* control column, never a candidate.

**On SENSEX, wider is better on every dimension. On NIFTY, tighter is better on
every dimension.** Same structure, same construction, same 18-day count, opposite
sign. So "expiry gamma defeats stops" cannot be the explanation — it would have to
apply to both.

## 2. What the numbers actually say

The `fires` column carries the mechanism, and it is the interesting part.

- **NIFTY: 6/18 fires at 20%, and still 6/18 at 60%.** The *same six days* blow
  through every level from 20% to 60%. On NIFTY expiry, a 20% adverse move is the
  start of a trend that keeps going — so the stop is doing exactly its job, and
  cutting at 20% is better than riding to 60%.
- **SENSEX: 9/18 at 20%, 4/18 at 50%, 1/18 at 75%, 0/18 at 100%.** Breaches
  *revert*. On SENSEX expiry, a 20% adverse move is usually noise that comes back —
  so a tight stop books a loss and then the premium decays without us.

**A 20% adverse move on NIFTY expiry is signal; on SENSEX expiry it is noise.**
That is a mechanical, checkable statement, and it fits what we already knew about
the venue — research/97 concluded SENSEX bid/ask is too noisy to price against and
forced the move to `ltp + slippage`. A percentage stop on a noisy combined premium
gets tripped by the quote, not by the market.

## 3. Recommendation — widen SENSEX DTE0 from 50% to 75%

| | 50% (live today) | **75% (proposed)** | change |
|---|---:|---:|---|
| net | ₹2,82,278 | ₹4,13,978 | **+₹1,31,700** |
| t | 1.74 | 2.71 | better |
| maxDD | −₹74,987 | **−₹64,399** | **also better** |
| fires | 4/18 | 1/18 | still active |

It is the rare change that improves return *and* drawdown together.

**Why 75% and not 100%.** At 100% the stop fires **0 of 18** times — it is
protection in name only, and would be indistinguishable from having none on every
day in the sample. At 75% it still fires once, so it is a live constraint that
bounds the tail on the day this 92-day sample does not contain. **The deliverable
here is a width, never a removal.**

**Why this is a proposal and not a deployment.** It is a rule change to a live
book: it needs Arun's decision, its own STATUS-MD, and an after-15:40 deploy. It
also rests on 18 expiry days in one regime, and 9 widths were tested on them.
The monotonicity is what makes it credible — SENSEX improves steadily from 20%
through 100% with no peak to overfit to — but monotonic-to-the-edge also means
the sweep never found a turning point, which is the same caveat research/80
flagged about its own strike sweep.

## 4. A bonus finding: the NIFTY cell we kept is correctly set

`NAS_COMB20`'s surviving live cell is DTE0/Tuesday at **SL25**. The NIFTY walk puts
20% first (₹2,69,860, t 1.88) and **25% immediately behind** (₹2,60,175, t 1.78),
with identical drawdown. So the one cell still carrying real money on that book is
already at, or one notch from, its best tested setting. Nothing to change there.

## 5. Files

| file | purpose |
|---|---|
| `scripts/dte0_width.py` | the stop-width walk, both venues |
| `scripts/sensex_study.py` | builds the SENSEX recorded-chain study |
| `scripts/sensex_grid.py` | phase 1 DTE × stop grid |

---

# PHASE 3 — the stop-width recommendation was aimed at PAPER books. Withdrawn.

**2026-08-31. Arun: "im not ok with 50% itself as thr risk is huge right?"**

The right response to that turned out not to be another width. It was to check
what is actually exposed on SENSEX expiry day, which I should have done before
recommending anything.

## 1. What is actually live on SENSEX expiry day (Thursday / DTE0)

| book | mode | DTE0 cell | lots |
|---|---|---|---:|
| `sensex_atm` | **LIVE** | 09:16 → 15:15 | 2 |
| `sensex_atm2` | **LIVE** | 09:16 → 15:15 | 2 |
| `sensex_atm4` | **LIVE** | 09:16 → 15:15 | 2 |
| `CSL_TIMEB_SENSEX` | paper | 13:00–15:20, SL none | 6 |
| `CSL30F_SENSEX` | paper | 09:16–15:20, SL none | 3 |

**Live SENSEX expiry-day exposure is 6 lots — the 9:16 suite. Every book that runs
a combined-premium stop on DTE0 is PAPER.**

So the entire 50% / 60% / 75% analysis in phases 1–2 describes the **TimeB and
CSL30F paper books**. Changing that width would not alter a single rupee of live
risk. **The recommendation is withdrawn as a live proposal.** It remains valid for
the paper books, where it is also nearly costless to leave alone.

## 2. What actually governs live SENSEX expiry-day risk

Not a combined-premium stop — the 9:16 suite does not use one. Three controls, in
order of what binds first:

1. **Per-leg stops are DISABLED on expiry day** (research/114): on DTE0 there is no
   per-leg stop at all, because the stop turned +₹2,630/lot/day at 92% into −₹227
   at 25% — expiry gamma tripped it and the premium then decayed without us. That
   is the same effect phase 1 rediscovered, and it was already acted on.
2. **The book-level portfolio stop is the guard**, and on SENSEX expiry it is
   **widened to −₹3,000/lot** (`services/nas_portfolio_stop.py`: `STOP_PER_LOT =
   1300.0`, overridden to `3000.0` on DTE0). Take-profit is ₹4,000/lot.
3. **Size: 6 lots.**

**So the designed worst case on a live SENSEX expiry day is 6 × ₹3,000 = −₹18,000**,
against a live DTE0 record of **+₹30,894 over 5 Thursdays** (mean ≈ +₹6,179). Roughly
1 : 3 risk to observed reward.

**And the stop is not theoretical — it has fired.** `sensex_atm`'s exit-reason
tally shows `PORTFOLIO_STOP` on 2 trades (−₹3,398 total), alongside `PORTFOLIO_TP`
on 1 (+₹3,096).

## 3. So how big is the risk, honestly

| framing | number | what it is |
|---|---:|---|
| designed worst case | **−₹18,000** | 6 lots × the −₹3,000/lot expiry book stop |
| research/118 unstopped tail | −₹1,29,000 | 6 lots × −₹21,500/lot, over 127 DTE0 days |

The gap between those two is **execution risk, not design risk** — it is what you
face only if the book stop fails to act: a gap through the level, a fast market
where the monitor cannot fill, or the process being down. It is a real residual and
worth naming, but it is a different problem from stop width, and no width setting
addresses it.

**The honest summary: at 6 lots with a −₹3,000/lot book stop that has demonstrably
fired, the live SENSEX expiry-day risk is bounded by design at −₹18,000. That is not
"huge" relative to the book. The thing that would make it huge is the book stop
failing, and the defence against that is size and execution monitoring, not width.**

## 4. What I got wrong, plainly

- I ran the whole width analysis at **10 lots**, on a **full-day combined-stop
  construction**, and presented conclusions as if they bore on the live book. Live
  is **6 lots** on a **per-leg / book-stop** construction. Those are different
  systems, and I conflated them.
- I recommended 75% partly on "maxDD also improves", which an 18-day sample cannot
  measure — corrected in phase 2, but it should not have been offered at all.
- Phase 1 called the SENSEX result a reproduction of research/114 and explained it
  as gamma. The NIFTY cross-check refuted the general claim (phase 2), and it now
  turns out research/114's conclusion was **already deployed** in exactly the place
  it belonged: expiry-day per-leg stops are off, and the book stop is the guard.

## 5. The genuinely open questions, if any

Neither is urgent, and neither is a width:

1. **Is −₹3,000/lot the right expiry-day book stop?** It was set by research/114 to
   be loose enough not to be tripped by gamma. Whether −₹2,000 or −₹2,500 would
   keep the edge while tightening the designed loss is measurable, but it needs a
   faithful replay of three interacting sleeves (ATM trail, ATM2 one-and-done, ATM4
   roll-to-match) under a shared book stop — not the single-straddle grid used here.
2. **Should SENSEX expiry-day size be 6 lots?** This is the lever that moves the
   execution-failure tail linearly and costs nothing in mean per lot. If the concern
   is the −₹1,29,000 scenario rather than the −₹18,000 one, **size is the only
   control that touches it.**

**Nothing deployed. No live rule changed by phases 1–3.**

---

# PHASE 4 — the "hidden live books" alarm was WRONG. Rs18,000 stands.

**2026-08-31. Raised by me, and retracted by me in the same session.**

## What happened

Arun asked whether the Rs18,000 SENSEX expiry-day figure included the COMB/TimeB
sleeves. Checking, I found the executor's event log carrying explicit
`[REAL MONEY]` entries with `source: "REAL"` for three books that carry NO
`"mode": "live"` flag:

- `CSL_TIMEB_SENSEX` — *"closed 77300 straddle -> P&L +11868 (8 lots, cum +23659) [REAL MONEY]"*
- `CSL_TIMEB_NIFTY` — *"SOLD 24250 straddle @ 189.85 credit (6 lots) [REAL MONEY]"*
- `CSL30F_SENSEX_WED`

I concluded the flag was not the operative gate and that SENSEX expiry-day risk
was **Rs39,380, not Rs18,000 — 2.2x higher.**

**That was wrong.** I did not date the events before drawing the conclusion.

## The dates settle it

| book | first REAL event | **last REAL event** | live now |
|---|---|---|---|
| `CSL30F_SENSEX_WED` | 2026-08-26 | 2026-08-26 | no |
| `CSL_TIMEB_SENSEX` | 2026-08-19 | **2026-08-27** | no |
| `CSL_TIMEB_NIFTY` | 2026-08-17 | **2026-08-28** | no |
| `NAS_COMB20` | 2026-08-17 | **2026-08-31 (today)** | **YES** |

**TimeB was pulled from LIVE on 2026-08-28**, which is already recorded in the ops
registry: *"Arun pulled TimeB from LIVE on 2026-08-28 after -Rs8,152 in one
10:00-12:00 NIFTY window at 6 lots (TIME_EXIT; the 20% stop never fired). All TimeB
books continue on PAPER."*

So the `[REAL MONEY]` events are a truthful record of a period that has ENDED. The
flag is correct **today**. `is_live_book()` returns `['NAS_COMB20']` and that is right.

## What therefore stands, and what changes

**STANDS — the live risk answer.** SENSEX expiry-day designed risk is
**Rs18,000** (6 suite lots x the -Rs3,000/lot expiry book stop). No COMB or TimeB
sleeve is live on SENSEX. Arun's "I'm ok with this risk" was answered against the
right number after all.

**CHANGES — the historical live P&L.** Those TimeB trades WERE real money between
17-Aug and 28-Aug, so the live book's history should include them. My earlier
"live book = 7 sleeves, Rs2,38,557" excluded them:

| | |
|---|---:|
| previously reported live net | Rs 2,38,557 |
| + `CSL_TIMEB_SENSEX` (real 19–27 Aug) | Rs 23,659 |
| + `CSL_TIMEB_NIFTY` (real 17–28 Aug) | Rs 3,089 |
| + `CSL30F_SENSEX_WED` (real 26 Aug) | Rs 2,115 |
| **corrected historical live net** | **Rs 2,67,420** |

## The lesson worth keeping

**A live roster has two different questions and they need different evidence:**

- *"What is live NOW?"* -> the flag / config, read at this moment. `is_live_book()`
  answers this correctly.
- *"What was ever real money?"* -> the event log, which is append-only history.

Reading history as present state is what produced the false alarm. Any future audit
of this book must **date-filter REAL events** before concluding anything about
current exposure. The `research/139` script `real_roster.py` now prints the last
event date per book for exactly this reason.

## Ops registry

The 2026-09-05 review *"Confirm which CSL/COMB books are REALLY live"* is
**RESOLVED**: `NAS_COMB20` only. The comments asserting REAL on TimeB books are
historically accurate, not stale errors — but they read as present-tense, which is
what misled this audit. Worth one clarifying word in the source when someone next
touches it; not worth an edit to live-trading code on its own.
