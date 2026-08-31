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
