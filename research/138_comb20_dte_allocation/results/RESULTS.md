# RESULTS — COMB20 day allocation, and what the live book is actually doing

**Verdict: SIGNAL — one change to a live book is justified by the evidence.
Nothing deployed. One earlier claim of mine is retracted below.**

research/138 · phase 2 · 2026-08-31

---

## 0. First, a correction that changes the size of everything

Every portfolio figure I gave earlier this session pooled the live systems with
their paper shadows, and normalised everything to 10 lots. That was the right
footing for *comparing constructions* — which is what was asked — but it is the
wrong instrument for "what should change in the live book", because most of what
it measured is not the live book.

Taking the roster from the two files the executors actually gate on
(`backtest_data/nas_day_matrix.json` for `live: true` + enabled DTEs, and the
`BOOKS` dict in `research/111/scripts/csl_paper_exec.py` for `"mode": "live"`):

| | pooled view (what I reported) | **real money only** |
|---|---|---|
| sleeves | 20 | **7** |
| days | 97 | 43 |
| net | ₹12,30,504 | **₹2,38,557** |
| max drawdown | −₹5,99,106 | **−₹36,082** |
| return / DD | 2.05 | **6.61** |
| t on daily P&L | 1.29 | **2.32** |
| green days | 54% | 63% |

**The live book is materially healthier than the pooled view suggested.** The
−₹5.99L drawdown and the 2.05 ratio belong to a set of paper shadows, not to
Arun's money. The real book is 43 days, ret/DD 6.61, t 2.32.

The live roster is exactly seven sleeves:

| sleeve | venue | live days | real net | maxDD | t |
|---|---|---|---:|---:|---:|
| `nas_916_atm` | NIFTY | Mon, Tue | ₹1,11,785 | −₹7,692 | 2.10 |
| `nas_916_atm4` | NIFTY | Mon, Tue | ₹75,327 | −₹20,342 | 1.52 |
| `nas_916_atm2` | NIFTY | Mon, Tue | ₹31,233 | −₹42,200 | 0.53 |
| `sensex_atm2` | SENSEX | Wed, Thu | ₹13,779 | −₹4,276 | 0.89 |
| `sensex_atm4` | SENSEX | Wed, Thu | ₹8,880 | −₹4,857 | 0.94 |
| `sensex_atm` | SENSEX | Wed, Thu | ₹7,642 | −₹3,398 | 1.08 |
| **`NAS_COMB20`** | NIFTY | Mon, Tue | **−₹10,089** | −₹16,095 | **−1.00** |

Everything else — `nas_atm`/`atm2`/`atm4` and the other thirteen CSL/COMB books
— is `paper_shadow`. **`NAS_COMB20` is the only live sleeve losing money**, and
the only one with a negative t.

---

## 1. RETRACTED: "Friday is the best day and nobody is live on it"

An intermediate run showed the NIFTY shadow earning **+₹2,86,628 on 19 Fridays,
t 2.18**, against a live book that trades no Fridays at all. That looked like the
largest single opportunity on the table.

It is an artefact. That figure summed **raw** P&L across a shadow whose lot size
changed over the window (1, 2, 3, 5 and 10 lots all appear in the same table), so
the later, larger-lot Fridays dominated the sum. Normalised to a constant 10 lots,
the same 16 Fridays are:

> net **+₹83,407** · mean ₹5,213 · **t 0.41** · maxDD −₹114,217 · green 8/16 ·
> **top 3 days = 272% of the total**

Top-3 at 272% of the total means three days carry more than the entire result and
the other thirteen give most of it back. That is noise. **Friday is not an
opportunity for this construction and I withdraw the claim.** It is a good
illustration of why the lot-normalisation Arun insisted on is not cosmetic.

---

## 2. The confound test: is Thursday good, or is 30% good on Thursday?

Phase 1 found DTE3 (Thursday) the strongest cell of the 30% stop — ₹1,55,265 over
18 days, t 3.85. That alone could not justify a change, because it might have been
a fact about the stop rather than the day. The source
(`static/app/options_study.json`) holds the **full untruncated** 5-min premium path
per recorded day, so the whole grid can be re-priced.

Thursday (DTE3), every stop level, same 18 days:

| stop | net | t | maxDD | stops fired |
|---|---:|---:|---:|---:|
| SL 15 | ₹99,885 | 1.61 | −₹34,090 | 2/18 |
| SL 20 | ₹1,55,265 | 3.85 | −₹15,790 | **0/18** |
| SL 25 | ₹1,55,265 | 3.85 | −₹15,790 | 0/18 |
| SL 30 | ₹1,55,265 | 3.85 | −₹15,790 | 0/18 |
| SL 40 | ₹1,55,265 | 3.85 | −₹15,790 | 0/18 |
| no stop | ₹1,55,265 | 3.85 | −₹15,790 | 0/18 |

**Answer: it is the day, not the stop.** From SL20 upward the row is *identical* —
in 18 Thursdays the combined premium never once rose 20% above entry. The stop is
inert on Thursday; only SL15 is tight enough to fire, and it destroys the cell.

This resolves the question in a useful direction: the live paper twin
`NAS_COMB20_THU` already runs DTE3 at **SL 20**, which the grid shows is
*identical* to SL 30. So there is **no stop change to make** — the only thing
separating the paper twin from the phase-1 headline is **size** (5 lots vs 10)
and the fact that it is paper.

Monday (DTE1) for contrast — the cell COMB20 runs live at 30%:

| stop | net | t | maxDD | stops |
|---|---:|---:|---:|---:|
| SL 15 | ₹1,46,235 | 1.28 | −₹64,180 | 6/20 |
| SL 20 | ₹1,20,625 | 1.00 | −₹73,540 | 6/20 |
| SL 25 | ₹1,22,640 | 1.02 | −₹62,880 | 5/20 |
| **SL 30 (live)** | ₹1,44,805 | 1.23 | −₹59,630 | 2/20 |
| SL 40 | ₹1,78,735 | 1.61 | −₹59,630 | 1/20 |
| no stop | ₹1,61,380 | 1.33 | −₹63,665 | 0/20 |

Monday is positive at every stop but **weak everywhere** (t 1.0–1.6) and carries
~4× Thursday's drawdown. The live 30% is mid-pack; SL40 is nominally best but
that is one pick out of six on 20 days, so it is not evidence. **There is no stop
fix for Monday** — the cell itself is the problem, not its parameter.

Stability (each cell split in half chronologically — a cell that only works in one
half is a period artefact):

| DTE | day | 1st half | 2nd half | verdict |
|---|---|---:|---:|---|
| 3 | Thu | ₹72,595 | ₹82,670 | **both +, and level** |
| 2 | Fri | ₹25,665 | ₹80,590 | both + |
| 0 | Tue | ₹87,610 | ₹1,90,570 | both + |
| 1 | Mon | ₹1,31,845 | ₹12,960 | **both +, but −90%** |
| 4 | Wed | ₹49,975 | −₹10,345 | flips |

Thursday is the steadiest cell in the grid. Monday's edge is nearly all in the
first half. DTE4/Wednesday flips — it was already removed from COMB20 on
2026-08-13, and this confirms that call.

**30 cells were tested here.** At that count one t≈2 arises by luck, so Tuesday's
2.09 and Monday's 1.61 are not to be trusted on their own. Thursday's 3.85, which
is stop-invariant *and* stable across both halves, is a different quality of
result.

---

## 3. The finding that actually matters: two constructions, opposite weekdays

The live NIFTY book runs two different mechanics **on the same two days**:

- the 9:16 suite (`nas_916_atm`/`atm2`/`atm4`) — per-leg stops, trails, rolls
- `NAS_COMB20` — a held straddle with one combined-premium stop

Normalised to 10 lots, their weekday profiles are close to *opposite*:

| | Mon | Tue | Wed | Thu | Fri |
|---|---:|---:|---:|---:|---:|
| **9:16 suite** (pooled) | **₹5,96,652** | −₹33,662 | ₹2,65,220 | −₹44,522 | ₹83,407 |
| *t* | **2.89** | −0.24 | 0.50 | −0.37 | 0.41 |
| **held straddle + CSL** | ₹1,44,805 | ₹2,78,180 | ₹39,630 | **₹1,55,265** | ₹1,06,255 |
| *t* | 1.23 | 1.85 | 0.35 | **3.85** | 0.94 |

The suite earns on Monday and gives it back on Tuesday and Thursday. The held
straddle earns on Tuesday and Thursday, and is weakest on Monday.

**The live configuration puts both on Mon+Tue.** So COMB20 stacks correlated size
onto Monday — the suite's single best day, which needs no help — and sits out
Thursday, its own best cell and one of the two days the suite loses money.

That is the whole of COMB20's −₹10,089 / t −1.00, and it is a *day-allocation*
error rather than a parameter or execution one. It also explains why the replay
and the live record disagreed on Mondays: the replay's Monday profit is
concentrated in April–June, and COMB20 only started trading Mondays in August.

Live Monday record for COMB20, 10-lot normalised: 17-Aug −₹605 · 24-Aug −₹24,495
· 31-Aug −₹31,530 = **−₹56,630 across three Mondays, all three losing.**

---

## 4. The proposal — NOT deployed

**Move `NAS_COMB20`'s DTE1 (Monday) allocation to DTE3 (Thursday).** Keep DTE0
(Tuesday) as it is. Size unchanged at 2 lots. No stop change — SL20 and SL30 are
identical on Thursday.

Allocation test over the same 92 recorded days (10-lot basis):

| allocation | net | maxDD | ret/DD | t |
|---|---:|---:|---:|---:|
| live today | ₹6,28,388 | −₹1,21,850 | 5.16 | 2.89 |
| drop DTE1 | ₹4,83,582 | −₹88,475 | 5.47 | 2.62 |
| **drop DTE1, DTE3 at full size** | ₹5,61,215 | −₹88,555 | **6.34** | **3.01** |
| DTE3 to full size (keep DTE1) | ₹7,06,020 | −₹1,21,930 | 5.79 | 3.22 |

**Three things that argue against doing this immediately, stated plainly:**

1. **Margin.** NIFTY was deliberately taken off Thursdays on 2026-08-27 because
   "Thursday is SENSEX expiry and NIFTY was competing for the same margin on
   SENSEX's best day". That constraint is real and this proposal walks back into
   it. It needs a margin check at 2 lots before it is actionable — 2 lots is
   small, but small is not zero.
2. **The live Thursday record is two days and negative** (−₹13,430 at 10-lot
   equivalent, from `NAS_COMB20_THU` paper). Thursday's case rests entirely on
   the 18-day replay. Two live days is not a contradiction, but it is not
   confirmation either.
3. **Monday is not negative on the replay** at any stop. The case for moving is
   comparative (Thursday is better and steadier, and Monday duplicates the
   suite), not that Monday loses money in the abstract.

Given (1) and (2), the honest recommendation is the **cheap** version: leave real
money where it is and **let `NAS_COMB20_THU` keep running on paper until it has
~8 Thursdays**, then re-read. The one thing worth doing now is deciding whether
Monday's COMB20 cell should keep running at all, given it has lost on all three
live Mondays and adds correlated size to the day that least needs it.

Any of this is a **strategy change**: its own STATUS-MD, its own evidence, and an
after-15:40 deploy. Not a parameter tweak, and not something to fold into other
work.

---

## 5. One thing to check that is not a research question

In `research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py`, `is_live_book()`
requires `B.get("mode") == "live"`. **Only `NAS_COMB20` carries that flag.**
But the inline comments assert two other books are real money:

- `CSL_TIMEB_SENSEX` — *"2026-08-18: 6L->8L REAL (notional parity w/ NIFTY TB@8L)"*
- `CSL_TIMEB2_LIVE` — *"research/125 expiry-Tuesday afternoon window — REAL (user 2026-08-25)"*

Neither has `"mode": "live"`, so by that function both evaluate as paper, and
`/app/straddles` renders them as paper. Either they are executed by a different
path, or the flag is missing and two books Arun believes are live are not
trading real money. **I have not touched it** — it is live-trading code. Worth
confirming against the broker.

---

## Files

| file | purpose | commit |
|---|---|---|
| `scripts/confound.py` | DTE × stop grid, stability split, allocation test | yes |
| `scripts/monday.py` | Monday replay vs live, in time order | yes |
| `scripts/dte_cell.py` | phase 1 — 30% stop by DTE | yes |
| `scripts/live_book.py` (repo `scripts/`) | pooled 20-sleeve view, all lot-normalised | yes |
| `scripts/real_book.py` (repo `scripts/`) | real-money-only book, live DTEs, real lots | yes |
| `scripts/verify_friday_thursday.py` (repo `scripts/`) | the two checks that killed claim 1 | yes |
