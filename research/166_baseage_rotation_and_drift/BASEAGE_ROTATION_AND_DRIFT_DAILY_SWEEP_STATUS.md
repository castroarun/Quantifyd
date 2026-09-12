# Open Alpha · Base Age — Rotation (swap a weak holding for a better signal) and Drift (trim a bloated winner)

**STATUS: DONE** — 12-Sep-2026 19:45 → 13-Sep-2026 00:30 IST on the VPS (`/home/arun/quantifyd`).
**Verdict: axis A rotation = SIGNAL (misses the bar by 0.006 Calmar); axis B drift = NO EDGE;
axis C interaction = NO EDGE. Nothing deployed. Full write-up in `results/RESULTS.md`.**

Research folder: `research/166_baseage_rotation_and_drift/`. Book under test: the **research/161
adopted Open Alpha · Base Age spec**, unchanged except on the axes below. Nothing here is
deployed; the live conversion (research/165) is a separate job.

---

## 1. The Ask

**What Arun asked (12-Sep-2026, evening):**

> "if we have some new entrants coming up and let's say we have all slots already taken, is
> there a way we could rank our positions, swap the better one in for a bad one out? can we
> roll out some kind of study?"

**What research/164 found that widens the question.** At *every* slot count tested, the
commonest reason a qualifying Base Age signal is NOT taken is that the book has **no cash**,
not that it has **no free slot**. At the incumbent 16 slots: 3,619 qualifying events →
**688 entries**, **977 refused for want of a slot**, **1,955 refused for want of cash**. The
book never trims a winner, so a handful of bloated positions can absorb ~95% of NAV while
slots sit nominally free. Rotation and drift are therefore one family and this study covers
both.

**What we are actually testing.** On the frozen research/161 Base Age event list
(3,619 qualifying signals, 2005-01-03 → 2026-09-11):

1. **Rotation.** When a qualifying signal arrives and the book cannot take it (no free slot,
   or a free slot but not enough cash), does **selling the weakest existing holding at the
   next open and buying the new signal at that same open** improve the book after tax, after
   25 bps a side, and after the extra tax the early sale crystallises? Ranked by *what*
   measure of "weakest"? With *what* margin of improvement required before the swap fires?
2. **Drift / trimming.** Does **trimming a position that has grown past a multiple of its
   target weight** back to target — at month-end, or on demand when a signal is cash-blocked
   — convert cash-refusals into entries and improve the book? Does **sizing a new entry to
   the cash actually available** (instead of refusing it) do the same?
3. **Interaction.** Best rotation rule × best drift rule, and each at 40 and 60 bps.

**Success criterion:** after-tax Calmar over the full window, paired on the same 30 seeds,
subject to after-tax CAGR ≥ the baseline and with the tradeability gate shown every time.

**Falsification.** If no rotation rule beats the random-swap null on ≥ 20 of 30 paired seeds
in BOTH windows, rotation is declared **NO EDGE** and the answer to Arun is "no, do not swap".
If trimming and cash-sizing convert refusals without lifting after-tax CAGR, drift is declared
**NO EDGE** and the answer is "let the winners run".

---

## 2. The Base — what is held fixed

The **research/161 adopted spec**, reproduced by research/164's engine copy and verified here
before any cell runs.

| Element | Setting |
|---|---|
| Entry signal | first **close above the prior all-time-high close**, where the prior ATH close is **≥ 60 trading bars old** and the stock fell **≥ 20%** below it in between; **20-day median traded value ≥ ₹2 cr** at the trigger; funds excluded (`etf_exclusions.json`); **60-bar per-symbol re-arm**; no volume filter, no saucer filter |
| Entry fill | **next day's open** |
| Exit | **SuperTrend(14, 4) close trail** — signalled on the close, filled at the **next open**. No hard stop, no target, no time stop |
| Slots / size | **16 slots @ 6.25% of NAV** per new position |
| Contested slot (baseline) | **seeded random draw** — this is the research/161 incumbent and the null that every ranked rule is measured against |
| Contested slot (live tie-break) | **largest 20-day traded value (tv20)** — research/164's live recommendation; reported as a second baseline because a live book cannot "draw a seed" |
| Capital | ₹10,00,000, NSE cash CNC |
| Costs | **25 bps per side** (ladder to 40 and 60 on the shortlist) |
| Tax | 20% STCG / 12.5% LTCG above 365 days, **Indian FY loss-netting**, settled 1 April |
| Idle cash | **5.2% p.a. post-tax, credited daily on the cash balance only** (Arun's new standard as of 12-Sep-2026 — not 5.0, not 5.5) |
| Robustness basis | **30 random-draw seeds**; median [worst..best] reported, worst seed named |
| Window | 2005-01-03 → 2026-09-11 (5,378 trading days, 1,698 symbols in the panel) |

**Causality convention used by every NEW rule in this study.** A rule is decided on the
**close of the signal bar** and executed at the **next open** — the same convention the
inherited entry and exit already use. Concretely: rotation scores are read at `close[i-1]` and
the swap (sell + buy) executes at `open[i]`; a month-end trim is decided at the month-end
close and executed at the next open; a demand trim is decided at `close[i-1]` and executed at
`open[i]`. *Inherited inconsistency, disclosed:* research/161's entry sizing marks the book at
`close[i]` while buying at `open[i]`. That is kept verbatim so the baseline reproduces
bit-exactly; it is not extended to any new rule.

---

## 3. Plan — the axes and the cell count

### Axis A — Rotation (Arun's question)

When a qualifying signal cannot be taken, rank the eligible holdings by a score, take the
**weakest**, and swap if the entrant beats it by a margin. Entrant priority among several
unfilled candidates = **largest tv20** (the live tie-break), with rs252 and base age tested as
variants on the winner. At most **1 swap per day** by default (a variant tests 3).

All five hold-scores are causal, read from `close[i-1]`:

| score | "weakest holding" means | entrant's paired score | margin units | margins tested |
|---|---|---|---|---|
| `cushion` | **closest to its SuperTrend(14,4) exit line** — the one nearest to being stopped out | the entrant's own cushion above its ST line | percentage points | 0, 5, 10, 25 |
| `rs` | **lowest 12-month price return** (IBD-style relative strength) | the entrant's own rs252 | percentage points | 0, 10, 25, 50 |
| `athdist` | **furthest below its own running-max close** | 0 by construction (the entrant is making a new ATH close) | percentage points | 0, 10, 20, 30 |
| `unreal` | **lowest unrealised return since entry** | 0 (a new position has no P&L) | percentage points under water | 0, 5, 10, 20 |
| `held` | **longest held** | 0 bars | trading bars | 40, 80, 160, 320 |
| `rand` | **a uniformly random holding** — the NULL control, swapped with probability p | — | probability p | 0.05, 0.15, 0.35 |

Swap execution: sell the weakest at the **next open**, buy the entrant at that **same open**;
both legs are placeable. The early sale realises its gain or loss, which flows through the
FY-netting tax engine exactly as any other exit — **the tax cost of churn is modelled, never
approximated by a haircut**.

**Cells: 5 scores × 4 margins = 20, plus 3 null cells, plus 2 max-swaps-per-day variants and
2 entrant-priority variants on the winner = 27.**

### Axis B — Drift / trimming (research/164's finding)

| cell family | rule | values |
|---|---|---|
| B1 month-end trim | at the last close of each month, any position worth more than **K × its 6.25% target** is sold back to target at the next open | K = 1.5, 2.0, 3.0 |
| B2 demand trim | trim the **most bloated** position back to target **only when** a qualifying signal is about to be refused for cash | K = 1.5, 2.0 |
| B3 partial fill | when cash is short of a full slot, buy what the cash affords instead of refusing, provided it is at least **f × a full slot** | f = 0.25, 0.50, 0.75 |
| B4 both | best trim × best partial fill | 1 |

**Cells: 9.**

### Axis C — Interaction and cost

Best rotation × best drift (2 cells). Cost ladder 40 and 60 bps on the shortlist, and the
zero-idle-yield attribution — **re-scorings of already-selected cells, not new selection**.

### Cell budget

**Pre-registered selection budget: ≤ 120 cells. Planned: 27 (A) + 9 (B) + 2 (C) = 38.**
Scan at 10 seeds; confirm the shortlist (≤ 5 cells) at 30 seeds. Any cell added after this
point to execute a pre-registered plateau test is disclosed separately in RESULTS.md.

### Recorded per cell

swaps/yr · trims/yr · cash refusals · slot refusals · turnover ×NAV · average invested % ·
tax paid (and its ST/LT split) · ten-best-trades share of profit · median position as a share
of the held name's own 20-day traded value · win rate · avg win / avg loss · max losing streak
· trades/yr.

### Pre-registered windows, ranking metric and adoption bar

- **Fit window W1 = 2005-01 → 2015-12. Holdout W2 = 2016-01 → 2026-09**, opened once at the
  end. A cell whose **W2 CAGR falls more than 4 percentage points below its W1 CAGR is not
  robust** and cannot be recommended whatever its full-window number says.
- **Ranking metric:** after-tax **Calmar** over the full window, paired on the same 30 seeds,
  **subject to after-tax CAGR ≥ the 5.2% baseline** and the tradeability gate being shown.
- **Adoption bar (strict — this is a live book three weeks from a paper-book decision):**

  > **≥ +0.10 Calmar OR ≥ +2pp CAGR at no worse drawdown, on ≥ 20 of 30 paired seeds, in BOTH
  > windows, sitting on a plateau (its margin neighbours within ±2pp of CAGR), surviving
  > 40 bps AND 60 bps, and beating the random-swap null.**

- **Null control:** random swap-out at a matched swap rate (axis A `rand`). A rule that does
  not beat the null is noise, whatever it does to the headline.

---

## 4. Harness proof (run before any selection cell)

`sim166.py` is research/164's `sim164.simulate` — itself research/161's `bt_core.simulate`
with slots/size/tie-break made live — extended on **rotation** and **drift** only. With both
switched off it must reproduce research/164 exactly:

| | CAGR med | worst seed | MaxDD | Calmar | invested |
|---|---|---|---|---|---|
| research/164 published, 5.0% idle cash | 20.94% | 19.81% | −35.50% | 0.601 | 72.9% |
| **this harness at 5.0%** | **20.935%** | **19.81%** | **−35.50%** | **0.601** | **72.9%** |
| research/163 independent 5.2% re-run | 20.975% | 20.010% | −34.05% | 0.6135 | 72.91% |
| **this harness at 5.2% — THE BASELINE** | **20.975%** | **20.010%** | **−34.05%** | **0.6135** | **72.91%** |
| research/161 published, 5.5% idle cash | 21.26% | 19.87% | −34.80% | 0.618 | — |
| **this harness at 5.5%** | **21.26%** | **19.87%** | **−34.80%** | **0.618** | 72.9% |

**Per-seed, not just per-median.** All 30 per-seed CAGR / MaxDD / Calmar / trade-count /
win-rate values are IDENTICAL to research/164's `seedstats_proof.csv` (max abs difference
0.000000), and the 30 per-seed CAGRs are identical to research/163's independent 5.2%
implementation (0.000000). The harness also reproduces research/164's axis D at 5.0% to the
digit (tv20 21.72 / −32.67 / 0.665; rs252 20.59 / −36.05 / 0.571; age 21.75 / −37.59 / 0.579).

**The one new input.** `st166.pkl` holds the SuperTrend(14,4) LINE values (the panel stores
only the exit signal). Self-check: the direction rebuilt from those lines agrees with the
panel's stored `ST_14_4` signal on **5,122,891 of 5,122,891 bars — 100.0000%**.

---

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-12 19:40 | Read r/161, r/162, r/164; VPS checked | load 1.12, 4 cores, no competing sweep; other agents are on frontend/ and services/ only |
| 2026-09-12 19:50 | Sections 1-4 written and shipped — **pre-registration closed** | 38 planned selection cells, budget 120 |
| 2026-09-12 23:25 | `st166.pkl` built (44 s) | SuperTrend line self-check 5,122,891/5,122,891 bars = 100.0000% |
| 2026-09-12 23:35 | **Harness proof PASSED** | per-seed identical to research/164 (5.0%) AND research/163 (5.2%); baseline locked at 20.975% / −34.05% / 0.6135, worst seed 20.010% |
| 2026-09-12 23:45 | Scan done, 33 cells @ 10 seeds (64 s) | rotation raises CAGR but blows the drawdown; only `unreal` beats baseline Calmar; every null loses |
| 2026-09-12 23:52 | Plateau + interaction cells, 20 @ 10 seeds (41 s) | entrant priority matters as much as the exit rank; month-end trimming has no plateau |
| 2026-09-13 00:02 | All 53 cells re-run @ 30 seeds (5.5 min) | best cell `X_entrs_unre_m010` 22.73 / −31.78 / 0.715, 30/30 paired seeds |
| 2026-09-13 00:08 | **Controls run** — the decisive ones | sell-without-buying = 0.603 (WORSE than doing nothing) → the replacement is the edge, not the sale; unconditional −10% stop = 0.677 but fails the CAGR clause |
| 2026-09-13 00:12 | Cost ladder 40/60 bps, 27 cells each | ranking unchanged at every rung |
| 2026-09-13 00:20 | Schema drift caught in `cells_full.csv`; whole 30-seed stage regenerated | 3 config columns had been added mid-run; rebuilt clean, 59 cells |
| 2026-09-13 00:26 | **Tie-break instability found and verified** | research/164's tv20 Calmar advantage REVERSES between 5.1% and 5.2% idle cash; my harness reproduces r/164 at 5.0% exactly, so it is a single-path artifact, not a bug |
| 2026-09-13 00:30 | Outlier deletion, report, RESULTS.md, PUBLISH_NOTE.md written | verdict: SIGNAL / NO EDGE / NO EDGE; nothing deployed |

---

## 6. Crash recovery — how to resume without Claude

Everything runs on the VPS at `/home/arun/quantifyd`, python `venv/bin/python3`.

```bash
cd /home/arun/quantifyd/research/166_baseage_rotation_and_drift/scripts

# 0. what finished?
ls -la ../results/*.csv ../results/*.log
tail -20 ../results/*.log

# 1. is anything still running?
pgrep -af run166.py

# 2. inputs (regenerate only if missing)
#    events166.csv  — copied from research/164/results/events164.csv (frozen event list)
#    panel164.pkl   — read READ-ONLY from ../../164_baseage_slots_sizing/results/
#    st166.pkl      — SuperTrend(14,4) line values; rebuild with:
/home/arun/quantifyd/venv/bin/python3 build_st166.py        # ~4 min, reads market_data.db

# 3. resume a stage — every stage skips cells already present in its cells_<stage>.csv
/home/arun/quantifyd/venv/bin/python3 -u run166.py --stage=proof --seeds=30 --workers=1
/home/arun/quantifyd/venv/bin/python3 -u run166.py --stage=scan  --seeds=10 --workers=2
/home/arun/quantifyd/venv/bin/python3 -u run166.py --stage=full  --seeds=30 --workers=2
/home/arun/quantifyd/venv/bin/python3 -u run166.py --stage=cost  --seeds=30 --workers=2 --bps=40
```

**Safe to inspect:** every file under `results/`.
**Do NOT touch:** `../../164_baseage_slots_sizing/` (read-only input), `frontend/`,
`services/`, `static/app/`, `ops_center.py`, `TODO.md`, `docs/` — two other agents were
working in those trees on 12-Sep-2026.

**Resume rule:** a stage's `cells_<stage>.csv` is the done-set. Delete a row to re-run that
cell; delete the file to re-run the stage.

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `BASEAGE_ROTATION_AND_DRIFT_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/bt_core.py` | research/159→161 panel + indicator core, copied verbatim (needed to unpickle the panel) | yes |
| `scripts/build_st166.py` | builds `st166.pkl`, the SuperTrend(14,4) **line values** the cushion score needs | yes |
| `scripts/sim166.py` | the simulator: research/164's engine + rotation + drift | yes |
| `scripts/run166.py` | cell grid, incremental CSV, resumable | yes |
| `scripts/report166.py` | tables, YoY, paired tests, charts | yes |
| `results/events166.csv` | frozen research/161 event list with ranking attributes | yes (small) |
| `results/cells_*.csv` | one row per cell — the decision evidence | yes |
| `results/seedstats_full.csv`, `seedstats_ctrl.csv` | one row per (cell, seed) for the 30-seed decisions | yes (the 10-seed scan and cost-rung ones are gitignored) |
| `results/paired166.csv` | paired seed-win table | yes |
| `results/RESULTS.md`, `results/PUBLISH_NOTE.md` | verdict + app study entry | yes |
| `results/st166.pkl`, `results/navs_*/` | heavy, regenerable | NO — gitignored |

---

## 8. Findings

**Full write-up: `results/RESULTS.md`. Tables: `results/tables166.md`, `results/outliers166.md`.
Figure: `results/r166_curves.png`.**

1. **Rotation is real but misses the bar.** The only ranking that works is **swap out a holding
   more than 10% under water** — +1.68pp after-tax CAGR at a shallower drawdown, 30/30 paired
   seeds, both windows, 4.2 swaps a year, surviving 40 and 60 bps. Paired ΔCalmar **+0.094**
   against a pre-registered **+0.10** bar. Not adopted.
2. **The obvious rankings are the worst.** Rotating into momentum, or out of whatever is closest
   to its trailing stop, raises CAGR to 24.08% and the drawdown to −43.8%. Calmar falls in all
   twelve of those cells, on 0–2 of 30 seeds.
3. **Two controls decide the mechanism.** Selling the loser WITHOUT buying the entrant scores
   0.603 — worse than never swapping (0.613). So the edge is the replacement, not the sale. A
   rate-matched random swap loses 2.29pp of CAGR to the ranked rule.
4. **Trimming is NO EDGE, and the reason is the finding.** Demand-trim + partial fill abolishes
   the cash constraint (1,953 cash refusals → 67) and buys +0.97pp CAGR / +0.006 Calmar,
   because slot refusals rise 978 → 2,796. Cash and slots are ONE constraint — slot-time —
   wearing two hats. This refines research/164's closing claim.
5. **Correction to research/164.** The tv20 contested-slot tie-break is DETERMINISTIC and has
   no ensemble. Its CAGR advantage is stable across idle-cash rates (21.40–21.76%); its
   −32.67% drawdown / 0.665 Calmar is a coin flip that reverses at 5.2%. **Adopt tv20 for the
   CAGR; strike the Calmar claim.**
6. **Recommendation for the live book (research/165): convert unchanged.** Dated re-test
   registered for **2027-03-13** after six months of live operation.
