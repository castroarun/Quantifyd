# Open Alpha re-optimised on an entry that can actually be placed — STATUS: RUNNING

One line: every parameter this book runs was chosen by sweeping against an entry no order can
execute. The optimum for the honest entry has never been looked for.

## 1. The Ask

**What Arun asked (11-Sep-2026):**

> "pls do more assessments/optimizations, i need the rturns to be +25% CAGR"

and, on sequencing:

> "all three in that order" — (1) re-optimise OA on the honest entry, (2) the eight-year
> fundamental test, (3) blend and allocation across TN / IPO / gold.

**What is actually being tested here (piece 1).** research/142 swept 680 cells — slots,
sizing, stop depth, trail length, RS threshold, liquidity floor, market gate — and every one
of them was evaluated on `trig = close > pivot` filled at `max(pivot, open)` **on the same
bar**. That entry needs the closing price known at the open. research/158 measured what it
costs: **40.8% becomes −1.7%** once the same rule is filled the way an order actually fills.

So the live book's entire parameter set is the answer to a question about a mechanic that
does not exist. This asks the same question of the mechanics that do.

**This is not a hunt for a 25% cell.** Best-of-N is luck; the playbook (§4, §5A, §6) is
explicit and was amended today with this exact scar. The deliverable is a *surface*: does
the honest entry have a broad plateau anywhere, and where. A peak that is not a plateau is
reported as noise and not adopted.

**Stated prior, before running.** The honest entry starts at 9.5% CAGR against the
look-ahead's 40.8%, and NIFTYBEES over the same window returns 11.5%. Re-optimising a
surface typically buys single digits, not triples. I expect the low-to-mid teens and will
report that plainly if that is where it stops. **25% is not a promise and may not be
reachable on this signal.**

## 2. The Base — what is fixed and what moves

**Fixed** (unchanged from the live book, so the comparison is clean):
universe = NSE dailies, 20-day median traded value ≥ ₹5cr, **funds excluded by the
name-based list** (`backtest_data/etf_exclusions.json` — r/158 found 221 gold, silver and
index funds in the universe), base depth ≤ 20% below the pivot, equal-weight slots,
25 bps per side, 2006-01-01 → 2026-08-31, 30-seed random-selection ensemble, medians
reported with the full seed range.

**Moves — stage A (entry economics):**

| Axis | Values | n |
|---|---|---|
| Entry mechanic | `stop_above_candle`, `close_same`, `open_next` | 3 |
| Trail (SMA on close) | 10, 15, 20, 30, 50 | 5 |
| Hard stop | 6%, 8%, 10%, 15%, none | 5 |

= **75 cells**, each a 30-seed ensemble. `open_same` (the look-ahead) is run once as a
reference row and labelled as unplaceable; it is never a candidate.

**Stage B (book shape), only on the region stage A finds:**

| Axis | Values | n |
|---|---|---|
| Slots | 8, 12, 16, 20, 24 | 5 |
| RS floor | 50, 60, 70, 80, 90 | 5 |
| Market gate | none, NIFTYBEES 100-SMA, 200-SMA | 3 |

= **75 cells**. Total disclosed: **150 + 1 reference**.

**Success criterion, pre-registered.** A cell is a candidate only if:
1. its CAGR beats NIFTYBEES (11.5%) after costs **and** after tax, and
2. its neighbours in every swept axis are within a reasonable band of it — a plateau, not a
   spike, and
3. its worst seed is not negative.

If no cell clears all three, the answer is that this signal does not support the target, and
that is the finding.

## 3. Plan

| Stage | State |
|---|---|
| A — 75 cells, entry × trail × stop | RUNNING |
| B — 75 cells, slots × RS × gate, on A's plateau | QUEUED |
| After-tax re-run of the surviving region | QUEUED |
| Verdict + RESULTS.md | — |

Frames are loaded once per trail value (5 loads) and every cell for that trail runs
in-process, per the mandatory optimisation rules in CLAUDE.md. CSV is written incrementally
and completed cells are skipped on restart.

## 4. Status

| Time (IST) | Event |
|---|---|
| ~16:05 | r/158 closed: 26 clean-universe arms + 2 abort arms. Abort without slot recycling also fails (−21.2% / −13.3%). |
| ~16:15 | r/159 opened; stage A launched |

## 5. Findings

Pending.

## 6. Crash Recovery

```bash
ssh arun@94.136.185.54 'pgrep -af sweep_honest'
ssh arun@94.136.185.54 'tail -5 /tmp/mpf/r159_stageA.log'
ssh arun@94.136.185.54 'wc -l /home/arun/quantifyd/research/159_oa_honest_reoptimization/results/stageA.csv'
```

Resume: re-run `scripts/sweep_honest.py` — it reads the existing CSV and skips finished
cells. Nothing here touches live state; no order is placed and no book file is written.

**Live state is unchanged by this study.** Open Alpha entries remain PAUSED from 11-Sep
11:18 (crontab backup `/tmp/mpf/ct.bak.20260911-111828`); exits, stops and the trail keep
running. TN and IPO untouched.

## 7. Files

| File | Purpose |
|---|---|
| `scripts/sweep_honest.py` | stage A + B runner, frames loaded once per trail |
| `results/stageA.csv` | one row per cell, written incrementally |
| `results/RESULTS.md` | verdict |

## 8. Verdict

Pending.
