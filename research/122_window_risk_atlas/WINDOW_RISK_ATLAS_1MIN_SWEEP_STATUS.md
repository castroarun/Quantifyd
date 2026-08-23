# Window Risk Atlas — Decay (options) x Violent-Move Probability (years of price action), Every Window

STATUS: **DONE** (2026-08-21 19:0x IST; full report in results/RESULTS.md)

## 2. The Ask

**What Arun asked (2026-08-22):** "did you look at other windows? I also asked for the
probability of the worst-case scenarios. For the decay look at the options data we have; for
the violent market move, look at the price action data we have for years (the least timeframe,
1-min or whatever) and come up with your report/recommendations in a table clearly with R:R,
90th percentile, 95th, probability of losses, other relevant stats."

**What we are building: a risk atlas of intraday short-straddle windows.** For every
candidate window (not only the five deployed TimeB cells), one row that marries:
- **the decay actually earned** — from the recorded option chain (~16 days per weekday), and
- **the probability of the move that hurts** — from the LONG price-action sample
  (SENSEX 1-min 2021→now, ~1,350 days; NIFTY per the r/121 licence below),
so each window shows its typical profit, its tail percentiles, its loss probabilities and an
honest R:R — and a recommendation per deployed cell: keep / move / downsize / drop.

## 3. The Base

- **Windows to cover (per venue, per applicable DTE):** the 5 deployed cells PLUS the
  alternatives: every start 09:20→14:30 in 30-min steps x durations 60/90/120 min and
  hold-to-15:20 — but REPORTED as an atlas, with the deployed cells highlighted. r/120
  already built this surface for Friday; extend the same harness to Mon/Tue/Wed/Thu.
- **Stage A (decay, rupee truth):** `options_data.db :: option_chain`, 1-min, 2026-04-20 →
  2026-08-22, READ-ONLY. Per window: median/mean net decay, win%, worst observed, MAE inside
  the window. Costs 0.5/1.0 pt per leg-side + Rs30/leg-side/lot. Reject frozen-chain holidays
  (<50 distinct spot prints; known: 2026-05-01, 05-28, 06-26).
- **Stage B (violent moves, the long clock):** `market_data.db :: market_data_unified`.
  SENSEX `minute` 2021-01→now. **NIFTY: no 1-min series exists** — use the 5-min series under
  the r/121 licence (for MAX EXCURSION inside a fixed window, 5-min == 1-min exactly; the
  no-5-min rule bites on the path only) and say so. Per window: distribution of the maximum
  adverse excursion (in bp of spot), p50/p90/p95/p99/max, and the probability that the move
  exceeds what the window's credit could absorb.
- **The bridge:** convert Stage-B move percentiles into rupee outcomes via the 2026 credit
  observed for that window (credit ladder, not one number: use the p25/median/p75 credits).
  This is the only place the two datasets meet — document the conversion.
- **Label days by DTE, not weekday**, where expiry matters (r/118: the SENSEX expiry day moved
  twice inside the long sample — Fri 2024, Tue 2025H1, Thu 2025-09+).

## 4. The table Arun asked for (per window row)

| col | meaning |
|---|---|
| window | venue, DTE, start-end |
| n_opt / n_px | days of options / price-action evidence |
| median net P&L | options sample, 10 lots, net of costs |
| win% | options sample |
| P(loss day) | long sample: probability the window's move exceeds breakeven decay |
| p90 / p95 / p99 adverse | long sample MAE percentiles, converted to Rs at 10 lots |
| P(move > SL20 cap) | probability the day would have hit the deployed stop |
| worst observed (opt) | real worst in the options sample |
| **R:R @p90 / @p95** | median profit vs the p90/p95 adverse outcome — the honest ratio |
| verdict | keep / move / downsize / drop |

## 5. Status (live log)

State: DONE. All stages complete, RESULTS.md written, committed.

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-21 18:40 IST | Resumed from session-limit kill; scripts/ and results/ were empty, starting from scratch | plan re-read from r/120+r/121 |
| 2026-08-21 18:45 IST | Stage A launched: scripts/stage_a_alldays.py (nice -n 10, bg) | grid 12 starts x 60/90/120/HOLD + 5 deployed windows x SL20/SL25/NOSTOP, both venues, all recorded days; holiday guard (<50 spot prints) + partial-session guard (last snap < 15:15); 2026-08-21 INCLUDED (session complete, last snap 15:40) |
| 2026-08-21 18:47 IST | Stage A DONE (~82 days/venue kept, 26,046 rows) | frozen-chain holidays + partials auto-skipped |
| 2026-08-21 18:47 IST | Stage B launched + DONE: scripts/stage_b_allweekday_clock.py | SENSEX 1-min 1,354 days (2021->) + NIFTY50 5-min 2,754 days (2015->), DTE-labelled per expiry-era tables (r/118); 217,631 window-day rows |
| 2026-08-21 18:48 IST | build_atlas.py: RECONCILIATION PASSED | FRI = r/120 exact (+399/lot, worst -344, 13/14); MON = r/121 exact; WED gap resolved = one holiday-shifted expiry week (05-27 Wed was DTE0; true DTE1 = Tue 05-26); weekday-selection reproduces r/121 to the rupee |
| 2026-08-21 18:52 IST | P(loss) model upgraded to per-cell Theil-Sen (net vs terminal move); atlas rebuilt | crude decay-arithmetic version disagreed with observed loss rates (FRI 34% vs 7%); Theil-Sen version agrees on 4 of 5 cells |
| 2026-08-21 18:55 IST | analyze_alternatives.py: dominance + plateau scan | TUE/THU: no dominators; MON/WED/FRI dominators all late-day COMB-family, window overlaps, or r/120 already-adjudicated - none recommended |
| 2026-08-21 ~19:00 IST | SESSION-LIMIT interruptions x2 (during doc reads and during RESULTS write) | resumed from disk both times; no artifacts lost, no query re-run needed |
| 2026-08-21 19:05 IST | RESULTS.md published, STATUS -> DONE, INDEX row added, committed | verdicts: KEEP x4, DOWNSIZE Monday |

### Crash recovery
- Stage A output: `results/stage_a_alldays.csv` (append per day, flush per day); progress: `tail results/stage_a.log`
- Alive? `pgrep -af stage_a_alldays`
- Re-run whole stage safely (READ-ONLY on DB, output overwritten):
  `cd /home/arun/quantifyd/research/122_window_risk_atlas && nice -n 10 python3 scripts/stage_a_alldays.py`
- Stage B: `nice -n 10 python3 scripts/stage_b_allweekday_clock.py` -> `results/stage_b_window_days.csv`
- Atlas build: `nice -n 10 python3 scripts/build_atlas.py` -> `results/atlas.csv` + percentile tables
- Do not touch: `backtest_data/options_data.db`, `backtest_data/market_data.db` (read-only mounts of live data)

Multiple-testing note (restored): the atlas REPORTS the surface; it does not crown a new
winner unless its neighbours agree (r/120 plateau rule) and it survives the family-wise
haircut. The deliverable is honest rows for the deployed cells first, the alternatives as
context.


## 8. Findings (final summary; full report = results/RESULTS.md)

Atlas of 1,590 window rows (12 starts x 4 durations + 5 deployed windows, x venue x DTE x
3 stop arms), options truth 15-17 days/cell x long-sample move risk 122-358 DTE-matched days.

| Cell | median @10L | win% | R:R @p90/@p95 | bridged p95/p99 adverse | worst obs | verdict |
|---|---|---|---|---|---|---|
| MON NIFTY DTE1 13:00-14:00 SL20 | +1,240 | 71 | 1:9.6 / 1:11.8 | 14.6k / 23.2k | -4,840 | **DOWNSIZE/DROP** |
| TUE NIFTY DTE0 09:30-11:00 SL25 | +9,525 | 81 | 1:1.3 / 1:1.5 | 14.6k / 20.5k | -27,040 | **KEEP** |
| WED SENSEX DTE1 10:30-12:00 SL20 | +3,370 | 71 | 1:2.5 / 1:2.9 | 9.7k / 14.8k | -11,440 | **KEEP** |
| THU SENSEX DTE0 13:00-15:20 NOSTOP | +15,070 | 82 | 1:2.7 / 1:3.1 | 46.9k / 70.0k | -54,100 | **KEEP, size for tail** |
| FRI NIFTY DTE2 10:00-12:00 SL20 | +3,120 | 93 | 1:5.4 / 1:6.9 | 21.7k / 29.4k | -3,440 | **KEEP** |

No alternative window recommended: TUE/THU have no dominators; MON/WED/FRI dominators are
last-hour COMB-family cells, overlaps of the deployed window, or r/120's already-deferred
"start Friday earlier" - all 16-17-day medians inside a ~220-comparison family (r/120
Westfall-Young precedent: nothing at this n survives). Strongest single exhibit: the book's
biggest earner (Thursday, 58% of the five-cell take) runs stop-less by necessity (any SL20
would fire on 85% of expiry afternoons) and has already printed -54,100 in 17 recorded
Thursdays against a bridged p99 of -70k @10 lots - size is its only risk dial.
