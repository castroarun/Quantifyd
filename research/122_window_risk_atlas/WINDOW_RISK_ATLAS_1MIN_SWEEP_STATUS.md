# Window Risk Atlas — Decay (options) x Violent-Move Probability (years of price action), Every Window

STATUS: **RUNNING** (launched 2026-08-22 by the ops session; executed by a research agent)

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

## 5-8. (standard: status log; crash recovery — read-only, scripts re-runnable, niced;
files — scripts/, results/ incl. the full atlas CSV, RESULTS.md; findings by the agent)

Multiple-testing note: the atlas REPORTS the surface; it does not crown a new winner unless
its neighbours agree (r/120 rule) and it survives the family-wise haircut. The deliverable is
honest rows for the deployed cells first, the alternatives as context.
