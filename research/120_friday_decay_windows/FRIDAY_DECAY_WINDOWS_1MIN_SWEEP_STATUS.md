# Friday Decay Windows — Where Does Premium Bleed Safest, and Is a Second TimeB Slot Worth It?

STATUS: **RUNNING** (launched 2026-08-21 by the ops session; executed by a research agent)

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

(to be written by the research agent)
