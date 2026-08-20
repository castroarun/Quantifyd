# Static vs Ratcheting Backstop — Should the Defence Move as the Trade Wins?

STATUS: **QUEUED** — runs after 15:40 IST on 2026-08-20 (read-only; queued behind the
live deploy so nothing competes for attention during market hours)

## 2. The Ask

**What Arun asked (2026-08-20):** "let's say we start with a backstop as 50%, the trade is
on, we reach some decent profit levels, now the distance from this point up to that
original 50% is a lot — now it's no more just a bad crack in the market but a big bad
crack. Should we still hold the same defence which is now way extended? What else can we
do in this situation — should be assessed deeply."

**What we are actually testing.** Our combined-SL and backstop levels are anchored to the
**entry credit** and never move. As the straddle decays, open profit grows while the stop
stays put, so the *give-back at risk* grows with every point of profit:

> TB-SENSEX today: credit 173.79, backstop 260.7 (1.5x). Decay to a combined 80 means
> about +Rs15k open on 8 lots — but the stop is still 180 points away, i.e. roughly
> Rs28.9k of give-back before the defence engages.

So: **does a ratcheting defence — one that tightens as profit accrues — beat the static
one, and which ratchet?** Applies to every combined-SL book we run (COMB, TimeB, the
SL-none Thursday books), so the answer is portfolio-wide, not one sleeve.

## 3. The Base

- **Data:** `options_data.db :: option_chain`, 1-minute, ~85 days, NIFTY + SENSEX,
  read-only. Same replay harness as research/113 and research/114.
- **Constructions:** the live ones — 09:16 ATM straddle to 15:20 (COMB shape) and the
  per-DTE TimeB windows — per venue, per DTE.
- **Costs:** 1.0 pt/leg-side SENSEX, 0.5 NIFTY, plus Rs30/leg-side/lot. Ratcheting rules
  do not add trades unless they fire, so cost differences come only from exit frequency.

## 4. Plan — the variants

| # | Rule | Definition |
|---|---|---|
| 0 | **STATIC (baseline)** | stop = 1.5 x credit, fixed. What we run today. |
| 1 | Breakeven clamp | when combined <= 0.6 x credit, move stop to credit (worst case becomes ~flat) |
| 2 | Multiplicative ratchet | stop = min(prev_stop, 1.5 x current_combined) — always 50% above the *current* premium, never widening |
| 3 | Ratchet k=1.3 / 2.0 | as #2 at tighter and looser multiples, to test monotonicity |
| 4 | Peak-giveback trail | exit if open profit retraces 30% / 50% of peak profit |
| 5 | Rupee giveback | exit on a fixed Rs/lot retrace from peak (the shape the NIFTY book stop already uses) |
| 6 | Time-scaled | static until 12:00, then ratchet (theta earned, gamma rising) |
| 7 | Hybrid | breakeven clamp + peak-giveback, whichever binds first |

**Metrics — not just total.** For each variant: net total, mean/day, median, win%, worst
day, AND the two that answer Arun's actual question:
- **give-back distribution:** peak open profit minus realised, per day
- **rescue count:** days where the ratchet exited materially better than static

**Success criterion:** a ratchet must improve the give-back distribution *without*
materially cutting the total — research/114 showed premature exits are the main way we
destroy this edge, so a ratchet that fires early and often will look "safer" while being
worse. Both must be reported side by side.

## 5. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-20 ~13:4x IST | Question raised, study queued | runs after the 15:40 deploy |

## 6. Crash Recovery

Read-only replay; no live state. Scripts in `scripts/`, outputs in `results/`. Re-run
`venv/bin/python3 research/116_ratcheting_backstop/scripts/<runner>.py` any time.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `RATCHETING_BACKSTOP_1MIN_SWEEP_STATUS.md` | this file | yes |
| `scripts/run_ratchet_sweep.py` | the bake-off | yes |
| `results/*.csv` | per variant-day detail + summary | yes |
| `results/RESULTS.md` | verdict + recommendation | yes |

## 8. Findings

(pending)
