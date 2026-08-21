# Static vs Ratcheting Backstop — Should the Defence Move as the Trade Wins?

STATUS: **DONE** (2026-08-21) — verdict **NO EDGE**, no live change. Read-only replay; nothing
in services/, config or the live books was touched.

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
| 2026-08-21 ~13:0x IST | Sweep launched (read-only, niced) | 17 defence variants x 4 live constructions x 85 recorded days; today (08-21) excluded as partial |
| 2026-08-21 ~13:3x IST | Sweep DONE | 4,012 rows / 236 construction-days written to `results/ratchet_detail.csv` |
| 2026-08-21 ~13:4x IST | Aggregation DONE | pooled + per-construction + per venue-DTE tables; **no variant beats STATIC** |
| 2026-08-21 ~14:0x IST | Give-back anatomy DONE | peak sits at 90th pct of window (median); only 2/236 days went deep-then-hit-stop |
| 2026-08-21 ~14:1x IST | RESULTS.md written, STATUS closed | verdict **NO EDGE** — leave the defence alone |

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

**Verdict: NO EDGE — do not ratchet the defence. Leave the live rule exactly as it is.**
Full write-up + all tables: `results/RESULTS.md`.

### The premise is right; the feared event is not there

- At the moment of **peak** open profit the stop really is a median **Rs 5,185/lot away**
  (p90 8,294, max 13,019). Arun's observation is quantitatively correct.
- But of 236 construction-days, **38 (16%) ever went deep** (open profit >= 40% of credit),
  and of those **exactly 2 later came back and touched the static stop** — 0.85% of all days
  (COMB_NIFTY DTE0 on 2026-06-02 and 2026-06-30).
- And there is little to trail: peak open profit lands at the **90th percentile of the window
  (median)** and in the **last 10% of the window on 50% of days**. Median give-back peak->close
  is **Rs 289/lot**, about one round trip's costs.

### Every ratchet loses, and none improves the give-back

Pooled, net Rs/lot, n=236: **STATIC 136,683**. Best defended alternative RATCHET_K2.5 =
135,334 (-1,349). The shapes that actually engage cost -51k to **-82k**, up to **60% of the
book's entire profit**. Median give-back gets **worse** as the rule tightens
(289 -> 302 -> 324 -> 373 -> 428) — a trailing rule can only fire *after* a retrace, so it
manufactures the give-back it is meant to prevent. Only RS_GB_1000 halves the p90 give-back
(3,401 -> 1,590), for -Rs 81,520.

### Monotonicity: the gradient runs to the boundary

k = 1.5 / 1.75 / 2.0 / 2.5 / STATIC -> 78,882 / 112,401 / 132,648 / 135,334 / **136,683**.
No interior optimum. The best ratchet is no ratchet.

### My stated prior was WRONG

I predicted the breakeven clamp would be the strongest candidate. It is nearly inert
(changes 0-6 of 236 days) but produced **more cut-shorts than rescues at every trigger level**
(0/2, 1/4, 3/6) and a negative uplift (-2,449 / -4,383 / -10,472), with gb_p50 unchanged and
gb_p90 slightly worse. Asymmetry only pays if the bad tail exists at meaningful frequency.

### Byproduct for a SEPARATE study (not acted on here)

The NO_DEFENCE control prices the existing defence: pooled it costs Rs 4,913 and buys a worst
day of -6,667 instead of -16,527 — a sane premium, keep it. But it is paid almost entirely on
**SENSEX expiry Thursday**, where 4 backstop firings cost **Rs 28,059** while improving the
worst day by Rs 370. Independently reproduces research/114 and the 2026-08-19 config note. That
is a *level* question with n=4 — own STATUS-MD, own evidence, alongside the 11-SEP SX-Thu review.

### Recommendation

**Leave the defence alone.** For the position that prompted this (TimeB NIFTY, credit 175.13,
8 lots, 20% combined-SL at 210.2, Rs 18,214 max loss): that Rs 18,214 is a fixed ceiling, it does
not grow as the trade wins. The right lever for discomfort with the distance is **size**, not
stop placement. Re-open only if the recorder accumulates a stressed regime, or if the
deep-then-reverse rate rises materially above the 2-in-236 observed here.
