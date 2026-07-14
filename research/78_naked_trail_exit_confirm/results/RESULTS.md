# research/78 — Naked-leg trail: instant vs confirmed exit → **SIGNAL (weak), and a bigger finding**

**Verdict on the question asked: K=3 does NOT measurably beat K=1.** It is +₹275/episode on the
mean (+5.5%) but it is **better on only 31 episodes and worse on 52** — a fat-tailed mean, not an
edge. Keeping K=3 is harmless and mildly positive, but it is **not the lever that matters**.

**The finding that does matter: on this sample the trailing stop itself is expensive.** Riding the
naked leg with NO trail earns **+₹9,507/episode** vs **+₹4,984** for the instant stop — the trail
costs roughly **₹4,500 per episode, ~47% of the mean** — and its entire benefit is halving the
worst observed loss (**−₹20,280 → −₹9,750**). That is a risk-preference decision, and **this
sample cannot settle it** (see the caveats — they are the whole story).

## The numbers (111 naked-leg episodes, 2026-04-06 → 2026-07-14, real chain prints)

| rule | mean | median | win% | worst | vs K1 (better/worse) |
|---|---|---|---|---|---|
| **K1** (instant — the old live behaviour) | +4,984 | +2,239 | 73% | −9,750 | — |
| K2 | +4,799 | +2,194 | 73% | −9,750 | −185 (26 / 57) |
| **K3** (what is deployed) | **+5,259** | +2,503 | 76% | −9,750 | **+275 (31 / 52)** |
| K5 | +6,071 | +2,685 | 72% | −9,750 | +1,087 (28 / 55) |
| K8 | +6,537 | +2,590 | 75% | −9,750 | +1,552 (31 / 52) |
| close5 (5-min close) | +4,953 | +2,003 | 77% | −9,750 | −31 (42 / 35) |
| **none** (no trail, ride to EOD) | **+9,507** | +3,546 | 81% | **−20,280** | +4,523 (51 / 32) |

**Monotonic in K: the less the stop fires, the more you make.** That is the signature of a stop
that is pure cost *in this sample*. Costs are irrelevant here (0→30bp moves the mean by <₹60).

Note every trail rule shares the SAME worst case (−9,750) while no-trail is −20,280: the stop is
doing exactly one job — capping the tail — and it is charging ~₹4.5k an episode to do it.

## Why you must NOT read this as "remove the trail"

1. **Selection bias, and it is severe.** A naked leg exists *because its sibling was stopped out* —
   i.e. the underlying moved AWAY from it. The population is therefore **selected for legs already
   winning**. Of course riding them looks good. This is not a sample of "short options in general".
2. **There is no crash in the window.** 2026-04→07 had no gap-down/vol-spike day. A naked short with
   no stop is precisely the position a crash destroys. **−20,280 is the worst *observed*, not the
   worst *possible*.** Selling tail insurance always looks profitable until the one day it isn't.
3. **Snapshot ≠ tick.** The recorder prints every ~4s (median), not ~1s. So the K=3 measured here is
   ~12s of confirmation vs ~3s live — the deployed rule is *even closer to K1* than this shows,
   i.e. its real effect is smaller than the already-small +₹275.

## What I am doing about it

- **Keep K=3.** It costs nothing, its mean is mildly positive, and it does filter the exact
  single-print whipsaw we watched happen live today (12:27:41, exit at 35.70 on a spike that fell
  back to ~31 within minutes). But I will not claim it as an edge — it is a noise filter.
- **Do NOT remove the trail.** The evidence for removing it comes entirely from a benign,
  selection-biased window, and what it insures against is not in the data.
- **The real lever is trail WIDTH, not confirmation.** Monotonicity says: fire less, earn more —
  but keep the tail cap. That points at a **wider ATR multiplier** (currently 3), not at deleting
  the stop. Testing multiplier 3/4/5/6/8 is the obvious next study and was NOT run here.

## Method (auditable)

`scripts/run_trail_confirm_sweep.py` — for each of 111 naked legs, rebuilds the leg's own 5-min
premium candles from 09:15, computes the deployed trailing stop (ATR7, ×3, ratchet-down), walks the
REAL chain prints inside `[entry_time, exit_time]`, and applies each rule. Forward-only; the stop at
any instant uses only bars closed by then. No orders, no live state touched.

**Self-correction:** the first run reported K8 / close5 / none as byte-identical. That was a bug in
*my sweep*, not a result — I had copied the live function's habit of re-arming the stop above a
breakout, which is harmless live (the breach closes the position) but in simulation makes the stop
leap above the price so a breach can never register. Fixed to a pure ratchet; every number above is
post-fix.
