# Naked-Leg Trailing Stop — Does Confirming a Breach Beat Exiting on the First Tick?  STATUS: DONE

**One line:** when a naked (survivor) short option leg's premium breaches its trailing stop, is it
better to exit on the FIRST print above the line, or to wait for the breach to hold for N
consecutive prints (and does waiting ever cost more than it saves)?

---

## 1. The Ask

**What you asked (2026-07-14):**
> *"The premium spiked to 35.70, breaching the trail at 35.48 — so we exit at the instant when the
> breach happened or await a few ticks?"* → then: *"N consecutive ticks may be 3"* → then:
> *"I'd like to measure whether the instant-vs-confirmed exit actually pays — proceed."*

**What we are actually testing.** The naked-leg trail currently exits on a SINGLE tick above the
stop. Today (12:27:41) that closed two live legs at 35.70 on a spike that fell back to ~31 within
minutes — a visible whipsaw. I shipped a 3-tick confirmation on judgement, NOT on evidence. This
study asks whether that judgement is right:

> Across every naked-leg episode we have on record, replaying each leg's REAL premium path:
> does requiring the breach to persist for K consecutive prints produce a better net P&L than
> exiting on the first print — and at what K does the delay start costing more than it saves?

Success criterion: **mean net P&L per naked-leg episode**, K=1 (instant) vs K=2,3,5,8, plus a
5-min-close rule and a no-trail (ride-to-EOD) baseline. A rule only wins if it beats K=1 on mean
P&L **and** does not materially worsen the left tail (that is the whole point of a stop).

---

## 2. The Base — what is being tested

- **Instrument:** the naked survivor leg of a NAS ATM/ATM4 strangle — a SHORT option, alone, after
  its sibling was stopped out. Its per-leg premium SL is deliberately disabled (`sl_price=999999`,
  "ride the winner"), so the trailing stop is its ONLY protection before the 15:15 squareoff.
- **The stop (unchanged, as now deployed):** `compute_short_trailing_stop()` — ATR(7) bands on the
  leg's own 5-min premium candles, stop = upper band (hl2 + 3×ATR), ratcheting DOWN only, seeded
  from 09:15 TODAY only. Sits ABOVE the premium; a short loses when the premium rises.
- **Exit rules under test** (the ONLY thing that varies):
  | Rule | Exit when |
  |---|---|
  | `K=1` (today's old behaviour) | first print with `premium > stop` |
  | `K=2,3,5,8` | `premium > stop` on K consecutive prints; any print back below RESETS the count |
  | `close5` | first 5-min candle CLOSE above the stop |
  | `none` | no trail at all — ride to the 15:15 squareoff (the pre-2026-07-14 reality) |
- **Fill assumption:** exit at the premium of the print that triggers (a market order). Cost
  sensitivity applied on top (see §6).
- **Universe:** every naked-leg episode in `nas_atm`, `nas_atm4`, `nas_916_atm`, `nas_916_atm4`.
  (`nas_atm2` / `nas_916_atm2` have ZERO naked legs by design — the move-stop closes both legs
  together — so they are correctly excluded.)
- **Period:** 2026-04-06 → 2026-07-14 (as far back as the chain recorder + the books go).
- **Sample:** **121 naked-leg episodes.**

---

## 3. The Data (and its honest limit)

- **Premium path:** `options_data.db.option_chain`, matched by `tradingsymbol`, restricted to
  `[entry_time, exit_time]` of each leg. **Cadence measured, not assumed: median gap 4s, mean 20s,
  p90 55s** (645 snapshots on a session).
- **LIMIT — snapshots are not ticks.** Live, a tick arrives ~1/sec, so 3 ticks ≈ 3s of
  confirmation. Here, 3 snapshots ≈ **12s** (median). So this study measures a **SLOWER, more
  conservative** confirmation than the one deployed. Read it as: *if K=3 at 12s is not harmful,
  K=3 at ~3s is safely fine.* It CANNOT resolve sub-second tick noise, and I will not claim it does.
- **Survivorship / look-ahead:** none by construction — each leg's path is walked forward in time
  only, within its own live window, and the stop at each point uses only candles up to that point.

---

## 4. Plan

1. Pull all 121 naked-leg episodes (entry, exit, actual exit price/reason, qty).
2. For each: rebuild the leg's 5-min candles from 09:15 today → compute the trailing stop series.
3. Walk the real chain prints inside `[naked_from, exit_time]`; apply each exit rule; record the
   exit price each rule would have got, and the P&L at the leg's real qty.
4. Compare: mean/median P&L, win rate, worst episode (left tail), whipsaw count (breaches that
   reverted), and how much later K>1 exits vs K=1.
5. Cost sensitivity: 0 / 0.15% / 0.30% slippage per exit.

**Gate:** if K=3 does not beat K=1 on BOTH mean P&L and the worst-case tail, revert to K=1 and say
so plainly. A judgement call I shipped this morning is not evidence and gets no benefit of the doubt.

---

## 5. Status

| Time (IST) | Event |
|---|---|
| 2026-07-14 ~13:0x | Folder + doc written. Data verified: 121 legs, cadence median 4s. |
| 2026-07-14 ~13:1x | Sweep run; 111 episodes had usable chain paths. |
| 2026-07-14 ~13:2x | BUG in my own sweep: K8/close5/none came out identical -- I had copied the live re-arm, which hides breaches in simulation. Fixed to a pure ratchet, re-ran. |
| 2026-07-14 ~13:3x | DONE. Verdict: K3 does not beat K1 (+275/ep, better on 31 / worse on 52). Bigger finding: the trail costs ~4,500/ep and only caps the tail. See results/RESULTS.md. |

## 6. Files

| File | Purpose | Commit? |
|---|---|---|
| `NAKED_TRAIL_EXIT_CONFIRM_SWEEP_STATUS.md` | this doc | yes |
| `scripts/run_trail_confirm_sweep.py` | the sweep | yes |
| `results/episodes.csv` | per-episode, per-rule outcome | yes (small) |
| `results/RESULTS.md` | verdict | yes |

## 7. Findings

**K=3 is NOT validated as an edge** (+275/episode, better on 31 / worse on 52) -- it is a noise
filter, kept because it is free and it kills the single-print whipsaw seen live today.

**The trail itself costs ~4,500/episode (47% of the mean)** and buys exactly one thing: the worst
case halves (-20,280 -> -9,750). Monotonic: the less it fires, the more you earn.

**Do NOT remove it on this evidence** -- the sample is selection-biased (naked legs exist because
the underlying moved AWAY from them, so they are pre-selected winners) and contains NO crash day,
which is precisely what the stop insures against.

**Next: trail WIDTH (ATR multiplier 3/4/5/6/8), not confirmation.** That is where monotonicity
points -- fire less, keep the tail cap. Not run here.
