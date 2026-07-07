# ATM2 exit-trigger revalidation — 30% premium SL vs strike-gated SL vs 0.4% move-stop

Pre-live-money check before arming the 9:16 systems at 2 lots. Chain replay on `options_data.db`
(53 sessions), 916 + squeeze ATM2 entries, all cascade/re-enter, no cooldown (isolate the trigger).
Three mechanics:
- **premium30** — old live behaviour: 30% per-leg SL → close+reopen at current ATM (can be the SAME
  strike = the churn Arun saw live).
- **prem30_gate** — Arun's fix (mechanic C): 30% SL breach AND price reached the NEXT ATM strike →
  only then close+reopen; else HOLD (never same-strike churn).
- **move0.4** — current v3 (live): ±0.4% underlying move → re-center at new ATM.

## 916 ATM2 (what's going live), per lot, 53 days

| Mechanic | Overall | Mon (DTE1) | Tue/expiry (DTE0) | Wed (DTE6) |
|---|---|---|---|---|
| premium30 | −8.4k | +29.6k | −12.1k | −29.2k |
| prem30_gate (C) | **−16.5k** | +28.1k | **−18.7k** | −29.2k |
| **move0.4 (live)** | **+30.8k** | +28.1k | **+18.5k** | −19.7k |

SmartGate trades 916 live only **Mon+Tue**, so those two columns decide it: **Mon ~tie, Tue move0.4 wins
massively** (+18.5k vs −12/−19k).

## Verdict — KEEP move0.4 on ATM2 (no change)
1. **The strike-gate (C) barely changes anything** — identical to plain 30% SL on Wed/Thu/Fri, worse on
   Tue. A 30% premium rise needs a ~70+ pt move, which is **already > one strike (50)**, so by the time
   the SL fires the ATM has almost always already shifted → the gate rarely blocks. Where it binds
   (expiry) it holds a losing straddle into gamma and bleeds more.
2. **The same-strike churn is already solved by move0.4** — it re-centers only when `new_atm != cur_k`,
   so it structurally cannot close+reopen the same strike (the exact problem the move-stop change fixed).
3. **move0.4 is simply the best mechanic for the 9:16 entry** (+30.8k vs −8.4k/−16.5k), driven by
   Tuesday/expiry — consistent with research/74 (expiry theta collapse; move-stop re-centers to ride it,
   premium SL whipsaws on the 15:15 IV pop).

## Caveats
- **Squeeze ATM2 disagrees**: there plain premium30 is best overall (+17.4k vs move +12.6k, gate +10.4k)
  and move0.4 loses its Tuesday — but that's the later squeeze entry, NOT what's arming now.
- Per-weekday n small (10–12); backtest may under-model rapid live premium whipsaw — another reason to
  prefer the structurally churn-free move-stop.

Data: `results/atm2_trigger_revalidation.csv`. Runner: `scripts/revalidate_atm2_trigger.py`.
