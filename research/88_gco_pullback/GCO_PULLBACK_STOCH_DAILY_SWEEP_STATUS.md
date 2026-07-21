# GCO-Pullback-Stoch — Golden-Cross First-Pullback Entry with Stochastic Trigger (Daily, F&O Futures)

STATUS: DONE — NO EDGE (best cell loses to random-entry control; see results/RESULTS.md)

## 1. The Ask

**What you asked (verbatim):** "on a fresh golden crossover (gco), where a 20
sma crosses above 50, we wait for the 1st pullback and then a green candle
where stochastics crossover happens below its value of 40, the we go long in
futures with SL below that green candle or below the previous candles or so.
Reverse for shorts."

**What we're testing:** On daily bars, F&O universe, does entering the first
stochastic-confirmed pullback after a fresh 20/50 SMA golden cross (mirror:
death cross for shorts) produce positive NET per-trade returns — and, the
lesson of research/87 applied — does it beat a RANDOM-ENTRY baseline running
the identical exit mechanics (the drift control)?

## 2. The Base (locked)

- **Setup (long):** sma20 crosses above sma50 (fresh = within F bars). Then
  the 1st pullback: a close below sma20. Then trigger: a GREEN candle where
  stoch %K (14,3,3) crosses above %D with %K < 40. One trigger per cross.
  Setup invalidated by opposite cross or F-window expiry. Short = mirror
  (death cross, pullback above sma20, red candle, stoch cross-down, %K > 60).
- **Entry:** open of the bar after the trigger candle.
- **Stop:** below trigger-candle low (SL1) or min of trigger + prior candle
  lows (SL2). Gap-through fills at open; stop checked before target same-bar.
- **Exits tested:** E1 = 2R target + SL (30-bar cap) · E2 = trail: close
  crosses back through sma20 + SL · E3 = 15-bar time stop + SL.
- **Universe/data:** ~86 F&O names, daily, warmup 2000.
- **Splits:** IS 2005-01-01→2017-12-31 · Val 2018-01-01→2022-06-30 · OOS
  2022-07-01+ QUARANTINED.
- **Costs:** 10bps round-trip.
- **Control (pre-registered):** RANDOM-ENTRY baseline — every 10th bar entry,
  same stops (synthetic SL at same median stop-distance), same exits, same
  window. **A cell only "works" if it beats its baseline arm AND net>0 with
  t ≥ 2.5, ≥55% names positive.** Val gate: net>0, t ≥ 2.0, no sign flip.

## 3. Plan — grid (24 cells + 6 baseline arms)

| Axis | Values |
|---|---|
| Fresh window F | 10, 20 bars |
| Stop | SL1 (trigger low), SL2 (min 2 lows) |
| Exit | E1 2R, E2 sma20-trail, E3 time-15 |
| Direction | L, S |

2×2×3×2 = **24 cells** + 6 baselines (exit × dir). Multiple-testing ledger
(program r/87+r/88): 104 + 24 = 128 signal cells. Phase 2 (only if G1
passes): 60-min timeframe variant, its own splits, ≤24 cells.

## 4. Status / event log

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-21 ~20:45 IST | Pre-registration + runner authored | |
| 2026-07-21 (see log) | G1 IS launched on VPS | results/gco_g1.log |

## 5. Crash recovery

- Progress: `tail research/88_gco_pullback/results/gco_g1.log`
- Alive: `ps -eo args | grep '[r]un_gco_battery'`
- Resume: rerun `venv/bin/python3 research/88_gco_pullback/scripts/run_gco_battery.py`
  (env `R88_WIN_START/R88_WIN_END` override the window; skips done cells).

## 6. Files

| File | Purpose | Committable |
|---|---|---|
| `scripts/run_gco_battery.py` | detector + stop/target sim + baselines | yes |
| `results/gco_g1.csv` / `.log` | per-cell aggregates / log | yes |
| `results/RESULTS.md` | verdict | yes (after) |

## 7. Findings

(populated during/after run)
