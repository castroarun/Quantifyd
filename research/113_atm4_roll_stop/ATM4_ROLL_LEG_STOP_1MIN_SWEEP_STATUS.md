# ATM4 Roll-Leg Stop Calibration — 9 Stop Rules on 83 Days of Real 1-min NIFTY Chain

STATUS: DONE (2026-08-18)

## 2. The Ask

**What you asked (2026-08-18, after watching the live book):** "the adjustment PE got
stopped out immediately... 15 to 11 is 30% but it's too small in absolute, will likely
get triggered more often than not, we have to work on this, pls assess based on our past
options data on what best can be done."

**What we're actually testing:** In the live NAS-ATM4 (roll-to-match) system, when the
first leg SLs, the stopped side is re-sold at a premium-matched OTM strike and that
rolled leg gets `SL = roll_premium x 1.3`. Rolled premiums are small (today: 12.1 → SL
15.7, only ~3.6 pts of room), so the stop sits inside expiry-day noise. Across all 83
recorded days of real 1-minute NIFTY option-chain data: **which roll-leg stop rule
maximises net P&L without blowing up the tail?** Is the fast re-stop a real cost, and
what beats it — a wider %, an absolute floor, parity with the survivor's stop, no stop,
a min-premium gate on rolling, or not rolling at all?

## 3. The Base — what's being tested

Faithful replay of the live ATM4 mechanic (from `services/nas_atm4_executor.py`):

- **Entry:** 09:16 snapshot. ATM strike = round(spot/50)x50, SELL CE+PE (front weekly
  expiry) at snapshot LTP. Per-leg SL = 1.3 x own entry premium.
- **First SL:** close stopped leg at its (1-min) breach print. price_x = survivor's LTP
  that minute. Roll: scan OTM strikes (>= 50 pts OTM, up to 15 steps of 50) for premium
  closest to price_x; SELL it. Survivor SL := 1.3 x price_x. Rolled-leg SL := **the
  variant under test**.
- **Second SL (either leg):** close stopped leg; remaining leg's SL := price_x (flat).
  Third breach closes it. No further rolls.
- **EOD:** 15:15 square-off of everything still open.
- **Universe/period:** NIFTY weekly options, 2026-04-20 → 2026-08-14 (83 trading days
  of full-chain 1-min snapshots, `options_data.db::option_chain`, real LTPs).
- **Size:** per 1 lot (65). Live runs 2 lots (x130) — scale linearly.
- **Costs (net):** slippage 0.5 pt per leg-side + Rs30 charges per leg-side per lot.
  Variants trade different leg counts, so costs scale with orders. Gross also reported.
- **Success criterion:** total net P&L/lot across all days, sanity-checked against
  median/day, p05 (tail), % of rolled legs re-stopped, and per-DTE stability. A winner
  must not win by fattening the left tail.

**Known caveats (stated up front):** 1-min data understates intra-minute SL touches
(both for the status quo AND the alternatives — comparison remains fair); single
overlapping 4-month window, no OOS; ST-trail on the post-second-SL naked survivor is
NOT modeled (flat price_x stop per the coded CASE 2) — identical across variants.

## 4. Plan — variant grid

One axis: the ROLLED leg's stop rule (everything else frozen):

| # | Variant | Rolled-leg stop |
|---|---------|-----------------|
| 1 | SQ (status quo) | 1.3 x roll_prem |
| 2 | P150 | 1.5 x roll_prem |
| 3 | P200 | 2.0 x roll_prem |
| 4 | F8 | roll_prem + max(0.3 x roll_prem, 8 pts) |
| 5 | F12 | roll_prem + max(0.3 x roll_prem, 12 pts) |
| 6 | SURV | 1.3 x price_x (absolute parity with survivor stop) |
| 7 | NOSL | no premium stop on the rolled leg (EOD only; survivor rules unchanged) |
| 8 | MIN15 | roll only if price_x >= 15, else no roll (survivor tightened 1.3 x price_x) |
| 9 | NOROLL | never roll: first SL closes the leg, survivor SL := 1.3 x price_x |

9 variants x 83 days = 747 day-replays (in-memory per day; minutes of compute).

## 5. Status (live log)

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-18 ~11:3x IST | Recon done | 83 NIFTY days, ~376 snaps/day (1-min), ltp+bid+ask present |
| 2026-08-18 ~11:4x IST | STATUS written, runner launched on VPS | results/ incremental CSV |

## 6. Crash Recovery

- Runner: `research/113_atm4_roll_stop/scripts/run_atm4_roll_sweep.py` (self-contained,
  reads `backtest_data/options_data.db` read-only).
- Check progress: `tail -20 research/113_atm4_roll_stop/results/run.log` and
  `wc -l research/113_atm4_roll_stop/results/atm4_roll_daily.csv` (1 row per variant-day).
- Alive? `pgrep -af run_atm4_roll_sweep`.
- Resume: just re-run the script — it skips (variant, day) pairs already in the CSV.
  `cd /home/arun/quantifyd && nohup python3 research/113_atm4_roll_stop/scripts/run_atm4_roll_sweep.py >> /tmp/atm4_roll.log 2>&1 &`
- Aggregate-only (if daily CSV complete): re-run with `--aggregate-only`.
- Safe to inspect anytime: everything under `results/`. Do not touch `options_data.db`.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `ATM4_ROLL_LEG_STOP_1MIN_SWEEP_STATUS.md` | this file | yes |
| `scripts/run_atm4_roll_sweep.py` | replay runner | yes |
| `results/atm4_roll_daily.csv` | per variant-day detail | yes (small) |
| `results/atm4_roll_summary.csv` | per-variant aggregate + per-DTE | yes |
| `results/run.log` | progress log | yes |
| `results/RESULTS.md` | verdict | yes |

## 8. Findings

**Verdict: SIGNAL (mechanic upgrade) — see results/RESULTS.md.** Live rule (1.3x roll_prem) = churniest variant (32% restop). Rolling itself strongly validated (NOROLL -49k vs +143k). DEPLOYED: MAXV — rolled-leg SL = max(price_x, roll_prem) x 1.3 (Arun refinement; +5.5k over SURV, same best tail, 19% restop) — SURV strictly dominates live rule on total/tail/win/churn and is best on DTE0. P200 = max-total alternative with fatter tail. 81 days, 1 window, no OOS.
