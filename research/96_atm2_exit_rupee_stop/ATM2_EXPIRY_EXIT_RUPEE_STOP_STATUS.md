# ATM2 Exit Redesign — Rupee MTM Stop replaces the 0.4% Spot-Move Stop (expiry-gamma fix)

STATUS: **DEPLOYED 2026-07-28 ~21:45 IST** (post-close; service restarted; master flipped back to live for Wed)

## 1. The Ask

**What Arun said (2026-07-28, mid-session, live ATM2 open):** "im worried ab the exit points
for ATM 2 on an expiry day like today coz the premiums are too low and at that stated exit, the
premium wud hv shot up a lot leading to losses with no support from the opposite leg as its
premium was also very low in this travel."

**What we're actually fixing:** ATM2's only intraday exit is a **0.4% spot-move stop** (close both
legs when |spot − entry_spot| ≥ 0.4%). A stop keyed to *spot distance* produces a **DTE-dependent
loss**: near expiry, gamma is huge so the losing leg balloons ~1:1 with spot while the OTM leg
(already tiny) can only cushion to ~0 → the 0.4% travel crystallises an asymmetric loss with no
offset. The same 0.4% costs far less on a far-DTE day (fat premium, vega/theta cushion). Goal:
replace it with a **DTE-agnostic stop that caps the rupee loss uniformly**.

## 2. The Base (current ATM2 exit, both variants)

- Configs: `NAS_ATM2_DEFAULTS` (squeeze `nas_atm2`) + `NAS_916_ATM2_DEFAULTS` (9:16 `nas_916_atm2`).
  Both: `move_stop_pct=0.004`, `move_stop_reenter=True`, `re_enter_on_sl=False`, per-leg 30% SL.
- Exit code: `services/nas_atm2_executor.py::check_positions` (inherited by `Nas916Atm2Executor`).
  The move-stop is the SOLE trigger; the 30% per-leg SL is DISABLED whenever `move_stop_pct>0`
  (line ~163: `if (not move_stop_pct) and live_prem >= sl_price`). After a move-stop it re-centers
  to the new ATM (churn risk on trending/expiry days).
- Other 6 NAS variants (ATM, ATM4, OTM in both families): `move_stop_pct=None` — DIFFERENT exit
  logic (ATM = cascade re-enter-on-SL, ATM4 = rolls, OTM = own). **NOT in scope. Untouched.**

## 3. Calibration (evidence for the thresholds)

Runner: `scripts/run_atm2_exit_calib.py`. 68 recorded chain days (2026-04-20 → 07-28), entry =
09:16 ATM straddle, 2 lots (QTY 130), net-of-cost, **one-and-done** (stop isolated, no re-center).
Split by DTE. avg = ₹/trade.

| Exit rule | Near-expiry (DTE<=1) avg | worst | win% | Far (DTE>=2) avg | All-days avg |
|---|---|---|---|---|---|
| CURRENT 0.4% move | +1,386 | -6,887 | 62% | **-160** (bleeds) | +499 |
| RUPEE 2,000/lot | +1,594 | **-5,490** | 66% | -209 | +560 |
| **RUPEE 2,500/lot (CHOSEN)** | **+2,153** | -6,972 | 69% | -98 | +861 |
| RUPEE 3,000/lot | +1,793 | -8,681 | 69% | **+309** | **+942** |
| PER-LEG 1.3x (rec-3 idea) | +352 | -2,415 | 52% | -149 | +64 |
| PREM x1.4..2.0 | +1,140..+1,869 | -10k..-19k | 69% | — | deep tails |

**Reads:**
1. Rupee MTM stop beats the 0.4% move-stop in every DTE bucket and is DTE-agnostic by construction.
   ₹2,500/lot near-expiry: +2,153/tr vs +1,386 (+55%), tail ≈ identical (-6,972 vs -6,887), win 69/62.
2. Premium-MULTIPLE stops are WORSE for the expiry worry — on a low-premium day "2x premium"
   triggers late in rupee terms → -16k to -31k tails. Flat **rupee** cap is the right primitive.
3. The 30% per-leg SL (rec-3) is TOO TIGHT to be a backstop — it pre-empts the rupee stop on
   almost every dip and collapses avg to +64/tr. **DROP it** (the rupee stop is the cap).
4. Re-center adds churn on trending/expiry days → **one-and-done**.

## 4. Decision (Arun approved 2026-07-28)

**ATM2 only, both variants (`nas_atm2` + `nas_916_atm2`):**
- Primary exit = **rupee MTM stop, ₹2,500 / lot** (₹5,000 per 2-lot strangle, pre-slippage).
- **Drop** the 30% per-leg SL.
- **One-and-done** — no move-stop re-center.

## 5. Exact code changes (apply at deploy)

a) `config.py` — `NAS_ATM2_DEFAULTS` AND `NAS_916_ATM2_DEFAULTS`:
```
'rupee_stop_per_lot': 2500,   # 2026-07-28: DTE-agnostic MTM stop (research/96) — replaces move-stop
'move_stop_pct': 0,           # was 0.004 — disable the spot-move stop
'move_stop_reenter': False,   # one-and-done, no re-center
```
b) `services/nas_atm2_executor.py::check_positions` — add, BEFORE the move-stop block:
   - if `cfg.get('rupee_stop_per_lot')`: for each ACTIVE strangle, sum live-leg P&L
     `= sum((entry_price - live_prem) * qty)`; lots = qty/lot_size; if
     `total_pnl <= -(rupee_stop_per_lot * lots)` → phantom-guard via `_broker_holds_any`, then
     `_close_leg` both legs (reason `RUPEE_STOP`), `_record_trade`, NO re-enter. Add sid to
     `exited_strangles`.
   - Gate the per-leg SL: change line ~163 to
     `if (not cfg.get('move_stop_pct')) and (not cfg.get('rupee_stop_per_lot')) and live_prem >= sl_price:`
     so the per-leg SL stays off when the rupee stop is active.

## 6. Deploy checklist (post-close, after 15:30 IST — NOT during market)

1. `git pull` on VPS (or edit in place), apply 5a + 5b.
2. **Flip master mode back to live** (currently persisted 'paper' from today's manual-close event):
   `curl -s -X POST http://127.0.0.1:5000/api/nas/master-mode -d '{"mode":"live"}' -H 'Content-Type: application/json'`
   OR it applies on the restart via `_restore_nas_master_mode_on_boot` if the mode file = 'live'.
   -> WRITE `backtest_data/nas_master_mode.json` = {"mode":"live"} before restart.
3. Confirm no kill flag: `ls backtest_data/nas_kill.flag` should be absent.
4. `sudo systemctl restart quantifyd` (allowed post-close).
5. Verify: GET /api/nas/master-mode == 'live'; ATM2 configs show rupee_stop_per_lot=2500,
   move_stop_pct=0; Wed day-matrix -> SENSEX live Wed/Thu, NIFTY paper Wed.

## 7. Current live state (as of 2026-07-28 ~11:00 IST)

- Broker FLAT (Arun closed live ATM2/916 manually on the expiry-gamma concern). DB reconciled.
- Master mode = **paper** (all 8 NAS variants paper, recording). Kill flag CLEARED.
- => No real money today; paper trades continue and record. MUST flip to 'live' before Wed (step 2).

## 8. Caveats / honesty

68 days only; optimistic minute-granularity fills (the rupee "worst" already shows some gap-through
beyond the threshold, so slippage is partly captured); one-and-done modeled (matches the deploy).
Rupee family is monotonic (2000/2500/3000 all sensible) — not a lucky single point. The venue-level
portfolio stop (−₹1,300/lot, research/90) still sits above this as aggregate cover; the per-strangle
rupee stop adds per-position protection for the single-straddle blowout Arun flagged.


## Deploy log (2026-07-28)

| Time | Event |
|---|---|
| 21:40 | Backups: nas_atm2_executor.py.bak_rupee96, config.py.bak_rupee96 |
| 21:42 | config.py: NAS_ATM2_DEFAULTS move_stop_pct 0.004->0, move_stop_reenter->False, +rupee_stop_per_lot=2500 (inherits to NAS_916_ATM2_DEFAULTS; verified both) |
| 21:42 | nas_atm2_executor.py: RUPEE_STOP block added (per-strangle MTM <= -2500xlots -> close both, one-and-done, manual-exit guard kept); 30% per-leg SL gated OFF when rupee stop active (sl_price kept as ticker wake trigger only) |
| 21:45 | Committed + service restarted + verified; NAS master-mode -> live for Wed SENSEX |

Cadence note: rupee stop evaluates on the same cadence as the old move-stop
(periodic check_positions + SL-tick piggyback) — not tick-level; acceptable per
calibration (the move-stop it replaces had identical latency).
