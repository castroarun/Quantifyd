# SENSEX Portfolio Bracket — Own Per-Lot Stop Calibration, 1-min Replay over Recorded SENSEX Chain

STATUS: DONE

## 2. The Ask
The deployed NAS portfolio stop uses -Rs1300/lot, calibrated on NIFTY (research/90). SENSEX has only
~2 live days, so its threshold is provisional. Run the identical 64-day, 1-min faithful replay on the
recorded **SENSEX** chain to give the SENSEX 3-system book its OWN per-lot stop, and confirm the
"no take-profit / no trail" finding holds for SENSEX too.

## 3. The Base
- Systems (2 lots each, QTY 40): SENSEX ATM (per-leg 1.3x SL + ST(7,2) survivor trail), SENSEX ATM2
  (0.4% underlying move-stop, one-and-done, re-center), SENSEX ATM4 (roll-to-match once). Same rules
  as the live SENSEX executors.
- Engine: research/90 MTM engine adapted for SENSEX (symbol=SENSEX, LOT=20, strike step 100). Front
  weekly expiry (Thursday). 09:16 entry, 15:15 force exit. Per-minute MTM.
- Data: options_data.db SENSEX chain, per-minute, ~64 days (2026-04-20+).
- Same optimism caveat: LTP fills, no slippage, 1-min resolution.

## 4. Plan
- TP in {none,3k,4k,5k,6k,8k,10k,12k,15k} x SL in {none,-2k,-3k,-4k,-5k,-6k,-8k,-10k,-12k}.
- Rank by total + Calmar; per-half stability; report best STOP as per-lot (/6).
- Deliverable: SENSEX per-lot stop number; update the live threshold if it differs materially from
  -Rs1300/lot.

## 5-8: see results/ (sensex_sweep.csv, RESULTS.md). Runner:
research/91_sensex_portfolio_bracket/scripts/run_sensex_bracket.py (caches results/sensex_day_paths.json).
Crash recovery: re-run the runner; cached paths make the sweep instant.
