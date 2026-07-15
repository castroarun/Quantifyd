# Nifty-500 5-Min History Backfill — 2015→2024 for ~370 Names (Kite, VPS)

STATUS: RUNNING

Part of **research/81 Swing Edge Discovery** (`EDGE_DISCOVERY_81_STUDY_STATE.md` at repo root).
Full pre-registration (Ask/Base/Plan) is preserved in git history of this file;
this copy is rewritten live by `scripts/backfill_5min_history.py`.

## Status (live)

- **State:** RUNNING
- **Progress:** 69 done · 1 skipped (already deep) · 2 failed · of 381 symbols
- **Current symbol:** BRIGADE
- **Elapsed:** 212 min · **crude ETA:** 15.4 h remaining
- **Last update:** 2026-07-15T22:17:31 IST

## Event log (last 30)

| Time | Event |
|---|---|
| 2026-07-15 20:43 IST | [43/381] AUBANK done in 3.0m — earliest 2024-03-18 → 2017-07-10 |
| 2026-07-15 20:47 IST | [44/381] AUROPHARMA done in 3.5m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 20:50 IST | [45/381] AXISBANK done in 3.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 20:53 IST | [46/381] BAJAJ-AUTO done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 20:57 IST | [47/381] BAJAJCON done in 3.2m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:00 IST | [48/381] BAJAJELEC done in 3.2m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:03 IST | [49/381] BAJAJFINSV done in 3.2m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:06 IST | [50/381] BAJAJHLDNG done in 3.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:09 IST | [51/381] BAJFINANCE done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:12 IST | [52/381] BALKRISIND done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:15 IST | [53/381] BALMLAWRIE done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:19 IST | [54/381] BALRAMCHIN done in 3.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:22 IST | [55/381] BANDHANBNK done in 2.9m — earliest 2024-03-18 → 2018-03-27 |
| 2026-07-15 21:25 IST | [56/381] BANKBARODA done in 3.2m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:28 IST | [57/381] BANKINDIA done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:31 IST | [58/381] BASF done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:34 IST | [59/381] BATAINDIA done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:37 IST | [60/381] BAYERCROP done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:40 IST | [61/381] BDL done in 3.0m — earliest 2024-03-18 → 2018-03-23 |
| 2026-07-15 21:43 IST | [62/381] BEL done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:47 IST | [63/381] BEML done in 3.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:50 IST | [64/381] BERGEPAINT done in 3.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:53 IST | [65/381] BHARATFORG done in 3.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 21:56 IST | [66/381] BHEL done in 3.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 22:00 IST | [67/381] BIOCON done in 3.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 22:03 IST | [68/381] BIRLACORPN done in 3.2m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 22:06 IST | [69/381] BLUEDART FAILED: BLUEDART: Execution failed |
| 2026-07-15 22:10 IST | [70/381] BLUESTARCO done in 3.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 22:13 IST | [71/381] BOSCHLTD done in 3.7m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 22:17 IST | [72/381] BPCL done in 3.8m — earliest 2024-03-18 → 2015-02-02 |

## Crash recovery

Resume (idempotent — skips checkpointed symbols):
```
cd /home/arun/quantifyd
nohup bash -c 'venv/bin/python3 research/81_swing_edge_discovery/scripts/backfill_5min_history.py && venv/bin/python3 scripts/backfill_market_data_vps.py --timeframe 5minute' > /tmp/backfill_5min_history.log 2>&1 &
```
Checkpoint: `research/81_swing_edge_discovery/results/backfill_checkpoint.json`
Log: `/tmp/backfill_5min_history.log` · If stuck WAITING_AUTH after 09:00 IST:
`curl -X POST http://127.0.0.1:5000/api/auth/auto-login`
