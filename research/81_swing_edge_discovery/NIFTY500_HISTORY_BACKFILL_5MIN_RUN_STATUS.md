# Nifty-500 5-Min History Backfill — 2015→2024 for ~370 Names (Kite, VPS)

STATUS: RUNNING

Part of **research/81 Swing Edge Discovery** (`EDGE_DISCOVERY_81_STUDY_STATE.md` at repo root).
Full pre-registration (Ask/Base/Plan) is preserved in git history of this file;
this copy is rewritten live by `scripts/backfill_5min_history.py`.

## Status (live)

- **State:** RUNNING
- **Progress:** 80 done · 262 skipped (already deep) · 1 failed · of 382 symbols
- **Current symbol:** STAR
- **Elapsed:** 352 min · **crude ETA:** 2.8 h remaining
- **Last update:** 2026-07-16T18:58:39 IST

## Event log (last 30)

| Time | Event |
|---|---|
| 2026-07-16 16:43 IST | [314/382] PRAJIND done in 4.2m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 16:49 IST | [315/382] PRESTIGE done in 5.8m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 16:54 IST | [316/382] PTC done in 4.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 16:58 IST | [317/382] RADICO done in 4.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 17:02 IST | [318/382] RAIN done in 4.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 17:06 IST | [319/382] RALLIS done in 3.8m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 17:10 IST | [320/382] RAMCOCEM done in 4.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 17:14 IST | [321/382] RATNAMANI done in 4.5m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 17:21 IST | [322/382] RAYMOND done in 6.9m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 17:26 IST | [323/382] RBLBANK done in 4.5m — earliest 2024-03-18 → 2016-08-31 |
| 2026-07-16 17:31 IST | [324/382] RCF done in 5.2m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 17:36 IST | [325/382] RECLTD done in 4.5m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 17:40 IST | [326/382] REDINGTON done in 4.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 17:45 IST | [327/382] RELAXO done in 5.2m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 17:50 IST | [328/382] RITES done in 4.9m — earliest 2024-03-18 → 2018-07-03 |
| 2026-07-16 17:54 IST | [329/382] RVNL done in 3.5m — earliest 2024-03-18 → 2015-08-12 |
| 2026-07-16 17:58 IST | [330/382] SAIL done in 4.7m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 18:04 IST | [331/382] SANOFI done in 5.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 18:09 IST | [332/382] SCHAEFFLER done in 5.2m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 18:09 IST | [333/382] SENSEX FAILED: SENSEX: Instrument token not found for SENSEX |
| 2026-07-16 18:14 IST | [334/382] SFL done in 4.9m — earliest 2024-03-18 → 2016-12-09 |
| 2026-07-16 18:20 IST | [335/382] SHK done in 6.0m — earliest 2024-03-18 → 2015-11-16 |
| 2026-07-16 18:24 IST | [336/382] SIS done in 4.0m — earliest 2024-03-18 → 2017-08-10 |
| 2026-07-16 18:29 IST | [337/382] SJVN done in 5.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 18:34 IST | [338/382] SKFINDIA done in 5.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 18:39 IST | [339/382] SOBHA done in 4.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 18:44 IST | [340/382] SOLARINDS done in 5.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 18:50 IST | [341/382] SONATSOFTW done in 5.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 18:54 IST | [342/382] SOUTHBANK done in 4.9m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 18:58 IST | [343/382] SRF done in 3.7m — earliest 2024-03-18 → 2015-02-02 |

## Crash recovery

Resume (idempotent — skips checkpointed symbols):
```
cd /home/arun/quantifyd
nohup bash -c 'venv/bin/python3 research/81_swing_edge_discovery/scripts/backfill_5min_history.py && venv/bin/python3 scripts/backfill_market_data_vps.py --timeframe 5minute' > /tmp/backfill_5min_history.log 2>&1 &
```
Checkpoint: `research/81_swing_edge_discovery/results/backfill_checkpoint.json`
Log: `/tmp/backfill_5min_history.log` · If stuck WAITING_AUTH after 09:00 IST:
`curl -X POST http://127.0.0.1:5000/api/auth/auto-login`
