# Nifty-500 5-Min History Backfill — 2015→2024 for ~370 Names (Kite, VPS)

STATUS: DONE_WITH_ERRORS

Part of **research/81 Swing Edge Discovery** (`EDGE_DISCOVERY_81_STUDY_STATE.md` at repo root).
Full pre-registration (Ask/Base/Plan) is preserved in git history of this file;
this copy is rewritten live by `scripts/backfill_5min_history.py`.

## Status (live)

- **State:** DONE_WITH_ERRORS
- **Progress:** 119 done · 262 skipped (already deep) · 1 failed · of 382 symbols
- **Current symbol:** ZYDUSWELL
- **Elapsed:** 534 min · **crude ETA:** 0.0 h remaining
- **Last update:** 2026-07-16T22:01:16 IST

## Event log (last 30)

| Time | Event |
|---|---|
| 2026-07-16 19:51 IST | [354/382] TCIEXP done in 4.4m — earliest 2024-03-18 → 2016-12-15 |
| 2026-07-16 19:57 IST | [355/382] THERMAX done in 5.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:02 IST | [356/382] THYROCARE done in 4.9m — earliest 2024-03-18 → 2016-05-10 |
| 2026-07-16 20:05 IST | [357/382] TIINDIA done in 3.6m — earliest 2024-03-18 → 2017-11-02 |
| 2026-07-16 20:09 IST | [358/382] TIMKEN done in 4.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:14 IST | [359/382] TORNTPHARM done in 4.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:19 IST | [360/382] TORNTPOWER done in 5.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:23 IST | [361/382] TRIDENT done in 3.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:27 IST | [362/382] TRITURBINE done in 3.8m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:31 IST | [363/382] TTKPRESTIG done in 4.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:35 IST | [364/382] TVSMOTOR done in 4.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:40 IST | [365/382] UBL done in 4.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:44 IST | [366/382] UCOBANK done in 4.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:49 IST | [367/382] UNIONBANK done in 4.9m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:54 IST | [368/382] UPL done in 4.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 20:58 IST | [369/382] VBL done in 4.2m — earliest 2024-03-18 → 2016-11-08 |
| 2026-07-16 21:02 IST | [370/382] VGUARD done in 4.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 21:07 IST | [371/382] VINATIORGA done in 4.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 21:12 IST | [372/382] VIPIND done in 4.7m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 21:16 IST | [373/382] VMART done in 4.2m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 21:22 IST | [374/382] VRLLOG done in 6.4m — earliest 2024-03-18 → 2015-04-30 |
| 2026-07-16 21:28 IST | [375/382] VSTIND done in 5.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 21:32 IST | [376/382] WELCORP done in 4.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 21:37 IST | [377/382] WHIRLPOOL done in 5.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 21:42 IST | [378/382] WOCKPHARMA done in 4.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 21:47 IST | [379/382] YESBANK done in 5.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 21:52 IST | [380/382] ZEEL done in 4.5m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 21:56 IST | [381/382] ZENSARTECH done in 4.5m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 22:01 IST | [382/382] ZYDUSWELL done in 4.8m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 22:01 IST | Run complete — done=119 skipped=262 failed=1 |

## Crash recovery

Resume (idempotent — skips checkpointed symbols):
```
cd /home/arun/quantifyd
nohup bash -c 'venv/bin/python3 research/81_swing_edge_discovery/scripts/backfill_5min_history.py && venv/bin/python3 scripts/backfill_market_data_vps.py --timeframe 5minute' > /tmp/backfill_5min_history.log 2>&1 &
```
Checkpoint: `research/81_swing_edge_discovery/results/backfill_checkpoint.json`
Log: `/tmp/backfill_5min_history.log` · If stuck WAITING_AUTH after 09:00 IST:
`curl -X POST http://127.0.0.1:5000/api/auth/auto-login`
