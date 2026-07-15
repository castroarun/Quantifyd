# Nifty-500 5-Min History Backfill — 2015→2024 for ~370 Names (Kite, VPS)

STATUS: RUNNING

Part of **research/81 Swing Edge Discovery** (`EDGE_DISCOVERY_81_STUDY_STATE.md` at repo root).
Full pre-registration (Ask/Base/Plan) is preserved in git history of this file;
this copy is rewritten live by `scripts/backfill_5min_history.py`.

## Status (live)

- **State:** RUNNING
- **Progress:** 24 done · 1 skipped (already deep) · 1 failed · of 381 symbols
- **Current symbol:** AKZOINDIA
- **Elapsed:** 59 min · **crude ETA:** 14.1 h remaining
- **Last update:** 2026-07-15T19:44:56 IST

## Event log (last 30)

| Time | Event |
|---|---|
| 2026-07-15 18:45 IST | Run start — 381 symbols, target from 2015-02-01 |
| 2026-07-15 18:45 IST | [1/381] BANKNIFTY FAILED: BANKNIFTY: Instrument token not found for BANKNIFTY |
| 2026-07-15 18:46 IST | [3/381] ICICIBANK done in 1.0m — earliest 2018-01-01 → 2015-02-02 |
| 2026-07-15 18:47 IST | [4/381] RELIANCE done in 1.0m — earliest 2018-01-01 → 2015-02-02 |
| 2026-07-15 18:48 IST | [5/381] INFY done in 0.9m — earliest 2018-01-01 → 2015-02-02 |
| 2026-07-15 18:49 IST | [6/381] TCS done in 0.9m — earliest 2018-01-01 → 2015-02-02 |
| 2026-07-15 18:50 IST | [7/381] SBIN done in 0.9m — earliest 2018-01-01 → 2015-02-02 |
| 2026-07-15 18:51 IST | [8/381] ITC done in 1.0m — earliest 2018-01-01 → 2015-02-02 |
| 2026-07-15 18:52 IST | [9/381] HINDUNILVR done in 1.0m — earliest 2018-01-01 → 2015-02-02 |
| 2026-07-15 18:53 IST | [10/381] KOTAKBANK done in 1.0m — earliest 2018-01-01 → 2015-02-02 |
| 2026-07-15 18:54 IST | [11/381] BHARTIARTL done in 0.9m — earliest 2018-01-01 → 2015-02-02 |
| 2026-07-15 18:57 IST | [12/381] 3MINDIA done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 19:00 IST | [13/381] AARTIIND done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 19:03 IST | [14/381] AAVAS done in 2.9m — earliest 2024-03-18 → 2018-10-08 |
| 2026-07-15 19:06 IST | [15/381] ABBOTINDIA done in 3.2m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 19:09 IST | [16/381] ABCAPITAL done in 3.3m — earliest 2024-03-18 → 2017-09-04 |
| 2026-07-15 19:13 IST | [17/381] ABFRL done in 3.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 19:16 IST | [18/381] ACC done in 3.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 19:21 IST | [19/381] ADANIENT done in 4.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 19:24 IST | [20/381] ADANIGREEN done in 3.0m — earliest 2024-03-18 → 2018-06-18 |
| 2026-07-15 19:27 IST | [21/381] ADANIPORTS done in 3.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 19:31 IST | [22/381] ADANIPOWER done in 3.7m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 19:34 IST | [23/381] ADVENZYMES done in 3.3m — earliest 2024-03-18 → 2016-08-01 |
| 2026-07-15 19:37 IST | [24/381] AFFLE done in 3.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 19:41 IST | [25/381] AIAENG done in 3.5m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-15 19:44 IST | [26/381] AJANTPHARM done in 3.4m — earliest 2024-03-18 → 2015-02-02 |

## Crash recovery

Resume (idempotent — skips checkpointed symbols):
```
cd /home/arun/quantifyd
nohup bash -c 'venv/bin/python3 research/81_swing_edge_discovery/scripts/backfill_5min_history.py && venv/bin/python3 scripts/backfill_market_data_vps.py --timeframe 5minute' > /tmp/backfill_5min_history.log 2>&1 &
```
Checkpoint: `research/81_swing_edge_discovery/results/backfill_checkpoint.json`
Log: `/tmp/backfill_5min_history.log` · If stuck WAITING_AUTH after 09:00 IST:
`curl -X POST http://127.0.0.1:5000/api/auth/auto-login`
