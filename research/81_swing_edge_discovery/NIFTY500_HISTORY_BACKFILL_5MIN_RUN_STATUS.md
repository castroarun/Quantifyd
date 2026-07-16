# Nifty-500 5-Min History Backfill — 2015→2024 for ~370 Names (Kite, VPS)

STATUS: WAITING_AUTH

Part of **research/81 Swing Edge Discovery** (`EDGE_DISCOVERY_81_STUDY_STATE.md` at repo root).
Full pre-registration (Ask/Base/Plan) is preserved in git history of this file;
this copy is rewritten live by `scripts/backfill_5min_history.py`.

## Status (live)

- **State:** WAITING_AUTH
- **Progress:** 184 done · 1 skipped (already deep) · 4 failed · of 381 symbols
- **Current symbol:** IFBIND
- **Elapsed:** 767 min · **crude ETA:** 13.1 h remaining
- **Last update:** 2026-07-16T07:32:56 IST

## Event log (last 30)

| Time | Event |
|---|---|
| 2026-07-16 03:11 IST | [160/381] GRASIM done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 03:14 IST | [161/381] GRINDWELL done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 03:19 IST | [162/381] GSFC done in 4.4m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 03:20 IST | [163/381] GSPL: no earlier history at Kite (listed later?) |
| 2026-07-16 03:23 IST | [164/381] GUJALKALI done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 03:24 IST | [165/381] GUJGASLTD: no earlier history at Kite (listed later?) |
| 2026-07-16 03:27 IST | [166/381] GULFOILLUB done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 03:30 IST | [167/381] HAL done in 2.9m — earliest 2024-03-18 → 2018-04-02 |
| 2026-07-16 03:33 IST | [168/381] HATSUN done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 03:36 IST | [169/381] HAVELLS done in 3.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 03:40 IST | [170/381] HCLTECH done in 3.3m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 03:43 IST | [171/381] HDFCAMC done in 3.1m — earliest 2024-03-18 → 2018-08-06 |
| 2026-07-16 03:46 IST | [172/381] HDFCLIFE done in 3.4m — earliest 2024-03-18 → 2015-02-11 |
| 2026-07-16 03:50 IST | [173/381] HEG done in 3.6m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 03:53 IST | [174/381] HEIDELBERG done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 03:56 IST | [175/381] HERITGFOOD done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 03:59 IST | [176/381] HEROMOTOCO done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 04:02 IST | [177/381] HFCL done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 04:05 IST | [178/381] HINDALCO done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 04:08 IST | [179/381] HINDCOPPER done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 04:11 IST | [180/381] HINDPETRO done in 3.1m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 04:14 IST | [181/381] HINDZINC done in 3.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 04:19 IST | [182/381] HONAUT done in 4.5m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 04:22 IST | [183/381] HUDCO done in 3.1m — earliest 2024-03-18 → 2017-05-19 |
| 2026-07-16 04:25 IST | [184/381] ICICIGI done in 3.1m — earliest 2024-03-18 → 2017-09-27 |
| 2026-07-16 04:28 IST | [185/381] ICICIPRULI done in 3.1m — earliest 2024-03-18 → 2016-09-29 |
| 2026-07-16 04:30 IST | [186/381] IDBI UNVERIFIED (token died mid-run) — will retry |
| 2026-07-16 06:06 IST | [187/381] IDEA done in 6.0m — earliest 2024-03-18 → 2015-02-02 |
| 2026-07-16 06:09 IST | [188/381] IDFCFIRSTB done in 3.2m — earliest 2024-03-18 → 2015-11-06 |
| 2026-07-16 06:12 IST | [189/381] IEX UNVERIFIED (token died mid-run) — will retry |

## Crash recovery

Resume (idempotent — skips checkpointed symbols):
```
cd /home/arun/quantifyd
nohup bash -c 'venv/bin/python3 research/81_swing_edge_discovery/scripts/backfill_5min_history.py && venv/bin/python3 scripts/backfill_market_data_vps.py --timeframe 5minute' > /tmp/backfill_5min_history.log 2>&1 &
```
Checkpoint: `research/81_swing_edge_discovery/results/backfill_checkpoint.json`
Log: `/tmp/backfill_5min_history.log` · If stuck WAITING_AUTH after 09:00 IST:
`curl -X POST http://127.0.0.1:5000/api/auth/auto-login`
