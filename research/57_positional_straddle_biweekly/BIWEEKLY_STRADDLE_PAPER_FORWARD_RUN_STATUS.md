# Bi-Weekly Short Straddle — FORWARD PAPER RUN (research/57)

STATUS: LIVE (paper) since 2026-06-04. First paper entry: next trading day 09:20.

## What is running
A standalone **forward paper logger** validating the research/57 recipe out-of-sample. PAPER ONLY,
1 lot, **no Kite orders, no gunicorn involvement** (pure cron → no service restart ever needed).
Reads the live options recorder (`backtest_data/options_data.db` latest snapshot) so paper == backtest.

- **Script:** `scripts/biweekly_paper.py`
- **DB:** `results/paper_straddle.db` (tables: `positions`, `actions`)
- **Cron:** `*/5 9-15 * * 1-5` → runs every 5 min, Mon-Fri, 09:00-15:55 IST.
- **Log:** `/tmp/biweekly_paper.log`

## The recipe being logged (locked SIGNAL from G0-G4)
1. **09:20** (if flat): SELL the **at-the-money straddle** in the **2nd-nearest weekly expiry**
   (bi-weekly, ~8-12 DTE). Record strikes, premiums, credit, entry spot.
2. **Every ~5 min:** CRASH stop — if NIFTY has moved **>=2.0%** from entry → exit immediately.
3. **15:20 (EOD):** exit if NIFTY moved **≥1.5%** (move-stop) OR profit **≥40% of credit** (profit-
   target) OR expiry **DTE ≤ 1** (roll). Else HOLD overnight.
4. After a close, **re-enter at the next 09:20** (one straddle at a time, sequential book).

## v2 (LIVE 2026-06-04): immediate 15:20 re-entry + overnight +/-500pt wings (tracked separately).
## (historical v1 note — add after first clean weeks)
- **Overnight far-OTM wings** (the gap-disaster insurance) — buy at 15:20, sell 09:20. The 5-min
  crash poll canNOT catch an overnight gap; only wings can. Track separately when added.
- **Immediate-CMP re-entry** on an intraday crash exit (v1 waits to next 09:20).

## Why this exists
The 30-day backtest is a SIGNAL (calm single regime: +7.8k/trade, but the crash-stop never fired
and the tail/gap behaviour is untested). Forward paper accumulates REAL out-of-sample trades across
real Wed/Thu moves, gaps, and expiries — the only way to earn the SIGNAL→STRATEGY upgrade. Do NOT
size to real money on the in-sample number; let this run ≥50-100 trades / ≥2 regimes first.

## How to monitor (no me needed)
```
# latest actions + open position + running P&L
cd /home/arun/quantifyd && ./venv/bin/python3 -c "
import sqlite3; c=sqlite3.connect('research/57_positional_straddle_biweekly/results/paper_straddle.db')
print('open:', [dict(zip([d[0] for d in c.execute('PRAGMA table_info(positions)')],r)) for r in c.execute(\"SELECT * FROM positions WHERE status='OPEN'\")])
print('closed total pnl:', c.execute(\"SELECT COALESCE(SUM(pnl),0),COUNT(*) FROM positions WHERE status='CLOSED'\").fetchone())
for r in c.execute('SELECT ts,action,detail FROM actions ORDER BY rowid DESC LIMIT 8'): print(' ',*r)"
tail -30 /tmp/biweekly_paper.log
```

## Crash recovery / resume
- The cron is self-healing — each 5-min tick re-reads state from the DB and the live snapshot; a
  missed tick just resumes next tick. No in-memory state to lose.
- To pause: `crontab -e` and comment the `biweekly_paper.py` line. To resume: uncomment.
- The DB is the sole source of truth (positions + actions). Safe to inspect any time (read-only use).
- If the recorder (`options_data.db`) is stale, the logger logs `NO-SNAP` and does nothing (safe).

## Status log
| Date | Event |
|---|---|
| 2026-06-04 | Logger built + cron installed; first paper entry next 09:20. Recipe = G0-G4 SIGNAL. |
| 2026-06-04 | G5: crash-stop set to 2.0%% (1.75%% whipsaws +36k vs +74k; 2%%=safe). Intraday move-from-entry median 2.12%%, max 4.32%%. Wings (v2) still pending - user deciding distance. |

| 2026-06-04 | v2 LIVE: (a) re-enter IMMEDIATELY at 15:20 (stay short overnight, +Rs5.9k vs next-0920 but deeper DD), (b) OVERNIGHT WINGS added (+/-500pt ~2%% OTM, buy 15:20/sell 09:20, P&L tracked separately) to cap the overnight gap that staying-short exposes. Combines both user instincts. crash=2.0%%. Tested full wing cycle OK. |