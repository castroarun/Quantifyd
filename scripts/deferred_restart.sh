#!/bin/bash
# Deferred quantifyd restart — NEVER before 15:40 IST on a trading day.
#
# Standing rule (Arun, 2026-08-17): all restarts happen post-15:40, which
# leaves the 15:15-15:35 EOD jobs (squareoffs, summaries, snapshots) room to
# finish before the service bounces.
#
# Usage: deferred_restart.sh [reason]
exec > /tmp/deferred_restart.log 2>&1
REASON="${1:-scheduled deploy}"
echo "armed at $(date) — reason: $REASON"

CUTOFF=1540
now=$(date +%s)
target=$(date -d '15:40' +%s)
if [ "$target" -gt "$now" ]; then
    sleep $((target - now))
fi

hm=$(date +%H%M)
dow=$(date +%u)
echo "woke at $(date) hm=$hm dow=$dow"

# Standalone gate, evaluated at execution time — not trusting the sleep.
if [ "$dow" -le 5 ] && [ "$hm" -lt "$CUTOFF" ]; then
    echo "ABORT: before the ${CUTOFF} IST cutoff on a trading day"
    exit 1
fi

open_before=$(/home/arun/quantifyd/venv/bin/python3 - <<'PY'
import sqlite3
c = sqlite3.connect('/home/arun/quantifyd/backtest_data/strangle_trading.db')
print(c.execute("SELECT COUNT(*) FROM strangle_positions WHERE status='OPEN'").fetchone()[0])
PY
)
echo "open strangle positions before restart: $open_before"

sudo systemctl restart quantifyd
sleep 25
systemctl is-active quantifyd
curl -s localhost:5000/api/nas/state -o /dev/null -w 'nas api: %{http_code}\n'

/home/arun/quantifyd/venv/bin/python3 - <<'PY'
import sqlite3
c = sqlite3.connect('/home/arun/quantifyd/backtest_data/strangle_trading.db')
print('variants still enabled:',
      c.execute('SELECT COUNT(*) FROM strangle_variants WHERE enabled=1').fetchone()[0])
print('open positions:',
      c.execute("SELECT COUNT(*) FROM strangle_positions WHERE status='OPEN'").fetchone()[0])
PY

journalctl -u quantifyd -q --since '-1min' | grep -i 'strangle' | head -3
echo 'DEFERRED RESTART DONE'
