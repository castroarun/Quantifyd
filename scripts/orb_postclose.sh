#!/bin/bash
# ORB paper-SL fix: deploy after the close, then purge today's phantom rows.
# Never before 15:40 IST — re-checks the clock on wake (a sleep is not a gate).
exec >> /home/arun/quantifyd/logs/orb_postclose_20260820.log 2>&1
echo "=== armed at $(TZ=Asia/Kolkata date) ==="

target=$(date -d '15:40' +%s); now=$(date +%s)
[ "$target" -gt "$now" ] && sleep $((target - now))

hm=$(TZ=Asia/Kolkata date +%H%M)
dow=$(TZ=Asia/Kolkata date +%u)
if [ "$dow" -le 5 ] && [ "$hm" -lt 1540 ]; then
    echo "ABORT: woke at $hm on a weekday, before 15:40"; exit 1
fi
echo "--- $(TZ=Asia/Kolkata date): restarting quantifyd (ORB paper-SL fix) ---"
/usr/bin/sudo /usr/bin/systemctl restart quantifyd || { echo "restart FAILED"; exit 1; }
sleep 20
echo "service: $(/usr/bin/systemctl is-active quantifyd)"

cd /home/arun/quantifyd || exit 1
echo "--- purging today's phantom rows ---"
./venv/bin/python3 scripts/orb_purge_phantom.py
echo "--- ORB state after ---"
curl -s http://127.0.0.1:5000/api/orb/state | ./venv/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print('today_pnl', d.get('today_pnl'), '| closed today', len(d.get('today_closed') or []), '| open', len(d.get('open_positions') or []))"
echo "=== done $(TZ=Asia/Kolkata date) ==="
