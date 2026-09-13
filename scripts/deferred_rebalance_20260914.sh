#!/bin/bash
# One-shot: rebalance the three Momentum Portfolio books to 37.5 / 37.5 / 25 on
# Monday 14-Sep-2026, after the open settles. Approved by Arun on 13-Sep-2026.
#
# A sleep is not a gate: it can wake early after a suspend, or days late. The date and the
# window are re-checked on waking, and the Python executor checks them a second time.
exec >> /home/arun/quantifyd/logs/deferred_rebalance_20260914.log 2>&1
echo "armed at $(date) pid $$"

target=$(date -d '2026-09-14 09:45' +%s)
now=$(date +%s)
if [ "$target" -gt "$now" ]; then
    sleep $((target - now))
fi

d=$(date +%F)
hm=$(date +%H%M)
echo "woke at $(date) d=$d hm=$hm"
if [ "$d" != "2026-09-14" ] || [ "$hm" -lt 940 ] || [ "$hm" -gt 1430 ]; then
    echo "ABORT: outside 2026-09-14 09:40-14:30"
    exit 1
fi

cd /home/arun/quantifyd && venv/bin/python -u scripts/rebalance_mpf_20260914.py --execute
echo "executor exited $? at $(date)"
