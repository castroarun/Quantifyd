#!/bin/bash
# One-shot post-close restart (2026-08-07 15:32 IST) to activate breakout-paper
# cash-model v2 (settled buffer + T+1 liquid-fund settlement). Self-removes.
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] post-close restart: activating breakout-paper cash-model v2"
if /usr/bin/sudo -n /bin/systemctl restart quantifyd; then
    sleep 25
    echo "[$(ts)] service state: $(/usr/bin/systemctl is-active quantifyd)"
    curl -s http://localhost:5000/api/breakout-paper/state | python3 -c "
import json, sys
d = json.load(sys.stdin)
ok = 'cash_buffer' in d
print('cash-model v2 live:', ok)
print('  nav', d.get('nav'), '| buffer', d.get('cash_buffer'),
      '| in-transit', d.get('cash_in_transit'), '| fund', d.get('cash_liquid_fund'),
      '| interest', d.get('interest_earned'), '| gate', d.get('gate'))
" || echo "[$(ts)] WARN: state endpoint not parseable yet"
else
    echo "[$(ts)] ERROR: restart failed (rc=$?) — service left as-was"
fi

# self-remove this one-shot cron line
crontab -l | grep -v bp_v2_restart.sh | crontab -
echo "[$(ts)] one-shot cron entry removed"
