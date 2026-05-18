#!/bin/bash
# Pre-open clean restart (called by cron at 09:00 IST Mon-Fri).
#
# Defense-in-depth for the 2026-05-18 NAS reactor-poisoning incident.
# This gunicorn process can live for days; KiteTicker's Twisted reactor
# is one-shot per process, so once an overnight stale-token 403 loop
# runs+stops it, the live price feed can never recover in-process and
# all NAS Squeeze variants go silent for the whole session.
#
# Restarting unconditionally at 09:00 — AFTER the 08:55 auto-login token
# refresh and BEFORE the 09:15 session / 09:16 NAS autostart — guarantees
# every trading day starts from a fresh interpreter with a virgin reactor
# and a fresh token. This is the explicitly-sanctioned pre-open safe
# window (the no-restart rule covers 09:15-15:30 only). The in-app 09:05
# conditional reactor-heal then acts as a pure backstop.
#
# Cron entry (in `crontab -e` for user `arun`):
#   0 9 * * 1-5 /home/arun/quantifyd/scripts/preopen_restart.sh >> /home/arun/quantifyd/logs/preopen_restart.log 2>&1
#
# Requires NOPASSWD sudo for `systemctl restart quantifyd` (already
# configured — proven by deploy restarts working non-interactively).

set -u

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] pre-open restart: issuing 'sudo systemctl restart quantifyd'"

if /usr/bin/sudo /usr/bin/systemctl restart quantifyd; then
    sleep 8
    state=$(/usr/bin/systemctl is-active quantifyd 2>/dev/null || echo "unknown")
    echo "[$(ts)] restart issued OK — service state: ${state}"
    exit 0
else
    echo "[$(ts)] ERROR: 'sudo systemctl restart quantifyd' failed (rc=$?)"
    exit 1
fi
