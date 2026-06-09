#!/bin/bash
# 09:06 pre-open TOKEN safety net (incident 2026-06-09).
#
# Second layer behind the hardened 08:50-08:59:30 auto-login. If the whole
# pre-restart login window was bad, the 09:00 process boots with a stale token
# and everything (NAS ticker + options recorder) stays dark all session. This
# job runs AFTER the 09:00 restart and BEFORE the 09:16 NAS entry: if — and
# ONLY if — the token is still invalid, it does a fresh TOTP login and restarts
# the service so it boots clean with a valid token + virgin reactor.
#
# GATED ON TOKEN VALIDITY (the real failure signal), NOT on ticker-connected,
# so on a normal day (token already valid) it is a strict NO-OP — no restart,
# no interference with the existing 09:05 reactor-heal.
#
# Cron (user arun):  6 9 * * 1-5 /home/arun/quantifyd/token_heal.sh >> /home/arun/quantifyd/logs/token_heal.log 2>&1

cd /home/arun/quantifyd || exit 1
set -a; . ./.env 2>/dev/null; set +a
ts(){ date '+%F %T'; }

valid=$(./venv/bin/python3 -c "
import logging; logging.disable(logging.CRITICAL)
try:
    from services.kite_service import get_kite
    get_kite().profile()
    print('VALID')
except Exception:
    print('INVALID')
" 2>/dev/null | tail -1)

if [ "$valid" = "VALID" ]; then
    echo "[$(ts)] token valid — no action"
    exit 0
fi

echo "[$(ts)] token INVALID after 09:00 boot — re-login (TOTP) + restart"
./venv/bin/python3 -c "
import logging; logging.disable(logging.CRITICAL)
from services.kite_auth import auto_login
print('relogin token:', bool(auto_login()))
" 2>&1 | tail -2

if /usr/bin/sudo /usr/bin/systemctl restart quantifyd; then
    echo "[$(ts)] service restarted — clean boot with fresh token before 09:16"
else
    echo "[$(ts)] ERROR: restart failed"
    exit 1
fi
