#!/bin/bash
# Daily Kite auto-login (cron @ 08:50 IST Mon-Fri).
#
# Zerodha invalidates the access_token every morning. A valid token MUST be in
# backtest_data/access_token.json BEFORE the 09:00 pre-open restart, because the
# fresh 09:00 process boots its in-process recorder + ticker off that token. If
# the token is stale at 09:00 boot, the options-chain recorder starts broken
# (in-process cached kite) and the 09:16 NAS ticker can't connect — blanking
# NAS trades AND the straddle recorder for the whole session (incident 2026-06-09).
#
# Hardened 2026-06-09: instead of 3 quick retries (~3 min), retry every 20s
# until a hard DEADLINE of 08:59:30 — i.e. up to ~9 min of runway — so a
# transient Kite login-flow slowdown is absorbed before the 09:00 restart.
# All login+ticker-start happens in THIS pre-restart process, so it never
# conflicts with the 09:16 autostart's one-shot Twisted reactor.
#
# Cron (user arun):  50 8 * * 1-5 /home/arun/quantifyd/auto_login.sh >> /home/arun/quantifyd/auto_login.log 2>&1

set -u

URL="http://localhost:5000/api/auth/auto-login"
DEADLINE_HHMMSS="085930"     # stop retrying at 08:59:30, before the 09:00 restart
RETRY_SECS=20

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
nowc(){ date '+%H%M%S'; }

echo "[$(ts)] auto_login starting (retry until 08:59:30)"

attempt=0
while [ "$(nowc)" -lt "$DEADLINE_HHMMSS" ]; do
    attempt=$((attempt + 1))
    response=$(curl -s -w '\nHTTP_CODE:%{http_code}' -X POST --max-time 40 "$URL")
    http_code=$(echo "$response" | grep -oE 'HTTP_CODE:[0-9]+' | cut -d: -f2)
    body=$(echo "$response" | grep -v 'HTTP_CODE:')

    if [ "$http_code" = "200" ] && echo "$body" | grep -q '"status":"success"'; then
        echo "[$(ts)] attempt $attempt SUCCESS: $body"
        exit 0
    fi

    echo "[$(ts)] attempt $attempt FAILED (http=$http_code): $body"
    # stop early if the next sleep would cross the deadline
    if [ "$(nowc)" -ge "$DEADLINE_HHMMSS" ]; then break; fi
    sleep "$RETRY_SECS"
done

echo "[$(ts)] auto_login EXHAUSTED — token NOT refreshed before 09:00 restart (attempts=$attempt). NAS + recorder will be dark; manual /api/auth/auto-login needed."
exit 1
