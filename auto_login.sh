#!/bin/bash
# Daily Kite auto-login (called by cron at 08:55 IST Mon-Fri).
#
# Zerodha invalidates the access_token every morning, so we need to
# re-login before the ORB engine's 09:14 initialize_day cron fires.
# This script calls the Flask auto-login endpoint (uses TOTP from env).
#
# Retries 3 times with 20s backoff in case the endpoint transiently
# fails (e.g. Kite login page temporarily slow at market open rush).
#
# Cron entry (in `crontab -e` for user `arun`):
#   55 8 * * 1-5 /home/arun/quantifyd/auto_login.sh >> /home/arun/quantifyd/auto_login.log 2>&1

set -u

URL="http://localhost:5000/api/auth/auto-login"
MAX_ATTEMPTS=3
BACKOFF_SECS=20

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] auto_login starting"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    response=$(curl -s -w '\nHTTP_CODE:%{http_code}' -X POST --max-time 45 "$URL")
    http_code=$(echo "$response" | grep -oE 'HTTP_CODE:[0-9]+' | cut -d: -f2)
    body=$(echo "$response" | grep -v 'HTTP_CODE:')

    if [ "$http_code" = "200" ] && echo "$body" | grep -q '"status":"success"'; then
        echo "[$(ts)] attempt $attempt SUCCESS: $body"
        exit 0
    fi

    echo "[$(ts)] attempt $attempt FAILED (http=$http_code): $body"

    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
        echo "[$(ts)] retrying in ${BACKOFF_SECS}s..."
        sleep "$BACKOFF_SECS"
    fi
done

echo "[$(ts)] auto_login EXHAUSTED all $MAX_ATTEMPTS attempts"
exit 1
