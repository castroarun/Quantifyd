#!/bin/bash
# Run the given command ONLY on an NSE trading day.
#
# Cron fires "mon-fri" regardless of exchange holidays. On a closed day a book's jobs either
# send orders the exchange refuses, record a fake flat day in a value curve, or repeat the
# previous session's decisions on stale data. This wrapper asks the project's trading calendar
# (services/trading_calendar.py, config/nse_holidays_<year>.json) and exits quietly on a
# non-trading day.
#
# Answer cached once per day in /tmp/nse_trading_day_<date>, so the per-minute jobs do not
# start the calendar every minute.
#
# FAILS OPEN: if the calendar cannot be read the command runs, and that is logged. A broken
# holiday file must never silently stop a trading day.
#
# Usage in cron:  cd /home/arun/quantifyd && scripts/on_trading_day.sh <command...>
# Test:           NSE_TD_DATE=2026-09-14 scripts/on_trading_day.sh echo hi
ROOT=/home/arun/quantifyd
d=${NSE_TD_DATE:-$(date +%F)}
cache=/tmp/nse_trading_day_$d

if [ -s "$cache" ]; then
    r=$(cat "$cache")
else
    r=$("$ROOT/venv/bin/python" - "$d" 2>/dev/null <<'PY'
import sys
sys.path.insert(0, '/home/arun/quantifyd')
from datetime import date
from services.trading_calendar import get_default_calendar
print('1' if get_default_calendar().is_trading_day(date.fromisoformat(sys.argv[1])) else '0')
PY
)
    r=$(echo "$r" | tail -1)
    if [ "$r" = "0" ] || [ "$r" = "1" ]; then
        echo "$r" > "$cache"
    else
        echo "[$(date '+%F %T')] on_trading_day: calendar unreadable - running anyway: $*" >&2
        r=1
    fi
fi

if [ "$r" != "1" ]; then
    echo "[$(date '+%F %T')] skipped - $d is not an NSE trading day: $*"
    exit 0
fi
exec "$@"
