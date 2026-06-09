#!/bin/bash
# ONE-SHOT (2026-06-09): after today's NAS paper session, flip master mode back
# to 'live' so tomorrow's 09:00 pre-open restart boots the 3 live 916 ATM
# variants. Then remove this cron line so it never runs again. Idempotent +
# self-deleting: a stray extra run just re-sets 'live' and re-removes itself.
set -u
MODE_FILE="/home/arun/quantifyd/backtest_data/nas_master_mode.json"
LOG="/home/arun/quantifyd/logs/flip_master.log"

printf '%s' '{"mode": "live"}' > "$MODE_FILE"
echo "[$(date '+%F %T')] master mode set to 'live' (one-shot flip-back); file=$(cat "$MODE_FILE")" >> "$LOG"

# self-remove this job from crontab
crontab -l 2>/dev/null | grep -v 'flip_master_live_once.sh' | crontab - \
  && echo "[$(date '+%F %T')] removed own cron line" >> "$LOG"
