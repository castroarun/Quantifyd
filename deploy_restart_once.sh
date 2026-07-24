#!/bin/bash
# ONE-SHOT (2026-06-11 15:36): activate the NAS Day & Gap matrix backend.
# The code was deployed to disk during market hours, so the restart was deferred
# to after close. After close NAS is paper-only and EOD-squared, so this is safe.
# Self-removes its cron line so it runs exactly once.
set -u
LOG=/home/arun/quantifyd/logs/matrix_deploy.log
if /usr/bin/sudo /usr/bin/systemctl restart quantifyd; then
    sleep 8
    echo "[$(date '+%F %T')] restarted to activate day-matrix backend; service=$(/usr/bin/systemctl is-active quantifyd)" >> "$LOG"
else
    echo "[$(date '+%F %T')] ERROR restarting for day-matrix activation" >> "$LOG"
fi
crontab -l 2>/dev/null | grep -v 'deploy_restart_once.sh' | crontab - \
  && echo "[$(date '+%F %T')] removed own cron line" >> "$LOG"
