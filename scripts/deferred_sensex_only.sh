#!/bin/bash
LOG=/tmp/sensex_only.log
cd /home/arun/quantifyd || exit 1
echo "[$(TZ=Asia/Kolkata date)] armed - waiting for Fri 28-Aug 15:45" >> $LOG
while true; do D=$(TZ=Asia/Kolkata date +%Y-%m-%d); H=$(TZ=Asia/Kolkata date +%H%M); if [ "$D" \> "2026-08-28" ] || { [ "$D" = "2026-08-28" ] && [ "$H" -ge 1545 ]; }; then break; fi; sleep 300; done
echo "[$(TZ=Asia/Kolkata date)] applying SENSEX-only" >> $LOG
venv/bin/python3 scripts/apply_sensex_only.py >> $LOG 2>&1
echo "[$(TZ=Asia/Kolkata date)] rc=$? done" >> $LOG
