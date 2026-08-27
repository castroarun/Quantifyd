#!/bin/bash
LOG=/tmp/rules_true.log
cd /home/arun/quantifyd || exit 1
while true; do H=$(TZ=Asia/Kolkata date +%H%M); if [ "$H" -ge 1525 ]; then break; fi; sleep 60; done
echo "[$(TZ=Asia/Kolkata date)] running rules-true reconstruction" >> $LOG
nice -n 15 venv/bin/python3 scripts/rules_true_0827.py >> $LOG 2>&1
echo "[$(TZ=Asia/Kolkata date)] rc=$? done" >> $LOG
