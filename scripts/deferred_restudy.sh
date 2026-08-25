#!/bin/bash
# Re-run r/122 + r/124 on the MEASURED cost model (Arun 2026-08-25 "yes go").
# Heavy chain replay -> gate on the real clock, never run during market hours.
LOG=/tmp/restudy.log
cd /home/arun/quantifyd || exit 1
echo "[$(TZ=Asia/Kolkata date)] restudy armed (waiting for 15:40 IST)" >> $LOG
while true; do
  H=$(TZ=Asia/Kolkata date +%H%M); D=$(TZ=Asia/Kolkata date +%u)
  if [ "$D" -ge 6 ] || [ "$H" -ge 1540 ]; then break; fi
  sleep 120
done
echo "[$(TZ=Asia/Kolkata date)] gate passed -> r/122 stage A" >> $LOG
nice -n 15 venv/bin/python3 -u research/122_window_risk_atlas/scripts/stage_a_alldays.py >> $LOG 2>&1
echo "[$(TZ=Asia/Kolkata date)] r/122 stage A rc=$? -> build_atlas" >> $LOG
nice -n 15 venv/bin/python3 -u research/122_window_risk_atlas/scripts/build_atlas.py >> $LOG 2>&1
echo "[$(TZ=Asia/Kolkata date)] r/122 atlas rc=$? -> r/124 stage A" >> $LOG
nice -n 15 venv/bin/python3 -u research/124_monday_window_rehab/scripts/stage_a_monday.py >> $LOG 2>&1
echo "[$(TZ=Asia/Kolkata date)] r/124 stage A rc=$? -> build_monday_atlas" >> $LOG
nice -n 15 venv/bin/python3 -u research/124_monday_window_rehab/scripts/build_monday_atlas.py >> $LOG 2>&1
echo "[$(TZ=Asia/Kolkata date)] r/124 atlas rc=$? -- DONE" >> $LOG
