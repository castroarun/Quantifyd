#!/bin/bash
# Deferred CSL Best-Config Lab regen (2026-08-24). The lab's cost model changed from a flat
# Rs160 to a venue/size-aware round trip, so the whole sweep must re-run to restate every cell
# net. That is heavy 3-sec compute -> it must NOT run during market hours (a rogue poller
# starved the live monitors on 2026-08-14). Gate on the real clock, not on a sleep.
LOG=/tmp/lab_regen.log
cd /home/arun/quantifyd || exit 1
echo "[$(TZ=Asia/Kolkata date)] deferred lab regen armed (waiting for 15:40 IST)" >> $LOG
while true; do
  H=$(TZ=Asia/Kolkata date +%H%M); D=$(TZ=Asia/Kolkata date +%u)
  if [ "$D" -ge 6 ] || [ "$H" -ge 1540 ]; then break; fi
  sleep 120
done
echo "[$(TZ=Asia/Kolkata date)] clock gate passed -> sweep starting" >> $LOG
nice -n 15 venv/bin/python3 -u research/111_sensex_manual_mgmt/scripts/entry_exit_sweep.py >> $LOG 2>&1
echo "[$(TZ=Asia/Kolkata date)] sweep rc=$? -> frontend build" >> $LOG
export PATH=/home/arun/.nvm/versions/node/v20.20.2/bin:$PATH
cd /home/arun/quantifyd/frontend && nice -n 15 npm run build >> $LOG 2>&1
echo "[$(TZ=Asia/Kolkata date)] build rc=$? -- DONE" >> $LOG
