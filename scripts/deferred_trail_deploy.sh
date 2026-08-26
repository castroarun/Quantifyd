#!/bin/bash
# research/128 SENSEX naked-survivor trail fix -> needs a gunicorn restart (app.py changed).
# Gate on the REAL clock, and refuse to restart while any option leg is still open.
LOG=/tmp/trail_deploy.log
cd /home/arun/quantifyd || exit 1
echo "[$(TZ=Asia/Kolkata date)] trail deploy armed (waiting for 15:40 IST)" >> $LOG
while true; do
  H=$(TZ=Asia/Kolkata date +%H%M); D=$(TZ=Asia/Kolkata date +%u)
  if [ "$D" -ge 6 ] || [ "$H" -ge 1540 ]; then break; fi
  sleep 120
done
echo "[$(TZ=Asia/Kolkata date)] clock gate passed - checking open legs" >> $LOG
set -a; . ./.env; set +a
OPEN=$(venv/bin/python3 - <<'PY'
import sys; sys.path.insert(0,".")
try:
    from services.kite_service import get_kite
    k=get_kite(); n=0
    for p in (k.positions().get("net") or []):
        ts=p.get("tradingsymbol") or ""
        if int(p.get("quantity") or 0)!=0 and (ts.startswith("NIFTY") or ts.startswith("SENSEX")):
            print("OPEN %s qty %s"%(ts,p.get("quantity")), file=sys.stderr); n+=1
    print(n)
except Exception as e:
    print("ERR", file=sys.stderr); print(99)
PY
)
echo "[$(TZ=Asia/Kolkata date)] open option legs: $OPEN" >> $LOG
if [ "$OPEN" != "0" ]; then
  echo "[$(TZ=Asia/Kolkata date)] ABORT - legs still open (or broker unreadable). Restart NOT done." >> $LOG
  exit 1
fi
sudo systemctl restart quantifyd >> $LOG 2>&1
echo "[$(TZ=Asia/Kolkata date)] restart rc=$? -- DONE" >> $LOG
sleep 20
curl -s -o /dev/null -w "health %{http_code}\n" http://127.0.0.1:5000/api/nas/ticker/status >> $LOG 2>&1
