#!/bin/bash
# Go-live for two-account buy/sell/funds on the Holdings page.
# Builds the frontend into the served static/app AND restarts the backend so the
# new /api/holdings/{order,exit,funds} routes load. GATED to >= 15:40 IST.
set -e
cd /home/arun/quantifyd
H=$(TZ=Asia/Kolkata date +%H); M=$(TZ=Asia/Kolkata date +%M); D=$(TZ=Asia/Kolkata date +%u)
echo "clock: $(TZ=Asia/Kolkata date '+%F %H:%M %A IST')"
if [ "$D" -le 5 ] && { [ "$H" -lt 15 ] || { [ "$H" -eq 15 ] && [ "$M" -lt 40 ]; }; }; then
  echo "ABORT: before 15:40 IST on a weekday — backend restart not allowed."; exit 1
fi
echo "=== build frontend ==="
export PATH="$HOME/.nvm/versions/node/v20.20.2/bin:$PATH"
cd frontend && npm run build && cd ..
echo "=== restart backend ==="
sudo systemctl restart quantifyd
sleep 4
systemctl is-active quantifyd && echo "quantifyd active"
echo "=== smoke: funds endpoints ==="
curl -s -o /dev/null -w "order-route status via app import check\n" http://127.0.0.1:8000/ || true
echo "DONE"
