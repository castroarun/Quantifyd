#!/usr/bin/env bash
# Deploy the Breakout Scanner — RUN AFTER 15:30 IST ONLY (frontend build wipes
# static/app + backend needs a restart). Safe to re-run.
set -e
cd /home/arun/quantifyd
echo "[1/5] ensure node (via nvm, no sudo)"
if ! command -v node >/dev/null 2>&1; then
  export NVM_DIR="$HOME/.nvm"
  [ -d "$NVM_DIR" ] || (curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash)
  . "$NVM_DIR/nvm.sh"; nvm install 20; nvm use 20
fi
. "$HOME/.nvm/nvm.sh" 2>/dev/null || true
echo "[2/5] npm install (frontend)"; cd frontend; [ -d node_modules ] || npm install
echo "[3/5] npm run build (tsc + vite -> static/app)"; npm run build
cd /home/arun/quantifyd
echo "[4/5] restart backend (activates /api/breakout-scanner/*)"; sudo systemctl restart quantifyd; sleep 5
echo "[5/5] verify"; curl -s http://127.0.0.1:5000/api/breakout-scanner/settings | head -c 400; echo
echo "DONE — open http://94.136.185.54:5000/app/breakout-scanner"
