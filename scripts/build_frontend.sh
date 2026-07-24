#!/bin/bash
# Safe frontend build: vite wipes static/app/, so ALWAYS rebuild runtime files after.
export PATH=/home/arun/.nvm/versions/node/v20.20.2/bin:$PATH
cd /home/arun/quantifyd/frontend && npm run build || exit 1
cd /home/arun/quantifyd
./venv/bin/python3 research/82_sensex_expiryday/scripts/sensex_paper.py
./venv/bin/python3 scripts/sensex_live_writer.py
./venv/bin/python3 research/58_intraday_recenter_straddle/scripts/straddle_paper_live.py
./venv/bin/python3 research/80_farDTE_rescue/scripts/condor_paper.py
bash research/58_intraday_recenter_straddle/scripts/regen_v2_both.sh
bash research/90_nifty_strangle_rules/scripts/regen_travel.sh
echo "build + runtime files restored"
