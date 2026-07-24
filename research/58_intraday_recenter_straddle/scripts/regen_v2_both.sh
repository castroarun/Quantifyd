#!/bin/bash
# Regenerate BOTH V2 stop-variant backtests (1.5% + 2.0%) from the recorded chain.
# Runs daily after close via cron. Frontend serves v2_1.5.json / v2_2.0.json.
cd /home/arun/quantifyd || exit 1
for m in 1.5 2.0; do
  V2_MOVE=$m PYTHONPATH=. ./venv/bin/python research/58_intraday_recenter_straddle/scripts/v2_curves.py >/dev/null 2>&1
  cp research/58_intraday_recenter_straddle/results/v2_positional.json frontend/public/straddles/v2_${m}.json
  cp research/58_intraday_recenter_straddle/results/v2_positional.json static/app/straddles/v2_${m}.json
done
echo "$(date "+%F %T") regenerated v2_1.5.json + v2_2.0.json"
