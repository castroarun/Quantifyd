#!/bin/bash
# Regenerate ALL straddle backtest/replay JSONs from the recorded chain. Daily post-close cron.
cd /home/arun/quantifyd || exit 1
S=research/58_intraday_recenter_straddle
# V1 all-days replay (writes static/app/straddles/v1_daily.json directly)
PYTHONPATH=. ./venv/bin/python $S/scripts/straddle_daily_journeys.py >/dev/null 2>&1
cp static/app/straddles/v1_daily.json frontend/public/straddles/v1_daily.json 2>/dev/null
# V1 backtest (0.4% / 0-1-DTE) -> v1_oad_4_dte1.json -> v1.json
PYTHONPATH=. ./venv/bin/python $S/scripts/v1_oad.py >/dev/null 2>&1
cp $S/results/v1_oad_4_dte1.json static/app/straddles/v1.json 2>/dev/null
cp $S/results/v1_oad_4_dte1.json frontend/public/straddles/v1.json 2>/dev/null
# V2 both stop-variants
for m in 1.5 2.0; do
  V2_MOVE=$m PYTHONPATH=. ./venv/bin/python $S/scripts/v2_curves.py >/dev/null 2>&1
  cp $S/results/v2_positional.json static/app/straddles/v2_${m}.json 2>/dev/null
  cp $S/results/v2_positional.json frontend/public/straddles/v2_${m}.json 2>/dev/null
done
# V2 variant lab (naked vs iron-fly x move-stop) + strategy leaderboard (rate all systems)
PYTHONPATH=. ./venv/bin/python $S/scripts/v2_variants.py >/dev/null 2>&1
PYTHONPATH=. ./venv/bin/python $S/scripts/strategy_rankings.py >/dev/null 2>&1
# V1 + 30% combined-premium SL backtest (reads the fresh options_study.json)
./venv/bin/python $S/scripts/sl30_journeys.py >/dev/null 2>&1
echo "$(date "+%F %T") regenerated v1_daily + v1 + v2 + variants + rankings + sl30"
cd /home/arun/quantifyd && venv/bin/python3 research/111_sensex_manual_mgmt/scripts/csl_paper_backfill.py >> /tmp/csl_backfill.log 2>&1
cd /home/arun/quantifyd && venv/bin/python3 research/111_sensex_manual_mgmt/scripts/nas_baseline.py >> /tmp/nas_baseline.log 2>&1
