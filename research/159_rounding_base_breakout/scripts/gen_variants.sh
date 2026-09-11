#!/bin/bash
# research/159 backtest Phase A: generate v3 entry-variant event sets.
# Single sequential niced process - deliberately gentle, the box runs live trading.
cd /home/arun/quantifyd || exit 1
for S in 15 20; do
  for K in 2 3 5; do
    for A in 0.90 0.95 1.00; do
      OUT="rounding_base_events_v3_s${S}_k${K}_a${A}.csv"
      if [ -s "research/159_rounding_base_breakout/results/$OUT" ]; then
        echo "skip $OUT (exists)"; continue
      fi
      echo "=== $(TZ=Asia/Kolkata date +%H:%M) generating S=$S K=$K ATH=$A ==="
      nice -n 19 venv/bin/python -u \
        research/159_rounding_base_breakout/scripts/detect_rounding_base_v3.py \
        --shelf=$S --k=$K --ath=$A --out="$OUT" 2>&1 | tail -3
    done
  done
done
echo "ALL VARIANTS DONE $(TZ=Asia/Kolkata date +%H:%M)"
