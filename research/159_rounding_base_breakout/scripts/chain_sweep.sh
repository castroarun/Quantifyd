#!/bin/bash
cd /home/arun/quantifyd || exit 1
R=research/159_rounding_base_breakout/results
while [ "$(ls $R | grep -c 'v3_s')" -lt 18 ]; do sleep 30; done
echo "=== all 18 variants at $(TZ=Asia/Kolkata date +%H:%M); starting 3 sweep shards ==="
for i in 0 1 2; do
  nice -n 12 venv/bin/python -u research/159_rounding_base_breakout/scripts/bt_sweep.py --shard=$i/3 \
    > /tmp/r159_sweep_$i.log 2>&1 &
done
wait
echo "=== ALL SHARDS FINISHED $(TZ=Asia/Kolkata date +%H:%M) ==="
