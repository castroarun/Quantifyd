#!/bin/bash
# research/174 — the full sweep, to be run AFTER the long-dated bhav backfill is merged
# into backtest_data/market_data.db (which happens after 15:40 IST).
#
# Each phase writes its own CSV under results/ and logs to /tmp/r174_<phase>.log.
# Safe to re-run: every phase rewrites its own outputs from scratch.
set -e
cd /home/arun/quantifyd
S=research/174_long_dated_short_premium/scripts
export R174_TAG=""

echo "=== P1 tenor bake-off ==="
R174_TENORS=30,45,60,75,90,105,120,150,180,210,240,270,300,365 \
  venv/bin/python -u $S/run_p1_tenor.py 2>&1 | tail -5

echo "=== P2 structures ==="
R174_ARMS=45:21,60:27,90:40,120:54,180:81,240:108 \
  venv/bin/python -u $S/run_p2_structures.py 2>&1 | tail -8

echo "=== P3 stops (paired) ==="
R174_ARMS=45:21,60:21,90:21,120:54,180:81,240:108 \
  venv/bin/python -u $S/run_p3_stops.py 2>&1 | tail -5

echo "=== P5 finalists ==="
venv/bin/python -u $S/run_p5_finals.py 2>&1 | tail -20

echo "ALL DONE"
