#!/bin/bash
cd /home/arun/quantifyd
R=research/142_bananapatterns_replication/scripts/bluesky_replay.py
COMMON="--start 2006-01-01 --end 2026-08-31 --slots 16 --skip-weak --fill-realistic --cost 25 --trail-sma 15 --ensemble 10"
echo "=== ARM A: close > pivot (the published mechanic, NOT tradeable) ==="
venv/bin/python -u $R $COMMON --tag close142
echo
echo "=== ARM B: high >= pivot (a resting stop = WHAT THE LIVE BOOK DOES) ==="
venv/bin/python -u $R $COMMON --poke-trigger --tag poke142
echo
echo ALL_DONE
