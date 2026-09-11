#!/bin/bash
# Trigger mechanic x trail length, at the ADOPTED spec otherwise.
# RESULTS.md line 286: "Adopted spec (2026-09-03): trail-20, -8% stop,
# 16 slots @6.25%, NO market gate." Published basis: 30-seed median 37.8% CAGR.
# So: no --skip-weak (that ADDS a gate the spec rejects), 30 seeds, 16 slots, 25bps.
cd /home/arun/quantifyd
R=research/142_bananapatterns_replication/scripts/bluesky_replay.py
COMMON="--start 2006-01-01 --end 2026-08-31 --slots 16 --fill-realistic --cost 25 --ensemble 30"
for TRAIL in 20 15; do
  for TRIG in close poke; do
    FLAG=""
    [ "$TRIG" = poke ] && FLAG="--poke-trigger"
    echo "########## trail-${TRAIL}  trigger=${TRIG} ##########"
    venv/bin/python -u $R $COMMON --trail-sma "$TRAIL" $FLAG --tag "t${TRAIL}_${TRIG}"
    echo
  done
done
echo ALL_DONE
