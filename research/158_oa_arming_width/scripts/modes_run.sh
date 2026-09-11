#!/bin/bash
# Entry-mechanic bake-off at the LIVE spec: trail-15, -8% stop, 16 slots @ 6.25%,
# NO market gate, 25 bps/side, 2006-01-01 -> 2026-08-31, 30-seed ensemble.
# Only the entry mechanic varies. open_same is the control: it must reproduce the
# 40.8% the unforked engine produced, or the fork is not trustworthy.
cd /home/arun/quantifyd
R=research/158_oa_arming_width/scripts/oa_entry_mechanics.py
C="--start 2006-01-01 --end 2026-08-31 --slots 16 --fill-realistic --cost 25 --ensemble 30 --trail-sma 15"
for M in open_same close_same open_next; do
  echo "########## entry=$M  PRE-TAX ##########"
  venv/bin/python -u $R $C --entry-mode "$M" --tag "pre_$M"
  echo
done
for M in close_same open_next; do
  echo "########## entry=$M  AFTER-TAX (20% STCG / 12.5% LTCG) ##########"
  venv/bin/python -u $R $C --entry-mode "$M" --stcg --tag "tax_$M"
  echo
done
echo ALL_DONE
