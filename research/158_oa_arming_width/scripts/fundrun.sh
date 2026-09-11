#!/bin/bash
# OA: does Arun's fundamental screen rescue an entry that loses without it?
#
# Window 2024-08-01 -> 2026-09-04, the only span where the three annual years a decision
# needs were already filed when the trade was taken. --base-start 2005-01-01 keeps the panel
# long so the pivot is a true all-time-high close and not an 18-month high.
#
# Tested on the two best HONEST entries, because an overlay on the look-ahead entry would
# mean nothing:
#   stop_above_candle + trail-20   the best tradeable variant found (9.9% over 20y)
#   close_same        + trail-15   Arun's 15:10 entry (8.5% over 20y)
#
# Masks: strict = all five criteria. The leave-one-out variants say WHICH criterion binds.
# missing=fail vs missing=pass brackets the data-coverage bias; both are reported.
#
# A two-year window is one regime and cannot establish an edge. It can show whether the
# screen changes the outcome, and the unfiltered arm in the SAME window is the comparator.
cd /home/arun/quantifyd
R=research/158_oa_arming_width/scripts/oa_entry_mechanics.py
RES=research/158_oa_arming_width/results
C="--start 2024-08-01 --end 2026-09-04 --base-start 2005-01-01 --slots 16 --fill-realistic --cost 25 --ensemble 30"
A1="--entry-mode stop_above_candle --trail-sma 20"
A2="--entry-mode close_same --trail-sma 15"

for NAME in above_t20 close_t15; do
  if [ "$NAME" = above_t20 ]; then ARM="$A1"; else ARM="$A2"; fi
  echo "################ $NAME : NO FILTER (baseline for this window) ################"
  venv/bin/python -u $R $C $ARM --tag "w2y_${NAME}_nofilter"
  echo
  for V in strict no_de no_roce no_growth growth_only; do
    for MISS in fail pass; do
      echo "################ $NAME : mask=$V missing=$MISS ################"
      venv/bin/python -u $R $C $ARM --fund-mask "$RES/fund_mask_$V.npz" \
        --fund-missing $MISS --tag "w2y_${NAME}_${V}_${MISS}"
      echo
    done
  done
done
echo FUNDRUN_DONE
