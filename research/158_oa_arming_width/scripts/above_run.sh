#!/bin/bash
# Arun's stop-above-the-breakout-candle entry. Two filters in series, both knowable before
# the order is placed: the breakout closed above the pivot, and the next session exceeded
# the breakout candle's high. An ordinary resting buy-stop, so it needs no live process.
#
# Live spec otherwise: trail-15, -8% close stop, 16 slots @ 6.25%, no market gate,
# realistic fills, 2006-01-01 -> 2026-08-31.
cd /home/arun/quantifyd
R=research/158_oa_arming_width/scripts/oa_entry_mechanics.py
C="--start 2006-01-01 --end 2026-08-31 --slots 16 --trail-sma 15 --fill-realistic --entry-mode stop_above_candle"

until grep -q ABORT_DONE /tmp/mpf/abort.log 2>/dev/null; do sleep 20; done

echo "########## STOP-ABOVE-CANDLE: 25bps, 30 seeds ##########"
venv/bin/python -u $R $C --cost 25 --ensemble 30 --tag above_25

echo
echo "########## STOP-ABOVE-CANDLE: 25bps AFTER-TAX, 30 seeds ##########"
venv/bin/python -u $R $C --cost 25 --ensemble 30 --stcg --tag above_tax

echo
echo "########## STOP-ABOVE-CANDLE: 25bps, RS selection (the live pick order) ##########"
venv/bin/python -u $R $C --cost 25 --selection rs --tag above_rs

echo
echo "########## STOP-ABOVE-CANDLE: trail-20 variant, 25bps, 30 seeds ##########"
venv/bin/python -u $R --start 2006-01-01 --end 2026-08-31 --slots 16 --trail-sma 20 \
  --fill-realistic --entry-mode stop_above_candle --cost 25 --ensemble 30 --tag above_t20
echo
echo ABOVE_DONE
