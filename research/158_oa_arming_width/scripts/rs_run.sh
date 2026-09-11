#!/bin/bash
# The 30-seed ensembles pick RANDOMLY among candidates when more qualify than there are
# slots. The live book does not: it takes highest relative strength first. If RS selection
# is materially better than the random median, every tradeable number so far understates
# the live book, and the verdict has to wait.
#
# sel=rs is deterministic, so one path per arm and no ensemble. Compared against the SAME
# arm's random median, which is what the tables already hold.
cd /home/arun/quantifyd
R=research/158_oa_arming_width/scripts/oa_entry_mechanics.py
C="--start 2006-01-01 --end 2026-08-31 --slots 16 --trail-sma 15 --cost 25 --selection rs"

until grep -q FILLS_DONE /tmp/mpf/fills.log 2>/dev/null; do sleep 20; done

echo "########## RS selection: touch trigger + realistic fill (THE LIVE BOOK) ##########"
venv/bin/python -u $R $C --poke-trigger --fill-realistic --tag rs_touch_real

echo
echo "########## RS selection: buy at the breakout close (Arun 15:10) ##########"
venv/bin/python -u $R $C --entry-mode close_same --fill-realistic --tag rs_close_same

echo
echo "########## RS selection: the look-ahead reference, for scale ##########"
venv/bin/python -u $R $C --fill-realistic --tag rs_open_same
echo
echo RS_DONE
