#!/bin/bash
# Arun's same-day abort: stop resting at the pivot (cheap fill), sold at the close if the
# close did not finish above the pivot (the breakout-held filter, bought with a small toll
# instead of a -8% stop). Slot frees the same day, so the next candidate can use it.
#
# Live spec otherwise: trail-15, -8% close stop, 16 slots @ 6.25%, no market gate,
# 2006-01-01 -> 2026-08-31. Realistic fills throughout: max(pivot, open), never the pivot
# on a gap-up day.
#
# Four arms, because the answer hinges on what the toll costs:
#   25bps 30-seed   the honest headline
#   0bps  30-seed   how much of the result the churn eats
#   after-tax       the decision gate per the house rule
#   RS selection    the live book's actual pick order, deterministic, single path
cd /home/arun/quantifyd
R=research/158_oa_arming_width/scripts/oa_entry_mechanics.py
C="--start 2006-01-01 --end 2026-08-31 --slots 16 --trail-sma 15 --poke-trigger --fill-realistic --eod-abort"

until grep -q RS_DONE /tmp/mpf/rs.log 2>/dev/null; do sleep 20; done

echo "########## ABORT: 25bps, 30 seeds ##########"
venv/bin/python -u $R $C --cost 25 --ensemble 30 --tag abort_25

echo
echo "########## ABORT: ZERO cost, 30 seeds (how much does the churn eat?) ##########"
venv/bin/python -u $R $C --cost 0 --ensemble 30 --tag abort_free

echo
echo "########## ABORT: 25bps AFTER-TAX, 30 seeds ##########"
venv/bin/python -u $R $C --cost 25 --ensemble 30 --stcg --tag abort_tax

echo
echo "########## ABORT: 25bps, RS selection (the live pick order) ##########"
venv/bin/python -u $R $C --cost 25 --selection rs --tag abort_rs
echo
echo ABORT_DONE
