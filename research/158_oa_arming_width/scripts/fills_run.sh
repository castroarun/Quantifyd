#!/bin/bash
# Which convention carries the published result: the TRIGGER or the FILL?
#
# Phase 1 of r/142 validated the entry as a buy-stop resting at the all-time-high close,
# "filled intraday at the pivot price on the first day price trades through it", and the
# published page states "filled at max(pivot, open)". So the honest trigger is the TOUCH
# (high >= pivot), which is what the live scanner does.
#
# Two separate optimisms sit in the published engine, and they have to be told apart:
#   TRIGGER  counting a trade only when the CLOSE held above the pivot, while pricing it at
#            the open. A resting order cannot decline the touches that close back below.
#   FILL     booking the PIVOT price even when the stock gapped open above it. The
#            validation script flagged this as fill inflation in its own docstring.
#
# 2x2. Everything else is the live spec: trail-15, -8% close stop, 16 slots @ 6.25%,
# no market gate, 2006-01-01 -> 2026-08-31, 30 seeds.
# The zero-cost pivot-fill touch arm is the closest thing to the site's own convention and
# is there to see whether their number is reachable at all on our data.
cd /home/arun/quantifyd
R=research/158_oa_arming_width/scripts/oa_entry_mechanics.py
C="--start 2006-01-01 --end 2026-08-31 --slots 16 --ensemble 30 --trail-sma 15"

# wait for the mechanic bake-off so the two runs do not fight for the box during market hours
until grep -q ALL_DONE /tmp/mpf/modes.log 2>/dev/null; do sleep 20; done

echo "########## touch trigger + PIVOT fill (their convention) + 25bps ##########"
venv/bin/python -u $R $C --poke-trigger --cost 25 --tag fill_touch_pivot

echo
echo "########## touch trigger + PIVOT fill + ZERO cost (their convention, no frictions) ##########"
venv/bin/python -u $R $C --poke-trigger --cost 0 --tag fill_touch_pivot_free

echo
echo "########## close trigger + PIVOT fill + 25bps (both optimisms) ##########"
venv/bin/python -u $R $C --cost 25 --tag fill_close_pivot

echo
echo "########## touch trigger + REALISTIC fill + ZERO cost (is it cost, or the signal?) ##########"
venv/bin/python -u $R $C --poke-trigger --fill-realistic --cost 0 --tag fill_touch_real_free
echo
echo FILLS_DONE
