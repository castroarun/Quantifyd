# E1 Cross-Sectional 5-Day RS Rotation — F&O Daily, IS Screen

STATUS: PRE-REGISTERED
EXP-E1 of research/81, family E (cross-sectional). G0: at ~1-week horizon the
literature says REVERSAL, at months MOMENTUM — this screen lets the data pick
the sign net-of-cost on F&O futures (shortable). Fresh entry into top/bottom
decile of universe 5-day return; skip1 variant strips 1-day reversal.

GRID LOCKED: skip {0,1} × dir {L,S} × ts {2,4} = 16 cells (ledger +16).
Stop FIXED 2.5×ATR14, no target.
Gate/falsification: standard G1. If long-decile and short-decile BOTH lose,
family E daily rotation = NO EDGE at this horizon.
Runner: scripts/run_e1_xsec_rs.py

## VERDICT (2026-07-15 ~21:00 IST): NO EDGE

All 16 cells net-negative (t -4.1..-8.9), both directions, both skips: at a
5-day horizon neither continuation nor reversal survives costs on F&O daily
rotation. Consistent with research/46 (reversal decayed/eaten by costs).
STATUS: DONE
