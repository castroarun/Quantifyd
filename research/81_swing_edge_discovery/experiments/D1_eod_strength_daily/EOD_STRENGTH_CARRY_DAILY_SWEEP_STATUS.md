# D1 End-of-Day Strength Carry — F&O Daily, IS Screen

STATUS: PRE-REGISTERED
EXP-D1 of research/81, family D (intraday-pattern→swing). G0: institutional
orders split across days; close pinned at day-extreme with ≥1% move implies
unfinished flow → continuation next 2-4 sessions.

Signal: CLV=(C−L)/(H−L). Long CLV≥thr & ret≥+1%; Short CLV≤1−thr & ret≤−1%.
GRID LOCKED: thr {0.8,0.9} × dir {L,S} × ts {2,4} = 16 cells (ledger +16).
Stop FIXED 2.5×ATR14, no target.
Gate/falsification: standard G1 (t≥3, ≥55% sym-pos, stability across thr).
Runner: scripts/run_d1_eod_strength.py

## VERDICT (2026-07-15 ~21:00 IST): NO EDGE

Only clv0.9_S_ts2 marginally positive (+6.6bps, t1.19); all long cells
negative net, most negative gross. EOD strength does NOT carry 2-4 days on
this universe net of costs. STATUS: DONE
