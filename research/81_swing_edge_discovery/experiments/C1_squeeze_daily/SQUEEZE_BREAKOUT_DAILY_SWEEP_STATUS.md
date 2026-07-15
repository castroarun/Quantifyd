# C1 Volatility Squeeze→Expansion Breakout — F&O Daily, IS Screen

STATUS: PRE-REGISTERED
EXP-C1 of research/81, family C (volatility). G0: vol clusters; expansion
out of compression attracts breakout flow (counterparty: theta sellers,
range traders). In-house prior: V2 combo research/61 — compression predicts
TREND on weeklies.

Signal: BB(20,2) rel-width trailing-252d percentile ≤ q YESTERDAY, today
closes beyond band. GRID LOCKED: q {0.10,0.20} × dir {L,S} × ts {2,4} = 16
cells (ledger +16). Stop FIXED 2.0×ATR14, no target.
Gate/falsification: standard G1 (t≥3, ≥55% sym-pos, stability across q).
Runner: scripts/run_c1_squeeze_daily.py

## VERDICT (2026-07-15 ~21:00 IST): NO EDGE (incoherent grid)

4/16 cells net-positive (best sq10_L_ts4 +17.6bps t1.70) but the pattern is a
spike, not a plateau: longs work only at ts4, shorts only at ts2, and the
sq10/sq20 ordering flips by cell. Only ~half of years positive. Fails the
pre-registered stability requirement -> NO EDGE for daily squeeze-breakout
with 2-4d exits. (Weekly-compression->trend remains real per research/61/67 -
the effect appears to live at longer horizons than this brief's time-stop.)
STATUS: DONE
