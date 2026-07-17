# research/83 — Faithful Turtle (S1/S2, both directions) on Indian F&O Equities

STATUS: PRE-REGISTERED → RUNNING

User mandate 2026-07-17: Turtle Traders system (Dennis/Covel), restricted to
Nifty-200/our stocks (MCX declined after Phase-0 audit). NEW information vs
prior art: (a) SHORT side at multi-week trailing holds — the last unclosed
short cell; (b) faithful S1(20/10)+S2(55/20) dual-channel mechanics with 2N
stops. Long side expected ≈ research/71 (live breakout paper book) — stated
prior, not a discovery claim.

GRID LOCKED (EXP-T1, 4 cells, ledger +4): {S1, S2} × {L, S}, F&O universe
(~81, CA-guarded), IS 2005-2017, futures-proxy 3bp, no pyramiding (phase 2
only if G1 passes), no time-stop (trailing channel exit is the exit).
Gate: standard G1 (net t≥3, ≥55% syms, per-year sanity). Falsification:
shorts negative → the short question closes across ALL horizons; longs
redundant vs r/71 → no build.

Dedicated simulator (trailing exits; engine is time-stop-only) with a
synthetic sanity check that must pass before the sweep.
Runner: scripts/run_turtle_probe.py · log /tmp/turtle_probe.log (VPS)
Crash doc: this file. OOS ledger: nothing consumed.
