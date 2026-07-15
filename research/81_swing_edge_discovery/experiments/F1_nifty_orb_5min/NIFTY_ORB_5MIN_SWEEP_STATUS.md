# F1 NIFTY Opening-Range Breakout → 1-4 Session Hold — 5-Min, IS Screen

STATUS: DONE — SIGNAL (see Verdict)

EXP-F1 of research/81 (`EDGE_DISCOVERY_81_STUDY_STATE.md`). Family F: index
futures systems — deepest liquidity, cleanest execution, most automatable.
First 5-min-native experiment; NIFTY50 5-min history 2015-02→present already
in DB (BANKNIFTY deferred to post-backfill repair).

## 1. The Ask

Does the classic opening-range breakout on NIFTY, held from intraday to a
2–4 session swing, clear costs on 11 years of 5-min data? The in-house ORB
cash system (live since 2026-04) trades stocks intraday; this tests the INDEX
with multi-day holds — different animal, futures costs.

## 2. Economic hypothesis (G0)

The first 30–60 min aggregates overnight information; a range break signals
directional imbalance that continues as slower participants react (index
flows, hedging). Counterparty: mean-reversion scalpers fading the open, and
option sellers pinned to yesterday's range. Decay risk: crowded classic;
survives (if at all) on the lowest-cost instrument.

## 3. The Base — locked mechanics

- **Data:** NIFTY50 5-min, canonical loader (audit-clean). Futures-proxy
  costs, slippage 3 bps/side (conservative for NIFTY futures).
- **Splits (5-min series, pre-registered):** IS 2015-02-01→2021-09-30 (~60%),
  Val 2021-10-01→2023-12-31, OOS 2024-01-01→end. This screen touches IS ONLY.
- **Opening range:** first W five-min bars of the session (W = 6 → 09:15-09:45,
  W = 12 → 09:15-10:15).
- **Signal:** first 5-min close beyond the OR after the window (long above
  OR-high, short below OR-low; one signal/side/day). Entry next bar open.
- **Stop:** opposite OR bound (absolute). No target.
- **Time-stop:** 1 (same-session close), 2, 4 sessions.

## 4. Plan — pre-registered grid (LOCKED)

| Axis | Values |
|---|---|
| OR window W (bars) | 6, 12 |
| Direction | long, short |
| Time-stop sessions | 1, 2, 4 |

**12 cells**, single symbol (ledger +12 → 108). Expected n ≈ 700–1,500/cell.

**G1 gate:** net expectancy > 0 with t ≥ 3 (single-symbol, so no sym-breadth
criterion; instead ≥60% of YEARS net-positive) and coherence across W.
**Falsification:** all cells net-negative → index ORB swing = NO EDGE, family
F moves to other index mechanics post-backfill (BANKNIFTY, gap systems).

## 5. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-15 ~21:10 IST | Pre-registered | runner: `scripts/run_f1_nifty_orb.py` |

## 6. Crash Recovery

Re-run `scripts/run_f1_nifty_orb.py` (idempotent, overwrites results CSVs);
log `/tmp/f1_orb.log`; results in `experiments/F1_nifty_orb_5min/results/`.

## 7. Findings

(after run)

## VERDICT (2026-07-15 ~21:20 IST): SIGNAL — strongest of the daily+index screens; G2 follow-up earned

LONG-only, coherent structure: net rises monotonically with hold (ts1 neg ->
ts2 +4..6 -> ts4 +11..12 bps) consistently across BOTH W; shorts uniformly
negative; W12_L_ts4 positive 6/7 years and IMPROVING (2017-21: +10..+26bps/yr;
2015-16 flat). n=463-512.

Cost sensitivity (pre-declared): slippage 1bp (realistic NIFTY futures) ->
net +14.7..15.6 bps, t=2.2-2.3; 3bp (conservative) -> t=1.65-1.67; 6bp stress
-> t=0.7-0.8 (dies at 2x conservative slippage - fragility flagged per brief).

Gate t>=3 NOT met -> label SIGNAL. Earned follow-ups (to pre-register):
EXP-F2 = conditioning filters on W12_L_ts4 (gap direction/size, prev-day
trend, INDIAVIX regime) aiming to concentrate the edge; then Val-split
confirmation of the LOCKED best cell; BANKNIFTY replica post-backfill.
STATUS: DONE
