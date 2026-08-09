# research/110 — Intraday Alternative-Information Signals: RESULTS

## Verdict: **NO EDGE — 0 of 14 cells. The alt-info door (as far as OHLCV proxies can open it) is closed.**

150 names, 2015-2023, entry 10:15 -> exit 15:15, 10bps RT, market-neutral
same-day benchmark for cross-sectional cells.

- **Cross-sectional morning RS: dead in BOTH directions.** Top-decile
  momentum longs -8bps, bottom-decile reversal longs -13bps (t -9.7!) —
  the 10:15 morning-return ranking contains no exploitable afternoon
  information net of costs; continuation and reversal both lose.
- **Event proxy (RVOL>=3 + gap>=1%): following the gap loses hard
  (-40bps IS); fading it looked alive IS (+19.7, t=1.2, halves unstable)
  and flipped negative in Val (-12.7, t=-2.0).** Fake.
- **Flow proxies (up-volume imbalance, CLV accumulation): negative both
  directions, both windows** (VOLIMB_S t=-13.7 IS / -8.6 Val).
- Nothing reached even the IS gate; every cell is negative in Val.

## Program conclusion (r/89 + r/109 + r/110: 58 intraday constructions)
Intraday single-name edges cannot be harvested from ANY OHLCV-derived
information — absolute price signals, indicators, patterns, cross-sectional
ranks, or flow proxies — against ~10bps friction. Closing the intraday
research line. What would change the picture: genuinely external data
(real-time news/earnings feeds, order-book depth) — none of which is in
our data estate. The >=20% CAGR ambition remains routed through multi-day
books + derivatives + leverage on validated curves.

Ledger: +14 (program total 470). OOS 2024+ still untouched.
