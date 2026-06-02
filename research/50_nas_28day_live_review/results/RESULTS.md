# NAS 8-System 28-Day Live/Paper Review — RESULTS

**Window:** 2026-04-20 → 2026-06-02 (25 trading days) · **180 trades** · **net ₹592,208** · win 54% · combined MaxDD ₹-6,829

**VERDICT: SIGNAL/BEHAVIOUR AUDIT ONLY** — 28-day single-regime sample on actual (not replayed) fills. Real per-trade economics are informative; cross-system ranking is indicative at best. NOT strategy validation.

## Per-system (net ₹ since 2026-04-20)

| System | Trades | Days | WinPct | NetPnL | PerLot | AvgTrade | MaxDD | BestDay | WorstDay | Adj |
|---|---|---|---|---|---|---|---|---|---|---|
| Squeeze OTM | 12 | 9 | 100.0 | 115029 | 1933 | 9586 | 0 | 29128 | 0 | 18 |
| Squeeze ATM | 22 | 14 | 54.5 | -1282 | -133 | -58 | -10335 | 7584 | -5712 | 0 |
| Squeeze ATM2 | 26 | 9 | 42.3 | -2219 | -125 | -85 | -6032 | 1470 | -3590 | 0 |
| Squeeze ATM4 | 12 | 8 | 58.3 | 5800 | 388 | 483 | -2072 | 3922 | -2072 | 0 |
| 916 OTM | 15 | 15 | 100.0 | 204266 | 2576 | 13618 | 0 | 29352 | 0 | 3 |
| 916 ATM | 22 | 22 | 59.1 | 68430 | 980 | 3110 | -14701 | 51252 | -8560 | 0 |
| 916 ATM2 | 56 | 17 | 33.9 | 137126 | 501 | 2449 | -27896 | 142340 | -10736 | 0 |
| 916 ATM4 | 15 | 15 | 60.0 | 65058 | 1587 | 4337 | -19314 | 49528 | -10754 | 0 |

## Caveats

- Lot size 10→2 mid-window → RAW cumulative not comparable over time; use per-lot panel.
- Mixed paper/live; restarts; 2026-06-02 trades distorted by exit/adjustment bugs, ghost legs and MANUAL_FLATTEN_RECONCILE handled that day.
- Exit-reason mix (all systems): SL_EXIT_BOTH=71, eod_squareoff=67, EOD_SQUAREOFF=15, time_exit=11, ST_EXIT=11, SL_HIT=3, adj_boundary_exit=1, BOUNDARY_EXIT=1