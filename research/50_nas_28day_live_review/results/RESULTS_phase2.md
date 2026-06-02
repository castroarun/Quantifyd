# NAS 8-System — Phase 2: chain-re-priced P&L

Closed legs re-priced at REAL recorded option_chain premiums (entry+exit), within 10-min tolerance.

**Recorded-table net ₹-17,170,104 → re-priced net ₹1,971.**

| System | Legs | Repriced | CovPct | Recorded | RepricedNet | ReprWin |
|---|---|---|---|---|---|---|
| Squeeze OTM | 68 | 65 | 96.0 | -17627 | -15549 | 49.0 |
| Squeeze ATM | 45 | 44 | 98.0 | 2238 | 2613 | 41.0 |
| Squeeze ATM2 | 52 | 52 | 100.0 | 1941 | 4321 | 50.0 |
| Squeeze ATM4 | 35 | 34 | 97.0 | 30786 | 34700 | 38.0 |
| 916 OTM | 50 | 48 | 96.0 | -17435314 | 10419 | 69.0 |
| 916 ATM | 46 | 45 | 98.0 | 73383 | -7087 | 38.0 |
| 916 ATM2 | 112 | 110 | 98.0 | 146029 | 14687 | 44.0 |
| 916 ATM4 | 46 | 45 | 98.0 | 28460 | -42132 | 36.0 |

**Read:** where re-priced << recorded, the recorder over-stated P&L (esp. OTM exit=0).
Low coverage % = legs at strikes/times outside the recorded chain window (excluded, not zero).
Still a 25-day single-regime sample — signal/audit, not validation.