# G4 Portfolio Construction — Gap-Up + ORB Long Combined Book

STATUS: DONE — BORDERLINE (see Verdict)

EXP-G4 of research/81. Construction: index sleeve (NIFTY W12, 1.0% risk/trade)
+ stock sleeve (9 names W6, 0.5% risk/trade per G3 MC), daily MTM NAV
2015-02..2023-12 (IS+Val; OOS locked), charges deducted at exit.
Two construction variants tried (logged): cap 6 vs cap 10 concurrent.

## VERDICT (2026-07-15 ~23:40 IST)

LOCKED construction = cap 6 (cap 10 raised CAGR 13.2->14.0% but widened DD
-17.2->-20.5% and cut Calmar 0.77->0.68 — clustered long risk; rejected).

| Metric | Book | NIFTY B&H | Brief gate | Pass? |
|---|---|---|---|---|
| CAGR | 13.2% | 10.7% | — | — |
| Sharpe | 1.00 | 0.70 | >=1.0 | borderline PASS |
| Sortino | 1.60 | — | — | — |
| MaxDD | -17.2% | -38.2% | <=20% | PASS |
| Calmar | 0.77 | 0.28 | >=1.0 | FAIL |
| DD duration | ~16 mo (2016-18 flat) | — | <=9 mo | FAIL |
| Corr vs NIFTY | 0.58 | — | — | long-only, expected |

Only negative year 2016 (-5.3%). Reading: a REAL, robust long sleeve that
beats buy-and-hold on every axis but is too monoline to clear the Calmar/
DD-duration gates alone. Diversification (uncorrelated short-side sleeve,
e.g. family-B deep-z fade, or breadth from the broad universe post-backfill)
is the designed next lever — NOT more parameter tuning.
