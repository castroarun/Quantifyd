# research/160 - Arun's own holdings vs the mechanical screen

Source: `backtest_data/holdings_snapshots.db`, 105 daily snapshots, 2026-04-20 to 2026-09-11, read-only.
81 distinct symbols ever held: 69 equities in the universe, 2 funds/ETFs, 10 not in
the price database at all.

**The two cohorts are not the same evidence.** 26 names were already held when the
snapshot history begins on 2026-04-20, so for them the screen is being asked "does this pass
today", not "would it have been picked on purchase". 43 names first appear later; the
snapshot runs daily, so those first appearances ARE purchase days and they are the
honest sample.

## Hit rate of the mechanical screen on his real picks

| cohort | n | has fundamental data | passes the full screen | passes screen AND near-ATH |
|---|---:|---:|---:|---:|
| Observed buys (2026-04-21 onward) | 43 | 43 (100%) | **3 (7%)** | 2 (5%) |
| Pre-existing at 2026-04-20 | 26 | 26 (100%) | **5 (19%)** | 3 (12%) |
| All equities held | 69 | 69 (100%) | **8 (12%)** | 5 (7%) |

## Where the screen and the man disagree

| criterion | of 69 equities held: pass | fail | n/a (lender or no data) |
|---|---:|---:|---:|
| growth | 19 | 50 | 0 |
| roe | 43 | 26 | 0 |
| roce | 47 | 18 | 4 |
| de | 36 | 33 | 0 |
| mcap | 68 | 1 | 0 |
| no_neg | 64 | 5 | 0 |

Most common reason a held name fails: growth 50, de 33, roe 26, roce 18, no_neg 5, mcap 1.

## Which dial rejects his picks

A low hit rate is only useful if you know what would fix it. Same 69 equities, same
evaluation dates, one criterion relaxed at a time:

| variant | of 69 held equities, pass |
|---|---:|
| the full screen as written | 8 |
| growth bar lowered to 15% | 19 |
| growth bar lowered to 10% | 20 |
| growth on SALES only (profit growth dropped) | 10 |
| growth dropped entirely | 27 |
| debt/equity dropped | 14 |
| ROE dropped | 11 |
| ROCE dropped | 8 |
| market-cap floor dropped | 8 |
| growth AND debt/equity both dropped | 40 |
| near-ATH condition alone (no fundamentals) | 42 |

Read it as a description of his process, not a scoring of it: he is buying names that are
near their highs and profitable, without insisting on 20%+ growth on BOTH the top and
bottom line, and without the debt-free constraint. Those two dials are what separate the
mechanical screen from the book.

## Every equity held, at its evaluation date

| symbol | cohort | eval date | n FY | sales g3 | profit g3 | ROE3 | ROCE | D/E | mcap Rs cr | % from ATH | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DYCL | observed buy | 2026-09-04 | 12 | 21.43 | 39.4 | 17.84 | 26.0 | 0.09 | 2603.52 | -10.4 | PASS (not near ATH) |
| RPEL | observed buy | 2026-07-23 | 5 | 25.99 | 27.14 | 18.11 | 26.0 | 0.04 | 5727.92 | -1.8 | PASS |
| SILVERTUC | observed buy | 2026-08-14 | 12 | 27.75 | 53.25 | 17.29 | 30.0 | 0.2 | 2522.38 | -0.3 | PASS |
| ADANIENSOL | observed buy | 2026-08-11 | 12 | 27.55 | 23.15 | 7.68 | 10.0 | 1.93 | 198333.14 | -60.5 | fails roe,roce,de |
| ADANIPOWER | observed buy | 2026-08-11 | 12 | 11.84 | 6.54 | 30.39 | 17.0 | 0.84 | 407723.47 | -15.9 | fails growth,de |
| ANTHEM | observed buy | 2026-08-05 | 7 | 26.18 | 15.42 | 19.08 | 30.0 | 0.02 | 45315.2 | -4.5 | fails growth |
| AUROPHARMA | observed buy | 2026-09-02 | 12 | 10.63 | 22.02 | 10.18 | 13.0 | 0.21 | 97631.4 | -4.0 | fails growth,roe,roce,de |
| AYMSYNTEX | observed buy | 2026-07-02 | 5 | 10.38 | nan | 1.41 | 8.0 | 0.35 | 1437.99 | -13.7 | fails growth,roe,roce,de |
| BHEL | observed buy | 2026-08-11 | 12 | 13.07 | 34.74 | 3.14 | 9.0 | 0.31 | 141549.0 | -7.8 | fails growth,roe,roce,de |
| DIVGIITTS | observed buy | 2026-07-02 | 8 | -2.18 | -19.49 | 6.73 | 6.0 | 0.0 | 2963.55 | -12.5 | fails growth,roe,roce |
| EBGNG | observed buy | 2026-09-08 | 7 | 41.95 | 58.72 | 26.61 | 20.0 | 0.57 | 7480.75 | -1.4 | fails de |
| ENTERO | observed buy | 2026-09-07 | 7 | 25.93 | nan | 5.76 | 10.0 | 0.4 | 7942.0 | 0.0 | fails growth,roe,roce,de |
| FIEMIND | observed buy | 2026-08-12 | 12 | 15.07 | 22.28 | 19.84 | 29.0 | 0.05 | 6040.58 | 0.0 | fails growth |
| GVT&D | observed buy | 2026-07-14 | 11 | 11.86 | nan | 16.25 | 55.0 | 0.02 | 124950.0 | -18.8 | fails growth,no_neg |
| IDEA | observed buy | 2026-09-01 | 12 | 2.09 | nan | -9.23 | -2.0 | nan | 152330.26 | -88.5 | fails growth,roe,roce,de,no_neg |
| INDNIPPON | observed buy | 2026-06-12 | 11 | 14.29 | 17.92 | 9.86 | 15.0 | 0.0 | 1962.51 | -17.6 | fails growth,roe,roce |
| INDSWFTLAB | observed buy | 2026-09-07 | 12 | -19.02 | -5.12 | 23.17 | 5.0 | 0.01 | 3202.91 | -1.1 | fails growth,roce |
| INOXINDIA | observed buy | 2026-08-18 | 8 | 17.99 | 18.51 | 26.39 | 33.0 | 0.07 | 17172.0 | -6.5 | fails growth |
| IOLCP | observed buy | 2026-09-07 | 12 | 1.51 | -0.48 | 7.34 | 11.0 | 0.07 | 5619.16 | 0.0 | fails growth,roe,roce |
| IRISDOREME | observed buy | 2026-09-07 | 12 | 18.84 | 25.99 | 14.84 | 16.0 | 0.24 | 1099.15 | -2.2 | fails growth,roe,de |
| KTKBANK | observed buy | 2026-09-07 | 7 | 7.29 | 3.57 | 10.83 | nan | nan | 12508.02 | -1.9 | fails growth,roe,de |
| LAURUSLABS | observed buy | 2026-08-11 | 12 | 4.09 | 3.92 | 9.58 | 18.0 | 0.48 | 98047.8 | -0.5 | fails growth,roe,de |
| LUMAXIND | observed buy | 2026-08-12 | 11 | 21.72 | 18.64 | 17.8 | 18.0 | 1.06 | 4756.05 | -11.0 | fails growth,de |
| MANINDS | observed buy | 2026-09-07 | 12 | 16.9 | 36.38 | 8.38 | 16.0 | 0.3 | 5796.9 | 0.0 | fails growth,roe,de |
| MANORAMA | observed buy | 2026-08-19 | 12 | 56.97 | 98.01 | 23.21 | 35.0 | 0.51 | 9627.0 | -0.4 | fails de |
| NACLIND | observed buy | 2026-07-02 | 11 | -8.89 | nan | -5.51 | -8.0 | 0.93 | 4447.8 | -33.6 | fails growth,roe,roce,de,no_neg |
| NATIONALUM | observed buy | 2026-08-11 | 12 | 7.76 | 59.25 | 23.41 | 40.0 | 0.0 | 63957.06 | -12.8 | fails growth |
| NITINSPIN | observed buy | 2026-09-07 | 12 | 10.09 | 2.56 | 12.32 | 12.0 | 0.76 | 3532.2 | -2.0 | fails growth,roe,roce,de |
| OFSS | observed buy | 2026-07-01 | 11 | 9.46 | 8.0 | 26.97 | 41.0 | 0.01 | 93396.0 | -13.4 | fails growth |
| PAISALO | observed buy | 2026-07-02 | 11 | 24.91 | 36.28 | 11.52 | nan | nan | 6375.6 | -24.7 | fails roe,de |
| POWERINDIA | observed buy | 2026-08-11 | 8 | 22.16 | 119.01 | 13.42 | 29.0 | 0.02 | 144697.5 | -9.2 | fails roe |
| PRECOT | observed buy | 2026-07-02 | 11 | -4.38 | -32.0 | 1.68 | 13.0 | 0.73 | 881.58 | -15.2 | fails growth,roe,roce,de,mcap,no_neg |
| PRICOLLTD | observed buy | 2026-09-08 | 10 | 27.29 | 26.15 | 17.71 | 24.0 | 0.3 | 9564.0 | -5.9 | fails de |
| RADICO | observed buy | 2026-08-11 | 12 | 24.43 | 40.01 | 13.84 | 24.0 | 0.15 | 58657.5 | -1.0 | fails roe |
| SBCL | observed buy | 2026-09-07 | 12 | 6.7 | 6.71 | 21.15 | 27.0 | 0.15 | 6287.4 | -5.5 | fails growth |
| SETL | observed buy | 2026-09-07 | 7 | 7.24 | 27.03 | 6.36 | 9.0 | 0.05 | 7702.3 | -5.0 | fails growth,roe,roce |
| SHILPAMED | observed buy | 2026-09-07 | 12 | 13.64 | nan | 4.81 | 11.0 | 0.25 | 17794.0 | -2.6 | fails growth,roe,roce,de |
| SMSPHARMA | observed buy | 2026-07-02 | 11 | 14.61 | 3.63 | 6.2 | 13.0 | 0.49 | 3680.55 | -4.7 | fails growth,roe,roce,de,no_neg |
| SPORTKING | observed buy | 2026-09-07 | 12 | 4.22 | -3.13 | 10.73 | 13.0 | 0.53 | 2988.83 | -3.8 | fails growth,roe,roce,de |
| SSWL | observed buy | 2026-09-07 | 12 | 8.65 | 1.36 | 13.86 | 16.0 | 0.48 | 5716.0 | 0.0 | fails growth,roe,de |
| THYROCARE | observed buy | 2026-08-26 | 12 | 16.3 | 36.55 | 19.2 | 35.0 | 0.09 | 9206.1 | -6.8 | fails growth |
| TMB | observed buy | 2026-09-07 | 9 | 12.55 | 9.15 | 13.3 | nan | nan | 14347.98 | -0.6 | fails growth,roe,de |
| WELCORP | observed buy | 2026-09-07 | 12 | 19.78 | 101.13 | 21.18 | 23.0 | 0.26 | 67415.04 | -1.7 | fails growth,de |
| ANANDRATHI | pre-existing | 2026-04-20 | 7 | 32.21 | 33.32 | 38.53 | 56.0 | 0.12 | 13081.32 | -0.7 | PASS |
| BSE | pre-existing | 2026-04-20 | 11 | 56.3 | 75.37 | 20.3 | 47.0 | 0.0 | 38712.6 | -1.7 | PASS |
| PERSISTENT | pre-existing | 2026-04-20 | 11 | 27.86 | 26.59 | 22.48 | 31.0 | 0.05 | 78765.96 | -20.2 | PASS (not near ATH) |
| POLYCAB | pre-existing | 2026-04-20 | 11 | 22.45 | 30.66 | 20.73 | 30.0 | 0.02 | 103875.0 | -4.8 | PASS |
| PRUDENT | pre-existing | 2026-04-20 | 7 | 35.24 | 34.8 | 30.54 | 44.0 | 0.05 | 9503.34 | -22.9 | PASS (not near ATH) |
| APLAPOLLO | pre-existing | 2026-04-20 | 11 | 16.56 | 6.94 | 19.88 | 22.0 | 0.15 | 54174.4 | -6.0 | fails growth |
| BANCOINDIA | pre-existing | 2026-04-20 | 11 | 17.95 | 37.12 | 26.48 | 32.0 | 0.44 | 7824.93 | -29.9 | fails growth,de |
| BEL | pre-existing | 2026-04-20 | 11 | 15.64 | 30.4 | 24.2 | 39.0 | 0.0 | 306069.7 | -2.3 | fails growth |
| CAMS | pre-existing | 2026-04-20 | 8 | 16.04 | 17.45 | 38.81 | 54.0 | 0.08 | 16056.08 | -29.0 | fails growth |
| CDSL | pre-existing | 2026-04-20 | 11 | 25.22 | 19.01 | 27.12 | 42.0 | 0.0 | 24720.52 | -30.2 | fails growth |
| COALINDIA | pre-existing | 2026-04-20 | 11 | 15.53 | 26.82 | 44.05 | 48.0 | 0.09 | 276965.22 | -6.0 | fails growth |
| DATAPATTNS | pre-existing | 2026-04-20 | 11 | 31.54 | 33.16 | 13.03 | 21.0 | 0.0 | 17091.8 | -0.8 | fails roe |
| GUJTHEM | pre-existing | 2026-04-20 | 11 | 9.5 | 3.65 | 29.35 | 27.0 | 0.12 | 2717.22 | -29.1 | fails growth |
| HAL | pre-existing | 2026-04-20 | 11 | 7.96 | 18.08 | 24.93 | 34.0 | 0.0 | 245209.44 | -22.7 | fails growth |
| HDFCAMC | pre-existing | 2026-04-20 | 11 | 18.57 | 20.88 | 27.02 | 43.0 | 0.0 | 50086.7 | -6.0 | fails growth |
| KEI | pre-existing | 2026-04-20 | 11 | 19.34 | 22.78 | 16.3 | 21.0 | 0.04 | 39280.6 | -6.3 | fails growth |
| KMEW | pre-existing | 2026-04-20 | 8 | 48.79 | 33.52 | 25.74 | 25.0 | 0.61 | 3471.82 | -11.8 | fails de |
| LUMAXTECH | pre-existing | 2026-04-20 | 11 | 34.1 | 40.81 | 20.8 | 19.0 | 0.96 | 11154.5 | -2.8 | fails de |
| NMDC | pre-existing | 2026-04-20 | 11 | -2.72 | -11.57 | 22.8 | 30.0 | 0.14 | 68711.43 | -2.2 | fails growth |
| NUVAMA | pre-existing | 2026-04-20 | 6 | 32.77 | 4.75 | 21.12 | 20.0 | 2.25 | 20500.2 | -16.8 | fails growth,de |
| POWERGRID | pre-existing | 2026-04-20 | 11 | 3.23 | -2.65 | 17.73 | 13.0 | 1.41 | 272333.28 | -12.5 | fails growth,roce,de |
| RECLTD | pre-existing | 2026-04-20 | 11 | 12.8 | 16.53 | 20.01 | nan | nan | 84808.93 | -40.7 | fails growth,de |
| SHRIPISTON | pre-existing | 2026-04-20 | 11 | 19.79 | 46.52 | 21.2 | 26.0 | 0.21 | 13310.88 | -2.9 | fails growth,de |
| SOLARINDS | pre-existing | 2026-04-20 | 11 | 24.06 | 41.45 | 28.91 | 38.0 | 0.22 | 115263.0 | -15.3 | fails de |
| SWARAJENG | pre-existing | 2026-04-20 | 11 | 13.91 | 15.05 | 38.77 | 56.0 | 0.0 | 3995.64 | -12.4 | fails growth |
| TDPOWERSYS | pre-existing | 2026-04-20 | 11 | 17.07 | 35.71 | 17.72 | 30.0 | 0.01 | 13370.3 | 0.0 | fails growth |

**Sensitivity, pre-existing cohort at the avg-cost-implied purchase date** (estimate, not a record): 3 of 26 pass the screen there vs 5 of 26 at 2026-04-20. A position built in tranches makes that date a blur; treat it as a direction, not a number.

