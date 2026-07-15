# Phase 0 Data Audit — market_data.db (research/81)

Generated: 2026-07-15T18:50:25 IST · DB: `/home/arun/quantifyd/backtest_data/market_data.db` · read-only snapshot **while 5-min history backfill is running** (broad-universe 5-min coverage will deepen; deep/daily series audited here are final).

## A. Inventory

| Timeframe | Symbols | First bar | Last bar | Rows |
|---|---|---|---|---|
| 30minute | 50 | 2020-01-01 09:15:00 | 2026-05-05 15:15:00 | 44,480 |
| 5minute | 381 | 2015-02-02 09:15:00 | 2026-07-15 15:25:00 | 15,085,677 |
| 60minute | 95 | 2018-01-01 09:15:00 | 2026-03-19 14:15:00 | 1,189,675 |
| day | 1642 | 2000-01-03 | 2026-07-15 | 3,520,851 |

5-min start-year histogram (per-symbol earliest bar):

| Start year | Symbols |
|---|---|
| 2015 | 8 |
| 2018 | 4 |
| 2024 | 369 |

## B. Integrity

| Timeframe | Dup (sym,tf,date) | H < max(O,C) | L > min(O,C) | H < L | Price ≤ 0 | Zero-vol rows (equity day) |
|---|---|---|---|---|---|---|
| day | 0 | 63 | 55 | 1 | 10 | 84113 |
| 60minute | 0 | 3 | 3 | 0 | 35 |  |
| 5minute | 0 | 36 | 278 | 2 | 2550 |  |
| 30minute | 0 | 0 | 0 | 0 | 0 |  |

## C. Daily session coverage (vs consensus calendar)

Consensus sessions (≥50 symbols traded): **5,326** (2005-01-03 → 2026-07-15)
Symbols with ≥250 expected sessions: 1,470; of those, **22 have <95% session coverage** (gaps within their own span).

| Symbol | Span | Have | Expected | Coverage |
|---|---|---|---|---|
| FUSION | 2016-02-22→2026-05-15 | 991 | 2533 | 39.1% |
| COALINDIA | 2010-11-04→2026-07-15 | 1537 | 3882 | 39.6% |
| DELHIVERY | 2016-01-18→2026-07-15 | 1036 | 2598 | 39.9% |
| STARHEALTH | 2016-01-04→2026-05-15 | 1159 | 2567 | 45.1% |
| COHANCE | 2015-01-01→2026-05-15 | 1554 | 2815 | 55.2% |
| ONGC | 2005-01-03→2026-07-15 | 2956 | 5326 | 55.5% |
| SBICARD | 2015-01-01→2026-07-15 | 1591 | 2856 | 55.7% |
| RAINBOW | 2019-04-01→2026-05-15 | 1036 | 1766 | 58.7% |
| LATENTVIEW | 2019-02-22→2026-05-15 | 1127 | 1790 | 63.0% |
| MAZDOCK | 2017-12-04→2026-07-15 | 1436 | 2134 | 67.3% |
| SOLEX | 2018-02-05→2026-05-15 | 1380 | 2050 | 67.3% |
| KSHITIJPOL | 2018-10-08→2026-02-17 | 1242 | 1828 | 67.9% |
| ANGELONE | 2018-01-08→2026-07-15 | 1448 | 2110 | 68.6% |
| HOMEFIRST | 2018-03-21→2026-07-15 | 1439 | 2061 | 69.8% |
| JAIPURKURT | 2017-10-11→2026-05-15 | 1531 | 2130 | 71.9% |
| SERVOTECH | 2017-10-11→2026-05-15 | 1549 | 2130 | 72.7% |
| ZODIAC | 2017-12-05→2026-02-17 | 1596 | 2035 | 78.4% |
| SHEKHAWATI | 2011-01-12→2026-02-17 | 3022 | 3737 | 80.9% |
| GATECH | 2023-07-10→2026-05-15 | 614 | 708 | 86.7% |
| DIGJAMLMTD | 2016-07-13→2026-05-15 | 2133 | 2438 | 87.5% |

## D. 5-min bar structure — deep-history names

NSE regular session 09:15–15:30 → 75 five-min bars (09:15…15:25).

| Symbol | Sessions | Median bars/sess | Sess <70 bars | Sess <30 bars (half-day/muhurat?) | Bars outside 09:15–15:29 |
|---|---|---|---|---|---|
| NIFTY50 | 2,706 | 75 | 17 | 13 | 118 |
| INDIAVIX | 2,813 | 75 | 15 | 14 | 115 |
| BANKNIFTY | 456 | 75 | 6 | 3 | 12 |
| HDFCBANK | 2,496 | 75 | 16 | 11 | 96 |
| ICICIBANK | 2,455 | 75 | 16 | 11 | 96 |
| RELIANCE | 2,496 | 75 | 17 | 11 | 96 |
| INFY | 2,455 | 75 | 16 | 11 | 96 |
| TCS | 2,455 | 75 | 16 | 11 | 96 |
| SBIN | 2,455 | 75 | 16 | 11 | 96 |
| ITC | 2,496 | 75 | 16 | 11 | 96 |
| HINDUNILVR | 2,496 | 75 | 16 | 11 | 96 |
| KOTAKBANK | 1,825 | 75 | 13 | 9 | 72 |
| BHARTIARTL | 1,825 | 75 | 13 | 9 | 72 |

## E. Corporate-action suspects — overnight |gap| > 25% (daily, equities)

Rows scanned: 3,506,487 · overnight |gap|>25% events (prev close >₹5): **321** across **182** symbols.

| Symbol | >25% gap events | Worst gap |
|---|---|---|
| GNFC | 25 | 61% |
| HINDPETRO | 16 | 42% |
| KINGFA | 12 | 227% |
| YESBANK | 11 | 58% |
| VEDL | 8 | 65% |
| ADANIENT | 5 | 90% |
| IDEA | 4 | 27% |
| JINDALSTEL | 4 | 29% |
| INDIAMART | 4 | 104% |
| ZEEL | 4 | 40% |
| PCJEWELLER | 4 | 44% |
| MPHASIS | 4 | 39% |
| IIFL | 3 | 33% |
| LICI | 3 | 105% |
| IOC | 3 | 36% |
| IMAGICAA | 3 | 97% |
| MFSL | 3 | 79% |
| ASHOKLEY | 3 | 28% |
| CUPID | 3 | 402% |
| OFSS | 3 | 36% |
| SAMMAANCAP | 3 | 34% |
| UEL | 3 | 268% |
| SUZLON | 3 | 40% |
| SUMEETINDS | 3 | 419% |
| INFIBEAM | 2 | 71% |

Interpretation: these are unadjusted splits/bonuses OR genuine crashes/circuits OR illiquid junk. Any symbol used by a strategy must be checked/adjusted; headline results prefer the F&O-liquid subset where Kite adjusts splits.

## F. Backfill splice consistency (new 2015–24 history vs pre-existing rows)

Backfilled symbols checked so far: 9 · **splice jumps >15%: 1**

| Symbol | Boundary | Close (new hist) | Open (old rows) | Jump |
|---|---|---|---|---|
| KOTAKBANK | 2018-01-01 | 201.9 | 1011.8 | 401% |

**Action:** these symbols split/bonused after their original download; their pre-existing rows are on the OLD adjustment basis. Fix = delete + full re-download of the symbol's 5-min series (idempotent downloader).

_Re-run this audit after the backfill completes to cover all symbols._

## G. Survivorship, sessions & standing caveats

- **Survivorship:** the 5-min universe is ~381 of *today's* liquid names — survivorship-biased for any pre-2024 cross-sectional claim. Daily table (1,642 symbols incl. many delisted/inactive) is broader. Every report must state this; headline claims prefer NIFTY/BANKNIFTY + F&O-liquid large-caps.
- **Futures:** no futures series exist. Per user decision (2026-07-15), cash/index series are the futures proxy with a futures-style cost model. Basis risk noted.
- **Sessions:** counts <75 bars are early closes/outages; sessions <30 bars are usually muhurat (evening, ~1h) — strategies must not treat them as regular days.
- **60-min table:** stale (ends 2026-03) and thin (95 syms) — the study aggregates 5-min → 15/30/60-min itself; daily table covers pre-2015.

_Audit runtime: 181s_
