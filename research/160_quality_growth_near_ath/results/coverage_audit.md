# research/160 - Coverage & survivorship audit of the fundamental leg

Universe = every `timeframe='day'` symbol in `market_data.db`, minus the 346 funds in
`backtest_data/etf_exclusions.json` and the legacy ticker regex, minus symbols with
< 250 daily rows or nothing after 2015-01-01. Screener tickers are the DB symbol with
the NSE series suffix stripped; a symbol and its `-BE` twin share one Screener page.

## 1. Coverage

| | count | share |
|---|---:|---:|
| Universe symbols | 2158 | 100% |
| With Screener fundamental history | 2131 | 98.7% |
| Without | 27 | 1.3% |
| — with >= 4 fiscal years (screenable at all) | 2126 | 98.5% |
| — with a quarterly table | 2109 | 97.7% |

Page used: consolidated 1555, standalone 576. Consolidated is preferred and standalone is read only when consolidated is absent
or thin; which page answered is recorded per symbol, because silently mixing the two
makes debt and equity inconsistent between names.

## 2. Survivorship exposure — the number, not the sentence

| | universe | Screener has it | Screener does NOT |
|---|---:|---:|---:|
| All symbols | 2158 | 2131 | 27 |
| Dead (last bar < 2026-06-01) | 102 | 101 | 1 |
| Dead AND ever tv20 >= Rs 1cr | 90 | 89 | 1 |
| Dead AND ever tv20 >= Rs 5cr | 63 | 62 | 1 |
| Alive | 2056 | 2030 | 26 |

**The expected bias did not appear, and the reason matters: Screener KEEPS the pages of
delisted companies.** Of the 102 universe names whose price series has stopped, 101 still
carry fundamental history. Exactly **1** stopped name that once traded >= Rs 5cr/day has
no page — 0.05% of the universe. On the Screener side this study is close to
survivorship-clean. That is the opposite of what r/158 assumed from a 638-name sample, and
it is worth stating plainly rather than repeating an inherited caveat that the data refutes.

**The real exposure has moved upstream, into `market_data.db` itself, where this leg cannot
measure it.** The universe carries only 102 stopped series out of 2158 (4.7%) across eleven
years — fewer than the NSE actually delisted or suspended over that period. A company that
never entered the price database is invisible to both legs AND to this audit; it cannot be
counted from inside. So the fundamental filter adds almost no survivorship of its own, and
the open question the study must carry is the price database's own coverage.

And a stopped series is not always a dead company. The one liquid name missing here is
TATAMOTORS — renames, series moves and demergers end a symbol without
ending the company (TATAMOTORS's series stops at a demerger, not a delisting). The engine
should treat a series that simply stops as a data event to inspect, not as a bankruptcy.

Largest by peak traded value:

| symbol | last bar | peak tv20 (Rs cr) |
|---|---|---:|
| TATAMOTORS | 2025-10-17 | 4145.4 |

## 3. Fiscal-year depth

| filed years on the page | symbols |
|---:|---:|
| 0 | 27 |
| 2 | 2 |
| 3 | 3 |
| 4 | 9 |
| 5 | 14 |
| 6 | 107 |
| 7 | 157 |
| 8 | 145 |
| 9 | 106 |
| 10 | 89 |
| 11 | 99 |
| 12 | 1375 |
| 13 | 25 |

Screener serves about twelve years to a signed-out reader, so the earliest fiscal year
is FY2015 for most names. **This is a hard floor on the study: four filed years — the
minimum a three-year growth rate needs — do not exist before FY2018 is filed.**

## 4. The honest study start

Share of names that were trading in that month and had ever turned over
>= Rs 5cr/day, which had four filed fiscal years and could therefore be screened:

| month | live & ever-liquid | screenable |
|---|---:|---:|
| 2015-01-01 | 875 | 4% |
| 2015-07-01 | 901 | 4% |
| 2016-01-01 | 927 | 4% |
| 2016-07-01 | 960 | 4% |
| 2017-01-01 | 991 | 4% |
| 2017-07-01 | 1021 | 5% |
| 2018-01-01 | 1074 | 6% |
| 2018-07-01 | 1107 | 7% |
| 2019-01-01 | 1126 | 87% |
| 2019-07-01 | 1146 | 86% |
| 2020-01-01 | 1170 | 90% |
| 2020-07-01 | 1179 | 89% |
| 2021-01-01 | 1213 | 91% |
| 2021-07-01 | 1246 | 90% |
| 2022-01-01 | 1309 | 92% |
| 2022-07-01 | 1347 | 91% |
| 2023-01-01 | 1386 | 95% |
| 2023-07-01 | 1408 | 94% |
| 2024-01-01 | 1479 | 97% |
| 2024-07-01 | 1523 | 96% |
| 2025-01-01 | 1600 | 99% |
| 2025-07-01 | 1659 | 98% |
| 2026-01-01 | 1691 | 99% |
| 2026-07-01 | 1629 | 99% |

**First month >= 50% screenable: 2018-08-01. First month >= 80%: 2018-08-01.**
The study should start at the 80% month; anything earlier is measuring Screener's
depth rather than the screen. Coverage before that is not random — it is whichever
companies happen to have longer pages — so an early start is a selection effect, not
merely a smaller sample.

## 5. Known data defect touching this leg: split scale

`market_data.db` is not retroactively split-adjusted: pre-split rows keep the old, higher
price scale, so `mcap_pit = shares x close` is INFLATED for any month preceding a split.
Each panel row carries `mcap_scale_suspect`, set when a one-day close collapse below
0.55x lies in that row's future.

- symbols with at least one suspect month: **65 of 2131** (3.1%)
- panel rows flagged: **4088 of 256828** (1.6%)

The flag is a candidate, not a confirmation — a genuine one-day 45%% collapse (a fraud
or a blow-up) trips it too. It is deliberately generous: the engine should run the
market-cap floor with and without the flagged rows and report the gap, in the same way
it reports the missing-data policy both ways. The share count itself is sound —
reconciliation against Screener's own market cap is within 10%% for ~95%% of names.

## 6. What the study must carry

1. **Screener coverage is NOT this study's survivorship problem.** 2131 of 2158 universe names
   have fundamental history, delisted ones included, and only 1 liquid stopped name is
   missing. Still run screened arms against an unscreened arm on the SAME sub-universe so
   the comparison is like-for-like — but the open survivorship question belongs to
   `market_data.db`'s own universe (section 2), and cannot be answered from inside it.
2. **Restated, not as-reported.** Screener shows figures as they stand today. The
   filing lag controls the timing; it cannot undo a restatement.
3. **No pre-2018-08-01 study window** — before that, coverage is the result, not the screen.
4. **Quarterly OPM is a recent-window feature** (~mid-2023 on): Screener carries about
   thirteen quarters, not a history.
5. **Market-cap floor is the softest criterion** — it depends on face value, share count
   and an unadjusted price series, where the others need only the filed statements.
