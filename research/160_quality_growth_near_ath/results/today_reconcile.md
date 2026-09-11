# research/160 — reconciliation against what screener.in shows today

Everything in the "today" column is TODAY's value with no filing lag. It exists only to
prove the point-in-time panel is wired correctly, and is never evidence about the
strategy.

## 1. How many names pass, on today's figures

2084 tickers with >= 4 fiscal years.

| criterion | passes | fails |
|---|---:|---:|
| growth | 278 | 1806 |
| roe | 623 | 1461 |
| roce | 774 | 1310 |
| de | 850 | 1234 |
| mcap | 1306 | 778 |
| no_neg | 1630 | 454 |
| all5 | 46 | 2038 |

**All five criteria on today's figures: 46 names.**
(Arun's live query adds `price >= 0.9 x all-time high`, which this table deliberately
does not apply — that is the engine leg's condition and it cuts the list further.)

Largest 25 of them: BSE, POLYCAB, HDFCAMC, PERSISTENT, MCX, COFORGE, NAM-INDIA, HINDCOPPER, CPPLUS, CUPID, ANANDRATHI, GODFRYPHLP, GRSE, RRKABEL, TDPOWERSYS, FORCEMOT, HBLENGINE, TRITURBINE, ASTRAZEN, KFINTECH, ZENTEC, PRUDENT, IGIL, DOMS, KRN

## 2. Point-in-time row for 2026-09-01 vs today

| | names |
|---|---:|
| screenable (n_fy_usable >= 4) at 2026-09-01 | 2126 |
| passing all five, point-in-time | **46** |
| passing all five, today's figures | 46 |

Overlap 45; PIT-only 1; today-only 1.
The two differ for one honest reason: on 2026-09-01 the FY2026 annuals (year-end
31-Mar-2026) are NOT yet usable under the 4-month lag — they become usable on
01-Aug-2026, so most names do have them, but any company whose latest page year is
FY2026 and whose FY2025 was weaker will differ. A large gap in either direction is
a bug; a small one is the lag doing its job.

## 3. Market-cap scale check — the 20 largest AND the 20 smallest

`equity_capital (Rs cr) / face_value (Rs)` = shares in **crores**; times a rupee price
= market cap in **Rs crore**. A lakh/crore slip anywhere would show as a ~100x error
and would switch the `mcap > 1000` criterion fully on or fully off.

**20 largest by Screener market cap**

| ticker | Screener mcap (Rs cr) | computed (Rs cr) | error |
|---|---:|---:|---:|
| RELIANCE | 1711460 | 1711798 | +0.0% |
| BHARTIARTL | 1150062 | 1123124 | -2.3% |
| HDFCBANK | 1085158 | 1083456 | -0.2% |
| ICICIBANK | 995439 | 993092 | -0.2% |
| SBIN | 919831 | 919308 | -0.1% |
| TCS | 799995 | 800382 | +0.0% |
| BAJFINANCE | 642768 | 641904 | -0.1% |
| LT | 541500 | 541200 | -0.1% |
| LICI | 509732 | 254898 | -50.0% |
| HINDUNILVR | 455821 | 455900 | +0.0% |
| SUNPHARMA | 441274 | 441360 | +0.0% |
| TITAN | 441163 | 442241 | +0.2% |
| INFY | 421245 | 420182 | -0.3% |
| KOTAKBANK | 415514 | 415910 | +0.1% |
| ADANIENT | 413202 | 393837 | -4.7% |
| ADANIPORTS | 407490 | 407754 | +0.1% |
| ADANIPOWER | 406232 | 406914 | +0.2% |
| MARUTI | 390645 | 390145 | -0.1% |
| AXISBANK | 389016 | 388750 | -0.1% |
| M&M | 387707 | 348592 | -10.1% |

**20 smallest by Screener market cap**

| ticker | Screener mcap (Rs cr) | computed (Rs cr) | error |
|---|---:|---:|---:|
| SHANTI | 7 | 7 | -0.2% |
| FLEXITUFF | 7 | 7 | +0.6% |
| PREMIER | 8 | 8 | +0.5% |
| GATECHDVR | 9 | 59 | +581.8% |
| ARENTERP | 10 | 10 | +1.5% |
| TVVISION | 11 | 11 | +0.2% |
| DCMFINSERV | 11 | 11 | +0.2% |
| ANTGRAPHIC | 12 | 12 | +0.2% |
| BLUECHIP | 12 | 12 | +0.1% |
| EUROTEXIND | 12 | 12 | -0.5% |
| GLOBALE | 12 | 12 | -0.2% |
| UNIINFO | 12 | 12 | -0.4% |
| CREATIVEYE | 12 | 11 | -5.9% |
| SHYAMTEL | 13 | 13 | -1.9% |
| BILVYAPAR | 13 | 13 | -0.1% |
| UMESLTD | 13 | 13 | -0.6% |
| ASTRON | 14 | 13 | -0.8% |
| GLFL | 14 | 14 | +1.3% |
| MTEDUCARE | 14 | 14 | +0.2% |
| ORTINGLOBE | 14 | 14 | +0.3% |

Across all 2083 reconcilable names: median |error| 0.72%, 95.6% within 10%, 58.6% within 1%. Worst offenders: GATECHDVR +582%, WEL +116%, GENUSPAPER +90%, ANURAS +28%, JSWSTEEL +25%.

A handful of large |errors| is expected and is not a unit bug: companies with a second
listed class (DVRs), partly-paid shares, or an equity issue after the last balance
sheet date all move the share count off the equity-capital route. A UNIT error would
show as ~100x on every name, not on a few.

## 4. Computed ratios vs Screener's published ones

- **ROE (3-yr avg here, latest there)**: 2011 comparable, median |difference| 2.6 pp, 70% within 5 pp
- **ROCE (latest FY both)**: 1906 comparable, median |difference| 0.2 pp, 100% within 5 pp

ROE is expected to differ: this panel averages three years, Screener publishes the
latest. ROCE is taken from Screener's own row, so it should agree almost exactly —
any material gap there means the wrong row is being read.

