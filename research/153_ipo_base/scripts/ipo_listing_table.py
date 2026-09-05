"""research/153 Phase 0d — build the VETTED listing table used by the sweep.

Three problems the raw "first row in market_data_unified" proxy has, all measured in
Phase 0a-0c and each handled here:

  (1) COVERAGE FLOOR + BULK ONBOARDING WAVES. 451 symbols begin on 2005-01-03, 95 on
      2015-01-01, 45 on 2026-08-17, 41 on 2026-04-20, 15 on 2025-05-26 (ABB, 360ONE...).
      These are download waves, not listings. -> reject any symbol whose start day is
      shared by >=8 symbols (real waves are 12-451; genuine multi-IPO days are 2-6).
  (2) PRE-LISTING JUNK ROWS. DELHIVERY carries 8 rows at ~Rs 5-11 from 2016 before its
      real 2022-05-24 listing at Rs 536 (a different instrument on the same ticker).
      -> strip leading rows using a price-jump AND a date-gap AND a dust-volume rule.
  (3) A REAL LISTING LOOKS LIKE ONE. Day-1 volume is typically many times the next 20
      days' median (known-IPO median 15x) and the day-1 range is wide.
      -> accept if vol_ratio >= 1.5 OR day-1 range >= 8%.

Output: results/listing_dates.csv with columns
  symbol, first_row, junk_rows, list_date, last_row, n_rows, bulk_n, vol_ratio,
  day1_range_pct, accepted
`list_date` is also the date before which ALL price rows are masked in the sweep.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
DB = Path("/home/arun/quantifyd/backtest_data/market_data.db")
if not DB.exists():
    DB = Path(__file__).resolve().parents[3] / "backtest_data" / "market_data.db"
con = sqlite3.connect(str(DB))

cov = pd.read_sql_query(
    "select symbol, count(*) n, min(date) d0, max(date) d1 from market_data_unified "
    "where timeframe='day' group by symbol", con)
cov['d0'] = pd.to_datetime(cov['d0'].str[:10])
cov['d1'] = pd.to_datetime(cov['d1'].str[:10])

KNOWN_IPO = {  # public record, mainboard NSE
    "DMART": "2017-03-21", "HDFCLIFE": "2017-11-17", "IEX": "2017-10-26",
    "BANDHANBNK": "2018-03-27", "IRCTC": "2019-10-14", "SBICARD": "2020-03-16",
    "HAPPSTMNDS": "2020-09-17", "MAZDOCK": "2020-10-12", "CAMS": "2020-10-05",
    "GLAND": "2020-11-20", "RBA": "2020-12-14", "INDIGOPNTS": "2021-02-02",
    "NUVOCO": "2021-08-23", "CHEMPLASTS": "2021-08-24", "NYKAA": "2021-11-10",
    "PAYTM": "2021-11-18", "POLICYBZR": "2021-11-15", "LATENTVIEW": "2021-11-23",
    "STARHEALTH": "2021-12-10", "MAPMYINDIA": "2021-12-21", "AWL": "2022-02-08",
    "CAMPUS": "2022-05-09", "LICI": "2022-05-17", "DELHIVERY": "2022-05-24",
    "FUSION": "2022-11-15", "MEDANTA": "2022-11-18", "MANKIND": "2023-05-09",
    "IDEAFORGE": "2023-07-07", "JIOFIN": "2023-08-21", "IREDA": "2023-11-29",
    "TATATECH": "2023-11-30", "BHARTIHEXA": "2024-04-12", "OLAELEC": "2024-08-09",
    "BAJAJHFL": "2024-09-16", "HYUNDAI": "2024-10-22", "SWIGGY": "2024-11-13",
    "NTPCGREEN": "2024-11-27", "VMM": "2024-12-18", "HEXT": "2025-02-19",
    "ACMESOLAR": "2024-11-13", "SAGILITY": "2024-11-12", "NIVABUPA": "2024-11-14",
    "PREMIERENE": "2024-09-03", "AJAXENGG": "2025-02-17", "POLYCAB": "2019-04-16",
    "LTTS": "2016-09-23", "PERSISTENT": "2010-04-06", "ASTRAL": "2007-03-20",
}
# unambiguously long-listed before 2005 => any post-2005 "first row" is an ONBOARDING
KNOWN_ONBOARD = ["ABB", "SIEMENS", "CUMMINSIND", "HAVELLS", "PIDILITIND", "BOSCHLTD",
                 "SUPREMEIND", "COFORGE", "MPHASIS", "OFSS", "TATAELXSI", "360ONE"]

vc = cov['d0'].value_counts()

rows = []
for s, n, d0, d1 in cov[['symbol', 'n', 'd0', 'd1']].itertuples(index=False):
    if n < 25:
        continue
    d = pd.read_sql_query(
        "select date, open, high, low, close, volume from market_data_unified "
        "where symbol=? and timeframe='day' order by date limit 300", con, params=(s,))
    dt = pd.to_datetime(d['date'].str[:10])
    c = d['close'].astype(float).values
    v = d['volume'].astype(float).values
    W = min(250, len(c) - 1)
    j = 0
    if W > 2:
        # (a) price jump >3x or <1/3 -- a different instrument on the same ticker
        r = c[1:W + 1] / np.maximum(c[:W], 1e-9)
        bad = np.nonzero((r > 3) | (r < 1 / 3.0))[0]
        if len(bad):
            j = max(j, int(bad[-1]) + 1)
        # (b) date gap > 30 calendar days -- placeholder rows, not a traded series
        gaps = (dt.values[1:W + 1] - dt.values[:W]).astype('timedelta64[D]').astype(int)
        bg = np.nonzero(gaps > 30)[0]
        if len(bg):
            j = max(j, int(bg[-1]) + 1)
        # (c) absolute dust volume (<5,000 shares) in the first 60 rows
        Wd = min(60, W)
        dust = np.nonzero(v[:Wd] < 5000)[0]
        if len(dust):
            j = max(j, int(dust[-1]) + 1)
    j = min(j, len(d) - 5) if len(d) > 5 else 0
    start = pd.Timestamp(str(d['date'].iloc[j])[:10])
    v0 = v[j]
    nxt = v[j + 1:j + 21]
    vratio = v0 / max(np.median(nxt), 1) if len(nxt) else np.nan
    rng0 = 100 * (d['high'].iloc[j] - d['low'].iloc[j]) / max(d['close'].iloc[j], 1e-9)
    rows.append(dict(symbol=s, first_row=str(d0.date()), junk_rows=int(j),
                     list_date=str(start.date()), last_row=str(d1.date()), n_rows=int(n),
                     bulk_n=int(vc.get(start, 0)), vol_ratio=float(vratio),
                     day1_range_pct=float(rng0)))
ld = pd.DataFrame(rows)
ld['accepted'] = ((ld.bulk_n < 8) & (ld.list_date > '2006-01-01')
                  & ((ld.vol_ratio >= 1.5) | (ld.day1_range_pct >= 8.0)))
ld.to_csv(RES / "listing_dates.csv", index=False)

print(f"profiled {len(ld)} symbols; ACCEPTED as real listings: {int(ld.accepted.sum())}")
print("\naccepted per year:")
ld['ly'] = ld.list_date.str[:4]
for y, k in ld[ld.accepted].groupby('ly').size().items():
    print(f"  {y}: {k:4d}")

print("\n--- RECALL: known IPOs ---")
hit = miss = 0
datehit = 0
for s, kd in sorted(KNOWN_IPO.items()):
    row = ld[ld.symbol == s]
    if not len(row):
        print(f"  {s:<12} ABSENT from DB")
        continue
    r = row.iloc[0]
    dd = (pd.Timestamp(r.list_date) - pd.Timestamp(kd)).days
    ok = bool(r.accepted)
    hit += ok
    miss += (not ok)
    datehit += abs(dd) <= 3
    flag = "OK " if ok else "REJ"
    print(f"  {s:<12} known {kd}  vetted {r.list_date} ({dd:+4d}d) junk={r.junk_rows:3d} "
          f"bulk={r.bulk_n:2d} vr={r.vol_ratio:7.2f} rng={r.day1_range_pct:5.1f}  {flag}")
tot = hit + miss
print(f"  recall: {hit}/{tot} accepted ({100*hit/max(tot,1):.0f}%); "
      f"listing DATE within +-3d for {datehit}/{tot} ({100*datehit/max(tot,1):.0f}%)")

print("\n--- PRECISION: known onboardings (must all be REJECTED) ---")
badkeep = 0
for s in KNOWN_ONBOARD:
    row = ld[ld.symbol == s]
    if not len(row):
        continue
    r = row.iloc[0]
    badkeep += bool(r.accepted)
    print(f"  {s:<12} start {r.list_date} bulk={r.bulk_n:3d} vr={r.vol_ratio:7.2f} "
          f"rng={r.day1_range_pct:5.1f}  {'WRONGLY KEPT' if r.accepted else 'rejected'}")
print(f"  precision leak: {badkeep}/{len(KNOWN_ONBOARD)}")
print("\nDONE -> results/listing_dates.csv")
