"""research/153 Phase 0b — validate the listing-date proxy against KNOWN NSE listing dates,
and characterise the pre-listing junk rows found by the split-scale scan (DELHIVERY etc).
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

# Known NSE listing dates (mainboard), from public record.
KNOWN = {
    "DMART": "2017-03-21", "HDFCLIFE": "2017-11-17", "IEX": "2017-10-26",
    "BANDHANBNK": "2018-03-27", "IRCTC": "2019-10-14", "SBICARD": "2020-03-16",
    "ROUTE": "2020-01-28", "HAPPSTMNDS": "2020-09-17", "MAZDOCK": "2020-10-12",
    "CAMS": "2020-10-01", "GLAND": "2020-11-20", "BURGERKING": "2020-12-14",
    "RBA": "2020-12-14", "INDIGOPNTS": "2021-02-02", "NUVOCO": "2021-08-23",
    "CHEMPLASTS": "2021-08-24", "ZOMATO": "2021-07-23", "NYKAA": "2021-11-10",
    "PAYTM": "2021-11-18", "POLICYBZR": "2021-11-15", "LATENTVIEW": "2021-11-23",
    "STARHEALTH": "2021-12-10", "MAPMYINDIA": "2021-12-21", "ADANIWILMAR": "2022-02-08",
    "AWL": "2022-02-08", "CAMPUS": "2022-05-09", "LICI": "2022-05-17",
    "DELHIVERY": "2022-05-24", "FUSION": "2022-11-15", "MEDANTA": "2022-11-16",
    "GLOBALHEALTH": "2022-11-16", "MANKIND": "2023-05-09", "IDEAFORGE": "2023-07-07",
    "JIOFIN": "2023-08-21", "IREDA": "2023-11-29", "TATATECH": "2023-11-30",
    "BHARTIHEXA": "2024-04-12", "OLAELEC": "2024-08-09", "BRAINBEES": "2024-08-13",
    "BAJAJHFL": "2024-09-16", "HYUNDAI": "2024-10-22", "SWIGGY": "2024-11-13",
    "NTPCGREEN": "2024-11-27", "VMM": "2024-12-18", "HEXT": "2025-02-19",
    "PREMIERENE": "2024-09-03", "NIVABUPA": "2024-11-14", "SAGILITY": "2024-11-12",
    "ACMESOLAR": "2024-11-13", "SENORES": "2025-01-01", "STANDARD": "2025-01-01",
    "AJAXENGG": "2025-02-17", "QUALITYPOWER": "2025-02-24",
}

print("=" * 92)
print("[A] LISTING-DATE PROXY vs KNOWN NSE LISTING DATES")
print("=" * 92)
print(f"{'symbol':<14}{'known':<13}{'first_row':<13}{'delta_d':>8}  {'first_close':>11} "
      f"{'first_vol':>12}  verdict")
rows = []
for s, kd in sorted(KNOWN.items()):
    d = pd.read_sql_query(
        "select date, open, high, low, close, volume from market_data_unified "
        "where symbol=? and timeframe='day' order by date limit 12", con, params=(s,))
    if not len(d):
        print(f"{s:<14}{kd:<13}{'ABSENT':<13}{'':>8}")
        rows.append(dict(symbol=s, known=kd, first=None, delta=None, verdict="absent"))
        continue
    f = str(d['date'].iloc[0])[:10]
    delta = (pd.Timestamp(f) - pd.Timestamp(kd)).days
    v = ("EXACT" if abs(delta) <= 2 else
         ("EARLY-JUNK" if delta < -2 else "LATE"))
    print(f"{s:<14}{kd:<13}{f:<13}{delta:>8}  {d['close'].iloc[0]:>11.2f} "
          f"{d['volume'].iloc[0]:>12.0f}  {v}")
    rows.append(dict(symbol=s, known=kd, first=f, delta=delta, verdict=v,
                     first_close=float(d['close'].iloc[0]),
                     first_vol=float(d['volume'].iloc[0])))
res = pd.DataFrame(rows)
res.to_csv(RES / "recon_listing_validation.csv", index=False)
ok = res[res.verdict == "EXACT"]
print(f"\n  EXACT (|delta| <= 2 trading-ish days): {len(ok)}/{len(res)} "
      f"({100*len(ok)/len(res):.0f}%)")
for v in ("EARLY-JUNK", "LATE", "absent"):
    sub = res[res.verdict == v]
    if len(sub):
        print(f"  {v}: {len(sub)} -> " + ", ".join(
            f"{r.symbol}({r.delta:+.0f}d)" if r.delta == r.delta else r.symbol
            for r in sub.itertuples()))

# ---------------------------------------------------------------- junk row anatomy
print("\n" + "=" * 92)
print("[B] ANATOMY OF THE PRE-LISTING JUNK ROWS (the split-scale scan's biggest suspects)")
print("=" * 92)
for s in ["DELHIVERY", "FUSION", "LATENTVIEW", "COHANCE", "GOYALALUM"]:
    d = pd.read_sql_query(
        "select date, open, high, low, close, volume from market_data_unified "
        "where symbol=? and timeframe='day' order by date", con, params=(s,))
    if not len(d):
        continue
    d['date'] = d['date'].str[:10]
    c = d['close'].astype(float).values
    r = c[1:] / np.maximum(c[:-1], 1e-9)
    i = int(np.argmax(np.abs(np.log(np.maximum(r, 1e-9)))))
    print(f"\n  {s}: {len(d)} rows {d['date'].iloc[0]} -> {d['date'].iloc[-1]}; "
          f"biggest jump at row {i+1} ({d['date'].iloc[i+1]}) x{r[i]:.1f}")
    print("    first 4 rows:", d.head(4).to_dict('records'))
    print("    rows around the jump:")
    print(d.iloc[max(0, i - 1):i + 3].to_string(index=False))

# ---------------------------------------------------------------- generic junk detector
print("\n" + "=" * 92)
print("[C] GENERIC PRE-LISTING-JUNK DETECTOR over all post-2006-first-row symbols")
print("     rule: within the first 60 rows there is a close jump >5x or <1/5")
print("=" * 92)
cov = pd.read_sql_query(
    "select symbol, count(*) n, min(date) d0 from market_data_unified "
    "where timeframe='day' group by symbol", con)
cov['d0'] = pd.to_datetime(cov['d0'].str[:10])
cand = cov[(cov.d0 >= '2006-01-01') & (cov.n >= 30)]
junk = []
for s in cand['symbol']:
    d = pd.read_sql_query(
        "select date, close, volume from market_data_unified where symbol=? "
        "and timeframe='day' order by date limit 60", con, params=(s,))
    c = d['close'].astype(float).values
    if len(c) < 5:
        continue
    r = c[1:] / np.maximum(c[:-1], 1e-9)
    bad = np.nonzero((r > 5) | (r < 0.2))[0]
    if len(bad):
        j = int(bad[-1])
        junk.append(dict(symbol=s, first=str(d['date'].iloc[0])[:10],
                         junk_rows=j + 1, true_start=str(d['date'].iloc[j + 1])[:10],
                         ratio=float(r[j])))
jdf = pd.DataFrame(junk)
jdf.to_csv(RES / "recon_prelisting_junk.csv", index=False)
print(f"  {len(jdf)} / {len(cand)} symbols show pre-listing junk "
      f"({100*len(jdf)/max(len(cand),1):.1f}%)")
if len(jdf):
    print(f"  median junk rows before the real start: {jdf.junk_rows.median():.0f}")
    print(jdf.sort_values('junk_rows', ascending=False).head(20).to_string(index=False))

# ---------------------------------------------------------------- 2025 stale cohort
print("\n" + "=" * 92)
print("[D] THE 2025 COHORT'S 48.6% 'ENDS EARLY' — delisting or feed gap?")
print("=" * 92)
cov2 = pd.read_sql_query(
    "select symbol, count(*) n, min(date) d0, max(date) d1 from market_data_unified "
    "where timeframe='day' group by symbol", con)
cov2['d0'] = pd.to_datetime(cov2['d0'].str[:10]); cov2['d1'] = pd.to_datetime(cov2['d1'].str[:10])
dbmax = cov2['d1'].max()
c25 = cov2[(cov2.d0 >= '2025-01-01') & (cov2.d0 < '2026-01-01')].copy()
c25['stale_d'] = (dbmax - c25.d1).dt.days
print(f"  2025-first-row symbols: {len(c25)}; median rows {c25.n.median():.0f}")
print("  stale-days distribution:")
print(c25['stale_d'].describe().to_string())
print("\n  20 examples of the stale ones (symbol, rows, first, last):")
print(c25[c25.stale_d > 90].sort_values('stale_d', ascending=False)
      .head(20)[['symbol', 'n', 'd0', 'd1']].to_string(index=False))
print("\n  ... and the fresh ones for contrast:")
print(c25[c25.stale_d <= 90].head(10)[['symbol', 'n', 'd0', 'd1']].to_string(index=False))
print("\nDONE")
