"""research/153 Phase 0c — the decisive data question.

`first row in market_data_unified` is contaminated by BULK DATA-ONBOARDING WAVES:
long-listed companies (e.g. ABB, listed in the 1990s) whose series begins on the day
they were added to the nightly Kite download. Those would be read as "2025 IPOs".

This script:
  1. counts first-row dates and finds the bulk-add days,
  2. builds a listing-day fingerprint (day-1 volume vs the next 20 days' median; day-1
     high-low range) and checks it separates KNOWN IPOs from KNOWN onboardings,
  3. proposes and scores an accepted-listing filter,
  4. writes results/listing_dates.csv — the vetted listing table the sweep will use.
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

print("=" * 92)
print("[A] BULK-ONBOARDING WAVES — dates on which many symbols' series begin")
print("=" * 92)
vc = cov['d0'].value_counts().sort_values(ascending=False)
print("  top 30 first-row dates by symbol count:")
for d, k in vc.head(30).items():
    print(f"    {d.date()}  {k:4d} symbols")
big = set(vc[vc >= 5].index)
print(f"\n  dates with >=5 new symbols: {len(big)}; they account for "
      f"{int(vc[vc >= 5].sum())} of {len(cov)} symbols "
      f"({100*vc[vc>=5].sum()/len(cov):.0f}%)")
for k in (3, 5, 8, 12, 20):
    dd = vc[vc >= k]
    print(f"    threshold >={k:2d}: {len(dd):4d} dates, {int(dd.sum()):5d} symbols affected")

# ------------------------------------------------------- 2. listing-day fingerprint
print("\n" + "=" * 92)
print("[B] LISTING-DAY FINGERPRINT")
print("=" * 92)
KNOWN_IPO = ["DMART", "HDFCLIFE", "BANDHANBNK", "IRCTC", "SBICARD", "HAPPSTMNDS",
             "MAZDOCK", "CAMS", "GLAND", "RBA", "INDIGOPNTS", "NUVOCO", "CHEMPLASTS",
             "NYKAA", "PAYTM", "POLICYBZR", "LATENTVIEW", "STARHEALTH", "MAPMYINDIA",
             "AWL", "CAMPUS", "LICI", "DELHIVERY", "FUSION", "MANKIND", "IDEAFORGE",
             "JIOFIN", "IREDA", "TATATECH", "BHARTIHEXA", "OLAELEC", "BAJAJHFL",
             "HYUNDAI", "SWIGGY", "NTPCGREEN", "VMM", "HEXT", "ACMESOLAR", "SAGILITY",
             "NIVABUPA", "PREMIERENE", "AJAXENGG", "MEDANTA", "IEX", "SENORES"]
# long-listed companies that CANNOT be 2025 IPOs -> their first row is an onboarding
KNOWN_OLD = ["ABB", "360ONE", "SIEMENS", "CUMMINSIND", "HAVELLS", "PIDILITIND",
             "BOSCHLTD", "SUPREMEIND", "ASTRAL", "POLYCAB", "APLAPOLLO", "COFORGE",
             "PERSISTENT", "LTTS", "MPHASIS", "OFSS", "TATAELXSI", "KPITTECH"]


def fingerprint(sym):
    d = pd.read_sql_query(
        "select date, open, high, low, close, volume from market_data_unified "
        "where symbol=? and timeframe='day' order by date limit 40", con, params=(sym,))
    if len(d) < 6:
        return None
    v = d['volume'].astype(float).values
    h, l, c = (d[k].astype(float).values for k in ('high', 'low', 'close'))
    med_next = np.median(v[1:21]) if len(v) > 5 else np.nan
    return dict(symbol=sym, first=str(d['date'].iloc[0])[:10],
                v0=v[0], vratio=v[0] / max(med_next, 1),
                rng0=100 * (h[0] - l[0]) / max(c[0], 1e-9))


fi = pd.DataFrame([x for x in (fingerprint(s) for s in KNOWN_IPO) if x])
fo = pd.DataFrame([x for x in (fingerprint(s) for s in KNOWN_OLD) if x])
print("  KNOWN IPOs (n=%d): day-1 volume / next-20d median volume" % len(fi))
print(f"     median {fi.vratio.median():.2f}  p10 {fi.vratio.quantile(.1):.2f}  "
      f"p90 {fi.vratio.quantile(.9):.2f} | day-1 range%% median {fi.rng0.median():.1f}")
print("  KNOWN long-listed (onboarded) (n=%d):" % len(fo))
print(f"     median {fo.vratio.median():.2f}  p10 {fo.vratio.quantile(.1):.2f}  "
      f"p90 {fo.vratio.quantile(.9):.2f} | day-1 range%% median {fo.rng0.median():.1f}")
print("\n  onboarded sample first-rows:")
print(fo[['symbol', 'first', 'vratio', 'rng0']].to_string(index=False))

# ------------------------------------------------------- 3. accepted-listing filter
print("\n" + "=" * 92)
print("[C] ACCEPTED-LISTING FILTER")
print("=" * 92)


def bulk_flag(d0, k):
    return vc.get(d0, 0) >= k


for k in (3, 5, 8):
    ipo_kept = sum(1 for _, r in fi.iterrows() if not bulk_flag(pd.Timestamp(r['first']), k))
    old_kept = sum(1 for _, r in fo.iterrows() if not bulk_flag(pd.Timestamp(r['first']), k))
    print(f"  bulk-day threshold >={k}: keeps {ipo_kept}/{len(fi)} known IPOs, "
          f"wrongly keeps {old_kept}/{len(fo)} known onboardings")

for vr in (2.0, 3.0, 5.0):
    ipo_kept = int((fi.vratio >= vr).sum())
    old_kept = int((fo.vratio >= vr).sum())
    print(f"  vol-ratio >= {vr}: keeps {ipo_kept}/{len(fi)} known IPOs, "
          f"wrongly keeps {old_kept}/{len(fo)} known onboardings")

for k, vr in ((5, 2.0), (5, 3.0), (3, 3.0)):
    ipo_kept = int(sum(1 for _, r in fi.iterrows()
                       if not bulk_flag(pd.Timestamp(r['first']), k) and r['vratio'] >= vr))
    old_kept = int(sum(1 for _, r in fo.iterrows()
                       if not bulk_flag(pd.Timestamp(r['first']), k) and r['vratio'] >= vr))
    print(f"  COMBINED bulk>={k} AND volratio>={vr}: keeps {ipo_kept}/{len(fi)} IPOs, "
          f"wrongly keeps {old_kept}/{len(fo)} onboardings")

# ------------------------------------------------------- 4. build the listing table
print("\n" + "=" * 92)
print("[D] BUILDING results/listing_dates.csv")
print("=" * 92)
rows = []
for s, n, d0, d1 in cov[['symbol', 'n', 'd0', 'd1']].itertuples(index=False):
    if n < 25:
        continue
    d = pd.read_sql_query(
        "select date, open, high, low, close, volume from market_data_unified "
        "where symbol=? and timeframe='day' order by date limit 80", con, params=(s,))
    c = d['close'].astype(float).values
    v = d['volume'].astype(float).values
    # strip pre-listing junk: last >5x / <0.2x close jump inside the first 60 rows
    j = 0
    if len(c) > 3:
        r = c[1:min(60, len(c))] / np.maximum(c[:min(59, len(c) - 1)], 1e-9)
        bad = np.nonzero((r > 5) | (r < 0.2))[0]
        if len(bad):
            j = int(bad[-1]) + 1
    start = pd.Timestamp(str(d['date'].iloc[j])[:10]) if j < len(d) else d0
    v0 = v[j] if j < len(v) else np.nan
    nxt = v[j + 1:j + 21]
    vratio = v0 / max(np.median(nxt), 1) if len(nxt) else np.nan
    rows.append(dict(symbol=s, first_row=str(d0.date()), junk_rows=j,
                     list_date=str(start.date()), n_rows=int(n),
                     last_row=str(d1.date()),
                     bulk_n=int(vc.get(start, 0)), vol_ratio=float(vratio) if vratio == vratio else np.nan,
                     day1_range_pct=float(100 * (d['high'].iloc[j] - d['low'].iloc[j])
                                          / max(d['close'].iloc[j], 1e-9)) if j < len(d) else np.nan))
ld = pd.DataFrame(rows)
ld['accepted'] = (ld.bulk_n < 5) & (ld.vol_ratio >= 2.0) & (ld.list_date > '2006-01-01')
ld.to_csv(RES / "listing_dates.csv", index=False)
print(f"  {len(ld)} symbols profiled; {int(ld.accepted.sum())} ACCEPTED as real listings "
      f"(post-2006, non-bulk day, day-1 volume >= 2x next-20d median)")
print("\n  accepted listings per year:")
ld['ly'] = ld.list_date.str[:4]
g = ld[ld.accepted].groupby('ly').size()
for y, k in g.items():
    print(f"    {y}: {k:4d}")
print("\n  sanity: known IPOs accepted?")
miss = [s for s in KNOWN_IPO if s in set(ld.symbol) and
        not bool(ld.loc[ld.symbol == s, 'accepted'].iloc[0])]
print(f"    {len(KNOWN_IPO)-len(miss)}/{len([s for s in KNOWN_IPO if s in set(ld.symbol)])} "
      f"known IPOs accepted; rejected: {miss}")
print("  sanity: known long-listed wrongly accepted?")
wrong = [s for s in KNOWN_OLD if s in set(ld.symbol) and
         bool(ld.loc[ld.symbol == s, 'accepted'].iloc[0])]
print(f"    {wrong}")
print("\nDONE -> results/listing_dates.csv")
