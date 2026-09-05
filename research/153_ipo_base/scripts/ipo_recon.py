"""research/153 Phase 0 — data recon for the IPO-Base screen.

Answers, before any backtest is run:
  1. LISTING-DATE PROXY. first row in market_data_unified (timeframe='day') per symbol.
     How many symbols have a first-row date after the DB's coverage floor? Cohort by year.
  2. SURVIVORSHIP. how many symbols' data ENDS well before the DB max date (delisting /
     stale proxy), split by listing cohort. Compare "IPO cohort" survival to the whole DB.
  3. PHANTOM HOLIDAY ROWS. scan for sparse days with >90% zero-volume rows.
  4. SPLIT-SCALE sanity: symbols with a >3x single-day close gap (unadjusted split).
  5. HOW MANY SIGNALS CAN EVEN EXIST: for each max-age band, count symbol-days that are
     both young and liquid (>=Rs 5cr 20d median traded value).
Writes results/recon_*.csv|json and prints a readable report.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
RES.mkdir(exist_ok=True)
DB = Path("/home/arun/quantifyd/backtest_data/market_data.db")
if not DB.exists():
    DB = Path(__file__).resolve().parents[3] / "backtest_data" / "market_data.db"

con = sqlite3.connect(str(DB))

print("=" * 78)
print("PHASE 0 RECON — research/153 IPO Base")
print("DB:", DB)
print("=" * 78)

# ---------------------------------------------------------------- 1. coverage
q = """
select symbol, count(*) n, min(date) d0, max(date) d1
from market_data_unified where timeframe='day' group by symbol
"""
cov = pd.read_sql_query(q, con)
cov["d0"] = pd.to_datetime(cov["d0"].str[:10])
cov["d1"] = pd.to_datetime(cov["d1"].str[:10])
dbmax = cov["d1"].max()
dbmin = cov["d0"].min()
print(f"\n[1] COVERAGE: {len(cov)} symbols, dates {dbmin.date()} -> {dbmax.date()}")
print(f"    rows total: {int(cov.n.sum()):,}")

# trading calendar = union of dates on the most complete symbol set
cal = pd.read_sql_query(
    "select distinct date from market_data_unified where timeframe='day' order by date", con)
cal["date"] = pd.to_datetime(cal["date"].str[:10])
CAL = pd.DatetimeIndex(cal["date"].unique())
print(f"    distinct trading dates: {len(CAL)}")

# ------------------------------------------------- 2. listing-date proxy cohorts
cov["list_year"] = cov["d0"].dt.year
byyear = cov.groupby("list_year").agg(
    n_symbols=("symbol", "count"),
    median_rows=("n", "median"),
).reset_index()
print("\n[2] FIRST-ROW-DATE COHORTS (the listing-date proxy)")
print("    year  n_syms  median_rows   (a spike at the coverage floor = NOT real listings)")
for _, r in byyear.iterrows():
    bar = "#" * min(60, int(r.n_symbols / 4))
    print(f"    {int(r.list_year)}  {int(r.n_symbols):5d}  {int(r.median_rows):8d}   {bar}")
byyear.to_csv(RES / "recon_listing_cohorts.csv", index=False)

# how many symbols start "late" (i.e. proxy is plausibly a real listing)
for cut in ("2005-01-01", "2010-01-01", "2015-01-01", "2018-01-01", "2020-01-01"):
    k = int((cov["d0"] >= cut).sum())
    print(f"    symbols with first row >= {cut}: {k}")

# ------------------------------------------------- 3. survivorship / stale-end probe
cov["days_stale"] = (dbmax - cov["d1"]).dt.days
cov["ends_early"] = cov["days_stale"] > 90
print("\n[3] SURVIVORSHIP PROBE — symbols whose series ENDS >90d before DB max")
print(f"    {int(cov.ends_early.sum())} / {len(cov)} "
      f"({100*cov.ends_early.mean():.1f}%) end early")
print("    by listing cohort (proxy):")
g = cov.groupby("list_year").agg(n=("symbol", "count"), dead=("ends_early", "sum"))
g["dead_pct"] = 100 * g["dead"] / g["n"]
for y, r in g.iterrows():
    if y >= 2005:
        print(f"      {int(y)}: {int(r.n):4d} listed-proxy, {int(r.dead):3d} end early "
              f"({r.dead_pct:5.1f}%)")
g.to_csv(RES / "recon_survivorship_by_cohort.csv")

# terminal drawdown of the "dead" names -- did they die low (real delisting) or is it
# just a stale feed?
dead = cov[cov.ends_early & (cov.list_year >= 2010)].copy()
print(f"\n    inspecting {len(dead)} early-ending post-2010-listing symbols for "
      f"terminal price behaviour (sample <=60) ...")
samp = dead.sample(min(60, len(dead)), random_state=1) if len(dead) else dead
term = []
for s in samp["symbol"]:
    d = pd.read_sql_query(
        "select date, close from market_data_unified where symbol=? and timeframe='day' "
        "order by date", con, params=(s,))
    if len(d) < 30:
        continue
    c = d["close"].astype(float)
    term.append(dict(symbol=s, last_close=c.iloc[-1], peak=c.max(),
                     from_peak=100 * (c.iloc[-1] / c.max() - 1),
                     first=c.iloc[0], tot=100 * (c.iloc[-1] / c.iloc[0] - 1)))
if term:
    tdf = pd.DataFrame(term)
    tdf.to_csv(RES / "recon_dead_terminal.csv", index=False)
    print(f"      median terminal drawdown from peak: {tdf.from_peak.median():.1f}%")
    print(f"      median total return listing->last:  {tdf.tot.median():.1f}%")
    print(f"      share ending >50% below peak: {100*(tdf.from_peak < -50).mean():.0f}%")

# ------------------------------------------------- 4. phantom-holiday-row check
print("\n[4] PHANTOM HOLIDAY ROWS (sparse day, >90% zero volume)")
ph = pd.read_sql_query(
    "select date, count(*) n, sum(case when volume=0 or volume is null then 1 else 0 end) z "
    "from market_data_unified where timeframe='day' and date >= '2024-01-01' "
    "group by date order by date", con)
ph["date"] = ph["date"].str[:10]
ph["zpct"] = 100 * ph["z"] / ph["n"]
bad = ph[(ph.zpct > 90) & (ph.n > 20)]
if len(bad):
    print("    SUSPECT DAYS:")
    print(bad.to_string(index=False))
else:
    print("    none found since 2024-01-01 — purge intact")
ph.to_csv(RES / "recon_zero_volume_days.csv", index=False)

# ------------------------------------------------- 5. split-scale sanity
print("\n[5] SPLIT-SCALE SANITY — single-day close gaps >3x or <1/3 (post-2010 listings)")
suspect = []
cand = cov[(cov.list_year >= 2010) & (cov.n >= 120)]["symbol"].tolist()
print(f"    scanning {len(cand)} post-2010-listing symbols ...")
for s in cand:
    d = pd.read_sql_query(
        "select date, close from market_data_unified where symbol=? and timeframe='day' "
        "order by date", con, params=(s,))
    c = d["close"].astype(float).values
    if len(c) < 30:
        continue
    r = c[1:] / np.maximum(c[:-1], 1e-9)
    i = np.argmax(np.abs(np.log(np.maximum(r, 1e-9))))
    if r[i] > 3 or r[i] < 1 / 3:
        suspect.append(dict(symbol=s, date=d["date"].iloc[i + 1][:10],
                            prev=c[i], now=c[i + 1], ratio=r[i]))
sdf = pd.DataFrame(suspect)
sdf.to_csv(RES / "recon_split_suspects.csv", index=False)
print(f"    {len(sdf)} suspects (listed in results/recon_split_suspects.csv)")
if len(sdf):
    print(sdf.head(25).to_string(index=False))

# ------------------------------------------------- 6. how many candidates exist at all
print("\n[6] CANDIDATE DENSITY — young + liquid symbol-days")
ETF_RE = r"(BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50)"
elig_syms = cov[(cov.n >= 40) & (~cov.symbol.str.contains(ETF_RE, regex=True))].copy()
print(f"    {len(elig_syms)} non-ETF symbols with >=40 daily rows")

rows = []
listmap = dict(zip(cov.symbol, cov.d0))
for s in elig_syms["symbol"]:
    d = pd.read_sql_query(
        "select date, close, volume from market_data_unified where symbol=? "
        "and timeframe='day' order by date", con, params=(s,))
    if len(d) < 40:
        continue
    d["date"] = pd.to_datetime(d["date"].str[:10])
    tv = (d["close"].astype(float) * d["volume"].astype(float)).rolling(20).median().shift(1)
    age_d = (d["date"] - listmap[s]).dt.days
    liq = tv >= 5e7
    for m in (3, 6, 12, 24, 36):
        young = (age_d > 0) & (age_d <= m * 30.44)
        rows.append(dict(symbol=s, months=m, n_young=int(young.sum()),
                         n_young_liq=int((young & liq).sum()),
                         first=str(listmap[s].date())))
dens = pd.DataFrame(rows)
dens.to_csv(RES / "recon_candidate_density.csv", index=False)
print("\n    months  symbols_with_any_young_liquid_day   total_young_liquid_symbol_days")
for m in (3, 6, 12, 24, 36):
    sub = dens[dens.months == m]
    print(f"    {m:6d}  {int((sub.n_young_liq > 0).sum()):34d}  "
          f"{int(sub.n_young_liq.sum()):30,d}")

# per-year count of NEW listings that ever become liquid
liq_syms = set(dens[(dens.months == 24) & (dens.n_young_liq > 0)]["symbol"])
cov["ever_liq_young"] = cov.symbol.isin(liq_syms)
print("\n    new-listing cohort (proxy) that becomes liquid within 24m of listing:")
gg = cov[cov.list_year >= 2010].groupby("list_year").agg(
    n=("symbol", "count"), liq=("ever_liq_young", "sum"))
for y, r in gg.iterrows():
    print(f"      {int(y)}: {int(r.n):4d} proxy-listings, {int(r.liq):4d} become liquid-young")

json.dump(dict(db=str(DB), db_min=str(dbmin.date()), db_max=str(dbmax.date()),
               n_symbols=int(len(cov)),
               n_first_row_ge_2010=int((cov.d0 >= "2010-01-01").sum()),
               n_first_row_ge_2015=int((cov.d0 >= "2015-01-01").sum()),
               ends_early_pct=float(100 * cov.ends_early.mean()),
               split_suspects=int(len(sdf))),
          open(RES / "recon_summary.json", "w"), indent=2)
cov.to_csv(RES / "recon_symbol_coverage.csv", index=False)
print("\nDONE — recon artefacts in results/recon_*.csv|json")
