"""
REAL NIFTY futures basis from NSE F&O bhavcopy — validate the hedge-carry assumption,
especially the crash-time BACKWARDATION risk. Samples crash windows (daily) + normal months.
Matches near-month future settle vs NIFTY 50 spot (from our DB); classifies each date by
whether NIFTYBEES daily ST(7,3) was RED (hedge would be ON). Reports carry distribution
hedge-ON vs OFF + worst backwardation. Saves results/real_basis.csv.
"""
import os, sys, io, zipfile, urllib.request, datetime as dt, time, sqlite3
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
from services.technical_indicators import calc_supertrend

DB = os.path.join(ROOT, 'backtest_data', 'market_data.db')
RES = os.path.join(HERE, '..', 'results')
MON = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
HDRS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Accept': '*/*',
        'Referer': 'https://www.nseindia.com/'}


def bhav_url(d):
    return (f"https://archives.nseindia.com/content/historical/DERIVATIVES/"
            f"{d.year}/{MON[d.month-1]}/fo{d.day:02d}{MON[d.month-1]}{d.year}bhav.csv.zip")


def fetch_nifty_fut(d):
    """Return (near_fut_close, expiry_date) for NIFTY FUTIDX near-month on date d, or None."""
    try:
        req = urllib.request.Request(bhav_url(d), headers=HDRS)
        raw = urllib.request.urlopen(req, timeout=20).read()
    except Exception:
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
        df = pd.read_csv(z.open(z.namelist()[0]))
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    f = df[(df['INSTRUMENT'] == 'FUTIDX') & (df['SYMBOL'] == 'NIFTY')].copy()
    if f.empty:
        return None
    f['EXP'] = pd.to_datetime(f['EXPIRY_DT'], format='%d-%b-%Y', errors='coerce')
    f = f[f['EXP'] >= pd.Timestamp(d)].sort_values('EXP')
    if f.empty:
        return None
    row = f.iloc[0]
    px = row['CLOSE'] if row['CLOSE'] > 0 else row['SETTLE_PR']
    return float(px), row['EXP'].date()


def daterange(a, b):
    d = a
    while d <= b:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


# sample dates: crash windows DAILY + normal months (1st business day)
dates = []
dates += list(daterange(dt.date(2020, 2, 20), dt.date(2020, 5, 29)))      # COVID crash+recovery (daily)
dates += [d for i, d in enumerate(daterange(dt.date(2022, 1, 3), dt.date(2022, 6, 30))) if i % 2 == 0]  # 2022
dates += [d for i, d in enumerate(daterange(dt.date(2018, 9, 1), dt.date(2018, 11, 30))) if i % 2 == 0]  # 2018
dates += list(daterange(dt.date(2015, 8, 17), dt.date(2015, 9, 11)))      # Aug-2015 China shock (daily)
# normal-regime monthly sample 2016-2024
for y in range(2016, 2025):
    for m in range(1, 13, 2):
        dates.append(dt.date(y, m, 5))
dates = sorted(set(dates))
print(f"Sampling {len(dates)} dates (crash windows daily + normal months)...", flush=True)

# spot from DB
c = sqlite3.connect(DB)
spot = pd.read_sql_query("SELECT date,close FROM market_data_unified WHERE symbol='NIFTY50' "
                         "AND timeframe='day' ORDER BY date", c)
nb = pd.read_sql_query("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTYBEES' "
                       "AND timeframe='day' ORDER BY date", c)
c.close()
spot['date'] = pd.to_datetime(spot['date']); spot = spot.set_index('date')['close']
nb['date'] = pd.to_datetime(nb['date']); nb = nb.set_index('date')
_, dser = calc_supertrend(nb[['high', 'low', 'close']].assign(open=nb['open']), 7, 3)
st_red = (pd.Series(dser.values, index=nb.index) == -1)

rows = []
ok = 0
for k, d in enumerate(dates):
    r = fetch_nifty_fut(d)
    time.sleep(0.25)
    if r is None:
        continue
    fut, exp = r
    ts = pd.Timestamp(d)
    if ts not in spot.index:
        continue
    sp = spot.loc[ts]
    dte = (exp - d).days
    if dte <= 0 or sp <= 0:
        continue
    basis = fut / sp - 1
    carry = basis * (365.0 / dte)                 # annualised
    red = bool(st_red.reindex([ts]).ffill().iloc[0]) if ts >= st_red.index[0] else None
    rows.append(dict(date=d.isoformat(), fut=round(fut, 1), spot=round(sp, 1), dte=dte,
                     basis_pct=round(100 * basis, 3), carry_ann_pct=round(100 * carry, 2),
                     hedge_on=red))
    ok += 1
    if ok % 40 == 0:
        print(f"  {ok} fetched (last {d})...", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, 'real_basis.csv'), index=False)
print(f"\nGot {len(df)} real basis points.\n")

def summ(sub, name):
    if sub.empty:
        print(f"{name:<28} (no data)"); return
    c = sub['carry_ann_pct']
    print(f"{name:<28} n={len(sub):<4} carry ann%: mean {c.mean():+.1f}  median {c.median():+.1f}  "
          f"min {c.min():+.1f}  max {c.max():+.1f}  backwardation(<0): {100*(c<0).mean():.0f}%")

print("=== REAL annualised carry by regime ===")
summ(df, 'ALL sampled')
summ(df[df['hedge_on'] == True], 'HEDGE-ON (ST red)')
summ(df[df['hedge_on'] == False], 'HEDGE-OFF (ST green)')
print("\n=== crash windows (hedge would be ON a lot) ===")
for lbl, a, b in [('COVID 2020 Feb-May', '2020-02-20', '2020-05-29'),
                  ('2022 selloff', '2022-01-01', '2022-06-30'),
                  ('2018 correction', '2018-09-01', '2018-11-30'),
                  ('Aug-2015 shock', '2015-08-17', '2015-09-11')]:
    summ(df[(df['date'] >= a) & (df['date'] <= b)], lbl)
print("\nWorst 8 backwardation days (lowest carry):")
for _, r in df.nsmallest(8, 'carry_ann_pct').iterrows():
    print(f"  {r['date']}  basis={r['basis_pct']:+.2f}% carry={r['carry_ann_pct']:+.1f}%/yr dte={r['dte']} hedge_on={r['hedge_on']}")
print("DONE")
