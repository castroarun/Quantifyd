"""Year-wise return AND drawdown, side by side, for the five individual systems
(TN, OA, VCP, MYB, IPO) plus the NIFTY 50 benchmark.

Conventions (r/154, 2026-09-05):
  - per-year return: seed/offset MEDIAN of each path's calendar-year return
  - per-year drawdown: seed/offset MEDIAN of the worst drawdown experienced DURING that
    year, measured from the RUNNING PEAK OF THE FULL CURVE (not the year's first bar) —
    this is the correction r/154 made; the old convention hid falls that began in December
  - all curves after-tax, net 25 bps/side, idle cash 5% p.a.
Windows differ per system and are printed with the summary row.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/arun/quantifyd/research')
OUT = R / '154_multi_system_blends' / 'results'

SRC = {
    'OA':  (R / '154_multi_system_blends/results/oa_navs30.csv', None),
    'TN':  (R / '154_multi_system_blends/results/tn_navs12.csv', None),
    'VCP': (R / '151_vcp_breakout/results/vcp_equity_seeds.csv', None),
    'MYB': (R / '152_multiyear_breakout/results/myb_equity_seeds.csv', None),
    'IPO': (R / '153_ipo_base/results/ipo_equity_seeds.csv', None),
}


def load(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime([str(x)[:10] for x in df.index])
    return df.sort_index().astype(float)


def year_stats(nav):
    """(year -> (return %, drawdown % from the full curve's running peak))"""
    peak = nav.cummax()
    dd = nav / peak - 1.0
    out = {}
    for yr, seg in nav.groupby(nav.index.year):
        prev = nav[nav.index.year < yr]
        base = prev.iloc[-1] if len(prev) else seg.iloc[0]
        ret = (seg.iloc[-1] / base - 1.0) * 100
        out[yr] = (ret, float(dd[dd.index.year == yr].min() * 100))
    return out


def full_stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
    mdd = float((nav / nav.cummax() - 1).min() * 100)
    return cagr, mdd, (cagr / abs(mdd) if mdd else np.nan)


rows, summ, windows = {}, {}, {}
for name, (path, _) in SRC.items():
    df = load(path)
    per = [year_stats(df[c]) for c in df.columns]
    years = sorted({y for p in per for y in p})
    rows[name] = {y: (float(np.median([p[y][0] for p in per if y in p])),
                      float(np.median([p[y][1] for p in per if y in p]))) for y in years}
    fs = [full_stats(df[c]) for c in df.columns]
    summ[name] = (float(np.median([f[0] for f in fs])),
                  float(np.median([f[1] for f in fs])),
                  float(np.median([f[2] for f in fs])))
    windows[name] = f'{df.index[0].date()} to {df.index[-1].date()}  ({len(df.columns)} paths)'
    print(f'{name}: {windows[name]}  CAGR {summ[name][0]:.2f}%  MaxDD {summ[name][1]:.2f}%')

# ---- NIFTY 50 benchmark (NIFTYBEES total price series from the market DB) ----
import sqlite3
con = sqlite3.connect('/home/arun/quantifyd/backtest_data/market_data.db')
q = con.execute("SELECT date, close FROM market_data_unified WHERE symbol='NIFTYBEES' "
                "AND timeframe='day' AND date>='2006-01-01' ORDER BY date").fetchall()
con.close()
bench = pd.Series({pd.Timestamp(str(d)[:10]): float(c) for d, c in q}).sort_index()
rows['NIFTY'] = year_stats(bench)
summ['NIFTY'] = full_stats(bench)
windows['NIFTY'] = f'{bench.index[0].date()} to {bench.index[-1].date()}  (index)'
print(f"NIFTY: CAGR {summ['NIFTY'][0]:.2f}%  MaxDD {summ['NIFTY'][1]:.2f}%")

ORDER = ['OA', 'TN', 'VCP', 'MYB', 'IPO']
all_years = sorted({y for n in ORDER for y in rows[n]})

recs = []
for y in all_years:
    rec = {'year': y}
    avail = []
    for n in ORDER:
        v = rows[n].get(y)
        rec[f'{n}_ret'] = None if v is None else round(v[0], 1)
        rec[f'{n}_dd'] = None if v is None else round(v[1], 1)
        if v:
            avail.append((n, v))
    b = rows['NIFTY'].get(y)
    rec['NIFTY_ret'] = None if b is None else round(b[0], 1)
    rec['NIFTY_dd'] = None if b is None else round(b[1], 1)
    if avail:
        rec['best_ret'] = max(avail, key=lambda t: t[1][0])[0]
        rec['least_dd'] = max(avail, key=lambda t: t[1][1])[0]
        rec['best_overall'] = max(avail, key=lambda t: t[1][0] + t[1][1])[0]
    recs.append(rec)

df = pd.DataFrame(recs)
df.to_csv(OUT / 'yoy_five_systems.csv', index=False)
json.dump({'rows': recs,
           'summary': {n: dict(cagr=round(summ[n][0], 2), maxdd=round(summ[n][1], 2),
                               calmar=round(summ[n][2], 2), window=windows[n])
                       for n in ORDER + ['NIFTY']}},
          open(OUT / 'yoy_five_systems.json', 'w'), indent=1)

print('\n=== YEAR | ' + ' | '.join(f'{n} ret(dd)' for n in ORDER) + ' | NIFTY | best ret | least dd | best overall')
for r in recs:
    cells = []
    for n in ORDER:
        rr, dd = r[f'{n}_ret'], r[f'{n}_dd']
        cells.append('     --      ' if rr is None else f'{rr:+7.1f}({dd:6.1f})')
    b = '' if r['NIFTY_ret'] is None else f"{r['NIFTY_ret']:+6.1f}({r['NIFTY_dd']:6.1f})"
    print(f"{r['year']} | " + ' | '.join(cells) + f" | {b} | {r.get('best_ret','')} | "
          f"{r.get('least_dd','')} | {r.get('best_overall','')}")
print('\n=== SUMMARY (full period, each on its own window) ===')
for n in ORDER + ['NIFTY']:
    c, m, k = summ[n]
    print(f'{n:6s} CAGR {c:6.2f}%  MaxDD {m:7.2f}%  Calmar {k:5.2f}   {windows[n]}')
print('\nwrote yoy_five_systems.csv / .json')
