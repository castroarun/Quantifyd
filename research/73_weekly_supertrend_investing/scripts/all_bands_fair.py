"""
All-band fair comparison (research/73): for EVERY index band, compare the weekly-ST
timing book vs (a) same-basket equal-weight buy-and-hold (survivorship-matched) and
(b) NIFTYBEES (Nifty 50). Full year-by-year. Answers "does timing beat owning the index".
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from st_weekly_engine import get_conn, load_daily, to_weekly, ROOT

RES = os.path.join(HERE, '..', 'results')
START, END = '2010-01-01', '2026-07-07'

BANDS = {
    'Nifty50': 'nifty50_official.csv',
    'Nifty200': 'nifty200_official.csv',
    'Midcap150': 'niftymidcap150_official.csv',
    'Smallcap250': 'niftysmallcap250_official.csv',
    'Nifty500': 'nifty500_proxy.csv',
}


def stats(nav):
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    r = nav.pct_change().dropna()
    sh = (r.mean() * 52) / (r.std() * np.sqrt(52)) if r.std() else 0
    return c * 100, dd * 100, (c / abs(dd) if dd else 0), sh, nav.iloc[-1] / nav.iloc[0]


def yr_ret(nav, y):
    s = nav[nav.index.year == y]
    return 100 * (s.iloc[-1] / s.iloc[0] - 1) if len(s) >= 2 else float('nan')


conn = get_conn()
nb = to_weekly(load_daily(conn, 'NIFTYBEES'))['close']

summary_rows = []
peryear = {}   # band -> list of (yr, st, basket, nifty)
for band, csvf in BANDS.items():
    eqf = os.path.join(RES, f'g4_{band}_core_nogate_equity.csv')
    if not os.path.exists(eqf):
        print(f'!! {band}: no ST equity ({eqf}) — skip', flush=True); continue
    stC = pd.read_csv(eqf, index_col=0, parse_dates=True).iloc[:, 0]
    base = stC.index
    syms = [s.strip() for s in pd.read_csv(os.path.join(ROOT, 'backtest_data', csvf))['Symbol'].dropna()]
    closes = {}
    for s in syms:
        d = load_daily(conn, s)
        if d.empty:
            continue
        closes[s] = to_weekly(d)['close']
    px = pd.DataFrame(closes).sort_index()
    px = px[(px.index >= START) & (px.index <= END)]
    ew = (1 + px.pct_change().mean(axis=1, skipna=True).fillna(0)).cumprod()
    stC = (stC / stC.iloc[0])
    ewB = ew.reindex(base).ffill().bfill(); ewB = ewB / ewB.iloc[0]
    nbB = nb.reindex(base).ffill().bfill(); nbB = nbB / nbB.iloc[0]
    cst, dst, calst, shst, tst = stats(stC)
    cb, db, calb, shb, tb = stats(ewB)
    cn, dn, caln, shn, tn = stats(nbB)
    summary_rows.append(dict(band=band, st_cagr=round(cst, 1), basket_cagr=round(cb, 1),
                             nifty_cagr=round(cn, 1), edge_vs_basket=round(cst - cb, 1),
                             edge_vs_nifty=round(cst - cn, 1),
                             st_dd=round(dst, 1), basket_dd=round(db, 1),
                             st_cal=round(calst, 2), basket_cal=round(calb, 2)))
    yrs = range(base[0].year, base[-1].year + 1)
    peryear[band] = [(y, yr_ret(stC, y), yr_ret(ewB, y), yr_ret(nbB, y)) for y in yrs]
    print(f'{band}: done', flush=True)
conn.close()

sdf = pd.DataFrame(summary_rows)
sdf.to_csv(os.path.join(RES, 'fair_allbands_summary.csv'), index=False)
print('\n================ SUMMARY: ST timing vs same-basket B&H vs NIFTYBEES (2010-2026) ================')
print(f"{'Band':<12}{'ST CAGR':>8}{'Basket':>8}{'Nifty50':>8}{'ST-Basket':>10}{'ST-Nifty':>9}{'ST DD':>7}{'Bskt DD':>8}{'ST Cal':>7}{'Bskt Cal':>9}")
for r in summary_rows:
    print(f"{r['band']:<12}{r['st_cagr']:>7.1f}%{r['basket_cagr']:>7.1f}%{r['nifty_cagr']:>7.1f}%"
          f"{r['edge_vs_basket']:>+9.1f}{r['edge_vs_nifty']:>+9.1f}{r['st_dd']:>6.0f}%{r['basket_dd']:>7.0f}%"
          f"{r['st_cal']:>7.2f}{r['basket_cal']:>9.2f}")

# per-year tables
for band, rows in peryear.items():
    print(f'\n---- {band}: year-by-year (ST% | Basket% | Nifty50% | ST-Nifty | ST-Basket) ----')
    for y, st, bk, nf in rows:
        if any(np.isnan([st, bk, nf])):
            continue
        print(f"  {y}  ST {st:6.1f} | Bskt {bk:6.1f} | N50 {nf:6.1f} | vsN {st-nf:+6.1f} | vsB {st-bk:+6.1f}")

# write a tidy per-year CSV
allrows = []
for band, rows in peryear.items():
    for y, st, bk, nf in rows:
        allrows.append(dict(band=band, year=y, st=round(st, 1) if not np.isnan(st) else '',
                            basket=round(bk, 1) if not np.isnan(bk) else '',
                            nifty50=round(nf, 1) if not np.isnan(nf) else ''))
pd.DataFrame(allrows).to_csv(os.path.join(RES, 'fair_allbands_peryear.csv'), index=False)
print('\nDONE -> fair_allbands_summary.csv + fair_allbands_peryear.csv')
