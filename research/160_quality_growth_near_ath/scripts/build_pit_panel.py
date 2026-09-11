# -*- coding: utf-8 -*-
"""research/160 step 4 - the point-in-time fundamental panel.

One row per (monthly decision date, symbol), built ONLY from fiscal years and quarters that
had been filed by that date. This is the file that makes the study honest; everything the
engine screens on comes from here.

THE POINT-IN-TIME RULE
  * Indian fiscal years end 31-March and audited annuals are filed within about four months,
    so FY ending 31-Mar-YYYY becomes usable on 01-Aug-YYYY.  --lag-months makes that a
    parameter; 3 rebuilds an aggressive variant without re-fetching a single page.
  * A quarter ending Q becomes usable at Q + 60 days.
  * Nothing else is read. A year Screener shows today but which had not been filed on the
    decision date does not exist as far as this panel is concerned.

WHAT THE DATA CANNOT DO, STATED HERE RATHER THAN DISCOVERED LATER
  * Screener's quarterly table carries only the last ~13 quarters - it is today's window, not
    a history. So opm_q_slope8 / opm_q_std8 are computable only for decision dates from about
    mid-2023 onward and are NaN before that. Any mask built on them is a RECENT-WINDOW mask.
    The annual OPM series (opm_slope3, opm_range3, opm_min3) has the full depth and is the one
    to use for the long window.
  * Figures are as they stand today, not as first reported. Restatements are usually small,
    but this is not an as-reported vintage.
  * Delisted companies are absent from Screener entirely - see coverage_audit.py.

SHARES AND MARKET CAP - and the split trap
  Equity Capital (Rs cr) / Face Value (Rs) = share count in crores. Equity capital is
  invariant to a split (shares x face value is unchanged), so this gives a share count on
  TODAY's face-value basis. market_data.db is NOT retroactively split-adjusted: pre-split rows
  keep the old, higher price scale. Multiplying the two therefore INFLATES mcap for any period
  BEFORE a split. Every symbol is scanned for one-day collapses (close ratio < 0.55) and each
  panel row is stamped `mcap_scale_suspect` when such an event lies in its FUTURE - which is
  exactly the condition under which its mcap is on the wrong scale. The engine can drop, keep
  or sensitivity-test those rows, but it is told.

Face value is read from the #top-ratios block, which is a TODAY value. It is used only as a
per-share denominator - not as a feature - and a change of face value (a split) does not
change equity capital, so the ratio stays coherent through splits.
"""
import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/160_quality_growth_near_ath/results'
CACHE = RES / 'screener_cache'
DB = ROOT / 'backtest_data/market_data.db'

START, END = '2015-01-01', '2026-09-01'
QUARTER_LAG_DAYS = 60
MIN_YEARS = 4              # four annual points = one genuine three-year growth rate
SPLIT_RATIO = 0.55         # one-day close collapse below this = split-scale suspect


def cagr_pct(a, b, yrs):
    """Growth a -> b over yrs years, in percent. UNDEFINED when either end is not positive:
    a swing out of a loss is not a growth rate, and pretending otherwise hands the screen a
    parade of turnarounds. The no-negatives rule is what handles that case."""
    if a is None or b is None or yrs <= 0 or a <= 0 or b <= 0:
        return np.nan
    return 100.0 * ((b / a) ** (1.0 / yrs) - 1.0)


def ols_slope(x, y):
    """Least-squares slope, in y-units per x-unit. None if fewer than 2 finite points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    x, y = x[ok], y[ok]
    vx = ((x - x.mean()) ** 2).sum()
    if vx == 0:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).sum() / vx)


def load_cache():
    """screener ticker -> (annual {Timestamp: {...}}, quarterly {...}, face_value, source, top)"""
    out = {}
    for f in sorted(CACHE.glob('*.json')):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not d.get('annual'):
            continue
        ann = {pd.Timestamp(k): v for k, v in d['annual'].items()}
        qtr = {pd.Timestamp(k): v for k, v in (d.get('quarterly') or {}).items()}
        top = d.get('top') or {}
        out[d['ticker']] = dict(annual=ann, quarterly=qtr, top=top,
                                face_value=top.get('face_value'), source=d.get('source'))
    return out


def load_prices(symbols):
    """symbol -> (np.array of ISO dates, np.array of closes), plus split-suspect dates."""
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    df = pd.read_sql_query(
        "select symbol, date, close from market_data_unified where timeframe='day' "
        "order by symbol, date", con)
    con.close()
    px, splits = {}, {}
    keep = set(symbols)
    for sym, g in df.groupby('symbol', sort=False):
        if sym not in keep:
            continue
        d = g.date.values.astype('U10')
        c = g.close.values.astype(float)
        px[sym] = (d, c)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = c[1:] / np.where(c[:-1] == 0, np.nan, c[:-1])
        hit = np.where(r < SPLIT_RATIO)[0] + 1
        if len(hit):
            splits[sym] = d[hit]
    return px, splits


def assess(annual, quarterly, asof, lag_months):
    """Everything knowable about one company on one date. None if nothing is."""
    usable = sorted(k for k in annual if k + pd.DateOffset(months=lag_months) <= asof)
    if not usable:
        return None
    m = dict(n_fy_usable=len(usable), fy_latest=str(usable[-1].date()))
    latest = annual[usable[-1]]

    def eq(y):
        d = annual[y]
        ec, rv = d.get('equity_capital'), d.get('reserves')
        return None if (ec is None or rv is None) else ec + rv

    # --- growth over the last three usable years -------------------------------------
    if len(usable) >= MIN_YEARS:
        a, b = usable[-MIN_YEARS], usable[-1]
        yrs = (b - a).days / 365.25
        m['sales_g3'] = cagr_pct(annual[a].get('sales'), latest.get('sales'), yrs)
        m['profit_g3'] = cagr_pct(annual[a].get('net_profit'), latest.get('net_profit'), yrs)
        m['growth_span_yrs'] = round(yrs, 2)
    else:
        m['sales_g3'] = m['profit_g3'] = np.nan
        m['growth_span_yrs'] = np.nan

    # --- ROE, averaged over the last three usable years ------------------------------
    roes = []
    for y in usable[-3:]:
        e, np_ = eq(y), annual[y].get('net_profit')
        if e and np_ is not None and e != 0:
            roes.append(100.0 * np_ / e)
    m['roe_avg3'] = float(np.mean(roes)) if roes else np.nan
    m['n_roe_yrs'] = len(roes)
    m['roe_latest'] = roes[-1] if roes else np.nan

    # --- ROCE: Screener's own row, never computed ------------------------------------
    m['roce_latest'] = latest.get('roce_pct', np.nan)
    if m['roce_latest'] is None:
        m['roce_latest'] = np.nan
    # a lender shows no ROCE on ANY usable year while ROE is computable
    any_roce = any(annual[y].get('roce_pct') is not None for y in usable)
    m['is_lender'] = bool((not any_roce) and roes)

    # --- debt / equity ---------------------------------------------------------------
    e_latest, bor = eq(usable[-1]), latest.get('borrowings')
    m['de_latest'] = (bor / e_latest) if (e_latest and bor is not None and e_latest > 0) else np.nan

    # --- operating margin, annual ----------------------------------------------------
    opm = [(y, annual[y].get('opm_pct')) for y in usable]
    opm = [(y, v) for y, v in opm if v is not None]
    m['opm_latest'] = opm[-1][1] if opm else np.nan
    last3 = opm[-3:]
    if len(last3) >= 2:
        x = [(y - last3[0][0]).days / 365.25 for y, _ in last3]
        m['opm_slope3'] = ols_slope(x, [v for _, v in last3])
        vals = [v for _, v in last3]
        m['opm_range3'] = float(max(vals) - min(vals))
        m['opm_min3'] = float(min(vals))
    else:
        m['opm_slope3'] = m['opm_range3'] = m['opm_min3'] = np.nan
    m['n_opm_yrs'] = len(opm)

    # --- operating margin, quarterly (recent window only - see the docstring) --------
    qu = sorted(k for k in quarterly
                if k + pd.Timedelta(days=QUARTER_LAG_DAYS) <= asof)
    qv = [(k, quarterly[k].get('opm_pct')) for k in qu[-8:]]
    qv = [(k, v) for k, v in qv if v is not None]
    m['n_q_usable'] = len(qu)
    if len(qv) >= 4:
        m['opm_q_slope8'] = ols_slope(list(range(len(qv))), [v for _, v in qv])
        m['opm_q_std8'] = float(np.std([v for _, v in qv], ddof=1))
    else:
        m['opm_q_slope8'] = m['opm_q_std8'] = np.nan

    # --- "no negative figures in the last three years" -------------------------------
    m['neg3'] = bool(any(
        annual[y].get(k) is not None and annual[y][k] < 0
        for y in usable[-3:] for k in ('sales', 'net_profit')))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lag-months', type=int, default=4,
                    help='filing lag applied to fiscal years (default 4)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    lag = a.lag_months
    out_path = Path(a.out) if a.out else (
        RES / ('features_pit_monthly.csv.gz' if lag == 4
               else 'features_pit_monthly_lag%d.csv.gz' % lag))

    uni = pd.read_csv(RES / 'universe.csv')
    cache = load_cache()
    print('universe %d symbols, screener cache %d tickers' % (len(uni), len(cache)), flush=True)
    px, splits = load_prices(set(uni.symbol))
    print('prices loaded for %d symbols; %d carry a one-day collapse < %.2f (split-scale '
          'suspects)' % (len(px), len(splits), SPLIT_RATIO), flush=True)

    months = pd.date_range(START, END, freq='MS')
    rows = []
    n_no_cache = 0
    for rec in uni.itertuples():
        c = cache.get(rec.screener_ticker)
        if c is None:
            n_no_cache += 1
            continue
        pr = px.get(rec.symbol)
        fv = c['face_value']
        sp = splits.get(rec.symbol)
        for d in months:
            m = assess(c['annual'], c['quarterly'], d, lag)
            if m is None:
                continue
            close = np.nan
            if pr is not None:
                i = np.searchsorted(pr[0], str(d.date()), 'right') - 1
                if i >= 0 and (d - pd.Timestamp(pr[0][i])).days <= 15:
                    close = float(pr[1][i])
            ec = c['annual'][pd.Timestamp(m['fy_latest'])].get('equity_capital')
            shares = (ec / fv) if (fv and ec is not None and fv > 0) else np.nan
            rows.append(dict(
                date=str(d.date()), symbol=rec.symbol,
                screener_ticker=rec.screener_ticker, source=c['source'],
                close_pit=close, face_value=fv, shares_pit=shares,
                mcap_pit=(shares * close if (shares == shares and close == close) else np.nan),
                mcap_scale_suspect=bool(sp is not None and (sp > str(d.date())).any()),
                **m))
    p = pd.DataFrame(rows)
    # keep the column order the contract promises
    lead = ['date', 'symbol', 'screener_ticker', 'source', 'n_fy_usable', 'fy_latest',
            'sales_g3', 'profit_g3', 'growth_span_yrs', 'roe_avg3', 'roe_latest', 'n_roe_yrs',
            'roce_latest', 'is_lender', 'de_latest', 'opm_latest', 'opm_slope3', 'opm_range3',
            'opm_min3', 'n_opm_yrs', 'opm_q_slope8', 'opm_q_std8', 'n_q_usable', 'neg3',
            'face_value', 'shares_pit', 'close_pit', 'mcap_pit', 'mcap_scale_suspect']
    p = p[[c for c in lead if c in p.columns]]
    for c in ('sales_g3', 'profit_g3', 'roe_avg3', 'roe_latest', 'roce_latest', 'de_latest',
              'opm_latest', 'opm_slope3', 'opm_range3', 'opm_min3', 'opm_q_slope8',
              'opm_q_std8', 'shares_pit', 'close_pit', 'mcap_pit'):
        p[c] = p[c].astype(float).round(4)
    p.to_csv(out_path, index=False, compression='gzip')

    print()
    print('rows %d   symbols %d   months %d   (%d universe symbols had no Screener record)'
          % (len(p), p.symbol.nunique(), p.date.nunique(), n_no_cache))
    last = p[p.date == p.date.max()]
    print('at %s: %d symbols, %d with n_fy_usable>=4, %d lenders, %d mcap-scale suspects'
          % (p.date.max(), len(last), int((last.n_fy_usable >= 4).sum()),
             int(last.is_lender.sum()), int(last.mcap_scale_suspect.sum())))
    print('quarterly OPM available on %d of %d rows (it is a recent-window feature)'
          % (int(p.opm_q_slope8.notna().sum()), len(p)))

    # --- share-count reconciliation, the pre-registered sanity check -----------------
    chk = []
    for t, c in cache.items():
        top = c['top']
        if not (top.get('market_cap') and top.get('current_price') and c['face_value']):
            continue
        ann = c['annual']
        if not ann:
            continue
        ec = ann[max(ann)].get('equity_capital')
        if not ec:
            continue
        sh = ec / c['face_value']
        chk.append((top['market_cap'], 100.0 * (sh * top['current_price'] - top['market_cap'])
                    / top['market_cap'], t))
    chk.sort(reverse=True)
    big = chk[:20]
    print()
    print('share-count reconciliation on the 20 largest names '
          '(equity capital / face value x current price, vs Screener\'s own market cap):')
    for mc, err, t in big:
        print('   %-14s mcap %9.0f cr   error %+7.1f%%' % (t, mc, err))
    errs = np.array([abs(e) for _, e, _ in big])
    print('   median |error| %.1f%%   worst %.1f%%' % (np.median(errs), errs.max()))
    allerr = np.array([abs(e) for _, e, _ in chk])
    print('   across all %d reconcilable names: median %.1f%%, %.0f%% within 10%%'
          % (len(allerr), np.median(allerr), 100.0 * (allerr <= 10).mean()))
    print()
    print('wrote %s (%.1f MB)' % (out_path, out_path.stat().st_size / 1e6))


if __name__ == '__main__':
    sys.exit(main())
