"""
Crash-exit overlay test (research/73 phase 2): own the same-basket EW book, layer a
MARKET-LEVEL de-risk signal (index gate / index ST / vol-spike) that flattens the WHOLE
book in confirmed downtrends and re-enters. Goal: cut drawdown WITHOUT giving back CAGR.

Daily resolution (so a fast crash exit is meaningful). Causal: signal at close t -> act at
open/next-day t+1. Cost charged on each exposure switch. Idle cash earns 6.5%/yr.
Reports vs the plain always-in basket. NOTE: tax on full-book exits flagged via switches/yr.
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from st_weekly_engine import get_conn, load_daily, ROOT
sys.path.insert(0, ROOT)
from services.technical_indicators import calc_supertrend

RES = os.path.join(HERE, '..', 'results')
START, END = '2010-01-01', '2026-07-07'
COST = 0.0015          # per side on the whole book at a switch
IDLE_D = (1 + 0.065) ** (1 / 252) - 1

BANDS = {'Nifty200': 'nifty200_official.csv', 'Smallcap250': 'niftysmallcap250_official.csv',
         'Nifty500': 'nifty500_proxy.csv'}


def daily_basket(conn, csvf):
    syms = [s.strip() for s in pd.read_csv(os.path.join(ROOT, 'backtest_data', csvf))['Symbol'].dropna()]
    closes = {}
    for s in syms:
        d = load_daily(conn, s)
        if d.empty:
            continue
        closes[s] = d['close']
    px = pd.DataFrame(closes).sort_index()
    px = px[(px.index >= START) & (px.index <= END)]
    return px.pct_change().mean(axis=1, skipna=True).fillna(0.0)   # EW daily basket return


def market_signals(conn):
    nb = load_daily(conn, 'NIFTYBEES')
    nb = nb[(nb.index >= '2008-01-01') & (nb.index <= END)]
    c = nb['close']
    sig = {}
    sig['sma200'] = (c > c.rolling(200).mean())
    sig['sma50'] = (c > c.rolling(50).mean())
    for ap, m, name in [(10, 3, 'dST_10_3'), (7, 3, 'dST_7_3'), (20, 3, 'dST_20_3')]:
        _, d = calc_supertrend(nb[['high', 'low', 'close']].assign(open=nb['open']), ap, m)
        sig[name] = (pd.Series(d.values, index=nb.index) == 1)
    # vol-spike: exit when 10d realized vol > trailing 1y 90th pct (panic gauge, research/70 spirit)
    rv = c.pct_change().rolling(10).std() * np.sqrt(252)
    thr = rv.rolling(252).quantile(0.90)
    sig['volcalm'] = ~(rv > thr)     # in when NOT spiking
    return {k: v.reindex(c.index) for k, v in sig.items()}, c.index


def stats(nav):
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    r = nav.pct_change().dropna()
    sh = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() else 0
    return c * 100, dd * 100, (c / abs(dd) if dd else 0), sh


def apply_overlay(bask_ret, exposure, tax=False):
    """exposure in {0,1} aligned to bask_ret index (already lagged). Returns NAV + n_switches.
    If tax=True, model STCG15%/LTCG10% realised on the whole book at each 1->0 exit
    (cost-basis tracking, holding-period aware). Sell-the-cash-basket assumption."""
    exp = exposure.reindex(bask_ret.index).ffill().fillna(1.0).astype(float).values
    br = bask_ret.values
    idx = bask_ret.index
    n = len(br)
    nav = np.empty(n)
    v = 1.0
    basis = 1.0                     # cost basis of the currently-held book
    entry_i = 0
    nsw = 0
    for i in range(n):
        prev = exp[i - 1] if i > 0 else 1.0
        cur = exp[i]
        if cur != prev:
            nsw += 1
            v *= (1 - COST)         # switching cost on the book
            if tax and prev > 0.5 and cur <= 0.5:   # full exit -> realise gains
                gain = v - basis
                if gain > 0:
                    held_yrs = (idx[i] - idx[entry_i]).days / 365.25
                    v -= gain * (0.10 if held_yrs > 1 else 0.15)
            if cur > 0.5:           # (re)entry -> reset basis + holding clock
                basis = v
                entry_i = i
        v *= (1 + (br[i] if cur > 0.5 else IDLE_D))
        nav[i] = v
    return pd.Series(nav, index=idx), nsw


def main():
    conn = get_conn()
    sig, idx = market_signals(conn)
    out = []
    peryear_store = {}
    for band, csvf in BANDS.items():
        bret = daily_basket(conn, csvf)
        base_nav = pd.Series((1 + bret.values).cumprod(), index=bret.index)
        c0, d0, cal0, sh0 = stats(base_nav)
        print(f"\n===== {band} =====", flush=True)
        print(f"{'Overlay':<16}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>8}{'Sharpe':>7}{'sw/yr':>7}"
              f"{'CAGRtax':>8}{'Caltax':>8}", flush=True)
        print(f"{'PLAIN (tax-def)':<16}{c0:>6.1f}%{d0:>7.1f}%{cal0:>8.2f}{sh0:>7.2f}{'0':>7}"
              f"{c0:>7.1f}%{cal0:>8.2f}", flush=True)
        yrs = (bret.index[-1] - bret.index[0]).days / 365.25
        for name in ['sma200', 'sma50', 'dST_20_3', 'dST_10_3', 'dST_7_3', 'volcalm']:
            expo = sig[name].shift(1)          # act next day (causal)
            nav, nsw = apply_overlay(bret, expo, tax=False)
            navt, _ = apply_overlay(bret, expo, tax=True)
            c, d, cal, sh = stats(nav)
            ct, dt, calt, sht = stats(navt)
            print(f"{name:<16}{c:>6.1f}%{d:>7.1f}%{cal:>8.2f}{sh:>7.2f}{nsw/yrs:>7.1f}"
                  f"{ct:>7.1f}%{calt:>8.2f}", flush=True)
            out.append(dict(band=band, overlay=name, cagr=round(c, 1), maxdd=round(d, 1),
                            calmar=round(cal, 2), sharpe=round(sh, 2), sw_yr=round(nsw / yrs, 1),
                            cagr_nettax=round(ct, 1), calmar_nettax=round(calt, 2),
                            d_cagr=round(c - c0, 1), d_dd=round(d - d0, 1)))
        peryear_store[band] = (base_nav, {n: apply_overlay(bret, sig[n].shift(1))[0] for n in ['sma200', 'dST_10_3', 'dST_7_3']})
    conn.close()
    pd.DataFrame(out).to_csv(os.path.join(RES, 'crash_overlay_summary.csv'), index=False)

    # per-year for Nifty200: plain vs best two overlays, focus on crash years
    print("\n\n==== Nifty200 per-year: PLAIN vs sma200 vs dST_10_3 vs dST_7_3 (annual %) ====")
    base, ov = peryear_store['Nifty200']
    def yr(nav, y):
        s = nav[nav.index.year == y]
        return 100 * (s.iloc[-1] / s.iloc[0] - 1) if len(s) >= 2 else float('nan')
    print(f"{'Yr':<6}{'PLAIN':>8}{'sma200':>8}{'dST10_3':>9}{'dST7_3':>8}")
    for y in range(2010, 2027):
        print(f"{y:<6}{yr(base,y):>7.1f}{yr(ov['sma200'],y):>8.1f}{yr(ov['dST_10_3'],y):>9.1f}{yr(ov['dST_7_3'],y):>8.1f}")
    print('\nDONE -> crash_overlay_summary.csv')


if __name__ == '__main__':
    main()
