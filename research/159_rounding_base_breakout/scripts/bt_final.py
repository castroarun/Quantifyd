"""
research/159 — final analysis for the chosen configuration (STATUS 11.6).

Produces, for the sweep winner:
  * 30-seed band (median [min..max], worst seed) at 25 bps
  * cost ladder 25 / 40 / 60 bps
  * two windows (pre-2016, 2016+), drawdowns measured from the FULL curve's running peak
  * outlier dependence: top-10 trades removed; winners capped at +50% and +100%
  * tradeability gate columns
  * YoY table in the house format (annual return with intra-year max drawdown beneath)
  * correlation and blend value vs Open Alpha / True North (research/154 curves)
  * curves exported for the factsheet

Usage: bt_final.py --exit ST_14_4 --hard 0 --tstop 0 --shelf 15 --k 3 --ath 0.90
                   [--obv 0] [--gate 0]
"""
import argparse
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt_core as B

RES = Path(__file__).resolve().parents[1] / 'results'
DB = Path(__file__).resolve().parents[3] / 'backtest_data' / 'market_data.db'
R154 = Path(__file__).resolve().parents[2] / '154_multi_system_blends' / 'results'
SEEDS = list(range(1, 31))


def ap():
    p = argparse.ArgumentParser()
    p.add_argument('--exit', default='ST_14_4')
    p.add_argument('--hard', type=int, default=0)
    p.add_argument('--tstop', type=int, default=0)
    p.add_argument('--shelf', type=int, default=15)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--ath', default='0.90')
    p.add_argument('--obv', type=int, default=0)
    p.add_argument('--gate', type=int, default=0)
    p.add_argument('--slots', type=int, default=16)
    p.add_argument('--tag', default='')
    return p.parse_args()


def window_dd(nav_full, cal, lo, hi):
    """Drawdown for a window measured from the running peak of the FULL curve (r/154)."""
    s = pd.Series(nav_full, index=pd.to_datetime(cal)).dropna()
    peak = s.cummax()
    sub = (s / peak - 1.0)[(s.index >= lo) & (s.index <= hi)]
    return float(sub.min() * 100) if len(sub) else np.nan


def yoy(nav, cal):
    s = pd.Series(nav, index=pd.to_datetime(cal)).dropna()
    peak = s.cummax()
    dd = s / peak - 1.0
    out = {}
    for y, g in s.groupby(s.index.year):
        prev = s[s.index < g.index[0]]
        base = prev.iloc[-1] if len(prev) else g.iloc[0]
        out[int(y)] = (round(100 * (g.iloc[-1] / base - 1), 1),
                       round(100 * dd[dd.index.year == y].min(), 1))
    return out


def main():
    a = ap()
    B.SLOTS = a.slots
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    cal = [d for d in pd.read_sql_query(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "AND symbol='NIFTYBEES' ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']

    vp = RES / ('rounding_base_events_v3_s%d_k%d_a%s.csv' % (a.shelf, a.k, a.ath))
    src = vp if vp.exists() else (RES / 'rounding_base_events_v3.csv')
    ev = pd.read_csv(src)
    ev = ev[ev['symbol'] != 'SILLYMONKS'].sort_values('pattern_quality', ascending=False)
    ev = ev.drop_duplicates(['symbol', 'trigger_date'])
    if a.obv:
        ev = ev[ev['obv_filter_pass'] == 1]
    print('config: exit=%s hard=%d tstop=%d S=%d K=%d ATH=%s obv=%d gate=%d slots=%d'
          % (a.exit, a.hard, a.tstop, a.shelf, a.k, a.ath, a.obv, a.gate, a.slots))
    print('events source: %s  (%d events)' % (src.name, len(ev)))

    with open(RES / 'panel_full.pkl', 'rb') as f:
        panel = pickle.load(f)
    events = []
    for sym, ed in zip(ev['symbol'], ev['entry_date']):
        if isinstance(ed, str) and panel.has(sym) and ed in panel.pos:
            events.append(dict(symbol=sym, entry_i=panel.pos[ed]))
    print('tradeable: %d' % len(events))

    nb = pd.read_sql_query("SELECT date,close FROM market_data_unified WHERE timeframe='day' "
                           "AND symbol='NIFTYBEES' ORDER BY date", con)
    nb = nb[nb['date'].isin(set(cal))].set_index('date')['close'].reindex(cal).ffill()
    bh = (nb / nb.iloc[0] * B.START_CAPITAL).to_numpy()
    con.close()
    gate = None
    if a.gate:
        g = np.array((nb > nb.rolling(100, min_periods=100).mean()).to_numpy(), copy=True)
        g[:100] = True
        gate = g

    report = {}
    # ---- cost ladder, 30 seeds -----------------------------------------------------
    print('\n--- COST LADDER (30 seeds, after tax) ---')
    print('%-6s %9s %9s %9s %9s %8s %8s' % ('bps', 'CAGRmed', 'worst', 'best', 'MaxDDmed', 'Calmar', 'trades'))
    navs25 = []
    for bps in (25.0, 40.0, 60.0):
        ms, nv = [], []
        for sd in SEEDS:
            nav, tr = B.simulate(events, panel, dict(exit=a.exit, hard_stop=bool(a.hard),
                                                     time_stop=a.tstop, cost_bps=bps,
                                                     gate_ok=gate), sd)
            m = B.metrics(nav, cal, tr)
            if m:
                ms.append((m, tr)); nv.append(nav)
        cg = sorted(m['cagr'] for m, _ in ms)
        print('%-6.0f %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.3f %8.0f'
              % (bps, np.median(cg), cg[0], cg[-1],
                 np.median([m['maxdd'] for m, _ in ms]),
                 np.median([m['calmar'] for m, _ in ms]),
                 np.median([m.get('trades', 0) for m, _ in ms])))
        if bps == 25.0:
            navs25 = nv; base_ms = ms
            report['cagr_med'] = float(np.median(cg)); report['cagr_worst'] = cg[0]
            report['cagr_best'] = cg[-1]
            report['maxdd_med'] = float(np.median([m['maxdd'] for m, _ in ms]))
            report['maxdd_worst'] = float(min(m['maxdd'] for m, _ in ms))
            report['calmar_med'] = float(np.median([m['calmar'] for m, _ in ms]))
            for k in ('trades', 'win_rate', 'avg_win', 'avg_loss', 'expectancy',
                      'max_loss_streak', 'trades_per_yr', 'sharpe'):
                report[k] = float(np.median([m.get(k, np.nan) for m, _ in ms]))

    med_i = int(np.argsort([m['cagr'] for m, _ in base_ms])[len(base_ms) // 2])
    nav_med = navs25[med_i]
    trades_med = base_ms[med_i][1]

    # ---- benchmark ----------------------------------------------------------------
    bhm = B.metrics(bh, cal)
    print('\n--- BENCHMARK NIFTYBEES B&H: CAGR %.2f%%  MaxDD %.2f%%  Calmar %.3f ---'
          % (bhm['cagr'], bhm['maxdd'], bhm['calmar']))
    report['bh'] = bhm

    # ---- two windows ---------------------------------------------------------------
    print('\n--- TWO WINDOWS (window DD from the FULL curve peak) ---')
    print('%-10s %10s %10s %10s %10s' % ('window', 'sysCAGR', 'sysDD', 'bhCAGR', 'bhDD'))
    wins = {}
    for lo, hi, lab in (('2005-01-03', '2015-12-31', 'pre-2016'),
                        ('2016-01-01', '2026-12-31', '2016+')):
        idx = [i for i, d in enumerate(cal) if lo <= d <= hi]
        sl = slice(idx[0], idx[-1] + 1)
        sm = B.metrics(nav_med[sl], cal[sl]); bm = B.metrics(bh[sl], cal[sl])
        sd_ = window_dd(nav_med, cal, lo, hi); bd = window_dd(bh, cal, lo, hi)
        wins[lab] = dict(sys_cagr=sm.get('cagr'), sys_dd=sd_, bh_cagr=bm.get('cagr'), bh_dd=bd)
        print('%-10s %9.2f%% %9.2f%% %9.2f%% %9.2f%%' % (lab, sm['cagr'], sd_, bm['cagr'], bd))
    report['windows'] = wins

    # ---- outlier dependence --------------------------------------------------------
    print('\n--- OUTLIER DEPENDENCE (median-seed trade list, %d trades) ---' % len(trades_med))
    t = pd.DataFrame(trades_med)
    tot = (1 + t['ret_pct'] / 100).prod()
    drop10 = (1 + t.nlargest(10, 'ret_pct')['ret_pct'].pipe(lambda x: t.drop(x.index)['ret_pct']) / 100).prod()
    cap50 = (1 + t['ret_pct'].clip(upper=50) / 100).prod()
    cap100 = (1 + t['ret_pct'].clip(upper=100) / 100).prod()
    print('  product of (1+r): all %.2f | top-10 removed %.2f | capped +50%% %.2f | capped +100%% %.2f'
          % (tot, drop10, cap50, cap100))
    report['outlier'] = dict(all=float(tot), drop10=float(drop10), cap50=float(cap50),
                             cap100=float(cap100))

    # ---- tradeability --------------------------------------------------------------
    print('\n--- TRADEABILITY GATE ---')
    print('  trades %.0f (%.1f/yr) | win %.1f%% | avg win %+.2f%% | avg loss %+.2f%% | '
          'expectancy %+.3f%% | max losing streak %.0f'
          % (report['trades'], report['trades_per_yr'], report['win_rate'], report['avg_win'],
             report['avg_loss'], report['expectancy'], report['max_loss_streak']))

    # ---- YoY ------------------------------------------------------------------------
    y_sys, y_bh = yoy(nav_med, cal), yoy(bh, cal)
    print('\n--- YEAR BY YEAR (return, intra-year DD from full-curve peak) ---')
    print('%-6s %18s %18s' % ('year', 'V3 book', 'NIFTYBEES'))
    for y in sorted(y_sys):
        a1, d1 = y_sys[y]; a2, d2 = y_bh.get(y, (np.nan, np.nan))
        print('%-6d %10.1f%% (%5.1f%%) %10.1f%% (%5.1f%%)' % (y, a1, d1, a2, d2))
    report['yoy_sys'] = y_sys; report['yoy_bh'] = y_bh

    # ---- correlation / blend vs OA and TN -------------------------------------------
    oa_p = R154 / 'oa_navs30.csv'
    if oa_p.exists():
        oa = pd.read_csv(oa_p, parse_dates=['date']).set_index('date')
        oa_nav = oa.median(axis=1)
        sysser = pd.Series(nav_med, index=pd.to_datetime(cal)).dropna()
        j = pd.concat([sysser.rename('sys'), oa_nav.rename('oa')], axis=1).dropna()
        dr = j.pct_change().dropna()
        mr = j.resample('ME').last().pct_change().dropna()
        print('\n--- PORTFOLIO FIT vs OPEN ALPHA (r/154 30-seed median NAV) ---')
        print('  daily corr %.3f | monthly corr %.3f | overlap %d days'
              % (dr['sys'].corr(dr['oa']), mr['sys'].corr(mr['oa']), len(j)))
        report['corr_oa_daily'] = float(dr['sys'].corr(dr['oa']))
        report['corr_oa_monthly'] = float(mr['sys'].corr(mr['oa']))
        for w in (0.10, 0.20, 0.33):
            bl = (1 - w) * (j['oa'] / j['oa'].iloc[0]) + w * (j['sys'] / j['sys'].iloc[0])
            m = B.metrics(bl.to_numpy() * B.START_CAPITAL, [d.strftime('%Y-%m-%d') for d in bl.index])
            print('  OA %.0f%% + V3 %.0f%%: CAGR %.2f%%  MaxDD %.2f%%  Calmar %.3f'
                  % (100 * (1 - w), 100 * w, m['cagr'], m['maxdd'], m['calmar']))
            report['blend_%d' % int(w * 100)] = m
        m_oa = B.metrics((j['oa'] / j['oa'].iloc[0]).to_numpy() * B.START_CAPITAL,
                         [d.strftime('%Y-%m-%d') for d in j.index])
        print('  OA alone over the same overlap: CAGR %.2f%%  MaxDD %.2f%%  Calmar %.3f'
              % (m_oa['cagr'], m_oa['maxdd'], m_oa['calmar']))
        report['oa_alone'] = m_oa

    tag = a.tag or ('%s_h%d_s%d_k%d_a%s_sl%d' % (a.exit, a.hard, a.shelf, a.k, a.ath, a.slots))
    np.savez_compressed(RES / ('final_curve_%s.npz' % tag),
                        nav=nav_med, bh=bh, dates=np.array(cal))
    pd.DataFrame(trades_med).to_csv(RES / ('final_trades_%s.csv' % tag), index=False)
    json.dump(report, open(RES / ('final_report_%s.json' % tag), 'w'), indent=1, default=str)
    print('\nwrote final_{curve,trades,report}_%s' % tag)


if __name__ == '__main__':
    main()
