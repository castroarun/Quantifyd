"""
research/161 — 30-seed decision cells, robustness, controls and the YoY table.

Runs the pre-registered decision set at 30 seeds:
  WINNER      X=60, depth>=20%, K=none, saucer off, ST(14,4) no stop
  NEIGHBOUR   X=40, same
  PLAIN_ATH   X=0, no filters, ST(14,4) no stop      (what the age/depth filter is added to)
  OA_PROXY    X=0, no filters, 15-SMA trail + -8% stop, liquidity >= Rs5cr (OA's own exits)
  WIN_VOL     winner + K>=2                          (what volume confirmation costs)
  WIN_SAUCER  winner + saucer on                      (what the r/159 shape costs)
  RANDOM      date-matched random entries from the liquid universe

For each: cost ladder, CAGR with and without the 5.5% idle-cash yield, two windows
(window drawdown from the FULL curve's peak), outlier dependence, tradeability gate,
and the house YoY table.
"""
import json
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt_core as B

RES = Path(__file__).resolve().parents[1] / 'results'
DB = Path(__file__).resolve().parents[3] / 'backtest_data' / 'market_data.db'
SEEDS = list(range(1, 31))
REARM = 60

CELLS = [
    ('WINNER',     dict(X=60, dep=20.0, K=0.0, sau=0, ex='ST_14_4', hard=False, liq=2.0)),
    ('NEIGHBOUR',  dict(X=40, dep=20.0, K=0.0, sau=0, ex='ST_14_4', hard=False, liq=2.0)),
    ('PLAIN_ATH',  dict(X=0,  dep=0.0,  K=0.0, sau=0, ex='ST_14_4', hard=False, liq=2.0)),
    ('OA_PROXY',   dict(X=0,  dep=0.0,  K=0.0, sau=0, ex='SMA15',   hard=True,  liq=5.0)),
    ('WIN_VOL',    dict(X=60, dep=20.0, K=2.0, sau=0, ex='ST_14_4', hard=False, liq=2.0)),
    ('WIN_SAUCER', dict(X=60, dep=20.0, K=0.0, sau=1, ex='ST_14_4', hard=False, liq=2.0)),
    ('WINNER_5CR', dict(X=60, dep=20.0, K=0.0, sau=0, ex='ST_14_4', hard=False, liq=5.0)),
]


def rearm(df):
    keep = []
    for _, g in df.sort_values(['symbol', 'hist_bars']).groupby('symbol', sort=False):
        last = -10 ** 9
        for idx, hb in zip(g.index, g['hist_bars'].to_numpy()):
            if hb - last >= REARM:
                keep.append(idx); last = hb
    return df.loc[keep]


def window_dd(nav, cal, lo, hi):
    s = pd.Series(nav, index=pd.to_datetime(cal)).dropna()
    dd = s / s.cummax() - 1.0
    sub = dd[(dd.index >= lo) & (dd.index <= hi)]
    return float(sub.min() * 100) if len(sub) else np.nan


def yoy(nav, cal):
    s = pd.Series(nav, index=pd.to_datetime(cal)).dropna()
    dd = s / s.cummax() - 1.0
    out = {}
    for y, g in s.groupby(s.index.year):
        prev = s[s.index < g.index[0]]
        base = prev.iloc[-1] if len(prev) else g.iloc[0]
        out[int(y)] = (round(100 * (g.iloc[-1] / base - 1), 1),
                       round(100 * dd[dd.index.year == y].min(), 1))
    return out


def main():
    t0 = time.time()
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    cal = [d for d in pd.read_sql_query(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "AND symbol='NIFTYBEES' ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']
    nb = pd.read_sql_query("SELECT date,close FROM market_data_unified WHERE timeframe='day' "
                           "AND symbol='NIFTYBEES' ORDER BY date", con)
    nb = nb[nb['date'].isin(set(cal))].set_index('date')['close'].reindex(cal).ffill()
    bh = (nb / nb.iloc[0] * B.START_CAPITAL).to_numpy()
    con.close()
    bhm = B.metrics(bh, cal)
    panel = pickle.load(open(RES / 'panel161.pkl', 'rb'))
    raw = pd.read_csv(RES / 'ath_events.csv')
    raw = raw[raw['symbol'].isin(panel.close) & raw['entry_date'].isin(panel.pos)]
    raw['entry_i'] = raw['entry_date'].map(panel.pos)
    lm = np.load(RES.parents[1] / '159_rounding_base_breakout' / 'results' / 'liquid_matrix.npz',
                 allow_pickle=True)
    liquid, lsyms = lm['liquid'], list(lm['symbols'])
    liq_idx = {s: i for i, s in enumerate(lsyms)}

    print('=== NIFTYBEES B&H: CAGR %.2f%%  MaxDD %.2f%%  Calmar %.3f (%.1f yrs) ===\n'
          % (bhm['cagr'], bhm['maxdd'], bhm['calmar'], bhm['years']), flush=True)

    def build(c):
        s = raw[raw['tv20_cr'] >= c['liq']]
        if c['X']:
            s = s[s['x_bars'] >= c['X']]
        if c['dep']:
            s = s[s['depth_pct'] >= c['dep']]
        if c['K']:
            s = s[s['vol_mult'] >= c['K']]
        if c['sau']:
            s = s[s['saucer_ok'] == 1]
        s = rearm(s)
        return [dict(symbol=a, entry_i=int(b)) for a, b in zip(s['symbol'], s['entry_i'])]

    report, curves = {}, {}
    print('%-11s %9s %9s %9s %9s %8s %7s %7s %7s %8s %7s'
          % ('cell', 'CAGRmed', 'worst', 'best', 'MaxDD', 'Calmar', 'WR%', 'avgW', 'avgL',
             'exp%', 'tr/yr'), flush=True)
    for name, c in CELLS:
        ev = build(c)
        cfg = dict(exit=c['ex'], hard_stop=c['hard'], time_stop=0, cost_bps=25.0, gate_ok=None)
        ms, navs = [], []
        for sd in SEEDS:
            nav, tr = B.simulate(ev, panel, cfg, sd)
            m = B.metrics(nav, cal, tr)
            if m:
                ms.append((m, tr)); navs.append(nav)
        cg = sorted(m['cagr'] for m, _ in ms)
        med_i = int(np.argsort([m['cagr'] for m, _ in ms])[len(ms) // 2])
        r = dict(n_events=len(ev), cagr_med=float(np.median(cg)), cagr_worst=cg[0],
                 cagr_best=cg[-1],
                 maxdd_med=float(np.median([m['maxdd'] for m, _ in ms])),
                 maxdd_worst=float(min(m['maxdd'] for m, _ in ms)),
                 calmar_med=float(np.median([m['calmar'] for m, _ in ms])))
        for k in ('win_rate', 'avg_win', 'avg_loss', 'expectancy', 'max_loss_streak',
                  'trades_per_yr', 'trades', 'sharpe'):
            r[k] = float(np.median([m.get(k, np.nan) for m, _ in ms]))
        # cost ladder + idle-cash off
        for bps in (40.0, 60.0):
            cc = dict(cfg); cc['cost_bps'] = bps
            v = [B.metrics(B.simulate(ev, panel, cc, sd)[0], cal)['cagr'] for sd in SEEDS[:10]]
            r['cagr_%dbps' % int(bps)] = float(np.median(v))
        cz = dict(cfg); cz['idle_yield'] = 0.0
        v = [B.metrics(B.simulate(ev, panel, cz, sd)[0], cal)['cagr'] for sd in SEEDS[:10]]
        r['cagr_no_idle'] = float(np.median(v))
        # windows
        nav_med = navs[med_i]
        for lo, hi, lab in (('2005-01-03', '2015-12-31', 'pre2016'),
                            ('2016-01-01', '2026-12-31', 'post2016')):
            idx = [i for i, d in enumerate(cal) if lo <= d <= hi]
            sl = slice(idx[0], idx[-1] + 1)
            r['%s_cagr' % lab] = B.metrics(nav_med[sl], cal[sl]).get('cagr')
            r['%s_dd' % lab] = window_dd(nav_med, cal, lo, hi)
            r['%s_bh_cagr' % lab] = B.metrics(bh[sl], cal[sl]).get('cagr')
            r['%s_bh_dd' % lab] = window_dd(bh, cal, lo, hi)
        # outliers
        t = pd.DataFrame(ms[med_i][1])
        r['outlier_all'] = float((1 + t.ret_pct / 100).prod())
        r['outlier_drop10'] = float((1 + t.drop(t.nlargest(10, 'ret_pct').index).ret_pct / 100).prod())
        r['outlier_cap50'] = float((1 + t.ret_pct.clip(upper=50) / 100).prod())
        r['yoy'] = yoy(nav_med, cal)
        report[name] = r
        curves[name] = nav_med
        print('%-11s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.3f %6.1f%% %7.2f %7.2f %7.2f%% %7.1f'
              % (name, r['cagr_med'], r['cagr_worst'], r['cagr_best'], r['maxdd_med'],
                 r['calmar_med'], r['win_rate'], r['avg_win'], r['avg_loss'],
                 r['expectancy'], r['trades_per_yr']), flush=True)

    # ---- date-matched random control against the WINNER -------------------------
    win_ev = build(dict(CELLS[0][1]))
    by_day = defaultdict(int)
    for e in win_ev:
        by_day[e['entry_i']] += 1
    pool = [s for s in panel.close if s in liq_idx]
    ms = []
    for sd in SEEDS[:10]:
        rng = np.random.default_rng(50_000 + sd)
        ev = []
        for day, k in by_day.items():
            ok = [s for s in pool if liquid[liq_idx[s], day] and np.isfinite(panel.open[s][day])]
            if ok:
                for j in rng.choice(len(ok), size=min(k, len(ok)), replace=False):
                    ev.append(dict(symbol=ok[j], entry_i=day))
        nav, tr = B.simulate(ev, panel, dict(exit='ST_14_4', hard_stop=False, time_stop=0,
                                             cost_bps=25.0, gate_ok=None), sd)
        m = B.metrics(nav, cal, tr)
        if m:
            ms.append(m)
    cg = sorted(m['cagr'] for m in ms)
    report['RANDOM'] = dict(cagr_med=float(np.median(cg)), cagr_worst=cg[0], cagr_best=cg[-1],
                            maxdd_med=float(np.median([m['maxdd'] for m in ms])),
                            calmar_med=float(np.median([m['calmar'] for m in ms])),
                            win_rate=float(np.median([m.get('win_rate', np.nan) for m in ms])),
                            expectancy=float(np.median([m.get('expectancy', np.nan) for m in ms])),
                            seeds=len(ms))
    print('%-11s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.3f %6.1f%%           %7.2f%%'
          % ('RANDOM', report['RANDOM']['cagr_med'], report['RANDOM']['cagr_worst'],
             report['RANDOM']['cagr_best'], report['RANDOM']['maxdd_med'],
             report['RANDOM']['calmar_med'], report['RANDOM']['win_rate'],
             report['RANDOM']['expectancy']), flush=True)

    report['BH'] = bhm
    report['BH']['yoy'] = yoy(bh, cal)
    json.dump(report, open(RES / 'final161.json', 'w'), indent=1, default=str)
    np.savez_compressed(RES / 'curves161.npz', dates=np.array(cal), bh=bh,
                        **{k: v for k, v in curves.items()})
    print('\nwrote final161.json + curves161.npz  (%.0fs)' % (time.time() - t0))


if __name__ == '__main__':
    main()
