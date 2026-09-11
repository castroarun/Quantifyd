"""
research/159 — the decisive comparison (STATUS 11.4).

Runs the SAME book and the SAME exits on four entry sources, 30 seeds each:

  V3          the saucer -> shelf breakout near the ATH
  CTRL_ATH    every near-ATH volume thrust, NO saucer and NO shelf  (11,862 events)
  CTRL_DM     the same control, DATE-MATCHED to v3: on each v3 trigger date, draw the
              same number of control signals that fired that day
  CTRL_RND    date-matched RANDOM entries from the liquid universe

If V3 does not beat CTRL_DM after tax, the saucer + shelf shape adds nothing over simply
buying strength near the all-time high, and the verdict is
"NO INCREMENTAL EDGE OVER OPEN ALPHA".

Also prints the NIFTYBEES buy-and-hold bar the whole study is judged against.
"""
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
PANEL_PKL = RES / 'panel_full.pkl'
SEEDS = list(range(1, 31))
COST = 25.0


def load_panel(con, need, cal):
    if PANEL_PKL.exists():
        with open(PANEL_PKL, 'rb') as f:
            p = pickle.load(f)
        if not (need - set(p.close)):
            print('panel: cache hit (%d symbols)' % len(p.close), flush=True)
            return p
    print('panel: building for %d symbols...' % len(need), flush=True)
    t = time.time()
    p = B.Panel(con, sorted(need), cal)
    with open(PANEL_PKL, 'wb') as f:
        pickle.dump(p, f, protocol=4)
    print('panel: built %d symbols in %.0fs' % (len(p.close), time.time() - t), flush=True)
    return p


def to_events(df, panel):
    ev = []
    for sym, ed in zip(df['symbol'], df['entry_date']):
        if not isinstance(ed, str) or not panel.has(sym):
            continue
        i = panel.pos.get(ed)
        if i is not None:
            ev.append(dict(symbol=sym, entry_i=i))
    return ev


def main():
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    cal = [d for d in pd.read_sql_query(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "AND symbol='NIFTYBEES' ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']
    pos = {d: i for i, d in enumerate(cal)}

    v3 = pd.read_csv(RES / 'rounding_base_events_v3.csv')
    v3 = v3[v3['symbol'] != 'SILLYMONKS'].sort_values('pattern_quality', ascending=False)
    v3 = v3.drop_duplicates(['symbol', 'trigger_date'])
    ctl = pd.read_csv(RES / 'control_ath_events.csv')
    ctl = ctl[ctl['symbol'] != 'SILLYMONKS']

    need = set(v3['symbol']) | set(ctl['symbol'])
    panel = load_panel(con, need, cal)

    nb = pd.read_sql_query(
        "SELECT date,close FROM market_data_unified WHERE timeframe='day' AND symbol='NIFTYBEES' "
        "ORDER BY date", con)
    nb = nb[nb['date'].isin(set(cal))].set_index('date')['close'].reindex(cal).ffill()
    bh = (nb / nb.iloc[0] * B.START_CAPITAL).to_numpy()
    bhm = B.metrics(bh, cal)
    con.close()

    lm = np.load(RES / 'liquid_matrix.npz', allow_pickle=True)
    liquid, lsyms = lm['liquid'], list(lm['symbols'])

    ev_v3 = to_events(v3, panel)
    ev_ctl = to_events(ctl, panel)
    print('events -> V3 %d | CTRL_ATH %d' % (len(ev_v3), len(ev_ctl)), flush=True)

    # v3 entries per calendar day, for date-matching
    v3_by_day = defaultdict(int)
    for e in ev_v3:
        v3_by_day[e['entry_i']] += 1
    ctl_by_day = defaultdict(list)
    for e in ev_ctl:
        ctl_by_day[e['entry_i']].append(e)
    liq_idx = {s: i for i, s in enumerate(lsyms)}
    pool_syms = [s for s in panel.close if s in liq_idx]

    def make_dm(seed):
        rng = np.random.default_rng(10_000 + seed)
        out = []
        for day, k in v3_by_day.items():
            avail = ctl_by_day.get(day, [])
            if not avail:
                continue
            take = min(k, len(avail))
            for j in rng.choice(len(avail), size=take, replace=False):
                out.append(avail[j])
        return out

    def make_rnd(seed):
        rng = np.random.default_rng(20_000 + seed)
        out = []
        for day, k in v3_by_day.items():
            ok = [s for s in pool_syms if liquid[liq_idx[s], day] and np.isfinite(panel.open[s][day])]
            if not ok:
                continue
            for j in rng.choice(len(ok), size=min(k, len(ok)), replace=False):
                out.append(dict(symbol=ok[j], entry_i=day))
        return out

    print('\n=== NIFTYBEES BUY & HOLD: CAGR %.2f%%  MaxDD %.2f%%  Calmar %.3f (%.1f yrs) ==='
          % (bhm['cagr'], bhm['maxdd'], bhm['calmar'], bhm['years']), flush=True)
    print('=== adoption bar: >20%% after-tax CAGR, beat NIFTYBEES on CAGR *and* DD, beat CTRL_DM ===\n',
          flush=True)

    rows = []
    print('%-10s %-8s %-5s %8s %8s %8s %8s %8s %7s %7s'
          % ('system', 'exit', 'stop', 'CAGRmed', 'CAGRmin', 'CAGRmax', 'MaxDDmed', 'Calmar',
             'trades', 'win%'), flush=True)
    for ex in B.EXITS:
        for hard in (False, True):
            for name in ('V3', 'CTRL_ATH', 'CTRL_DM', 'CTRL_RND'):
                res = []
                for sd in SEEDS:
                    if name == 'V3':
                        ev = ev_v3
                    elif name == 'CTRL_ATH':
                        ev = ev_ctl
                    elif name == 'CTRL_DM':
                        ev = make_dm(sd)
                    else:
                        ev = make_rnd(sd)
                    nav, tr = B.simulate(ev, panel, dict(exit=ex, hard_stop=hard, time_stop=0,
                                                         cost_bps=COST, gate_ok=None), sd)
                    m = B.metrics(nav, cal, tr)
                    if m:
                        res.append(m)
                if not res:
                    continue
                cg = sorted(r['cagr'] for r in res)
                row = dict(system=name, exit=ex, hard_stop=int(hard),
                           cagr_med=float(np.median(cg)), cagr_min=cg[0], cagr_max=cg[-1],
                           maxdd_med=float(np.median([r['maxdd'] for r in res])),
                           maxdd_worst=min(r['maxdd'] for r in res),
                           calmar_med=float(np.median([r['calmar'] for r in res])),
                           trades=float(np.median([r.get('trades', 0) for r in res])),
                           win_rate=float(np.median([r.get('win_rate', np.nan) for r in res])),
                           expectancy=float(np.median([r.get('expectancy', np.nan) for r in res])),
                           seeds=len(res))
                rows.append(row)
                print('%-10s %-8s %-5s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %8.3f %7.0f %6.1f%%'
                      % (name, ex, hard, row['cagr_med'], row['cagr_min'], row['cagr_max'],
                         row['maxdd_med'], row['calmar_med'], row['trades'], row['win_rate']),
                      flush=True)
            print('', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / 'compare_v3_vs_controls.csv', index=False)
    print('wrote %s' % (RES / 'compare_v3_vs_controls.csv'))

    # paired verdict per exit
    print('\n=== PAIRED: V3 minus CTRL_DM (same exit, same seeds) ===')
    for ex in B.EXITS:
        for hard in (0, 1):
            a = df[(df.system == 'V3') & (df.exit == ex) & (df.hard_stop == hard)]
            b = df[(df.system == 'CTRL_DM') & (df.exit == ex) & (df.hard_stop == hard)]
            if len(a) and len(b):
                print('  %-8s stop=%d  V3 %.2f%% vs CTRL_DM %.2f%%  -> %+.2fpp'
                      % (ex, hard, a.cagr_med.iloc[0], b.cagr_med.iloc[0],
                         a.cagr_med.iloc[0] - b.cagr_med.iloc[0]))


if __name__ == '__main__':
    main()
