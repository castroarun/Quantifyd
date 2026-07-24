"""
G1 signal probe (research/73) — does weekly-ST(10,3) timing beat just owning the names?

Per band (Nifty50/200/Midcap150/Smallcap250): every long weekly-ST trade,
gross + net@0.30%, vs (a) buy-&-hold same names, (b) random-duration placebo.
Reports full-period and modern sub-period (2015+). Writes incremental CSVs.
"""
import os, sys, csv, glob, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from st_weekly_engine import (get_conn, load_daily, weekly_st_trades,
                              buy_hold_return, placebo_trades, ROOT)

RESULTS = os.path.join(HERE, '..', 'results')
os.makedirs(RESULTS, exist_ok=True)
COST_RT = 0.003  # 0.30% round-trip

BANDS = {
    'Nifty50': 'nifty50_official.csv',
    'Nifty200': 'nifty200_official.csv',
    'Midcap150': 'niftymidcap150_official.csv',
    'Smallcap250': 'niftysmallcap250_official.csv',
}


def load_symbols(csvfile):
    df = pd.read_csv(os.path.join(ROOT, 'backtest_data', csvfile))
    return [s.strip() for s in df['Symbol'].dropna().tolist()]


def tstat(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def agg(trades, key='net'):
    if not trades:
        return dict(n=0)
    r = np.array([t[key] for t in trades], float)
    wins = r[r > 0]
    losses = r[r <= 0]
    return dict(
        n=len(r), win_pct=100 * len(wins) / len(r),
        mean=100 * r.mean(), median=100 * np.median(r),
        avg_win=100 * wins.mean() if len(wins) else 0.0,
        avg_loss=100 * losses.mean() if len(losses) else 0.0,
        avg_weeks=np.mean([t['weeks'] for t in trades]),
        tstat=tstat(r), total_r=float(r.sum()))


def run_band(name, csvfile, start=None, tag='full'):
    conn = get_conn()
    syms = load_symbols(csvfile)
    all_trades, all_placebo, bh = [], [], []
    t0 = time.time()
    for k, sym in enumerate(syms):
        try:
            daily = load_daily(conn, sym)
        except Exception:
            continue
        if daily.empty:
            continue
        tr = weekly_st_trades(daily, sym, start=start, cost_rt=COST_RT)
        all_trades.extend(tr)
        pb = placebo_trades(daily, sym, tr, start=start, cost_rt=COST_RT)
        all_placebo.extend(pb)
        r = buy_hold_return(daily, start=start)
        if r is not None:
            bh.append(r)
        if (k + 1) % 50 == 0:
            print(f'  [{name}/{tag}] {k+1}/{len(syms)} syms, {len(all_trades)} trades, '
                  f'{time.time()-t0:.0f}s', flush=True)
    conn.close()

    st_net = agg(all_trades, 'net')
    st_gross = agg(all_trades, 'gross')
    pb_net = agg(all_placebo, 'net')
    bh_mean = 100 * np.mean(bh) if bh else 0.0

    # per-trade CSV (heavy) only for full period
    if tag == 'full' and all_trades:
        with open(os.path.join(RESULTS, f'g1_{name}_trades.csv'), 'w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=['symbol', 'entry_dt', 'exit_dt', 'entry_px',
                                                'exit_px', 'weeks', 'year', 'gross', 'net',
                                                'open_at_end'])
            wtr.writeheader()
            for t in all_trades:
                wtr.writerow(t)

    row = dict(band=name, period=tag,
               n_trades=st_net['n'],
               st_win=round(st_net.get('win_pct', 0), 1),
               st_mean_net=round(st_net.get('mean', 0), 2),
               st_median_net=round(st_net.get('median', 0), 2),
               st_mean_gross=round(st_gross.get('mean', 0), 2),
               avg_win=round(st_net.get('avg_win', 0), 2),
               avg_loss=round(st_net.get('avg_loss', 0), 2),
               avg_weeks=round(st_net.get('avg_weeks', 0), 1),
               tstat=round(st_net.get('tstat', 0), 2),
               placebo_mean_net=round(pb_net.get('mean', 0), 2),
               placebo_win=round(pb_net.get('win_pct', 0), 1),
               bh_mean=round(bh_mean, 2),
               edge_vs_placebo=round(st_net.get('mean', 0) - pb_net.get('mean', 0), 2))
    return row, all_trades


def per_year(trades):
    rows = {}
    for t in trades:
        y = t['year']
        rows.setdefault(y, []).append(t['net'])
    out = []
    for y in sorted(rows):
        r = np.array(rows[y], float)
        out.append((y, len(r), round(100 * r.mean(), 2), round(100 * (r > 0).mean(), 1)))
    return out


def main():
    summary_path = os.path.join(RESULTS, 'g1_summary.csv')
    fields = ['band', 'period', 'n_trades', 'st_win', 'st_mean_net', 'st_median_net',
              'st_mean_gross', 'avg_win', 'avg_loss', 'avg_weeks', 'tstat',
              'placebo_mean_net', 'placebo_win', 'bh_mean', 'edge_vs_placebo']
    done = set()
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for r in csv.DictReader(f):
                done.add((r['band'], r['period']))
    write_header = not os.path.exists(summary_path)
    f = open(summary_path, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()

    print('=== G1 signal probe: weekly SuperTrend(10,3) ===', flush=True)
    for name, csvfile in BANDS.items():
        for tag, start in [('full', None), ('2015+', '2015-01-01')]:
            if (name, tag) in done:
                print(f'skip {name}/{tag}', flush=True)
                continue
            print(f'\n>>> {name} / {tag}', flush=True)
            row, trades = run_band(name, csvfile, start=start, tag=tag)
            w.writerow(row); f.flush()
            print(f'  RESULT {name}/{tag}: n={row["n_trades"]} win={row["st_win"]}% '
                  f'meanNet={row["st_mean_net"]}% (gross {row["st_mean_gross"]}%) '
                  f'placebo={row["placebo_mean_net"]}% EDGE={row["edge_vs_placebo"]}pp '
                  f't={row["tstat"]} avgHold={row["avg_weeks"]}w', flush=True)
            if tag == 'full':
                print('  per-year (yr,n,meanNet%,win%):', per_year(trades), flush=True)
    f.close()
    print('\nDONE. Summary ->', summary_path, flush=True)


if __name__ == '__main__':
    main()
