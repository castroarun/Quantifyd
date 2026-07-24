#!/usr/bin/env python3
"""Phase B: filter-overlay marginal-add study on the RSI 70/40 regime system.

Question: do trend/momentum filters MATERIALLY RESCUE the single-name RSI-regime
system, or do they just reduce exposure? Base = entry 70, exit 40, len 14, cost 15bps.

Targets:
  - RELIANCE (single name)
  - BASKET = equal-weight mean of per-name single-name NAV curves for
    RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC, LT.

For each filter config: net CAGR, gross CAGR, MaxDD, Calmar, Sharpe, n_trades,
exposure%, DELTA vs base. Decompose Calmar gains: higher return vs lower exposure.

Incremental CSV, resumable, progress per config.
"""
import csv, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rsi_regime_engine as E

RESULTS = os.path.join(os.path.dirname(HERE), 'results')
os.makedirs(RESULTS, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULTS, 'phaseB_filters.csv')

START = '2015-01-01'
END = None
BASE = dict(entry_th=70, exit_th=40, rsi_len=14, cost_bps=15.0)
BASKET = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'ITC', 'LT']
BENCH = 'NIFTYBEES'

# Filter configs: label -> kwargs passed to E.run beyond BASE
FILTERS = [
    ('base',              dict()),
    ('ma200',             dict(ma_len=200)),
    ('ma50',              dict(ma_len=50)),
    ('adx20',             dict(adx_min=20)),
    ('adx25',             dict(adx_min=25)),
    ('wrsi50',            dict(wrsi_min=50)),
    ('wrsi60',            dict(wrsi_min=60)),
    ('st_regime',         dict(st_regime=True)),
    ('donchian20',        dict(donchian=20)),
    # combos (best-2 style; filled from singles that helped)
    ('ma200_adx20',       dict(ma_len=200, adx_min=20)),
    ('ma200_wrsi50',      dict(ma_len=200, wrsi_min=50)),
    ('ma200_donchian20',  dict(ma_len=200, donchian=20)),
    ('ma50_adx20',        dict(ma_len=50, adx_min=20)),
]

FIELDNAMES = ['target', 'label', 'net_cagr', 'gross_cagr', 'maxdd', 'calmar',
              'sharpe', 'n_trades', 'exposure_pct', 'total_return_x',
              'd_net_cagr', 'd_maxdd', 'd_calmar', 'd_exposure',
              'verdict', 'bh_cagr', 'bh_maxdd']


def exposure_from_trades(df, trades):
    """Fraction of trading days spent in-position (causal, from trade dates)."""
    if not trades:
        return 0.0
    idx = df.index
    days_in = 0
    for t in trades:
        e = pd.Timestamp(t['entry']); x = pd.Timestamp(t['exit'])
        days_in += int(((idx >= e) & (idx <= x)).sum())
    return round(100.0 * days_in / len(idx), 2)


def build_nav(df, kw, cost_bps):
    """Return (net_eqs, gross_eqs, trades) for one name with given filter kwargs."""
    net_eqs, trades, _ = E.run(df, BASE['entry_th'], BASE['exit_th'],
                               BASE['rsi_len'], cost_bps, **kw)
    gross_eqs, _, _ = E.run(df, BASE['entry_th'], BASE['exit_th'],
                            BASE['rsi_len'], 0.0, **kw)
    return net_eqs, gross_eqs, trades


def basket_nav(dfs, kw, cost_bps):
    """Equal-weight mean of per-name NAV curves on common dates.
    Returns (net_eqs, gross_eqs, avg_exposure, total_trades)."""
    net_curves, gross_curves, exps, ntr = [], [], [], 0
    for sym, df in dfs.items():
        n, g, tr = build_nav(df, kw, cost_bps)
        net_curves.append(n.rename(sym))
        gross_curves.append(g.rename(sym))
        exps.append(exposure_from_trades(df, tr))
        ntr += len(tr)
    # align on intersection of dates, forward-fill within each then dropna on union-intersection
    net_df = pd.concat(net_curves, axis=1).dropna()
    gross_df = pd.concat(gross_curves, axis=1).dropna()
    # renormalise each column to start at 1 on the common start, then average
    net_df = net_df / net_df.iloc[0]
    gross_df = gross_df / gross_df.iloc[0]
    net_avg = net_df.mean(axis=1)
    gross_avg = gross_df.mean(axis=1)
    return net_avg, gross_avg, round(float(np.mean(exps)), 2), ntr


def _f(x):
    try:
        return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
    except Exception:
        return 0.0


def row_for(target, label, net_eqs, gross_eqs, exposure, n_trades, base_row, bh_cagr, bh_maxdd):
    m_net = E.metrics(net_eqs)
    m_gross = E.metrics(gross_eqs)
    row = dict(
        target=target, label=label,
        net_cagr=m_net['cagr'], gross_cagr=m_gross['cagr'],
        maxdd=m_net['maxdd'], calmar=m_net['calmar'], sharpe=m_net['sharpe'],
        n_trades=n_trades, exposure_pct=exposure,
        total_return_x=m_net['total_return_x'],
        bh_cagr=bh_cagr, bh_maxdd=bh_maxdd,
    )
    if base_row is not None:
        row['d_net_cagr'] = round(_f(row['net_cagr']) - _f(base_row['net_cagr']), 2)
        row['d_maxdd'] = round(_f(row['maxdd']) - _f(base_row['maxdd']), 2)
        row['d_calmar'] = round(_f(row['calmar']) - _f(base_row['calmar']), 2)
        row['d_exposure'] = round(_f(row['exposure_pct']) - _f(base_row['exposure_pct']), 2)
        # Decompose: did Calmar improve via return or via exposure cut?
        dc = row['d_calmar']; dret = row['d_net_cagr']; dexp = row['d_exposure']
        if dc > 0.05:
            if dret > 0.3:
                row['verdict'] = 'ADDS RETURN'
            elif dexp < -2:
                row['verdict'] = 'JUST CUTS EXPOSURE/DD'
            else:
                row['verdict'] = 'marginal'
        elif dc < -0.05:
            row['verdict'] = 'WORSE'
        else:
            row['verdict'] = 'neutral'
    else:
        row['d_net_cagr'] = 0.0; row['d_maxdd'] = 0.0
        row['d_calmar'] = 0.0; row['d_exposure'] = 0.0
        row['verdict'] = 'BASE'
    return row


def main():
    t0 = time.time()
    # resume
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            for r in csv.DictReader(f):
                done.add((r['target'], r['label']))
        print(f'Resuming: {len(done)} rows already done')
    else:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # preload
    print('Loading data...', flush=True)
    dfs = {}
    for s in BASKET:
        df = E.load(s, start=START, end=END)
        df = df[df['close'] > 0]
        dfs[s] = df
        print(f'  {s}: {len(df)} rows {df.index[0].date()}..{df.index[-1].date()}', flush=True)
    bench = E.load(BENCH, start=START, end=END)
    bench = bench[bench['close'] > 0]
    m_bench = E.metrics(E.buyhold(bench))
    print(f'  {BENCH} B&H: CAGR={m_bench["cagr"]}% MaxDD={m_bench["maxdd"]}% '
          f'Calmar={m_bench["calmar"]}', flush=True)

    # buy&hold references
    bh = {s: E.metrics(E.buyhold(dfs[s])) for s in BASKET}
    # basket B&H = equal-weight mean of per-name B&H NAVs
    bh_curves = pd.concat([(dfs[s]['close']/dfs[s]['close'].iloc[0]).rename(s)
                           for s in BASKET], axis=1).dropna()
    bh_curves = bh_curves / bh_curves.iloc[0]
    m_bh_basket = E.metrics(bh_curves.mean(axis=1))
    print(f'  BASKET B&H: CAGR={m_bh_basket["cagr"]}% MaxDD={m_bh_basket["maxdd"]}% '
          f'Calmar={m_bh_basket["calmar"]}', flush=True)

    def append(row):
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
            f.flush()

    # ---- RELIANCE single-name ----
    base_rel = None
    rel_df = dfs['RELIANCE']
    for i, (label, kw) in enumerate(FILTERS, 1):
        key = ('RELIANCE', label)
        if key in done:
            print(f'[REL {i}/{len(FILTERS)}] {label} skip (done)', flush=True)
            if label == 'base':
                # need base_row for deltas; recompute lightweight
                pass
            continue
        ts = time.time()
        net, gross, trades = build_nav(rel_df, kw, BASE['cost_bps'])
        exp = exposure_from_trades(rel_df, trades)
        row = row_for('RELIANCE', label, net, gross, exp, len(trades),
                      base_rel, bh['RELIANCE']['cagr'], bh['RELIANCE']['maxdd'])
        if label == 'base':
            base_rel = row
        append(row)
        print(f'[REL {i}/{len(FILTERS)}] {label} {time.time()-ts:.0f}s | '
              f'net={row["net_cagr"]}% DD={row["maxdd"]}% Cal={row["calmar"]} '
              f'exp={row["exposure_pct"]}% trades={row["n_trades"]} '
              f'-> {row["verdict"]}', flush=True)

    # ---- BASKET ----
    base_bkt = None
    for i, (label, kw) in enumerate(FILTERS, 1):
        key = ('BASKET', label)
        if key in done:
            print(f'[BKT {i}/{len(FILTERS)}] {label} skip (done)', flush=True)
            continue
        ts = time.time()
        net, gross, exp, ntr = basket_nav(dfs, kw, BASE['cost_bps'])
        row = row_for('BASKET', label, net, gross, exp, ntr,
                      base_bkt, m_bh_basket['cagr'], m_bh_basket['maxdd'])
        if label == 'base':
            base_bkt = row
        append(row)
        print(f'[BKT {i}/{len(FILTERS)}] {label} {time.time()-ts:.0f}s | '
              f'net={row["net_cagr"]}% DD={row["maxdd"]}% Cal={row["calmar"]} '
              f'exp={row["exposure_pct"]}% trades={row["n_trades"]} '
              f'-> {row["verdict"]}', flush=True)

    print(f'DONE in {time.time()-t0:.0f}s. CSV: {OUTPUT_CSV}', flush=True)
    print(f'BENCH NIFTYBEES CAGR={m_bench["cagr"]}% DD={m_bench["maxdd"]}%')


if __name__ == '__main__':
    import logging
    logging.disable(logging.WARNING)
    main()
