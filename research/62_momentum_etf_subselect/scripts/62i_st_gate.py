"""
research/62 phase — swap the momentum book's regime gate (NIFTYBEES 100-DMA, weekly)
for a daily-SuperTrend gate on NIFTYBEES. Everything else identical to the WINNER
config (rsblend, N=8, buffer=22, donchian=15, gate_mode=cash, STCG 20%).

A/B: no-gate vs 100-DMA (current) vs 50-DMA vs daily-ST(7,3)/(10,3)/(14,3) as the risk-off signal.
Gross AND net-post-tax, per-year, Calmar/DD.
"""
import os, sys, importlib.util
import numpy as np, pandas as pd
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
from services.technical_indicators import calc_supertrend

# import the digit-prefixed engine module
spec = importlib.util.spec_from_file_location('mom62', os.path.join(HERE, '62_mom30_subselect.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

WIN = dict(score_fn='rsblend', N=8, buffer=22, donchian=15)   # research/62 winner
STCG = 0.20


def nifty_st_roff(dates, atr, mult):
    """daily SuperTrend on NIFTYBEES -> {date: risk_off(bool)} (dir==-1)."""
    db = os.path.join(ROOT, 'backtest_data', 'market_data.db')
    c = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT date,open,high,low,close FROM market_data_unified "
                           "WHERE symbol='NIFTYBEES' AND timeframe='day' ORDER BY date", c)
    c.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['open', 'high', 'low', 'close']).set_index('date')
    df = df[df['close'] > 0]
    _, d = calc_supertrend(df[['high', 'low', 'close']].assign(open=df['open']), atr, mult)
    ro = (pd.Series(d.values, index=df.index) == -1)
    ro = ro.reindex(dates, method='ffill').fillna(False)
    return {dt: bool(v) for dt, v in ro.items()}


def summ(g):
    return dict(cagr=g['cagr'], dd=g['dd'], sharpe=g['sharpe'], calmar=g['calmar'], py=g['py'],
                derisk=g['st']['gate_derisk'])


def main():
    print('Loading price+tv ...', flush=True)
    close, tv = m.rs2.load()
    dates = close.index[close.index >= m.START]
    print(f'  {close.shape[1]} syms, {dates.min().date()}..{dates.max().date()}', flush=True)

    # precompute ST risk-off signals on NIFTYBEES
    roff = {k: nifty_st_roff(close.index, a, mm) for k, (a, mm) in
            {'st7': (7, 3), 'st10': (10, 3), 'st14': (14, 3)}.items()}

    configs = [
        ('NO-GATE',            dict(gate=None)),
        ('100-DMA (current)',  dict(gate=100)),
        ('50-DMA',             dict(gate=50)),
        ('dST(7,3)',           dict(gate=1, gate_roff=roff['st7'])),
        ('dST(10,3)',          dict(gate=1, gate_roff=roff['st10'])),
        ('dST(14,3)',          dict(gate=1, gate_roff=roff['st14'])),
    ]

    bn = m.bench_nav(close); bm = m.stats_from_nav(bn)
    print(f"\n[ref] NIFTYBEES CAGR={bm['cagr']:.1f}% DD={bm['dd']:.1f}% Calmar={bm['calmar']:.2f}\n", flush=True)

    print(f"{'Gate signal':<20}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>7}{'Sharpe':>7}"
          f"{'nCAGR':>7}{'nDD':>8}{'nCal':>7}{'derisk':>7}", flush=True)
    rows = {}
    for lbl, kw in configs:
        gg = m.run(close, tv, WIN['score_fn'], WIN['N'], WIN['buffer'],
                   donchian=WIN['donchian'], **kw)                    # gross
        tt = m.run(close, tv, WIN['score_fn'], WIN['N'], WIN['buffer'],
                   donchian=WIN['donchian'], stcg=STCG, **kw)         # net post-tax
        rows[lbl] = (summ(gg), summ(tt))
        print(f"{lbl:<20}{gg['cagr']:>6.1f}%{gg['dd']:>7.1f}%{gg['calmar']:>7.2f}{gg['sharpe']:>7.2f}"
              f"{tt['cagr']:>6.1f}%{tt['dd']:>7.1f}%{tt['calmar']:>7.2f}{gg['st']['gate_derisk']:>7}",
              flush=True)

    # per-year net-post-tax: current vs best ST
    print("\n== Per-year NET-post-tax CAGR%: 100-DMA vs dST(10,3) vs dST(7,3) vs NIFTYBEES ==", flush=True)
    a = rows['100-DMA (current)'][1]['py']; b = rows['dST(10,3)'][1]['py']
    cc = rows['dST(7,3)'][1]['py']; nb = bm['py']
    yrs = sorted(set(a.index) | set(b.index))
    print(f"{'Yr':<6}{'100DMA':>8}{'dST10':>8}{'dST7':>8}{'NIFTY':>8}", flush=True)
    for y in yrs:
        print(f"{y:<6}{a.get(y,float('nan')):>8.1f}{b.get(y,float('nan')):>8.1f}"
              f"{cc.get(y,float('nan')):>8.1f}{nb.get(y,float('nan')):>8.1f}", flush=True)

    out = []
    for lbl, (gg, tt) in rows.items():
        out.append(dict(gate=lbl, gross_cagr=round(gg['cagr'],1), gross_dd=round(gg['dd'],1),
                        gross_calmar=round(gg['calmar'],2), net_cagr=round(tt['cagr'],1),
                        net_dd=round(tt['dd'],1), net_calmar=round(tt['calmar'],2),
                        derisk_events=gg['derisk']))
    pd.DataFrame(out).to_csv(os.path.join(HERE, '..', 'results', '62i_st_gate.csv'), index=False)
    print('\nDONE -> results/62i_st_gate.csv', flush=True)


if __name__ == '__main__':
    main()
