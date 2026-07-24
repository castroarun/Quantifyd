"""
G3 robustness (research/73): param sensitivity + OOS split on the winning
Nifty200 core config (blind ST exit, no gate, no booking, net cost+tax).
Sweep ATR period x multiplier; report full + OOS (train<=2019 / test 2020+).
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from st_weekly_engine import get_conn, load_daily, to_weekly, ROOT
sys.path.insert(0, ROOT)
from services.technical_indicators import calc_supertrend
from g4_portfolio import build_panel, bench_weekly, run_portfolio, metrics, per_year

RES = os.path.join(HERE, '..', 'results')

CFG = dict(max_pos=16, weight=0.0625, cost_side=0.0015, gate=False, booking=False,
           idle_rate=0.065, tax=True)


def run(atr, mult, start, end):
    conn = get_conn()
    syms = [s.strip() for s in pd.read_csv(os.path.join(ROOT, 'backtest_data', 'nifty200_official.csv'))['Symbol'].dropna()]
    panel = build_panel(conn, syms, atr_period=atr, multiplier=mult)
    bench_w, risk_on = bench_weekly(conn, start, end)
    conn.close()
    weeks = [w for w in bench_w.index if pd.Timestamp(start) <= w <= pd.Timestamp(end)]
    bw = bench_w['close']
    eq, trades = run_portfolio(panel, weeks, bw, {d: bool(v) for d, v in risk_on.items()}, CFG)
    m = metrics(eq, bw)
    return m, len(trades)


def main():
    print('=== G3 param sensitivity — Nifty200 core (blind ST, no gate/booking) ===', flush=True)
    print(f"{'atr':>4} {'mult':>4} | {'CAGR':>6} {'MaxDD':>7} {'Calmar':>6} {'Sharpe':>6} {'trades':>6}", flush=True)
    rows = []
    for atr in [7, 10, 14]:
        for mult in [2.0, 3.0, 4.0]:
            m, nt = run(atr, mult, '2010-01-01', '2026-07-07')
            rows.append((atr, mult, m))
            print(f"{atr:>4} {mult:>4} | {m['cagr']:>5.1f}% {m['maxdd']:>6.1f}% "
                  f"{m['calmar']:>6.2f} {m['sharpe']:>6.2f} {nt:>6}", flush=True)
    # OOS split on base 10/3
    print('\n=== OOS split (10,3) ===', flush=True)
    for lbl, s, e in [('train 2010-2019', '2010-01-01', '2019-12-31'),
                      ('test  2020-2026', '2020-01-01', '2026-07-07')]:
        m, nt = run(10, 3.0, s, e)
        print(f"  {lbl}: CAGR {m['cagr']:.1f}%  MaxDD {m['maxdd']:.1f}%  Calmar {m['calmar']:.2f}  "
              f"bench {m['bench_cagr']:.1f}%/Cal {m['bench_calmar']:.2f}  trades={nt}", flush=True)
    pd.DataFrame([dict(atr=a, mult=mm, **{k: round(v, 2) if isinstance(v, float) else v for k, v in m.items()}) for a, mm, m in rows]).to_csv(os.path.join(RES, 'g3_param_sens.csv'), index=False)
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
