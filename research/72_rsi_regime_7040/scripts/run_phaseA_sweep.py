#!/usr/bin/env python3
"""Phase A: RELIANCE RSI-regime threshold x length sweep (75 cells).
Incremental, resumable CSV. Ranks by net Calmar / CAGR vs NIFTYBEES & RELIANCE B&H."""
import csv, os, sys, itertools
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rsi_regime_engine as E

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'results' / 'phaseA_sweep.csv'
OUT.parent.mkdir(exist_ok=True)

SYMBOL = os.environ.get('SYM', 'RELIANCE')
START = '2015-01-01'
COST = 15.0
ENTRIES = [60, 65, 70, 75, 80]
EXITS = [30, 35, 40, 45, 50]
LENS = [7, 14, 21]

FIELDS = ['label', 'symbol', 'rsi_len', 'entry', 'exit', 'n_trades', 'win_rate',
          'avg_hold', 'exposure_pct', 'cagr_net', 'cagr_gross', 'sharpe', 'sortino',
          'maxdd', 'calmar', 'vol', 'total_x',
          'bench_cagr', 'bench_maxdd', 'bench_x', 'stock_bh_cagr', 'stock_bh_maxdd',
          'beat_bench_ratio', 'lower_dd_than_bench']


def exposure(eqs):
    return round(100 * float((eqs.pct_change() != 0).mean()), 1)


def main():
    df = E.load(SYMBOL, start=START)
    bench = E.load('NIFTYBEES', start=str(df.index[0].date()))
    m_bench = E.metrics(E.buyhold(bench), 'bench')
    m_stock = E.metrics(E.buyhold(df), 'stock')

    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {r['label'] for r in csv.DictReader(f)}
    write_header = not OUT.exists()
    with open(OUT, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        cells = [(ln, en, ex) for ln, en, ex in itertools.product(LENS, ENTRIES, EXITS) if ex < en]
        for i, (ln, en, ex) in enumerate(cells, 1):
            label = f'{SYMBOL}_L{ln}_E{en}_X{ex}'
            if label in done:
                continue
            eqs, trades, _ = E.run(df, en, ex, ln, COST)
            eqs_g, _, _ = E.run(df, en, ex, ln, 0)
            m = E.metrics(eqs); mg = E.metrics(eqs_g)
            wins = [t for t in trades if t['ret_pct'] > 0]
            row = dict(
                label=label, symbol=SYMBOL, rsi_len=ln, entry=en, exit=ex,
                n_trades=len(trades),
                win_rate=round(100*len(wins)/len(trades), 1) if trades else 0,
                avg_hold=round(sum(t['hold_days'] for t in trades)/len(trades), 1) if trades else 0,
                exposure_pct=exposure(eqs),
                cagr_net=m.get('cagr'), cagr_gross=mg.get('cagr'),
                sharpe=m.get('sharpe'), sortino=m.get('sortino'),
                maxdd=m.get('maxdd'), calmar=m.get('calmar'), vol=m.get('vol'),
                total_x=m.get('total_return_x'),
                bench_cagr=m_bench.get('cagr'), bench_maxdd=m_bench.get('maxdd'),
                bench_x=m_bench.get('total_return_x'),
                stock_bh_cagr=m_stock.get('cagr'), stock_bh_maxdd=m_stock.get('maxdd'),
                beat_bench_ratio=round(m.get('cagr', 0)/m_bench['cagr'], 2) if m_bench.get('cagr') else None,
                lower_dd_than_bench=abs(m.get('maxdd', 0)) < abs(m_bench.get('maxdd', 0)),
            )
            w.writerow(row); f.flush()
            print(f"[{i}/{len(cells)}] {label} netCAGR={row['cagr_net']} DD={row['maxdd']} "
                  f"Calmar={row['calmar']} beat={row['beat_bench_ratio']}x exp={row['exposure_pct']}%", flush=True)
    print('DONE. bench NIFTYBEES CAGR', m_bench['cagr'], 'DD', m_bench['maxdd'],
          '| stock RELIANCE_BH CAGR', m_stock['cagr'], 'DD', m_stock['maxdd'])


if __name__ == '__main__':
    main()
