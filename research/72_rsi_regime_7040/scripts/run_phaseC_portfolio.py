#!/usr/bin/env python3
"""Phase C: slot-based RSI-regime portfolio sweep. Resumable, incremental CSV."""
import sys, csv, os, time
from pathlib import Path
import numpy as np
import pandas as pd

SD = Path(__file__).resolve().parent
sys.path.insert(0, str(SD))
import rsi_regime_engine as E
import portfolio_engine as P

ROOT = SD.parent  # research/72_rsi_regime_7040
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / 'phaseC_portfolio.csv'

BENCH = 'NIFTYBEES'
RSI_LEN = 14
COST_BPS = 15.0
START = '2015-01-01'  # modern-period proxy window (matches qualified-name filter)
EX_PAIRS = [(70, 40), (60, 40), (70, 50), (65, 45), (60, 30)]
NS = [5, 10, 15, 20]
UNIVERSES = ['nifty50', 'nifty200']

FIELDS = ['label', 'universe', 'N', 'entry', 'exit', 'n_names',
          'gross_cagr', 'net_cagr', 'net_sharpe', 'net_sortino', 'net_maxdd',
          'net_calmar', 'net_vol', 'net_total_x', 'trades_per_yr',
          'bench_cagr', 'bench_maxdd', 'beat_ratio', 'lower_dd', 'gate_pass',
          'start', 'end']


def read_symbols(universe):
    f = ROOT.parents[1] / 'backtest_data' / f'{universe}_official.csv'
    df = pd.read_csv(f)
    return df['Symbol'].astype(str).str.strip().tolist()


def main():
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {r['label'] for r in csv.DictReader(f)}
        print(f'Resuming: {len(done)} cells already done', flush=True)
    else:
        with open(OUT, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    # Benchmark
    bdf = E.load(BENCH)
    bdf = bdf[bdf['close'] > 0]

    total = len(UNIVERSES) * len(NS) * len(EX_PAIRS)
    i = 0
    for uni in UNIVERSES:
        syms = read_symbols(uni)
        t0 = time.time()
        price = P.load_universe_prices(syms)
        print(f'[{uni}] {len(price)}/{len(syms)} names qualify (>=2015 start, >=1500 rows) '
              f'loaded in {time.time()-t0:.0f}s', flush=True)
        # common window = earliest date any qualified name has, but bench-clip per cell
        for N in NS:
            for entry, exit_ in EX_PAIRS:
                i += 1
                label = f'{uni}_N{N}_E{entry}X{exit_}'
                if label in done:
                    print(f'[{i}/{total}] {label} skip (done)', flush=True)
                    continue
                gross_eqs, net_eqs, st = P.run_portfolio(
                    price, entry, exit_, RSI_LEN, N, cost_bps=COST_BPS, start=START)
                gm = E.metrics(gross_eqs, 'gross')
                nm = E.metrics(net_eqs, 'net')
                # bench over identical window
                start, end = net_eqs.index[0], net_eqs.index[-1]
                bw = bdf[(bdf.index >= start) & (bdf.index <= end)]
                beq = E.buyhold(bw)
                bmet = E.metrics(beq, 'bench')
                beat = round(nm['cagr'] / bmet['cagr'], 2) if bmet['cagr'] else np.nan
                lower_dd = abs(nm['maxdd']) < abs(bmet['maxdd'])
                gate = (nm['cagr'] >= 1.5 * bmet['cagr']) and lower_dd
                row = dict(label=label, universe=uni, N=N, entry=entry, exit=exit_,
                           n_names=len(price),
                           gross_cagr=gm['cagr'], net_cagr=nm['cagr'],
                           net_sharpe=nm['sharpe'], net_sortino=nm['sortino'],
                           net_maxdd=nm['maxdd'], net_calmar=nm['calmar'],
                           net_vol=nm['vol'], net_total_x=nm['total_return_x'],
                           trades_per_yr=st['trades_per_yr'],
                           bench_cagr=bmet['cagr'], bench_maxdd=bmet['maxdd'],
                           beat_ratio=beat, lower_dd=lower_dd, gate_pass=gate,
                           start=str(start.date()), end=str(end.date()))
                with open(OUT, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                    f.flush()
                print(f'[{i}/{total}] {label} netCAGR={nm["cagr"]:.2f} DD={nm["maxdd"]:.2f} '
                      f'Calmar={nm["calmar"]:.2f} vs bench({bmet["cagr"]:.2f}/{bmet["maxdd"]:.2f}) '
                      f'beat={beat}x lowerDD={lower_dd} GATE={"PASS" if gate else "no"}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
