#!/usr/bin/env python3
"""Phase D: adversarial robustness on the Phase-C winner (N20, E60/X30, len14).
Kills survivorship bias (broad 2015-vintage universe vs today's Nifty50),
OOS split (2015-2020 vs 2021-2026), and cost-stress. Ranks by net Calmar."""
import sys, os, sqlite3, json
from pathlib import Path
import pandas as pd
SD = Path(__file__).resolve().parent
sys.path.insert(0, str(SD))
import rsi_regime_engine as E
import portfolio_engine as P

DB = E.DB
# non-equity tickers to exclude from the broad universe
BLOCK = {'NIFTYBEES','BANKBEES','JUNIORBEES','GOLDBEES','SILVERBEES','LIQUIDBEES',
         'ITBEES','PSUBNKBEES','INFRABEES','NIFTY50','NIFTY 50','NIFTY500','SENSEX',
         'CPSEETF','SETFNIF50','ICICINIFTY','KOTAKNIFTY','NIFTY100','NIFTY200'}


def read_symbols(csv_path):
    import csv as _csv
    out = []
    with open(csv_path) as f:
        for r in _csv.DictReader(f):
            out.append(r['Symbol'].strip())
    return out


def broad_universe():
    """All symbols with day data reaching back to <=end-2015 and >=1500 rows,
    excluding obvious index/ETF tickers and anything with a space."""
    con = sqlite3.connect(str(DB)); cur = con.cursor()
    rows = cur.execute(
        "SELECT symbol, COUNT(*) c, MIN(date) mn FROM market_data_unified "
        "WHERE timeframe='day' GROUP BY symbol HAVING c>=1500 AND mn<='2015-12-31'").fetchall()
    con.close()
    syms = [s for (s, c, mn) in rows
            if s not in BLOCK and ' ' not in s and 'BEES' not in s and 'ETF' not in s]
    return syms


def stat_line(net_eqs, gross_eqs, label):
    m = E.metrics(net_eqs); mg = E.metrics(gross_eqs)
    return dict(label=label, net_cagr=m.get('cagr'), gross_cagr=mg.get('cagr'),
                maxdd=m.get('maxdd'), calmar=m.get('calmar'), sharpe=m.get('sharpe'),
                sortino=m.get('sortino'), total_x=m.get('total_return_x'))


def run_case(price, name, entry, exit_, N, cost, start, end, results):
    g, n, s = P.run_portfolio(price, entry, exit_, 14, N, cost_bps=cost, start=start, end=end)
    line = stat_line(n, g, name)
    line.update(n_names=len(price), cost_bps=cost,
                window=f'{start or "full"}..{end or "full"}')
    results.append(line)
    print(f"{name:42s} netCAGR={line['net_cagr']:>6} DD={line['maxdd']:>7} "
          f"Calmar={line['calmar']:>5} Sharpe={line['sharpe']:>5} names={len(price)}", flush=True)
    return line


def bench(start, end):
    b = E.load('NIFTYBEES', start=start, end=end)
    m = E.metrics(E.buyhold(b))
    return m


def main():
    results = []
    n50 = read_symbols(str(DB.parent / 'nifty50_official.csv'))
    broad = broad_universe()
    print(f"nifty50 list={len(n50)}  broad universe={len(broad)} names\n")

    price50 = P.load_universe_prices(n50)
    priceB = P.load_universe_prices(broad)
    print(f"qualified: nifty50={len(price50)}  broad={len(priceB)}\n")

    E60X30_N20 = dict(entry=60, exit_=30, N=20)
    # 1) survivorship: same config, nifty50 vs broad, full window + cost 15 & 30
    print("== SURVIVORSHIP KILL (winner config N20 60/30) ==")
    run_case(price50, 'nifty50_full_c15', **E60X30_N20, cost=15, start='2015-01-01', end=None, results=results)
    run_case(priceB,  'broad_full_c15',   **E60X30_N20, cost=15, start='2015-01-01', end=None, results=results)
    run_case(priceB,  'broad_full_c30',   **E60X30_N20, cost=30, start='2015-01-01', end=None, results=results)
    # 2) OOS split on BROAD (the honest universe)
    print("\n== OOS SPLIT (broad universe) ==")
    run_case(priceB, 'broad_IS_2015_2020', **E60X30_N20, cost=15, start='2015-01-01', end='2020-12-31', results=results)
    run_case(priceB, 'broad_OOS_2021_2026', **E60X30_N20, cost=15, start='2021-01-01', end=None, results=results)
    run_case(price50, 'nifty50_IS_2015_2020', **E60X30_N20, cost=15, start='2015-01-01', end='2020-12-31', results=results)
    run_case(price50, 'nifty50_OOS_2021_2026', **E60X30_N20, cost=15, start='2021-01-01', end=None, results=results)
    # 3) param neighborhood on BROAD (is it a plateau or a lone peak?)
    print("\n== PARAM NEIGHBORHOOD (broad, full) ==")
    for (en, ex, N) in [(60,30,15),(60,30,25),(65,35,20),(70,40,20),(60,40,20)]:
        run_case(priceB, f'broad_N{N}_E{en}X{ex}', entry=en, exit_=ex, N=N, cost=15,
                 start='2015-01-01', end=None, results=results)

    # benchmark rows
    b_full = bench('2015-01-01', None)
    b_is = bench('2015-01-01', '2020-12-31')
    b_oos = bench('2021-01-01', None)
    print(f"\nNIFTYBEES full: CAGR {b_full['cagr']} DD {b_full['maxdd']} Calmar {b_full['calmar']}")
    print(f"NIFTYBEES IS 15-20: CAGR {b_is['cagr']} DD {b_is['maxdd']}")
    print(f"NIFTYBEES OOS 21-26: CAGR {b_oos['cagr']} DD {b_oos['maxdd']}")

    out = dict(results=results, bench=dict(full=b_full, IS=b_is, OOS=b_oos),
               n_broad=len(priceB), n_nifty50=len(price50))
    (SD.parent / 'results' / 'phaseD_robustness.json').write_text(json.dumps(out, indent=2, default=str))
    print("\nsaved results/phaseD_robustness.json")


if __name__ == '__main__':
    main()
