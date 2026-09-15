"""research/176 stage 0 — build the universe and scan it for the known data defects.

Universe = the symbols that carry 5-minute bars spanning 2015-02 -> 2026 (this IS
the liquid F&O panel built in research/81), intersected with daily history and a
traded-value floor, ETFs excluded by instrument NAME (never by ticker regex).

Writes results/universe.csv with one row per symbol:
  symbol, day_n, day_start, day_end, min5_n, med_tv_cr, split_flags, worst_gap, gap_dates
"""
import json
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import connect, load_bars, split_defect_flags

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'universe.csv')
ETF_JSON = '/home/arun/quantifyd/backtest_data/etf_exclusions.json'


def main():
    con = connect()
    print('finding 5-min panel...', flush=True)
    rows = con.execute(
        "select symbol, count(*), min(date), max(date) from market_data_unified "
        "where timeframe='5minute' group by symbol").fetchall()
    panel = [(s, n, a, b) for (s, n, a, b) in rows
             if str(a)[:4] <= '2015' and str(b)[:4] >= '2026' and n > 100000]
    print(f'  5-min panel spanning 2015->2026: {len(panel)} symbols', flush=True)

    etfs = set()
    if os.path.exists(ETF_JSON):
        try:
            j = json.load(open(ETF_JSON))
            if isinstance(j, dict):
                for v in j.values():
                    if isinstance(v, list):
                        etfs |= set(v)
            elif isinstance(j, list):
                etfs = set(j)
        except Exception as e:
            print('  etf json unreadable:', e, flush=True)
    print(f'  etf exclusion list: {len(etfs)} names', flush=True)

    recs = []
    for i, (sym, n5, a5, b5) in enumerate(panel):
        if sym in etfs:
            continue
        d = load_bars(con, sym, 'day')
        if d.empty or len(d) < 500:
            continue
        tv = (d['close'] * d['volume']).rolling(20).median()
        med_tv_cr = float(np.nanmedian(tv.values)) / 1e7
        nf, worst, dates = split_defect_flags(d)
        recs.append(dict(symbol=sym, day_n=len(d),
                         day_start=str(d.index[0].date()), day_end=str(d.index[-1].date()),
                         min5_n=n5, min5_start=str(a5)[:10], min5_end=str(b5)[:10],
                         med_tv_cr=round(med_tv_cr, 2),
                         split_flags=nf, worst_gap=round(worst, 3),
                         gap_dates=';'.join(dates[:6])))
        if (i + 1) % 40 == 0:
            print(f'  {i+1}/{len(panel)} scanned', flush=True)
    con.close()

    df = pd.DataFrame(recs).sort_values('symbol')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}  ({len(df)} symbols)', flush=True)
    print('\n--- liquidity ---', flush=True)
    for thr in (5, 10, 25, 50, 100):
        print(f'  med 20d traded value >= Rs{thr:3d} cr : {(df.med_tv_cr >= thr).sum()}', flush=True)
    print('\n--- daily history depth ---', flush=True)
    for y in ('2006', '2010', '2015'):
        print(f'  daily starts on/before {y}-12-31 : {(df.day_start <= y + "-12-31").sum()}', flush=True)
    print('\n--- split / corporate-action step flags (overnight gap outside 0.65x-1.55x) ---', flush=True)
    print(f'  symbols with >=1 flag : {(df.split_flags > 0).sum()} of {len(df)}', flush=True)
    bad = df[df.split_flags > 0].sort_values('worst_gap', ascending=False)
    print(bad[['symbol', 'split_flags', 'worst_gap', 'gap_dates']].head(25).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
