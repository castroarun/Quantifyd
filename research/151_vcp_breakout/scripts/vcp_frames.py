"""Build + cache the wide daily frames used by every research/151 sweep cell.

NaN discipline (playbook §3 / the r/142 gate bug): every rolling statistic is computed
on the symbol's OWN dropna()'d series and only then aligned onto the union index, so a
missing/phantom row cannot NaN-poison the window for months afterwards.

Cache: results/frames.npz  (float32 wide matrices) + results/frames_meta.json
Columns/rows are shared: rows = union trading dates, cols = symbols.
"""
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
DB = ROOT / 'backtest_data' / 'market_data.db'
CACHE = STUDY / 'results' / 'frames.npz'
META = STUDY / 'results' / 'frames_meta.json'

ETF_RE = re.compile(r'(BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50)')
SMAS = (15, 20, 50, 150)
BASE_START = '2004-06-01'          # 550d of warmup before a 2006 study start


def build():
    conn = sqlite3.connect(str(DB))
    syms = [r[0] for r in conn.execute(
        "select symbol from (select symbol, count(*) n from market_data_unified "
        "where timeframe='day' group by symbol) where n >= 260")]
    print(f'{len(syms)} symbols with >=260 daily rows', flush=True)
    cols = {}
    t0 = time.time()
    for i, s in enumerate(syms):
        df = pd.read_sql_query(
            "select date, open, high, low, close, volume from market_data_unified "
            "where symbol=? and timeframe='day' order by date", conn, params=(s,))
        df['date'] = pd.to_datetime(df['date'].str[:10])
        df = df.drop_duplicates('date').set_index('date').sort_index()
        # drop phantom rows: zero volume AND a flat O=H=L=C bar
        flat = (df['open'] == df['high']) & (df['high'] == df['low']) & (df['low'] == df['close'])
        df = df[~(flat & (df['volume'] <= 0))]
        c = df['close'].dropna()
        if len(c) < 260:
            continue
        d = dict(close=c, high=df['high'], low=df['low'], open=df['open'],
                 tv20=(df['close'] * df['volume']).rolling(20, min_periods=10).median())
        for n in SMAS:
            d[f'sma{n}'] = c.rolling(n, min_periods=max(5, n // 2)).mean()
        m = df.index >= BASE_START
        if not m.any():
            continue
        cols[s] = {k: v[v.index >= BASE_START] for k, v in d.items()}
        if (i + 1) % 500 == 0:
            print(f'  loaded {i+1}/{len(syms)} ({time.time()-t0:.0f}s)', flush=True)
    conn.close()

    keys = ['close', 'high', 'low', 'open', 'tv20'] + [f'sma{n}' for n in SMAS]
    wide = {k: pd.DataFrame({s: v[k] for s, v in cols.items()}).astype('float32')
            for k in keys}
    idx = wide['close'].index
    symbols = list(wide['close'].columns)
    print(f'wide frames: {wide["close"].shape} ({time.time()-t0:.0f}s)', flush=True)
    np.savez_compressed(CACHE, **{k: v.values for k, v in wide.items()})
    json.dump(dict(dates=[str(d.date()) for d in idx], symbols=symbols,
                   smas=list(SMAS), base_start=BASE_START,
                   etfs=[s for s in symbols if ETF_RE.search(s)]),
              open(META, 'w'))
    print(f'cached -> {CACHE} ({CACHE.stat().st_size/1e6:.0f} MB)')


def load():
    meta = json.load(open(META))
    z = np.load(CACHE)
    dates = pd.DatetimeIndex(meta['dates'])
    return {k: z[k] for k in z.files}, dates, meta['symbols'], meta


if __name__ == '__main__':
    build()
