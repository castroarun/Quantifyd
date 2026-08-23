#!/usr/bin/env python3
"""Fetch EXACT index constituents + TRUE weights (free-float mcap share)
from NSE's equity-stockIndices API, one JSON per index.

Output: backtest_data/index_constituents/<STORE_SYM>.json
  { symbols: [...], weights: {SYM: pct}, source: 'nse_ffmc',
    index: 'NIFTY REALTY', fetched: iso-ts }

NSE needs a browser-ish session (homepage first for cookies). If an index
fetch fails, any existing JSON is left untouched and index_pulse falls back
to its CSV/industry-mapping path — the page degrades, never breaks.
Run weekly (composition changes are ~monthly) and after rebalances.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

OUT = Path('/home/arun/quantifyd/backtest_data/index_constituents')
OUT.mkdir(parents=True, exist_ok=True)

NSE_NAMES = {
    'NIFTY50':        'NIFTY 50',
    'NIFTYNEXT50':    'NIFTY NEXT 50',
    'NIFTYMIDCAP150': 'NIFTY MIDCAP 150',
    'NIFTYSMLCAP250': 'NIFTY SMALLCAP 250',
    'NIFTY500':       'NIFTY 500',
    'BANKNIFTY':      'NIFTY BANK',
    'NIFTYFINSRV':    'NIFTY FINANCIAL SERVICES',
    'NIFTYIT':        'NIFTY IT',
    'NIFTYPHARMA':    'NIFTY PHARMA',
    'NIFTYAUTO':      'NIFTY AUTO',
    'NIFTYMETAL':     'NIFTY METAL',
    'NIFTYFMCG':      'NIFTY FMCG',
    'NIFTYENERGY':    'NIFTY ENERGY',
    'NIFTYREALTY':    'NIFTY REALTY',
    'NIFTYPSUBANK':   'NIFTY PSU BANK',
}

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({'User-Agent': UA, 'Accept': 'application/json',
                      'Accept-Language': 'en-US,en;q=0.9',
                      'Referer': 'https://www.nseindia.com/'})
    s.get('https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%2050', timeout=15)   # cookie warm-up on the matching page
    time.sleep(1.0)
    return s


def fetch_index(s: requests.Session, nse_name: str) -> dict:
    url = ('https://www.nseindia.com/api/equity-stock-indices?index='
           + requests.utils.quote(nse_name))
    r = s.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


def main():
    only = sys.argv[1:] or list(NSE_NAMES)
    s = session()
    ok = fail = 0
    for store_sym in only:
        nse_name = NSE_NAMES.get(store_sym)
        if not nse_name:
            continue
        try:
            data = fetch_index(s, nse_name)
            rows = [d for d in data.get('data', [])
                    if d.get('symbol') and d['symbol'] != nse_name
                    and d.get('priority', 0) != 1]        # drop the index row
            ffmc = {d['symbol']: float(d.get('ffmc') or 0) for d in rows}
            total = sum(ffmc.values())
            if not rows or total <= 0:
                raise ValueError(f'empty payload ({len(rows)} rows)')
            weights = {k: round(v / total * 100, 3)
                       for k, v in sorted(ffmc.items(), key=lambda x: -x[1])}
            out = {
                'index': nse_name,
                'symbols': list(weights.keys()),
                'weights': weights,
                'source': 'nse_ffmc',
                'fetched': datetime.now().isoformat(timespec='seconds'),
            }
            (OUT / f'{store_sym}.json').write_text(json.dumps(out, indent=1))
            print(f'  {store_sym:16s} {len(weights):4d} names  '
                  f'top: {list(weights)[0]} {weights[list(weights)[0]]:.1f}%')
            ok += 1
        except Exception as e:
            print(f'  {store_sym:16s} FAILED: {e} — existing file (if any) kept')
            fail += 1
        time.sleep(1.2)
    print(f'DONE: {ok} ok, {fail} failed')


if __name__ == '__main__':
    main()
