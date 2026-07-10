#!/usr/bin/env python3
"""Generate static/holdings_ohlc.json — daily OHLC for every current holding.

Same pattern as static/nifty_5m.json: an out-of-band data file the Holdings
Chart Wall (Focus mode) fetches to draw candlesticks. No app.py route, no
service restart. Uses the SAME yfinance symbol mapping as the digest sparkline
(services.holdings_service.get_yahoo_symbol), so symbols like ACCENTMIC-SM
resolve identically. Safe to run any time (read-only); wire to a daily cron.
"""
import json
import os
import sys
import time
import datetime
import urllib.request

sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')

import yfinance as yf  # noqa: E402
from services.holdings_service import get_yahoo_symbol  # noqa: E402

OUT = '/home/arun/quantifyd/static/holdings_ohlc.json'


def holdings_symbols():
    with urllib.request.urlopen('http://localhost:5000/api/holdings/digest', timeout=45) as r:
        digest = json.load(r)
    return [h['tradingsymbol'] for h in digest.get('holdings', [])]


def fetch_bars(symbol, attempts=3):
    """Fetch daily OHLC from yfinance; retry a few times for flaky symbols."""
    best = []
    for i in range(attempts):
        try:
            hist = yf.Ticker(get_yahoo_symbol(symbol)).history(period='1y')
        except Exception as e:  # noqa: BLE001
            print('  err', symbol, e)
            hist = None
        if hist is not None and not hist.empty:
            rows = []
            for idx, row in hist.iterrows():
                o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
                if any(v != v for v in (o, h, l, c)):  # skip NaN rows
                    continue
                rows.append({
                    't': idx.strftime('%Y-%m-%d'),
                    'o': round(float(o), 2), 'h': round(float(h), 2),
                    'l': round(float(l), 2), 'c': round(float(c), 2),
                    'v': int(row['Volume']) if row['Volume'] == row['Volume'] else 0,
                })
            if len(rows) > len(best):
                best = rows
            if len(best) >= 20:      # good enough, stop retrying
                break
        if i < attempts - 1:
            time.sleep(1.2)          # let yfinance breathe before retry
    return best


def main():
    try:
        syms = holdings_symbols()
    except Exception as e:  # noqa: BLE001
        print('FATAL digest fetch failed:', e)
        sys.exit(1)
    print(f'{len(syms)} symbols')

    # never downgrade: keep the best OHLC we've ever captured per symbol
    prev = {}
    if os.path.exists(OUT):
        try:
            prev = (json.load(open(OUT)) or {}).get('symbols', {})
        except Exception:  # noqa: BLE001
            prev = {}

    out = {}
    for s in syms:
        rows = fetch_bars(s)
        keep = rows if len(rows) >= len(prev.get(s, [])) else prev[s]
        if keep:
            out[s] = keep
            tag = 'ok' if len(keep) >= 20 else 'THIN'
            src = '' if keep is rows else ' (kept previous)'
            print(tag, s, len(keep), src)
        else:
            print('NO-DATA', s)
        time.sleep(0.3)

    payload = {
        'updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbols': out,
    }
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    os.replace(tmp, OUT)  # atomic
    print('WROTE', OUT, 'symbols:', len(out), 'bytes:', os.path.getsize(OUT))


if __name__ == '__main__':
    main()
