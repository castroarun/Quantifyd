#!/usr/bin/env python3
"""Generate static/momentum_scan.json + static/scan_ohlc.json for the Holdings scanner UI.

- momentum_scan.json: the Momentum-30 (research/62) ranked Nifty-200 scan (rsblend RS score).
- scan_ohlc.json: daily OHLC (from market_data.db) for the top-N ranked names, so watchlist
  picks get the same candlesticks as holdings.
Read-only over market_data.db. Slow (~1-2 min cold panel build) — a cron/background job, never
a request. Atomic writes. Cron daily after EOD refresh + on-demand.
"""
import sys, os, json, sqlite3, datetime
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')

from services.momentum_scan import scan

OUT_SCAN = '/home/arun/quantifyd/static/momentum_scan.json'
OUT_OHLC = '/home/arun/quantifyd/static/scan_ohlc.json'
MARKET_DB = '/home/arun/quantifyd/backtest_data/market_data.db'
OHLC_TOP_N = 80   # generate charts for the top-80 ranked names (watchlist candidates)


def _atomic_write(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, separators=(',', ':'))
    os.replace(tmp, path)


def main():
    d = scan()
    d['generated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _atomic_write(OUT_SCAN, d)
    print('WROTE', OUT_SCAN, 'count', d.get('count'), 'asof', d.get('asof'), 'err', d.get('error'))

    # OHLC for the top-N ranked names, from market_data.db (daily)
    syms = [r['symbol'] for r in d.get('rows', [])[:OHLC_TOP_N]]
    con = sqlite3.connect(MARKET_DB)
    out = {}
    for s in syms:
        rows = con.execute(
            "SELECT date,open,high,low,close,volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' ORDER BY date DESC LIMIT 300", (s,)
        ).fetchall()
        if len(rows) < 2:
            continue
        bars = []
        for dt, o, h, l, c, v in reversed(rows):
            if None in (o, h, l, c):
                continue
            bars.append({'t': str(dt)[:10], 'o': round(o, 2), 'h': round(h, 2),
                         'l': round(l, 2), 'c': round(c, 2), 'v': int(v or 0)})
        if bars:
            out[s] = bars
    con.close()
    payload = {'generated': d['generated'], 'symbols': out}
    _atomic_write(OUT_OHLC, payload)
    print('WROTE', OUT_OHLC, 'symbols', len(out), 'bytes', os.path.getsize(OUT_OHLC))


if __name__ == '__main__':
    main()
