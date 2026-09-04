"""Generate static/oa_real_ohlc.json — daily OHLC for the OA REAL book's holdings.

Same shape as momentum_ohlc.json so HoldingsCharts renders it unchanged. The
per-bar "stop" line is the book's actual exit level: the 15-SMA close trail,
floored at the fixed -8%-from-entry stop on bars after the entry date.
"""
import json
import sqlite3
from datetime import datetime, timedelta

ROOT = '/home/arun/quantifyd'
OUT = f'{ROOT}/static/oa_real_ohlc.json'
DB = f'{ROOT}/backtest_data/market_data.db'
STATE = f'{ROOT}/backtest_data/oa_real_state.json'
DAYS = 400
TRAIL_N = 15


def main():
    st = json.load(open(STATE))
    positions = {p['symbol']: p for p in st['positions']}
    syms = sorted(positions)
    since = (datetime.now() - timedelta(days=DAYS)).date().isoformat()
    con = sqlite3.connect(DB)
    out = {}
    for s in syms:
        rows = con.execute(
            "SELECT date,open,high,low,close,volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' AND date>=? AND close>0 ORDER BY date",
            (s, since)).fetchall()
        if not rows:
            print(f'  no data: {s}')
            continue
        p = positions[s]
        closes = [float(c) for d, o, h, l, c, v in rows]
        bars = []
        for i, (d, o, h, l, c, v) in enumerate(rows):
            sma = sum(closes[max(0, i - TRAIL_N + 1):i + 1]) / min(i + 1, TRAIL_N) \
                if i + 1 >= TRAIL_N else None
            level = sma
            if d[:10] >= p['entry_date'] and level is not None:
                level = max(level, p['stop'])
            elif d[:10] >= p['entry_date']:
                level = p['stop']
            bars.append({'t': d[:10], 'o': float(o or c), 'h': float(h or c),
                         'l': float(l or c), 'c': float(c), 'v': int(v or 0),
                         'stop': round(level, 2) if level is not None else None})
        out[s] = bars
    con.close()
    pos = {s: {'entry': p['buy'], 'entry_date': p['entry_date'], 'stop': p['stop'],
               'qty': p['qty']} for s, p in positions.items()}
    json.dump({'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               'trail': TRAIL_N, 'positions': pos, 'symbols': out}, open(OUT, 'w'))
    print(f'wrote {OUT}: {len(out)} symbols')


if __name__ == '__main__':
    main()
