"""Publish today's LIVE NIFTY 5-min OHLC to static/nifty_5m.json for the /app/nas
chart. Source = options_data.db underlying_spot (per-minute NIFTY spot, written
live by the options recorder from the Kite feed). Aggregated into 5-min buckets.
Runs every minute via cron during market hours. No backend dependency."""
import sqlite3, json
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DB = BASE / 'backtest_data' / 'options_data.db'
OUT = BASE / 'static' / 'nifty_5m.json'


def ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


day = ist_now().strftime('%Y-%m-%d')
c = sqlite3.connect(str(DB))
rows = c.execute(
    "SELECT snapshot_time, spot_price FROM underlying_spot "
    "WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND spot_price>0 "
    "ORDER BY snapshot_time", (day,)).fetchall()

buckets = {}
order = []
for ts, px in rows:
    hh, mm = int(ts[11:13]), int(ts[14:16])
    key = '%02d:%02d' % (hh, (mm // 5) * 5)
    if key not in buckets:
        buckets[key] = []
        order.append(key)
    buckets[key].append(px)

candles = []
for key in order:
    ps = buckets[key]
    candles.append({'t': key, 'o': round(ps[0], 2), 'h': round(max(ps), 2),
                    'l': round(min(ps), 2), 'c': round(ps[-1], 2)})

out = {
    'updated': ist_now().strftime('%H:%M:%S'),
    'day': day,
    'last': candles[-1]['c'] if candles else None,
    'candles': candles,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
tmp = OUT.with_suffix('.json.tmp')
tmp.write_text(json.dumps(out))
tmp.replace(OUT)
print('wrote %s: %d candles, last %s @ %s' % (OUT, len(candles), out['last'], out['updated']))
