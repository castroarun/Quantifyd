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

def hlc(d1, d2=None):
    d2 = d2 or d1
    r = c.execute(
        "SELECT max(spot_price), min(spot_price), "
        "(SELECT spot_price FROM underlying_spot WHERE symbol='NIFTY' AND "
        " substr(snapshot_time,1,10)<=? AND spot_price>0 ORDER BY snapshot_time DESC LIMIT 1) "
        "FROM underlying_spot WHERE symbol='NIFTY' AND substr(snapshot_time,1,10) BETWEEN ? AND ? "
        "AND spot_price>0", (d2, d1, d2)).fetchone()
    return r if r and r[0] else None


def cpr(hlc_tuple):
    if not hlc_tuple:
        return None
    h, l, cl = hlc_tuple
    p = (h + l + cl) / 3.0
    bc = (h + l) / 2.0
    tc = 2 * p - bc
    return {
        'P': round(p, 1), 'TC': round(max(tc, bc), 1), 'BC': round(min(tc, bc), 1),
        'R1': round(2 * p - l, 1), 'S1': round(2 * p - h, 1),
        'R2': round(p + (h - l), 1), 'S2': round(p - (h - l), 1),
    }


from datetime import date as _date
_today = ist_now().date()
_all_days = [r[0] for r in c.execute(
    "SELECT DISTINCT substr(snapshot_time,1,10) d FROM underlying_spot WHERE symbol='NIFTY' ORDER BY d")]
_prev_day = max((d for d in _all_days if d < day), default=None)
_this_mon = _today - timedelta(days=_today.weekday())
_prev_mon = (_this_mon - timedelta(days=7)).isoformat()
_prev_fri = (_this_mon - timedelta(days=3)).isoformat()

daily_cpr = cpr(hlc(_prev_day)) if _prev_day else None
weekly_cpr = cpr(hlc(_prev_mon, _prev_fri))

out = {
    'updated': ist_now().strftime('%H:%M:%S'),
    'day': day,
    'last': candles[-1]['c'] if candles else None,
    'candles': candles,
    'dailyCpr': daily_cpr,
    'weeklyCpr': weekly_cpr,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
tmp = OUT.with_suffix('.json.tmp')
tmp.write_text(json.dumps(out))
tmp.replace(OUT)
print('wrote %s: %d candles, last %s @ %s' % (OUT, len(candles), out['last'], out['updated']))
