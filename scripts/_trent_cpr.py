"""Verify TRENT's prev-day HLC and recompute today's CPR width.

Compares:
  (1) what _fetch_daily_candles would return
  (2) raw Kite daily bars
  (3) user-observed Kite chart CPR → implied prev C
"""
import sys, json
sys.path.insert(0, '/home/arun/quantifyd')
from datetime import datetime as dt, timedelta
from services.kite_service import get_kite

k = get_kite()
with open('/home/arun/quantifyd/backtest_data/instrument_tokens.json') as f:
    cache = json.load(f)
tok = cache['symbol_to_token']['TRENT']
print('TRENT token:', tok)

today = dt.now().replace(hour=0, minute=0, second=0, microsecond=0)
frm = today - timedelta(days=12)
daily = k.historical_data(tok, frm, today + timedelta(days=1), 'day')
print('\nRaw daily candles (oldest -> newest):')
for c in daily:
    d = c['date']
    dstr = d.strftime('%Y-%m-%d %H:%M:%S%z') if hasattr(d, 'strftime') else str(d)
    print('  {}  O={:.2f}  H={:.2f}  L={:.2f}  C={:.2f}  V={}'.format(
        dstr, c['open'], c['high'], c['low'], c['close'], c.get('volume', 0)))

# Replicate engine's filter: prev = last candle with date() < today
today_date = today.date()
prev = None
for c in reversed(daily):
    d = c['date']
    dd = d.date() if hasattr(d, 'date') else dt.strptime(str(d)[:10], '%Y-%m-%d').date()
    if dd < today_date:
        prev = c; prev_date = dd
        break

# Also check what prev_candles[-1] in engine logic would pick
prev_engine = None
for c in daily:
    d = c['date']
    dd = d.date() if hasattr(d, 'date') else dt.strptime(str(d)[:10], '%Y-%m-%d').date()
    if dd < today_date:
        prev_engine = c  # engine does prev_candles[-1] = most recent

print('\n--- Engine-style prev (prev_candles[-1]):', prev_engine['date'] if prev_engine else None)
print('--- Reverse-loop prev (my script):      ', prev['date'] if prev else None)

if prev_engine:
    H, L, C = prev_engine['high'], prev_engine['low'], prev_engine['close']
    pivot = (H + L + C) / 3
    bc = (H + L) / 2
    tc = 2 * pivot - bc
    width_pct = abs(tc - bc) / pivot * 100
    print('\nEngine-style CPR for today:')
    print('  prev H={}  L={}  C={}'.format(H, L, C))
    print('  pivot={:.2f}  tc={:.2f}  bc={:.2f}'.format(pivot, tc, bc))
    print('  width={:.4f}% (our engine logs 0.6400%)'.format(width_pct))

# Kite chart values from user
print('\n--- User Kite chart ---')
user_tc = 4225.50
user_bc = 4202.35
user_pivot = (user_tc + user_bc) / 2
print('  tc={}  bc={}  pivot={}'.format(user_tc, user_bc, user_pivot))
# Back-solve: what H, L, C would produce these?
# If bc = (H+L)/2, pivot = (H+L+C)/3 → H+L = 2*bc, C = 3*pivot - 2*bc
implied_C = 3 * user_pivot - 2 * user_bc
implied_H_plus_L = 2 * user_bc
print('  implied C={:.2f}  implied H+L={:.2f}'.format(implied_C, implied_H_plus_L))
