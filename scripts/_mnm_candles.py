"""Upload & run on VPS: check M&M 5-min candles vs OR levels today."""
import sys
sys.path.insert(0, '/home/arun/quantifyd')
from datetime import datetime as dt
from services.kite_service import get_kite

k = get_kite()
insts = k.instruments('NSE')
tok = next(i['instrument_token'] for i in insts if i['tradingsymbol'] == 'M&M')
start = dt.now().replace(hour=9, minute=15, second=0, microsecond=0)
candles = k.historical_data(tok, start, dt.now(), '5minute')
or_high = 3251.0
or_low = 3212.0
print('M&M OR: [{}, {}]'.format(or_low, or_high))
print('time   open      high      low       close     breach')
for c in candles:
    t = c['date']
    tstr = t.strftime('%H:%M') if hasattr(t, 'strftime') else str(t)[11:16]
    if tstr < '09:30':
        continue
    flags = []
    if c['high'] > or_high: flags.append('H>OR_H')
    if c['close'] > or_high: flags.append('C>OR_H *BREAKOUT*')
    if c['low'] < or_low: flags.append('L<OR_L')
    if c['close'] < or_low: flags.append('C<OR_L *BREAKDOWN*')
    print('{}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {:>8.2f}  {}'.format(
        tstr, c['open'], c['high'], c['low'], c['close'], ' | '.join(flags)))
