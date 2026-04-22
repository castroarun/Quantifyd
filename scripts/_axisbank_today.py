"""Dump today's 5-min candles for AXISBANK and annotate OR crossings."""
from datetime import datetime
from services.kite_service import get_kite
from services.orb_db import get_orb_db

db = get_orb_db()
today = '2026-04-22'
sym = 'AXISBANK'
ds = db.get_or_create_daily_state(sym, today)
or_h, or_l = ds['or_high'], ds['or_low']
print(f'OR: {or_l}-{or_h}  CPR: tc={ds["cpr_tc"]} bc={ds["cpr_bc"]}  gap%={ds["gap_pct"]}')
print()

k = get_kite()
token = 1510401  # AXISBANK NSE token
from datetime import date
candles = k.historical_data(
    instrument_token=token,
    from_date=date(2026, 4, 22),
    to_date=datetime.now(),
    interval='5minute',
)
print(f"{'time':<20} {'open':>8} {'high':>8} {'low':>8} {'close':>8} {'vs_OR':<20}")
prev_close = None
for c in candles:
    t = c['date'].strftime('%H:%M')
    flag = ''
    if prev_close is not None:
        if c['close'] > or_h and prev_close <= or_h:
            flag = ' <<< LONG breakout'
        elif c['close'] < or_l and prev_close >= or_l:
            flag = ' <<< SHORT breakout'
    rel = ''
    if c['close'] > or_h:
        rel = 'above_OR_high'
    elif c['close'] < or_l:
        rel = 'below_OR_low'
    else:
        rel = 'inside_OR'
    print(f"{t:<20} {c['open']:>8.2f} {c['high']:>8.2f} {c['low']:>8.2f} {c['close']:>8.2f} {rel:<16}{flag}")
    prev_close = c['close']
