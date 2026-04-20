import os, sys
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')
from dotenv import load_dotenv
load_dotenv('.env')

from services.kite_service import get_kite
from services.orb_db import get_orb_db
from datetime import datetime, timedelta, date

kite = get_kite()
db = get_orb_db()
today = date.today().isoformat()

universe = ['ADANIENT','TATASTEEL','BEL','VEDL','BPCL','M&M','BAJFINANCE','TRENT','HAL','IRCTC','GRASIM','GODREJPROP','RELIANCE','AXISBANK','APOLLOHOSP']

insts = kite.instruments('NSE')
tok_map = {i['tradingsymbol']: i['instrument_token'] for i in insts if i['tradingsymbol'] in universe}

now = datetime.now()
session_start = now.replace(hour=9, minute=15, second=0, microsecond=0)

print(f"{'Stock':12s} {'OR_H':>8s} {'OR_L':>8s} {'Gap%':>6s} | {'LONG BRK':>12s} | {'SHORT BRK':>12s} | RSI  | Verdict")
print("-" * 105)

missed = []
for sym in universe:
    ds = db.get_or_create_daily_state(sym, today)
    or_h = ds.get('or_high')
    or_l = ds.get('or_low')
    wide = ds.get('is_wide_cpr_day')
    gap = ds.get('gap_pct') or 0
    rsi = ds.get('rsi_15m')

    if wide:
        print(f"{sym:12s}     SKIP — Wide CPR ({ds.get('cpr_width_pct', 0):.3f}%)")
        continue
    if not or_h or not or_l:
        print(f"{sym:12s}     (no OR data)")
        continue

    token = tok_map.get(sym)
    if not token:
        continue
    try:
        candles = kite.historical_data(token, session_start, now, '5minute')
    except Exception as e:
        print(f"{sym:12s}     ERR: {e}")
        continue

    long_breaks = []
    short_breaks = []
    for i in range(1, len(candles)):
        prev = candles[i-1]
        curr = candles[i]
        t = curr['date'].strftime('%H:%M')
        if t < '09:30':
            continue
        if curr['close'] > or_h and prev['close'] <= or_h:
            long_breaks.append((t, curr['close']))
        if curr['close'] < or_l and prev['close'] >= or_l:
            short_breaks.append((t, curr['close']))

    lb_str = f"{long_breaks[0][0]}@{long_breaks[0][1]:.1f}" if long_breaks else '-'
    sb_str = f"{short_breaks[0][0]}@{short_breaks[0][1]:.1f}" if short_breaks else '-'
    rsi_str = f"{rsi:.1f}" if rsi else '?'

    gap_block = (gap or 0) > 1.0

    # Verdict
    verdict_parts = []
    if long_breaks:
        if gap_block:
            verdict_parts.append(f"LONG @{long_breaks[0][0]} blocked by GAP")
        elif rsi and rsi < 60:
            verdict_parts.append(f"LONG @{long_breaks[0][0]} blocked by RSI<60")
        else:
            verdict_parts.append(f"LONG @{long_breaks[0][0]} MISSED")
            missed.append((sym, 'LONG', long_breaks[0][0], long_breaks[0][1]))
    if short_breaks:
        if rsi and rsi > 40:
            verdict_parts.append(f"SHORT @{short_breaks[0][0]} blocked by RSI>40")
        else:
            verdict_parts.append(f"SHORT @{short_breaks[0][0]} MISSED")
            missed.append((sym, 'SHORT', short_breaks[0][0], short_breaks[0][1]))

    verdict = ' | '.join(verdict_parts) if verdict_parts else 'No breakout'
    print(f"{sym:12s} {or_h:>8.2f} {or_l:>8.2f} {gap:>5.2f}% | {lb_str:>12s} | {sb_str:>12s} | {rsi_str:>4s} | {verdict}")

print()
print(f"=== MISSED TRADES: {len(missed)} ===")
for sym, dir, t, price in missed:
    print(f"  {t} {sym} {dir} @ {price:.1f}")
