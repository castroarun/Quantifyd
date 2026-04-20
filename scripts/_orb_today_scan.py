"""ORB live-equivalent scan for today — runs on VPS via _vps_query.py.

Scans the ORB universe for 5-min breakouts above/below OR15 range, classifies
each signal as TAKEN / BLOCKED (filter name) / MISSED, and prints a summary.
"""
import os, sys
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')
from dotenv import load_dotenv
load_dotenv('.env')

from services.kite_service import get_kite
from services.orb_db import get_orb_db
from datetime import datetime, date

kite = get_kite()
db = get_orb_db()
today = date.today().isoformat()

universe = ['ADANIENT','TATASTEEL','BEL','VEDL','BPCL','M&M','BAJFINANCE','TRENT','HAL','IRCTC','GRASIM','GODREJPROP','RELIANCE','AXISBANK','APOLLOHOSP']

insts = kite.instruments('NSE')
tok_map = {i['tradingsymbol']: i['instrument_token'] for i in insts if i['tradingsymbol'] in universe}

now = datetime.now()
session_start = now.replace(hour=9, minute=15, second=0, microsecond=0)

print(f"{'Stock':12s} {'OR_H':>8s} {'OR_L':>8s} {'Gap%':>6s} {'RSI':>5s} | {'LONG':>13s} | {'SHORT':>13s} | Verdict")
print('-' * 120)

taken = []
blocked = []
missed = []
no_signal = []

for sym in universe:
    ds = db.get_or_create_daily_state(sym, today)
    or_h = ds.get('or_high')
    or_l = ds.get('or_low')
    wide = ds.get('is_wide_cpr_day')
    gap = ds.get('gap_pct') or 0
    rsi = ds.get('rsi_15m')

    if wide:
        print(f"{sym:12s}     SKIP -- Wide CPR ({ds.get('cpr_width_pct', 0):.3f}%)")
        blocked.append((sym, '-', 'WIDE_CPR', ds.get('cpr_width_pct', 0)))
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
        print(f'{sym:12s}     ERR: {e}')
        continue

    long_breaks = []
    short_breaks = []
    for i in range(1, len(candles)):
        prev, curr = candles[i-1], candles[i]
        t = curr['date'].strftime('%H:%M')
        if t < '09:30':
            continue
        if curr['close'] > or_h and prev['close'] <= or_h:
            long_breaks.append((t, curr['close']))
        if curr['close'] < or_l and prev['close'] >= or_l:
            short_breaks.append((t, curr['close']))

    lb_str = f"{long_breaks[0][0]}@{long_breaks[0][1]:.1f}" if long_breaks else '-'
    sb_str = f"{short_breaks[0][0]}@{short_breaks[0][1]:.1f}" if short_breaks else '-'
    rsi_str = f'{rsi:.1f}' if rsi else '?'
    gap_block = (gap or 0) > 1.0

    # Compute realized P&L if it were taken (SL = 0.5%, T1 = R, flexible exit at last candle)
    verdicts = []
    for direction, break_info in [('LONG', long_breaks[0] if long_breaks else None),
                                     ('SHORT', short_breaks[0] if short_breaks else None)]:
        if not break_info:
            continue
        t0, entry = break_info
        # filter check
        filter_name = None
        if direction == 'LONG':
            if gap_block: filter_name = f'GAP {gap:.2f}%'
            elif rsi and rsi < 60: filter_name = f'RSI {rsi:.1f}<60'
        else:
            if rsi and rsi > 40: filter_name = f'RSI {rsi:.1f}>40'

        # simulate PnL assuming entry at break close, SL 0.5%, target 1R, else exit at EOD
        sl_pct = 0.005
        if direction == 'LONG':
            sl = entry * (1 - sl_pct)
            target = entry + (entry - sl)  # 1R
        else:
            sl = entry * (1 + sl_pct)
            target = entry - (sl - entry)  # 1R

        # Find exit in subsequent candles
        exit_t = None; exit_price = None; exit_reason = None
        # Find the starting index corresponding to the breakout candle
        start_idx = next((j for j in range(len(candles))
                          if candles[j]['date'].strftime('%H:%M') == t0), None)
        if start_idx is None:
            continue
        for j in range(start_idx + 1, len(candles)):
            c = candles[j]
            t_hm = c['date'].strftime('%H:%M')
            if t_hm >= '15:20':
                exit_t, exit_price, exit_reason = t_hm, c['close'], 'EOD'
                break
            if direction == 'LONG':
                if c['low'] <= sl:
                    exit_t, exit_price, exit_reason = t_hm, sl, 'SL'
                    break
                if c['high'] >= target:
                    exit_t, exit_price, exit_reason = t_hm, target, 'TARGET'
                    break
            else:
                if c['high'] >= sl:
                    exit_t, exit_price, exit_reason = t_hm, sl, 'SL'
                    break
                if c['low'] <= target:
                    exit_t, exit_price, exit_reason = t_hm, target, 'TARGET'
                    break
        if exit_price is None and candles:
            last = candles[-1]
            exit_t, exit_price, exit_reason = last['date'].strftime('%H:%M'), last['close'], 'OPEN'

        pnl_pct = ((exit_price - entry) / entry * 100) if direction == 'LONG' \
                  else ((entry - exit_price) / entry * 100)

        if filter_name:
            verdicts.append(f'{direction}@{t0} BLOCK [{filter_name}] would-be {pnl_pct:+.2f}%')
            blocked.append((sym, direction, filter_name, t0, entry, exit_t, exit_price, exit_reason, round(pnl_pct, 2)))
        else:
            verdicts.append(f'{direction}@{t0} TAKE entry={entry:.1f} exit={exit_price:.1f} [{exit_reason}] {pnl_pct:+.2f}%')
            missed.append((sym, direction, t0, entry, exit_t, exit_price, exit_reason, round(pnl_pct, 2)))

    if not verdicts:
        no_signal.append(sym)
        verdicts.append('No breakout')

    verdict = ' | '.join(verdicts)
    print(f'{sym:12s} {or_h:>8.2f} {or_l:>8.2f} {gap:>5.2f}% {rsi_str:>5s} | {lb_str:>13s} | {sb_str:>13s} | {verdict}')

print()
print('=' * 80)
print(f'MISSED SIGNALS (passed filters, would have been trades): {len(missed)}')
print('=' * 80)
total_pct = 0
for sym, direction, t0, entry, exit_t, exit_price, exit_reason, pnl_pct in missed:
    print(f'  {t0} -> {exit_t}  {sym:12s} {direction:5s} entry={entry:.1f} exit={exit_price:.1f} [{exit_reason}]  {pnl_pct:+.2f}%')
    total_pct += pnl_pct
print(f'  TOTAL would-be P&L across missed: {total_pct:+.2f}% across {len(missed)} trades')

print()
print('=' * 80)
print(f'BLOCKED BY FILTERS: {len(blocked)}')
print('=' * 80)
block_pct = 0
for b in blocked:
    if len(b) == 4:
        sym, direction, fname, val = b
        print(f'  {sym:12s} {direction:5s}  {fname}')
    else:
        sym, direction, fname, t0, entry, exit_t, exit_price, exit_reason, pnl_pct = b
        print(f'  {t0} -> {exit_t}  {sym:12s} {direction:5s} entry={entry:.1f} exit={exit_price:.1f} [{exit_reason}]  {pnl_pct:+.2f}%  BLOCKED by {fname}')
        block_pct += pnl_pct
print(f'  TOTAL would-be P&L if blocked taken: {block_pct:+.2f}%')

print()
print(f'NO BREAKOUT today: {len(no_signal)} -> {no_signal}')
