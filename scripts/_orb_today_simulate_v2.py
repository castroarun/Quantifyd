"""ORB today-sim — uses CORRECT live-engine SL/TGT logic:
   SL = opposite OR boundary, Target = entry ± 1.5 × risk.
"""
import os, sys
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')
from dotenv import load_dotenv
load_dotenv('.env')

from datetime import datetime, date, timedelta
from services.kite_service import get_kite

kite = get_kite()

UNIVERSE = ['ADANIENT','TATASTEEL','BEL','VEDL','BPCL','M&M','BAJFINANCE','TRENT','HAL','IRCTC','GRASIM','GODREJPROP','RELIANCE','AXISBANK','APOLLOHOSP']

CAPITAL_PER_TRADE = 20000
R_MULTIPLE = 1.5
GAP_LONG_BLOCK_PCT = 1.0
RSI_LONG_MIN = 60
RSI_SHORT_MAX = 40
CPR_WIDTH_THRESHOLD = 0.5
OR_MINUTES = 15

insts = kite.instruments('NSE')
tok_map = {i['tradingsymbol']: i['instrument_token'] for i in insts if i['tradingsymbol'] in UNIVERSE}

now = datetime.now()
today = date.today()
session_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
or_end = session_start + timedelta(minutes=OR_MINUTES)


def _naive(dt):
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        ch = closes[i] - closes[i-1]
        if ch > 0: gains += ch
        else: losses -= ch
    if losses == 0: return 100.0
    rs = (gains/period) / (losses/period)
    rsi = 100 - 100/(1+rs)
    for i in range(period+1, len(closes)):
        ch = closes[i] - closes[i-1]
        g = max(ch, 0); l = max(-ch, 0)
        gains = gains * (period-1) + g
        losses = losses * (period-1) + l
        if losses == 0: rsi = 100.0
        else:
            rs = (gains/period) / (losses/period)
            rsi = 100 - 100/(1+rs)
    return rsi


def fifteen_min_closes(candles_5m):
    out = []; bucket = []; cur_start = None
    for c in candles_5m:
        t = _naive(c['date'])
        b = t.replace(minute=(t.minute//15)*15, second=0, microsecond=0)
        if cur_start is None or b != cur_start:
            if bucket: out.append(bucket[-1]['close'])
            bucket = [c]; cur_start = b
        else:
            bucket.append(c)
    if bucket: out.append(bucket[-1]['close'])
    return out


header = f"{'Stock':12s} {'OR_H':>8s} {'OR_L':>8s} {'Gap%':>6s} {'CPR%':>6s} {'RSI':>5s} | Action"
print(header); print('-'*len(header))

taken = []; blocked = []

for sym in UNIVERSE:
    token = tok_map.get(sym)
    if not token: continue
    try:
        candles = kite.historical_data(token, session_start, now, '5minute')
    except Exception as e:
        print(f'{sym:12s}  ERR 5m: {e}'); continue
    if not candles: continue

    today_open = candles[0]['open']
    try:
        daily = kite.historical_data(token,
            datetime.combine(today-timedelta(days=15), datetime.min.time()),
            datetime.combine(today, datetime.min.time()), 'day')
    except Exception as e:
        print(f'{sym:12s}  ERR daily: {e}'); continue

    prev = None
    for d in reversed(daily or []):
        d_date = d['date'].date() if hasattr(d['date'], 'date') else d['date']
        if d_date < today: prev = d; break
    if not prev:
        print(f'{sym:12s}  no prev day'); continue

    ph, pl, pc = prev['high'], prev['low'], prev['close']
    pivot = (ph+pl+pc)/3
    bc = (ph+pl)/2; tc = 2*pivot - bc
    cpr_w = abs(tc-bc)/pivot*100 if pivot > 0 else 0
    gap_pct = (today_open - pc)/pc*100 if pc else 0

    or_candles = [c for c in candles if _naive(c['date']) < or_end]
    if len(or_candles) < 3: continue
    or_high = max(c['high'] for c in or_candles)
    or_low = min(c['low'] for c in or_candles)
    rsi_val = compute_rsi(fifteen_min_closes(candles), 14)
    rsi_str = f'{rsi_val:.1f}' if rsi_val is not None else '--'

    if cpr_w > CPR_WIDTH_THRESHOLD:
        print(f'{sym:12s} {or_high:8.2f} {or_low:8.2f} {gap_pct:6.2f} {cpr_w:6.3f} {"":>5s} | SKIP wide CPR')
        continue

    long_break = short_break = None
    for i in range(1, len(candles)):
        prev_c, cur = candles[i-1], candles[i]
        ct = _naive(cur['date'])
        if ct < or_end: continue
        if ct.time() >= datetime.strptime('15:00', '%H:%M').time(): break
        if not long_break and cur['close'] > or_high and prev_c['close'] <= or_high:
            long_break = (ct, cur['close'], i)
        if not short_break and cur['close'] < or_low and prev_c['close'] >= or_low:
            short_break = (ct, cur['close'], i)

    def simulate(direction, br):
        t0, entry, idx = br
        # CORRECT live-engine logic: SL = opposite OR boundary
        if direction == 'LONG':
            sl = or_low
            risk = entry - sl
            tgt = entry + R_MULTIPLE * risk
        else:
            sl = or_high
            risk = sl - entry
            tgt = entry - R_MULTIPLE * risk
        for j in range(idx+1, len(candles)):
            c = candles[j]; t = _naive(c['date'])
            if t.time() >= datetime.strptime('15:20', '%H:%M').time():
                pct = (c['close']-entry)/entry*100 if direction=='LONG' else (entry-c['close'])/entry*100
                return t, c['close'], 'EOD', pct, sl, tgt, risk
            if direction == 'LONG':
                if c['low'] <= sl:
                    return t, sl, 'SL', -risk/entry*100, sl, tgt, risk
                if c['high'] >= tgt:
                    return t, tgt, 'TGT', R_MULTIPLE*risk/entry*100, sl, tgt, risk
            else:
                if c['high'] >= sl:
                    return t, sl, 'SL', -risk/entry*100, sl, tgt, risk
                if c['low'] <= tgt:
                    return t, tgt, 'TGT', R_MULTIPLE*risk/entry*100, sl, tgt, risk
        last = candles[-1]
        pct = (last['close']-entry)/entry*100 if direction=='LONG' else (entry-last['close'])/entry*100
        return _naive(last['date']), last['close'], 'OPEN', pct, sl, tgt, risk

    actions = []
    for direction, br in [('LONG', long_break), ('SHORT', short_break)]:
        if not br: continue
        t0, entry, _ = br
        block_reason = None
        if direction == 'LONG':
            if gap_pct > GAP_LONG_BLOCK_PCT: block_reason = f'gap {gap_pct:.2f}%>1%'
            elif rsi_val is None or rsi_val < RSI_LONG_MIN: block_reason = f'RSI {rsi_str}<60'
        else:
            if rsi_val is None or rsi_val > RSI_SHORT_MAX: block_reason = f'RSI {rsi_str}>40'

        exit_t, exit_px, reason, pnl_pct, sl, tgt, risk = simulate(direction, br)
        qty = max(1, int(CAPITAL_PER_TRADE/entry))
        pnl_inr = round((exit_px - entry) * qty if direction == 'LONG' else (entry - exit_px) * qty, 2)

        trade_info = dict(sym=sym, dir=direction, t0=t0.strftime('%H:%M'),
                          entry=entry, sl=sl, tgt=tgt, risk=risk,
                          exit_t=exit_t.strftime('%H:%M'), exit=exit_px, reason=reason,
                          pnl_pct=round(pnl_pct, 2), pnl_inr=pnl_inr, qty=qty)

        if block_reason:
            actions.append(f'{direction}@{t0.strftime("%H:%M")} BLOCK {block_reason} (would be {pnl_pct:+.2f}% / Rs{pnl_inr:+.0f})')
            trade_info['block'] = block_reason; blocked.append(trade_info)
        else:
            actions.append(f'{direction}@{t0.strftime("%H:%M")} entry={entry:.1f} SL={sl:.1f} TGT={tgt:.1f} -> {exit_px:.1f}@{exit_t.strftime("%H:%M")} [{reason}] {pnl_pct:+.2f}% Rs{pnl_inr:+.0f}')
            taken.append(trade_info)

    if not actions: actions.append('No breakout')
    print(f'{sym:12s} {or_high:8.2f} {or_low:8.2f} {gap_pct:6.2f} {cpr_w:6.3f} {rsi_str:>5s} | {" | ".join(actions)}')

print()
print('='*90)
print(f'TRADES TAKEN (correct SL=OR_opposite, TGT=1.5R): {len(taken)}')
print('='*90)
total = 0
for t in taken:
    print(f'  {t["t0"]}->{t["exit_t"]}  {t["sym"]:12s} {t["dir"]:5s} qty={t["qty"]:4d} entry={t["entry"]:.2f} SL={t["sl"]:.2f} TGT={t["tgt"]:.2f} risk={t["risk"]:.2f} exit={t["exit"]:.2f} [{t["reason"]}] {t["pnl_pct"]:+.2f}% Rs{t["pnl_inr"]:+.0f}')
    total += t['pnl_inr']
print(f'  TOTAL: Rs{total:+.0f}')

print()
print('='*90)
print(f'BLOCKED: {len(blocked)}')
print('='*90)
for b in blocked:
    print(f'  {b["t0"]}->{b["exit_t"]}  {b["sym"]:12s} {b["dir"]:5s} entry={b["entry"]:.2f} SL={b["sl"]:.2f} TGT={b["tgt"]:.2f} exit={b["exit"]:.2f} [{b["reason"]}] Rs{b["pnl_inr"]:+.0f}  BLOCK: {b["block"]}')
