"""Simulated ORB live-run for today. Computes everything from raw Kite candles,
applies live-engine filters, and reports every signal with entry/exit/P&L.

Runs on VPS via _vps_query.py.
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

# Live-engine defaults (mirror config.py ORB_DEFAULTS)
CAPITAL_PER_TRADE = 20000
SL_PCT = 0.005        # 0.5% stop loss (from live engine)
R_MULTIPLE = 1.0      # 1R target
GAP_THRESHOLD_PCT = 1.0   # longs blocked if gap > 1%
RSI_LONG_MIN = 60         # long requires RSI >= 60
RSI_SHORT_MAX = 40        # short requires RSI <= 40
OR_MINUTES = 15
CPR_WIDTH_THRESHOLD = 0.5  # wide CPR if width% > 0.5

insts = kite.instruments('NSE')
tok_map = {i['tradingsymbol']: i['instrument_token'] for i in insts if i['tradingsymbol'] in UNIVERSE}

now = datetime.now()
today = date.today()
session_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
or_end = session_start + timedelta(minutes=OR_MINUTES)  # 09:30


def _naive(dt):
    """Strip tzinfo so comparisons against naive session_start work."""
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        ch = closes[i] - closes[i - 1]
        if ch > 0: gains += ch
        else: losses -= ch
    if losses == 0: return 100.0
    rs = (gains / period) / (losses / period)
    rsi = 100 - 100 / (1 + rs)
    # wilder smoothing for remaining
    for i in range(period + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        g = max(ch, 0); l = max(-ch, 0)
        gains = (gains * (period - 1) + g) / period * period
        losses = (losses * (period - 1) + l) / period * period
        if losses == 0: rsi = 100.0
        else:
            rs = (gains / period) / (losses / period)
            rsi = 100 - 100 / (1 + rs)
    return rsi


def fifteen_min_closes(candles_5m):
    """Aggregate 5-min candles into 15-min closes."""
    out = []
    bucket = []
    cur_start = None
    for c in candles_5m:
        t = _naive(c['date'])
        boundary_min = (t.minute // 15) * 15
        boundary = t.replace(minute=boundary_min, second=0, microsecond=0)
        if cur_start is None or boundary != cur_start:
            if bucket:
                out.append(bucket[-1]['close'])
            bucket = [c]
            cur_start = boundary
        else:
            bucket.append(c)
    if bucket:
        out.append(bucket[-1]['close'])
    return out


header = f"{'Stock':12s} {'OR_H':>9s} {'OR_L':>9s} {'Gap%':>6s} {'CPR%':>6s} {'RSI':>5s} | Action"
print(header)
print('-' * len(header))

taken = []   # passed filters -> simulated trade
blocked = [] # broke out but filter blocked
no_breakout = []

for sym in UNIVERSE:
    token = tok_map.get(sym)
    if not token:
        print(f'{sym:12s}  MISSING TOKEN')
        continue

    # Fetch today's 5-min candles
    try:
        candles = kite.historical_data(token, session_start, now, '5minute')
    except Exception as e:
        print(f'{sym:12s}  ERR fetch 5m: {e}')
        continue
    if not candles:
        print(f'{sym:12s}  no data today')
        continue

    # Today's open = first 5-min candle open (≈9:15)
    today_open = candles[0]['open']

    # Fetch prev day daily for gap + CPR — exclude today (Kite returns today's
    # partial candle which would make CPR computation use today instead of
    # yesterday).
    try:
        daily = kite.historical_data(token,
                                     datetime.combine(today - timedelta(days=15), datetime.min.time()),
                                     datetime.combine(today, datetime.min.time()),
                                     'day')
    except Exception as e:
        print(f'{sym:12s}  ERR fetch daily: {e}')
        continue
    # Pick the last candle whose date < today (strictly previous trading day)
    prev = None
    for d in reversed(daily or []):
        d_date = d['date'].date() if hasattr(d['date'], 'date') else d['date']
        if d_date < today:
            prev = d
            break
    if not prev:
        print(f'{sym:12s}  no prev day data')
        continue
    prev_high, prev_low, prev_close = prev['high'], prev['low'], prev['close']
    pivot = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2
    tc = 2 * pivot - bc
    cpr_width_pct = abs(tc - bc) / pivot * 100
    is_wide_cpr = cpr_width_pct > CPR_WIDTH_THRESHOLD
    gap_pct = (today_open - prev_close) / prev_close * 100

    # OR15 from first 3 candles (9:15, 9:20, 9:25)
    or_candles = [c for c in candles if _naive(c['date']) < or_end]
    if len(or_candles) < 3:
        print(f'{sym:12s}  only {len(or_candles)} OR candles, skip')
        continue
    or_high = max(c['high'] for c in or_candles)
    or_low = min(c['low'] for c in or_candles)

    # RSI(14) from 15-min aggregated closes (use 5m candles prior to latest time)
    rsi_closes = fifteen_min_closes(candles)
    rsi_val = compute_rsi(rsi_closes, period=14)

    if is_wide_cpr:
        print(f'{sym:12s}  {or_high:9.2f} {or_low:9.2f} {gap_pct:6.2f} {cpr_width_pct:6.3f} {"":>5s} | SKIP wide CPR')
        blocked.append({'sym': sym, 'dir': '-', 'reason': f'wide CPR {cpr_width_pct:.2f}%'})
        continue

    rsi_str = f'{rsi_val:.1f}' if rsi_val is not None else '--'

    # Walk post-OR candles, take first breakout (close above/below OR)
    long_break = None
    short_break = None
    for i in range(1, len(candles)):
        prev_c, cur = candles[i-1], candles[i]
        if _naive(cur['date']) < or_end: continue
        if _naive(cur['date']).time() >= datetime.strptime('15:00', '%H:%M').time(): break
        if not long_break and cur['close'] > or_high and prev_c['close'] <= or_high:
            long_break = (_naive(cur['date']), cur['close'], i)
        if not short_break and cur['close'] < or_low and prev_c['close'] >= or_low:
            short_break = (_naive(cur['date']), cur['close'], i)
        if long_break and short_break:
            break

    def simulate(direction, break_info):
        """Return (exit_time, exit_price, exit_reason, pnl_pct)."""
        t0, entry, idx = break_info
        if direction == 'LONG':
            sl = entry * (1 - SL_PCT)
            tgt = entry + (entry - sl) * R_MULTIPLE
        else:
            sl = entry * (1 + SL_PCT)
            tgt = entry - (sl - entry) * R_MULTIPLE
        for j in range(idx + 1, len(candles)):
            c = candles[j]
            t = _naive(c['date'])
            if t.time() >= datetime.strptime('15:20', '%H:%M').time():
                return t, c['close'], 'EOD', ((c['close']-entry)/entry*100 if direction=='LONG' else (entry-c['close'])/entry*100)
            if direction == 'LONG':
                if c['low'] <= sl: return t, sl, 'SL', -SL_PCT*100
                if c['high'] >= tgt: return t, tgt, 'TGT', SL_PCT*100*R_MULTIPLE
            else:
                if c['high'] >= sl: return t, sl, 'SL', -SL_PCT*100
                if c['low'] <= tgt: return t, tgt, 'TGT', SL_PCT*100*R_MULTIPLE
        last = candles[-1]
        return _naive(last['date']), last['close'], 'OPEN', ((last['close']-entry)/entry*100 if direction=='LONG' else (entry-last['close'])/entry*100)

    actions = []
    for direction, br in [('LONG', long_break), ('SHORT', short_break)]:
        if not br: continue
        # Apply filters
        block_reason = None
        if direction == 'LONG':
            if gap_pct > GAP_THRESHOLD_PCT:
                block_reason = f'gap {gap_pct:.2f}%>1%'
            elif rsi_val is None or rsi_val < RSI_LONG_MIN:
                block_reason = f'RSI {rsi_str}<60'
        else:
            if rsi_val is None or rsi_val > RSI_SHORT_MAX:
                block_reason = f'RSI {rsi_str}>40'

        t0, entry, _ = br
        exit_t, exit_px, exit_reason, pnl_pct = simulate(direction, br)
        qty = max(1, int(CAPITAL_PER_TRADE / entry))
        pnl_inr = round((exit_px - entry) * qty if direction == 'LONG' else (entry - exit_px) * qty, 2)

        tag = f'{direction}@{t0.strftime("%H:%M")}'
        if block_reason:
            actions.append(f'{tag} BLOCK {block_reason} (would be {pnl_pct:+.2f}% / Rs{pnl_inr:+.0f})')
            blocked.append({'sym': sym, 'dir': direction, 'reason': block_reason,
                            'entry_time': t0.strftime('%H:%M'), 'entry': entry,
                            'exit_time': exit_t.strftime('%H:%M'), 'exit': exit_px,
                            'exit_reason': exit_reason, 'pnl_pct': round(pnl_pct, 2),
                            'pnl_inr': pnl_inr})
        else:
            actions.append(f'{tag} TAKE entry={entry:.1f} -> {exit_px:.1f}@{exit_t.strftime("%H:%M")} [{exit_reason}] {pnl_pct:+.2f}% / Rs{pnl_inr:+.0f}')
            taken.append({'sym': sym, 'dir': direction,
                          'entry_time': t0.strftime('%H:%M'), 'entry': entry,
                          'exit_time': exit_t.strftime('%H:%M'), 'exit': exit_px,
                          'exit_reason': exit_reason, 'pnl_pct': round(pnl_pct, 2),
                          'pnl_inr': pnl_inr})

    if not long_break and not short_break:
        actions.append('No breakout')
        no_breakout.append(sym)

    print(f'{sym:12s}  {or_high:9.2f} {or_low:9.2f} {gap_pct:6.2f} {cpr_width_pct:6.3f} {rsi_str:>5s} | {" | ".join(actions)}')

print()
print('='*90)
print(f'TRADES TAKEN (passed filters): {len(taken)}')
print('='*90)
total_inr = 0
for t in taken:
    print(f'  {t["entry_time"]}->{t["exit_time"]}  {t["sym"]:12s} {t["dir"]:5s}  entry={t["entry"]:.1f} exit={t["exit"]:.1f} [{t["exit_reason"]}]  {t["pnl_pct"]:+.2f}%  Rs{t["pnl_inr"]:+.0f}')
    total_inr += t['pnl_inr']
print(f'  TOTAL P&L: Rs{total_inr:+.0f} across {len(taken)} trades')

print()
print('='*90)
print(f'BLOCKED BY FILTERS: {len(blocked)}')
print('='*90)
blocked_inr = 0
for b in blocked:
    if 'entry' not in b:
        print(f'  {b["sym"]:12s} {b["dir"]:5s}  {b["reason"]}')
    else:
        print(f'  {b["entry_time"]}->{b["exit_time"]}  {b["sym"]:12s} {b["dir"]:5s}  entry={b["entry"]:.1f} exit={b["exit"]:.1f} [{b["exit_reason"]}]  {b["pnl_pct"]:+.2f}%  Rs{b["pnl_inr"]:+.0f}  BLOCKED: {b["reason"]}')
        blocked_inr += b['pnl_inr']
print(f'  Would-be P&L if blocked taken: Rs{blocked_inr:+.0f}')

print()
print(f'NO BREAKOUT: {len(no_breakout)} -> {no_breakout}')
