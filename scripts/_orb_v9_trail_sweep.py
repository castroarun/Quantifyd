"""ORB V9-trail variants — at 14:30 start trailing instead of force-closing.

Compares:
  V9_base            : force close at 14:30 (current live rule)
  V9t_BE             : at 14:30 move SL to entry; EOD 15:18 or SL
  V9t_ATR1           : at 14:30 start trailing SL by 1*ATR(14,5m); EOD 15:18 or SL
  V9t_ATR2           : at 14:30 start trailing SL by 2*ATR(14,5m); EOD 15:18 or SL
  V9t_lock50         : at 14:30 lock 50% of current profit; EOD 15:18 or SL

All use same entry: OR15 breakout + CPR<0.5% + gap<1% + RSI60/40.
Target: 1.5R. Initial SL: OR-opposite.

250 trading days x 15 stocks.
"""
import os, sys, csv
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')
from dotenv import load_dotenv
load_dotenv('.env')

from collections import defaultdict
from datetime import date, datetime, timedelta
import time as _time

from services.kite_service import get_kite
from services.data_manager import FNO_LOT_SIZES

from scripts._orb_variants_sweep import (
    compute_rsi, fifteen_min_closes, _naive, atr_14_5m,
    UNIVERSE, LOT, CAPITAL, OR_MINUTES, GAP_LONG_BLOCK_PCT,
    RSI_LONG_MIN, RSI_SHORT_MAX, CPR_WIDTH_THRESHOLD,
)

OUT_TRADES_CSV = '/home/arun/quantifyd/v9_trail_trades.csv'
OUT_DAILY_CSV = '/home/arun/quantifyd/v9_trail_daily.csv'
TRADE_FIELDS = ['date', 'sym', 'dir', 'variant', 'entry_t', 'entry',
                'exit_t', 'exit', 'reason', 'pnl_unit', 'pnl_mis', 'qty_mis']
DAILY_FIELDS = ['date', 'variant', 'day_pnl_mis', 'trades']

N_DAYS = int(os.environ.get('ORB_DAYS', '250'))

VARIANT_NAMES = ['V9_base', 'V9t_BE', 'V9t_ATR1', 'V9t_ATR2', 'V9t_lock50']

TRAIL_START = datetime.strptime('14:30', '%H:%M').time()
HARD_EOD = datetime.strptime('15:18', '%H:%M').time()
TGT_R = 1.5


def simulate(variant, candles, idx, entry, direction, or_high, or_low):
    """Walk forward from idx. Apply V9-trail-variant exit logic.
    Returns (exit_time, exit_price, reason, pnl_per_unit)."""
    if direction == 'LONG':
        sl = or_low
        risk = entry - sl
        tgt = entry + TGT_R * risk
    else:
        sl = or_high
        risk = sl - entry
        tgt = entry - TGT_R * risk

    # For the BASELINE V9: force-close at 14:30
    force_close_at = TRAIL_START if variant == 'V9_base' else HARD_EOD
    trail_activated = False

    for j in range(idx + 1, len(candles)):
        c = candles[j]
        t = _naive(c['date'])
        t_time = t.time()

        # --- At 14:30: for V9_base we force-close; for trail variants we activate trailing ---
        if t_time >= TRAIL_START and not trail_activated and variant != 'V9_base':
            trail_activated = True
            if variant == 'V9t_BE':
                # Move SL to entry (only if that tightens)
                if direction == 'LONG':
                    sl = max(sl, entry)
                else:
                    sl = min(sl, entry)
            elif variant == 'V9t_lock50':
                # Lock 50% of current profit
                cur = c['close']
                if direction == 'LONG':
                    gain = cur - entry
                    new_sl = entry + 0.5 * gain if gain > 0 else entry
                    sl = max(sl, new_sl)
                else:
                    gain = entry - cur
                    new_sl = entry - 0.5 * gain if gain > 0 else entry
                    sl = min(sl, new_sl)

        # --- Force close for V9_base ---
        if variant == 'V9_base' and t_time >= force_close_at:
            return t, c['close'], 'FORCE_1430', (c['close'] - entry) if direction == 'LONG' else (entry - c['close'])

        # --- Hard EOD 15:18 for trail variants ---
        if variant != 'V9_base' and t_time >= HARD_EOD:
            return t, c['close'], 'EOD_1518', (c['close'] - entry) if direction == 'LONG' else (entry - c['close'])

        # --- ATR trail (after activation) ---
        if trail_activated and variant in ('V9t_ATR1', 'V9t_ATR2'):
            atr = atr_14_5m(candles, j)
            if atr:
                mult = 1.0 if variant == 'V9t_ATR1' else 2.0
                if direction == 'LONG':
                    sl = max(sl, c['close'] - mult * atr)
                else:
                    sl = min(sl, c['close'] + mult * atr)

        # --- SL / TGT checks ---
        if direction == 'LONG':
            if c['low'] <= sl:
                return t, sl, 'SL', sl - entry
            if c['high'] >= tgt:
                return t, tgt, 'TGT', tgt - entry
        else:
            if c['high'] >= sl:
                return t, sl, 'SL', entry - sl
            if c['low'] <= tgt:
                return t, tgt, 'TGT', entry - tgt

    # Ran out of candles — close at last
    last = candles[-1]
    pnl = (last['close'] - entry) if direction == 'LONG' else (entry - last['close'])
    return _naive(last['date']), last['close'], 'OPEN', pnl


def main():
    kite = get_kite()
    import json as _json
    with open('/home/arun/quantifyd/backtest_data/instrument_tokens.json') as _f:
        _cache = _json.load(_f)
    s2t = _cache.get('symbol_to_token', {})
    tok_map = {s: s2t[s] for s in UNIVERSE if s in s2t}
    print(f'token cache: {len(tok_map)}/{len(UNIVERSE)}')

    today = date.today()
    dates = []
    d = today - timedelta(days=1)
    while len(dates) < N_DAYS:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    dates.reverse()

    done_pairs = set()
    if os.path.exists(OUT_TRADES_CSV) and os.path.getsize(OUT_TRADES_CSV) > 0:
        with open(OUT_TRADES_CSV) as f:
            for row in csv.DictReader(f, fieldnames=TRADE_FIELDS):
                if row.get('date') and row.get('sym') and row.get('variant'):
                    done_pairs.add((row['date'], row['sym'], row['variant']))
        print(f'RESUME: {len(done_pairs)} (date,sym,variant) done')
    else:
        with open(OUT_TRADES_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()
    if not os.path.exists(OUT_DAILY_CSV):
        with open(OUT_DAILY_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=DAILY_FIELDS).writeheader()

    print(f'Sweeping {len(dates)} days x {len(UNIVERSE)} stocks x {len(VARIANT_NAMES)} variants')
    print(f'First: {dates[0]}  last: {dates[-1]}', flush=True)

    agg = {v: {'day_pnl': defaultdict(float), 'trades': []} for v in VARIANT_NAMES}
    if os.path.exists(OUT_TRADES_CSV) and os.path.getsize(OUT_TRADES_CSV) > 0:
        with open(OUT_TRADES_CSV) as f:
            for row in csv.DictReader(f, fieldnames=TRADE_FIELDS):
                v = row.get('variant')
                if v not in agg:
                    continue
                try:
                    pnl = float(row['pnl_mis'])
                except (ValueError, TypeError):
                    continue
                agg[v]['day_pnl'][row['date']] += pnl
                agg[v]['trades'].append(pnl)

    t0 = _time.time()
    total_breakouts = 0

    for di, dt in enumerate(dates):
        day_iso = dt.isoformat()
        day_pnl_this = {v: 0.0 for v in VARIANT_NAMES}
        day_trades_this = {v: 0 for v in VARIANT_NAMES}

        for sym in UNIVERSE:
            if all((day_iso, sym, v) in done_pairs for v in VARIANT_NAMES):
                continue
            tok = tok_map.get(sym)
            if not tok:
                continue
            session_start = datetime.combine(dt, datetime.min.time()).replace(hour=9, minute=15)
            session_end = datetime.combine(dt, datetime.min.time()).replace(hour=15, minute=30)
            or_end = session_start + timedelta(minutes=OR_MINUTES)
            try:
                candles = kite.historical_data(tok, session_start, session_end, '5minute')
            except Exception as e:
                if 'Too many' in str(e):
                    _time.sleep(2)
                continue
            if not candles:
                continue
            today_open = candles[0]['open']
            try:
                daily = kite.historical_data(tok,
                    datetime.combine(dt - timedelta(days=15), datetime.min.time()),
                    datetime.combine(dt, datetime.min.time()), 'day')
            except Exception:
                continue
            prev = None
            for d_ in reversed(daily or []):
                dd = d_['date'].date() if hasattr(d_['date'], 'date') else d_['date']
                if dd < dt:
                    prev = d_
                    break
            if not prev:
                continue

            ph, pl, pc = prev['high'], prev['low'], prev['close']
            pivot = (ph + pl + pc) / 3
            bc = (ph + pl) / 2
            tc = 2 * pivot - bc
            cpr_w = abs(tc - bc) / pivot * 100 if pivot > 0 else 0
            gap_pct = (today_open - pc) / pc * 100 if pc else 0
            if cpr_w > CPR_WIDTH_THRESHOLD:
                continue

            or_candles = [c for c in candles if _naive(c['date']) < or_end]
            if len(or_candles) < 3:
                continue
            or_high = max(c['high'] for c in or_candles)
            or_low = min(c['low'] for c in or_candles)
            rsi_val = compute_rsi(fifteen_min_closes(candles), 14)

            long_br = short_br = None
            for i in range(1, len(candles)):
                prev_c, cur = candles[i - 1], candles[i]
                ct = _naive(cur['date'])
                if ct < or_end:
                    continue
                if ct.time() >= datetime.strptime('15:00', '%H:%M').time():
                    break
                if not long_br and cur['close'] > or_high and prev_c['close'] <= or_high:
                    long_br = (ct, cur['close'], i)
                if not short_br and cur['close'] < or_low and prev_c['close'] >= or_low:
                    short_br = (ct, cur['close'], i)

            for direction, br in [('LONG', long_br), ('SHORT', short_br)]:
                if not br:
                    continue
                if direction == 'LONG':
                    if gap_pct > GAP_LONG_BLOCK_PCT:
                        continue
                    if rsi_val is None or rsi_val < RSI_LONG_MIN:
                        continue
                else:
                    if rsi_val is None or rsi_val > RSI_SHORT_MAX:
                        continue

                t_entry, entry, idx = br
                qty_mis = int(CAPITAL / entry)
                total_breakouts += 1

                rows_to_write = []
                for v in VARIANT_NAMES:
                    if (day_iso, sym, v) in done_pairs:
                        continue
                    et, ep, er, pnl_per_unit = simulate(v, candles, idx, entry, direction, or_high, or_low)
                    pnl_mis = pnl_per_unit * qty_mis
                    day_pnl_this[v] += pnl_mis
                    day_trades_this[v] += 1
                    agg[v]['day_pnl'][day_iso] += pnl_mis
                    agg[v]['trades'].append(pnl_mis)
                    rows_to_write.append({
                        'date': day_iso, 'sym': sym, 'dir': direction,
                        'variant': v,
                        'entry_t': t_entry.strftime('%H:%M'), 'entry': round(entry, 2),
                        'exit_t': et.strftime('%H:%M'), 'exit': round(ep, 2), 'reason': er,
                        'pnl_unit': round(pnl_per_unit, 4),
                        'pnl_mis': round(pnl_mis, 2), 'qty_mis': qty_mis,
                    })

                if rows_to_write:
                    with open(OUT_TRADES_CSV, 'a', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
                        for r in rows_to_write:
                            w.writerow(r)

        with open(OUT_DAILY_CSV, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=DAILY_FIELDS)
            for v in VARIANT_NAMES:
                w.writerow({'date': day_iso, 'variant': v,
                            'day_pnl_mis': round(day_pnl_this[v], 2),
                            'trades': day_trades_this[v]})

        elapsed = _time.time() - t0
        print(f'  [{di+1}/{len(dates)}] {dt} · elapsed {elapsed:.0f}s · breakouts {total_breakouts}',
              flush=True)

    print()
    print('=' * 100)
    print(f'{"Variant":<14} {"Trades":>7} {"Wins":>6} {"WR":>6} {"PF":>6} {"Net P&L":>14} {"MaxDD":>10} {"Calmar":>8}')
    print('=' * 100)
    summaries = []
    for v in VARIANT_NAMES:
        pnls = agg[v]['trades']
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gw, gl = sum(wins), abs(sum(losses))
        pf = gw / gl if gl > 0 else 999
        total = sum(pnls)
        days = sorted(agg[v]['day_pnl'].keys())
        eq = peak = mdd = 0.0
        for d_ in days:
            eq += agg[v]['day_pnl'][d_]
            peak = max(peak, eq)
            mdd = max(mdd, peak - eq)
        calmar = total / mdd if mdd > 0 else 999
        wr = len(wins) / len(pnls) * 100 if pnls else 0
        summaries.append((v, len(pnls), wr, pf, total, mdd, calmar))
        print(f'{v:<14} {len(pnls):>7d} {len(wins):>6d} {wr:>5.1f}% {pf:>6.2f} {total:>+14,.0f} {mdd:>+10,.0f} {calmar:>8.2f}')

    print()
    print('RANKED BY CALMAR')
    for s in sorted(summaries, key=lambda x: -x[6]):
        ret_pct = s[4] / CAPITAL * 100
        dd_pct = s[5] / CAPITAL * 100
        print(f'  {s[0]:<14}  ret={ret_pct:+6.1f}%  DD={dd_pct:5.1f}%  Calmar={s[6]:5.2f}  '
              f'PF={s[3]:.2f}  trades={s[1]}  WR={s[2]:.1f}%')


if __name__ == '__main__':
    main()
