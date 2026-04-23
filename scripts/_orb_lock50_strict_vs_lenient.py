"""V9t_lock50 — STRICT (cut losers at BE at 14:30) vs LENIENT (keep OR-opposite SL).

STRICT  (matches _orb_v9_trail_sweep.py V9t_lock50 — the 676-Calmar backtest)
  At 14:30:
    profitable → SL = entry ± 0.5 × gain  (lock half)
    losing     → SL = entry              (breakeven)

LENIENT (matches current live activate_trail_lock50 as of 2026-04-23)
  At 14:30:
    profitable → SL = entry ± 0.5 × gain
    losing     → SL unchanged (OR-opposite)

Both use 15:18 hard EOD and same entry/target rules. 60-day window for
speed (the question is strict-vs-lenient RELATIVE performance, not an
absolute parameter sweep).
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
from scripts._orb_variants_sweep import (
    compute_rsi, fifteen_min_closes, _naive,
    UNIVERSE, CAPITAL, OR_MINUTES, GAP_LONG_BLOCK_PCT,
    RSI_LONG_MIN, RSI_SHORT_MAX, CPR_WIDTH_THRESHOLD,
)

OUT = '/home/arun/quantifyd/v9_lock50_compare.csv'
FIELDS = ['date', 'sym', 'dir', 'variant', 'entry_t', 'entry',
          'exit_t', 'exit', 'reason', 'pnl_unit', 'pnl_mis', 'qty_mis']

N_DAYS = int(os.environ.get('ORB_DAYS', '60'))
THROTTLE = float(os.environ.get('THROTTLE', '0.35'))  # 0.35s per Kite call ≈ 3 req/sec

TRAIL_START = datetime.strptime('14:30', '%H:%M').time()
HARD_EOD = datetime.strptime('15:18', '%H:%M').time()
TGT_R = 1.5


def simulate_both(candles, idx, entry, direction, or_high, or_low):
    """Walk forward once, return (strict_exit, lenient_exit) tuples.
    Each is (time, price, reason, pnl_per_unit)."""
    if direction == 'LONG':
        sl0 = or_low
        tgt = entry + TGT_R * (entry - sl0)
    else:
        sl0 = or_high
        tgt = entry - TGT_R * (sl0 - entry)

    sl_s, sl_l = sl0, sl0   # strict, lenient
    done_s = done_l = None  # (t, price, reason)
    trail_done = False

    for j in range(idx + 1, len(candles)):
        c = candles[j]
        t = _naive(c['date'])
        tt = t.time()

        # --- 14:30 trail activation (once) ---
        if tt >= TRAIL_START and not trail_done:
            trail_done = True
            cur = c['close']
            if direction == 'LONG':
                gain = cur - entry
                # strict: BE on loss, else lock half
                new_sl_s = (entry + 0.5 * gain) if gain > 0 else entry
                new_sl_l = (entry + 0.5 * gain) if gain > 0 else sl_l
                sl_s = max(sl_s, new_sl_s)
                sl_l = max(sl_l, new_sl_l)
            else:
                gain = entry - cur
                new_sl_s = (entry - 0.5 * gain) if gain > 0 else entry
                new_sl_l = (entry - 0.5 * gain) if gain > 0 else sl_l
                sl_s = min(sl_s, new_sl_s)
                sl_l = min(sl_l, new_sl_l)

        # --- 15:18 hard EOD ---
        if tt >= HARD_EOD:
            pnl = (c['close'] - entry) if direction == 'LONG' else (entry - c['close'])
            if not done_s:
                done_s = (t, c['close'], 'EOD_1518', pnl)
            if not done_l:
                done_l = (t, c['close'], 'EOD_1518', pnl)
            break

        # --- SL / TGT checks ---
        if direction == 'LONG':
            if not done_s and c['low'] <= sl_s:
                done_s = (t, sl_s, 'SL', sl_s - entry)
            if not done_l and c['low'] <= sl_l:
                done_l = (t, sl_l, 'SL', sl_l - entry)
            if not done_s and c['high'] >= tgt:
                done_s = (t, tgt, 'TGT', tgt - entry)
            if not done_l and c['high'] >= tgt:
                done_l = (t, tgt, 'TGT', tgt - entry)
        else:  # SHORT
            if not done_s and c['high'] >= sl_s:
                done_s = (t, sl_s, 'SL', entry - sl_s)
            if not done_l and c['high'] >= sl_l:
                done_l = (t, sl_l, 'SL', entry - sl_l)
            if not done_s and c['low'] <= tgt:
                done_s = (t, tgt, 'TGT', entry - tgt)
            if not done_l and c['low'] <= tgt:
                done_l = (t, tgt, 'TGT', entry - tgt)

        if done_s and done_l:
            break

    if not done_s:
        last = candles[-1]
        pnl = (last['close'] - entry) if direction == 'LONG' else (entry - last['close'])
        done_s = (_naive(last['date']), last['close'], 'OPEN', pnl)
    if not done_l:
        last = candles[-1]
        pnl = (last['close'] - entry) if direction == 'LONG' else (entry - last['close'])
        done_l = (_naive(last['date']), last['close'], 'OPEN', pnl)
    return done_s, done_l


def main():
    kite = get_kite()
    import json as _json
    with open('/home/arun/quantifyd/backtest_data/instrument_tokens.json') as f:
        cache = _json.load(f)
    s2t = cache.get('symbol_to_token', {})
    tok_map = {s: s2t[s] for s in UNIVERSE if s in s2t}
    print(f'token cache: {len(tok_map)}/{len(UNIVERSE)}', flush=True)

    today = date.today()
    dates = []
    d = today - timedelta(days=1)
    while len(dates) < N_DAYS:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    dates.reverse()
    print(f'Sweeping {len(dates)} days x {len(UNIVERSE)} stocks. First: {dates[0]} last: {dates[-1]}', flush=True)

    # Resume
    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        with open(OUT) as f:
            for row in csv.DictReader(f):
                done.add((row.get('date'), row.get('sym'), row.get('variant')))
        print(f'resume: {len(done)} rows already', flush=True)
    else:
        with open(OUT, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    agg = {'strict': defaultdict(float), 'lenient': defaultdict(float)}
    trades = {'strict': [], 'lenient': []}

    # Seed agg from CSV on resume
    if done:
        with open(OUT) as f:
            for row in csv.DictReader(f):
                v = row['variant']
                if v in agg:
                    pnl = float(row['pnl_mis'] or 0)
                    agg[v][row['date']] += pnl
                    trades[v].append(pnl)

    t0 = _time.time()
    total_breakouts = 0

    for di, dt in enumerate(dates):
        day_iso = dt.isoformat()

        for sym in UNIVERSE:
            if (day_iso, sym, 'strict') in done and (day_iso, sym, 'lenient') in done:
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
            _time.sleep(THROTTLE)
            if not candles:
                continue
            today_open = candles[0]['open']
            try:
                daily = kite.historical_data(tok,
                    datetime.combine(dt - timedelta(days=15), datetime.min.time()),
                    datetime.combine(dt, datetime.min.time()), 'day')
            except Exception:
                continue
            _time.sleep(THROTTLE)
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

                exit_s, exit_l = simulate_both(candles, idx, entry, direction, or_high, or_low)

                rows = []
                for variant_name, ex in [('strict', exit_s), ('lenient', exit_l)]:
                    if (day_iso, sym, variant_name) in done:
                        continue
                    et, ep, er, pnl_unit = ex
                    pnl_mis = pnl_unit * qty_mis
                    agg[variant_name][day_iso] += pnl_mis
                    trades[variant_name].append(pnl_mis)
                    rows.append({
                        'date': day_iso, 'sym': sym, 'dir': direction, 'variant': variant_name,
                        'entry_t': t_entry.strftime('%H:%M'), 'entry': round(entry, 2),
                        'exit_t': et.strftime('%H:%M'), 'exit': round(ep, 2), 'reason': er,
                        'pnl_unit': round(pnl_unit, 4),
                        'pnl_mis': round(pnl_mis, 2), 'qty_mis': qty_mis,
                    })
                if rows:
                    with open(OUT, 'a', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=FIELDS)
                        for r in rows:
                            w.writerow(r)

        elapsed = _time.time() - t0
        print(f'  [{di+1}/{len(dates)}] {dt} · elapsed {elapsed:.0f}s · breakouts {total_breakouts}', flush=True)

    # --- Summary ---
    import statistics
    def metrics(variant):
        pnls = trades[variant]
        daily = list(agg[variant].values())
        if not pnls:
            return {}
        total = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        daily_sum = list(agg[variant].items())
        # Equity curve and max drawdown
        eq = 0
        peak = 0
        max_dd = 0
        for d, p in sorted(daily_sum):
            eq += p
            peak = max(peak, eq)
            max_dd = min(max_dd, eq - peak)
        sharpe = (statistics.mean(daily) / statistics.stdev(daily) * (252 ** 0.5)) if len(daily) > 1 and statistics.stdev(daily) > 0 else 0
        cagr = total / CAPITAL * (252 / max(len(daily), 1)) * 100  # rough annualized %
        calmar = (cagr / abs(max_dd / CAPITAL * 100)) if max_dd < 0 else 0
        return {
            'trades': len(pnls),
            'total_pnl_mis': round(total, 2),
            'avg_pnl': round(total / len(pnls), 2) if pnls else 0,
            'win_rate': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            'avg_win': round(statistics.mean(wins), 2) if wins else 0,
            'avg_loss': round(statistics.mean(losses), 2) if losses else 0,
            'days': len(daily),
            'sharpe_rough': round(sharpe, 2),
            'cagr_rough_%': round(cagr, 1),
            'max_dd_mis': round(max_dd, 2),
            'calmar_rough': round(calmar, 2),
        }

    print()
    print('=' * 80)
    print('STRICT (backtest fidelity)  vs  LENIENT (current live)')
    print('=' * 80)
    ms = metrics('strict')
    ml = metrics('lenient')
    for k in sorted(set(list(ms.keys()) + list(ml.keys()))):
        print(f'  {k:<20s}  strict={ms.get(k, 0):>12}  lenient={ml.get(k, 0):>12}')
    print()
    if ms.get('total_pnl_mis', 0) > ml.get('total_pnl_mis', 0):
        edge = ms['total_pnl_mis'] - ml['total_pnl_mis']
        print(f'  >> STRICT better by Rs {edge:,.0f} over {ms["days"]} days ({edge / ms["days"]:.0f}/day)')
    else:
        edge = ml.get('total_pnl_mis', 0) - ms.get('total_pnl_mis', 0)
        print(f'  >> LENIENT better by Rs {edge:,.0f} over {ml["days"]} days ({edge / ml["days"]:.0f}/day)')


if __name__ == '__main__':
    main()
