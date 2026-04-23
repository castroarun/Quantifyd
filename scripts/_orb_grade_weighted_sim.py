"""Grade-weighted ORB position sizing — simulate + analyse.

PHASE 1: walk 250 days × 15 stocks, find valid breakouts (same filters as
live), tag each trade with its conviction grade (same 4-star rubric as
live), run STRICT V9t_lock50 exit, log to CSV with grade column.

PHASE 2 (post-processing, no Kite calls): compute aggregate metrics
under 4 weighting schemes:
  flat      : A+/A/B/C all size at 0.8% risk (baseline, what's live)
  gentle    : A+ 1.2%, A 1.0%, B 0.8%, C 0.6%
  aggressive: A+ 1.6%, A 1.2%, B 0.8%, C 0.4%
  a_only    : A+ 1.5%, A 1.0%, B/C skipped

4-star rubric (matches services/orb_live_engine._compute_conviction):
  ⭐ CPR < 0.3%
  ⭐ OR width < 0.8% of entry
  ⭐ RSI ≥ 65 (LONG) / ≤ 35 (SHORT)
  ⭐ past_pct in 0.2–0.7% (at entry, distance past OR)
Grade: A+ = 4 stars, A = 3, B = 2, C = ≤1
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

OUT = '/home/arun/quantifyd/v9_grade_weighted.csv'
FIELDS = ['date', 'sym', 'dir', 'entry_t', 'entry', 'exit_t', 'exit', 'reason',
          'pnl_unit', 'pnl_mis', 'qty_mis',
          'grade', 'score',
          'cpr_narrow', 'tight_risk', 'rsi_conviction', 'clean_past',
          'cpr_pct', 'or_width_pct', 'rsi_val', 'past_pct']

N_DAYS = int(os.environ.get('ORB_DAYS', '250'))
THROTTLE = float(os.environ.get('THROTTLE', '0.35'))

TRAIL_START = datetime.strptime('14:30', '%H:%M').time()
HARD_EOD = datetime.strptime('15:18', '%H:%M').time()
TGT_R = 1.5


def compute_conviction(side, past_pct, cpr_pct, or_width_pct, rsi):
    """Returns (grade, score, star flags dict). Same rubric as live."""
    stars = {
        'cpr_narrow':     cpr_pct is not None and cpr_pct < 0.3,
        'tight_risk':     or_width_pct is not None and or_width_pct < 0.8,
        'rsi_conviction': (rsi is not None and rsi >= 65) if side == 'LONG'
                          else (rsi is not None and rsi <= 35),
        'clean_past':     past_pct is not None and 0.2 <= past_pct <= 0.7,
    }
    score = sum(1 for v in stars.values() if v)
    grade = 'A+' if score == 4 else 'A' if score == 3 else 'B' if score == 2 else 'C'
    return grade, score, stars


def simulate_strict_lock50(candles, idx, entry, direction, or_high, or_low):
    """V9t_lock50 STRICT exit. Returns (exit_t, exit_price, reason, pnl_per_unit)."""
    if direction == 'LONG':
        sl = or_low
        tgt = entry + TGT_R * (entry - sl)
    else:
        sl = or_high
        tgt = entry - TGT_R * (sl - entry)

    trail_done = False
    for j in range(idx + 1, len(candles)):
        c = candles[j]
        t = _naive(c['date'])
        tt = t.time()

        # 14:30 strict trail: lock half on profit, breakeven on loss
        if tt >= TRAIL_START and not trail_done:
            trail_done = True
            cur = c['close']
            if direction == 'LONG':
                gain = cur - entry
                new_sl = (entry + 0.5 * gain) if gain > 0 else entry
                sl = max(sl, new_sl)
            else:
                gain = entry - cur
                new_sl = (entry - 0.5 * gain) if gain > 0 else entry
                sl = min(sl, new_sl)

        if tt >= HARD_EOD:
            pnl = (c['close'] - entry) if direction == 'LONG' else (entry - c['close'])
            return t, c['close'], 'EOD_1518', pnl

        if direction == 'LONG':
            if c['low'] <= sl:   return t, sl,  'SL',  sl - entry
            if c['high'] >= tgt: return t, tgt, 'TGT', tgt - entry
        else:
            if c['high'] >= sl:  return t, sl,  'SL',  entry - sl
            if c['low'] <= tgt:  return t, tgt, 'TGT', entry - tgt

    last = candles[-1]
    pnl = (last['close'] - entry) if direction == 'LONG' else (entry - last['close'])
    return _naive(last['date']), last['close'], 'OPEN', pnl


def phase1_sim(kite, tok_map, dates):
    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        with open(OUT) as f:
            for row in csv.DictReader(f):
                done.add((row['date'], row['sym'], row['dir']))
        print(f'resume: {len(done)} trades already logged', flush=True)
    else:
        with open(OUT, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    t0 = _time.time()
    total_breakouts = 0

    for di, dt in enumerate(dates):
        day_iso = dt.isoformat()
        for sym in UNIVERSE:
            # Skip if both LONG and SHORT slots already done (rare; usually max 1)
            if (day_iso, sym, 'LONG') in done and (day_iso, sym, 'SHORT') in done:
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
                if (day_iso, sym, direction) in done:
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

                # Conviction at entry
                past_pct = (entry - or_high) / or_high * 100 if direction == 'LONG' \
                    else (or_low - entry) / or_low * 100
                or_width_pct = ((or_high - or_low) / entry * 100) if entry else None
                grade, score, stars = compute_conviction(
                    direction, past_pct, cpr_w, or_width_pct, rsi_val,
                )

                et, ep, er, pnl_unit = simulate_strict_lock50(
                    candles, idx, entry, direction, or_high, or_low,
                )
                pnl_mis = pnl_unit * qty_mis

                row = {
                    'date': day_iso, 'sym': sym, 'dir': direction,
                    'entry_t': t_entry.strftime('%H:%M'), 'entry': round(entry, 2),
                    'exit_t': et.strftime('%H:%M'), 'exit': round(ep, 2), 'reason': er,
                    'pnl_unit': round(pnl_unit, 4),
                    'pnl_mis': round(pnl_mis, 2), 'qty_mis': qty_mis,
                    'grade': grade, 'score': score,
                    'cpr_narrow':     int(stars['cpr_narrow']),
                    'tight_risk':     int(stars['tight_risk']),
                    'rsi_conviction': int(stars['rsi_conviction']),
                    'clean_past':     int(stars['clean_past']),
                    'cpr_pct':        round(cpr_w, 4),
                    'or_width_pct':   round(or_width_pct, 4) if or_width_pct else None,
                    'rsi_val':        round(rsi_val, 2) if rsi_val else None,
                    'past_pct':       round(past_pct, 4),
                }
                with open(OUT, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                done.add((day_iso, sym, direction))

        elapsed = _time.time() - t0
        print(f'  [{di+1}/{len(dates)}] {dt} · elapsed {elapsed:.0f}s · breakouts {total_breakouts}', flush=True)


def phase2_analyse():
    """Apply weighting schemes to the trade log and print aggregate metrics."""
    import statistics
    schemes = {
        'flat':       {'A+': 1.0, 'A': 1.0, 'B': 1.0, 'C': 1.0},     # 0.8% baseline
        'gentle':     {'A+': 1.5, 'A': 1.25, 'B': 1.0, 'C': 0.75},   # 1.2/1.0/0.8/0.6
        'aggressive': {'A+': 2.0, 'A': 1.5, 'B': 1.0, 'C': 0.5},     # 1.6/1.2/0.8/0.4
        'a_only':     {'A+': 1.875, 'A': 1.25, 'B': 0.0, 'C': 0.0},  # 1.5/1.0/skip/skip
    }

    trades = []
    with open(OUT) as f:
        for row in csv.DictReader(f):
            try:
                row['pnl_mis'] = float(row['pnl_mis'])
                row['qty_mis'] = int(row['qty_mis'])
                trades.append(row)
            except (ValueError, TypeError):
                continue

    print()
    print(f'Trade log: {len(trades)} trades')
    # Grade distribution
    from collections import Counter
    gc = Counter(t['grade'] for t in trades)
    print(f'Grade distribution: ' + ' · '.join(f'{g}={gc.get(g, 0)}' for g in ('A+', 'A', 'B', 'C')))

    # Per-grade baseline stats
    print()
    print('Per-grade stats (flat-baseline P&L):')
    print(f"  {'grade':<4} {'n':>5} {'win%':>7} {'avg_pnl':>10} {'total':>14} {'avg_win':>10} {'avg_loss':>10}")
    for g in ('A+', 'A', 'B', 'C'):
        bucket = [t for t in trades if t['grade'] == g]
        if not bucket:
            continue
        pnls = [t['pnl_mis'] for t in bucket]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        avg_pnl = statistics.mean(pnls)
        win_rate = len(wins) / len(pnls) * 100
        total = sum(pnls)
        avg_w = statistics.mean(wins) if wins else 0
        avg_l = statistics.mean(losses) if losses else 0
        print(f"  {g:<4} {len(pnls):>5} {win_rate:>6.1f}% {avg_pnl:>10,.0f} {total:>14,.0f} {avg_w:>10,.0f} {avg_l:>10,.0f}")

    # Now apply weighting schemes
    print()
    print('=' * 100)
    print('Weighting schemes (multiplier on baseline qty per grade)')
    print('=' * 100)
    print(f"  {'scheme':<12} {'trades':>7} {'total_pnl':>14} {'avg_daily':>11} {'maxDD':>12} {'calmar':>9} {'sharpe':>8} {'win%':>6}")
    for name, mult in schemes.items():
        daily = defaultdict(float)
        pnls = []
        kept = 0
        for t in trades:
            m = mult.get(t['grade'], 0)
            if m == 0:
                continue
            p = t['pnl_mis'] * m
            pnls.append(p)
            daily[t['date']] += p
            kept += 1
        if not pnls:
            continue
        total = sum(pnls)
        days = list(daily.values())
        avg_daily = total / len(daily)
        # Max DD on cumulative curve
        eq = 0; peak = 0; mdd = 0
        for d, p in sorted(daily.items()):
            eq += p
            peak = max(peak, eq)
            mdd = min(mdd, eq - peak)
        cagr = total / CAPITAL * (252 / max(len(daily), 1)) * 100
        calmar = (cagr / abs(mdd / CAPITAL * 100)) if mdd < 0 else 0
        sharpe = (statistics.mean(days) / statistics.stdev(days) * (252 ** 0.5)) \
                 if len(days) > 1 and statistics.stdev(days) > 0 else 0
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f"  {name:<12} {kept:>7} {total:>14,.0f} {avg_daily:>11,.0f} {mdd:>12,.0f} "
              f"{calmar:>9.1f} {sharpe:>8.2f} {win_rate:>5.1f}%")


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
    print(f'Sweeping {len(dates)} days x {len(UNIVERSE)} stocks. '
          f'First: {dates[0]} last: {dates[-1]}', flush=True)

    phase1_sim(kite, tok_map, dates)
    phase2_analyse()


if __name__ == '__main__':
    main()
