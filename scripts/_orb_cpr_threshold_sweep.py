"""ORB CPR-width threshold sweep — 250 trading days × 5 threshold values.

Tests how the cpr_width_threshold_pct parameter affects Calmar, trade count,
and MaxDD when applied to the live V9 (close@14:30) strategy. Keeps all
other filters intact (RSI 60/40, gap 1%, entry 15min OR).

Usage: nohup venv/bin/python3 scripts/_orb_cpr_threshold_sweep.py > cpr_sweep.log 2>&1 &
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
    simulate_variant, compute_rsi, fifteen_min_closes, _naive,
    UNIVERSE, LOT, CAPITAL, OR_MINUTES, GAP_LONG_BLOCK_PCT,
    RSI_LONG_MIN, RSI_SHORT_MAX,
)

OUT_TRADES_CSV = '/home/arun/quantifyd/cpr_threshold_sweep_trades.csv'
OUT_DAILY_CSV = '/home/arun/quantifyd/cpr_threshold_sweep_daily.csv'
TRADE_FIELDS = ['date', 'sym', 'dir', 'threshold', 'entry_t', 'entry',
                'exit_t', 'exit', 'reason', 'pnl_unit', 'pnl_mis', 'qty_mis']
DAILY_FIELDS = ['date', 'threshold', 'day_pnl_mis', 'trades', 'blocked_by_cpr']

# V9 variant only (winner of the earlier sweep)
VARIANT_CFG = {'r_mult': 1.5, 'trail_be_at': None, 'partial_at': None,
               'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '14:30'}

THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7]
N_DAYS = int(os.environ.get('ORB_DAYS', '250'))


def main():
    kite = get_kite()
    # Use cached instrument tokens (avoid rate limit on kite.instruments('NSE'))
    import json as _json
    try:
        with open('/home/arun/quantifyd/backtest_data/instrument_tokens.json') as _f:
            _cache = _json.load(_f)
        s2t = _cache.get('symbol_to_token', {})
        tok_map = {s: s2t[s] for s in UNIVERSE if s in s2t}
        print(f'token cache: {len(tok_map)}/{len(UNIVERSE)} symbols resolved')
    except Exception as e:
        print(f'token cache load failed ({e}) — falling back to API')
        insts = kite.instruments('NSE')
        tok_map = {i['tradingsymbol']: i['instrument_token']
                   for i in insts if i['tradingsymbol'] in UNIVERSE}

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
                if row.get('date') and row.get('sym') and row.get('threshold'):
                    done_pairs.add((row['date'], row['sym'], row['threshold']))
        print(f'RESUME: {len(done_pairs)} (date,sym,thr) triples done')
    else:
        with open(OUT_TRADES_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()
    if not os.path.exists(OUT_DAILY_CSV):
        with open(OUT_DAILY_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=DAILY_FIELDS).writeheader()

    print(f'Sweeping {len(dates)} days x {len(UNIVERSE)} stocks x {len(THRESHOLDS)} thresholds')
    print(f'First: {dates[0]}  last: {dates[-1]}', flush=True)

    # Per-threshold aggregates rebuilt from trades CSV
    agg = {str(t): {'day_pnl': defaultdict(float), 'trades': []} for t in THRESHOLDS}
    if os.path.exists(OUT_TRADES_CSV) and os.path.getsize(OUT_TRADES_CSV) > 0:
        with open(OUT_TRADES_CSV) as f:
            for row in csv.DictReader(f, fieldnames=TRADE_FIELDS):
                thr = row.get('threshold')
                if not thr or thr == 'threshold' or thr not in agg:
                    continue
                try:
                    pnl = float(row['pnl_mis'])
                except (ValueError, TypeError):
                    continue
                agg[thr]['day_pnl'][row['date']] += pnl
                agg[thr]['trades'].append(pnl)

    t0 = _time.time()
    total_breakouts = 0

    for di, dt in enumerate(dates):
        day_iso = dt.isoformat()
        # Per-threshold daily counters
        day_pnl_this = {str(t): 0.0 for t in THRESHOLDS}
        day_trades_this = {str(t): 0 for t in THRESHOLDS}
        day_blocked = {str(t): 0 for t in THRESHOLDS}

        for sym in UNIVERSE:
            if all((day_iso, sym, str(t)) in done_pairs for t in THRESHOLDS):
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
                # Simulate V9 once; price is the same, filter applied per-threshold
                et, ep, er, pnl_per_unit = simulate_variant(
                    candles, idx, entry, direction, or_high, or_low, VARIANT_CFG)
                pnl_mis = pnl_per_unit * qty_mis

                rows_to_write = []
                for thr in THRESHOLDS:
                    thr_s = str(thr)
                    if (day_iso, sym, thr_s) in done_pairs:
                        continue
                    if cpr_w > thr:
                        # blocked — still log a zero row so resume logic sees it
                        day_blocked[thr_s] += 1
                        continue
                    day_pnl_this[thr_s] += pnl_mis
                    day_trades_this[thr_s] += 1
                    agg[thr_s]['day_pnl'][day_iso] += pnl_mis
                    agg[thr_s]['trades'].append(pnl_mis)
                    rows_to_write.append({
                        'date': day_iso, 'sym': sym, 'dir': direction,
                        'threshold': thr_s,
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

        # Per-threshold daily row
        with open(OUT_DAILY_CSV, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=DAILY_FIELDS)
            for thr in THRESHOLDS:
                thr_s = str(thr)
                w.writerow({
                    'date': day_iso, 'threshold': thr_s,
                    'day_pnl_mis': round(day_pnl_this[thr_s], 2),
                    'trades': day_trades_this[thr_s],
                    'blocked_by_cpr': day_blocked[thr_s],
                })

        elapsed = _time.time() - t0
        print(f'  [{di+1}/{len(dates)}] {dt} · elapsed {elapsed:.0f}s · breakouts {total_breakouts}',
              flush=True)

    # ===== Summary =====
    print()
    print('=' * 100)
    print(f'{"CPR thr":>8} {"Trades":>7} {"Wins":>6} {"WR":>6} {"PF":>6} {"Net P&L":>14} {"MaxDD":>10} {"Calmar":>8}')
    print('=' * 100)
    summaries = []
    for thr in THRESHOLDS:
        thr_s = str(thr)
        pnls = agg[thr_s]['trades']
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gw, gl = sum(wins), abs(sum(losses))
        pf = gw / gl if gl > 0 else 999
        total = sum(pnls)
        days = sorted(agg[thr_s]['day_pnl'].keys())
        eq = peak = mdd = 0.0
        for d_ in days:
            eq += agg[thr_s]['day_pnl'][d_]
            peak = max(peak, eq)
            mdd = max(mdd, peak - eq)
        calmar = total / mdd if mdd > 0 else 999
        wr = len(wins) / len(pnls) * 100 if pnls else 0
        summaries.append((thr, len(pnls), wr, pf, total, mdd, calmar))
        print(f'{thr:>8.2f} {len(pnls):>7d} {len(wins):>6d} {wr:>5.1f}% {pf:>6.2f} {total:>+14,.0f} {mdd:>+10,.0f} {calmar:>8.2f}')

    print()
    print('RANKED BY CALMAR')
    for s in sorted(summaries, key=lambda x: -x[6]):
        ret_pct = s[4] / CAPITAL * 100
        dd_pct = s[5] / CAPITAL * 100
        print(f'  thr={s[0]:.2f}  ret={ret_pct:+6.1f}%  DD={dd_pct:5.1f}%  Calmar={s[6]:5.2f}  '
              f'PF={s[3]:.2f}  trades={s[1]}  WR={s[2]:.1f}%')


if __name__ == '__main__':
    main()
