"""ORB + prev-day-breakout confluence sweep.

Tests whether ORB signals that ALSO break the previous day's high (long) or
low (short) have better conviction than plain ORB breakouts.

All three variants use V9 exits (force close 14:30). Only the ENTRY filter
differs:
  B (baseline)      : plain V9 — 5min close past OR boundary
  B+PDB             : entry close must also be past prev_day_high (long)
                      or prev_day_low (short)
  B+PDB_strict      : same as PDB + candle high > prev_day_high (long)
                      or low < prev_day_low (short) — confirms intraday break

250 trading days × 15 stocks.
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
    RSI_LONG_MIN, RSI_SHORT_MAX, CPR_WIDTH_THRESHOLD,
)

OUT_TRADES_CSV = '/home/arun/quantifyd/prev_day_confluence_trades.csv'
OUT_DAILY_CSV = '/home/arun/quantifyd/prev_day_confluence_daily.csv'
TRADE_FIELDS = ['date', 'sym', 'dir', 'variant', 'entry_t', 'entry',
                'exit_t', 'exit', 'reason', 'pnl_unit', 'pnl_mis', 'qty_mis',
                'prev_high', 'prev_low']
DAILY_FIELDS = ['date', 'variant', 'day_pnl_mis', 'trades']

V9 = {'r_mult': 1.5, 'trail_be_at': None, 'partial_at': None,
      'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '14:30'}
VARIANT_NAMES = ['B_baseline', 'B_PDB', 'B_PDB_strict']
N_DAYS = int(os.environ.get('ORB_DAYS', '250'))


def main():
    kite = get_kite()
    # Cached instrument tokens
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
        print(f'RESUME: {len(done_pairs)} (date,sym,variant) triples done')
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
                    long_br = (ct, cur['close'], i, cur)
                if not short_br and cur['close'] < or_low and prev_c['close'] >= or_low:
                    short_br = (ct, cur['close'], i, cur)

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

                t_entry, entry, idx, brk_candle = br
                qty_mis = int(CAPITAL / entry)
                total_breakouts += 1

                # Simulate V9 once; the result applies wherever an entry is taken.
                et, ep, er, pnl_per_unit = simulate_variant(
                    candles, idx, entry, direction, or_high, or_low, V9)
                pnl_mis = pnl_per_unit * qty_mis

                # Decide which variants take this trade
                # B_baseline: always takes
                # B_PDB: close past prev day level
                # B_PDB_strict: close past prev day level AND candle reach past too
                variants_to_take = ['B_baseline']
                if direction == 'LONG':
                    if brk_candle['close'] > ph:
                        variants_to_take.append('B_PDB')
                        if brk_candle['high'] > ph:
                            variants_to_take.append('B_PDB_strict')
                else:
                    if brk_candle['close'] < pl:
                        variants_to_take.append('B_PDB')
                        if brk_candle['low'] < pl:
                            variants_to_take.append('B_PDB_strict')

                rows_to_write = []
                for v in VARIANT_NAMES:
                    if (day_iso, sym, v) in done_pairs:
                        continue
                    if v not in variants_to_take:
                        continue
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
                        'prev_high': round(ph, 2), 'prev_low': round(pl, 2),
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

    # ===== Summary =====
    print()
    print('=' * 100)
    print(f'{"Variant":<18} {"Trades":>7} {"Wins":>6} {"WR":>6} {"PF":>6} {"Net P&L":>14} {"MaxDD":>10} {"Calmar":>8}')
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
        print(f'{v:<18} {len(pnls):>7d} {len(wins):>6d} {wr:>5.1f}% {pf:>6.2f} {total:>+14,.0f} {mdd:>+10,.0f} {calmar:>8.2f}')

    print()
    print('RANKED BY CALMAR')
    for s in sorted(summaries, key=lambda x: -x[6]):
        ret_pct = s[4] / CAPITAL * 100
        dd_pct = s[5] / CAPITAL * 100
        print(f'  {s[0]:<18}  ret={ret_pct:+6.1f}%  DD={dd_pct:5.1f}%  Calmar={s[6]:5.2f}  '
              f'PF={s[3]:.2f}  trades={s[1]}  WR={s[2]:.1f}%')


if __name__ == '__main__':
    main()
