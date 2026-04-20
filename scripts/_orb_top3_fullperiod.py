"""ORB top-3 variants — extended backtest over ~1 year of 5-min candles.

Tests ONLY V0 (baseline), V5 (trail-BE @ 1R), V6 (partial 50% @ 1R + BE trail),
V9 (close@14:30). Produces the same ranked-by-Calmar output as the 60-day
sweep but across 250 trading days (full Kite 5-min historical depth).

Reuses the simulate_variant() engine from _orb_variants_sweep.py.
Resumable: skips (date, symbol) pairs already in the trades CSV.
"""
import os, sys
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')
from dotenv import load_dotenv
load_dotenv('.env')

import csv
from collections import defaultdict
from datetime import date, datetime, timedelta
import time as _time

from services.kite_service import get_kite
from services.data_manager import FNO_LOT_SIZES

# Import engine functions from the existing sweep module
from scripts._orb_variants_sweep import (
    simulate_variant, compute_rsi, fifteen_min_closes, _naive,
    UNIVERSE, LOT, CAPITAL, OR_MINUTES, GAP_LONG_BLOCK_PCT,
    RSI_LONG_MIN, RSI_SHORT_MAX, CPR_WIDTH_THRESHOLD,
)

OUT_TRADES_CSV = '/home/arun/quantifyd/variants_top3_trades.csv'
OUT_DAILY_CSV = '/home/arun/quantifyd/variants_top3_daily.csv'
TRADE_FIELDS = ['date','sym','dir','variant','entry_t','entry','exit_t','exit','reason',
                'pnl_unit','pnl_mis','pnl_fut','qty_mis','qty_fut']
DAILY_FIELDS = ['date','variant','day_pnl_mis','trades']

# Only the top 3 + baseline
VARIANTS = [
    ('V0_baseline',         {'r_mult': 1.5, 'trail_be_at': None, 'partial_at': None, 'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '15:20'}),
    ('V5_trailBE_1R',       {'r_mult': 1.5, 'trail_be_at': 1.0,  'partial_at': None, 'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '15:20'}),
    ('V6_partial50_BEtrail',{'r_mult': 1.5, 'trail_be_at': None, 'partial_at': 1.0,  'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '15:20'}),
    ('V9_close_1430',       {'r_mult': 1.5, 'trail_be_at': None, 'partial_at': None, 'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '14:30'}),
]

N_DAYS = int(os.environ.get('ORB_DAYS', '250'))


def main():
    kite = get_kite()
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

    # Resume: track processed (date, sym) pairs so we can pick up after a halt
    done_pairs = set()
    if os.path.exists(OUT_TRADES_CSV) and os.path.getsize(OUT_TRADES_CSV) > 0:
        with open(OUT_TRADES_CSV) as f:
            for row in csv.DictReader(f, fieldnames=TRADE_FIELDS):
                if row.get('date') and row.get('sym'):
                    done_pairs.add((row['date'], row['sym']))
        print(f'RESUME: {len(done_pairs)} (date,sym) pairs already processed')
    else:
        with open(OUT_TRADES_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()
    if not os.path.exists(OUT_DAILY_CSV):
        with open(OUT_DAILY_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=DAILY_FIELDS).writeheader()

    print(f'Sweeping {len(dates)} trading days x {len(UNIVERSE)} stocks x {len(VARIANTS)} variants')
    print(f'First date: {dates[0]}, last date: {dates[-1]}', flush=True)

    # Rebuild aggregates from CSV (for resume)
    variant_day_pnl = {name: defaultdict(float) for name, _ in VARIANTS}
    variant_trades = {name: [] for name, _ in VARIANTS}
    if os.path.exists(OUT_TRADES_CSV) and os.path.getsize(OUT_TRADES_CSV) > 0:
        with open(OUT_TRADES_CSV) as f:
            for row in csv.DictReader(f, fieldnames=TRADE_FIELDS):
                vname = row.get('variant')
                if not vname or vname == 'variant' or vname not in variant_day_pnl:
                    continue
                try:
                    pnl_mis = float(row['pnl_mis'])
                    pnl_fut = float(row['pnl_fut'])
                except (TypeError, ValueError):
                    continue
                variant_day_pnl[vname][row['date']] += pnl_mis
                variant_trades[vname].append({**row, 'pnl_mis': pnl_mis, 'pnl_fut': pnl_fut})

    t0 = _time.time()
    total_trades_counted = sum(len(v) for v in variant_trades.values()) // max(1, len(VARIANTS))

    for di, dt in enumerate(dates):
        day_pnl_this_date = defaultdict(float)
        day_trades_this_date = defaultdict(int)

        for sym in UNIVERSE:
            if (dt.isoformat(), sym) in done_pairs:
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
                msg = str(e)
                if 'Too many' in msg:
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
                qty_fut = LOT.get(sym, 350)
                total_trades_counted += 1

                rows_to_write = []
                for vname, vcfg in VARIANTS:
                    et, ep, er, pnl_per_unit = simulate_variant(
                        candles, idx, entry, direction, or_high, or_low, vcfg)
                    pnl_mis = pnl_per_unit * qty_mis
                    pnl_fut = pnl_per_unit * qty_fut
                    variant_day_pnl[vname][dt.isoformat()] += pnl_mis
                    day_pnl_this_date[vname] += pnl_mis
                    day_trades_this_date[vname] += 1
                    row = {
                        'date': dt.isoformat(), 'sym': sym, 'dir': direction,
                        'variant': vname,
                        'entry_t': t_entry.strftime('%H:%M'), 'entry': round(entry, 2),
                        'exit_t': et.strftime('%H:%M'), 'exit': round(ep, 2), 'reason': er,
                        'pnl_unit': round(pnl_per_unit, 4),
                        'pnl_mis': round(pnl_mis, 2), 'pnl_fut': round(pnl_fut, 2),
                        'qty_mis': qty_mis, 'qty_fut': qty_fut,
                    }
                    variant_trades[vname].append(row)
                    rows_to_write.append(row)

                with open(OUT_TRADES_CSV, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
                    for r in rows_to_write:
                        w.writerow(r)

            done_pairs.add((dt.isoformat(), sym))

        with open(OUT_DAILY_CSV, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=DAILY_FIELDS)
            for vname, _ in VARIANTS:
                w.writerow({'date': dt.isoformat(), 'variant': vname,
                            'day_pnl_mis': round(day_pnl_this_date.get(vname, 0.0), 2),
                            'trades': day_trades_this_date.get(vname, 0)})

        elapsed = _time.time() - t0
        print(f'  [{di+1}/{len(dates)}] {dt} · elapsed {elapsed:.0f}s · trades so far {total_trades_counted}',
              flush=True)

    print()
    print('=' * 110)
    print(f'{"Variant":<28} {"Trades":>7} {"Wins":>6} {"WinRate":>8} {"PF":>6} {"Net P&L (MIS)":>15} {"MaxDD":>12} {"Calmar":>7}')
    print('=' * 110)

    summaries = []
    for vname, _ in VARIANTS:
        trades = variant_trades[vname]
        pnls = [t['pnl_mis'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gw = sum(wins)
        gl = abs(sum(losses))
        pf = gw / gl if gl > 0 else float('inf')
        total = sum(pnls)
        days = sorted(variant_day_pnl[vname].keys())
        eq = 0.0
        peak = 0.0
        max_dd = 0.0
        for d in days:
            eq += variant_day_pnl[vname][d]
            peak = max(peak, eq)
            max_dd = max(max_dd, peak - eq)
        calmar = total / max_dd if max_dd > 0 else float('inf')
        wr = len(wins) / len(pnls) * 100 if pnls else 0
        summaries.append({
            'name': vname, 'trades': len(pnls), 'wins': len(wins),
            'win_rate': wr, 'pf': pf, 'net': total, 'max_dd': max_dd,
            'calmar': calmar,
        })
        print(f'{vname:<28} {len(pnls):>7d} {len(wins):>6d} {wr:>7.1f}% {pf:>6.2f} {total:>+15,.0f} {max_dd:>+12,.0f} {calmar:>7.2f}')

    print()
    print('RANKED BY CALMAR (risk-adjusted return)')
    for s in sorted(summaries, key=lambda x: x['calmar'], reverse=True):
        dd_pct = s['max_dd'] / CAPITAL * 100
        ret_pct = s['net'] / CAPITAL * 100
        print(f'  {s["name"]:<28}  ret={ret_pct:+6.1f}%  DD={dd_pct:5.1f}%  Calmar={s["calmar"]:5.2f}  PF={s["pf"]:.2f}  WR={s["win_rate"]:.1f}%')


if __name__ == '__main__':
    main()
