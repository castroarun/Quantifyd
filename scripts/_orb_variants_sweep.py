"""ORB variants sweep — run all exit-rule variants against 60 days of cached
candles in a single pass, store per-variant per-trade P&L, print comparison.

Variants tested (all share the same entry logic + filters as baseline):
  V0  Baseline                     1.5R target, OR-opposite SL, EOD exit at 15:20
  V1  R=1.0                        1.0R target
  V2  R=2.0                        2.0R target
  V3  R=3.0                        3.0R target
  V4  Trail-BE-at-0.5R             1.5R target; move SL to entry after +0.5R
  V5  Trail-BE-at-1R               1.5R target; move SL to entry after +1.0R
  V6  Partial 50% at 1R + BE trail Exit 50% qty at 1R; remainder trails with SL=entry
  V7  ATR trail after 1R           1.5R target; after +1R profit, trail SL by 1 ATR(14, 5m)
  V8  Time-tighten 12:00           SL moves halfway to entry after 12:00
  V9  Close-all 14:30              Force exit at 14:30 (no EOD 15:20)

For MIS sizing, uses capital = Rs 23,10,000 at 5x leverage with
max_concurrent_trades = 5 -> qty per trade = capital / entry (floor).
Futures P&L uses lot size.
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

OUT_TRADES_CSV = '/home/arun/quantifyd/variants_sweep_trades.csv'
OUT_DAILY_CSV = '/home/arun/quantifyd/variants_sweep_daily.csv'
OUT_LOG = '/home/arun/quantifyd/variants_sweep.log'
TRADE_FIELDS = ['date','sym','dir','variant','entry_t','entry','exit_t','exit','reason',
                'pnl_unit','pnl_mis','pnl_fut','qty_mis','qty_fut']
DAILY_FIELDS = ['date','variant','day_pnl_mis','trades']

UNIVERSE = ['ADANIENT','TATASTEEL','BEL','VEDL','BPCL','M&M','BAJFINANCE',
            'TRENT','HAL','IRCTC','GRASIM','GODREJPROP','RELIANCE','AXISBANK','APOLLOHOSP']
LOT = {s: FNO_LOT_SIZES.get(s, 350) for s in UNIVERSE}

CAPITAL = 2_310_000
OR_MINUTES = 15
GAP_LONG_BLOCK_PCT = 1.0
RSI_LONG_MIN = 60
RSI_SHORT_MAX = 40
CPR_WIDTH_THRESHOLD = 0.5

# ===== Variant definitions =====
VARIANTS = [
    ('V0_baseline',        {'r_mult': 1.5, 'trail_be_at': None, 'partial_at': None, 'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '15:20'}),
    ('V1_R_1.0',           {'r_mult': 1.0, 'trail_be_at': None, 'partial_at': None, 'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '15:20'}),
    ('V2_R_2.0',           {'r_mult': 2.0, 'trail_be_at': None, 'partial_at': None, 'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '15:20'}),
    ('V3_R_3.0',           {'r_mult': 3.0, 'trail_be_at': None, 'partial_at': None, 'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '15:20'}),
    ('V4_trailBE_0.5R',    {'r_mult': 1.5, 'trail_be_at': 0.5,  'partial_at': None, 'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '15:20'}),
    ('V5_trailBE_1R',      {'r_mult': 1.5, 'trail_be_at': 1.0,  'partial_at': None, 'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '15:20'}),
    ('V6_partial50_BEtrail',{'r_mult': 1.5,'trail_be_at': None, 'partial_at': 1.0,  'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '15:20'}),
    ('V7_ATR_trail_1R',    {'r_mult': 1.5, 'trail_be_at': None, 'partial_at': None, 'atr_trail_at': 1.0,  'time_tight_at': None, 'force_exit': '15:20'}),
    ('V8_tight_1200',      {'r_mult': 1.5, 'trail_be_at': None, 'partial_at': None, 'atr_trail_at': None, 'time_tight_at': '12:00', 'force_exit': '15:20'}),
    ('V9_close_1430',      {'r_mult': 1.5, 'trail_be_at': None, 'partial_at': None, 'atr_trail_at': None, 'time_tight_at': None, 'force_exit': '14:30'}),
]


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
    out = []; bucket = []; cur = None
    for c in candles_5m:
        t = _naive(c['date'])
        b = t.replace(minute=(t.minute//15)*15, second=0, microsecond=0)
        if cur is None or b != cur:
            if bucket: out.append(bucket[-1]['close'])
            bucket = [c]; cur = b
        else:
            bucket.append(c)
    if bucket: out.append(bucket[-1]['close'])
    return out


def atr_14_5m(candles, idx, period=14):
    """ATR(14) computed from 5-min candles up to idx (exclusive)."""
    if idx < period + 1:
        return None
    trs = []
    for j in range(max(1, idx - period), idx):
        c = candles[j]; p = candles[j-1]
        tr = max(c['high'] - c['low'], abs(c['high'] - p['close']), abs(c['low'] - p['close']))
        trs.append(tr)
    return sum(trs) / len(trs) if trs else None


def simulate_variant(candles, idx, entry, direction, or_high, or_low, variant_cfg):
    """Walk forward from `idx` applying the variant's exit rules.
    Returns (exit_time, exit_price, exit_reason, pnl_per_unit)."""
    # Initial SL (always OR-opposite) and initial risk
    if direction == 'LONG':
        sl = or_low
        risk = entry - sl
        tgt = entry + variant_cfg['r_mult'] * risk
    else:
        sl = or_high
        risk = sl - entry
        tgt = entry - variant_cfg['r_mult'] * risk

    eod = datetime.strptime(variant_cfg['force_exit'], '%H:%M').time()
    tt_at = variant_cfg.get('time_tight_at')
    tt_time = datetime.strptime(tt_at, '%H:%M').time() if tt_at else None

    trail_be_at = variant_cfg.get('trail_be_at')  # move SL to entry after this R
    partial_at = variant_cfg.get('partial_at')    # close 50% at this R, trail rest
    atr_trail_at = variant_cfg.get('atr_trail_at')  # after this R, trail by 1 ATR

    half_exited = False
    half_exit_pnl = 0.0  # per unit (on 0.5 qty weighting)
    current_sl = sl
    time_tightened = False

    for j in range(idx + 1, len(candles)):
        c = candles[j]
        t = _naive(c['date'])
        t_time = t.time()

        # Time-based SL tighten (once)
        if tt_time and not time_tightened and t_time >= tt_time:
            # Move SL halfway to entry
            current_sl = (current_sl + entry) / 2
            time_tightened = True

        # Force EOD exit
        if t_time >= eod:
            if direction == 'LONG':
                pnl = (c['close'] - entry)
            else:
                pnl = (entry - c['close'])
            # If half-exited earlier, blend: 50% at half_exit, 50% at this close
            if half_exited:
                per_unit = 0.5 * half_exit_pnl + 0.5 * pnl
            else:
                per_unit = pnl
            return t, c['close'], 'EOD', per_unit

        # Compute current profit in R terms
        if direction == 'LONG':
            cur_profit_r = (c['high'] - entry) / risk if risk > 0 else 0
            # Trail-BE
            if trail_be_at is not None and cur_profit_r >= trail_be_at:
                current_sl = max(current_sl, entry)
            # ATR trail
            if atr_trail_at is not None and cur_profit_r >= atr_trail_at:
                atr = atr_14_5m(candles, j)
                if atr:
                    current_sl = max(current_sl, c['close'] - atr)
            # Partial exit at R
            if partial_at is not None and not half_exited and cur_profit_r >= partial_at:
                half_exit_pnl = partial_at * risk  # locked at +partial_at * risk per unit
                half_exited = True
                current_sl = max(current_sl, entry)  # BE trail for remainder
            # SL hit
            if c['low'] <= current_sl:
                pnl_per_unit = current_sl - entry
                if half_exited:
                    per_unit = 0.5 * half_exit_pnl + 0.5 * pnl_per_unit
                else:
                    per_unit = pnl_per_unit
                return t, current_sl, 'SL', per_unit
            # Target hit (only if partial not taken)
            if not half_exited and c['high'] >= tgt:
                return t, tgt, 'TGT', tgt - entry
        else:  # SHORT
            cur_profit_r = (entry - c['low']) / risk if risk > 0 else 0
            if trail_be_at is not None and cur_profit_r >= trail_be_at:
                current_sl = min(current_sl, entry)
            if atr_trail_at is not None and cur_profit_r >= atr_trail_at:
                atr = atr_14_5m(candles, j)
                if atr:
                    current_sl = min(current_sl, c['close'] + atr)
            if partial_at is not None and not half_exited and cur_profit_r >= partial_at:
                half_exit_pnl = partial_at * risk
                half_exited = True
                current_sl = min(current_sl, entry)
            if c['high'] >= current_sl:
                pnl_per_unit = entry - current_sl
                if half_exited:
                    per_unit = 0.5 * half_exit_pnl + 0.5 * pnl_per_unit
                else:
                    per_unit = pnl_per_unit
                return t, current_sl, 'SL', per_unit
            if not half_exited and c['low'] <= tgt:
                return t, tgt, 'TGT', entry - tgt

    # No candle hit anything — exit on last
    last = candles[-1]
    if direction == 'LONG':
        pnl = last['close'] - entry
    else:
        pnl = entry - last['close']
    if half_exited:
        per_unit = 0.5 * half_exit_pnl + 0.5 * pnl
    else:
        per_unit = pnl
    return _naive(last['date']), last['close'], 'OPEN', per_unit


def main():
    kite = get_kite()
    insts = kite.instruments('NSE')
    tok_map = {i['tradingsymbol']: i['instrument_token']
               for i in insts if i['tradingsymbol'] in UNIVERSE}

    # Build list of trading days (last 60 weekdays)
    today = date.today()
    dates = []
    d = today - timedelta(days=1)
    while len(dates) < 60:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    dates.reverse()

    # Resume: load already-processed dates from daily CSV
    # Use explicit fieldnames so resume works even if header row is missing/stripped.
    done_dates = set()
    if os.path.exists(OUT_DAILY_CSV) and os.path.getsize(OUT_DAILY_CSV) > 0:
        with open(OUT_DAILY_CSV) as f:
            r = csv.DictReader(f, fieldnames=DAILY_FIELDS)
            for row in r:
                if row.get('date') and row['date'] != 'date':
                    done_dates.add(row['date'])
        print(f'RESUME: {len(done_dates)} dates already processed')
    else:
        with open(OUT_DAILY_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=DAILY_FIELDS).writeheader()
    if not (os.path.exists(OUT_TRADES_CSV) and os.path.getsize(OUT_TRADES_CSV) > 0):
        with open(OUT_TRADES_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()

    print(f'Sweeping {len(dates)} trading days x {len(UNIVERSE)} stocks x {len(VARIANTS)} variants')
    print(f'First date: {dates[0]}, last date: {dates[-1]}', flush=True)

    # per-variant per-day pnl (in Rs, MIS sized) — rebuilt from CSV on start
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
                except (TypeError, ValueError):
                    continue
                variant_day_pnl[vname][row['date']] += pnl_mis
                variant_trades[vname].append({**row, 'pnl_mis': pnl_mis, 'pnl_fut': float(row['pnl_fut'])})

    t0 = _time.time()
    total_trades_counted = sum(len(v) for v in variant_trades.values()) // max(1, len(VARIANTS))

    for di, dt in enumerate(dates):
        if dt.isoformat() in done_dates:
            continue
        day_trades_this_date = defaultdict(int)  # variant -> count
        day_pnl_this_date = defaultdict(float)   # variant -> pnl
        session_start = datetime.combine(dt, datetime.min.time()).replace(hour=9, minute=15)
        session_end = datetime.combine(dt, datetime.min.time()).replace(hour=15, minute=30)
        or_end = session_start + timedelta(minutes=OR_MINUTES)

        for sym in UNIVERSE:
            tok = tok_map.get(sym)
            if not tok: continue
            try:
                candles = kite.historical_data(tok, session_start, session_end, '5minute')
            except Exception as e:
                if 'Too many' in str(e):
                    _time.sleep(2)
                continue
            if not candles: continue
            today_open = candles[0]['open']

            try:
                daily = kite.historical_data(tok,
                    datetime.combine(dt-timedelta(days=15), datetime.min.time()),
                    datetime.combine(dt, datetime.min.time()), 'day')
            except Exception:
                continue
            prev = None
            for d_ in reversed(daily or []):
                dd = d_['date'].date() if hasattr(d_['date'], 'date') else d_['date']
                if dd < dt: prev = d_; break
            if not prev: continue

            ph, pl, pc = prev['high'], prev['low'], prev['close']
            pivot = (ph+pl+pc)/3
            bc = (ph+pl)/2; tc = 2*pivot - bc
            cpr_w = abs(tc-bc)/pivot*100 if pivot > 0 else 0
            gap_pct = (today_open - pc)/pc*100 if pc else 0
            if cpr_w > CPR_WIDTH_THRESHOLD: continue

            or_candles = [c for c in candles if _naive(c['date']) < or_end]
            if len(or_candles) < 3: continue
            or_high = max(c['high'] for c in or_candles)
            or_low = min(c['low'] for c in or_candles)
            rsi_val = compute_rsi(fifteen_min_closes(candles), 14)

            # Find breakouts
            long_br = short_br = None
            for i in range(1, len(candles)):
                prev_c, cur = candles[i-1], candles[i]
                ct = _naive(cur['date'])
                if ct < or_end: continue
                if ct.time() >= datetime.strptime('15:00', '%H:%M').time(): break
                if not long_br and cur['close'] > or_high and prev_c['close'] <= or_high:
                    long_br = (ct, cur['close'], i)
                if not short_br and cur['close'] < or_low and prev_c['close'] >= or_low:
                    short_br = (ct, cur['close'], i)

            # Apply filters
            for direction, br in [('LONG', long_br), ('SHORT', short_br)]:
                if not br: continue
                if direction == 'LONG':
                    if gap_pct > GAP_LONG_BLOCK_PCT: continue
                    if rsi_val is None or rsi_val < RSI_LONG_MIN: continue
                else:
                    if rsi_val is None or rsi_val > RSI_SHORT_MAX: continue

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
                # Write trade rows for this breakout immediately
                with open(OUT_TRADES_CSV, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
                    for r in rows_to_write:
                        w.writerow(r)

        # Write daily summary for all variants
        with open(OUT_DAILY_CSV, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=DAILY_FIELDS)
            for vname, _ in VARIANTS:
                w.writerow({'date': dt.isoformat(), 'variant': vname,
                            'day_pnl_mis': round(day_pnl_this_date.get(vname, 0.0), 2),
                            'trades': day_trades_this_date.get(vname, 0)})

        elapsed = _time.time() - t0
        print(f'  [{di+1}/{len(dates)}] {dt} · elapsed {elapsed:.0f}s · trades so far {total_trades_counted}',
              flush=True)

    # ===== Summary =====
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
        gw = sum(wins); gl = abs(sum(losses))
        pf = gw / gl if gl > 0 else float('inf')
        total = sum(pnls)
        # Drawdown
        days = sorted(variant_day_pnl[vname].keys())
        eq = 0.0; peak = 0.0; max_dd = 0.0
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

    # Sort by Calmar
    print()
    print('RANKED BY CALMAR (risk-adjusted return)')
    for s in sorted(summaries, key=lambda x: x['calmar'], reverse=True):
        dd_pct = s['max_dd'] / CAPITAL * 100
        ret_pct = s['net'] / CAPITAL * 100
        print(f'  {s["name"]:<28}  ret={ret_pct:+6.1f}%  DD={dd_pct:5.1f}%  Calmar={s["calmar"]:5.2f}  PF={s["pf"]:.2f}  WR={s["win_rate"]:.1f}%')


if __name__ == '__main__':
    main()
