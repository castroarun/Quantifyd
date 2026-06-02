#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NWV Phase-1 backtest harness  (runs ON the Quantifyd host)
==========================================================
Replays the live NWV view engine over history, builds the next-week-expiry
debit spread the user specified (ATM long, 200-wide short, 5 lots), and
measures P&L under several management policies.

Pricing model
-------------
Real EOD option prices (nse_options_bhav) CALIBRATE a per-day implied vol
(invert the ATM contract of the chosen expiry). The engine's own
black_scholes_put/call (services/spread_structure.py) then prices the spread
at the 09:45 entry and at any intraday timestamp (15/30-min bars). Real EOD
is also used to VALIDATE the model (mean abs error reported).

Stage 1 (this run): IV calibration sanity + view distribution + baseline
P&L (enter 09:45 Mon, exit Fri EOD, no intraday management), stratified by
conviction. Intraday stop / morph / Friday-grid sweeps layer on next.
"""
import sqlite3, sys, math, csv, json
from datetime import datetime, date, timedelta
from collections import defaultdict, Counter

sys.path.insert(0, '/home/arun/quantifyd')
from services.nwv_engine import (
    compute_cpr, compute_pivots, classify_cpr_width, classify_first_candle,
    classify_gap, base_view_matrix, apply_gap_dampener, apply_monthly_override,
    score_conviction, adx_bucket,
    VIEW_BEARISH, VIEW_NTB, VIEW_NEUTRAL, VIEW_NTBULL, VIEW_BULLISH, VIEW_IGNORE,
)
from services.spread_structure import black_scholes_put, black_scholes_call, round_to_strike

DB = '/home/arun/quantifyd/backtest_data/market_data.db'
SPOT = 'NIFTY50'
LOT = 65
LOTS = 5
WIDTH = 200          # points between long and short strike
R = 0.06

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
cur = con.cursor()

# ───────────────────────── data access ─────────────────────────

def bars(timeframe):
    """All NIFTY50 bars for a timeframe, ordered, as list of dict."""
    rows = cur.execute(
        "SELECT date,open,high,low,close,volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe=? ORDER BY date", (SPOT, timeframe)).fetchall()
    out = []
    for r in rows:
        d = r['date']
        out.append({'date': d, 'open': r['open'], 'high': r['high'],
                    'low': r['low'], 'close': r['close'], 'volume': r['volume']})
    return out

DAY = bars('day')
M30 = bars('30minute')
# index 30min by calendar date -> list of bars that day
m30_by_day = defaultdict(list)
for b in M30:
    m30_by_day[b['date'][:10]].append(b)
day_by_date = {b['date'][:10]: b for b in DAY}
DAY_DATES = [b['date'][:10] for b in DAY]

# 5-min bars -> resample to 15-min (for the 15-min stop-confirmation timeframe)
M5 = bars('5minute')
def resample_15(m5):
    buckets = {}
    for b in m5:
        ts = b['date']; hh = int(ts[11:13]); mm = int(ts[14:16])
        floor = (mm // 15) * 15
        key = f"{ts[:11]}{hh:02d}:{floor:02d}:00"
        g = buckets.setdefault(key, {'date': key, 'open': b['open'], 'high': b['high'],
                                     'low': b['low'], 'close': b['close']})
        g['high'] = max(g['high'], b['high']); g['low'] = min(g['low'], b['low'])
        g['close'] = b['close']
    return sorted(buckets.values(), key=lambda x: x['date'])
M15 = resample_15(M5)
m15_by_day = defaultdict(list)
for b in M15:
    m15_by_day[b['date'][:10]].append(b)

def intraday_bars(tf_by_day, d0, d1):
    """All bars (tf) with date in [d0,d1], ascending."""
    out = []
    for d in sorted(tf_by_day):
        if d0 <= d <= d1:
            out += tf_by_day[d]
    return out

def stochastic(m30bars, k=14, d=3):
    """%K(14) fast stoch + %D(3) SMA, keyed by bar date. Bullish CO = %K>%D
    this bar and %K<=%D prev bar."""
    res = {}
    ks = []
    for i in range(len(m30bars)):
        win = m30bars[max(0, i - k + 1):i + 1]
        hh = max(b['high'] for b in win); ll = min(b['low'] for b in win)
        kv = 100 * (m30bars[i]['close'] - ll) / (hh - ll) if hh > ll else 50.0
        ks.append(kv)
        dv = sum(ks[max(0, i - d + 1):i + 1]) / min(i + 1, d)
        res[m30bars[i]['date']] = (kv, dv)
    return res

def prev_trading_days(upto_date_str, n):
    """Return up to n daily bars strictly before upto_date_str (ascending)."""
    res = [b for b in DAY if b['date'][:10] < upto_date_str]
    return res[-n:] if n else res

# ───────────────────────── options access ─────────────────────────

def expiries_after(d_str):
    rows = cur.execute(
        "SELECT DISTINCT expiry_date FROM nse_options_bhav "
        "WHERE symbol='NIFTY' AND expiry_date>=? ORDER BY expiry_date", (d_str,)).fetchall()
    return [r['expiry_date'] for r in rows]

def opt_eod(trade_date, expiry, strike, otype):
    r = cur.execute(
        "SELECT close,settle_price FROM nse_options_bhav WHERE symbol='NIFTY' "
        "AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?",
        (trade_date, expiry, float(strike), otype)).fetchone()
    if not r:
        return None
    return r['close'] if r['close'] and r['close'] > 0 else r['settle_price']

def implied_vol(price, spot, strike, dte_days, otype):
    """Bisection invert BS -> IV. Returns None if no solution."""
    if price is None or price <= 0 or dte_days <= 0:
        return None
    f = black_scholes_put if otype == 'PE' else black_scholes_call
    lo, hi = 0.01, 3.0
    plo = f(spot, strike, dte_days, lo, R) - price
    phi = f(spot, strike, dte_days, hi, R) - price
    if plo * phi > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        pm = f(spot, strike, dte_days, mid, R) - price
        if abs(pm) < 1e-4:
            return mid
        if plo * pm < 0:
            hi = mid; phi = pm
        else:
            lo = mid; plo = pm
    return (lo + hi) / 2

def calibrate_iv(trade_date, expiry, spot):
    """Daily IV from the ATM contract of `expiry` on `trade_date`."""
    dte = (datetime.strptime(expiry, '%Y-%m-%d').date()
           - datetime.strptime(trade_date, '%Y-%m-%d').date()).days
    if dte <= 0:
        return None
    atm = round_to_strike(spot, 100)
    for k in (atm, atm - 100, atm + 100, atm - 200, atm + 200):
        for ot in ('CE', 'PE'):
            px = opt_eod(trade_date, expiry, k, ot)
            iv = implied_vol(px, spot, k, dte, 'PE' if ot == 'PE' else 'CE')
            if iv and 0.03 < iv < 2.5:
                return iv
    return None

_legiv_cache = {}
def leg_iv_eod(trade_date, expiry, strike, otype, spot_eod):
    """Per-leg IV backed out of that leg's REAL EOD close (sticky-strike).
    Reprices the leg exactly at EOD; used to interpolate intraday."""
    key = (trade_date, expiry, strike, otype)
    if key in _legiv_cache:
        return _legiv_cache[key]
    dte = (datetime.strptime(expiry, '%Y-%m-%d').date()
           - datetime.strptime(trade_date, '%Y-%m-%d').date()).days
    px = opt_eod(trade_date, expiry, strike, otype)
    iv = implied_vol(px, spot_eod, strike, dte, otype) if (px and dte > 0) else None
    _legiv_cache[key] = iv
    return iv

def bs_leg(spot, strike, dte_days, iv, otype):
    f = black_scholes_put if otype == 'PE' else black_scholes_call
    return f(spot, strike, dte_days, iv, R)

def dte_at(ts_str, expiry):
    """DTE in days for an intraday timestamp 'YYYY-MM-DD HH:MM:00'.
    EOD (15:30) -> integer calendar days remaining (matches leg_iv_eod anchor)."""
    d = datetime.strptime(ts_str[:10], '%Y-%m-%d').date()
    edate = datetime.strptime(expiry, '%Y-%m-%d').date()
    base = (edate - d).days
    hh, mm = int(ts_str[11:13]), int(ts_str[14:16])
    mins_since_open = (hh * 60 + mm) - (9 * 60 + 15)
    frac_remaining = max(0.0, min(1.0, (375 - mins_since_open) / 375.0))
    return max(base + frac_remaining, 1.0 / 365)

def spread_value_intraday(ts_str, spot_now, long_k, short_k, otype, expiry,
                          iv_long, iv_short):
    """Value of the (long-short) debit spread at an intraday timestamp using
    per-leg sticky-strike IVs anchored to that day's real EOD."""
    dte = dte_at(ts_str, expiry)
    vl = bs_leg(spot_now, long_k, dte, iv_long, otype)
    vs = bs_leg(spot_now, short_k, dte, iv_short, otype)
    return vl - vs

# ───────────────────────── ADX (Wilder 14) ─────────────────────────

def adx_at(d_str, period=14):
    hist = prev_trading_days(d_str, 60)
    if len(hist) < period + 2:
        return None
    trs, pdms, ndms = [], [], []
    for i in range(1, len(hist)):
        h, l, pc = hist[i]['high'], hist[i]['low'], hist[i-1]['close']
        ph, pl = hist[i-1]['high'], hist[i-1]['low']
        tr = max(h - l, abs(h - pc), abs(l - pc))
        up, dn = h - ph, pl - l
        pdm = up if (up > dn and up > 0) else 0.0
        ndm = dn if (dn > up and dn > 0) else 0.0
        trs.append(tr); pdms.append(pdm); ndms.append(ndm)
    def wilder(x):
        s = sum(x[:period]); out = [s]
        for v in x[period:]:
            s = s - s / period + v; out.append(s)
        return out
    atr = wilder(trs); pdi = wilder(pdms); ndi = wilder(ndms)
    dxs = []
    for a, p, n in zip(atr, pdi, ndi):
        if a == 0:
            continue
        pdi_v = 100 * p / a; ndi_v = 100 * n / a
        denom = pdi_v + ndi_v
        if denom == 0:
            continue
        dxs.append(100 * abs(pdi_v - ndi_v) / denom)
    if len(dxs) < period:
        return None
    adx = sum(dxs[:period]) / period
    for v in dxs[period:]:
        adx = (adx * (period - 1) + v) / period
    return adx

# ───────────────────────── week iteration ─────────────────────────

def mondays(start, end):
    d = start
    while d.weekday() != 0:
        d += timedelta(days=1)
    while d <= end:
        yield d
        d += timedelta(days=7)

def first_30m_candle(d_str):
    day_bars = m30_by_day.get(d_str)
    if not day_bars:
        return None
    first = min(day_bars, key=lambda b: b['date'])
    if first['date'][11:16] not in ('09:15',):
        return None
    return first

def week_hlc(week_mon_str):
    """Prev week's H/L/C from daily bars (the 5 sessions before this Monday's week)."""
    mon = datetime.strptime(week_mon_str, '%Y-%m-%d').date()
    prev_mon = mon - timedelta(days=7)
    prev_fri = mon - timedelta(days=3)
    seg = [b for b in DAY if prev_mon.isoformat() <= b['date'][:10] <= prev_fri.isoformat()]
    if not seg:
        return None
    hi = max(b['high'] for b in seg); lo = min(b['low'] for b in seg)
    cl = seg[-1]['close']
    return hi, lo, cl, seg[-1]['date'][:10]

def month_hlc(d_str):
    """Prior calendar month H/L/C for monthly CPR."""
    d = datetime.strptime(d_str, '%Y-%m-%d').date()
    first_this = d.replace(day=1)
    last_prev = first_this - timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    seg = [b for b in DAY if first_prev.isoformat() <= b['date'][:10] <= last_prev.isoformat()]
    if not seg:
        return None
    return max(b['high'] for b in seg), min(b['low'] for b in seg), seg[-1]['close']

# ───────────────────────── main replay ─────────────────────────

def build_view(mon):
    """Replay the engine for a Monday -> dict (or None if no candle/week data)."""
    mon_str = mon.isoformat()
    fc = first_30m_candle(mon_str)
    if not fc:
        return None
    wh = week_hlc(mon_str)
    if not wh:
        return None
    prev_h, prev_l, prev_c, prev_fri_date = wh
    prev_fri = day_by_date.get(prev_fri_date)
    prev_fri_close = prev_fri['close'] if prev_fri else prev_c
    cpr = compute_cpr(prev_h, prev_l, prev_c)
    piv = compute_pivots(prev_h, prev_l, prev_c)
    spot = fc['close']
    cpr_w = abs(cpr['tc'] - cpr['bc']) / spot * 100
    bucket = classify_cpr_width(cpr_w)
    fcc = classify_first_candle(fc['open'], fc['high'], fc['low'], fc['close'],
                                cpr['tc'], cpr['bc'])
    gap_pct = (fc['open'] - prev_fri_close) / prev_fri_close * 100 if prev_fri_close else 0
    gap_tier, gap_dir = classify_gap(gap_pct)
    base = base_view_matrix(bucket, fcc['pos'], fcc['body'])
    v_gap, _ = apply_gap_dampener(base, gap_tier, gap_dir)
    mh = month_hlc(mon_str)
    mtc = mbc = None
    if mh:
        mcpr = compute_cpr(*mh); mtc, mbc = mcpr['tc'], mcpr['bc']
    view, _ = apply_monthly_override(v_gap, spot, mtc, mbc)
    adx = adx_at(mon_str)
    conv = score_conviction(fcc, adx, None)
    return {'mon': mon, 'mon_str': mon_str, 'spot': spot, 'view': view, 'conv': conv,
            'bucket': bucket, 'adx': adx, 'gap_pct': gap_pct, 'piv': piv,
            'r1': piv['r1'], 's1': piv['s1'], 'pp': piv['pp']}


def week_sessions(mon_str):
    """Trading-day strings Mon..Fri of this week present in DAY_DATES."""
    fri = (datetime.strptime(mon_str, '%Y-%m-%d').date() + timedelta(days=4)).isoformat()
    return [d for d in DAY_DATES if mon_str <= d <= fri]


def run():
    start, end = date(2024, 3, 4), date(2026, 3, 23)
    rows = []
    view_counts = Counter()
    # study accumulators
    P = defaultdict(list)   # policy_name -> [pnl,...]
    stop_extend = []        # 2020+ spot-only: did R1/S1 stop fire & save? (per directional wk)

    for mon in mondays(start, end):
        v = build_view(mon)
        if not v:
            continue
        view_counts[v['view']] += 1
        if v['view'] not in (VIEW_BEARISH, VIEW_NTB, VIEW_BULLISH, VIEW_NTBULL):
            continue
        bearish = v['view'] in (VIEW_BEARISH, VIEW_NTB)
        mon_str = v['mon_str']; spot = v['spot']

        exps = expiries_after(mon_str)
        if len(exps) < 2:
            continue
        expiry = exps[1]
        dte0 = (datetime.strptime(expiry, '%Y-%m-%d').date() - mon).days
        otype = 'PE' if bearish else 'CE'
        long_k = round_to_strike(spot, 100)
        short_k = long_k - WIDTH if bearish else long_k + WIDTH

        # ENTRY: real EOD debit on Monday (pure real data)
        rl0 = opt_eod(mon_str, expiry, long_k, otype)
        rs0 = opt_eod(mon_str, expiry, short_k, otype)
        if not (rl0 and rs0):
            continue
        debit = rl0 - rs0
        if debit <= 0:
            continue
        maxv = WIDTH  # spread max value

        sessions = week_sessions(mon_str)
        if not sessions:
            continue
        exit_date = sessions[-1]

        # per-day per-leg IVs (sticky strike, anchored to real EOD)
        leg_iv = {}
        for d in sessions:
            ed = day_by_date[d]['close']
            leg_iv[d] = (leg_iv_eod(d, expiry, long_k, otype, ed),
                         leg_iv_eod(d, expiry, short_k, otype, ed))

        def spread_val_eod(d):
            rl = opt_eod(d, expiry, long_k, otype); rs = opt_eod(d, expiry, short_k, otype)
            if rl and rs:
                return rl - rs
            il, is_ = leg_iv[d]
            if il and is_:
                dte = (datetime.strptime(expiry, '%Y-%m-%d').date()
                       - datetime.strptime(d, '%Y-%m-%d').date()).days
                return bs_leg(day_by_date[d]['close'], long_k, dte, il, otype) - \
                       bs_leg(day_by_date[d]['close'], short_k, dte, is_, otype)
            return None

        def spread_val_intraday(ts, spot_now):
            d = ts[:10]; il, is_ = leg_iv.get(d, (None, None))
            if not (il and is_):
                return None
            return spread_value_intraday(ts, spot_now, long_k, short_k, otype, expiry, il, is_)

        def pnl(exit_val):
            return (exit_val - debit) * LOT * LOTS

        # ── BASELINE: hold to Friday EOD (real) ──
        base_exit = spread_val_eod(exit_date)
        if base_exit is None:
            continue
        base_pnl = pnl(base_exit)
        P['baseline_FriEOD'].append(base_pnl)
        if v['conv'] >= 4: P['baseline_conv>=4'].append(base_pnl)
        if v['conv'] == 3: P['baseline_conv=3'].append(base_pnl)
        if v['conv'] <= 2: P['baseline_conv<=2'].append(base_pnl)
        (P['baseline_BEAR'] if bearish else P['baseline_BULL']).append(base_pnl)

        # ── STUDY C: structural stop, 15-min vs 30-min close beyond R1/S1 ──
        for tf_name, tf_by_day in (('15m', m15_by_day), ('30m', m30_by_day)):
            ibars = intraday_bars(tf_by_day, mon_str, exit_date)
            stop_val = None; stop_ts = None
            for b in ibars:
                if b['date'][11:16] == '09:15':   # don't stop on the entry candle itself
                    continue
                breached = (b['close'] > v['r1']) if bearish else (b['close'] < v['s1'])
                if breached:
                    sv = spread_val_intraday(b['date'], b['close'])
                    if sv is not None:
                        stop_val = sv; stop_ts = b['date']
                    break
            if stop_val is not None:
                P[f'stop_{tf_name}'].append(pnl(stop_val))
            else:
                P[f'stop_{tf_name}'].append(base_pnl)   # never stopped -> Friday exit
            if bearish and tf_name == '30m':
                stop_extend.append((mon_str, stop_ts is not None, base_pnl,
                                    pnl(stop_val) if stop_val is not None else base_pnl))

        # ── STUDY B: Friday exit-time grid ──
        fri = exit_date
        fbars = {b['date'][11:16]: b for b in m15_by_day.get(fri, [])}
        for t in ('09:45', '11:00', '13:00', '14:30', '15:15'):
            b = fbars.get(t)
            if b:
                sv = spread_val_intraday(b['date'], b['close'])
                P[f'fri_{t}'].append(pnl(sv) if sv is not None else base_pnl)
            else:
                P[f'fri_{t}'].append(base_pnl)
        # early profit-take: first EOD where spread >= 0.75*max -> take it
        taken = None
        for d in sessions:
            sv = spread_val_eod(d)
            if sv is not None and sv >= 0.75 * maxv:
                taken = pnl(sv); break
        P['pt_0.75'].append(taken if taken is not None else base_pnl)

        # ── STUDY A (corrected): morph to an ALL-PUT condor/butterfly by ADDING a
        #    BULL PUT SPREAD below the existing bear put spread, on a 30-min
        #    bullish stoch crossover. = iron-condor payoff via put-call parity. ──
        if bearish:
            m30w = intraday_bars(m30_by_day, mon_str, exit_date)
            st = stochastic(m30w)
            def trigger(require_pp):
                for i in range(1, len(m30w)):
                    b = m30w[i]; pb = m30w[i-1]
                    k1, d1 = st[b['date']]; k0, d0 = st[pb['date']]
                    if not ((k0 <= d0) and (k1 > d1)):
                        continue
                    if require_pp and b['close'] < v['pp']:
                        continue
                    if b['close'] > v['r1']:        # in stop region -> exit, don't morph
                        continue
                    return b['date'], b['close']
                return None, None
            def morph(require_pp, placement):
                trig_ts, trig_spot = trigger(require_pp)
                if trig_ts is None:
                    return None                      # no trigger -> baseline (hold)
                # placement of the ADDED bull put spread (short ks_hi / long ks_lo)
                if placement == 'fly':
                    ks_hi = short_k                  # share existing short -> butterfly @ M
                elif placement == 'condor':
                    ks_hi = long_k - 100             # band [M, L-100]
                else:                                # 'recenter' short at current price
                    ks_hi = round_to_strike(trig_spot, 100)
                ks_lo = ks_hi - WIDTH
                d = trig_ts[:10]; ed = day_by_date[d]['close']
                iv_hi = leg_iv_eod(d, expiry, ks_hi, 'PE', ed)
                iv_lo = leg_iv_eod(d, expiry, ks_lo, 'PE', ed)
                if not (iv_hi and iv_lo):
                    return None
                dte = dte_at(trig_ts, expiry)
                credit = (bs_leg(trig_spot, ks_hi, dte, iv_hi, 'PE')
                          - bs_leg(trig_spot, ks_lo, dte, iv_lo, 'PE'))   # received
                bh = opt_eod(exit_date, expiry, ks_hi, 'PE')
                bl = opt_eod(exit_date, expiry, ks_lo, 'PE')
                if bh is None or bl is None:
                    return None
                bp_exit_val = bh - bl                # cost to close the sold spread
                bullput_pnl = (credit - bp_exit_val) * LOT * LOTS
                return base_pnl + bullput_pnl        # original put spread + added bull put
            for name, req, place in (('morph_fly', False, 'fly'),
                                     ('morph_condor', False, 'condor'),
                                     ('morph_recenter', False, 'recenter'),
                                     ('morph_recenter_pp', True, 'recenter')):
                mp = morph(req, place)
                P[name].append(mp if mp is not None else base_pnl)

        rows.append({'week': mon_str, 'view': v['view'], 'conv': v['conv'],
                     'bucket': v['bucket'], 'side': 'BEAR' if bearish else 'BULL',
                     'adx': round(v['adx'], 1) if v['adx'] else None,
                     'spot': round(spot, 1), 'long_k': long_k, 'short_k': short_k,
                     'r1': round(v['r1'], 1), 's1': round(v['s1'], 1),
                     'debit': round(debit, 1), 'dte0': dte0,
                     'base_pnl': round(base_pnl)})

    # ── reports ──
    print("=== VIEW DISTRIBUTION (all weeks 2024-03..2026-03) ===")
    tot = sum(view_counts.values())
    for vname, n in view_counts.most_common():
        print(f"  {vname:22s} {n:3d}  {100*n/tot:4.1f}%")
    print(f"  total weeks: {tot}")

    print(f"\n=== POLICY COMPARISON (next-week expiry, {LOTS} lots, real-EOD pricing) ===")
    order = ['baseline_FriEOD', 'baseline_BEAR', 'baseline_BULL',
             'baseline_conv>=4', 'baseline_conv=3', 'baseline_conv<=2',
             'stop_15m', 'stop_30m',
             'fri_09:45', 'fri_11:00', 'fri_13:00', 'fri_14:30', 'fri_15:15', 'pt_0.75',
             'morph_fly', 'morph_condor', 'morph_recenter', 'morph_recenter_pp']
    for k in order:
        summarize_p(P.get(k, []), k)

    if stop_extend:
        fired = [x for x in stop_extend if x[1]]
        print(f"\n=== STUDY C detail (bearish, 30m stop) ===")
        print(f"  stop fired in {len(fired)}/{len(stop_extend)} bearish weeks")
        saved = sum(1 for _, f, bp, sp in stop_extend if f and sp > bp)
        print(f"  stop improved vs Friday-hold in {saved} of those")

    with open('/home/arun/quantifyd/research/nwv_p1_trades.csv', 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print("\nper-week CSV -> research/nwv_p1_trades.csv")


def summarize_p(pnls, label):
    if not pnls:
        print(f"  {label:20s}: (no trades)"); return
    n = len(pnls); wins = [p for p in pnls if p > 0]
    tot = sum(pnls); avg = tot / n; wr = 100 * len(wins) / n
    gw = sum(p for p in pnls if p > 0); gl = -sum(p for p in pnls if p < 0)
    pf = gw / gl if gl else float('inf')
    print(f"  {label:20s}: n={n:3d}  win%={wr:5.1f}  avg=Rs{avg:8.0f}  "
          f"tot=Rs{tot:10.0f}  PF={pf:5.2f}  worst={min(pnls):8.0f}")

def summarize(rs, label):
    if not rs:
        print(f"  {label:18s}: (no trades)"); return
    pnls = [r['pnl'] for r in rs]
    n = len(pnls); wins = [p for p in pnls if p > 0]
    tot = sum(pnls); avg = tot / n
    wr = 100 * len(wins) / n
    mx, mn = max(pnls), min(pnls)
    gross_w = sum(p for p in pnls if p > 0); gross_l = -sum(p for p in pnls if p < 0)
    pf = gross_w / gross_l if gross_l else float('inf')
    print(f"  {label:18s}: n={n:3d}  win%={wr:4.1f}  avg=Rs{avg:7.0f}  "
          f"tot=Rs{tot:9.0f}  PF={pf:4.2f}  best={mx:6.0f} worst={mn:7.0f}")

if __name__ == '__main__':
    run()
