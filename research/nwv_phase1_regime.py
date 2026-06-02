#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NWV Phase-1 — REGIME-ROBUSTNESS extension (2020 -> 2026)
========================================================
Options data only exists 2024-03+. To test the morph across the COVID crash,
2022 drawdown, etc., we price with Black-Scholes using a realized-vol IV proxy
(trailing 20d), with the vol LEVEL calibrated to 2024+ real-option IVs so the
absolute level is anchored. Daily bars are derived from the 30-min series
(available from 2020). Consistent pricing across the whole window => up-vs-down
regime comparison is not confounded by a pricing-method change.

Reports, by year/regime:
  - baseline (hold to Fri), 30m R1/S1 stop, morph (add opposite credit spread)
  - the key diagnostic: how often the morph CAPS a would-be winner vs SAVES a
    loser, split by up/down regime.
Validation: model vs real entry debit on 2024+ (mean abs error).
"""
import sqlite3, sys, math, statistics
from datetime import datetime, date, timedelta
from collections import defaultdict, Counter

sys.path.insert(0, '/home/arun/quantifyd')
from services.nwv_engine import (
    compute_cpr, compute_pivots, classify_cpr_width, classify_first_candle,
    classify_gap, base_view_matrix, apply_gap_dampener, apply_monthly_override,
    score_conviction,
    VIEW_BEARISH, VIEW_NTB, VIEW_BULLISH, VIEW_NTBULL,
)
from services.spread_structure import black_scholes_put, black_scholes_call, round_to_strike

DB = '/home/arun/quantifyd/backtest_data/market_data.db'
SPOT = 'NIFTY50'
LOT, LOTS, WIDTH, R = 65, 5, 200, 0.06

con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
cur = con.cursor()

def bars(tf):
    rows = cur.execute("SELECT date,open,high,low,close FROM market_data_unified "
                       "WHERE symbol=? AND timeframe=? ORDER BY date", (SPOT, tf)).fetchall()
    return [{'date': r['date'], 'open': r['open'], 'high': r['high'],
             'low': r['low'], 'close': r['close']} for r in rows]

M30 = bars('30minute')
m30_by_day = defaultdict(list)
for b in M30:
    m30_by_day[b['date'][:10]].append(b)

# derive DAILY OHLC from 30-min (covers 2020+)
DAY = []
for d in sorted(m30_by_day):
    bs = sorted(m30_by_day[d], key=lambda x: x['date'])
    DAY.append({'date': d, 'open': bs[0]['open'], 'high': max(x['high'] for x in bs),
                'low': min(x['low'] for x in bs), 'close': bs[-1]['close']})
day_by_date = {b['date']: b for b in DAY}
DAY_DATES = [b['date'] for b in DAY]
daily_close = {b['date']: b['close'] for b in DAY}

# ── realized vol (annualised), trailing window of daily closes ──
def realized_vol(d_str, window=20):
    closes = [b['close'] for b in DAY if b['date'] < d_str][-(window + 1):]
    if len(closes) < window + 1:
        return None
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    sd = statistics.pstdev(rets)
    return sd * math.sqrt(252)

# ── options (2024+) just for vol-level calibration + validation ──
def opt_eod(td, exp, k, ot):
    r = cur.execute("SELECT close,settle_price FROM nse_options_bhav WHERE symbol='NIFTY' "
                    "AND trade_date=? AND expiry_date=? AND strike=? AND option_type=?",
                    (td, exp, float(k), ot)).fetchone()
    if not r:
        return None
    return r['close'] if r['close'] and r['close'] > 0 else r['settle_price']

def expiries_after(d_str):
    rows = cur.execute("SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' "
                       "AND expiry_date>=? ORDER BY expiry_date", (d_str,)).fetchall()
    return [r['expiry_date'] for r in rows]

def implied_vol(price, spot, strike, dte, ot):
    if not price or price <= 0 or dte <= 0:
        return None
    f = black_scholes_put if ot == 'PE' else black_scholes_call
    lo, hi = 0.01, 3.0
    if (f(spot, strike, dte, lo, R) - price) * (f(spot, strike, dte, hi, R) - price) > 0:
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if (f(spot, strike, dte, lo, R) - price) * (f(spot, strike, dte, mid, R) - price) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

# ── view replay (uses derived daily) ──
def first_30m(d_str):
    bs = m30_by_day.get(d_str)
    if not bs:
        return None
    f = min(bs, key=lambda b: b['date'])
    return f if f['date'][11:16] == '09:15' else None

def seg_hlc(d0, d1):
    seg = [b for b in DAY if d0 <= b['date'] <= d1]
    if not seg:
        return None
    return max(b['high'] for b in seg), min(b['low'] for b in seg), seg[-1]['close']

def adx_at(d_str, period=14):
    hist = [b for b in DAY if b['date'] < d_str][-60:]
    if len(hist) < period + 2:
        return None
    trs, pdms, ndms = [], [], []
    for i in range(1, len(hist)):
        h, l, pc = hist[i]['high'], hist[i]['low'], hist[i-1]['close']
        ph, pl = hist[i-1]['high'], hist[i-1]['low']
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        up, dn = h - ph, pl - l
        pdms.append(up if (up > dn and up > 0) else 0.0)
        ndms.append(dn if (dn > up and dn > 0) else 0.0)
    def w(x):
        s = sum(x[:period]); o = [s]
        for v in x[period:]:
            s = s - s / period + v; o.append(s)
        return o
    atr, pdi, ndi = w(trs), w(pdms), w(ndms)
    dxs = []
    for a, p, n in zip(atr, pdi, ndi):
        if a == 0:
            continue
        pv, nv = 100*p/a, 100*n/a
        if pv + nv:
            dxs.append(100*abs(pv-nv)/(pv+nv))
    if len(dxs) < period:
        return None
    adx = sum(dxs[:period]) / period
    for v in dxs[period:]:
        adx = (adx*(period-1)+v)/period
    return adx

def mondays(s, e):
    d = s
    while d.weekday():
        d += timedelta(days=1)
    while d <= e:
        yield d
        d += timedelta(days=7)

def build_view(mon):
    ms = mon.isoformat()
    fc = first_30m(ms)
    if not fc:
        return None
    pw = seg_hlc((mon - timedelta(days=7)).isoformat(), (mon - timedelta(days=3)).isoformat())
    if not pw:
        return None
    ph, pl, pc = pw
    cpr = compute_cpr(ph, pl, pc); piv = compute_pivots(ph, pl, pc)
    spot = fc['close']
    bucket = classify_cpr_width(abs(cpr['tc'] - cpr['bc']) / spot * 100)
    fcc = classify_first_candle(fc['open'], fc['high'], fc['low'], fc['close'], cpr['tc'], cpr['bc'])
    prev_fri = [b for b in DAY if b['date'] < ms][-1]['close'] if [b for b in DAY if b['date'] < ms] else pc
    gap = (fc['open'] - prev_fri) / prev_fri * 100 if prev_fri else 0
    gt, gd = classify_gap(gap)
    base = base_view_matrix(bucket, fcc['pos'], fcc['body'])
    vg, _ = apply_gap_dampener(base, gt, gd)
    fm = mon.replace(day=1); lp = fm - timedelta(days=1); fp = lp.replace(day=1)
    mh = seg_hlc(fp.isoformat(), lp.isoformat())
    mtc = mbc = None
    if mh:
        mc = compute_cpr(*mh); mtc, mbc = mc['tc'], mc['bc']
    view, _ = apply_monthly_override(vg, spot, mtc, mbc)
    conv = score_conviction(fcc, adx_at(ms), None)
    return {'mon': mon, 'ms': ms, 'spot': spot, 'view': view, 'conv': conv,
            'r1': piv['r1'], 's1': piv['s1'], 'pp': piv['pp']}

def stochastic(b30, k=14, d=3):
    res = {}; ks = []
    for i in range(len(b30)):
        win = b30[max(0, i-k+1):i+1]
        hh = max(x['high'] for x in win); ll = min(x['low'] for x in win)
        kv = 100*(b30[i]['close']-ll)/(hh-ll) if hh > ll else 50.0
        ks.append(kv)
        res[b30[i]['date']] = (kv, sum(ks[max(0, i-d+1):i+1])/min(i+1, d))
    return res

def week_sessions(ms):
    fri = (datetime.strptime(ms, '%Y-%m-%d').date() + timedelta(days=4)).isoformat()
    return [d for d in DAY_DATES if ms <= d <= fri]

def intraday(d0, d1):
    out = []
    for d in sorted(m30_by_day):
        if d0 <= d <= d1:
            out += m30_by_day[d]
    return out

# ── vol multiplier: median(real ATM IV / realized vol) over 2024+ ──
def calibrate_multiplier():
    ratios = []
    for mon in mondays(date(2024, 3, 4), date(2026, 3, 16)):
        v = build_view(mon)
        if not v:
            continue
        rv = realized_vol(v['ms'])
        if not rv:
            continue
        exps = expiries_after(v['ms'])
        if len(exps) < 2:
            continue
        exp = exps[1]
        dte = (datetime.strptime(exp, '%Y-%m-%d').date() - mon).days
        atm = round_to_strike(v['spot'], 100)
        px = opt_eod(v['ms'], exp, atm, 'CE') or opt_eod(v['ms'], exp, atm, 'PE')
        ot = 'CE' if opt_eod(v['ms'], exp, atm, 'CE') else 'PE'
        iv = implied_vol(px, v['spot'], atm, dte, ot)
        if iv and rv:
            ratios.append(iv / rv)
    return statistics.median(ratios) if ratios else 1.3, len(ratios)

MULT, nmult = calibrate_multiplier()

def price(spot, k, dte, iv, ot):
    f = black_scholes_put if ot == 'PE' else black_scholes_call
    return f(spot, k, max(dte, 1/365), iv, R)

def dte_at(ts, exp_d):
    d = datetime.strptime(ts[:10], '%Y-%m-%d').date()
    base = (exp_d - d).days
    hh, mm = int(ts[11:13]), int(ts[14:16])
    rem = max(0.0, min(1.0, (375 - ((hh*60+mm) - 555)) / 375.0))
    return max(base + rem, 1/365)

def run():
    start, end = date(2020, 2, 3), date(2026, 3, 23)
    P = defaultdict(list)
    by_year = defaultdict(lambda: defaultdict(list))
    diag = []          # morph diagnostics
    val_err = []
    vcount = Counter()
    for mon in mondays(start, end):
        v = build_view(mon)
        if not v:
            continue
        vcount[v['view']] += 1
        if v['view'] not in (VIEW_BEARISH, VIEW_NTB, VIEW_BULLISH, VIEW_NTBULL):
            continue
        bearish = v['view'] in (VIEW_BEARISH, VIEW_NTB)
        ms, spot = v['ms'], v['spot']
        rv = realized_vol(ms)
        if not rv:
            continue
        iv = MULT * rv
        sess = week_sessions(ms)
        if len(sess) < 2:
            continue
        exit_d = sess[-1]
        # synthetic next-week Tuesday expiry (entry Monday + ~8 days)
        exp_d = mon + timedelta(days=8)
        dte0 = (exp_d - mon).days
        ot = 'PE' if bearish else 'CE'
        Lk = round_to_strike(spot, 100)
        Mk = Lk - WIDTH if bearish else Lk + WIDTH
        debit = price(spot, Lk, dte0, iv, ot) - price(spot, Mk, dte0, iv, ot)
        if debit <= 0:
            continue
        year = ms[:4]

        # validation vs real options (2024+)
        if ms >= '2024-03-01':
            exps = expiries_after(ms)
            if len(exps) >= 2:
                rl = opt_eod(ms, exps[1], Lk, ot); rs = opt_eod(ms, exps[1], Mk, ot)
                if rl and rs:
                    val_err.append(abs(debit - (rl - rs)))

        def val(spot_now, dte, legs):
            # legs: list of (qty, strike, otype); qty +long/-short
            return sum(q * price(spot_now, k, dte, iv, o) for q, k, o in legs)

        exit_spot = daily_close[exit_d]
        dte_x = (exp_d - datetime.strptime(exit_d, '%Y-%m-%d').date()).days
        base_legs = [(1, Lk, ot), (-1, Mk, ot)]
        base_exit = val(exit_spot, dte_x, base_legs)
        base_pnl = (base_exit - debit) * LOT * LOTS
        P['baseline'].append(base_pnl); by_year[year]['baseline'].append(base_pnl)
        (P['base_BEAR'] if bearish else P['base_BULL']).append(base_pnl)

        # 30m structural stop
        stop_pnl = base_pnl
        for b in intraday(ms, exit_d):
            if b['date'][11:16] == '09:15':
                continue
            if (b['close'] > v['r1']) if bearish else (b['close'] < v['s1']):
                d_x = dte_at(b['date'], exp_d)
                stop_pnl = (val(b['close'], d_x, base_legs) - debit) * LOT * LOTS
                break
        P['stop_30m'].append(stop_pnl); by_year[year]['stop_30m'].append(stop_pnl)

        # morph: add opposite credit spread (condor placement near existing short)
        b30 = intraday(ms, exit_d); st = stochastic(b30)
        morph_pnl = base_pnl; triggered = False
        for i in range(1, len(b30)):
            b, pb = b30[i], b30[i-1]
            k1, d1 = st[b['date']]; k0, d0 = st[pb['date']]
            if bearish:
                co = (k0 <= d0) and (k1 > d1)            # bullish CO vs bear trade
                guard = b['close'] < v['pp'] or b['close'] > v['r1']
            else:
                co = (k0 >= d0) and (k1 < d1)            # bearish CO vs bull trade
                guard = b['close'] > v['pp'] or b['close'] < v['s1']
            if not co or guard:
                continue
            triggered = True
            # add credit spread: bear->bull put spread below; bull->bear call spread above
            if bearish:
                ks_hi = Lk - 100; ks_lo = ks_hi - WIDTH; cot = 'PE'
                add_legs = [(-1, ks_hi, cot), (1, ks_lo, cot)]   # short hi / long lo (credit)
            else:
                ks_lo = Lk + 100; ks_hi = ks_lo + WIDTH; cot = 'CE'
                add_legs = [(-1, ks_lo, cot), (1, ks_hi, cot)]   # short lo / long hi (credit)
            d_t = dte_at(b['date'], exp_d)
            credit = -val(b['close'], d_t, add_legs)   # cash received (short legs dominate)
            add_exit = val(exit_spot, dte_x, add_legs)
            morph_pnl = base_pnl + (credit + add_exit) * LOT * LOTS
            break
        P['morph'].append(morph_pnl); by_year[year]['morph'].append(morph_pnl)

        if triggered:
            diag.append({'year': year, 'bearish': bearish, 'base': base_pnl,
                         'morph': morph_pnl, 'fwd_ret': (exit_spot - spot) / spot})

    print(f"vol multiplier (median real-IV/RV over 2024+, n={nmult}): {MULT:.2f}")
    if val_err:
        print(f"validation: model vs real entry debit, mean abs err = "
              f"{sum(val_err)/len(val_err):.1f} pts (n={len(val_err)})")
    print(f"\n=== view distribution 2020-2026 (n weeks) ===")
    tot = sum(vcount.values())
    for vv, n in vcount.most_common():
        print(f"  {vv:20s} {n:3d}  {100*n/tot:4.1f}%")

    print(f"\n=== POLICY (full 2020-2026, modeled BS, {LOTS} lots) ===")
    for k in ['baseline', 'base_BEAR', 'base_BULL', 'stop_30m', 'morph']:
        summ(P[k], k)

    print(f"\n=== BY YEAR (avg/wk, n) ===")
    print(f"  {'year':6s} {'n':>3s}  {'baseline':>10s} {'stop_30m':>10s} {'morph':>10s}")
    for y in sorted(by_year):
        b = by_year[y]['baseline']
        print(f"  {y:6s} {len(b):>3d}  {avg(b):>10s} {avg(by_year[y]['stop_30m']):>10s} "
              f"{avg(by_year[y]['morph']):>10s}")

    print(f"\n=== MORPH DIAGNOSTIC (when it triggered) ===")
    trig = diag
    up = [d for d in trig if d['fwd_ret'] > 0]; dn = [d for d in trig if d['fwd_ret'] <= 0]
    capped = [d for d in trig if d['base'] > 0 and d['morph'] < d['base']]
    saved = [d for d in trig if d['base'] < 0 and d['morph'] > d['base']]
    print(f"  triggered: {len(trig)} weeks  (price up after entry: {len(up)}, down: {len(dn)})")
    print(f"  CAPPED a would-be winner: {len(capped)} weeks, "
          f"give-up sum=Rs{sum(d['base']-d['morph'] for d in capped):,.0f}")
    print(f"  SAVED a loser:           {len(saved)} weeks, "
          f"rescue sum=Rs{sum(d['morph']-d['base'] for d in saved):,.0f}")
    bears = [d for d in trig if d['bearish']]; bulls = [d for d in trig if not d['bearish']]
    for lbl, grp in (('bear morphs', bears), ('bull morphs', bulls)):
        if grp:
            net = sum(d['morph']-d['base'] for d in grp)
            print(f"  {lbl}: n={len(grp)}  net morph-vs-base = Rs{net:,.0f}")

def avg(x):
    return f"Rs{sum(x)/len(x):,.0f}" if x else "-"

def summ(p, label):
    if not p:
        print(f"  {label:12s}: (none)"); return
    n = len(p); w = [x for x in p if x > 0]
    gw = sum(x for x in p if x > 0); gl = -sum(x for x in p if x < 0)
    print(f"  {label:12s}: n={n:3d}  win%={100*len(w)/n:5.1f}  avg=Rs{sum(p)/n:8.0f}  "
          f"tot=Rs{sum(p):10.0f}  PF={gw/gl if gl else 999:5.2f}  worst={min(p):8.0f}")

if __name__ == '__main__':
    run()
