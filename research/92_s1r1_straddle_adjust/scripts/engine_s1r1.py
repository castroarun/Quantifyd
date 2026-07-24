# -*- coding: utf-8 -*-
"""S1/R1-break engines on the recorded NIFTY chain — two strategy families:
  STRADDLE-ADJUST: short ATM straddle 09:16; on first 5-min close beyond R1/S1, cut the losing leg
                   and (optionally) trail the survivor with ST(7,3) on price or on the option premium.
  BREAKOUT-FLIP:   no 09:16 entry; on an R1 close go short PE (bullish), on an S1 close go short CE
                   (bearish); flip on the opposite break; ST(7,3) trail (price or premium) otherwise.
Reuses research/90 engine helpers (load_day, supertrend_up)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from datetime import time as dtime
from engine_mtm import load_day, supertrend_up, oc, days, LOT, LOTS, BROK_PER_LEG  # noqa

QTY = LOT * LOTS          # 130 (NIFTY lot 65 x 2)
EXIT = dtime(15, 15)
ENTRY = dtime(9, 16)


def daily_ohlc():
    out = {}
    dl = [r[0] for r in oc.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]
    for d in dl:
        hl = oc.execute("SELECT MIN(underlying_spot),MAX(underlying_spot) FROM option_chain "
                        "WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND underlying_spot IS NOT NULL",
                        (d,)).fetchone()
        o = oc.execute("SELECT underlying_spot FROM option_chain WHERE symbol='NIFTY' AND "
                       "substr(snapshot_time,1,10)=? AND underlying_spot IS NOT NULL ORDER BY snapshot_time LIMIT 1",
                       (d,)).fetchone()
        c = oc.execute("SELECT underlying_spot FROM option_chain WHERE symbol='NIFTY' AND "
                       "substr(snapshot_time,1,10)=? AND underlying_spot IS NOT NULL ORDER BY snapshot_time DESC LIMIT 1",
                       (d,)).fetchone()
        if hl and hl[0] and o and c:
            out[d] = dict(high=hl[1], low=hl[0], open=o[0], close=c[0])
    return out


def pivots(prev):
    H, L, C = prev['high'], prev['low'], prev['close']
    P = (H + L + C) / 3.0
    return 2 * P - L, 2 * P - H          # R1, S1


def _helpers(bundle):
    chain = bundle['chain']

    def prem(ts, t):
        if ts not in chain:
            return None
        ta, la, _, _ = chain[ts]
        idx = np.searchsorted(ta, np.datetime64(t), side='right') - 1
        if idx < 0:
            return None
        v = la[idx]
        return float(v) if v and v > 0 else None

    def tsym(strike, typ):
        for ts, (_, _, st, ty) in chain.items():
            if int(st) == int(strike) and ty == typ:
                return ts
        return None

    return prem, tsym


def _s5(spot_s):
    s5 = spot_s.resample('5min').agg(['first', 'max', 'min', 'last']).dropna()
    s5.columns = ['open', 'high', 'low', 'close']
    return s5


def _fin(legs, day, bd):
    out, tot = [], 0.0
    for g in ('CE', 'PE'):
        d = legs[g]
        pnl = (d['e'] - d['x']) * QTY - BROK_PER_LEG
        tot += pnl
        out.append(dict(leg=g, e=d['e'], x=d['x'], r=d['r'], pnl=round(pnl)))
    return dict(day=day, pnl=round(tot), bdir=bd, legs=out)


def sim_straddle(bundle, r1, s1, mode='hold', stp=7, stm=3.0):
    """mode: hold | cut | price | premium."""
    prem, tsym = _helpers(bundle)
    spot_s = bundle['spot_s']; times = bundle['times']; day = bundle['day']; chain = bundle['chain']
    ent = [t for t in times if t.time() >= ENTRY]
    if not ent:
        return None
    t0 = ent[0]; spot0 = float(spot_s.loc[t0]); atm = round(spot0 / 50) * 50
    ce, pe = tsym(atm, 'CE'), tsym(atm, 'PE')
    if not ce or not pe:
        return None
    ce_e, pe_e = prem(ce, t0), prem(pe, t0)
    if ce_e is None or pe_e is None:
        return None
    lastt = [t for t in times if t.time() <= EXIT][-1]
    legs = {'CE': dict(ts=ce, e=ce_e, x=None, r=None), 'PE': dict(ts=pe, e=pe_e, x=None, r=None)}

    if mode == 'hold':
        for g in ('CE', 'PE'):
            legs[g]['x'] = prem(legs[g]['ts'], lastt) or legs[g]['e']; legs[g]['r'] = 'eod'
        return _fin(legs, day, None)

    s5 = _s5(spot_s); bt = bd = None
    for idx, row in s5.iterrows():
        ct = idx + pd.Timedelta('5min')
        if ct <= t0:
            continue
        if ct.time() > EXIT:
            break
        if row['close'] > r1:
            bt, bd = ct, 'up'; break
        if row['close'] < s1:
            bt, bd = ct, 'down'; break
    if bd is None:
        for g in ('CE', 'PE'):
            legs[g]['x'] = prem(legs[g]['ts'], lastt) or legs[g]['e']; legs[g]['r'] = 'eod_nobreak'
        return _fin(legs, day, None)

    loser, surv = ('CE', 'PE') if bd == 'up' else ('PE', 'CE')
    legs[loser]['x'] = prem(legs[loser]['ts'], bt) or legs[loser]['e']; legs[loser]['r'] = 'cut'
    if mode == 'cut':
        legs[surv]['x'] = prem(legs[surv]['ts'], lastt) or legs[surv]['e']; legs[surv]['r'] = 'eod_surv'
        return _fin(legs, day, bd)

    # trail survivor (price or premium ST)
    if mode == 'price':
        st = supertrend_up(spot_s, period=stp, mult=stm, tf='5min')
        want = (lambda up: not up) if surv == 'PE' else (lambda up: up)
    else:
        ta, la, _, _ = chain[legs[surv]['ts']]
        ps = pd.Series(la, index=pd.DatetimeIndex(ta)).sort_index()
        st = supertrend_up(ps, period=stp, mult=stm, tf='5min')
        want = lambda up: up
    xt = lastt
    for b in st.index:
        ct = b + pd.Timedelta('5min')
        if ct <= bt:
            continue
        if ct.time() > EXIT:
            break
        if want(bool(st.loc[b])):
            xt = ct; break
    legs[surv]['x'] = prem(legs[surv]['ts'], xt) or legs[surv]['e']
    legs[surv]['r'] = 'st_exit' if xt.time() < EXIT else 'eod_trail'
    return _fin(legs, day, bd)


def sim_breakout(bundle, r1, s1, mode='price', stp=7, stm=3.0):
    """No 09:16 entry. R1 close -> short PE (bullish); S1 close -> short CE (bearish); flip on the
    opposite break; ST(7,3) trail (price/premium) otherwise. Returns summed P&L over the day."""
    prem, tsym = _helpers(bundle)
    spot_s = bundle['spot_s']; chain = bundle['chain']; day = bundle['day']
    lastt = [t for t in bundle['times'] if t.time() <= EXIT][-1]
    s5 = _s5(spot_s)
    price_st = supertrend_up(spot_s, period=stp, mult=stm, tf='5min') if mode == 'price' else None
    prem_st_cache = {}
    realized = []
    pos = None

    def open_pos(side, t):
        sp = float(spot_s.loc[t]) if t in spot_s.index else None
        if sp is None:
            return None
        ts = tsym(round(sp / 50) * 50, side)
        if not ts:
            return None
        e = prem(ts, t)
        return dict(side=side, ts=ts, e=e) if e is not None else None

    def close_pos(p, t, reason):
        x = prem(p['ts'], t) or p['e']
        realized.append(dict(side=p['side'], e=p['e'], x=x, r=reason, pnl=round((p['e'] - x) * QTY - BROK_PER_LEG)))

    def prem_up(ts, ct):
        st = prem_st_cache.get(ts)
        if st is None:
            ta, la, _, _ = chain[ts]
            st = supertrend_up(pd.Series(la, index=pd.DatetimeIndex(ta)).sort_index(), period=stp, mult=stm, tf='5min')
            prem_st_cache[ts] = st
        sub = st[st.index <= ct]
        return bool(sub.iloc[-1]) if len(sub) else False

    for idx, row in s5.iterrows():
        ct = idx + pd.Timedelta('5min')
        if ct.time() > EXIT:
            break
        sig = 'PE' if row['close'] > r1 else ('CE' if row['close'] < s1 else None)
        if sig:
            if pos is None:
                pos = open_pos(sig, ct)
            elif pos['side'] != sig:
                close_pos(pos, ct, 'flip'); pos = open_pos(sig, ct)
            continue
        if pos is not None:
            if mode == 'price':
                up = bool(price_st.loc[idx]) if idx in price_st.index else None
                exit_now = (up is not None) and ((not up) if pos['side'] == 'PE' else up)
            else:
                exit_now = prem_up(pos['ts'], ct)
            if exit_now:
                close_pos(pos, ct, 'st_exit'); pos = None
    if pos is not None:
        close_pos(pos, lastt, 'eod')
    return dict(day=day, pnl=round(sum(r['pnl'] for r in realized)), n_legs=len(realized), legs=realized)
