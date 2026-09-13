# -*- coding: utf-8 -*-
"""research/172 - portfolio simulator for the 52-week channel book.

Conventions (all pre-registered in the STATUS doc before any cell ran):
  * a CLOSE-based signal on day t is filled at the OPEN of day t+1, for BOTH legs
  * 20 slots at 1/slots of NAV, Rs 1,00,00,000 book, NSE cash CNC, long only
  * contested slots resolved by 252-day relative strength, descending (tie: symbol name)
  * costs in bps per side, applied to both legs
  * Indian FY tax: 20% STCG / 12.5% LTCG over 365 days, loss netting settled 1 April,
    Rs 1.25 lakh LTCG exemption per FY, carried-forward losses
  * idle cash accrues 5.2% post-tax, daily
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0
START_CAPITAL = 1e7                 # Rs 1 crore
IDLE_YIELD = 0.052                  # project standard, post-tax
STCG, LTCG = 0.20, 0.125
LTCG_EXEMPT = 125000.0
LTCG_DAYS = 365


def entry_signal(P, uni, lookback, ref='close', buffer=0.0, ath_within=None,
                 new_ath=False):
    """(T,N) bool of FILL days: True on day i means buy at open[i] (signal fired i-1)."""
    piv = P.PIV_C[lookback] if ref == 'close' else P.PIV_H[lookback]
    elig = P.eligible(uni, lookback)
    with np.errstate(invalid='ignore'):
        sig = elig & np.isfinite(piv) & (P.C > piv * (1.0 + buffer))
        if ath_within is not None:
            sig &= (P.C >= P.ATHC * (1.0 - ath_within))
        if new_ath:
            sig &= (P.C >= P.ATHC)
    out = np.zeros_like(sig)
    out[1:] = sig[:-1]
    return out


def exit_array(P, key):
    """(T,N) bool - the close-based exit rule says get out today. None = path-dependent."""
    if key.startswith('CC'):
        w = int(key[2:])
        lo = P.LO_C[w]
        with np.errstate(invalid='ignore'):
            return np.isfinite(lo) & (P.C < lo)
    if key.startswith('CL'):
        w = int(key[2:])
        lo = P.LO_L[w]
        with np.errstate(invalid='ignore'):
            return np.isfinite(lo) & (P.C < lo)
    if key in P.ST:
        return P.ST[key] == -1
    if key == 'SMA15':
        with np.errstate(invalid='ignore'):
            return np.isfinite(P.SMA[15]) & (P.C < P.SMA[15])
    if key == 'SMA50':
        with np.errstate(invalid='ignore'):
            return np.isfinite(P.SMA[50]) & (P.C < P.SMA[50])
    if key == 'EMA50':
        with np.errstate(invalid='ignore'):
            return np.isfinite(P.EMA50) & (P.C < P.EMA50)
    if key.startswith('ATR') or key == 'NONE':
        return None
    raise ValueError(key)


def simulate(P, trig, cfg):
    """trig: (T,N) bool fill-day mask. cfg keys:
       exit, atr_mult, hard_stop, time_stop, slots, cost_bps, gate, idle_yield,
       seed (None -> RS-rank contention), days (index array), same_close_fill
    """
    days = cfg['days']
    i0, i1 = int(days[0]), int(days[-1])
    slots = cfg.get('slots', 20)
    slot_pct = cfg.get('slot_pct', 1.0 / slots)
    cost = cfg.get('cost_bps', 15.0) / 10000.0
    gate = cfg.get('gate')
    seed = cfg.get('seed')
    rng = np.random.default_rng(seed) if seed is not None else None
    atr_mult = cfg.get('atr_mult', 0.0)
    hard = cfg.get('hard_stop', 0.0)
    tstop = cfg.get('time_stop', 0)
    iy = cfg.get('idle_yield', IDLE_YIELD)
    same_close = cfg.get('same_close_fill', False)
    daily_yield = (1.0 + iy) ** (1.0 / TRADING_DAYS) - 1.0
    taxon = 1.0 if cfg.get('tax', True) else 0.0
    EX = cfg.get('exit_arr')

    C, O = P.C, P.O
    CM = P.CM
    ATR, RS = P.ATR, P.RS252
    n = i1 - i0 + 1
    nav = np.full(n, np.nan)
    inv = np.full(n, np.nan)
    cash = START_CAPITAL
    open_pos = []
    trades = []
    pending_exit = []
    fy_st = fy_lt = 0.0
    carry = 0.0
    cur_fy = None
    tax_paid = 0.0
    gross_cost = 0.0

    for k in range(n):
        i = i0 + k
        day = P.dstr[i]
        fy = int(day[:4]) - (1 if day[5:7] < '04' else 0)
        if cur_fy is None:
            cur_fy = fy
        elif fy != cur_fy:
            st, lt = fy_st, fy_lt
            pool = carry
            if st < 0:
                pool += st
                st = 0.0
            if lt < 0:
                pool += lt
                lt = 0.0
            if pool < 0 and st > 0:
                use = min(st, -pool)
                st -= use
                pool += use
            if pool < 0 and lt > 0:
                use = min(lt, -pool)
                lt -= use
                pool += use
            carry = min(pool, 0.0)
            lt_taxable = max(0.0, lt - LTCG_EXEMPT)
            bill = (st * STCG + lt_taxable * LTCG) * taxon
            cash -= bill
            tax_paid += bill
            fy_st = fy_lt = 0.0
            cur_fy = fy

        # ---- sells queued from yesterday's close signal, at today's open
        still = []
        for p in pending_exit:
            px = O[i, p['j']]
            if not np.isfinite(px) or px <= 0:
                still.append(p)
                continue
            proceeds = p['shares'] * px * (1 - cost)
            gross_cost += p['shares'] * px * cost
            cash += proceeds
            pnl = proceeds - p['basis']
            held_days = int(P.dnum[i] - P.dnum[p['ei']])
            if held_days > LTCG_DAYS:
                fy_lt += pnl
            else:
                fy_st += pnl
            trades.append(dict(symbol=P.cols[p['j']], entry_date=P.dstr[p['ei']],
                               exit_date=day, entry_px=p['epx'], exit_px=float(px),
                               shares=p['shares'], pnl=pnl,
                               ret_pct=100.0 * (proceeds / p['basis'] - 1.0),
                               bars=i - p['ei'], days=held_days,
                               mae_pct=100.0 * (p['trough'] / p['epx'] - 1.0),
                               mfe_pct=100.0 * (p['peak'] / p['epx'] - 1.0),
                               reason=p['reason']))
        pending_exit = still

        # ---- new entries at today's open
        cand = np.nonzero(trig[i])[0]
        if len(cand) and (gate is None or gate[i]):
            held = {p['j'] for p in open_pos} | {p['j'] for p in pending_exit}
            cand = np.array([j for j in cand if j not in held], dtype=int)
            free = slots - len(open_pos)
            if len(cand) and free > 0:
                if len(cand) > free:
                    if rng is not None:
                        cand = cand[np.sort(rng.choice(len(cand), size=free, replace=False))]
                    else:
                        rs = RS[i, cand]
                        rs = np.where(np.isfinite(rs), rs, -1e9)
                        order = np.lexsort((np.array([P.cols[j] for j in cand]), -rs))
                        cand = cand[order[:free]]
                mv = sum(p['shares'] * _px(CM, i, p['j']) for p in open_pos)
                navnow = cash + mv
                for j in cand:
                    px = C[i, j] if same_close else O[i, j]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    alloc = navnow * slot_pct
                    shares = int(alloc // (px * (1 + cost)))
                    if shares <= 0:
                        continue
                    basis = shares * px * (1 + cost)
                    if basis > cash:
                        continue
                    gross_cost += shares * px * cost
                    cash -= basis
                    open_pos.append(dict(j=int(j), shares=shares, ei=i, epx=float(px),
                                         basis=basis, peak=float(px), trough=float(px),
                                         reason=''))

        # ---- mark, then evaluate close-based exits
        mv = 0.0
        keep = []
        for p in open_pos:
            c = _px(CM, i, p['j'])
            mv += p['shares'] * c
            if c > p['peak']:
                p['peak'] = c
            if c < p['trough']:
                p['trough'] = c
            out = None
            if hard and c <= p['epx'] * (1.0 - hard):
                out = 'HARD'
            elif tstop and (i - p['ei']) >= tstop:
                out = 'TIME'
            elif atr_mult:
                a = ATR[i, p['j']]
                if np.isfinite(a) and c <= p['peak'] - atr_mult * a:
                    out = 'ATRTRAIL'
            if out is None and EX is not None and EX[i, p['j']]:
                out = 'RULE'
            if out and i + 1 <= i1:
                p['reason'] = out
                pending_exit.append(p)
            else:
                keep.append(p)
        open_pos = keep
        cash *= (1.0 + daily_yield)
        nav[k] = cash + mv
        inv[k] = (mv / nav[k]) if nav[k] > 0 else 0.0

    # ---- liquidate at the final close (and pay the residual tax)
    if open_pos or pending_exit:
        i = i1
        for p in open_pos + pending_exit:
            px = _px(CM, i, p['j'])
            proceeds = p['shares'] * px * (1 - cost)
            cash += proceeds
            pnl = proceeds - p['basis']
            held_days = int(P.dnum[i] - P.dnum[p['ei']])
            if held_days > LTCG_DAYS:
                fy_lt += pnl
            else:
                fy_st += pnl
            trades.append(dict(symbol=P.cols[p['j']], entry_date=P.dstr[p['ei']],
                               exit_date=P.dstr[i], entry_px=p['epx'], exit_px=float(px),
                               shares=p['shares'], pnl=pnl,
                               ret_pct=100.0 * (proceeds / p['basis'] - 1.0),
                               bars=i - p['ei'], days=held_days,
                               mae_pct=100.0 * (p['trough'] / p['epx'] - 1.0),
                               mfe_pct=100.0 * (p['peak'] / p['epx'] - 1.0),
                               reason='EOD'))
        nav[-1] = cash

    st, lt = fy_st, fy_lt
    pool = carry
    if st < 0:
        pool += st
        st = 0.0
    if lt < 0:
        pool += lt
        lt = 0.0
    if pool < 0 and st > 0:
        use = min(st, -pool)
        st -= use
        pool += use
    if pool < 0 and lt > 0:
        use = min(lt, -pool)
        lt -= use
        pool += use
    bill = (st * STCG + max(0.0, lt - LTCG_EXEMPT) * LTCG) * taxon
    nav[-1] -= bill
    tax_paid += bill
    return dict(nav=nav, dates=P.dstr[i0:i1 + 1], trades=trades,
                tax_paid=tax_paid, cost_paid=gross_cost, invested=inv)


def _px(arr, i, j):
    v = arr[i, j]
    return float(v) if np.isfinite(v) else 0.0


# ----------------------------------------------------------------- metrics
def metrics(nav, dates, trades=None, peak_ref=None):
    s = pd.Series(nav, index=pd.to_datetime(dates)).dropna()
    if len(s) < 30 or s.iloc[0] <= 0:
        return {}
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    ref = s.cummax() if peak_ref is None else peak_ref.reindex(s.index).ffill()
    dd = s / ref - 1.0
    mdd = float(dd.min())
    r = s.pct_change().dropna()
    sharpe = (r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan
    out = dict(cagr=round(100 * cagr, 2), maxdd=round(100 * mdd, 2),
               calmar=round(cagr / abs(mdd), 3) if mdd < 0 else np.nan,
               sharpe=round(float(sharpe), 3), final=round(float(s.iloc[-1]), 0),
               years=round(yrs, 2))
    if trades:
        t = pd.DataFrame(trades)
        wins, loss = t[t.pnl > 0], t[t.pnl <= 0]
        streak = mx = 0
        for p in t.sort_values('exit_date')['pnl']:
            streak = streak + 1 if p <= 0 else 0
            mx = max(mx, streak)
        out.update(trades=len(t),
                   win_rate=round(100 * len(wins) / len(t), 1),
                   avg_win=round(float(wins.ret_pct.mean()), 2) if len(wins) else np.nan,
                   avg_loss=round(float(loss.ret_pct.mean()), 2) if len(loss) else np.nan,
                   expectancy=round(float(t.ret_pct.mean()), 3),
                   med_hold_d=int(t.days.median()),
                   worst_mae=round(float(t.mae_pct.min()), 1),
                   med_mae=round(float(t.mae_pct.median()), 1),
                   max_loss_streak=mx,
                   trades_per_yr=round(len(t) / yrs, 1))
    return out


def bh_equal_weight(P, uni, days, cost_bps=15.0, rebal='M'):
    """Drift control: equal-weight buy-and-hold of the SAME eligible universe."""
    i0, i1 = int(days[0]), int(days[-1])
    C = P.CM
    mask = P.universe_mask(uni)
    with np.errstate(invalid='ignore'):
        elig = mask & (P.TVp >= 5e7) & (P.BARS >= 60) & ~P.FUND[None, :]
    cost = cost_bps / 10000.0
    n = i1 - i0 + 1
    nav = np.full(n, np.nan)
    val = START_CAPITAL
    w = None
    cols_prev = None
    per = pd.PeriodIndex(pd.to_datetime(P.dstr[i0:i1 + 1]), freq='M')
    newmon = np.r_[True, per[1:] != per[:-1]]
    for k in range(n):
        i = i0 + k
        c = C[i]
        if w is not None and cols_prev is not None:
            prev = C[i - 1, cols_prev]
            cur = C[i, cols_prev]
            r = np.where(np.isfinite(prev) & np.isfinite(cur) & (prev > 0),
                         cur / prev - 1.0, 0.0)
            val *= (1.0 + float(np.dot(w, r)))
        if newmon[k]:
            j = np.nonzero(elig[i] & np.isfinite(c) & (c > 0))[0]
            if len(j):
                if cols_prev is not None:
                    turn = len(set(j.tolist()) ^ set(cols_prev.tolist())) / max(len(j), 1)
                    val *= (1.0 - cost * min(turn, 1.0))
                cols_prev = j
                w = np.full(len(j), 1.0 / len(j))
        nav[k] = val
    return nav
