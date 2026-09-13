# -*- coding: utf-8 -*-
"""research/172 Phase 2 - a general stop / trail EXIT STACK on the same book.

bt172.py is left untouched so every Phase 1 number stays bit-reproducible. This is a fork
of its simulator with one change: the exit is a STACK of levels rather than a single rule.

Every level is CLOSE-based and filled at the NEXT open, exactly as in Phase 1.

Stack components (all optional, all combinable):
  hard_pct      initial stop at entry_px * (1 - x)                         fixed
  hard_atr      initial stop at entry_px - k * ATR(14) measured AT ENTRY   fixed
  be_after      once the close is >= entry*(1+x), the floor becomes entry  sticky
  lock_after /  once the close is >= entry*(1+lock_after), the percentage
  lock_trail    trail tightens to lock_trail                               sticky
  trail_pct     trail at (highest close since entry) * (1 - x)
  trail_chand   chandelier: (highest high since entry) - k * ATR(22) today
  trail_atr     (highest close since entry) - k * ATR(14) today
  rule          a precomputed close-based boolean array (ST, EMA, Donchian ...)
  time_bars /   exit after time_bars if the gain is below time_min_gain
  time_min_gain
  block_bars    after a STOP-OUT (not a rule exit), the symbol cannot be re-entered
                for this many bars. 0 = re-entry allowed, the Phase 1 behaviour.
  book_dd_kill  BOOK-level trailing drawdown kill: liquidate everything and stay in
                cash until NIFTYBEES closes back above its 50-day SMA.

THE STOP LEVEL RATCHETS. It can never fall. Without the ratchet an expanding ATR would
loosen a chandelier stop after the fact, which no broker order does.
"""
from __future__ import annotations

import numpy as np

from bt172 import (TRADING_DAYS, START_CAPITAL, IDLE_YIELD, STCG, LTCG, LTCG_EXEMPT,
                   LTCG_DAYS, metrics, entry_signal, exit_array)   # noqa: F401

NEG = -1e18


def simulate_stack(P, trig, cfg):
    days = cfg['days']
    i0, i1 = int(days[0]), int(days[-1])
    slots = cfg.get('slots', 20)
    slot_pct = cfg.get('slot_pct', 1.0 / slots)
    cost = cfg.get('cost_bps', 15.0) / 10000.0
    gate = cfg.get('gate')
    seed = cfg.get('seed')
    rng = np.random.default_rng(seed) if seed is not None else None
    iy = cfg.get('idle_yield', IDLE_YIELD)
    taxon = 1.0 if cfg.get('tax', True) else 0.0
    daily_yield = (1.0 + iy) ** (1.0 / TRADING_DAYS) - 1.0

    hard_pct = cfg.get('hard_pct', 0.0)
    hard_atr = cfg.get('hard_atr', 0.0)
    be_after = cfg.get('be_after', 0.0)
    lock_after = cfg.get('lock_after', 0.0)
    lock_trail = cfg.get('lock_trail', 0.0)
    trail_pct = cfg.get('trail_pct', 0.0)
    trail_chand = cfg.get('trail_chand', 0.0)
    trail_atr = cfg.get('trail_atr', 0.0)
    EX = cfg.get('exit_arr')
    time_bars = cfg.get('time_bars', 0)
    time_min_gain = cfg.get('time_min_gain', 0.0)
    block_bars = cfg.get('block_bars', 0)
    book_kill = cfg.get('book_dd_kill', 0.0)

    C, O, H = P.C, P.O, P.H
    CM = P.CM
    ATR14, ATR22, RS = P.ATR, P.ATR22, P.RS252
    rearm = P.GATE50

    n = i1 - i0 + 1
    nav = np.full(n, np.nan)
    inv = np.full(n, np.nan)
    cash = START_CAPITAL
    open_pos, trades, pending_exit = [], [], []
    blocked = {}
    fy_st = fy_lt = 0.0
    carry = 0.0
    cur_fy = None
    tax_paid = 0.0
    gross_cost = 0.0
    reasons = {}
    book_peak = START_CAPITAL
    halted = False
    kills = 0

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
            bill = (st * STCG + max(0.0, lt - LTCG_EXEMPT) * LTCG) * taxon
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
            reasons[p['reason']] = reasons.get(p['reason'], 0) + 1
            if block_bars and p['reason'] == 'STOP':
                blocked[p['j']] = i + block_bars
            trades.append(dict(symbol=P.cols[p['j']], entry_date=P.dstr[p['ei']],
                               exit_date=day, entry_px=p['epx'], exit_px=float(px),
                               shares=p['shares'], pnl=pnl,
                               ret_pct=100.0 * (proceeds / p['basis'] - 1.0),
                               bars=i - p['ei'], days=held_days,
                               mae_pct=100.0 * (p['trough'] / p['epx'] - 1.0),
                               mfe_pct=100.0 * (p['peakc'] / p['epx'] - 1.0),
                               reason=p['reason']))
        pending_exit = still

        if halted and rearm[i]:
            halted = False

        # ---- new entries at today's open
        if not halted:
            cand = np.nonzero(trig[i])[0]
            if len(cand) and (gate is None or gate[i]):
                held = {p['j'] for p in open_pos} | {p['j'] for p in pending_exit}
                cand = np.array([j for j in cand
                                 if j not in held and blocked.get(j, -1) <= i], dtype=int)
                free = slots - len(open_pos)
                if len(cand) and free > 0:
                    if len(cand) > free:
                        if rng is not None:
                            cand = cand[np.sort(rng.choice(len(cand), size=free,
                                                           replace=False))]
                        else:
                            rs = RS[i, cand]
                            rs = np.where(np.isfinite(rs), rs, -1e9)
                            order = np.lexsort((np.array([P.cols[j] for j in cand]), -rs))
                            cand = cand[order[:free]]
                    mv = sum(p['shares'] * _px(CM, i, p['j']) for p in open_pos)
                    navnow = cash + mv
                    for j in cand:
                        px = O[i, j]
                        if not np.isfinite(px) or px <= 0:
                            continue
                        shares = int((navnow * slot_pct) // (px * (1 + cost)))
                        if shares <= 0:
                            continue
                        basis = shares * px * (1 + cost)
                        if basis > cash:
                            continue
                        gross_cost += shares * px * cost
                        cash -= basis
                        a14 = ATR14[i, j]
                        open_pos.append(dict(
                            j=int(j), shares=shares, ei=i, epx=float(px), basis=basis,
                            peakc=float(px), peakh=float(px), trough=float(px),
                            atr0=float(a14) if np.isfinite(a14) else 0.0,
                            lvl=NEG, be=False, lock=False, reason=''))

        # ---- mark, then evaluate the close-based exit stack
        mv = 0.0
        keep = []
        for p in open_pos:
            c = _px(CM, i, p['j'])
            mv += p['shares'] * c
            if c > p['peakc']:
                p['peakc'] = c
            if c < p['trough']:
                p['trough'] = c
            hh = H[i, p['j']]
            if np.isfinite(hh) and hh > p['peakh']:
                p['peakh'] = float(hh)
            gain = c / p['epx'] - 1.0
            if be_after and not p['be'] and gain >= be_after:
                p['be'] = True
            if lock_after and not p['lock'] and gain >= lock_after:
                p['lock'] = True

            lvl = NEG
            if hard_pct:
                lvl = max(lvl, p['epx'] * (1.0 - hard_pct))
            if hard_atr and p['atr0'] > 0:
                lvl = max(lvl, p['epx'] - hard_atr * p['atr0'])
            if p['be']:
                lvl = max(lvl, p['epx'])
            tp = lock_trail if (p['lock'] and lock_trail) else trail_pct
            if tp:
                lvl = max(lvl, p['peakc'] * (1.0 - tp))
            if trail_chand:
                a = ATR22[i, p['j']]
                if np.isfinite(a):
                    lvl = max(lvl, p['peakh'] - trail_chand * a)
            if trail_atr:
                a = ATR14[i, p['j']]
                if np.isfinite(a):
                    lvl = max(lvl, p['peakc'] - trail_atr * a)
            if lvl > p['lvl']:
                p['lvl'] = lvl                      # RATCHET: never loosen

            out = None
            if p['lvl'] > NEG and c <= p['lvl']:
                out = 'STOP'
            elif time_bars and (i - p['ei']) >= time_bars and gain < time_min_gain:
                out = 'TIME'
            elif EX is not None and EX[i, p['j']]:
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

        # ---- BOOK-level trailing drawdown kill
        if book_kill and not halted:
            book_peak = max(book_peak, nav[k])
            if nav[k] <= book_peak * (1.0 - book_kill) and (open_pos or pending_exit):
                for p in open_pos:
                    p['reason'] = 'BOOKKILL'
                    pending_exit.append(p)
                open_pos = []
                halted = True
                kills += 1
        elif book_kill:
            book_peak = max(book_peak, nav[k])

    # ---- liquidate at the final close
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
            reasons['EOD'] = reasons.get('EOD', 0) + 1
            trades.append(dict(symbol=P.cols[p['j']], entry_date=P.dstr[p['ei']],
                               exit_date=P.dstr[i], entry_px=p['epx'], exit_px=float(px),
                               shares=p['shares'], pnl=pnl,
                               ret_pct=100.0 * (proceeds / p['basis'] - 1.0),
                               bars=i - p['ei'], days=held_days,
                               mae_pct=100.0 * (p['trough'] / p['epx'] - 1.0),
                               mfe_pct=100.0 * (p['peakc'] / p['epx'] - 1.0),
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
    return dict(nav=nav, dates=P.dstr[i0:i1 + 1], trades=trades, tax_paid=tax_paid,
                cost_paid=gross_cost, invested=inv, reasons=reasons, book_kills=kills)


def _px(arr, i, j):
    v = arr[i, j]
    return float(v) if np.isfinite(v) else 0.0
