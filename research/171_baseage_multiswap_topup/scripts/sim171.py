# -*- coding: utf-8 -*-
"""research/171 MULTI-SWAP COPY of research/166 -- the Open Alpha - Base Age book with TWO new capabilities:

  A. ROTATION -- when a qualifying signal cannot be taken (no free slot, or a free slot but
     not enough cash), rank the holdings, and swap the WEAKEST out for the new signal if the
     entrant beats it by a margin.  Sell and buy both at the NEXT open; both legs placeable.

  B. DRIFT / TRIMMING -- trim a position that has grown past K x its target weight back to
     target (at month-end, or on demand when a signal is about to be refused for cash), and
     optionally size a new entry to the cash actually available instead of refusing it.

Everything else is research/164's `sim164.simulate`, which is research/161's
`bt_core.simulate` with slots / slot_pct / select made live.  With `rot_score=None`,
`trim_mult=0` and `min_fill_frac=0` this file reproduces research/164 bit-exactly: the rng is
drawn only on days that bind, in the same order, with the same call.

CAUSALITY.  Every NEW rule decides on the CLOSE of the signal bar and executes at the NEXT
open -- the convention the inherited entry and exit already use.  Rotation scores are read at
close[i-1] and the swap executes at open[i]; a month-end trim is decided at the month-end
close and executed at the next open; a demand trim is decided at close[i-1] and executed at
open[i].  Inherited inconsistency, kept verbatim so the baseline reproduces: research/161
sizes a new entry off a NAV marked at close[i] while buying at open[i].  That is not extended
to any new rule.

TAX.  A swap-out and a trim are ordinary realisations: they run through the same Indian
financial-year netting block as any exit (20% STCG / 12.5% LTCG above 365 days, losses
carried forward, settled 1 April).  The tax cost of extra churn is therefore MODELLED, never
approximated by a haircut.  `book['tax_paid']` reports what was actually paid.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252.0
STCG, LTCG = 0.20, 0.125
LTCG_DAYS = 365
START_CAPITAL = 1_000_000.0

# entrant tie-break when several unfilled candidates compete for one swap
ENT_KEYS = {'tv': 'tv20_cr', 'rs': 'rs252', 'age': 'x_bars'}

# rank key -> (event field, descending?)   'random' handled separately
SELECT_RULES = {
    'random': None,
    'rs':     ('rs252', True),
    'ext':    ('ext_pct', False),
    'tv':     ('tv20_cr', True),
    'age':    ('x_bars', True),
}

# scores on which a HELD position can be ranked "weakest".  All read at close[i-1].
ROT_SCORES = ('cushion', 'rs', 'athdist', 'unreal', 'held', 'rand')

# ───────────────────────── research/171 MULTI-SWAP AND TOP-UP ─────────────────────────
# Added by research/171 `patch171.py`.  New config keys, all inert at their defaults:
#
#   rot_max_per_day  k -- how many holdings may leave in one evening.  ALREADY in research/170.
#   rot_dest      'entrant' (default) | 'topup' | 'hybrid'   where a swap's proceeds go
#   rot_trigger   'signal'  (default) | 'any'                is a refused signal required?
#   rot_spill     'none'    (default) | 'cash' | 'topup'     an eligible loser with no entrant
#   topup_rank    'rs' | 'unreal' | 'cushion'                which HOLDING gets topped up
#   topup_split   1 (all to the top one) | 2 (equally across the top two)
#   topup_cap     0 = no cap, else the multiple of the 6.25% target a position may not exceed
#
# A top-up is NOT an entry: it consumes no event, takes no slot, and cannot re-arm anything
# (the 60-bar re-arm lives in the event generator and is blind to the book).  It keeps the
# position's original `entry_i`, blends `entry_px` to the weighted-average buy price so the
# under-water trigger is measured against what the book actually paid, and records its own
# tax lot.


def _px(arr, i):
    v = arr[i]
    return float(v) if np.isfinite(v) else 0.0


def _f(x):
    x = float(x)
    return x if np.isfinite(x) else np.nan


def hold_score(p, j, panel, aux, score):
    """Score of a HELD position on the close of bar j (= the signal bar).  LOWER is weaker."""
    sym = p['symbol']
    if score == 'cushion':
        st = _f(aux['st'][sym][j]) if sym in aux['st'] else np.nan
        c = _f(panel.close[sym][j])
        if not (np.isfinite(st) and np.isfinite(c)) or st <= 0:
            return np.nan
        return 100.0 * (c / st - 1.0)
    if score == 'rs':
        return _f(aux['rs'][sym][j])
    if score == 'athdist':
        return _f(aux['athd'][sym][j])
    if score == 'unreal':
        c = _f(panel.close[sym][j])
        if not np.isfinite(c) or p['entry_px'] <= 0:
            return np.nan
        return 100.0 * (c / p['entry_px'] - 1.0)
    if score == 'held':
        return -float(j - p['entry_i'])
    return np.nan


def ent_score(sym, j, panel, aux, score):
    """The ENTRANT's score on the same bar, in the same units."""
    if score == 'cushion':
        st = _f(aux['st'][sym][j]) if sym in aux['st'] else np.nan
        c = _f(panel.close[sym][j])
        if not (np.isfinite(st) and np.isfinite(c)) or st <= 0:
            return np.nan
        return 100.0 * (c / st - 1.0)
    if score == 'rs':
        return _f(aux['rs'][sym][j])
    if score == 'athdist':
        return 0.0            # the entrant IS at a new all-time-high close, by construction
    if score in ('unreal', 'held'):
        return 0.0            # a brand-new position has no P&L and no holding period
    return np.nan


def simulate(events, panel, aux, cfg, seed):
    """Returns (nav, trades, inv, book)."""
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(1_000_000 + seed)     # the null's own stream: never perturbs
    slots = int(cfg.get('slots', 16))
    slot_pct = float(cfg.get('slot_pct', 0.0625))
    select = cfg.get('select', 'random')
    rule = SELECT_RULES[select]
    n = panel.n
    cost = cfg['cost_bps'] / 10000.0
    exit_key = cfg['exit']
    hard = cfg.get('hard_stop', False)
    tstop = cfg.get('time_stop', 0)
    gate = cfg.get('gate_ok')
    iy = cfg.get('idle_yield', 0.052)
    daily_yield = (1.0 + iy) ** (1.0 / TRADING_DAYS) - 1.0

    rot_score = cfg.get('rot_score') or None
    rot_margin = float(cfg.get('rot_margin', 0.0))
    rot_max = int(cfg.get('rot_max_per_day', 1))
    ent_key = ENT_KEYS[cfg.get('rot_entrant', 'tv')]
    # sell the weakest but DO NOT buy the entrant -- the control that separates "a swap"
    # from "a conditional stop-loss that happens to be triggered by someone else's signal"
    sell_only = bool(cfg.get('rot_sell_only', False))
    hard_pct = float(cfg.get('hard_stop_pct', 0.92))
    trim_mult = float(cfg.get('trim_mult', 0.0) or 0.0)
    trim_when = cfg.get('trim_when', 'month')
    min_fill = float(cfg.get('min_fill_frac', 0.0) or 0.0)
    rot_dest = cfg.get('rot_dest') or 'entrant'
    rot_trigger = cfg.get('rot_trigger') or 'signal'
    rot_spill = cfg.get('rot_spill') or 'none'
    topup_rank = cfg.get('topup_rank') or 'rs'
    topup_split = max(1, int(cfg.get('topup_split', 1) or 1))
    topup_cap = float(cfg.get('topup_cap', 0.0) or 0.0)

    by_day = {}
    for e in events:
        by_day.setdefault(e['entry_i'], []).append(e)
    # last trading day of each calendar month
    month_end = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        if panel.cal[i][:7] != panel.cal[i + 1][:7]:
            month_end[i] = True

    cash = START_CAPITAL
    nav = np.full(n, np.nan)
    inv = np.full(n, np.nan)
    open_pos = []
    trades = []
    pending_exit = []
    pending_trim = []        # [(pos, shares_to_sell)] decided at a close, sold at the next open
    fy_st, fy_lt = 0.0, 0.0
    carry_st = 0.0
    cur_fy = None
    days_signal = days_bind = days_full = turned_away = 0
    cash_refused = entries_taken = 0
    swaps = swap_attempts = trims = partial_fills = 0
    topups = spills = topup_refused = 0
    topup_notional = 0.0
    elig_days = elig_total = elig_ge1 = elig_ge2 = elig_ge3 = elig_max = 0
    max_pos_w = 0.0
    swaps_today = 0
    trim_notional = traded_notional = tax_paid = tax_st = tax_lt = 0.0

    def _realise(pnl, held_bars):
        nonlocal fy_st, fy_lt
        if held_bars >= LTCG_DAYS * 252 / 365:
            fy_lt += pnl
        else:
            fy_st += pnl

    def _realise_pos(p, proceeds, i_now):
        """Realise a FULL sale lot by lot.  With a single lot -- every cell that does no
        top-ups -- this is arithmetically identical to `_realise(proceeds - cost_basis,
        i_now - entry_i)`, which is what the inherited code called."""
        lots = p.get('lots')
        if not lots or len(lots) == 1:
            _realise(proceeds - p['cost_basis'], i_now - p['entry_i'])
            return
        tot = float(p['shares'])
        if tot <= 0:
            return
        for (sh, bs, ei) in lots:
            _realise(proceeds * (sh / tot) - bs, i_now - ei)

    for i in range(n):
        day = panel.cal[i]
        j = i - 1
        # ---- financial-year boundary: settle tax on 1 April --------------------------
        fy = int(day[:4]) - (1 if day[5:7] < '04' else 0)
        if cur_fy is None:
            cur_fy = fy
        elif fy != cur_fy:
            st, lt = fy_st, fy_lt
            pool = carry_st
            if st < 0:
                pool += st; st = 0.0
            if lt < 0:
                pool += lt; lt = 0.0
            if pool < 0 and st > 0:
                use = min(st, -pool); st -= use; pool += use
            if pool < 0 and lt > 0:
                use = min(lt, -pool); lt -= use; pool += use
            carry_st = min(pool, 0.0)
            paid = st * STCG + lt * LTCG
            cash -= paid
            tax_paid += paid
            tax_st += st * STCG
            tax_lt += lt * LTCG
            fy_st = fy_lt = 0.0
            cur_fy = fy

        # ---- sell everything queued from yesterday's close signal ---------------------
        still = []
        for p in pending_exit:
            px = panel.open[p['symbol']][i]
            if not np.isfinite(px):
                still.append(p); continue
            proceeds = p['shares'] * px * (1 - cost)
            cash += proceeds
            traded_notional += p['shares'] * px
            pnl = proceeds - p['cost_basis']
            held = i - p['entry_i']
            _realise_pos(p, proceeds, i)
            trades.append(dict(symbol=p['symbol'], entry_date=panel.cal[p['entry_i']],
                               exit_date=day, entry_px=p['entry_px'], exit_px=float(px),
                               shares=p['shares'], pnl=pnl,
                               ret_pct=100.0 * (proceeds / p['cost_basis'] - 1.0),
                               bars=held, reason=p['reason'], kind='FULL',
                               notional=p['cost_basis'], tv20_cr=p['tv20_cr']))
        pending_exit = still

        # ---- execute trims queued from yesterday's close ------------------------------
        stillt = []
        for (p, nsell) in pending_trim:
            if not any(q is p for q in open_pos):
                continue                                  # it exited before the trim landed
            px = panel.open[p['symbol']][i]
            if not np.isfinite(px):
                stillt.append((p, nsell)); continue
            nsell = int(min(nsell, p['shares'] - 1))      # never trim a position to zero
            if nsell <= 0:
                continue
            proceeds = nsell * px * (1 - cost)
            frac = nsell / float(p['shares'])
            basis_part = p['cost_basis'] * frac
            cash += proceeds
            traded_notional += nsell * px
            trim_notional += nsell * px
            _realise(proceeds - basis_part, i - p['entry_i'])
            p['shares'] -= nsell
            p['cost_basis'] -= basis_part
            trims += 1
            trades.append(dict(symbol=p['symbol'], entry_date=panel.cal[p['entry_i']],
                               exit_date=day, entry_px=p['entry_px'], exit_px=float(px),
                               shares=nsell, pnl=proceeds - basis_part,
                               ret_pct=100.0 * (proceeds / basis_part - 1.0),
                               bars=i - p['entry_i'], reason='TRIM', kind='TRIM',
                               notional=basis_part, tv20_cr=p['tv20_cr']))
        pending_trim = stillt

        # ---- new entries ---------------------------------------------------------------
        cands_all = by_day.get(i, [])
        unfilled = []
        if cands_all:
            days_signal += 1
            free0 = slots - len(open_pos)
            if free0 <= 0:
                days_full += 1
                days_bind += 1
                turned_away += len(cands_all)
            elif len(cands_all) > free0:
                days_bind += 1
                turned_away += len(cands_all) - free0
        if cands_all and (gate is None or gate[i]):
            cands = cands_all
            free = slots - len(open_pos)
            if free <= 0:
                unfilled = list(cands_all)
            else:
                if len(cands) > free:
                    if rule is None:
                        pick = rng.choice(len(cands), size=free, replace=False)
                        sel = sorted(pick)
                        ss = set(int(x) for x in sel)
                        cands = [cands_all[k] for k in sel]
                        unfilled = [cands_all[k] for k in range(len(cands_all)) if k not in ss]
                    else:
                        key, desc = rule
                        sgn = -1.0 if desc else 1.0
                        order = sorted(cands_all, key=lambda e: (sgn * e[key], e['symbol']))
                        cands = order[:free]
                        unfilled = order[free:]
                mv = sum(p['shares'] * _px(panel.close[p['symbol']], i) for p in open_pos)
                navnow = cash + mv
                for e in cands:
                    px = panel.open[e['symbol']][i]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    alloc = navnow * slot_pct
                    shares = int(alloc // (px * (1 + cost)))
                    if shares <= 0:
                        cash_refused += 1
                        unfilled.append(e)
                        continue
                    basis = shares * px * (1 + cost)
                    if basis > cash:
                        # --- B2: raise cash by trimming the most bloated holding ---------
                        if trim_mult > 0 and trim_when == 'demand' and j >= 0:
                            raised = _demand_trim(open_pos, panel, i, j, cost, trim_mult,
                                                  slot_pct, cash)
                            if raised:
                                p_t, nsell, got = raised
                                px_t = panel.open[p_t['symbol']][i]
                                frac = nsell / float(p_t['shares'])
                                basis_part = p_t['cost_basis'] * frac
                                cash += got
                                traded_notional += nsell * px_t
                                trim_notional += nsell * px_t
                                _realise(got - basis_part, i - p_t['entry_i'])
                                p_t['shares'] -= nsell
                                p_t['cost_basis'] -= basis_part
                                trims += 1
                                trades.append(dict(
                                    symbol=p_t['symbol'],
                                    entry_date=panel.cal[p_t['entry_i']], exit_date=day,
                                    entry_px=p_t['entry_px'], exit_px=float(px_t),
                                    shares=nsell, pnl=got - basis_part,
                                    ret_pct=100.0 * (got / basis_part - 1.0),
                                    bars=i - p_t['entry_i'], reason='TRIM_DEMAND',
                                    kind='TRIM', notional=basis_part,
                                    tv20_cr=p_t['tv20_cr']))
                        # --- B3: buy what the cash affords, if that is a big enough bite --
                        if basis > cash:
                            if min_fill > 0:
                                aff = int(max(cash, 0.0) // (px * (1 + cost)))
                                if aff > 0 and aff * px * (1 + cost) >= min_fill * alloc:
                                    shares = aff
                                    basis = shares * px * (1 + cost)
                                    partial_fills += 1
                                else:
                                    cash_refused += 1
                                    unfilled.append(e)
                                    continue
                            else:
                                cash_refused += 1
                                unfilled.append(e)
                                continue
                    cash -= basis
                    traded_notional += shares * px
                    entries_taken += 1
                    open_pos.append(dict(symbol=e['symbol'], shares=shares, entry_i=i,
                                         entry_px=float(px), cost_basis=basis,
                                         peak=float(px), reason='',
                                         tv20_cr=float(e.get('tv20_cr', np.nan)),
                                         lots=[[shares, basis, i]], last_buy_i=i))

        # ---- A0: research/171 RECORDS how many holdings are eligible at once -----------
        #      Recording only.  It decides nothing and it draws no random number.
        if rot_score == 'unreal' and unfilled and open_pos and j >= 0:
            _ne = 0
            for _p in open_pos:
                if _p['entry_i'] >= i:
                    continue
                _s = hold_score(_p, j, panel, aux, 'unreal')
                if np.isfinite(_s) and (0.0 - _s) >= rot_margin:
                    _ne += 1
            elig_days += 1
            elig_total += _ne
            elig_ge1 += 1 if _ne >= 1 else 0
            elig_ge2 += 1 if _ne >= 2 else 0
            elig_ge3 += 1 if _ne >= 3 else 0
            elig_max = max(elig_max, _ne)

        # ---- A: ROTATION -- swap the weakest holding out for an unfilled signal ---------
        swaps_today = 0
        if rot_score and unfilled and open_pos and j >= 0 and rot_dest == 'entrant':
            swaps_today = 0
            held_syms = set(p['symbol'] for p in open_pos)
            ents = sorted(unfilled,
                          key=lambda e: (-_f(e.get(ent_key, np.nan)) if np.isfinite(
                              _f(e.get(ent_key, np.nan))) else 1e18, e['symbol']))
            for e in ents:
                if swaps_today >= rot_max:
                    break
                sym = e['symbol']
                if sym in held_syms:
                    continue
                px = panel.open[sym][i]
                if not np.isfinite(px) or px <= 0:
                    continue
                elig = []
                for p in open_pos:
                    if p['entry_i'] >= i:
                        continue                       # bought today: not swappable today
                    spx = panel.open[p['symbol']][i]
                    if not np.isfinite(spx) or spx <= 0:
                        continue
                    sc = hold_score(p, j, panel, aux, rot_score) if rot_score != 'rand' else 0.0
                    if not np.isfinite(sc):
                        continue
                    elig.append((sc, p['symbol'], p, float(spx)))
                if not elig:
                    break
                if rot_score == 'rand':
                    weak = elig[int(rng2.integers(len(elig)))]
                    fire = bool(rng2.random() < rot_margin)
                else:
                    elig.sort(key=lambda t: (t[0], t[1]))
                    weak = elig[0]
                    es = ent_score(sym, j, panel, aux, rot_score)
                    if not np.isfinite(es):
                        continue
                    fire = (es - weak[0]) >= rot_margin
                swap_attempts += 1
                if not fire:
                    continue
                p_out, spx = weak[2], weak[3]
                proceeds = p_out['shares'] * spx * (1 - cost)
                mv_rest = sum(q['shares'] * _px(panel.close[q['symbol']], i)
                              for q in open_pos if q is not p_out)
                cash2 = cash + proceeds
                alloc = (cash2 + mv_rest) * slot_pct
                shares = int(min(alloc, cash2) // (px * (1 + cost)))
                if shares <= 0 and not sell_only:
                    continue
                # --- execute: sell the weakest, buy the entrant, same open ---------------
                cash = cash2
                traded_notional += p_out['shares'] * spx
                pnl = proceeds - p_out['cost_basis']
                _realise_pos(p_out, proceeds, i)
                trades.append(dict(symbol=p_out['symbol'],
                                   entry_date=panel.cal[p_out['entry_i']], exit_date=day,
                                   entry_px=p_out['entry_px'], exit_px=spx,
                                   shares=p_out['shares'], pnl=pnl,
                                   ret_pct=100.0 * (proceeds / p_out['cost_basis'] - 1.0),
                                   bars=i - p_out['entry_i'], reason='SWAP_OUT', kind='FULL',
                                   notional=p_out['cost_basis'], tv20_cr=p_out['tv20_cr']))
                open_pos = [q for q in open_pos if q is not p_out]
                pending_trim = [(q, ns) for (q, ns) in pending_trim if q is not p_out]
                swaps += 1
                swaps_today += 1
                held_syms.discard(p_out['symbol'])
                if sell_only:
                    continue           # the freed slot and cash go back into the normal queue
                basis = shares * px * (1 + cost)
                cash -= basis
                traded_notional += shares * px
                entries_taken += 1
                held_syms.add(sym)
                open_pos.append(dict(symbol=sym, shares=shares, entry_i=i,
                                     entry_px=float(px), cost_basis=basis,
                                     peak=float(px), reason='',
                                     tv20_cr=float(e.get('tv20_cr', np.nan)),
                                     lots=[[shares, basis, i]], last_buy_i=i))

        # ---- research/171: SPILL, TOP-UP DESTINATION, ANY-EVENING TRIGGER ---------------
        #  Runs only when one of the new keys is off its default.  Same causality as every
        #  other rule here: scores read at close[i-1] (= j), both legs fill at open[i].
        if (rot_score == 'unreal' and open_pos and j >= 0
                and (rot_dest != 'entrant' or rot_spill != 'none')):
            dest = rot_spill if rot_dest == 'entrant' else rot_dest
            if dest != 'none' and (unfilled or rot_trigger == 'any'):
                for _ in range(max(0, rot_max - swaps_today)):
                    elig2 = []
                    for p in open_pos:
                        if p.get('last_buy_i', p['entry_i']) >= i:
                            continue
                        spx = panel.open[p['symbol']][i]
                        if not np.isfinite(spx) or spx <= 0:
                            continue
                        sc = hold_score(p, j, panel, aux, 'unreal')
                        if not np.isfinite(sc):
                            continue
                        elig2.append((sc, p['symbol'], p, float(spx)))
                    if not elig2:
                        break
                    elig2.sort(key=lambda t: (t[0], t[1]))
                    sc_out, _so, p_out, spx = elig2[0]
                    swap_attempts += 1
                    if (0.0 - sc_out) < rot_margin:
                        break
                    proceeds = p_out['shares'] * spx * (1 - cost)
                    ent_buy = None
                    money = proceeds
                    if dest == 'hybrid' and unfilled:
                        _hs = set(q['symbol'] for q in open_pos if q is not p_out)
                        ents2 = sorted(
                            [e for e in unfilled if e['symbol'] not in _hs],
                            key=lambda e: (-_f(e.get(ent_key, np.nan)) if np.isfinite(
                                _f(e.get(ent_key, np.nan))) else 1e18, e['symbol']))
                        for e2 in ents2:
                            epx = panel.open[e2['symbol']][i]
                            if np.isfinite(epx) and epx > 0:
                                ent_buy = (e2, float(epx))
                                break
                        if ent_buy is not None:
                            money = proceeds * 0.5
                    plan = []
                    if dest in ('topup', 'hybrid'):
                        mv_rest = sum(q['shares'] * _px(panel.close[q['symbol']], i)
                                      for q in open_pos if q is not p_out)
                        navnow2 = cash + proceeds + mv_rest
                        cap_val = (topup_cap * slot_pct * navnow2) if topup_cap > 0 else 1e30
                        tg = []
                        for q in open_pos:
                            if q is p_out:
                                continue
                            qs = hold_score(q, j, panel, aux, topup_rank)
                            if not np.isfinite(qs):
                                continue
                            qpx = panel.open[q['symbol']][i]
                            if not np.isfinite(qpx) or qpx <= 0:
                                continue
                            tg.append((qs, q['symbol'], q, float(qpx)))
                        tg.sort(key=lambda t: (-t[0], t[1]))
                        share_money = money / float(topup_split)
                        for (_qs, _qsym, q, qpx) in tg:
                            if len(plan) >= topup_split:
                                break
                            room = cap_val - q['shares'] * _px(panel.close[q['symbol']], i)
                            spend = min(share_money, max(room, 0.0))
                            nb = int(spend // (qpx * (1 + cost)))
                            if nb > 0:
                                plan.append((q, nb, qpx))
                        if not plan and ent_buy is None:
                            topup_refused += 1
                            break
                    # --- execute: sell the weakest -------------------------------------
                    cash += proceeds
                    traded_notional += p_out['shares'] * spx
                    pnl = proceeds - p_out['cost_basis']
                    _realise_pos(p_out, proceeds, i)
                    trades.append(dict(symbol=p_out['symbol'],
                                       entry_date=panel.cal[p_out['entry_i']], exit_date=day,
                                       entry_px=p_out['entry_px'], exit_px=spx,
                                       shares=p_out['shares'], pnl=pnl,
                                       ret_pct=100.0 * (proceeds / p_out['cost_basis'] - 1.0),
                                       bars=i - p_out['entry_i'], reason='SWAP_OUT',
                                       kind='FULL', notional=p_out['cost_basis'],
                                       tv20_cr=p_out['tv20_cr']))
                    open_pos = [q for q in open_pos if q is not p_out]
                    pending_trim = [(q, ns) for (q, ns) in pending_trim if q is not p_out]
                    swaps += 1
                    swaps_today += 1
                    if dest == 'cash':
                        spills += 1
                    # --- the entrant half of a hybrid ----------------------------------
                    if ent_buy is not None:
                        e2, epx = ent_buy
                        nb = int(min(proceeds * 0.5, cash) // (epx * (1 + cost)))
                        if nb > 0:
                            b2 = nb * epx * (1 + cost)
                            cash -= b2
                            traded_notional += nb * epx
                            entries_taken += 1
                            open_pos.append(dict(symbol=e2['symbol'], shares=nb, entry_i=i,
                                                 entry_px=float(epx), cost_basis=b2,
                                                 peak=float(epx), reason='',
                                                 tv20_cr=float(e2.get('tv20_cr', np.nan)),
                                                 lots=[[nb, b2, i]], last_buy_i=i))
                            unfilled = [e for e in unfilled if e is not e2]
                    # --- top up the strongest holding(s) -------------------------------
                    for (q, nb, qpx) in plan:
                        b2 = nb * qpx * (1 + cost)
                        if b2 > cash:
                            nb = int(max(cash, 0.0) // (qpx * (1 + cost)))
                            if nb <= 0:
                                continue
                            b2 = nb * qpx * (1 + cost)
                        cash -= b2
                        traded_notional += nb * qpx
                        topups += 1
                        topup_notional += nb * qpx
                        q['lots'].append([nb, b2, i])
                        q['entry_px'] = ((q['entry_px'] * q['shares'] + qpx * nb)
                                         / float(q['shares'] + nb))
                        q['shares'] += nb
                        q['cost_basis'] += b2
                        q['last_buy_i'] = i

        # ---- mark, then evaluate close-based exit signals --------------------------------
        mv = 0.0
        _dmax = 0.0
        keep = []
        for p in open_pos:
            c = _px(panel.close[p['symbol']], i)
            mv += p['shares'] * c
            _dmax = max(_dmax, p['shares'] * c)
            out = None
            if hard and c <= p['entry_px'] * hard_pct:
                out = 'STOP8'
            elif tstop and (i - p['entry_i']) >= tstop:
                out = 'TIME'
            elif panel.sig[p['symbol']][exit_key][i]:
                out = exit_key
            if out and i + 1 < n:
                p['reason'] = out
                pending_exit.append(p)
            else:
                keep.append(p)
        open_pos = keep
        exiting = set(id(p) for p in pending_exit)
        pending_trim = [(q, ns) for (q, ns) in pending_trim if id(q) not in exiting]

        cash *= (1.0 + daily_yield)
        nav[i] = cash + mv
        inv[i] = mv / nav[i] if nav[i] > 0 else np.nan
        if nav[i] > 0:
            max_pos_w = max(max_pos_w, _dmax / nav[i])

        # ---- B1: month-end trim of anything above K x its target weight ------------------
        if trim_mult > 0 and trim_when == 'month' and month_end[i] and i + 1 < n and nav[i] > 0:
            target = nav[i] * slot_pct
            for p in open_pos:
                c = _px(panel.close[p['symbol']], i)
                if c <= 0:
                    continue
                val = p['shares'] * c
                if val > trim_mult * target:
                    nsell = int((val - target) // c)
                    nsell = int(min(nsell, p['shares'] - 1))
                    if nsell > 0:
                        pending_trim.append((p, nsell))

    if open_pos:
        i = n - 1
        for p in open_pos:
            px = _px(panel.close[p['symbol']], i)
            proceeds = p['shares'] * px * (1 - cost)
            cash += proceeds
            trades.append(dict(symbol=p['symbol'], entry_date=panel.cal[p['entry_i']],
                               exit_date=panel.cal[i], entry_px=p['entry_px'],
                               exit_px=float(px), shares=p['shares'],
                               pnl=proceeds - p['cost_basis'],
                               ret_pct=100.0 * (proceeds / p['cost_basis'] - 1.0),
                               bars=i - p['entry_i'], reason='EOD', kind='FULL',
                               notional=p['cost_basis'], tv20_cr=p['tv20_cr']))
        nav[i] = cash

    yrs = max((pd.Timestamp(panel.cal[-1]) - pd.Timestamp(panel.cal[0])).days / 365.25, 1e-9)
    mean_nav = float(np.nanmean(nav))
    book = dict(days_signal=days_signal, days_bind=days_bind, days_full=days_full,
                turned_away=turned_away, cash_refused=cash_refused,
                entries_taken=entries_taken, swaps=swaps, swap_attempts=swap_attempts,
                swaps_per_yr=round(swaps / yrs, 2), trims=trims,
                trims_per_yr=round(trims / yrs, 2),
                trim_notional=round(trim_notional, 0),
                partial_fills=partial_fills,
                tax_paid=round(tax_paid, 0), tax_st=round(tax_st, 0),
                tax_lt=round(tax_lt, 0),
                topups=topups, topups_per_yr=round(topups / yrs, 2),
                topup_notional=round(topup_notional, 0), spills=spills,
                topup_refused=topup_refused, elig_days=elig_days, elig_total=elig_total,
                elig_ge1=elig_ge1, elig_ge2=elig_ge2, elig_ge3=elig_ge3, elig_max=elig_max,
                elig_ge2_pct=(round(100.0 * elig_ge2 / elig_days, 2) if elig_days else 0.0),
                elig_mean=(round(elig_total / float(elig_days), 3) if elig_days else 0.0),
                max_pos_w=round(100.0 * max_pos_w, 2),
                turnover_x=round(traded_notional / 2.0 / mean_nav / yrs, 3))
    return nav, trades, inv, book


def _demand_trim(open_pos, panel, i, j, cost, trim_mult, slot_pct, cash):
    """The most bloated holding measured on the SIGNAL bar's close (j = i-1), trimmed back to
    target and sold at open[i].  Returns (position, shares_to_sell, proceeds) or None."""
    mv = 0.0
    vals = []
    for p in open_pos:
        c = _px(panel.close[p['symbol']], j)
        v = p['shares'] * c
        mv += v
        vals.append((v, c, p))
    navj = cash + mv
    if navj <= 0:
        return None
    target = navj * slot_pct
    vals.sort(key=lambda t: (-t[0], t[2]['symbol']))
    for (v, c, p) in vals:
        if v <= trim_mult * target or c <= 0:
            continue
        px = panel.open[p['symbol']][i]
        if not np.isfinite(px) or px <= 0:
            continue
        nsell = int((v - target) // c)
        nsell = int(min(nsell, p['shares'] - 1))
        if nsell <= 0:
            continue
        return p, nsell, nsell * float(px) * (1 - cost)
    return None


# --------------------------------------------------------------------------- metrics
def metrics(nav, cal, trades=None):
    nav = pd.Series(nav, index=pd.to_datetime(cal)).dropna()
    if len(nav) < 30 or nav.iloc[0] <= 0:
        return {}
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = (nav / nav.cummax() - 1.0)
    mdd = dd.min()
    r = nav.pct_change().dropna()
    sharpe = (r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan
    out = dict(cagr=round(100 * cagr, 2), maxdd=round(100 * mdd, 2),
               calmar=round(cagr / abs(mdd), 3) if mdd < 0 else np.nan,
               sharpe=round(sharpe, 3), final=round(nav.iloc[-1], 0), years=round(yrs, 2))
    if trades is not None and trades:
        tt = pd.DataFrame(trades)
        t = tt[tt['kind'] == 'FULL'] if 'kind' in tt.columns else tt
        if not len(t):
            return out
        wins = t[t.pnl > 0]; loss = t[t.pnl <= 0]
        streak = mx = 0
        for p in t.sort_values('exit_date')['pnl']:
            streak = streak + 1 if p <= 0 else 0
            mx = max(mx, streak)
        out.update(trades=len(t),
                   win_rate=round(100 * len(wins) / len(t), 1),
                   avg_win=round(wins.ret_pct.mean(), 2) if len(wins) else np.nan,
                   avg_loss=round(loss.ret_pct.mean(), 2) if len(loss) else np.nan,
                   expectancy=round(t.ret_pct.mean(), 3),
                   max_loss_streak=mx,
                   trades_per_yr=round(len(t) / yrs, 1))
        tv = t['tv20_cr'].to_numpy(float) * 1e7
        ok = np.isfinite(tv) & (tv > 0)
        if ok.any():
            frac = 100.0 * t['notional'].to_numpy(float)[ok] / tv[ok]
            out['pos_rs_med'] = round(float(np.median(t['notional'])), 0)
            out['cap_pct_med'] = round(float(np.median(frac)), 3)
            out['cap_pct_p95'] = round(float(np.percentile(frac, 95)), 3)
            out['cap_over1pct'] = round(100.0 * float((frac > 1.0).mean()), 1)
        # share of total book profit earned by the ten best trades
        pn = t['pnl'].to_numpy(float)
        tot = float(pn[pn > 0].sum() + pn[pn <= 0].sum())
        top10 = float(np.sort(pn)[-10:].sum()) if len(pn) >= 10 else np.nan
        out['profit_total'] = round(tot, 0)
        out['top10_share'] = round(100.0 * top10 / tot, 1) if tot > 0 else np.nan
        # the same measure over EVERY realisation, trims included -- the honest version for a
        # trimming cell, where one name's profit is split across a trim and a final exit
        pna = tt['pnl'].to_numpy(float)
        tota = float(pna.sum())
        t10a = float(np.sort(pna)[-10:].sum()) if len(pna) >= 10 else np.nan
        out['top10_share_all'] = round(100.0 * t10a / tota, 1) if tota > 0 else np.nan
    return out


def window_dd(nav, cal, lo, hi):
    """Drawdown inside a window, measured from the running peak of the FULL curve."""
    s = pd.Series(nav, index=pd.to_datetime(cal)).dropna()
    dd = s / s.cummax() - 1.0
    sub = dd[(dd.index >= lo) & (dd.index <= hi)]
    return round(float(sub.min() * 100), 2) if len(sub) else np.nan


def window_cagr(nav, cal, lo, hi):
    idx = [i for i, d in enumerate(cal) if lo <= d <= hi]
    if not idx:
        return np.nan
    sl = slice(idx[0], idx[-1] + 1)
    m = metrics(np.asarray(nav)[sl], cal[sl])
    return m.get('cagr', np.nan)


def build_aux(panel, st):
    """rs252 (12-month price return, %) and athd (% below the running-max close since the
    panel start, 2005-01-03) for every symbol -- both causal, both derived from the panel's
    own forward-filled close, so they need no database read."""
    rs, athd = {}, {}
    n = panel.n
    for sym, c in panel.close.items():
        cc = np.asarray(c, dtype=np.float64)
        r = np.full(n, np.nan, np.float32)
        if n > 252:
            prev = cc[:-252]
            with np.errstate(invalid='ignore', divide='ignore'):
                v = np.where(prev > 0, 100.0 * (cc[252:] / prev - 1.0), np.nan)
            r[252:] = v
        rs[sym] = r
        cm = pd.Series(cc).cummax().to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            a = np.where(cm > 0, 100.0 * (cc / cm - 1.0), np.nan)
        athd[sym] = a.astype(np.float32)
    return dict(st=st, rs=rs, athd=athd)
