# -*- coding: utf-8 -*-
"""Generate `sim171.py` from research/170's `sim170.py` by exact-string patches.

WHY NOT JUST COPY AND EDIT IT.  Every number in research/171 is measured against
research/170's incumbent and against OA-ROT-1 as research/170 defined it.  That comparison
is worth nothing if the engine has quietly drifted, so the engine is never hand-edited: it is
regenerated from research/170's file by patches that must match EXACTLY ONCE each, and the
build aborts if any of them matches zero times or more than once.  Same discipline as
research/170 `patch_engine170.py` (11 patches on research/160 engine) and research/165's
`patch_probe170.py` (3 patches on this one).

WHAT THE PATCHES ADD (nothing else, and nothing that fires at the defaults):

  * `_realise_pos` -- a FULL sale is taxed lot by lot, so a topped-up position does not hand
    its LTCG clock to shares bought yesterday.  With one lot the arithmetic is identical.
  * `lots` / `last_buy_i` on every position -- inert until a top-up happens.
  * an ELIGIBILITY HISTOGRAM: how many holdings are simultaneously more than the margin under
    water on an evening when a signal is refused.  It records; it decides nothing and it draws
    no random number.
  * a new block AFTER the inherited rotation block implementing research/171's three
    mechanics: `rot_spill` (what happens to an eligible loser with no entrant to buy),
    `rot_dest` in ('topup', 'hybrid') (redeploy into EXISTING holdings), and
    `rot_trigger='any'` (fire without a refused signal).
  * `max_pos_w` -- the largest weight any single position ever reached.

AT THE DEFAULTS -- `rot_dest='entrant'`, `rot_trigger='signal'`, `rot_spill='none'` -- not one
of those paths executes, and `--verify` proves it by running research/170's own incumbent and
OA-ROT-1 cells through both modules on the same seed and comparing the NAV curve and the
trade list bit for bit.

    python3 patch171.py            # regenerate
    python3 patch171.py --verify   # regenerate, then prove the no-op
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / '170_qs_leeway_and_baseage_best_entrant' / 'scripts' / 'sim170.py'
DST = HERE / 'sim171.py'

HEADER = '''
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
'''

PATCHES = [
    # 1 ─ the header note, right after the score-name table
    ("ROT_SCORES = ('cushion', 'rs', 'athdist', 'unreal', 'held', 'rand')\n",
     "ROT_SCORES = ('cushion', 'rs', 'athdist', 'unreal', 'held', 'rand')\n" + HEADER),

    # 2 ─ read the new config keys
    ("""    min_fill = float(cfg.get('min_fill_frac', 0.0) or 0.0)
""",
     """    min_fill = float(cfg.get('min_fill_frac', 0.0) or 0.0)
    rot_dest = cfg.get('rot_dest') or 'entrant'
    rot_trigger = cfg.get('rot_trigger') or 'signal'
    rot_spill = cfg.get('rot_spill') or 'none'
    topup_rank = cfg.get('topup_rank') or 'rs'
    topup_split = max(1, int(cfg.get('topup_split', 1) or 1))
    topup_cap = float(cfg.get('topup_cap', 0.0) or 0.0)
"""),

    # 3 ─ the new counters
    ("""    swaps = swap_attempts = trims = partial_fills = 0
""",
     """    swaps = swap_attempts = trims = partial_fills = 0
    topups = spills = topup_refused = 0
    topup_notional = 0.0
    elig_days = elig_total = elig_ge1 = elig_ge2 = elig_ge3 = elig_max = 0
    max_pos_w = 0.0
    swaps_today = 0
"""),

    # 4 ─ lot-aware realisation of a FULL sale
    ("""    def _realise(pnl, held_bars):
        nonlocal fy_st, fy_lt
        if held_bars >= LTCG_DAYS * 252 / 365:
            fy_lt += pnl
        else:
            fy_st += pnl
""",
     """    def _realise(pnl, held_bars):
        nonlocal fy_st, fy_lt
        if held_bars >= LTCG_DAYS * 252 / 365:
            fy_lt += pnl
        else:
            fy_st += pnl

    def _realise_pos(p, proceeds, i_now):
        \"\"\"Realise a FULL sale lot by lot.  With a single lot -- every cell that does no
        top-ups -- this is arithmetically identical to `_realise(proceeds - cost_basis,
        i_now - entry_i)`, which is what the inherited code called.\"\"\"
        lots = p.get('lots')
        if not lots or len(lots) == 1:
            _realise(proceeds - p['cost_basis'], i_now - p['entry_i'])
            return
        tot = float(p['shares'])
        if tot <= 0:
            return
        for (sh, bs, ei) in lots:
            _realise(proceeds * (sh / tot) - bs, i_now - ei)
"""),

    # 5 ─ the ordinary exit realises lot by lot
    ("""            pnl = proceeds - p['cost_basis']
            held = i - p['entry_i']
            _realise(pnl, held)
""",
     """            pnl = proceeds - p['cost_basis']
            held = i - p['entry_i']
            _realise_pos(p, proceeds, i)
"""),

    # 6 ─ a normal entry opens one tax lot
    ("""                    open_pos.append(dict(symbol=e['symbol'], shares=shares, entry_i=i,
                                         entry_px=float(px), cost_basis=basis,
                                         peak=float(px), reason='',
                                         tv20_cr=float(e.get('tv20_cr', np.nan))))
""",
     """                    open_pos.append(dict(symbol=e['symbol'], shares=shares, entry_i=i,
                                         entry_px=float(px), cost_basis=basis,
                                         peak=float(px), reason='',
                                         tv20_cr=float(e.get('tv20_cr', np.nan)),
                                         lots=[[shares, basis, i]], last_buy_i=i))
"""),

    # 7 ─ the eligibility histogram, and the gate on the inherited rotation block
    ("""        # ---- A: ROTATION -- swap the weakest holding out for an unfilled signal ---------
        if rot_score and unfilled and open_pos and j >= 0:
            swaps_today = 0
""",
     """        # ---- A0: research/171 RECORDS how many holdings are eligible at once -----------
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
"""),

    # 8 ─ a swapped-in entrant opens one tax lot
    ("""                open_pos.append(dict(symbol=sym, shares=shares, entry_i=i,
                                     entry_px=float(px), cost_basis=basis,
                                     peak=float(px), reason='',
                                     tv20_cr=float(e.get('tv20_cr', np.nan))))
""",
     """                open_pos.append(dict(symbol=sym, shares=shares, entry_i=i,
                                     entry_px=float(px), cost_basis=basis,
                                     peak=float(px), reason='',
                                     tv20_cr=float(e.get('tv20_cr', np.nan)),
                                     lots=[[shares, basis, i]], last_buy_i=i))
"""),

    # 9 ─ the swap-out realises lot by lot
    ("""                pnl = proceeds - p_out['cost_basis']
                _realise(pnl, i - p_out['entry_i'])
""",
     """                pnl = proceeds - p_out['cost_basis']
                _realise_pos(p_out, proceeds, i)
"""),

    # 10 ─ THE NEW BLOCK, plus the per-day largest-position tracker
    ("""        # ---- mark, then evaluate close-based exit signals --------------------------------
        mv = 0.0
        keep = []
        for p in open_pos:
            c = _px(panel.close[p['symbol']], i)
            mv += p['shares'] * c
""",
     """        # ---- research/171: SPILL, TOP-UP DESTINATION, ANY-EVENING TRIGGER ---------------
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
"""),

    # 11 ─ record the largest weight, and report the new counters
    ("""        inv[i] = mv / nav[i] if nav[i] > 0 else np.nan
""",
     """        inv[i] = mv / nav[i] if nav[i] > 0 else np.nan
        if nav[i] > 0:
            max_pos_w = max(max_pos_w, _dmax / nav[i])
"""),

    ("""                tax_lt=round(tax_lt, 0),
                turnover_x=round(traded_notional / 2.0 / mean_nav / yrs, 3))
""",
     """                tax_lt=round(tax_lt, 0),
                topups=topups, topups_per_yr=round(topups / yrs, 2),
                topup_notional=round(topup_notional, 0), spills=spills,
                topup_refused=topup_refused, elig_days=elig_days, elig_total=elig_total,
                elig_ge1=elig_ge1, elig_ge2=elig_ge2, elig_ge3=elig_ge3, elig_max=elig_max,
                elig_ge2_pct=(round(100.0 * elig_ge2 / elig_days, 2) if elig_days else 0.0),
                elig_mean=(round(elig_total / float(elig_days), 3) if elig_days else 0.0),
                max_pos_w=round(100.0 * max_pos_w, 2),
                turnover_x=round(traded_notional / 2.0 / mean_nav / yrs, 3))
"""),
]


def build():
    src = SRC.read_text(encoding='utf-8')
    out = src
    for k, (old, new) in enumerate(PATCHES, 1):
        n = out.count(old)
        if n != 1:
            raise SystemExit('PATCH %d matched %d times, expected exactly 1:\n%r'
                             % (k, n, old[:160]))
        out = out.replace(old, new)
    out = out.replace('"""research/166', '"""research/171 MULTI-SWAP COPY of research/166', 1)
    DST.write_text(out, encoding='utf-8')
    print('wrote %s (%d bytes from %d, %d patches)'
          % (DST.name, len(out), len(src), len(PATCHES)))


BASE_CFG = dict(exit='ST_14_4', hard_stop=False, hard_stop_pct=0.92, rot_sell_only=False,
                time_stop=0, cost_bps=25.0, gate_ok=None, idle_yield=0.052, slots=16,
                slot_pct=0.0625, select='random', rot_score=None, rot_margin=0.0,
                rot_max_per_day=1, rot_entrant='tv', trim_mult=0.0, trim_when='month',
                min_fill_frac=0.0)


def verify():
    """The defaults must be a no-op: same NAV, same trades, on research/170's own two cells."""
    import pickle

    import numpy as np
    import pandas as pd
    root = HERE.parents[2]
    sys.path.insert(0, str(SRC.parent))
    sys.path.insert(0, str(HERE))
    import sim170 as A
    import sim171 as B
    panel = pickle.load(open(root / 'research/164_baseage_slots_sizing/results/panel164.pkl',
                             'rb'))
    st = pickle.load(open(root / 'research/166_baseage_rotation_and_drift/results/st166.pkl',
                          'rb'))
    ev = pd.read_csv(root / 'research/166_baseage_rotation_and_drift/results/events166.csv')
    events = ev.to_dict('records')
    for e in events:
        e['entry_i'] = int(e['entry_i'])
    aux = A.build_aux(panel, st)
    cells = [('INCUMBENT  (BASE_rand)', dict(BASE_CFG)),
             ('OA-ROT-1   (X_entrs_unre_m010)',
              dict(BASE_CFG, rot_score='unreal', rot_margin=10.0, rot_entrant='rs'))]
    ok = True
    for name, cfg in cells:
        for seed in (1001, 1017):
            navA, trA, invA, bookA = A.simulate(events, panel, aux, cfg, seed)
            navB, trB, invB, bookB = B.simulate(events, panel, aux, cfg, seed)
            same_nav = bool(np.array_equal(np.nan_to_num(navA), np.nan_to_num(navB)))
            same_tr = (len(trA) == len(trB) and all(a == b for a, b in zip(trA, trB)))
            same_inv = bool(np.array_equal(np.nan_to_num(invA), np.nan_to_num(invB)))
            same_book = all(bookA[k] == bookB[k] for k in bookA)
            print('%-32s seed %4d | NAV identical: %-5s  trades identical: %-5s  '
                  'invested identical: %-5s  book identical: %-5s  (%d trades, %d swaps)'
                  % (name, seed, same_nav, same_tr, same_inv, same_book,
                     len(trA), bookA['swaps']))
            ok = ok and same_nav and same_tr and same_inv and same_book
    if not ok:
        raise SystemExit('sim171 IS NOT A NO-OP AT THE DEFAULTS - do not use it')
    print('VERIFY OK - sim171 at its defaults is research/170 sim170, bit for bit')


if __name__ == '__main__':
    build()
    if '--verify' in sys.argv:
        verify()
