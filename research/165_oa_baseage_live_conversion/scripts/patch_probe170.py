# -*- coding: utf-8 -*-
"""Generate `sim170_probe.py` from research/170's `sim170.py` by exact-string patches.

WHY NOT JUST COPY IT AND EDIT. The replication gate's whole claim is that the live
`rot1_pick()` makes the same decision research/170's engine made. That claim is worth
nothing if the engine has quietly drifted, so the engine is never hand-edited: it is
regenerated from research/170's file by patches that must match EXACTLY ONCE each, and the
run aborts if any of them matches zero times or more than once. This is the same discipline
research/170 itself used to build its Part A engine out of research/160's
(`patch_engine170.py`, 11 exact-string patches).

The patches add a PROBE sink and nothing else. No arithmetic is touched: with `PROBE = None`
the generated file is behaviourally identical to the original, and `--verify` proves it by
re-running one seed through both and comparing the NAV curve bit for bit.

    python3 patch_probe170.py            # regenerate
    python3 patch_probe170.py --verify   # regenerate, then prove PROBE=None is a no-op
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / '170_qs_leeway_and_baseage_best_entrant' / 'scripts' / 'sim170.py'
DST = HERE / 'sim170_probe.py'

HEADER = '''
# ───────────────────────── research/165 DECISION PROBE ─────────────────────────
# Added by research/165 `patch_probe170.py`. With PROBE left at None this module is
# research/170's `sim170.py` exactly; set PROBE to a list and every rotation decision point
# is appended to it, so the LIVE `services.oa_baseage_entry.rot1_pick()` can be asked to
# re-decide the same inputs and the two answers compared.
PROBE = None


def _probe_open(day, i, j, cash, open_pos, ents, held_syms, panel, aux, ent_key, rot_score):
    """One record per decision point, appended BEFORE the engine decides anything."""
    def _c(sym, k):
        v = panel.close[sym][k]
        return float(v) if np.isfinite(v) else None

    def _o(sym, k):
        v = panel.open[sym][k]
        return float(v) if np.isfinite(v) else None

    rec = dict(day=day, i=int(i), j=int(j), cash=float(cash),
               holdings=[dict(symbol=p['symbol'], shares=int(p['shares']),
                              entry_px=float(p['entry_px']), entry_i=int(p['entry_i']),
                              close_j=_c(p['symbol'], j), close_i=_c(p['symbol'], i),
                              open_i=_o(p['symbol'], i),
                              hold_score=(None if rot_score == 'rand' else
                                          (lambda s: None if not np.isfinite(s) else float(s))(
                                              hold_score(p, j, panel, aux, rot_score))))
                         for p in open_pos],
               entrants=[dict(symbol=e['symbol'],
                              rs252=(lambda v: None if not np.isfinite(v) else float(v))(
                                  _f(aux['rs'][e['symbol']][j])
                                  if e['symbol'] in aux['rs'] else np.nan),
                              tv20_cr=float(e.get('tv20_cr', np.nan))
                              if np.isfinite(_f(e.get('tv20_cr', np.nan))) else None,
                              x_bars=int(e.get('x_bars', 0)),
                              open_i=_o(e['symbol'], i))
                         for e in ents],
               held=sorted(held_syms), fired=False, out_symbol=None, in_symbol=None,
               shares=0)
    PROBE.append(rec)
    return rec
'''

PATCHES = [
    # 1. the sink and its helper, right after the score-name table
    ("ROT_SCORES = ('cushion', 'rs', 'athdist', 'unreal', 'held', 'rand')\n",
     "ROT_SCORES = ('cushion', 'rs', 'athdist', 'unreal', 'held', 'rand')\n" + HEADER),
    # 2. open a record at the top of the rotation block, before the entrant walk
    ("""                              _f(e.get(ent_key, np.nan))) else 1e18, e['symbol']))
            for e in ents:
""",
     """                              _f(e.get(ent_key, np.nan))) else 1e18, e['symbol']))
            _rec = (_probe_open(day, i, j, cash, open_pos, ents, held_syms, panel, aux,
                                ent_key, rot_score) if PROBE is not None else None)
            for e in ents:
"""),
    # 3. START THE BOOK SOMEWHERE ELSE. Not a rule change: it moves the opening cash and lets
    #    the book open with positions already in it, so research/170's own engine can be run
    #    over a sub-window from the LIVE book's state. Absent both keys this is the original
    #    line. `entry_i = -1` means "bought before the window", so a seeded position is
    #    eligible to be swapped out on day one exactly as a real holding would be.
    ("""    cash = START_CAPITAL
""",
     """    cash = float(cfg.get('start_capital', START_CAPITAL))
"""),
    ("""    open_pos = []
    trades = []
""",
     """    open_pos = [dict(symbol=p['symbol'], shares=int(p['qty']), entry_i=-1,
                     entry_px=float(p['buy']),
                     cost_basis=float(p['qty']) * float(p['buy']),
                     peak=float(p['buy']), reason='', tv20_cr=float('nan'))
                for p in (cfg.get('seed_positions') or [])]
    trades = []
"""),
    # 4. record what it actually did, at the moment it does it
    ("""                swaps += 1
                swaps_today += 1
""",
     """                swaps += 1
                swaps_today += 1
                if _rec is not None:
                    _rec.update(fired=True, out_symbol=p_out['symbol'], in_symbol=sym,
                                shares=int(shares), sell_px=float(spx), buy_px=float(px))
"""),
]


def build():
    src = SRC.read_text(encoding='utf-8')
    out = src
    for k, (old, new) in enumerate(PATCHES, 1):
        n = out.count(old)
        if n != 1:
            raise SystemExit('PATCH %d matched %d times, expected exactly 1:\n%r'
                             % (k, n, old[:120]))
        out = out.replace(old, new)
    out = out.replace('"""research/166', '"""research/165 PROBE COPY of research/166', 1)
    DST.write_text(out, encoding='utf-8')
    print('wrote %s (%d bytes from %d, %d patches)'
          % (DST.name, len(out), len(src), len(PATCHES)))


def verify():
    """PROBE=None must be a no-op: same NAV, same trades, same book, on one seed."""
    import pickle

    import numpy as np
    import pandas as pd
    root = HERE.parents[2]
    sys.path.insert(0, str(SRC.parent))
    sys.path.insert(0, str(HERE))
    import sim170 as A
    import sim170_probe as B
    panel = pickle.load(open(root / 'research/164_baseage_slots_sizing/results/panel164.pkl',
                             'rb'))
    st = pickle.load(open(root / 'research/166_baseage_rotation_and_drift/results/st166.pkl',
                          'rb'))
    ev = pd.read_csv(root / 'research/166_baseage_rotation_and_drift/results/events166.csv')
    events = ev.to_dict('records')
    for e in events:
        e['entry_i'] = int(e['entry_i'])
    cfg = dict(exit='ST_14_4', hard_stop=False, hard_stop_pct=0.92, rot_sell_only=False,
               time_stop=0, cost_bps=25.0, gate_ok=None, idle_yield=0.052, slots=16,
               slot_pct=0.0625, select='random', rot_score='unreal', rot_margin=10.0,
               rot_max_per_day=1, rot_entrant='rs', trim_mult=0.0, trim_when='month',
               min_fill_frac=0.0)
    auxA = A.build_aux(panel, st)
    navA, trA, invA, bookA = A.simulate(events, panel, auxA, cfg, 1001)
    B.PROBE = None
    navB, trB, invB, bookB = B.simulate(events, panel, auxA, cfg, 1001)
    same_nav = bool(np.array_equal(np.nan_to_num(navA), np.nan_to_num(navB)))
    print('NAV identical            : %s' % same_nav)
    print('trades identical         : %s' % (len(trA) == len(trB)
                                             and all(a == b for a, b in zip(trA, trB))))
    print('book identical           : %s' % (bookA == bookB))
    print('swaps                    : %d' % bookA['swaps'])
    if not (same_nav and bookA == bookB):
        raise SystemExit('PROBE COPY IS NOT A NO-OP - do not use it')
    print('VERIFY OK - the probe copy is behaviourally identical with PROBE=None')


if __name__ == '__main__':
    build()
    if '--verify' in sys.argv:
        verify()
