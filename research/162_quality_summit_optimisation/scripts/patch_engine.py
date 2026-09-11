# -*- coding: utf-8 -*-
"""research/162 — derive `qg_engine2.py` from research/160's frozen `qg_engine.py`.

r/160's engine is the validated artefact (five self-tests, a look-ahead probe, a
random-selection null). It must not be edited: r/160's published numbers have to stay
regenerable from it. So this script produces r/162's engine as an explicit, auditable set
of exact-string patches on top of it, and fails loudly if any anchor has moved.

What the patches add — and nothing else:
  1. `--exits st_trail:14_4` and `--exits chand:22_3` — ATR-scaled close trails read from
     research/162's aux npz as precomputed boolean exit-signal frames.
  2. ranking axes `profit_g3`, `opm_slope3` and the cross-sectional composites
     `z_rs_profit` / `z_rs_opm`, computed among the day's candidates.
  3. `--weights invvol` — inverse-60-day-volatility position targets, capped at 2x the
     equal-weight target and floored at 0.25x, re-based so the top-`slots` candidates
     average exactly the equal-weight target.
  4. `weights` written into the output CSV.

Everything else — the fill convention, the tax model, the mask loader, the gate, the
metrics, the offset/seed ensembles, the resume logic — is byte-identical to r/160.
"""
import sys
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'research' / '160_quality_growth_near_ath' / 'scripts' / 'qg_engine.py'
DST = Path(__file__).resolve().parent / 'qg_engine2.py'

HEADER = '''# -*- coding: utf-8 -*-
# =========================================================================================
# research/162 — GENERATED from research/160's qg_engine.py by patch_engine.py.
# Do not hand-edit: edit patch_engine.py and regenerate, so the diff against the validated
# r/160 engine stays visible and auditable. r/160's engine itself is frozen.
# Added here: ATR-scaled close trails (SuperTrend / chandelier) from the r/162 aux npz,
# fundamental and composite ranking axes, and inverse-volatility position sizing.
# =========================================================================================
'''

AUX_HELPERS = '''

# ------------------------------------------------------------------ r/162 aux frames ---
_AUX_CACHE = {}


def aux_frame(path, key):
    """Lazily pull one frame out of research/162's aux npz, cached per process.

    The npz carries the SuperTrend / chandelier exit-signal booleans, the 60-day volatility
    and the point-in-time ranking fields, all on the r/160 panel's own date and symbol
    axes, so a frame can be indexed [day, col] exactly like a panel frame."""
    if not path:
        raise SystemExit('this cell needs --aux <aux_162.npz> (built by build_aux.py)')
    z = _AUX_CACHE.get(('z', path))
    if z is None:
        z = np.load(path)
        _AUX_CACHE[('z', path)] = z
    k = (path, key)
    if k not in _AUX_CACHE:
        if key not in z.files:
            raise SystemExit('aux npz %s has no frame %r (has: %s)'
                             % (path, key, sorted(x for x in z.files if x not in
                                                  ('dates', 'syms'))))
        _AUX_CACHE[k] = np.asarray(z[key])
    return _AUX_CACHE[k]
'''

ORDER_NEW = '''def _zc(v):
    """Cross-sectional z-score among the day's candidates. A name with no value scores
    -3 sigma rather than NaN, so a missing fundamental can never rank first."""
    v = np.asarray(v, dtype=np.float64)
    fin = np.isfinite(v)
    if fin.sum() < 3:
        return np.zeros_like(v)
    m, s = v[fin].mean(), v[fin].std()
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(v)
    z = (v - m) / s
    return np.where(np.isfinite(z), z, -3.0)


def _wfactors(cand, IVOL, i, slots):
    """Inverse-volatility position multipliers, re-based so the top-`slots` candidates
    average 1.0 (i.e. the equal-weight target), then clipped to [0.25, 2.0]. A name with
    no volatility estimate takes 1.0. Returns {} for equal weighting."""
    if IVOL is None:
        return {}
    top = [int(c) for c in cand[:max(int(slots), 1)]]
    iv = np.array([IVOL[i, c] for c in top], dtype=np.float64)
    inv = np.where(np.isfinite(iv) & (iv > 0), 1.0 / iv, np.nan)
    base = np.nanmean(inv)
    if not np.isfinite(base) or base <= 0:
        return {}
    out = {}
    for c in [int(x) for x in cand]:
        v = IVOL[i, c]
        f = (1.0 / v) / base if (np.isfinite(v) and v > 0) else 1.0
        out[c] = float(min(max(f, 0.25), 2.0))
    return out


def _order(cand, RV, i, rank, rng, RV2=None):
    """Rank the candidate columns. A random permutation is applied FIRST so that ties are
    broken randomly and the seed ensemble actually measures path dependence."""
    cand = rng.permutation(cand)
    if rank == 'random' or RV is None:
        return cand
    if RV2 is not None:
        v = _zc(RV[i, cand]) + _zc(RV2[i, cand])
    else:
        v = RV[i, cand]
        v = np.where(np.isfinite(v), v, -np.inf)
    return cand[np.argsort(-v, kind='stable')]
'''

RANK_NEW = '''    RV = RV2 = None
    if cell.rank == 'rs':
        RV = panel.f('score')
    elif cell.rank == 'profit_g3':
        RV = aux_frame(cell.aux, 'profit_g3')
    elif cell.rank == 'opm_slope3':
        RV = aux_frame(cell.aux, 'opm_slope3')
    elif cell.rank == 'z_rs_profit':
        RV, RV2 = panel.f('score'), aux_frame(cell.aux, 'profit_g3')
    elif cell.rank == 'z_rs_opm':
        RV, RV2 = panel.f('score'), aux_frame(cell.aux, 'opm_slope3')
    elif cell.rank == 'dist_ath':
        RV = C / ATH
    elif cell.rank == 'tv_desc':
        RV = TV
    elif cell.rank in ('mcap_asc', 'mcap_desc'):
        if der.shares is None:
            raise SystemExit('mcap ranking needs %s' % MCAP_SNAP)
        m = (C * der.shares[None, :].astype(np.float32))
        RV = m if cell.rank == 'mcap_desc' else -m
    elif cell.rank != 'random':
        raise SystemExit('unknown --rank %r' % cell.rank)
'''

PATCHES = [
    # ---- 1. the Cell gets two new axes -------------------------------------------------
    ("    arms: str = 'all'                   # all = gross/net/net_tax | tax = the net-of-tax arm only\n",
     "    arms: str = 'all'                   # all = gross/net/net_tax | tax = the net-of-tax arm only\n"
     "    aux: str = ''                       # r/162 derived-frame npz (ATR trails, vol, PIT ranks)\n"
     "    weights: str = 'equal'              # equal | invvol\n"),

    # ---- 2. the exit parser learns two ATR families ------------------------------------
    ("    out = dict(hard_stop=None, peak_dd=None, sma_trail=None, donch=None, months=None,\n"
     "               fund_fail=False)\n"
     "    known = {'none', 'fund_fail', 'sma_trail', 'peak_dd', 'donchian_low', 'time', 'hard_stop'}\n",
     "    out = dict(hard_stop=None, peak_dd=None, sma_trail=None, donch=None, months=None,\n"
     "               fund_fail=False, trail_key=None)\n"
     "    known = {'none', 'fund_fail', 'sma_trail', 'peak_dd', 'donchian_low', 'time',\n"
     "             'hard_stop', 'st_trail', 'chand'}\n"),
    ("        elif name == 'sma_trail':\n            out['sma_trail'] = int(arg)\n",
     "        elif name == 'st_trail':\n"
     "            out['trail_key'] = 'st_' + arg.replace('.0', '')\n"
     "        elif name == 'chand':\n"
     "            out['trail_key'] = 'ch_' + arg.replace('.0', '')\n"
     "        elif name == 'sma_trail':\n            out['sma_trail'] = int(arg)\n"),

    # ---- 3. the simulator sees the trail frame and the vol frame -----------------------
    ("    SMA, DL = ctx['SMA'], ctx['DL']\n",
     "    SMA, DL = ctx['SMA'], ctx['DL']\n"
     "    TRAIL, IVOL = ctx['TRAIL'], ctx['IVOL']\n"),
    ("                elif ex['sma_trail'] and i > p[0] and np.isfinite(SMA[i, col]) and c < SMA[i, col]:\n"
     "                    reason = 'sma_trail'\n",
     "                elif TRAIL is not None and i > p[0] and TRAIL[i, col]:\n"
     "                    reason = 'atr_trail'\n"
     "                elif ex['sma_trail'] and i > p[0] and np.isfinite(SMA[i, col]) and c < SMA[i, col]:\n"
     "                    reason = 'sma_trail'\n"),

    # ---- 4. inverse-vol position targets on the rebalance entry ------------------------
    ("                    tgt = tgt_pct * nav\n"
     "                    for col in cand:\n"
     "                        if free <= 0:\n"
     "                            break\n"
     "                        col = int(col)\n"
     "                        if col in pos and col not in selling:\n"
     "                            continue\n"
     "                        px = C[i, col]\n"
     "                        if not np.isfinite(px):\n"
     "                            continue\n"
     "                        qty = int(tgt / px)\n",
     "                    tgt = tgt_pct * nav\n"
     "                    wf = _wfactors(cand, IVOL, i, slots)\n"
     "                    cap = float(cell.max_position_pct) * nav\n"
     "                    for col in cand:\n"
     "                        if free <= 0:\n"
     "                            break\n"
     "                        col = int(col)\n"
     "                        if col in pos and col not in selling:\n"
     "                            continue\n"
     "                        px = C[i, col]\n"
     "                        if not np.isfinite(px):\n"
     "                            continue\n"
     "                        qty = int(min(tgt * wf.get(col, 1.0), cap) / px)\n"),

    # ---- 5. composites reach _order ----------------------------------------------------
    ("                cand = _order(cand, RV, i, cell.rank, rng)\n"
     "                rank_of = {int(c): r for r, c in enumerate(cand)}\n",
     "                cand = _order(cand, RV, i, cell.rank, rng, ctx.get('RV2'))\n"
     "                rank_of = {int(c): r for r, c in enumerate(cand)}\n"),
    ("                        cand = _order(cand, RV, i, cell.rank, rng)\n"
     "                        nav = cash + sum",
     "                        cand = _order(cand, RV, i, cell.rank, rng, ctx.get('RV2'))\n"
     "                        nav = cash + sum"),

    # ---- 6. the ranking block and the helper functions ---------------------------------
    ("    RV = None\n"
     "    if cell.rank == 'rs':\n"
     "        RV = panel.f('score')\n"
     "    elif cell.rank == 'dist_ath':\n"
     "        RV = C / ATH\n"
     "    elif cell.rank == 'tv_desc':\n"
     "        RV = TV\n"
     "    elif cell.rank in ('mcap_asc', 'mcap_desc'):\n"
     "        if der.shares is None:\n"
     "            raise SystemExit('mcap ranking needs %s' % MCAP_SNAP)\n"
     "        m = (C * der.shares[None, :].astype(np.float32))\n"
     "        RV = m if cell.rank == 'mcap_desc' else -m\n"
     "    elif cell.rank != 'random':\n"
     "        raise SystemExit('unknown --rank %r' % cell.rank)\n",
     RANK_NEW),

    ("    SMA = panel.f('sma%d' % ex['sma_trail']) if ex['sma_trail'] else None\n"
     "    DL = panel.f('donch_low%d' % ex['donch']) if ex['donch'] else None\n"
     "    weak = build_gate(panel, cell.index_gate)\n",
     "    SMA = panel.f('sma%d' % ex['sma_trail']) if ex['sma_trail'] else None\n"
     "    DL = panel.f('donch_low%d' % ex['donch']) if ex['donch'] else None\n"
     "    TRAIL = aux_frame(cell.aux, ex['trail_key']) if ex.get('trail_key') else None\n"
     "    if cell.weights not in ('equal', 'invvol'):\n"
     "        raise SystemExit('unknown --weights %r (equal | invvol)' % cell.weights)\n"
     "    IVOL = aux_frame(cell.aux, 'vol60') if cell.weights == 'invvol' else None\n"
     "    weak = build_gate(panel, cell.index_gate)\n"),

    ("    ctx = dict(cell=cell, C=C, O=O, ATH=ATH, TV=TV, QUAL=QUAL, MASKD=MASKD, ELIG=ELIG,\n"
     "               RV=RV, SMA=SMA, DL=DL, weak=weak, days=days, dts=dts, dt_all=dt_all, ex=ex)\n",
     "    ctx = dict(cell=cell, C=C, O=O, ATH=ATH, TV=TV, QUAL=QUAL, MASKD=MASKD, ELIG=ELIG,\n"
     "               RV=RV, RV2=RV2, TRAIL=TRAIL, IVOL=IVOL,\n"
     "               SMA=SMA, DL=DL, weak=weak, days=days, dts=dts, dt_all=dt_all, ex=ex)\n"),

    # ---- 7. helpers: replace _order wholesale, and bolt the aux loader on --------------
    ("def _order(cand, RV, i, rank, rng):\n"
     '    """Rank the candidate columns. A random permutation is applied FIRST so that ties are\n'
     '    broken randomly and the seed ensemble actually measures path dependence."""\n'
     "    cand = rng.permutation(cand)\n"
     "    if rank == 'random' or RV is None:\n"
     "        return cand\n"
     "    v = RV[i, cand]\n"
     "    v = np.where(np.isfinite(v), v, -np.inf)\n"
     "    return cand[np.argsort(-v, kind='stable')]\n",
     ORDER_NEW),

    # ---- 8. the CSV gains the weights column -------------------------------------------
    ("    'tv_floor', 'mask', 'mask_missing', 'exits', 'index_gate', 'gate_action', 'fill',\n",
     "    'tv_floor', 'mask', 'mask_missing', 'exits', 'weights', 'index_gate', 'gate_action',\n"
     "    'fill',\n"),
]


def main():
    src = SRC.read_text(encoding='utf-8')
    out = src
    for i, (old, new) in enumerate(PATCHES, 1):
        n = out.count(old)
        if n != 1:
            print('PATCH %d FAILED: anchor occurs %d times (expected 1):\n---\n%s\n---'
                  % (i, n, old[:400]))
            return 2
        out = out.replace(old, new, 1)
    # the aux loader goes right before the mask loader section
    anchor = '# -------------------------------------------------------------------------- mask/gate ---'
    if out.count(anchor) != 1:
        print('PATCH 9 FAILED: mask/gate anchor')
        return 2
    out = out.replace(anchor, AUX_HELPERS.strip('\n') + '\n\n\n' + anchor, 1)
    DST.write_text(HEADER + out, encoding='utf-8')
    print('wrote %s (%d lines, %d patches applied)'
          % (DST, out.count('\n') + 1, len(PATCHES) + 1))
    import py_compile
    py_compile.compile(str(DST), doraise=True)
    print('compiles clean')
    return 0


if __name__ == '__main__':
    sys.exit(main())
