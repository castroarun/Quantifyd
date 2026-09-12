# -*- coding: utf-8 -*-
"""research/163 - IPO Base re-optimised on a PLACEABLE entry.

research/153 decided on bar i's CLOSE and filled at bar i's OPEN (the 8th deadly sin,
playbook 5A). services/ipo_paper.py is correct: it triggers on tonight's close and carries a
buy-stop at the pivot into the NEXT morning. Every r/153 parameter was fitted against the
look-ahead surface. On Open Alpha that surface INVERTED once the entry was made placeable.

This fork:
  * keeps r/153's validated exit and book machinery (ipo_replay.simulate_ipo) untouched,
  * replaces the entry with each of the five mechanics enumerated in playbook 5A,
  * excludes funds by the instrument's LONG NAME (backtest_data/etf_exclusions.json), not by
    a ticker regex - 146 of the 1353 vetted listings are funds and r/153's regex caught 54,
  * reports after tax, 30 seeds, median [min..max] + worst seed, on three windows.

Stages:  stage0  entry-mechanic bake-off + diagnostics (both universes)
         stage1  exit economics    trail x target x stop         (72 cells)
         stage2a base geometry     age x L x depth x RS          (256 cells)
         stage2b slots / sizing                                  (18 cells)
         stage3  null controls                                   (8 cells)
         stage4  gate bake-off                                   (12 cells)
         report  aggregate everything already on disk
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
HERE = ROOT / 'research/167_ipo_base_honest_reopt'
RES = HERE / 'results'
RES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'research/153_ipo_base/scripts'))
sys.path.insert(0, str(ROOT / 'research/142_bananapatterns_replication/scripts'))
sys.path.insert(0, str(ROOT))

import bluesky_replay as br            # noqa: E402

SEEDS = list(range(1, 31))
WINDOWS = {'w2': ('2006-01-01', '2026-09-04'),
           'wa': ('2006-01-01', '2015-12-31'),
           'wb': ('2016-01-01', '2026-09-04')}

# the r/153 adopted spec, verbatim
INCUMBENT = dict(max_age_m=6, min_bars=25, L=25, max_depth=0.30, rs_policy='off',
                 rs_min=70.0, tight_max=None, pivot_mode='close',
                 stop=0.08, trail=20, target=0.25, slots=8, size_pct=0.1875,
                 risk_pct=None, stop_mode='pct', cost=0.0025, gate=False)

NO_STOP = 0.99          # engine needs a number; 0.99 puts the stop at 1% of fill
MECHANICS = ('ref_lookahead', 'close_fill', 'nextday_pivot', 'nextday_candle',
             'resting_stop')


# ───────────────────────────────────────────── fund mask (by long name, not by ticker)
class _NameMask:
    """Duck-types re.Pattern.search so ipo_replay's `br.ETF_RE.search(col)` keeps working."""

    def __init__(self, symbols):
        self.s = set(symbols)

    def search(self, c):
        return True if c in self.s else None


_OLD_ETF_RE = br.ETF_RE


def clean_fund_symbols():
    """etf_exclusions.json (long-name derived) UNION r/142's old ticker regex."""
    ex = json.load(open(ROOT / 'backtest_data/etf_exclusions.json'))
    syms = set(ex['symbols'])
    return syms


def install_fund_mask(clean: bool):
    if clean:
        br.ETF_RE = _NameMask(clean_fund_symbols())
    else:
        br.ETF_RE = _OLD_ETF_RE


def load_ctx(clean=True, verbose=True):
    install_fund_mask(clean)
    import ipo_replay as ir
    ctx = ir.Ctx(verbose=verbose)
    # the old regex is a *ticker* test; keep it as a belt-and-braces union on the clean arm
    if clean:
        extra = [c for c in ctx.cols if _OLD_ETF_RE.search(c)]
        if extra:
            j = [ctx.cols.index(c) for c in extra]
            ctx.ELIG[:, j] = False
            if verbose:
                print('[ctx] +%d ticker-regex funds masked on top of the name list' % len(j),
                      flush=True)
    return ctx, ir


# ───────────────────────────────────────────── setup + the five entry mechanics
def _rsok(ctx, cfg):
    pol, rs_min = cfg['rs_policy'], cfg['rs_min']
    if pol == 'off':
        return np.ones_like(ctx.C, dtype=bool)
    if pol == 'strict':
        return np.nan_to_num(ctx.RSF, nan=-1.0) >= rs_min
    if pol == 'relaxed':
        return np.isnan(ctx.RSF) | (np.nan_to_num(ctx.RSF, nan=-1.0) >= rs_min)
    if pol == 'short':
        return np.nan_to_num(ctx.RSS, nan=-1.0) >= rs_min
    raise ValueError(pol)


def build_setup(ctx, cfg):
    """Everything except the breakout cross itself."""
    piv, lo = ctx.pivot(cfg['L'], cfg.get('pivot_mode', 'close'))
    with np.errstate(invalid='ignore', divide='ignore'):
        depth = (piv - lo) / np.where(piv > 0, piv, np.nan)
    young = ((ctx.AGE > 0) & (ctx.AGE <= cfg['max_age_m'] * 30.44)
             & (ctx.BARS >= cfg['min_bars']))
    setup = (young & (depth <= cfg['max_depth']) & (ctx.PREVC < piv)
             & ctx.ELIG & _rsok(ctx, cfg) & ~np.isnan(piv))
    if cfg.get('tight_max') is not None:
        setup &= np.nan_to_num(ctx.atrp(cfg['L']), nan=1e9) <= cfg['tight_max']
    return np.nan_to_num(setup, nan=False), piv, lo


def _shift(a):
    b = np.zeros_like(a) if a.dtype == bool else np.full_like(a, np.nan)
    b[1:] = a[:-1]
    return b


def apply_mechanic(ctx, setup, piv, lo, mode):
    """Returns (TRIG, LEVEL, BASELOW, fill_close). LEVEL is the price the order rests at."""
    C, H = ctx.C, ctx.H
    with np.errstate(invalid='ignore'):
        if mode == 'ref_lookahead':                 # r/153: LOOK-AHEAD, reference only
            return setup & (C > piv), piv, lo, False
        if mode == 'close_fill':                    # buy at the signal close
            return setup & (C > piv), piv, lo, True
        if mode == 'nextday_pivot':                 # THE LIVE MECHANIC
            t, pv, ln = _shift(setup & (C > piv)), _shift(piv), _shift(lo)
            return t & (H >= pv) & np.isfinite(pv), pv, ln, False
        if mode == 'nextday_candle':                # stop above the signal candle
            t, hs, ln = _shift(setup & (C > piv)), _shift(H), _shift(lo)
            return t & (H > hs) & np.isfinite(hs), hs, ln, False
        if mode == 'resting_stop':                  # every crossing, incl. the failures
            return setup & (H >= piv) & np.isfinite(piv), piv, lo, False
    raise ValueError(mode)


# ───────────────────────────────────────────── one cell
def days_of(ctx, win):
    return np.array([i for i, d in enumerate(ctx.dates) if win[0] <= str(d.date()) <= win[1]])


_DAYCACHE: dict = {}


def days_cached(ctx, wk):
    if wk not in _DAYCACHE:
        _DAYCACHE[wk] = days_of(ctx, WINDOWS[wk])
    return _DAYCACHE[wk]


def full_curve_dd_by_year(nav):
    """Intra-year max drawdown measured from the running peak of the FULL curve (r/154)."""
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return {int(y): round(100 * float(g.min()), 2) for y, g in dd.groupby(dd.index.year)}


def run_cell(ctx, ir, cfg, mechanic='nextday_pivot', seeds=SEEDS, windows=('w2', 'wa', 'wb'),
             trig=None, level=None, lo=None, keep=False, **over):
    if trig is None:
        setup, piv, lo0 = build_setup(ctx, cfg)
        trig, level, lo, fc = apply_mechanic(ctx, setup, piv, lo0, mechanic)
    else:
        fc = over.pop('fill_close', False)
    sma = ctx.sma(cfg['trail'])
    weak = ctx.WEAK if cfg.get('gate') else ctx.NOWEAK
    if cfg.get('gate') == 'custom':
        weak = cfg['weak_series']
    out = {'n_signals': int(trig.sum())}
    kept = {}
    for wk in windows:
        days = days_cached(ctx, wk)
        du = ctx.dates[days]
        rows, navs, alltr = [], [], []
        for sd in seeds:
            eq, trd, _, inv = ir.simulate_ipo(
                sd, days, ctx.dates, ctx.C, ctx.O, level, lo, sma, ctx.RSF, ctx.TVp,
                trig, weak, cost=cfg['cost'], stop=cfg['stop'], slots=cfg['slots'],
                size_pct=cfg['size_pct'], risk_pct=cfg.get('risk_pct'),
                stop_mode=cfg.get('stop_mode', 'pct'), target=cfg.get('target'),
                fill_close=fc, **over)
            st, e = ir.stats_from(eq, du, trd, invested=inv)
            rows.append(st)
            if keep and wk == 'w2':
                navs.append(e)
                alltr.append(trd)
        d = pd.DataFrame(rows)
        out[f'{wk}_cagr'] = round(float(d.cagr.median()), 2)
        out[f'{wk}_cagr_lo'] = round(float(d.cagr.min()), 2)
        out[f'{wk}_cagr_hi'] = round(float(d.cagr.max()), 2)
        out[f'{wk}_dd'] = round(float(d.dd.median()), 2)
        out[f'{wk}_dd_worst'] = round(float(d.dd.min()), 2)
        out[f'{wk}_calmar'] = round(float(d.cagr.median() / abs(d.dd.median()))
                                    if d.dd.median() else np.nan, 3)
        out[f'{wk}_n'] = int(d.n.median())
        out[f'{wk}_tpy'] = round(float(d.tpy.median()), 1)
        out[f'{wk}_win'] = round(float(d.win.median()), 1)
        out[f'{wk}_mean'] = round(float(d['mean'].median()), 3)
        out[f'{wk}_netexp'] = round(float(d['mean'].median()) - 200 * cfg['cost'], 3)
        out[f'{wk}_avg_win'] = round(float(d.avg_win.median()), 2)
        out[f'{wk}_avg_loss'] = round(float(d.avg_loss.median()), 2)
        out[f'{wk}_hold'] = round(float(d.hold.median()), 0)
        out[f'{wk}_inv'] = round(float(d.invested_pct.median()), 1)
        out[f'{wk}_streak'] = int(d.max_loss_streak.median())
        if keep and wk == 'w2':
            kept = dict(navs=navs, trades=alltr, stats=d)
    return (out, kept) if keep else (out, None)


CFGKEYS = ['max_age_m', 'min_bars', 'L', 'max_depth', 'rs_policy', 'rs_min', 'tight_max',
           'pivot_mode', 'stop', 'trail', 'target', 'slots', 'size_pct', 'risk_pct',
           'stop_mode', 'cost', 'gate']
METRICS = ('cagr', 'cagr_lo', 'cagr_hi', 'dd', 'dd_worst', 'calmar', 'n', 'tpy', 'win',
           'mean', 'netexp', 'avg_win', 'avg_loss', 'hold', 'inv', 'streak')
FIELDS = (['label', 'mechanic', 'universe'] + CFGKEYS + ['n_signals']
          + [f'{w}_{m}' for w in ('w2', 'wa', 'wb') for m in METRICS] + ['secs'])


def sweep(ctx, ir, cells, path, log_every=1):
    done = set()
    if path.exists():
        with open(path) as f:
            done = {r['label'] for r in csv.DictReader(f)}
        print(f'resuming {path.name}: {len(done)} cells already done', flush=True)
    else:
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    t0 = time.time()
    todo = [c for c in cells if c[0] not in done]
    print(f'{path.name}: {len(todo)} of {len(cells)} cells to run', flush=True)
    for i, (label, cfg, mech, uni) in enumerate(todo, 1):
        t = time.time()
        out, _ = run_cell(ctx, ir, cfg, mechanic=mech)
        row = {'label': label, 'mechanic': mech, 'universe': uni,
               **{k: cfg.get(k) for k in CFGKEYS}, **out,
               'secs': round(time.time() - t, 1)}
        with open(path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow({k: row.get(k) for k in FIELDS})
        if i % log_every == 0 or i == len(todo):
            el = time.time() - t0
            eta = el / i * (len(todo) - i) / 60
            print(f'[{i}/{len(todo)}] {label:<46} sig {out["n_signals"]:5d} | '
                  f'w2 {out["w2_cagr"]:6.2f}% (worst {out["w2_cagr_lo"]:6.2f}) '
                  f'DD {out["w2_dd"]:7.2f} Cal {out["w2_calmar"]:5.2f} '
                  f'inv {out["w2_inv"]:4.1f}% tpy {out["w2_tpy"]:5.1f} | '
                  f'wa {out["wa_cagr"]:6.2f} wb {out["wb_cagr"]:6.2f} '
                  f'| {time.time()-t:.0f}s ETA {eta:.0f}m', flush=True)


# ───────────────────────────────────────────── stage 0
def stage0(ctx, ir):
    global _DAYCACHE
    path = RES / 'stage0_mechanics.csv'
    cells = [(f'm_{m}__clean', dict(INCUMBENT), m, 'clean') for m in MECHANICS]
    sweep(ctx, ir, cells, path)
    # contaminated universe, for the size of the fund defect
    print('\n--- rebuilding the panel on the r/153 (ticker-regex) universe ---', flush=True)
    ctx2, ir2 = load_ctx(clean=False)
    cells2 = [(f'm_{m}__r153univ', dict(INCUMBENT), m, 'r153_tickerregex')
              for m in MECHANICS]
    _DAYCACHE = {}
    sweep(ctx2, ir2, cells2, path)
    _DAYCACHE = {}
    del ctx2
    return path


def stage0b(ctx, ir):
    """Corporate-action exposure: r/153's engine has no data-event guard, the live book does."""
    cfg = dict(INCUMBENT)
    setup, piv, lo0 = build_setup(ctx, cfg)
    trig, lvl, lo, fc = apply_mechanic(ctx, setup, piv, lo0, 'nextday_pivot')
    out, kept = run_cell(ctx, ir, cfg, trig=trig, level=lvl, lo=lo, keep=True,
                         windows=('w2',), fill_close=fc)
    tr = pd.DataFrame([x for t in kept['trades'] for x in t])
    tr['seed'] = np.repeat(SEEDS, [len(t) for t in kept['trades']])
    cols = np.array(ctx.cols)
    tr['symbol'] = cols[tr['col'].values]
    dd = {}
    dd['trades_total_median'] = float(tr.groupby('seed').size().median())
    for thr in (-0.40, -0.50, -0.70):
        sel = tr[tr.ret <= thr]
        dd[f'trades_below_{int(thr*100)}pct'] = int(len(sel))
        dd[f'frac_below_{int(thr*100)}pct'] = round(100 * len(sel) / len(tr), 3)
    dd['worst_20_trades'] = (tr.nsmallest(20, 'ret')[['symbol', 'ret', 'held', 'reason']]
                             .assign(ret=lambda x: (100 * x.ret).round(1))
                             .to_dict('records'))
    # one-day close collapse on the exit bar => almost certainly a split/bonus, not a fall
    C = ctx.C
    bad = []
    for _, r in tr[tr.ret <= -0.40].iterrows():
        xi, c = int(r.xi), int(r.col)
        if xi > 0 and np.isfinite(C[xi, c]) and np.isfinite(C[xi - 1, c]):
            if C[xi, c] / C[xi - 1, c] - 1 <= -0.40:
                bad.append(dict(symbol=r.symbol, date=str(ctx.dates[xi].date()),
                                prev=round(float(C[xi - 1, c]), 2),
                                px=round(float(C[xi, c]), 2), ret=round(100 * r.ret, 1)))
    dd['one_day_collapse_exits'] = bad
    dd['n_one_day_collapse_exits'] = len(bad)
    dd['invested_pct_median'] = out['w2_inv']
    dd['cash_pct_median'] = round(100 - out['w2_inv'], 1)
    dd['cash_sweep_pp_estimate'] = round(5.0 * (100 - out['w2_inv']) / 100.0, 2)
    # same spec, no cash yield -> the sweep's real contribution
    out0, _ = run_cell(ctx, ir, cfg, trig=trig, level=lvl, lo=lo, windows=('w2',),
                       fill_close=fc, cash_yield=0.0)
    dd['w2_cagr_with_cash_yield'] = out['w2_cagr']
    dd['w2_cagr_zero_cash_yield'] = out0['w2_cagr']
    dd['cash_sweep_pp_measured'] = round(out['w2_cagr'] - out0['w2_cagr'], 2)
    json.dump(dd, open(RES / 'diagnostics.json', 'w'), indent=2, default=str)
    print(json.dumps({k: v for k, v in dd.items() if k not in
                      ('worst_20_trades', 'one_day_collapse_exits')}, indent=2), flush=True)
    return dd


# ───────────────────────────────────────────── stage 1 - exit economics
def cells_stage1():
    out = []
    for trail, tgt, stop in itertools.product((10, 15, 20, 30, 50, 75, 100, 150),
                                              (0.25, 0.50, 1.00, None),
                                              (0.06, 0.08, 0.10, 0.15, None)):
        cfg = dict(INCUMBENT)
        cfg.update(trail=trail, target=tgt, stop=NO_STOP if stop is None else stop)
        lab = (f's1_tr{trail}_'
               f'{"tp"+str(int(tgt*100)) if tgt else "notp"}_'
               f'{"sl"+str(int(stop*100)) if stop else "nosl"}')
        out.append((lab, cfg, 'nextday_pivot', 'clean'))
    return out


# ───────────────────────────────────────────── stage 2 - geometry, then book
def best_exits(n=1):
    df = pd.read_csv(RES / 'stage1_exits.csv')
    df = df[(df.wa_netexp > 0) & (df.wb_netexp > 0) & (df.w2_cagr_lo > 0)]
    if not len(df):
        df = pd.read_csv(RES / 'stage1_exits.csv')
    df = df.sort_values('w2_cagr', ascending=False)
    return df.head(n)


def cells_stage2a():
    e = best_exits(1).iloc[0]
    out = []
    for age, L, dep, pol in itertools.product((3, 6, 12, 24), (15, 25, 40, 60),
                                              (0.20, 0.30, 0.40, 0.60),
                                              ('off', 'short70')):
        cfg = dict(INCUMBENT)
        cfg.update(trail=int(e.trail), target=None if pd.isna(e.target) else float(e.target),
                   stop=float(e.stop), max_age_m=age, L=L, max_depth=dep,
                   min_bars=max(25, L))
        if pol == 'short70':
            cfg.update(rs_policy='short', rs_min=70.0)
        else:
            cfg.update(rs_policy='off', rs_min=70.0)
        out.append((f's2a_a{age}_L{L}_d{int(dep*100)}_{pol}', cfg, 'nextday_pivot', 'clean'))
    return out


def cells_stage2b():
    g = pd.read_csv(RES / 'stage2a_geometry.csv')
    g = g[(g.wa_netexp > 0) & (g.wb_netexp > 0)].sort_values('w2_cagr', ascending=False)
    tops = g.head(3)
    out = []
    for _, r in tops.iterrows():
        for slots, size in ((5, 0.20), (8, 0.125), (8, 0.1875), (10, 0.10),
                            (12, 0.0833), (16, 0.0625)):
            cfg = {k: (None if (isinstance(r[k], float) and pd.isna(r[k])) else r[k])
                   for k in CFGKEYS}
            cfg['trail'] = int(cfg['trail']); cfg['L'] = int(cfg['L'])
            cfg['slots'] = slots; cfg['size_pct'] = size
            cfg['gate'] = False
            out.append((f's2b_{r.label[4:]}_n{slots}x{int(size*10000)}',
                        cfg, 'nextday_pivot', 'clean'))
    return out


# ───────────────────────────────────────────── stage 3 - nulls
def stage3(ctx, ir):
    path = RES / 'stage3_nulls.csv'
    rows = []
    g = pd.read_csv(RES / 'stage2a_geometry.csv')
    gg = g[(g.wa_netexp > 0) & (g.wb_netexp > 0)].sort_values('w2_cagr', ascending=False)
    src = gg.iloc[0] if len(gg) else g.sort_values('w2_cagr', ascending=False).iloc[0]
    best = {k: (None if (isinstance(src[k], float) and pd.isna(src[k])) else src[k])
            for k in CFGKEYS}
    best['trail'] = int(best['trail']); best['L'] = int(best['L'])
    best['min_bars'] = int(best['min_bars']); best['max_age_m'] = int(best['max_age_m'])
    best['gate'] = False
    specs = [('incumbent', dict(INCUMBENT)), ('refit', best)]
    with open(path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS + ['arm', 'spec',
                                               'paired_delta_med', 'real_wins']).writeheader()

    def emit(label, out, cfg, arm, spec, pd_=None, wins=None):
        row = {'label': label, 'mechanic': 'nextday_pivot', 'universe': 'clean',
               **{k: cfg.get(k) for k in CFGKEYS}, **out, 'secs': 0,
               'arm': arm, 'spec': spec, 'paired_delta_med': pd_, 'real_wins': wins}
        with open(path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS + ['arm', 'spec', 'paired_delta_med',
                                                   'real_wins']).writerow(
                {k: row.get(k) for k in FIELDS + ['arm', 'spec', 'paired_delta_med',
                                                  'real_wins']})
        rows.append(row)

    for spec, cfg in specs:
        setup, piv, lo0 = build_setup(ctx, cfg)
        trig_real, lvl, lo, fc = apply_mechanic(ctx, setup, piv, lo0, 'nextday_pivot')
        young = ((ctx.AGE > 0) & (ctx.AGE <= cfg['max_age_m'] * 30.44)
                 & (ctx.BARS >= cfg['min_bars']) & ctx.ELIG)

        # ---- NULL A: name-selection null. BOTH arms fill at the next-day OPEN, no level.
        #      Isolates "which names were chosen" from "what price was paid".
        O = ctx.O
        openlvl = np.where(np.isfinite(O), 0.0, np.nan).astype('float32')  # level below open
        trig_openA = _shift(setup & (ctx.C > piv))      # signal last night, buy next open
        outA_real, kA = run_cell(ctx, ir, cfg, trig=trig_openA, level=openlvl, lo=lo,
                                 keep=True, windows=('w2',), fill_close=False)
        emit(f'n3_{spec}_A_real', outA_real, cfg, 'real_nextopen', spec)
        rng = np.random.default_rng(20260912)
        nperA = trig_openA.sum(axis=1)
        nper = trig_real.sum(axis=1)
        nullA = np.zeros_like(trig_real)
        yshift0 = _shift(young)
        for i in np.nonzero(nperA)[0]:
            pool = np.nonzero(yshift0[i] & np.isfinite(O[i]))[0]
            if not len(pool):
                continue
            k = min(int(nperA[i]), len(pool))
            nullA[i, rng.choice(pool, size=k, replace=False)] = True
        outA_null, kAn = run_cell(ctx, ir, cfg, trig=nullA, level=openlvl, lo=lo,
                                  keep=True, windows=('w2',), fill_close=False)
        dlt = kA['stats'].cagr.values - kAn['stats'].cagr.values
        emit(f'n3_{spec}_A_null', outA_null, cfg, 'null_random_nextopen', spec,
             round(float(np.median(dlt)), 2), int((dlt > 0).sum()))

        # ---- NULL B: structure null. Both arms carry a next-day buy-stop at the name's own
        #      L-bar pivot and only fill if the high reaches it. The ONLY difference is
        #      whether last night's close cleared the base high.
        pvn, lon = _shift(piv), _shift(lo)
        with np.errstate(invalid='ignore'):
            reach = ctx.H >= pvn
        outB_real, kB = run_cell(ctx, ir, cfg, trig=trig_real, level=pvn, lo=lon,
                                 keep=True, windows=('w2',), fill_close=False)
        emit(f'n3_{spec}_B_real', outB_real, cfg, 'real_nextday_pivot', spec)
        yshift = _shift(young)
        nullB = np.zeros_like(trig_real)
        for i in np.nonzero(nper)[0]:
            pool = np.nonzero(yshift[i] & reach[i] & np.isfinite(pvn[i]))[0]
            if not len(pool):
                continue
            k = min(int(nper[i]), len(pool))
            nullB[i, rng.choice(pool, size=k, replace=False)] = True
        outB_null, kBn = run_cell(ctx, ir, cfg, trig=nullB, level=pvn, lo=lon,
                                  keep=True, windows=('w2',), fill_close=False)
        dltB = kB['stats'].cagr.values - kBn['stats'].cagr.values
        emit(f'n3_{spec}_B_null', outB_null, cfg, 'null_random_nextday_pivot', spec,
             round(float(np.median(dltB)), 2), int((dltB > 0).sum()))

        # ---- cash null: the same book with every signal suppressed (pure 5% sweep)
        outC, _ = run_cell(ctx, ir, cfg, trig=np.zeros_like(trig_real), level=pvn, lo=lon,
                           windows=('w2',), fill_close=False)
        emit(f'n3_{spec}_C_cash_only', outC, cfg, 'cash_only_5pct', spec)

    # ---- cohort drift null: equal-weight hold of every young+liquid name (gross)
    cfg = dict(INCUMBENT)
    young = ((ctx.AGE > 0) & (ctx.AGE <= cfg['max_age_m'] * 30.44)
             & (ctx.BARS >= cfg['min_bars']) & ctx.ELIG)
    cl = pd.DataFrame(ctx.C, index=ctx.dates, columns=ctx.cols)
    rets = cl.pct_change()
    mask = pd.DataFrame(young, index=ctx.dates, columns=ctx.cols).shift(1).fillna(False)
    coh = rets.where(mask).mean(axis=1).fillna(0.0)
    d = days_cached(ctx, 'w2')
    cser = (1 + coh.iloc[d]).cumprod()
    yrs = (cser.index[-1] - cser.index[0]).days / 365.25
    coh_cagr = 100 * ((cser.iloc[-1]) ** (1 / yrs) - 1)
    coh_dd = 100 * float((cser / cser.cummax() - 1).min())
    json.dump(dict(cohort_cagr_gross=round(float(coh_cagr), 2), cohort_dd=round(float(coh_dd), 2)),
              open(RES / 'cohort_null.json', 'w'), indent=2)
    print(f'\ncohort equal-weight young+liquid drift (GROSS, no cost/tax): '
          f'CAGR {coh_cagr:.2f}%  DD {coh_dd:.2f}%', flush=True)
    print('\nstage3 done ->', path, flush=True)


# ───────────────────────────────────────────── stage 4 - gates
def stage4(ctx, ir):
    path = RES / 'stage4_gates.csv'
    g = pd.read_csv(RES / 'stage2b_book.csv') if (RES / 'stage2b_book.csv').exists() else None
    if g is None or not len(g):
        g = pd.read_csv(RES / 'stage2a_geometry.csv')
    gg = g[(g.wa_netexp > 0) & (g.wb_netexp > 0)].sort_values('w2_cagr', ascending=False)
    src = gg.iloc[0] if len(gg) else g.sort_values('w2_cagr', ascending=False).iloc[0]
    best = {k: (None if (isinstance(src[k], float) and pd.isna(src[k])) else src[k])
            for k in CFGKEYS}
    for k in ('trail', 'L', 'min_bars', 'max_age_m', 'slots'):
        best[k] = int(best[k])
    best['gate'] = False
    print('gate bake-off base spec:', json.dumps({k: best[k] for k in CFGKEYS},
                                                 default=str), flush=True)

    nb = ctx.close.get('NIFTYBEES').dropna()
    gates = {'none': np.zeros(len(ctx.dates), dtype=bool)}
    for n in (100, 150, 200):
        w = (nb < nb.rolling(n).mean()).shift(1)
        gates[f'nbees_sma{n}'] = w.reindex(ctx.dates).ffill().fillna(False).to_numpy(bool)
    for thr in (0.05, 0.10, 0.15, 0.20):
        w = (nb / nb.cummax() - 1 <= -thr).shift(1)
        gates[f'nbees_dd{int(thr*100)}'] = (w.reindex(ctx.dates).ffill().fillna(False)
                                            .to_numpy(bool))
    for n in (63, 126, 252):
        w = (nb / nb.shift(n) - 1 < 0).shift(1)
        gates[f'nbees_mom{n}neg'] = (w.reindex(ctx.dates).ffill().fillna(False)
                                     .to_numpy(bool))
    cells = []
    for name, series in gates.items():
        cfg = dict(best)
        cfg['gate'] = 'custom'
        cfg['weak_series'] = series
        cells.append((f's4_{name}', cfg, 'nextday_pivot', 'clean'))
    sweep(ctx, ir, cells, path)

    # paired deltas vs the no-gate arm
    df = pd.read_csv(path)
    base = df[df.label == 's4_none']
    if len(base):
        print('\n--- gate vs none (median CAGR delta, W2 after tax) ---', flush=True)
        b = float(base.iloc[0].w2_cagr)
        for _, r in df.iterrows():
            print(f'  {r.label:<22} {r.w2_cagr:6.2f}%  ({r.w2_cagr-b:+5.2f}pp)  '
                  f'DD {r.w2_dd:7.2f}  Cal {r.w2_calmar:5.2f}  inv {r.w2_inv:4.1f}%',
                  flush=True)


# ───────────────────────────────────────────── report
def report():
    print('\n' + '=' * 100)
    for nm in ('stage0_mechanics', 'stage1_exits', 'stage2a_geometry', 'stage2b_book',
               'stage3_nulls', 'stage4_gates'):
        p = RES / f'{nm}.csv'
        if not p.exists():
            print(f'{nm}: MISSING')
            continue
        df = pd.read_csv(p)
        print(f'\n### {nm}  ({len(df)} cells)')
        cols = ['label', 'w2_cagr', 'w2_cagr_lo', 'w2_dd', 'w2_calmar', 'w2_inv',
                'w2_tpy', 'w2_win', 'w2_netexp', 'wa_cagr', 'wb_cagr']
        cols = [c for c in cols if c in df.columns]
        print(df.sort_values('w2_cagr', ascending=False)[cols].head(25).to_string(index=False))
    p = RES / 'stage1_exits.csv'
    if p.exists():
        df = pd.read_csv(p)
        for tgt in sorted(df.target.fillna(-1).unique()):
            sub = df[df.target.fillna(-1) == tgt]
            print(f'\n--- stage1 surface: W2 after-tax CAGR, target='
                  f'{"none" if tgt < 0 else tgt} ---')
            print(sub.pivot_table(index='stop', columns='trail',
                                  values='w2_cagr').round(2).to_string())


# ───────────────────────────────────────────── main
if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what == 'report':
        report()
        sys.exit(0)
    t0 = time.time()
    ctx, ir = load_ctx(clean=True)
    print(f'[ctx] clean universe ready in {time.time()-t0:.0f}s, '
          f'{len(ctx.cols)} tradeable young-listing symbols', flush=True)
    if what in ('all', 'stage0'):
        stage0b(ctx, ir)
        stage0(ctx, ir)
    if what in ('all', 'stage1'):
        sweep(ctx, ir, cells_stage1(), RES / 'stage1_exits.csv')
    if what in ('all', 'stage2a'):
        sweep(ctx, ir, cells_stage2a(), RES / 'stage2a_geometry.csv')
    if what in ('all', 'stage2b'):
        sweep(ctx, ir, cells_stage2b(), RES / 'stage2b_book.csv')
    if what in ('all', 'stage3'):
        stage3(ctx, ir)
    if what in ('all', 'stage4'):
        stage4(ctx, ir)
    if what == 'all':
        report()
    print(f'\nALL DONE in {(time.time()-t0)/60:.1f} min', flush=True)
