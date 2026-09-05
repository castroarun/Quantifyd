"""research/152 — phase runner for the Multi-Year Breakout study. Resume-safe.

  --phase A   signal inventory + Open-Alpha overlap (no book)          72 matrices
  --phase B   G1 sweep: default book, 10 seeds, both windows           72 cells x 2 windows
  --phase C   G2 mechanics OFAT on the survivors named by --families
  --phase D   30-seed adoption run for one spec (--spec ...)

Every runner appends one row per completed cell immediately and skips rows already present.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import myb_replay as M                                   # noqa: E402

RES = Path(__file__).resolve().parents[1] / 'results'
RES.mkdir(exist_ok=True)

W1 = ('2020-01-01', '2025-12-31')            # the site's window
W2 = ('2010-01-01', '2026-09-04')            # longest window the data supports (N<=5)
W2B = ('2015-01-01', '2026-09-04')           # N=10 only (history-starved before 2015)

NS = [2, 3, 5, 10]
LEVELS = ['close', 'high']
ATHV = ['incl', 'excl', 'athonly']
AGES = [0, 6, 12]

DEF = dict(stop=0.08, slots=16, size_pct=0.0625, trail=50, cost=0.0025,
           gate=False, fill_close=False, take_profit=None, risk_pct=None)


# ───────────────────────── helpers ─────────────────────────
def days_for(dates, win):
    a, b = win
    return np.array([i for i, d in enumerate(dates) if a <= str(d.date()) <= b])


def append_row(path, row, fields):
    new = not path.exists()
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def done_keys(path, keycols):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if not len(df):
        return set()
    return {tuple(str(r[c]) for c in keycols) for _, r in df.iterrows()}


def bench(close, dates, win):
    nb = close['NIFTYBEES'].dropna()
    s = nb[(nb.index >= win[0]) & (nb.index <= win[1])]
    if len(s) < 60:
        return np.nan, np.nan
    y = (s.index[-1] - s.index[0]).days / 365.25
    return ((s.iloc[-1] / s.iloc[0]) ** (1 / y) - 1) * 100, float((s / s.cummax() - 1).min() * 100)


def ens(rows, key):
    v = [r[key] for r in rows]
    return float(np.median(v)), float(min(v)), float(max(v))


# ───────────────────────── context ─────────────────────────
class Ctx:
    def __init__(self, trail_list=(50,)):
        w = M.load_wide()
        self.w = w
        self.cm = M.common(w)
        self.close = self.cm['close']
        self.dates = self.close.index
        self.C = self.close.values
        self.O = w['open'].values
        self.trails = {t: self.close.rolling(t).mean().values.astype('float32')
                       for t in trail_list}
        self.OA_TRIG, self.OA_PIV = M.oa_signal(self.cm)
        self.events = M.split_blackout_events(self.close)
        print(f'split-scale events detected: {len(self.events)}', flush=True)
        self.weak_off = np.zeros(len(self.dates), dtype=bool)
        self.weak_on = M.weak_gate(self.close, self.dates, on=True)
        self.days = {}

    def d(self, win):
        if win not in self.days:
            self.days[win] = days_for(self.dates, win)
        return self.days[win]

    def trail(self, t):
        if t not in self.trails:
            self.trails[t] = self.close.rolling(t).mean().values.astype('float32')
        return self.trails[t]


def signal_cache(ctx, n, level):
    """Per-(N, level) heavy pieces, reused by the 9 (athvar x age) sub-cells."""
    src = ctx.close if level == 'close' else ctx.cm['high']
    W = int(n * M.YR)
    # min_periods=1 is DELIBERATE: on a union-index wide frame, min_periods=W demands
    # zero missing rows in W trading days, which silently voided almost every long
    # lookback (N=10 collapsed to 1 symbol). The genuine "N years of history"
    # requirement is enforced separately by the nrows_ok mask, which counts a symbol's
    # own non-NaN rows. (Playbook: NaN-robust indicators are mandatory.)
    piv = src.shift(1).rolling(W, min_periods=1).max()
    olds = {}
    for a in AGES:
        if a == 0:
            continue
        X = int(round(a * M.YR / 12))
        olds[a] = src.shift(1 + X).rolling(W - X, min_periods=1).max()
    bo = M.blackout_mask(ctx.C.shape, ctx.events, W)
    return piv, olds, bo


def build_trig(ctx, piv, olds, bo, athvar, age, maxdist=0.20, tight=None,
               apply_blackout=True):
    close = ctx.close
    prev_close = close.shift(1)
    W_ok = piv.notna()
    ok = ctx.cm['elig'] & ctx.cm['rs_ok'] & W_ok
    ok &= (prev_close < piv) & (prev_close >= (1 - maxdist) * piv)
    # min-history is applied by the caller via the nrows_ok mask (needs N)
    if age:
        ok &= (olds[age] >= piv)
    athp = ctx.cm['athp']
    if athvar == 'excl':
        ok &= (piv < athp) & (close <= athp)
    elif athvar == 'athonly':
        ok &= (piv >= 0.999 * athp)
    if tight:
        rng = ctx.cm['high'].rolling(60).max() / ctx.cm['low'].rolling(60).min() - 1
        ok &= (rng.shift(1) <= tight)
    trig = (ok & (close > piv)).fillna(False).values
    if apply_blackout:
        trig = trig & bo
    return trig


# ───────────────────────── phases ─────────────────────────
FIELDS_A = ['cell', 'n_years', 'level', 'athvar', 'age_m', 'window', 'signals_raw',
            'signals', 'blackout_cost_pct', 'signals_yr', 'uniq_syms',
            'oa_overlap_pct', 'oa_share_of_oa_pct']


def phase_a(ctx):
    path = RES / 'phaseA_signals.csv'
    done = done_keys(path, ['cell', 'window'])
    for n in NS:
        for level in LEVELS:
            t0 = time.time()
            piv, olds, bo = signal_cache(ctx, n, level)
            nrows_ok = (ctx.cm['nrows'] >= int(n * M.YR))
            for athvar in ATHV:
                for age in AGES:
                    cell = f'N{n}_{level}_{athvar}_age{age}'
                    if all((cell, wn) in done for wn in ('W2', 'W1')):
                        continue
                    trig_raw = build_trig(ctx, piv, olds, bo, athvar, age,
                                          apply_blackout=False) & nrows_ok.values
                    trig = trig_raw & bo
                    for wname, win in (('W2', W2 if n < 10 else W2B), ('W1', W1)):
                        if (cell, wname) in done:
                            continue
                        idx = ctx.d(win)
                        sr = int(trig_raw[idx].sum())
                        s = int(trig[idx].sum())
                        yrs = (ctx.dates[idx[-1]] - ctx.dates[idx[0]]).days / 365.25
                        sub = trig[idx]
                        uniq = int((sub.sum(axis=0) > 0).sum())
                        oa = ctx.OA_TRIG[idx]
                        inter = int((sub & oa).sum())
                        append_row(path, dict(
                            cell=cell, n_years=n, level=level, athvar=athvar, age_m=age,
                            window=wname, signals_raw=sr, signals=s,
                            blackout_cost_pct=round(100 * (sr - s) / sr, 2) if sr else 0,
                            signals_yr=round(s / yrs, 1), uniq_syms=uniq,
                            oa_overlap_pct=round(100 * inter / s, 1) if s else 0,
                            oa_share_of_oa_pct=round(100 * inter / max(1, int(oa.sum())), 1)),
                            FIELDS_A)
            print(f'[A] N={n} {level} done ({time.time()-t0:.0f}s)', flush=True)
    print('PHASE A DONE', flush=True)


FIELDS_B = ['cell', 'n_years', 'level', 'athvar', 'age_m', 'window', 'start', 'end',
            'signals', 'seeds', 'cagr_med', 'cagr_min', 'cagr_max', 'dd_med', 'dd_worst',
            'calmar_med', 'x_med', 'trades_med', 'trades_yr', 'win_med', 'mean_tr',
            'avg_win', 'avg_loss', 'streak', 'avg_hold', 'bench_cagr', 'elapsed_s']


def run_cell(ctx, trig, win, seeds=10, **kw):
    p = dict(DEF); p.update(kw)
    idx = ctx.d(win)
    dates_used = ctx.dates[idx]
    weak = ctx.weak_on if p['gate'] else ctx.weak_off
    trail = ctx.trail(p['trail']) if p['trail'] else None
    out = []
    for seed in range(1, seeds + 1):
        eq, tr, _ = M.simulate_ext(
            seed, idx, ctx.dates, ctx.C, ctx.O, ctx.PIV_CUR, trail, trig, weak,
            cost=p['cost'], stop=p['stop'], slots=p['slots'], size_pct=p['size_pct'],
            risk_pct=p['risk_pct'], take_profit=p['take_profit'],
            fill_close=p['fill_close'])
        st, e = M.stats_from(eq, dates_used, tr, dates=ctx.dates)
        st['_nav'] = e
        out.append(st)
    return out


def phase_b(ctx, seeds=10):
    path = RES / 'phaseB_g1.csv'
    done = done_keys(path, ['cell', 'window'])
    bmk = {}
    for n in NS:
        for level in LEVELS:
            piv, olds, bo = signal_cache(ctx, n, level)
            ctx.PIV_CUR = piv.values.astype('float32')
            nrows_ok = (ctx.cm['nrows'] >= int(n * M.YR)).values
            for athvar in ATHV:
                for age in AGES:
                    cell = f'N{n}_{level}_{athvar}_age{age}'
                    trig = None
                    for wname, win in (('W2', W2 if n < 10 else W2B), ('W1', W1)):
                        if (cell, wname) in done:
                            continue
                        if trig is None:
                            trig = build_trig(ctx, piv, olds, bo, athvar, age) & nrows_ok
                        t0 = time.time()
                        if win not in bmk:
                            bmk[win] = bench(ctx.close, ctx.dates, win)
                        idx = ctx.d(win)
                        nsig = int(trig[idx].sum())
                        if nsig < 5:
                            append_row(path, dict(cell=cell, n_years=n, level=level,
                                                  athvar=athvar, age_m=age, window=wname,
                                                  start=win[0], end=win[1], signals=nsig,
                                                  seeds=0, bench_cagr=round(bmk[win][0], 2),
                                                  elapsed_s=0), FIELDS_B)
                            continue
                        rows = run_cell(ctx, trig, win, seeds=seeds)
                        cm_, cmn, cmx = ens(rows, 'cagr')
                        dm, dmn, _ = ens(rows, 'dd')
                        append_row(path, dict(
                            cell=cell, n_years=n, level=level, athvar=athvar, age_m=age,
                            window=wname, start=win[0], end=win[1], signals=nsig, seeds=seeds,
                            cagr_med=round(cm_, 2), cagr_min=round(cmn, 2),
                            cagr_max=round(cmx, 2), dd_med=round(dm, 2),
                            dd_worst=round(dmn, 2),
                            calmar_med=round(float(np.median([r['calmar'] for r in rows])), 2),
                            x_med=round(float(np.median([r['x'] for r in rows])), 2),
                            trades_med=int(np.median([r['n'] for r in rows])),
                            trades_yr=round(float(np.median([r['trades_yr'] for r in rows])), 1),
                            win_med=round(float(np.median([r['win'] for r in rows])), 1),
                            mean_tr=round(float(np.median([r['mean'] for r in rows])), 3),
                            avg_win=round(float(np.median([r['avg_win'] for r in rows])), 2),
                            avg_loss=round(float(np.median([r['avg_loss'] for r in rows])), 2),
                            streak=int(np.median([r['max_lose_streak'] for r in rows])),
                            avg_hold=round(float(np.median([r['avg_hold'] for r in rows])), 1),
                            bench_cagr=round(bmk[win][0], 2),
                            elapsed_s=round(time.time() - t0, 1)), FIELDS_B)
                        print(f'[B] {cell:26s} {wname} sig={nsig:6d} '
                              f'CAGR {cm_:6.2f}% [{cmn:.1f}..{cmx:.1f}] DD {dm:6.1f}% '
                              f'({time.time()-t0:.0f}s)', flush=True)
    print('PHASE B DONE', flush=True)



# ───────────────────────── phase C: mechanics OFAT ─────────────────────────
FIELDS_C = ['family', 'arm', 'axis', 'value', 'window', 'signals', 'seeds', 'cagr_med',
            'cagr_min', 'cagr_max', 'dd_med', 'dd_worst', 'calmar_med', 'x_med',
            'trades_med', 'trades_yr', 'win_med', 'mean_tr', 'avg_win', 'avg_loss',
            'streak', 'avg_hold', 'bench_cagr', 'elapsed_s']

BASE_C = dict(stop=0.08, trail=50, take_profit=None, slots=16, size_pct=0.0625,
              risk_pct=None, gate=False, fill_close=False, cost=0.0025,
              maxdist=0.20, tight=None)


def c_arms():
    """OFAT arms: (arm_label, axis, value, overrides)."""
    arms = [('base', 'base', 'base', {})]
    for v in (0.07, 0.10):
        arms.append((f'stop{int(v*100)}', 'stop', v, dict(stop=v)))
    for t in (10, 15, 20, 25, 30, 150):        # 10/20/25 are the PLATEAU neighbours of 15
        arms.append((f'trail{t}', 'exit', f'trail{t}', dict(trail=t)))
    arms.append(('tp25_notrail', 'exit', 'tp25', dict(take_profit=0.25, trail=None)))
    arms.append(('tp25_trail50', 'exit', 'tp25+trail50', dict(take_profit=0.25)))
    for sl, sz in ((3, 0.30), (5, 0.20), (8, 0.1875), (10, 0.10)):
        arms.append((f'fixed{sl}x{int(sz*100)}', 'sizing', f'{sl}x{sz}',
                     dict(slots=sl, size_pct=sz)))
    for r in (0.01, 0.015, 0.02):
        arms.append((f'risk{r*100:g}pct', 'sizing', f'risk{r*100:g}%', dict(risk_pct=r)))
    arms.append(('gateON', 'gate', 'on', dict(gate=True)))
    arms.append(('fillclose', 'fill', 'close', dict(fill_close=True)))
    for md in (0.12, 0.30):
        arms.append((f'maxdist{int(md*100)}', 'basequality', f'maxdist{md}', dict(maxdist=md)))
    for tg in (0.35, 0.25):
        arms.append((f'tight{int(tg*100)}', 'basequality', f'tight{tg}', dict(tight=tg)))
    for c in (0.0040, 0.0060):
        arms.append((f'cost{int(c*10000)}', 'cost', f'{int(c*10000)}bps', dict(cost=c)))
    return arms


def parse_family(fam):
    import re as _re
    m = _re.match(r'N(\d+)_(close|high)_(incl|excl|athonly)_age(\d+)$', fam)
    assert m, f'bad family {fam}'
    return int(m.group(1)), m.group(2), m.group(3), int(m.group(4))


def phase_c(ctx, families, seeds=10):
    path = RES / 'phaseC_g2.csv'
    done = done_keys(path, ['family', 'arm', 'window'])
    bmk = {}
    for fam in families:
        n, level, athvar, age = parse_family(fam)
        piv, olds, bo = signal_cache(ctx, n, level)
        ctx.PIV_CUR = piv.values.astype('float32')
        nrows_ok = (ctx.cm['nrows'] >= int(n * M.YR)).values
        trig_cache = {}
        for arm, axis, value, ov in c_arms():
            p = dict(BASE_C); p.update(ov)
            tkey = (p['maxdist'], p['tight'])
            if tkey not in trig_cache:
                trig_cache[tkey] = build_trig(ctx, piv, olds, bo, athvar, age,
                                              maxdist=p['maxdist'],
                                              tight=p['tight']) & nrows_ok
            trig = trig_cache[tkey]
            for wname, win in (('W2', W2 if n < 10 else W2B), ('W1', W1)):
                if (fam, arm, wname) in done:
                    continue
                if win not in bmk:
                    bmk[win] = bench(ctx.close, ctx.dates, win)
                t0 = time.time()
                idx = ctx.d(win)
                nsig = int(trig[idx].sum())
                if nsig < 5:
                    continue
                rows = run_cell(ctx, trig, win, seeds=seeds, stop=p['stop'],
                                trail=p['trail'], take_profit=p['take_profit'],
                                slots=p['slots'], size_pct=p['size_pct'],
                                risk_pct=p['risk_pct'], gate=p['gate'],
                                fill_close=p['fill_close'], cost=p['cost'])
                cm_, cmn, cmx = ens(rows, 'cagr')
                dm, dmn, _ = ens(rows, 'dd')
                append_row(path, dict(
                    family=fam, arm=arm, axis=axis, value=value, window=wname,
                    signals=nsig, seeds=seeds, cagr_med=round(cm_, 2),
                    cagr_min=round(cmn, 2), cagr_max=round(cmx, 2), dd_med=round(dm, 2),
                    dd_worst=round(dmn, 2),
                    calmar_med=round(float(np.median([r['calmar'] for r in rows])), 2),
                    x_med=round(float(np.median([r['x'] for r in rows])), 2),
                    trades_med=int(np.median([r['n'] for r in rows])),
                    trades_yr=round(float(np.median([r['trades_yr'] for r in rows])), 1),
                    win_med=round(float(np.median([r['win'] for r in rows])), 1),
                    mean_tr=round(float(np.median([r['mean'] for r in rows])), 3),
                    avg_win=round(float(np.median([r['avg_win'] for r in rows])), 2),
                    avg_loss=round(float(np.median([r['avg_loss'] for r in rows])), 2),
                    streak=int(np.median([r['max_lose_streak'] for r in rows])),
                    avg_hold=round(float(np.median([r['avg_hold'] for r in rows])), 1),
                    bench_cagr=round(bmk[win][0], 2),
                    elapsed_s=round(time.time() - t0, 1)), FIELDS_C)
                print(f'[C] {fam:24s} {arm:14s} {wname} CAGR {cm_:6.2f}% '
                      f'[{cmn:.1f}..{cmx:.1f}] DD {dm:6.1f}% Cal '
                      f'{np.median([r["calmar"] for r in rows]):.2f} '
                      f'({time.time()-t0:.0f}s)', flush=True)
    print('PHASE C DONE', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--families', default='')
    a = ap.parse_args()
    ctx = Ctx()
    if a.phase == 'A':
        phase_a(ctx)
    elif a.phase == 'B':
        phase_b(ctx, seeds=a.seeds)
    elif a.phase == 'C':
        phase_c(ctx, [f for f in a.families.split(',') if f], seeds=a.seeds)
    else:
        sys.exit(f'unknown phase {a.phase}')


if __name__ == '__main__':
    main()
