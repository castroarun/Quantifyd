"""research/151 sweep runner — phases P4 (faithful replica) .. P7 (robustness).

Incremental + resume-safe: one CSV row per completed cell, cells already present in the
output CSV are skipped. Usage (from /home/arun/quantifyd):

  flock -w 7200 /tmp/qf_sweep.lock venv/bin/python -u \
      research/151_vcp_breakout/scripts/vcp_sweep.py --phase p4
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vcp_frames import load                      # noqa: E402
from vcp_replay import Cfg, build_signal, weak_array, simulate, stats, CAPITAL  # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / 'results'

FIELDS = ['cell', 'phase', 'params', 'seeds', 'window', 'x_med', 'cagr_med', 'cagr_min',
          'cagr_max', 'cagr_worst_seed', 'dd_med', 'dd_worst', 'calmar_med', 'n_med',
          'win_med', 'avg_win', 'avg_loss', 'mean_tr', 'median_tr', 'streak', 'tpy',
          'signals', 'passed_up', 'yearly_med', 'runtime_s']

_cache = {}


def get_signal(F, dates, symbols, meta, cfg):
    key = (cfg.pivot_n, cfg.near_pct, cfg.rs_min, cfg.exit_kind)
    if key not in _cache:
        if len(_cache) > 2:
            _cache.clear()
        _cache[key] = build_signal(F, dates, symbols, meta, cfg)
    return _cache[key]


def run_cell(F, dates, symbols, meta, cfg, seeds, label, phase):
    t0 = time.time()
    TRIG, PIV, TRAIL, RS = get_signal(F, dates, symbols, meta, cfg)
    weak = weak_array(F, dates, symbols, cfg)
    C, H, O = F['close'], F['high'], F['open']
    di = np.array([i for i, d in enumerate(dates)
                   if pd.Timestamp(cfg.start) <= d <= pd.Timestamp(cfg.end)])
    dates_used = dates[di]
    allst, curves = [], {}
    for s in seeds:
        eq, tr, passed = simulate(s, cfg, di, dates, C, H, O, PIV, TRAIL, TRIG, RS, weak)
        if cfg.drop_topn and tr:
            keep = sorted(tr, key=lambda t: t[4] / t[3])[:-cfg.drop_topn]
            tr = keep
        st, e = stats(eq, dates_used, tr)
        st['seed'] = s
        st['passed'] = passed
        allst.append(st)
        curves[f'seed{s}'] = e
    d = pd.DataFrame(allst)
    ymed = pd.DataFrame([s['yearly'] for s in allst]).median().round(1)
    row = dict(cell=label, phase=phase,
               params=json.dumps({k: v for k, v in asdict(cfg).items()}),
               seeds=len(seeds), window=f'{cfg.start}:{cfg.end}',
               x_med=round(d.x.median(), 2), cagr_med=round(d.cagr.median(), 2),
               cagr_min=round(d.cagr.min(), 2), cagr_max=round(d.cagr.max(), 2),
               cagr_worst_seed=round(d.cagr.min(), 2),
               dd_med=round(d.dd.median(), 2), dd_worst=round(d.dd.min(), 2),
               calmar_med=round(d.calmar.median(), 3), n_med=int(d.n.median()),
               win_med=round(d.win.median(), 1), avg_win=round(d.avg_win.median(), 2),
               avg_loss=round(d.avg_loss.median(), 2), mean_tr=round(d['mean'].median(), 3),
               median_tr=round(d['median'].median(), 3), streak=int(d.streak.median()),
               tpy=round(d.tpy.median(), 1), signals=int(TRIG[di].sum()),
               passed_up=int(d.passed.median()),
               yearly_med=json.dumps({int(k): v for k, v in ymed.items()}),
               runtime_s=round(time.time() - t0, 1))
    return row, curves, allst


def cells_p4(base):
    """Faithful replica of the site's own dials, their optimistic fills, no costs/tax."""
    out = []
    for slots, risk, stop in ((5, 0.02, 0.07), (5, 0.02, 0.08), (5, 0.015, 0.08),
                              (8, 0.02, 0.08), (8, 0.015, 0.08)):
        out.append((f'p4_slots{slots}_risk{risk}_stop{stop}',
                    replace(base, slots=slots, risk_pct=risk, stop_pct=stop,
                            cost_bps=0.0, fill='pivot', tax=False, cash_yield=0.0,
                            start='2020-01-01', end='2025-12-31')))
    return out


def cells_p5(base):
    """Honest baseline: realistic fills, 25bps, after tax, 5% idle cash, 3 windows."""
    out = []
    for win in (('2020-01-01', '2025-12-31'), ('2012-01-01', '2026-09-01'),
                ('2006-01-01', '2026-09-01')):
        for tag, kw in (('their_dials', dict(slots=5, risk_pct=0.02, stop_pct=0.07)),
                        ('stop8', dict(slots=5, risk_pct=0.02, stop_pct=0.08)),
                        ('fixed16x6.25', dict(slots=16, sizing='fixed', size_pct=0.0625,
                                              stop_pct=0.08)),
                        ('fixed8x18.75', dict(slots=8, sizing='fixed', size_pct=0.1875,
                                              stop_pct=0.08))):
            out.append((f'p5_{tag}_{win[0][:4]}',
                        replace(base, start=win[0], end=win[1], **kw)))
    return out


def cells_p6(base):
    """Optimization: their dials as axes, exits tested jointly with entries."""
    out = []
    long_win = dict(start='2012-01-01', end='2026-09-01')
    # A: pivot lookback x exit family (entry x exit, jointly)
    for n in (10, 20, 30, 50, 75, 120, 252):
        for ex in ('sma15', 'sma20', 'sma50', 'sma150', 'target25'):
            out.append((f'p6A_piv{n}_{ex}',
                        replace(base, pivot_n=n, exit_kind=ex, **long_win)))
    # B: slots x sizing
    for slots in (3, 5, 8, 10, 16):
        for sz in ('risk', 'fixed'):
            for v in ((0.01, 0.015, 0.02) if sz == 'risk' else (0.0625, 0.10, 0.1875)):
                kw = dict(risk_pct=v) if sz == 'risk' else dict(size_pct=v)
                out.append((f'p6B_slots{slots}_{sz}{v}',
                            replace(base, slots=slots, sizing=sz, **kw, **long_win)))
    # C: stop x breakeven
    for stop in (0.06, 0.07, 0.08, 0.10, 0.15):
        for be in (0.0, 0.05, 0.10, 0.20):
            out.append((f'p6C_stop{stop}_be{be}',
                        replace(base, stop_pct=stop, breakeven_at=be, **long_win)))
    # D: weak-market gate across index series x gate SMA
    for g in ('', 'NIFTYBEES', 'JUNIORBEES', 'NIFTYMIDCAP150', 'NIFTYSMLCAP250'):
        for gs in (100, 150, 200):
            if not g and gs != 200:
                continue
            out.append((f'p6D_gate{g or "off"}_{gs}',
                        replace(base, gate=g, gate_sma=gs, **long_win)))
    # E: proximity + RS threshold
    for near in (0.03, 0.05, 0.10, 0.20, 0.50):
        for rs in (0, 50, 70, 85):
            out.append((f'p6E_near{near}_rs{rs}',
                        replace(base, near_pct=near, rs_min=rs, **long_win)))
    return out


def cells_p6f(base):
    """The same axes under OUR fixed-weight sizing (P5 showed fixed 16 x 6.25% beats
    their concentrated risk sizing on Calmar and on the seed band)."""
    out = []
    lw = dict(start='2012-01-01', end='2026-09-01', sizing='fixed', size_pct=0.0625,
              slots=16, stop_pct=0.08)
    for n in (10, 15, 20, 30, 50, 75, 120):
        for ex in ('sma15', 'sma20', 'sma50'):
            out.append((f'p6F_piv{n}_{ex}', replace(base, pivot_n=n, exit_kind=ex, **lw)))
    for g, gs in (('', 200), ('NIFTYBEES', 200), ('NIFTYBEES', 100), ('NIFTYBEES', 150)):
        out.append((f'p6F_gate{g or "off"}{gs}',
                    replace(base, gate=g, gate_sma=gs, exit_kind='sma15', pivot_n=10, **lw)))
    for slots, sz in ((8, 0.125), (12, 0.0833), (16, 0.0625), (20, 0.05), (25, 0.04)):
        out.append((f'p6F_slots{slots}',
                    replace(base, **{**lw, 'slots': slots, 'size_pct': sz},
                            exit_kind='sma15', pivot_n=10)))
    for near in (0.05, 0.10, 0.20, 0.50):
        for rs in (0, 50, 70, 85):
            out.append((f'p6F_near{near}_rs{rs}',
                        replace(base, near_pct=near, rs_min=rs, exit_kind='sma15',
                                pivot_n=10, **lw)))
    for stop in (0.06, 0.08, 0.10, 0.15, 0.99):
        out.append((f'p6F_stop{stop}',
                    replace(base, **{**lw, 'stop_pct': stop}, exit_kind='sma15', pivot_n=10)))
    return out


def cells_p6g(base):
    """Null control on the entry: does the breakout lookback do ANY work?
    pivot_n -> 1 degenerates to 'buy any RS-qualified name that closes up', which is the
    matched null for the whole screen. Plus the cost ladder on the two candidate specs."""
    out = []
    lw = dict(start='2012-01-01', end='2026-09-01', sizing='fixed', size_pct=0.0625,
              slots=16, stop_pct=0.08, exit_kind='sma15')
    for n in (1, 2, 3, 5, 7, 10, 30):
        out.append((f'p6G_nullpiv{n}', replace(base, pivot_n=n, **lw)))
    for n in (10, 30):
        for bps in (40.0, 60.0, 100.0):
            out.append((f'p6G_cost{bps}_piv{n}',
                        replace(base, pivot_n=n, cost_bps=bps, **lw)))
    for n in (10, 30):
        out.append((f'p6G_notax_piv{n}', replace(base, pivot_n=n, tax=False, **lw)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--save-curves', default='')
    a = ap.parse_args()
    F, dates, symbols, meta = load()
    print(f'frames {F["close"].shape} {dates[0].date()}..{dates[-1].date()}', flush=True)

    base = Cfg()
    builder = dict(p4=cells_p4, p5=cells_p5, p6=cells_p6, p6f=cells_p6f, p6g=cells_p6g)[a.phase]
    cells = builder(base)
    seeds = list(range(1, a.seeds + 1))
    out_csv = RES / f'{a.phase}_cells.csv'
    done = set()
    if out_csv.exists():
        done = {r['cell'] for r in csv.DictReader(open(out_csv))}
        print(f'resume: {len(done)} cells already complete', flush=True)
    else:
        with open(out_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    for j, (label, cfg) in enumerate(cells, 1):
        if label in done:
            continue
        try:
            row, curves, allst = run_cell(F, dates, symbols, meta, cfg, seeds, label, a.phase)
        except SystemExit as e:
            print(f'[{j}/{len(cells)}] {label} SKIPPED: {e}', flush=True)
            continue
        with open(out_csv, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"[{j}/{len(cells)}] {label:34s} x{row['x_med']:8.2f} CAGR {row['cagr_med']:6.2f}% "
              f"[{row['cagr_min']:.1f}..{row['cagr_max']:.1f}] DD {row['dd_med']:6.1f}% "
              f"Calmar {row['calmar_med']:.2f} n={row['n_med']:4d} win {row['win_med']:.0f}% "
              f"sig={row['signals']} ({row['runtime_s']}s)", flush=True)
        if a.save_curves and label == a.save_curves:
            pd.DataFrame(curves).to_csv(RES / f'{label}_equity.csv')
    print('PHASE DONE', a.phase, flush=True)


if __name__ == '__main__':
    main()
