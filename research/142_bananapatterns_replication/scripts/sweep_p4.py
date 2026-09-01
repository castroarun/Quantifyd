"""
Phase-4 Stage-A wide OFAT sweep (research/142). Loads data ONCE, then for each
cell rebuilds only the cheap matrices and runs a 10-seed ensemble.

Axes around baseline (stop 8 / trail SMA50 / slots 8 / gate DMA200 / rs 70 / base 20%):
  stop_pct    : 4, 5, 6, 7, 10, 12, 15, 99 (=stop off, trail only)
  trail_sma   : 20, 30, 40, 75, 100, 150, 200
  slots       : 5, 6, 10, 12, 15, 20
  gate_dma    : 0 (off), 50, 100, 150, 250
  rs_min      : 60, 80, 90
  base_depth  : 0.10, 0.15, 0.25, 0.30, 9.9 (=no basing filter)

Fixed (config-D stack): 2006-01-01 -> 2026-08-31, realistic fills, 25bps/side,
TV>=Rs5cr, mcap>=Rs500cr PIT proxy, ETFs excluded, close>ATH-close trigger.

Incremental output: results/p4_sweep.csv (one row per cell: ensemble stats).
Resumable: cells already in the CSV are skipped.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br  # noqa: E402

START, END = '2006-01-01', '2026-08-31'
COST = 25 / 10000.0
SEEDS = list(range(1, 11))
OUT = STUDY / 'results' / 'p4_sweep.csv'
FIELDS = ['cell', 'axis', 'value', 'x_med', 'x_min', 'x_max', 'cagr_med', 'cagr_min',
          'cagr_max', 'dd_med', 'dd_worst', 'trades_med', 'win_med', 'signals']

BASE = dict(stop=8.0, trail=50, slots=8, gate=200, rs=70.0, depth=0.20, size=0.1875)
CELLS = [('baseline', 'baseline', 0, dict(BASE))]
for v in [4, 5, 6, 7, 10, 12, 15, 99]:
    CELLS.append((f'stop{v}', 'stop', v, dict(BASE, stop=float(v))))
for v in [20, 30, 40, 75, 100, 150, 200]:
    CELLS.append((f'trail{v}', 'trail', v, dict(BASE, trail=v)))
for v in [5, 6, 10, 12, 15, 20]:
    CELLS.append((f'slots{v}', 'slots', v, dict(BASE, slots=v)))
for v in [0, 50, 100, 150, 250]:
    CELLS.append((f'gate{v}', 'gate', v, dict(BASE, gate=v)))
for v in [60, 80, 90]:
    CELLS.append((f'rs{v}', 'rs', v, dict(BASE, rs=float(v))))
for v in [0.10, 0.15, 0.25, 0.30, 9.9]:
    CELLS.append((f'depth{v}', 'depth', v, dict(BASE, depth=v)))
# Arun's regime-switching idea: mcap floor conditioned on the index regime
CELLS.append(('adapt_floor_when_weak', 'adaptive', 1, dict(BASE, adaptive='floor_when_weak')))
CELLS.append(('adapt_floor_when_strong', 'adaptive', 2, dict(BASE, adaptive='floor_when_strong')))
CELLS.append(('no_mcap_floor', 'adaptive', 0, dict(BASE, adaptive='never')))
# ---- Stage B: trail plateau, combos, sizing x slots ----
CELLS.append(('trail15', 'trail', 15, dict(BASE, trail=15)))
CELLS.append(('trail25', 'trail', 25, dict(BASE, trail=25)))
CELLS.append(('c_t20_s99', 'combo', 1, dict(BASE, trail=20, stop=99.0)))
CELLS.append(('c_t20_s99_nofloor', 'combo', 2, dict(BASE, trail=20, stop=99.0, adaptive='never')))
CELLS.append(('c_t20_s99_nofloor_g0', 'combo', 3, dict(BASE, trail=20, stop=99.0, adaptive='never', gate=0)))
CELLS.append(('c_t20_nofloor', 'combo', 4, dict(BASE, trail=20, adaptive='never')))
CELLS.append(('c_s99_nofloor', 'combo', 5, dict(BASE, stop=99.0, adaptive='never')))
CELLS.append(('sz125_slots8', 'sizing', 1, dict(BASE, size=0.125)))
CELLS.append(('sz10_slots10', 'sizing', 2, dict(BASE, size=0.10, slots=10)))
CELLS.append(('sz0625_slots16', 'sizing', 3, dict(BASE, size=0.0625, slots=16)))


def main():
    t0 = time.time()
    base_start = '2004-07-01'
    w = br.load_frames(base_start)          # sma50 col unused; we compute per-cell
    close, high, open_, athcp, tv20 = (w[k] for k in ('close', 'high', 'open', 'athcp', 'tv20'))

    etf_cols = [c for c in close.columns if br.ETF_RE.search(c)]
    tv_prev = tv20.shift(1)
    prev_close = close.shift(1)
    eligible_base = tv_prev >= br.TV_FLOOR
    eligible_base[etf_cols] = False
    snap = json.load(open(STUDY / 'results' / 'mcap_snapshot.json'))
    shares = pd.Series({s: v['mcap'] / v['px'] for s, v in snap.items()
                        if v.get('mcap') and v.get('px')}).reindex(close.columns)
    mcap_ok = prev_close.mul(shares, axis=1) >= 500 * 1e7
    eligible = eligible_base & mcap_ok          # default (config-D) eligibility
    eligible_nomcap = eligible_base

    r63 = close / close.shift(63) - 1
    r126 = close / close.shift(126) - 1
    r189 = close / close.shift(189) - 1
    r252 = close / close.shift(252) - 1
    score_raw = 2 * r63 + r126 + r189 + r252
    nb = close['NIFTYBEES']

    dates = close.index
    days_idx = np.array([i for i, d in enumerate(dates) if START <= str(d.date()) <= END])
    dates_used = dates[days_idx]
    C, H, O, ATH = close.values, high.values, open_.values, athcp.values
    TVv = tv_prev.values
    print(f'setup done {time.time()-t0:.0f}s; {len(CELLS)} cells', flush=True)

    done = set()
    if OUT.exists():
        done = {r['cell'] for r in csv.DictReader(open(OUT))}
        print(f'skipping {len(done)} done cells', flush=True)
    else:
        with open(OUT, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    sma_cache, rs_cache = {}, {}
    for cell, axis, val, p in CELLS:
        if cell in done:
            continue
        tc = time.time()
        if p['trail'] not in sma_cache:
            sma_cache[p['trail']] = close.rolling(p['trail']).mean().values
        S50 = sma_cache[p['trail']]
        if 'rs' not in rs_cache:
            rs_cache['rs'] = (score_raw.where(eligible).rank(axis=1, pct=True) * 100).shift(1)
            rs_cache['rs_nomcap'] = (score_raw.where(eligible_nomcap)
                                     .rank(axis=1, pct=True) * 100).shift(1)
        rs = rs_cache['rs']

        def build_trig(elig, rs_m):
            setup = (prev_close < athcp) & (prev_close >= (1 - p['depth']) * athcp) \
                & elig & (rs_m >= p['rs'])
            return (setup & (close > athcp) & athcp.notna()).fillna(False).values

        if p['gate']:
            weak_arr = (nb < nb.rolling(p['gate']).mean()).shift(1).fillna(False).values
        else:
            weak_arr = np.zeros(len(dates), dtype=bool)

        adaptive = p.get('adaptive')
        if adaptive:
            t_m = build_trig(eligible, rs)
            t_n = build_trig(eligible_nomcap, rs_cache['rs_nomcap'])
            if adaptive == 'never':
                trig = t_n
                rs = rs_cache['rs_nomcap']
            elif adaptive == 'floor_when_weak':
                trig = np.where(weak_arr[:, None], t_m, t_n)
            else:  # floor_when_strong
                trig = np.where(weak_arr[:, None], t_n, t_m)
        else:
            trig = build_trig(eligible, rs)

        stats = []
        for seed in SEEDS:
            eq, trades, _ = br.simulate(seed, 'random', days_idx, dates, C, H, O, ATH,
                                        S50, rs.values, TVv, trig, weak_arr,
                                        True, COST, stop=p['stop'] / 100.0,
                                        slots=p['slots'], size_pct=p.get('size', 0.1875))
            st, _e = br.stats_from(eq, dates_used, trades, br.CAPITAL)
            stats.append(st)
        sdf = pd.DataFrame(stats)
        row = dict(cell=cell, axis=axis, value=val,
                   x_med=round(sdf.x.median(), 1), x_min=round(sdf.x.min(), 1),
                   x_max=round(sdf.x.max(), 1),
                   cagr_med=round(sdf.cagr.median(), 1), cagr_min=round(sdf.cagr.min(), 1),
                   cagr_max=round(sdf.cagr.max(), 1),
                   dd_med=round(sdf.dd.median(), 1), dd_worst=round(sdf.dd.min(), 1),
                   trades_med=int(sdf.n.median()), win_med=round(sdf.win.median(), 0),
                   signals=int(trig[days_idx].sum()))
        with open(OUT, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"{cell:12s} x_med {row['x_med']:>8} cagr {row['cagr_med']:>5}% "
              f"dd {row['dd_med']:>6}% sig {row['signals']} ({time.time()-tc:.0f}s)", flush=True)

    print(f'SWEEP DONE {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
