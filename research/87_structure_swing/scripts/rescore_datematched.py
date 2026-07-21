#!/usr/bin/env python3
"""research/87: date-matched rescore — each trade vs the SAME-DATE universe
mean forward return. Separates name-level signal from bear-regime timing."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_g1_daily_screen import (FAMILIES, HORIZONS, COST_SIDE, IS_START, IS_END,
                                 load_universe, load_daily, simulate)

def main():
    uni = load_universe()
    data = {}
    for s in uni:
        d = load_daily(s)
        if d is not None: data[s] = d
    is_masks = {s: ((df.index >= IS_START) & (df.index <= IS_END)) for s, df in data.items()}
    # per-date universe mean forward return per horizon (keyed by SIGNAL date)
    fwd = {h: {} for h in HORIZONS}
    for s, df in data.items():
        o = df['open'].to_numpy(); dates = df.index
        for h in HORIZONS:
            for t in range(len(o) - 1 - h):
                r = o[t + 1 + h] / o[t + 1] - 1
                fwd[h].setdefault(dates[t], []).append(r)
    datemean = {h: {d: float(np.mean(v)) for d, v in fwd[h].items()} for h in HORIZONS}
    sigs = {}
    for fname, fn in FAMILIES.items():
        for s, df in data.items():
            try: res = fn(df)
            except Exception: continue
            for p, pair in res.items():
                sigs.setdefault((fname, p), {})[s] = pair
    rows = []
    for (fam, par), d_ in sorted(sigs.items()):
        for dname, di in (('L', 1), ('S', -1)):
            for h in HORIZONS:
                rel = []; name_means = {}
                for s, (sl, ss) in d_.items():
                    sig = sl if dname == 'L' else ss
                    df = data[s]; o = df['open'].to_numpy(); dates = df.index
                    idx = np.flatnonzero(sig); busy = -1
                    for t in idx:
                        if t <= busy or t + 1 + h >= len(o) or not is_masks[s][t + 1]: continue
                        bm = datemean[h].get(dates[t])
                        if bm is None: continue
                        stock_r = o[t + 1 + h] / o[t + 1] - 1
                        r = di * (stock_r - bm)     # market-adjusted signal return
                        rel.append(r); name_means.setdefault(s, []).append(r)
                        busy = t + h
                if len(rel) < 100: continue
                rel = np.array(rel)
                m = rel.mean(); t_ = m / (rel.std() / np.sqrt(len(rel)) + 1e-12)
                npos = np.mean([np.mean(v) > 0 for v in name_means.values()])
                rows.append((fam, par, dname, h, len(rel), m * 1e4, (m - 2 * COST_SIDE) * 1e4, t_, npos))
    rows.sort(key=lambda r: -r[7])
    print(f"{'cell':30s} {'n':>6s} {'rel_bps':>8s} {'relnet':>7s} {'t':>6s} {'names+':>6s}")
    for fam, par, dn, h, n, rb, rn, t, np_ in rows[:18]:
        print(f'{fam}_{par}_{dn}_h{h:<3d} {n:>10d} {rb:8.1f} {rn:7.1f} {t:6.2f} {np_:6.2f}')
    ps = [r for r in rows if r[6] > 0 and r[7] >= 2.5 and r[8] >= 0.55]
    print(f'\nDATE-MATCHED PASSERS (relnet>0, t>=2.5, names+>=0.55): {len(ps)}')
    for fam, par, dn, h, n, rb, rn, t, np_ in ps:
        print(f'  {fam}_{par}_{dn}_h{h:<3d} n={n} relnet={rn:.1f} t={t:.2f} names+={np_:.2f}')
    print('DATEMATCH DONE')

if __name__ == '__main__':
    main()
