#!/usr/bin/env python3
"""research/87: re-score G1 cells as EXCESS over unconditional same-horizon
drift (the honest test — kills drift/survivorship impostors)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_g1_daily_screen import (FAMILIES, HORIZONS, COST_SIDE, IS_START, IS_END,
                                 load_universe, load_daily, simulate)

def main():
    uni = load_universe()
    data = {s: d for s in uni if (d := load_daily(s)) is not None}
    is_masks = {s: ((df.index >= IS_START) & (df.index <= IS_END)) for s, df in data.items()}
    # unconditional baseline per horizon: every-bar forward open-to-open return
    base = {}
    for h in HORIZONS:
        alls = []
        for s, df in data.items():
            o = df['open'].to_numpy(); m = is_masks[s]
            idx = np.flatnonzero(m)
            idx = idx[(idx + 1 + h) < len(o)]
            alls.append(o[idx + 1 + h] / o[idx + 1] - 1)
        allv = np.concatenate(alls)
        base[h] = allv.mean()
        print(f'baseline h={h}: {base[h]*1e4:.1f} bps (n={len(allv)})', flush=True)
    sigs = {}
    for fname, fn in FAMILIES.items():
        for s, df in data.items():
            try: res = fn(df)
            except Exception: continue
            for p, pair in res.items():
                sigs.setdefault((fname, p), {})[s] = pair
    print('\ncell excess results (gross vs drift; excess_net = excess - 10bp):', flush=True)
    rows = []
    for (fam, par), d_ in sorted(sigs.items()):
        for dname, di in (('L', 1), ('S', -1)):
            for h in HORIZONS:
                allg, name_ex = [], []
                bm = base[h] * di  # for shorts the drift benchmark flips sign
                for s, (sl, ss) in d_.items():
                    tr = simulate(data[s], sl if dname == 'L' else ss, di, h, is_masks[s])
                    if tr:
                        g = [x[2] for x in tr]
                        allg.extend(g)
                        name_ex.append(np.mean(g) - bm)
                if len(allg) < 100: continue
                g = np.array(allg)
                ex = g.mean() - bm
                t_ex = ex / (g.std() / np.sqrt(len(g)) + 1e-12)
                ex_net = ex - 2 * COST_SIDE
                rows.append((fam, par, dname, h, len(g), ex * 1e4, ex_net * 1e4, t_ex,
                             np.mean([e > 0 for e in name_ex])))
    rows.sort(key=lambda r: -r[7])
    print(f"{'cell':30s} {'n':>6s} {'exc_bps':>8s} {'excnet':>7s} {'t_exc':>6s} {'names+':>6s}")
    for fam, par, dn, h, n, exb, exn, t, np_ in rows[:20]:
        print(f'{fam}_{par}_{dn}_h{h:<3d} {n:>10d} {exb:8.1f} {exn:7.1f} {t:6.2f} {np_:6.2f}')
    passers = [(f'{fam}_{par}_{dn}_h{h}', n, exn, t, np_) for fam, par, dn, h, n, exb, exn, t, np_ in rows
               if exn > 0 and t >= 2.5 and np_ >= 0.55]
    print(f'\nEXCESS-GATE PASSERS: {len(passers)}')
    for c, n, exn, t, np_ in passers:
        print(f'  {c:30s} n={n} excnet={exn:.1f} t={t:.2f} names+={np_:.2f}')
    print('RESCORE DONE')

if __name__ == '__main__':
    main()
