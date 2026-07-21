#!/usr/bin/env python3
"""research/87 phase 1c: explicit chart-pattern geometry on daily bars.

CP4 head-and-shoulders / inverse; CP5 triangles & wedges (pivot-sequence
convergence); CP6 cup-with-handle / inverted. Scored date-matched (primary)
plus absolute, same harness conventions as the phase-1 rescore.
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_g1_daily_screen import (COST_SIDE, HORIZONS, IS_START, IS_END,
                                 confirmed_pivots, load_daily, load_universe)

COST_RT = 2 * COST_SIDE
WIN_START = os.getenv("R87_WIN_START", IS_START)
WIN_END = os.getenv("R87_WIN_END", IS_END)


def log(m):
    print(f"{datetime.now().isoformat(timespec='seconds')} {m}", flush=True)


def fam_cp4(df):
    """Head & shoulders (short) / inverse H&S (long). Pivot w=3."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    ph, pl, phi, pli = confirmed_pivots(h, l, 3)
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    highs, lows = [], []          # (value, formed_idx)
    armed_s = armed_l = None      # (neckline, expiry)
    for t in range(n):
        if not np.isnan(ph[t]):
            highs.append((ph[t], phi[t]))
            if len(highs) >= 3:
                (v1, i1), (v2, i2), (v3, i3) = highs[-3:]
                if (v2 > v1 and v2 > v3 and abs(v1 - v3) / v1 <= 0.03
                        and i3 - i1 >= 15):
                    seg = l[i1:i3 + 1]
                    neck = seg.min()
                    if v2 / neck - 1 >= 0.04:
                        armed_s = (neck, t + 40)
        if not np.isnan(pl[t]):
            lows.append((pl[t], pli[t]))
            if len(lows) >= 3:
                (v1, i1), (v2, i2), (v3, i3) = lows[-3:]
                if (v2 < v1 and v2 < v3 and abs(v1 - v3) / v1 <= 0.03
                        and i3 - i1 >= 15):
                    seg = h[i1:i3 + 1]
                    neck = seg.max()
                    if neck / v2 - 1 >= 0.04:
                        armed_l = (neck, t + 40)
        if armed_s:
            if t > armed_s[1]:
                armed_s = None
            elif c[t] < armed_s[0]:
                ss[t] = True
                armed_s = None
        if armed_l:
            if t > armed_l[1]:
                armed_l = None
            elif c[t] > armed_l[0]:
                sl[t] = True
                armed_l = None
    return {"w3_sh3pct": (sl, ss)}


def fam_cp5(df):
    """Triangles/wedges from pivot sequences. sym: lower highs + higher
    lows; risingw: both rising, converging; fallingw: both falling,
    converging. Break beyond the last pivot bound sets direction."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    ph, pl, phi, pli = confirmed_pivots(h, l, 3)
    out = {}
    for sub in ("sym", "risingw", "fallingw"):
        sl = np.zeros(n, bool)
        ss = np.zeros(n, bool)
        highs, lows = [], []
        for t in range(n):
            if not np.isnan(ph[t]):
                highs.append((ph[t], phi[t]))
            if not np.isnan(pl[t]):
                lows.append((pl[t], pli[t]))
            if len(highs) < 2 or len(lows) < 2:
                continue
            (h1, hi1), (h2, hi2) = highs[-2:]
            (l1, li1), (l2, li2) = lows[-2:]
            span = max(hi2, li2) - min(hi1, li1)
            if span < 10 or span > 60 or t - max(hi2, li2) > 25:
                continue
            if sub == "sym":
                cond = h2 < h1 and l2 > l1
            elif sub == "risingw":
                hs = (h2 - h1) / max(hi2 - hi1, 1)
                ls = (l2 - l1) / max(li2 - li1, 1)
                cond = h2 > h1 and l2 > l1 and ls > hs > 0
            else:
                hs = (h2 - h1) / max(hi2 - hi1, 1)
                ls = (l2 - l1) / max(li2 - li1, 1)
                cond = h2 < h1 and l2 < l1 and hs < ls < 0
            if not cond:
                continue
            if c[t] > h2:
                sl[t] = True
            elif c[t] < l2:
                ss[t] = True
        out[sub] = (sl, ss)
    return out


def fam_cp6(df):
    """Cup-with-handle (long) / inverted (short)."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    for t in range(60, n):
        w0 = max(0, t - 120)
        # long: prior peak >=30 bars back, base depth >=15%, recovery to
        # within 3%, handle drift <=5% deep for 3-10 bars, break of peak
        seg_end = t - 3
        peak_i = w0 + int(np.argmax(h[w0:seg_end]))
        if t - peak_i < 30:
            continue
        peak = h[peak_i]
        base = l[peak_i:seg_end].min()
        if peak <= 0 or base / peak > 0.85:
            pass
        else:
            for hl in range(3, 11):
                k = t - hl
                if k <= peak_i:
                    break
                if h[k] >= peak * 0.97:
                    hh_seg = h[k + 1:t]
                    ll_seg = l[k + 1:t]
                    if len(hh_seg) and (hh_seg.max() - ll_seg.min()) / peak <= 0.05 \
                            and c[t] > peak:
                        sl[t] = True
                    break
        trough_i = w0 + int(np.argmin(l[w0:seg_end]))
        if t - trough_i < 30:
            continue
        trough = l[trough_i]
        top = h[trough_i:seg_end].max()
        if trough <= 0 or top / trough < 1.15:
            continue
        for hl in range(3, 11):
            k = t - hl
            if k <= trough_i:
                break
            if l[k] <= trough * 1.03:
                hh_seg = h[k + 1:t]
                ll_seg = l[k + 1:t]
                if len(hh_seg) and (hh_seg.max() - ll_seg.min()) / trough <= 0.05 \
                        and c[t] < trough:
                    ss[t] = True
                break
    return {"d15_h5": (sl, ss)}


FAMILIES = {"CP4": fam_cp4, "CP5": fam_cp5, "CP6": fam_cp6}


def main():
    t0 = time.time()
    uni = load_universe()
    data = {}
    for s in uni:
        d = load_daily(s)
        if d is not None:
            data[s] = d
    masks = {s: ((df.index >= WIN_START) & (df.index <= WIN_END))
             for s, df in data.items()}
    log(f"loaded {len(data)} names | window {WIN_START}..{WIN_END}")

    fwd, dmean = {}, {}
    for h in HORIZONS:
        frames = []
        for s, df in data.items():
            o = df["open"].to_numpy()
            r = np.full(len(o), np.nan)
            if len(o) > h + 1:
                r[:-(h + 1)] = o[h + 1:] / o[1:len(o) - h] - 1
            frames.append(pd.Series(r, index=df.index, name=s))
        wide = pd.concat(frames, axis=1, sort=True)
        fwd[h] = wide
        dmean[h] = wide.mean(axis=1)

    sigs = {}
    for fname, fn in FAMILIES.items():
        for s, df in data.items():
            try:
                res = fn(df)
            except Exception as e:
                log(f"det {fname}/{s}: {e}")
                continue
            for p, pair in res.items():
                sigs.setdefault((fname, p), {})[s] = pair
        log(f"signals {fname} done ({time.time()-t0:.0f}s)")

    rows = []
    for (fam, par), d_ in sorted(sigs.items()):
        for dname, di in (("L", 1), ("S", -1)):
            for h in HORIZONS:
                wide = fwd[h]
                dm = dmean[h]
                raws, rels = [], []
                name_rel = {}
                for s, (sl, ss) in d_.items():
                    sig = sl if dname == "L" else ss
                    df = data[s]
                    fs = wide[s].reindex(df.index).to_numpy()
                    dms = dm.reindex(df.index).to_numpy()
                    m = masks[s]
                    busy = -1
                    for t in np.flatnonzero(sig):
                        if t <= busy or not m[t] or np.isnan(fs[t]) or np.isnan(dms[t]):
                            continue
                        raws.append(di * fs[t])
                        rel = di * (fs[t] - dms[t])
                        rels.append(rel)
                        name_rel.setdefault(s, []).append(rel)
                        busy = t + h
                if len(raws) < 60:
                    rows.append((f"{fam}_{par}_{dname}_h{h}", len(raws),
                                 None, None, None, None, None))
                    continue
                ra, re = np.array(raws), np.array(rels)
                rows.append((
                    f"{fam}_{par}_{dname}_h{h}", len(ra),
                    (ra.mean() - COST_RT) * 1e4,
                    (re.mean() - COST_RT) * 1e4,
                    re.mean() / (re.std() / np.sqrt(len(re)) + 1e-12),
                    np.mean([np.mean(v) > 0 for v in name_rel.values()]),
                    re.mean() * 1e4))
    rows.sort(key=lambda r: -(r[4] if r[4] is not None else -99))
    print(f"\n{'cell':26s} {'n':>6s} {'abs_net':>8s} {'rel_net':>8s} {'t_rel':>6s} {'names+':>6s}")
    for cell, n, absn, reln, trel, np_, relb in rows:
        if trel is None:
            print(f"{cell:26s} {n:>6d}   (n<60)")
        else:
            print(f"{cell:26s} {n:>6d} {absn:8.1f} {reln:8.1f} {trel:6.2f} {np_:6.2f}")
    ps = [r for r in rows if r[4] is not None and r[3] > 0 and r[4] >= 2.5 and r[5] >= 0.55]
    print(f"\nGATE PASSERS: {len(ps)}")
    for cell, n, absn, reln, trel, np_, relb in ps:
        print(f"  {cell} n={n} rel_net={reln:.1f} t={trel:.2f} names+={np_:.2f}")
    print("PATTERNS2 DONE")


if __name__ == "__main__":
    main()
