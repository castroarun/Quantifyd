#!/usr/bin/env python3
"""research/87 phase 1d: remaining named chart patterns (investingoal list).

AT/DT ascending-descending triangles, triple top/bottom, island reversal,
dead-cat bounce, megaphone, diamond (approx), channel bounce, parabolic
break. Daily bars, date-matched harness identical to phase 1c.
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


def fam_atdt(df):
    """Ascending triangle: flat resistance (>=2 pivot highs within 1%) +
    >=2 successively higher pivot lows. Descending: mirror. Break = dir."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    ph, pl, phi, pli = confirmed_pivots(h, l, 3)
    out = {}
    for sub in ("asc", "desc"):
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
            if sub == "asc":
                cond = abs(h1 - h2) / h1 <= 0.01 and l2 > l1 * 1.005
                res, sup = max(h1, h2), l2
            else:
                cond = abs(l1 - l2) / l1 <= 0.01 and h2 < h1 * 0.995
                res, sup = h2, min(l1, l2)
            if not cond:
                continue
            if c[t] > res:
                sl[t] = True
            elif c[t] < sup:
                ss[t] = True
        out[sub] = (sl, ss)
    return out


def fam_cp8(df):
    """Triple top (short) / triple bottom (long): 3 pivots within 1.5%,
    span >= 20 bars, neckline break."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    ph, pl, phi, pli = confirmed_pivots(h, l, 3)
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    highs, lows = [], []
    armed_s = armed_l = None
    for t in range(n):
        if not np.isnan(ph[t]):
            highs.append((ph[t], phi[t]))
            if len(highs) >= 3:
                (v1, i1), (v2, i2), (v3, i3) = highs[-3:]
                vs = [v1, v2, v3]
                if (max(vs) - min(vs)) / min(vs) <= 0.015 and i3 - i1 >= 20:
                    armed_s = (l[i1:i3 + 1].min(), t + 40)
        if not np.isnan(pl[t]):
            lows.append((pl[t], pli[t]))
            if len(lows) >= 3:
                (v1, i1), (v2, i2), (v3, i3) = lows[-3:]
                vs = [v1, v2, v3]
                if (max(vs) - min(vs)) / min(vs) <= 0.015 and i3 - i1 >= 20:
                    armed_l = (h[i1:i3 + 1].max(), t + 40)
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
    return {"tol1.5": (sl, ss)}


def fam_isl(df):
    """Island reversal: up-gap then down-gap within 10 bars -> short;
    mirror -> long."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(h)
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    for t in range(2, n):
        if h[t] < l[t - 1]:                      # down-gap today
            for a in range(max(1, t - 10), t):
                if l[a] > h[a - 1]:              # up-gap started island
                    ss[t] = True
                    break
        if l[t] > h[t - 1]:                      # up-gap today
            for a in range(max(1, t - 10), t):
                if h[a] < l[a - 1]:              # down-gap started island
                    sl[t] = True
                    break
    return {"g10": (sl, ss)}


def fam_dcb(df):
    """Dead-cat bounce: >=15%/5-bar crash, >=5% bounce, first lower close
    -> short. Mirror (melt-up, fade, first higher close) -> long."""
    c = df["close"].to_numpy()
    n = len(c)
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    for t in range(12, n):
        for j in range(2, 8):                    # bounce length
            k = t - j                            # crash low bar
            if k - 5 < 0:
                continue
            crash = c[k] / c[k - 5] - 1
            if crash <= -0.15 and c[t - 1] >= c[k] * 1.05 and c[t] < c[t - 1]:
                ss[t] = True
                break
            if crash >= 0.15 and c[t - 1] <= c[k] * 0.95 and c[t] > c[t - 1]:
                sl[t] = True
                break
    return {"c15b5": (sl, ss)}


def fam_meg(df):
    """Megaphone: >=2 successively higher pivot highs AND >=2 lower pivot
    lows within 40 bars (diverging); extreme break sets direction."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    ph, pl, phi, pli = confirmed_pivots(h, l, 3)
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
        if span < 10 or span > 40 or t - max(hi2, li2) > 20:
            continue
        if h2 > h1 * 1.005 and l2 < l1 * 0.995:
            if c[t] > h2:
                sl[t] = True
            elif c[t] < l2:
                ss[t] = True
    return {"div40": (sl, ss)}


def fam_dia(df):
    """Diamond approx: 10-bar range expands >=1.5x then contracts >=40%;
    break of the contraction extremes."""
    h, l, c = df["high"], df["low"], df["close"]
    rng = (h.rolling(10).max() - l.rolling(10).min()).to_numpy()
    cn = c.to_numpy()
    hh = h.rolling(10).max().shift(1).to_numpy()
    ll = l.rolling(10).min().shift(1).to_numpy()
    n = len(cn)
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    for t in range(40, n):
        r_now = rng[t - 1]
        r_mid = rng[t - 10]
        r_old = rng[t - 25]
        if r_old > 0 and r_mid >= 1.5 * r_old and r_now <= 0.6 * r_mid:
            if cn[t] > hh[t]:
                sl[t] = True
            elif cn[t] < ll[t]:
                ss[t] = True
    return {"x1.5c0.6": (sl, ss)}


def fam_chn(df):
    """Channel bounce: >=3 successively higher pivot lows, roughly
    collinear -> tag of the line through last two + close back above = long
    continuation. Falling mirror -> short."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    ph, pl, phi, pli = confirmed_pivots(h, l, 3)
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    highs, lows = [], []
    for t in range(n):
        if not np.isnan(ph[t]):
            highs.append((ph[t], phi[t]))
        if not np.isnan(pl[t]):
            lows.append((pl[t], pli[t]))
        if len(lows) >= 3:
            (v1, i1), (v2, i2), (v3, i3) = lows[-3:]
            if v1 < v2 < v3 and i3 - i1 <= 80 and t - i3 <= 30:
                slope = (v3 - v2) / max(i3 - i2, 1)
                line = v3 + slope * (t - i3)
                if line > 0 and l[t] <= line * 1.005 and c[t] > line:
                    sl[t] = True
        if len(highs) >= 3:
            (v1, i1), (v2, i2), (v3, i3) = highs[-3:]
            if v1 > v2 > v3 and i3 - i1 <= 80 and t - i3 <= 30:
                slope = (v3 - v2) / max(i3 - i2, 1)
                line = v3 + slope * (t - i3)
                if h[t] >= line * 0.995 and c[t] < line:
                    ss[t] = True
    return {"3piv": (sl, ss)}


def fam_para(df):
    """Parabolic break: >=25%/20-bar accelerating run, first close below
    the 3-bar low -> short. Mirror capitulation -> long."""
    c, l, h = df["close"].to_numpy(), df["low"].to_numpy(), df["high"].to_numpy()
    n = len(c)
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    for t in range(25, n):
        r20 = c[t - 1] / c[t - 21] - 1
        r5 = c[t - 1] / c[t - 6] - 1
        if r20 >= 0.25 and r5 >= r20 * 0.4:
            if c[t] < l[t - 3:t].min():
                ss[t] = True
        if r20 <= -0.25 and r5 <= r20 * 0.4:
            if c[t] > h[t - 3:t].max():
                sl[t] = True
    return {"r25a": (sl, ss)}


FAMILIES = {"ATDT": fam_atdt, "CP8": fam_cp8, "ISL": fam_isl, "DCB": fam_dcb,
            "MEG": fam_meg, "DIA": fam_dia, "CHN": fam_chn, "PARA": fam_para}


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
                wide, dm = fwd[h], dmean[h]
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
                                 None, None, None, None))
                    continue
                ra, re = np.array(raws), np.array(rels)
                rows.append((
                    f"{fam}_{par}_{dname}_h{h}", len(ra),
                    (ra.mean() - COST_RT) * 1e4,
                    (re.mean() - COST_RT) * 1e4,
                    re.mean() / (re.std() / np.sqrt(len(re)) + 1e-12),
                    np.mean([np.mean(v) > 0 for v in name_rel.values()])))
    rows.sort(key=lambda r: -(r[4] if r[4] is not None else -99))
    print(f"\n{'cell':24s} {'n':>6s} {'abs_net':>8s} {'rel_net':>8s} {'t_rel':>6s} {'names+':>6s}")
    for cell, n, absn, reln, trel, np_ in rows:
        if trel is None:
            print(f"{cell:24s} {n:>6d}   (n<60)")
        else:
            print(f"{cell:24s} {n:>6d} {absn:8.1f} {reln:8.1f} {trel:6.2f} {np_:6.2f}")
    ps = [r for r in rows if r[4] is not None and r[3] > 0 and r[4] >= 2.5 and r[5] >= 0.55]
    print(f"\nGATE PASSERS: {len(ps)}")
    for cell, n, absn, reln, trel, np_ in ps:
        print(f"  {cell} n={n} rel_net={reln:.1f} t={trel:.2f} names+={np_:.2f}")
    print("PATTERNS3 DONE")


if __name__ == "__main__":
    main()
