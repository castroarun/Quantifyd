#!/usr/bin/env python3
"""research/87 G1 IS screen: structure/pattern swing families, daily bars.

8 families x grids = 104 pre-registered cells (see STATUS doc). Signal on
close[t] -> enter open[t+1] -> exit open[t+1+h]. Non-overlap per cell.
Costs: 5bp/side (10bp RT). IS window 2005-01-01..2017-12-31.
Incremental CSV; safe to rerun (skips done cells).
"""
import csv
import os
import sqlite3
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(STUDY))
sys.path.insert(0, REPO)
DB = os.path.join(REPO, "backtest_data", "market_data.db")
RES = os.path.join(STUDY, "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "g1_screen.csv")

IS_START = os.getenv("R87_WIN_START", "2005-01-01")
IS_END = os.getenv("R87_WIN_END", "2017-12-31")
WARM_START = "2000-01-01"
COST_SIDE = 0.0005
HORIZONS = [3, 5, 10, 15]
FIELDS = ["cell", "family", "params", "dir", "h", "n", "gross_bps", "net_bps",
          "t", "win", "pct_names_pos", "pct_years_pos", "yrs"]


def log(msg):
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)


def load_universe():
    from services.data_manager import FNO_LOT_SIZES
    return sorted(FNO_LOT_SIZES.keys())


def load_daily(sym):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' AND date>=? AND date<=? ORDER BY date",
        con, params=(sym, WARM_START, IS_END + " 23:59"))
    con.close()
    if len(df) < 300:
        return None
    df["date"] = pd.to_datetime(df["date"].str[:10])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if len(df) < 300:
        return None
    return df.set_index("date")


# ---------------------------------------------------------------- helpers

def confirmed_pivots(high, low, w):
    """Pivot at i is CONFIRMED at i+w (causal). Returns arrays where
    ph[j]/pl[j] = pivot value confirmed at bar j (nan otherwise) and the
    pivot's formation index."""
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    phi = np.full(n, -1, dtype=int)
    pli = np.full(n, -1, dtype=int)
    for i in range(w, n - w):
        seg_h = high[i - w:i + w + 1]
        seg_l = low[i - w:i + w + 1]
        if high[i] == seg_h.max():
            ph[i + w] = high[i]
            phi[i + w] = i
        if low[i] == seg_l.min():
            pl[i + w] = low[i]
            pli[i + w] = i
    return ph, pl, phi, pli


def ffill_level(arr):
    out = np.full(len(arr), np.nan)
    cur = np.nan
    for i, v in enumerate(arr):
        if not np.isnan(v):
            cur = v
        out[i] = cur
    return out


# ---------------------------------------------------------------- families
# Each detector returns {param_str: (sig_long bool[n], sig_short bool[n])}

def fam_sr1(df):
    """Cross of the last confirmed pivot high (long) / pivot low (short)."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    out = {}
    for w in (3, 5):
        ph, pl, _, _ = confirmed_pivots(h, l, w)
        lvl_h, lvl_l = ffill_level(ph), ffill_level(pl)
        sl = np.zeros(len(c), bool)
        ss = np.zeros(len(c), bool)
        for t in range(1, len(c)):
            if not np.isnan(lvl_h[t]) and c[t] > lvl_h[t] and c[t - 1] <= lvl_h[t]:
                sl[t] = True
            if not np.isnan(lvl_l[t]) and c[t] < lvl_l[t] and c[t - 1] >= lvl_l[t]:
                ss[t] = True
        out[f"w{w}"] = (sl, ss)
    return out


def fam_sr2(df):
    """Bounce off a >=2-touch pivot cluster: tag within 0.5%, close holds."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    ph, pl, phi, pli = confirmed_pivots(h, l, 3)
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    lows, highs = [], []          # (value, formed_idx) confirmed so far
    for t in range(n):
        if not np.isnan(pl[t]):
            lows.append((pl[t], pli[t]))
        if not np.isnan(ph[t]):
            highs.append((ph[t], phi[t]))
        recent_l = [v for v, i in lows if t - i <= 250]
        recent_h = [v for v, i in highs if t - i <= 250]
        for L in recent_l:
            touches = sum(1 for v in recent_l if abs(v - L) / L <= 0.01)
            if touches >= 2 and l[t] <= L * 1.005 and c[t] > L:
                sl[t] = True
                break
        for R in recent_h:
            touches = sum(1 for v in recent_h if abs(v - R) / R <= 0.01)
            if touches >= 2 and h[t] >= R * 0.995 and c[t] < R:
                ss[t] = True
                break
    return {"tol1pct": (sl, ss)}


def fam_cp1(df):
    """Volatility-contraction breakout: 20d range width in bottom p% of
    252d, then close breaks the consolidation extreme."""
    h, l, c = df["high"], df["low"], df["close"]
    rng = (h.rolling(20).max() - l.rolling(20).min()) / c
    out = {}
    for p in (10, 20):
        pct = rng.rolling(252).rank(pct=True) * 100
        tight = (pct < p).to_numpy()
        hh = h.rolling(20).max().shift(1).to_numpy()
        ll = l.rolling(20).min().shift(1).to_numpy()
        cn = c.to_numpy()
        sl = np.zeros(len(cn), bool)
        ss = np.zeros(len(cn), bool)
        for t in range(1, len(cn)):
            if tight[t - 1]:
                if cn[t] > hh[t]:
                    sl[t] = True
                elif cn[t] < ll[t]:
                    ss[t] = True
        out[f"p{p}"] = (sl, ss)
    return out


def fam_cp2(df):
    """Flag: >=8%/5d impulse, then <=5%-deep drift for 3-8 bars, break of
    flag extreme (mirror for down-flags)."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    for t in range(20, n):
        for j in range(3, 9):                     # flag length
            k = t - j                             # impulse end bar
            if k - 5 < 0:
                continue
            imp = c[k] / c[k - 5] - 1
            fh = h[k + 1:t].max() if t > k + 1 else h[k]
            fl = l[k + 1:t].min() if t > k + 1 else l[k]
            depth = (fh - fl) / c[k]
            if imp >= 0.08 and depth <= 0.05 and c[t] > fh:
                sl[t] = True
                break
            if imp <= -0.08 and depth <= 0.05 and c[t] < fl:
                ss[t] = True
                break
    return {"i8d5": (sl, ss)}


def fam_cp3(df):
    """Double bottom (long) / double top (short): two pivots within 1.5%,
    >=10 bars apart, neckline >=4% away, close breaks neckline."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    ph, pl, phi, pli = confirmed_pivots(h, l, 3)
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    lows, highs = [], []
    armed_l = armed_s = None      # (neckline, expiry)
    for t in range(n):
        if not np.isnan(pl[t]):
            lows.append((pl[t], pli[t]))
            if len(lows) >= 2:
                (v1, i1), (v2, i2) = lows[-2], lows[-1]
                if abs(v1 - v2) / v1 <= 0.015 and i2 - i1 >= 10:
                    neck = h[i1:i2 + 1].max()
                    if neck >= max(v1, v2) * 1.04:
                        armed_l = (neck, t + 40)
        if not np.isnan(ph[t]):
            highs.append((ph[t], phi[t]))
            if len(highs) >= 2:
                (v1, i1), (v2, i2) = highs[-2], highs[-1]
                if abs(v1 - v2) / v1 <= 0.015 and i2 - i1 >= 10:
                    neck = l[i1:i2 + 1].min()
                    if neck <= min(v1, v2) * 0.96:
                        armed_s = (neck, t + 40)
        if armed_l:
            if t > armed_l[1]:
                armed_l = None
            elif c[t] > armed_l[0]:
                sl[t] = True
                armed_l = None
        if armed_s:
            if t > armed_s[1]:
                armed_s = None
            elif c[t] < armed_s[0]:
                ss[t] = True
                armed_s = None
    return {"tol1.5_neck4": (sl, ss)}


def fam_mtf1(df):
    """Weekly trend (close > 30w SMA; mirror <) AND daily breakout of
    {20,55}d extreme."""
    c = df["close"]
    wk = c.resample("W-FRI").last()
    wsma = wk.rolling(30).mean()
    up_w = (wk > wsma).shift(1)                    # last COMPLETED week
    up_d = up_w.reindex(df.index, method="ffill").fillna(False).to_numpy()
    out = {}
    for dtrig in (20, 55):
        hh = df["high"].rolling(dtrig).max().shift(1).to_numpy()
        ll = df["low"].rolling(dtrig).min().shift(1).to_numpy()
        cn = c.to_numpy()
        sl = (cn > hh) & up_d
        ss = (cn < ll) & ~up_d
        out[f"d{dtrig}"] = (sl, ss)
    return out


def fam_br1(df):
    """52-week extreme break with volume expansion."""
    c, v = df["close"], df["volume"]
    hh = df["high"].rolling(252).max().shift(1)
    ll = df["low"].rolling(252).min().shift(1)
    vr = v / v.rolling(20).mean().shift(1)
    out = {}
    for vm in (1.5, 2.0):
        sl = ((c > hh) & (vr >= vm)).to_numpy()
        ss = ((c < ll) & (vr >= vm)).to_numpy()
        out[f"v{vm}"] = (sl, ss)
    return out


def fam_br2(df):
    """Compression breaks: inside-bar mother break; NR7 range break."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    rng = h - l
    out = {}
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    for t in range(2, n):
        if h[t - 1] < h[t - 2] and l[t - 1] > l[t - 2]:
            if c[t] > h[t - 2]:
                sl[t] = True
            elif c[t] < l[t - 2]:
                ss[t] = True
    out["inside"] = (sl.copy(), ss.copy())
    sl = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    for t in range(7, n):
        if rng[t - 1] == rng[t - 7:t].min():
            if c[t] > h[t - 1]:
                sl[t] = True
            elif c[t] < l[t - 1]:
                ss[t] = True
    out["nr7"] = (sl, ss)
    return out


FAMILIES = {
    "SR1": fam_sr1, "SR2": fam_sr2, "CP1": fam_cp1, "CP2": fam_cp2,
    "CP3": fam_cp3, "MTF1": fam_mtf1, "BR1": fam_br1, "BR2": fam_br2,
}


# ---------------------------------------------------------------- sim

def simulate(df, sig, direction, h, is_mask):
    """Fixed-horizon, non-overlapping. Returns list of (year, net_ret, gross)."""
    o = df["open"].to_numpy()
    years = df.index.year
    n = len(o)
    trades = []
    busy_until = -1
    idx = np.flatnonzero(sig)
    for t in idx:
        if t <= busy_until or t + 1 + h >= n or not is_mask[t + 1]:
            continue
        e, x = o[t + 1], o[t + 1 + h]
        gross = direction * (x / e - 1)
        net = gross - 2 * COST_SIDE
        trades.append((int(years[t + 1]), net, gross))
        busy_until = t + h
    return trades


def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {r["cell"] for r in csv.DictReader(f)}
        log(f"resume: {len(done)} cells already done")
    else:
        with open(OUT, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    uni = load_universe()
    log(f"universe {len(uni)} names; loading daily bars...")
    data = {}
    for s in uni:
        df = load_daily(s)
        if df is not None:
            data[s] = df
    log(f"loaded {len(data)} names with daily history")

    # precompute signals per family per name
    t0 = time.time()
    sigs = {}      # (fam, param) -> {sym: (sl, ss)}
    for fname, fn in FAMILIES.items():
        for i, (s, df) in enumerate(data.items()):
            try:
                res = fn(df)
            except Exception as e:
                log(f"detector {fname}/{s}: {e}")
                continue
            for p, pair in res.items():
                sigs.setdefault((fname, p), {})[s] = pair
        log(f"signals {fname} done ({time.time()-t0:.0f}s)")

    is_masks = {s: ((df.index >= IS_START) & (df.index <= IS_END))
                for s, df in data.items()}

    cells = [(f, p, d, h) for (f, p) in sorted(sigs) for d in ("L", "S")
             for h in HORIZONS]
    log(f"{len(cells)} cells to evaluate")
    for ci, (fam, par, d, h) in enumerate(cells):
        cell = f"{fam}_{par}_{d}_h{h}"
        if cell in done:
            continue
        direction = 1 if d == "L" else -1
        allt = []
        name_means = []
        for s, (sl, ss) in sigs[(fam, par)].items():
            sig = sl if d == "L" else ss
            tr = simulate(data[s], sig, direction, h, is_masks[s])
            if tr:
                allt.extend(tr)
                name_means.append(np.mean([x[1] for x in tr]))
        if len(allt) < 30:
            row = dict(cell=cell, family=fam, params=par, dir=d, h=h, n=len(allt),
                       gross_bps="", net_bps="", t="", win="",
                       pct_names_pos="", pct_years_pos="", yrs="")
        else:
            nets = np.array([x[1] for x in allt])
            grosses = np.array([x[2] for x in allt])
            yr = {}
            for y, nv, _ in allt:
                yr.setdefault(y, []).append(nv)
            yr_means = {y: np.mean(v) for y, v in yr.items()}
            row = dict(
                cell=cell, family=fam, params=par, dir=d, h=h, n=len(nets),
                gross_bps=round(grosses.mean() * 1e4, 1),
                net_bps=round(nets.mean() * 1e4, 1),
                t=round(nets.mean() / (nets.std() / np.sqrt(len(nets)) + 1e-12), 2),
                win=round((nets > 0).mean(), 3),
                pct_names_pos=round(np.mean([m > 0 for m in name_means]), 2),
                pct_years_pos=round(np.mean([m > 0 for m in yr_means.values()]), 2),
                yrs=len(yr_means))
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        if (ci + 1) % 8 == 0:
            log(f"cells {ci+1}/{len(cells)} ({time.time()-t0:.0f}s)")
    log("G1 SCREEN DONE")


if __name__ == "__main__":
    main()
