#!/usr/bin/env python3
"""research/88 G1: GCO-Pullback-Stoch on daily bars, F&O universe.

Setup: fresh 20/50 golden cross -> 1st pullback (close < sma20) -> green
candle with stoch(14,3,3) %K x-up %D at %K<40 -> long at next open, SL below
trigger low(s). Mirror for shorts. Exits: 2R target / sma20 trail / time-15.
Control: random-entry baseline with identical exit mechanics (drift control).
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
OUT = os.path.join(RES, "gco_g1.csv")

WIN_START = os.getenv("R88_WIN_START", "2005-01-01")
WIN_END = os.getenv("R88_WIN_END", "2017-12-31")
COST_RT = 0.0010
FIELDS = ["cell", "F", "sl", "exit", "dir", "n", "net_bps", "t", "win",
          "pct_names_pos", "avg_hold", "avg_R"]


def log(m):
    print(f"{datetime.now().isoformat(timespec='seconds')} {m}", flush=True)


def load_universe():
    from services.data_manager import FNO_LOT_SIZES
    return sorted(FNO_LOT_SIZES.keys())


def load_daily(sym):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' AND date>=? AND date<=? ORDER BY date",
        con, params=(sym, "2000-01-01", WIN_END + " 23:59"))
    con.close()
    if len(df) < 400:
        return None
    df["date"] = pd.to_datetime(df["date"].str[:10])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if len(df) < 400:
        return None
    return df.set_index("date")


def stoch_kd(df, n=14, k_smooth=3, d_smooth=3):
    ll = df["low"].rolling(n).min()
    hh = df["high"].rolling(n).max()
    raw = 100 * (df["close"] - ll) / (hh - ll + 1e-12)
    k = raw.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k.to_numpy(), d.to_numpy()


def triggers(df, F, direction):
    """Return list of (trigger_bar, stop1, stop2). One trigger per cross."""
    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    s20 = df["close"].rolling(20).mean().to_numpy()
    s50 = df["close"].rolling(50).mean().to_numpy()
    K, D = stoch_kd(df)
    n = len(c)
    out = []
    state = 0          # 0 idle, 1 fresh-cross, 2 pullback-seen
    cross_bar = -1
    for t in range(51, n):
        if direction == 1:
            crossed = s20[t] > s50[t] and s20[t - 1] <= s50[t - 1]
            uncross = s20[t] < s50[t]
        else:
            crossed = s20[t] < s50[t] and s20[t - 1] >= s50[t - 1]
            uncross = s20[t] > s50[t]
        if crossed:
            state, cross_bar = 1, t
            continue
        if state == 0:
            continue
        if uncross or (t - cross_bar > F and state == 1):
            state = 0
            continue
        if state == 1:
            pull = c[t] < s20[t] if direction == 1 else c[t] > s20[t]
            if pull:
                state = 2
            continue
        # state 2: waiting for trigger candle
        if t - cross_bar > F + 30:
            state = 0
            continue
        if direction == 1:
            trig = (c[t] > o[t] and K[t] > D[t] and K[t - 1] <= D[t - 1]
                    and K[t] < 40)
            if trig:
                out.append((t, l[t], min(l[t], l[t - 1])))
                state = 0
        else:
            trig = (c[t] < o[t] and K[t] < D[t] and K[t - 1] >= D[t - 1]
                    and K[t] > 60)
            if trig:
                out.append((t, h[t], max(h[t], h[t - 1])))
                state = 0
    return out


def run_trade(df, t, stop, direction, exit_mode, s20):
    """Enter open[t+1]; simulate to exit. Returns (net, hold, R) or None."""
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(o)
    if t + 2 >= n:
        return None
    e = o[t + 1]
    risk = (e - stop) * direction
    if risk <= 0 or risk / e > 0.15:
        return None
    target = e + direction * 2 * risk if exit_mode == "E1" else None
    cap = 30 if exit_mode == "E1" else (15 if exit_mode == "E3" else 60)
    for j in range(t + 1, min(t + 1 + cap, n - 1)):
        # gap through stop at open
        if direction == 1 and o[j] <= stop:
            x = o[j]
            return ((x / e - 1) - COST_RT, j - t, (x - e) / risk)
        if direction == -1 and o[j] >= stop:
            x = o[j]
            return ((1 - x / e) - COST_RT, j - t, (e - x) / risk)
        # intrabar stop (precedence over target)
        if direction == 1 and l[j] <= stop:
            return ((stop / e - 1) - COST_RT, j - t, -1.0)
        if direction == -1 and h[j] >= stop:
            return ((1 - stop / e) - COST_RT, j - t, -1.0)
        # target
        if target is not None:
            if direction == 1 and h[j] >= target:
                return ((target / e - 1) - COST_RT, j - t, 2.0)
            if direction == -1 and l[j] <= target:
                return ((1 - target / e) - COST_RT, j - t, 2.0)
        # trail: close crosses back through sma20
        if exit_mode == "E2" and j > t + 1:
            if (direction == 1 and c[j] < s20[j]) or (direction == -1 and c[j] > s20[j]):
                if j + 1 < n:
                    x = o[j + 1]
                    return ((direction * (x / e - 1)) - COST_RT, j + 1 - t,
                            direction * (x - e) / risk)
    j = min(t + cap, n - 1)
    x = o[j]
    return ((direction * (x / e - 1)) - COST_RT, j - t, direction * (x - e) / risk)


def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {r["cell"] for r in csv.DictReader(f)}
    else:
        with open(OUT, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    uni = load_universe()
    data = {}
    for s in uni:
        d = load_daily(s)
        if d is not None:
            data[s] = d
    log(f"loaded {len(data)} names | window {WIN_START}..{WIN_END}")
    s20s = {s: df["close"].rolling(20).mean().to_numpy() for s, df in data.items()}
    masks = {s: ((df.index >= WIN_START) & (df.index <= WIN_END))
             for s, df in data.items()}

    t0 = time.time()
    cells = [(F, sl, ex, d) for F in (10, 20) for sl in ("SL1", "SL2")
             for ex in ("E1", "E2", "E3") for d in ("L", "S")]
    for F, sl, ex, dname in cells:
        cell = f"GCO_F{F}_{sl}_{ex}_{dname}"
        if cell in done:
            continue
        di = 1 if dname == "L" else -1
        allt = []
        name_means = {}
        for s, df in data.items():
            trigs = triggers(df, F, di)
            for (t, st1, st2) in trigs:
                if t + 1 >= len(df) or not masks[s][t + 1]:
                    continue
                stop = st1 if sl == "SL1" else st2
                r = run_trade(df, t, stop, di, ex, s20s[s])
                if r:
                    allt.append(r)
                    name_means.setdefault(s, []).append(r[0])
        row = dict(cell=cell, F=F, sl=sl, exit=ex, dir=dname, n=len(allt),
                   net_bps="", t="", win="", pct_names_pos="", avg_hold="", avg_R="")
        if len(allt) >= 30:
            nets = np.array([x[0] for x in allt])
            row.update(net_bps=round(nets.mean() * 1e4, 1),
                       t=round(nets.mean() / (nets.std() / np.sqrt(len(nets)) + 1e-12), 2),
                       win=round((nets > 0).mean(), 3),
                       pct_names_pos=round(np.mean([np.mean(v) > 0 for v in name_means.values()]), 2),
                       avg_hold=round(np.mean([x[1] for x in allt]), 1),
                       avg_R=round(np.mean([x[2] for x in allt]), 2))
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        log(f"{cell}: n={row['n']} net={row['net_bps']} t={row['t']}")

    # random-entry baselines: every 10th bar, stop at median 4% distance
    for ex in ("E1", "E2", "E3"):
        for dname in ("L", "S"):
            cell = f"BASE_{ex}_{dname}"
            if cell in done:
                continue
            di = 1 if dname == "L" else -1
            allt = []
            name_means = {}
            for s, df in data.items():
                o = df["open"].to_numpy()
                for t in range(60, len(df) - 2, 10):
                    if not masks[s][t + 1]:
                        continue
                    e = o[t + 1]
                    stop = e * (1 - di * 0.04)
                    r = run_trade(df, t, stop, di, ex, s20s[s])
                    if r:
                        allt.append(r)
                        name_means.setdefault(s, []).append(r[0])
            nets = np.array([x[0] for x in allt])
            row = dict(cell=cell, F="", sl="med4pct", exit=ex, dir=dname, n=len(allt),
                       net_bps=round(nets.mean() * 1e4, 1),
                       t=round(nets.mean() / (nets.std() / np.sqrt(len(nets)) + 1e-12), 2),
                       win=round((nets > 0).mean(), 3),
                       pct_names_pos=round(np.mean([np.mean(v) > 0 for v in name_means.values()]), 2),
                       avg_hold=round(np.mean([x[1] for x in allt]), 1),
                       avg_R=round(np.mean([x[2] for x in allt]), 2))
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            log(f"{cell}: n={row['n']} net={row['net_bps']} t={row['t']}")
    log(f"GCO G1 DONE ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
