#!/usr/bin/env python3
"""research/110 wave-1: cross-sectional / flow-proxy intraday signals.

Phase 1: stream 150 names -> per-(name,day) features by 10:15.
Phase 2: cross-sectional evaluation of 14 pre-registered cells.
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "109_intraday_stocks" / "scripts"))
from run_wave1 import universe, load5                  # noqa: E402

RES = STUDY / "results"
RES.mkdir(parents=True, exist_ok=True)
FEAT = RES / "features.csv"
DONE = RES / "done_syms.txt"
COST = 0.0010
IS_END = pd.Timestamp("2021-09-30")
H1_END = pd.Timestamp("2018-06-30")


def log(m):
    print(f"{datetime.now().isoformat(timespec='seconds')} {m}", flush=True)


def build_features():
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    uni = universe()
    new = not FEAT.exists()
    f = open(FEAT, "a")
    if new:
        f.write("symbol,date,gap,mret,rvol,volimb,clv,e1015,eod\n")
    t0 = time.time()
    for k, sym in enumerate(uni):
        if sym in done:
            continue
        df = load5(sym)
        if df is None:
            with open(DONE, "a") as fd:
                fd.write(f"{sym}\n")
            continue
        df["day"] = df["date"].dt.normalize()
        fh_vol_hist = deque(maxlen=20)
        prev_close = None
        for day, dd in df.groupby("day"):
            o = dd["open"].to_numpy()
            h = dd["high"].to_numpy()
            l = dd["low"].to_numpy()
            c = dd["close"].to_numpy()
            v = dd["volume"].to_numpy(float)
            tm = ((dd["date"] - day).dt.total_seconds() / 60 - 555).to_numpy()
            n = len(dd)
            if n < 30 or prev_close is None or prev_close <= 0:
                prev_close = c[-1] if n else prev_close
                continue
            fh = tm < 60
            e_idx = np.flatnonzero(tm >= 60)
            if fh.sum() < 8 or len(e_idx) < 10:
                prev_close = c[-1]
                continue
            i0 = int(e_idx[0])
            e1015 = o[i0]
            eod = c[max(0, n - 4)]
            gap = o[0] / prev_close - 1
            mret = c[i0 - 1] / o[0] - 1
            fh_vol = v[fh].sum()
            rvol = fh_vol / (np.mean(fh_vol_hist) + 1e-9) if len(fh_vol_hist) >= 10 else np.nan
            fh_vol_hist.append(fh_vol)
            up = c[fh] >= o[fh]
            volimb = v[fh][up].sum() / (fh_vol + 1e-9)
            rng = h[fh] - l[fh]
            clv_bars = np.where(rng > 0, (c[fh] - l[fh]) / np.where(rng > 0, rng, 1), 0.5)
            clv = float(np.mean(clv_bars))
            if e1015 > 0 and eod > 0 and not np.isnan(rvol):
                f.write(f"{sym},{day.date()},{gap:.5f},{mret:.5f},{rvol:.3f},"
                        f"{volimb:.3f},{clv:.3f},{e1015:.2f},{eod:.2f}\n")
            prev_close = c[-1]
        f.flush()
        with open(DONE, "a") as fd:
            fd.write(f"{sym}\n")
        if (k + 1) % 20 == 0:
            log(f"[{k+1}/{len(uni)}] features ({time.time()-t0:.0f}s)")
    f.close()
    log("FEATURES DONE")


def evaluate():
    d = pd.read_csv(FEAT, parse_dates=["date"])
    d["ret"] = d["eod"] / d["e1015"] - 1
    d = d[d["date"] <= pd.Timestamp("2023-12-31")]
    d["day_mean"] = d.groupby("date")["ret"].transform("mean")
    d["xret"] = d["ret"] - d["day_mean"]
    d["mrank"] = d.groupby("date")["mret"].rank(pct=True)
    cells = {}

    def add(name, sel, direction, use_x=True):
        g = d[sel].copy()
        r = direction * (g["xret"] if use_x else g["ret"]) - COST
        cells[name] = pd.DataFrame({"date": g["date"], "symbol": g["symbol"],
                                    "net": r,
                                    "abs_net": direction * g["ret"] - COST})

    add("XMOM_topL", d.mrank >= 0.9, 1)
    add("XMOM_botS", d.mrank <= 0.1, -1)
    add("XREV_botL", d.mrank <= 0.1, 1)
    add("XREV_topS", d.mrank >= 0.9, -1)
    ev = (d.rvol >= 3) & (d.gap.abs() >= 0.01)
    add("EVENT_follow", ev, 0, use_x=True)   # placeholder replaced below
    del cells["EVENT_follow"]
    g = d[ev].copy()
    sgn = np.sign(g["gap"])
    for name, mult in (("EVENT_follow", 1), ("EVENT_fade", -1)):
        r = mult * sgn * g["xret"] - COST
        cells[name] = pd.DataFrame({"date": g["date"], "symbol": g["symbol"],
                                    "net": r,
                                    "abs_net": mult * sgn * g["ret"] - COST})
    add("VOLIMB_L", d.volimb >= 0.65, 1)
    add("VOLIMB_S", d.volimb <= 0.35, -1)
    add("CLV_L", d.clv >= 0.7, 1)
    add("CLV_S", d.clv <= 0.3, -1)
    add("RVOL_trendL", (d.rvol >= 3) & (d.mret > 0), 1)
    add("RVOL_trendS", (d.rvol >= 3) & (d.mret < 0), -1)
    # long-short spread: top decile long + bottom decile short, same day
    print(f"\n{'cell':14s} {'win':>4s} {'n':>7s} {'xnet_bps':>9s} {'t':>7s} "
          f"{'absnet':>7s} {'h1':>6s} {'h2':>6s} {'names+':>6s}")
    for name, g in cells.items():
        for wname, sel in (("IS", g["date"] <= IS_END), ("VAL", g["date"] > IS_END)):
            gg = g[sel]
            if len(gg) < 300:
                continue
            v = gg["net"].to_numpy()
            t = v.mean() / (v.std() / np.sqrt(len(v)) + 1e-12)
            h1 = gg[gg["date"] <= H1_END]["net"].mean() if wname == "IS" else np.nan
            h2 = gg[gg["date"] > H1_END]["net"].mean() if wname == "IS" else np.nan
            np_ = (gg.groupby("symbol")["net"].mean() > 0).mean()
            print(f"{name:14s} {wname:>4s} {len(v):7d} {v.mean()*1e4:9.1f} "
                  f"{t:7.2f} {gg['abs_net'].mean()*1e4:7.1f} "
                  f"{(h1 or 0)*1e4:6.1f} {(h2 or 0)*1e4:6.1f} {np_:6.2f}")
    print("ALTINFO WAVE1 DONE")


if __name__ == "__main__":
    build_features()
    evaluate()
