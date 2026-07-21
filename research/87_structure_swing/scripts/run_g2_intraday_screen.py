#!/usr/bin/env python3
"""research/87 phase 1b: structure families on 30-MIN bars, F&O universe.

Same detectors as the daily pass (bar-parameterized), MTF1 redefined as
daily-trend x 30m-breakout. Controls in the primary pass: excess vs
unconditional drift AND date-matched universe mean. IS 2015-02..2021-09.
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
sys.path.insert(0, HERE)
from run_g1_daily_screen import (fam_sr1, fam_sr2, fam_cp1, fam_cp2, fam_cp3,
                                 fam_br1, fam_br2, load_universe)

DB = os.path.join(REPO, "backtest_data", "market_data.db")
RES = os.path.join(STUDY, "results")
OUT = os.path.join(RES, "g2_intraday.csv")

WIN_START = os.getenv("R87B_WIN_START", "2015-02-01")
WIN_END = os.getenv("R87B_WIN_END", "2021-09-30")
COST_RT = 0.0010
HORIZONS = [13, 39, 65, 130]          # ~1/3/5/10 sessions in 30m bars
FIELDS = ["cell", "family", "params", "dir", "h", "n", "net_bps", "t_abs",
          "rel_bps", "rel_net_bps", "t_rel", "pct_names_pos_rel", "drift_bps"]


def log(m):
    print(f"{datetime.now().isoformat(timespec='seconds')} {m}", flush=True)


def load_30m(sym):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=? ORDER BY date",
        con, params=(sym, "2015-01-01", WIN_END + " 23:59"))
    con.close()
    if len(df) < 5000:
        return None
    df["date"] = pd.to_datetime(df["date"].str[:19])
    df = df.set_index("date").sort_index()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    g = df.resample("30min", origin="start_day", offset="9h15min")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum()}).dropna()
    return out if len(out) > 3000 else None


def fam_mtf1_intraday(df):
    """Daily trend (close > 30-day SMA of daily closes; mirror <) AND 30m
    breakout of {26,65}-bar extreme (~2/5 sessions)."""
    c = df["close"]
    dly = c.resample("1D").last().dropna()
    dsma = dly.rolling(30).mean()
    up_d = (dly > dsma).shift(1)          # last COMPLETED day
    up = up_d.reindex(df.index.normalize()).fillna(False).to_numpy()
    out = {}
    for dtrig in (26, 65):
        hh = df["high"].rolling(dtrig).max().shift(1).to_numpy()
        ll = df["low"].rolling(dtrig).min().shift(1).to_numpy()
        cn = c.to_numpy()
        sl = (cn > hh) & up
        ss = (cn < ll) & ~up
        out[f"d{dtrig}"] = (sl, ss)
    return out


FAMILIES = {
    "SR1": fam_sr1, "SR2": fam_sr2, "CP1": fam_cp1, "CP2": fam_cp2,
    "CP3": fam_cp3, "MTF1": fam_mtf1_intraday, "BR1": fam_br1, "BR2": fam_br2,
}


def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {r["cell"] for r in csv.DictReader(f)}
        log(f"resume: {len(done)} done")
    else:
        with open(OUT, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    t0 = time.time()
    uni = load_universe()
    data = {}
    for s in uni:
        d = load_30m(s)
        if d is not None:
            data[s] = d
    log(f"loaded {len(data)} names, 30m bars ({time.time()-t0:.0f}s)")

    masks = {s: ((df.index >= WIN_START) & (df.index <= WIN_END + " 23:59"))
             for s, df in data.items()}

    # forward open-to-open returns per horizon; date-matched universe mean
    fwd = {}
    dmean = {}
    for h in HORIZONS:
        frames = []
        for s, df in data.items():
            o = df["open"].to_numpy()
            r = np.full(len(o), np.nan)
            if len(o) > h + 1:
                r[:-(h + 1)] = o[h + 1:] / o[1:len(o) - h] - 1
            frames.append(pd.Series(r, index=df.index, name=s))
        wide = pd.concat(frames, axis=1)
        fwd[h] = wide
        dmean[h] = wide.mean(axis=1)
        base = wide[(wide.index >= WIN_START) & (wide.index <= WIN_END)]
        log(f"h={h}: drift={np.nanmean(base.to_numpy())*1e4:.1f}bps")

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

    cells = [(f, p, d, h) for (f, p) in sorted(sigs) for d in ("L", "S")
             for h in HORIZONS]
    log(f"{len(cells)} cells")
    for ci, (fam, par, dname, h) in enumerate(cells):
        cell = f"{fam}_{par}_{dname}_h{h}"
        if cell in done:
            continue
        di = 1 if dname == "L" else -1
        wide = fwd[h]
        dm = dmean[h]
        drift = np.nanmean(wide[(wide.index >= WIN_START) & (wide.index <= WIN_END)].to_numpy())
        raws, rels = [], []
        name_rel = {}
        for s, (sl, ss) in sigs[(fam, par)].items():
            sig = sl if dname == "L" else ss
            df = data[s]
            fs = wide[s].to_numpy()
            dms = dm.reindex(df.index).to_numpy()
            m = masks[s]
            idx = np.flatnonzero(sig)
            busy = -1
            for t in idx:
                if t <= busy or not m[t] or np.isnan(fs[t]) or np.isnan(dms[t]):
                    continue
                raw = di * fs[t]
                rel = di * (fs[t] - dms[t])
                raws.append(raw)
                rels.append(rel)
                name_rel.setdefault(s, []).append(rel)
                busy = t + h
        row = dict(cell=cell, family=fam, params=par, dir=dname, h=h, n=len(raws),
                   net_bps="", t_abs="", rel_bps="", rel_net_bps="", t_rel="",
                   pct_names_pos_rel="", drift_bps=round(drift * 1e4, 1))
        if len(raws) >= 100:
            ra = np.array(raws)
            re = np.array(rels)
            row.update(
                net_bps=round((ra.mean() - COST_RT) * 1e4, 1),
                t_abs=round((ra.mean() - COST_RT) / (ra.std() / np.sqrt(len(ra)) + 1e-12), 2),
                rel_bps=round(re.mean() * 1e4, 1),
                rel_net_bps=round((re.mean() - COST_RT) * 1e4, 1),
                t_rel=round(re.mean() / (re.std() / np.sqrt(len(re)) + 1e-12), 2),
                pct_names_pos_rel=round(np.mean([np.mean(v) > 0 for v in name_rel.values()]), 2))
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        if (ci + 1) % 8 == 0:
            log(f"cells {ci+1}/{len(cells)} ({time.time()-t0:.0f}s)")
    log("G2 INTRADAY DONE")


if __name__ == "__main__":
    main()
