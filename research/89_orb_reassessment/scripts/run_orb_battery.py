#!/usr/bin/env python3
"""research/89: ORB reassessment battery.

B  — decay trace: locked r/81 config, net bps per quarter 2015..2026-07,
     regime splits (NIFTY>50DMA, VIX terciles).
C1 — live-replica autopsy cells on 2024-01..2026-07.
C2 — revival grid, per-period gates (2015-21 / 2022-23 / 2024-26).

Reuses the r/81 engine. All 2024-26 results are in-sample by construction
(OOS consumed 2026-07-16) — candidates go to paper-forward only.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
R81 = STUDY.parent / "81_swing_edge_discovery"
ROOT = STUDY.parents[1]
for p in (str(R81), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("ENGINE_MAX_TIME_STOP", "4")

from engine import loader                                # noqa: E402
from engine.backtester import BTConfig, run_symbol       # noqa: E402
from engine.costs import CostConfig                      # noqa: E402

RES = STUDY / "results"
RES.mkdir(parents=True, exist_ok=True)
OUT = RES / "orb_battery.csv"
QOUT = RES / "decay_quarters.csv"

CFG3 = BTConfig(cost=CostConfig(product="FUTURES_PROXY", slippage=0.0003))
CFG3_TS2 = BTConfig(cost=CostConfig(product="FUTURES_PROXY", slippage=0.0003),
                    time_stop_sessions=2)
STOCKS = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "TCS", "SBIN",
          "AXISBANK", "KOTAKBANK", "LT", "ITC", "BHARTIARTL", "MARUTI",
          "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL",
          "ADANIENT", "ADANIPORTS", "BAJFINANCE", "BAJAJFINSV", "HCLTECH",
          "WIPRO", "TECHM", "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB",
          "TITAN", "ASIANPAINT", "ULTRACEMCO", "GRASIM", "NTPC", "POWERGRID",
          "ONGC", "COALINDIA", "BPCL", "HINDUNILVR", "NESTLEIND",
          "BRITANNIA", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "M&M",
          "APOLLOHOSP", "INDUSINDBK", "SBILIFE", "HDFCLIFE", "DLF",
          "GODREJPROP", "AMBUJACEM", "SHREECEM", "PIDILITIND", "SIEMENS",
          "ABB", "BEL", "HAL", "TRENT", "DMART", "NAUKRI", "ZOMATO",
          "PAYTM", "PNB", "BANKBARODA", "CANBK", "FEDERALBNK", "IDFCFIRSTB",
          "AUBANK", "CHOLAFIN", "LICHSGFIN", "MUTHOOTFIN", "SRF", "UPL",
          "TATAPOWER", "TORNTPOWER", "GAIL", "IGL"]
FIELDS = ["cell", "phase", "params", "period", "n", "net_bps", "t", "win",
          "pct_names_pos"]


def log(m):
    print(f"{datetime.now().isoformat(timespec='seconds')} {m}", flush=True)


def orb_entries_long(df5, w, gap_thr, lo, hi):
    sess = df5.index.normalize()
    g = df5.groupby(sess)
    pos = g.cumcount()
    or_high = df5["high"].where(pos < w).groupby(sess).transform("max")
    or_low = df5["low"].where(pos < w).groupby(sess).transform("min")
    raw = (pos >= w) & (df5["close"] > or_high)
    first = raw & (raw.groupby(sess).cumsum() == 1)
    sig = df5.index[first]
    sig = sig[(sig >= pd.Timestamp(lo)) & (sig <= pd.Timestamp(hi + " 23:59:59"))]
    d = loader.to_daily(df5)
    gap = d["open"] / d["close"].shift(1) - 1
    ok_days = set(d.index[gap >= gap_thr])
    sig = pd.DatetimeIndex([t for t in sig if t.normalize() in ok_days])
    return pd.DataFrame({"direction": 1, "stop": or_low.loc[sig],
                         "target": np.nan}, index=sig)


def rsi15(df5, n=14):
    """15-min RSI resampled back onto the 5-min index (causal ffill)."""
    df15 = loader.resample_intraday(df5, 15)
    delta = df15["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    r = 100 - 100 / (1 + up / (dn + 1e-12))
    return r.reindex(df5.index, method="ffill")


def live_replica_trades(df5, lo, hi, direction_mode, use_rsi,
                        w_bars=3, gap_thr=None):
    """Approximate the production rules: OR of w_bars 5-min bars, first
    breakout each side, optional daily-gap filter, intraday only: 1.5R
    target, OR-opposite SL, 15:15 squareoff. Net returns incl. costs."""
    sess_all = df5.index.normalize()
    rsi = rsi15(df5) if use_rsi else None
    ok_days = None
    if gap_thr is not None:
        d = loader.to_daily(df5)
        gap = d["open"] / d["close"].shift(1) - 1
        ok_days = set(d.index[gap >= gap_thr])
    out = []
    for day, dd in df5.groupby(sess_all):
        if not (pd.Timestamp(lo) <= day <= pd.Timestamp(hi)):
            continue
        if ok_days is not None and day not in ok_days:
            continue
        if len(dd) < 10:
            continue
        orh = dd["high"].iloc[:w_bars].max()
        orl = dd["low"].iloc[:w_bars].min()
        o = dd["open"].to_numpy()
        h = dd["high"].to_numpy()
        l = dd["low"].to_numpy()
        c = dd["close"].to_numpy()
        for di, brk, stp in ((1, orh, orl), (-1, orl, orh)):
            if direction_mode == "L" and di == -1:
                continue
            ei = None
            for i in range(w_bars, len(dd) - 1):
                cond = c[i] > brk if di == 1 else c[i] < brk
                if cond:
                    if use_rsi:
                        rv = rsi.loc[dd.index[i]]
                        if np.isnan(rv) or (di == 1 and rv < 60) or (di == -1 and rv > 40):
                            continue
                    ei = i
                    break
            if ei is None:
                continue
            e = o[ei + 1] * (1 + di * 0.0003)          # next bar open + slip
            risk = (e - stp) * di
            if risk <= 0:
                continue
            tgt = e + di * 1.5 * risk
            x = None
            for j in range(ei + 1, len(dd)):
                if di == 1 and l[j] <= stp:
                    x = stp
                    break
                if di == -1 and h[j] >= stp:
                    x = stp
                    break
                if di == 1 and h[j] >= tgt:
                    x = tgt
                    break
                if di == -1 and l[j] <= tgt:
                    x = tgt
                    break
            if x is None:
                x = c[-3] if len(dd) > 3 else c[-1]    # ~15:15 squareoff
            gross = di * (x / e - 1)
            out.append(gross - 0.0010)                 # RT costs+slip approx
    return out


def write_row(row):
    exists = OUT.exists()
    with open(OUT, "a", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            wcsv.writeheader()
        wcsv.writerow(row)


def agg(rets_by_name, cell, phase, params, period):
    allr = np.concatenate([v for v in rets_by_name.values() if len(v)]) \
        if rets_by_name else np.array([])
    if len(allr) < 20:
        return dict(cell=cell, phase=phase, params=params, period=period,
                    n=len(allr), net_bps="", t="", win="", pct_names_pos="")
    t = allr.mean() / (allr.std(ddof=1) / np.sqrt(len(allr)) + 1e-12)
    return dict(cell=cell, phase=phase, params=params, period=period,
                n=len(allr), net_bps=round(allr.mean() * 1e4, 1),
                t=round(t, 2), win=round((allr > 0).mean(), 3),
                pct_names_pos=round(np.mean([np.mean(v) > 0
                                             for v in rets_by_name.values()
                                             if len(v)]), 2))


def main():
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {(r["cell"], r["period"]) for r in csv.DictReader(f)}
        log(f"resume: {len(done)} rows done")

    t0 = time.time()
    data = {}
    for s in STOCKS:
        try:
            df = loader.load_bars(s)
            if df is not None and len(df) > 20000:
                data[s] = df
        except Exception as e:
            log(f"load {s}: {e}")
    log(f"loaded {len(data)} names ({time.time()-t0:.0f}s)")

    # ---------------- B: decay trace, locked config -------------------
    if ("B_locked", "quarters") not in done:
        qrets = {}
        for s, df5 in data.items():
            ent = orb_entries_long(df5, 12, 0.0025, "2015-02-01", "2026-07-21")
            tr = run_symbol(df5, ent, CFG3, symbol=s)
            for ts, r in zip(pd.to_datetime(tr["entry_time"]), tr["net_ret"]):
                q = f"{ts.year}Q{(ts.month-1)//3+1}"
                qrets.setdefault(q, []).append(r)
        with open(QOUT, "w", newline="") as f:
            wq = csv.writer(f)
            wq.writerow(["quarter", "n", "net_bps", "t"])
            for q in sorted(qrets):
                v = np.array(qrets[q])
                tq = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)) + 1e-12) \
                    if len(v) > 2 else np.nan
                wq.writerow([q, len(v), round(v.mean() * 1e4, 1),
                             round(tq, 2) if tq == tq else ""])
        write_row(dict(cell="B_locked", phase="B", params="W12_gap25_ts4_L",
                       period="quarters", n=sum(len(v) for v in qrets.values()),
                       net_bps="", t="", win="", pct_names_pos="see decay_quarters.csv"))
        log(f"B decay trace done ({time.time()-t0:.0f}s)")

    # ---------------- C1: live-replica autopsy ------------------------
    LO, HI = "2024-01-01", "2026-07-21"
    c1 = [("C1a_live_replica_LS_rsi", "LS", True),
          ("C1b_live_replica_L_rsi", "L", True),
          ("C1c_live_replica_L_norsi", "L", False)]
    for cell, dmode, ursi in c1:
        if (cell, "2024-26") in done:
            continue
        rbn = {}
        for s, df5 in data.items():
            rbn[s] = live_replica_trades(df5, LO, HI, dmode, ursi,
                                         w_bars=3, gap_thr=0.003)
        write_row(agg(rbn, cell, "C1", f"{dmode}_rsi{ursi}", "2024-26"))
        log(f"{cell} done ({time.time()-t0:.0f}s)")
    if ("C1d_research_cfg", "2024-26") not in done:
        rbn = {}
        for s, df5 in data.items():
            ent = orb_entries_long(df5, 12, 0.0025, LO, HI)
            tr = run_symbol(df5, ent, CFG3, symbol=s)
            rbn[s] = tr["net_ret"].to_numpy(float)
        write_row(agg(rbn, "C1d_research_cfg", "C1", "W12_gap25_ts4_L", "2024-26"))
        log(f"C1d done ({time.time()-t0:.0f}s)")

    # ---------------- C2: revival grid --------------------------------
    periods = [("2015-21", "2015-02-01", "2021-12-31"),
               ("2022-23", "2022-01-01", "2023-12-31"),
               ("2024-26", "2024-01-01", "2026-07-21")]
    for w in (6, 12, 18):
        for gap in (0.0025, 0.004):
            for ex in ("ts2", "ts4", "eod"):
                cell = f"C2_W{w}_g{int(gap*1e4)}_{ex}_L"
                for pname, lo, hi in periods:
                    if (cell, pname) in done:
                        continue
                    rbn = {}
                    cfg = CFG3_TS2 if ex == "ts2" else CFG3
                    for s, df5 in data.items():
                        if ex == "eod":
                            rbn[s] = live_replica_trades(df5, lo, hi, "L",
                                                         False, w_bars=w,
                                                         gap_thr=gap)
                            continue
                        ent = orb_entries_long(df5, w, gap, lo, hi)
                        tr = run_symbol(df5, ent, cfg, symbol=s)
                        rbn[s] = tr["net_ret"].to_numpy(float)
                    write_row(agg(rbn, cell, "C2", f"W{w}_gap{gap}_{ex}", pname))
                log(f"{cell} done ({time.time()-t0:.0f}s)")
    log("ORB BATTERY DONE")


if __name__ == "__main__":
    main()
