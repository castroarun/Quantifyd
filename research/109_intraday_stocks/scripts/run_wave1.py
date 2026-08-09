#!/usr/bin/env python3
"""research/109 wave-1: 9 intraday signal families, 150 names, streaming.

Per symbol: iterate days, evaluate all family cells, record per-trade
(cell, date, entry_slot, gross_ret_to_eod). Costs and time-matched baseline
applied at aggregation. Checkpointed per symbol; safe to rerun.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT))
DB = ROOT / "backtest_data" / "market_data.db"
RES = STUDY / "results"
RES.mkdir(parents=True, exist_ok=True)
TRADES = RES / "wave1_trades.csv"
DONE = RES / "wave1_done_syms.txt"
SLOTS = RES / "wave1_slots.csv"          # per entry-slot baseline sums

WIN_END = "2023-12-31"                    # IS+Val only; OOS 2024+ untouched


def log(m):
    print(f"{datetime.now().isoformat(timespec='seconds')} {m}", flush=True)


def universe():
    from services.data_manager import FNO_LOT_SIZES
    fno = sorted(FNO_LOT_SIZES.keys())
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT symbol, count(*) c FROM market_data_unified "
        "WHERE timeframe='5minute' GROUP BY symbol ORDER BY c DESC").fetchall()
    con.close()
    extra = [s for s, c in rows if s not in fno and c > 100000
             and not s.startswith("NIFTY")
             and s not in ("INDIAVIX", "BANKNIFTY", "SENSEX")][:76]
    return fno + extra


def load5(sym):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' AND date<=? ORDER BY date",
        con, params=(sym, WIN_END + " 23:59"))
    con.close()
    if len(df) < 30000:
        return None
    df["date"] = pd.to_datetime(df["date"].str[:19])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return df


def day_signals(o, h, l, c, v, ts_min, prev):
    """Return list of (cell, entry_bar_idx). ts_min = minutes since 09:15
    per bar. prev = dict(H,L,C) of previous day. Long cells end _L etc."""
    out = []
    n = len(o)
    if n < 20:
        return out
    tp = (h + l + c) / 3.0
    cum_v = np.cumsum(v)
    vwap = np.cumsum(tp * v) / np.maximum(cum_v, 1)
    run_hi = np.maximum.accumulate(h)
    run_lo = np.minimum.accumulate(l)

    def first_idx(mask, lo_i=0):
        idx = np.flatnonzero(mask)
        idx = idx[idx >= lo_i]
        return int(idx[0]) if len(idx) else None

    # 1 TREND_BRK: after T, close breaks the pre-T day high (mirror low)
    for tmin, tag in ((105, "11"), (195, "1230")):
        pre = ts_min < tmin
        if pre.sum() < 5 or (~pre).sum() < 6:
            continue
        preH, preL = h[pre].max(), l[pre].min()
        i = first_idx((ts_min >= tmin) & (c > preH))
        if i is not None:
            out.append((f"TRBRK{tag}_L", i))
        i = first_idx((ts_min >= tmin) & (c < preL))
        if i is not None:
            out.append((f"TRBRK{tag}_S", i))

    # 2 VWAP_RECLAIM: >=k bars strictly below vwap then close above
    below = c < vwap
    for k in (6, 12):
        run = 0
        for i in range(n - 6):
            if below[i]:
                run += 1
            else:
                if run >= k and c[i] > vwap[i]:
                    out.append((f"VWREC{k}_L", i))
                    break
                run = 0
        run = 0
        for i in range(n - 6):
            if not below[i]:
                run += 1
            else:
                if run >= k and c[i] < vwap[i]:
                    out.append((f"VWREC{k}_S", i))
                    break
                run = 0

    # 3 VWAP_FADE: extension from vwap
    for x, tag in ((0.015, "15"), (0.025, "25")):
        i = first_idx((c > vwap * (1 + x)) & (ts_min >= 30))
        if i is not None:
            out.append((f"VWFADE{tag}_S", i))
        i = first_idx((c < vwap * (1 - x)) & (ts_min >= 30))
        if i is not None:
            out.append((f"VWFADE{tag}_L", i))

    # 4 CPR_TREND (needs prev day)
    if prev:
        P = (prev["H"] + prev["L"] + prev["C"]) / 3
        BC = (prev["H"] + prev["L"]) / 2
        TC = 2 * P - BC
        width = abs(TC - BC) / prev["C"]
        if width < 0.003:
            f15 = ts_min < 15
            if f15.sum() >= 3:
                c15 = c[f15][-1]
                o15 = o[0]
                if o15 > max(TC, BC) and c15 > o15:
                    out.append(("CPRTR_L", int(np.flatnonzero(f15)[-1])))
                elif o15 < min(TC, BC) and c15 < o15:
                    out.append(("CPRTR_S", int(np.flatnonzero(f15)[-1])))
        # 5 PIV_BOUNCE
        S1, R1 = 2 * P - prev["H"], 2 * P - prev["L"]
        tag_l = (l <= S1 * 1.0015) & (c > o) & (c > S1)
        i = first_idx(tag_l, 3)
        if i is not None:
            out.append(("PIVB_L", i))
        tag_s = (h >= R1 * 0.9985) & (c < o) & (c < R1)
        i = first_idx(tag_s, 3)
        if i is not None:
            out.append(("PIVB_S", i))
        # 8 PD_LEVEL
        tag_l = (l <= prev["L"] * 1.0015) & (c > o)
        i = first_idx(tag_l, 3)
        if i is not None:
            out.append(("PDL_L", i))
        tag_s = (h >= prev["H"] * 0.9985) & (c < o)
        i = first_idx(tag_s, 3)
        if i is not None:
            out.append(("PDH_S", i))

    # 6 MORN_FADE: first-hour move, then 15m (3-bar) reversal close
    fh = ts_min < 60
    if fh.sum() >= 6:
        mv = c[fh][-1] / o[0] - 1
        for x, tag in ((0.015, "15"), (0.025, "25")):
            if mv <= -x:
                for i in range(12, min(n - 6, 36)):
                    if c[i] > c[i - 3]:
                        out.append((f"MFADE{tag}_L", i))
                        break
            elif mv >= x:
                for i in range(12, min(n - 6, 36)):
                    if c[i] < c[i - 3]:
                        out.append((f"MFADE{tag}_S", i))
                        break

    # 7 FAILED_ORB: OR15 break then close back inside within 6 bars -> fade
    if n > 12:
        orh, orl = h[:3].max(), l[:3].min()
        brk = first_idx(c > orh, 3)
        if brk is not None:
            for j in range(brk + 1, min(brk + 7, n - 6)):
                if c[j] < orh:
                    out.append(("FORB_S", j))
                    break
        brk = first_idx(c < orl, 3)
        if brk is not None:
            for j in range(brk + 1, min(brk + 7, n - 6)):
                if c[j] > orl:
                    out.append(("FORB_L", j))
                    break

    # 9 LATE_MOM: 14:15 position in day range
    late = np.flatnonzero(ts_min >= 300)
    if len(late) and n - late[0] >= 4:
        i = int(late[0])
        rng = run_hi[i] - run_lo[i]
        if rng > 0:
            pos = (c[i] - run_lo[i]) / rng
            if pos >= 0.8:
                out.append(("LATEMOM_L", i))
            elif pos <= 0.2:
                out.append(("LATEMOM_S", i))
    return out


def main():
    done = set()
    if DONE.exists():
        done = set(DONE.read_text().split())
    uni = universe()
    log(f"universe {len(uni)}; {len(done)} already done")
    new_files = not TRADES.exists()
    ftr = open(TRADES, "a")
    if new_files:
        ftr.write("symbol,cell,date,slot,gross_ret\n")
    fsl = open(SLOTS, "a")
    if not SLOTS.exists() or os.path.getsize(SLOTS) == 0:
        fsl.write("symbol,date,slot,ret_to_eod\n")
    t0 = time.time()
    for k, sym in enumerate(uni):
        if sym in done:
            continue
        df = load5(sym)
        if df is None:
            DONE.write_text(DONE.read_text() + f"{sym}\n" if DONE.exists() else f"{sym}\n")
            continue
        df["day"] = df["date"].dt.normalize()
        prev = None
        n_tr = 0
        for day, dd in df.groupby("day"):
            o = dd["open"].to_numpy()
            h = dd["high"].to_numpy()
            l = dd["low"].to_numpy()
            c = dd["close"].to_numpy()
            v = dd["volume"].to_numpy(float)
            tm = ((dd["date"] - day).dt.total_seconds() / 60 - 555).to_numpy()
            n = len(dd)
            if n >= 20:
                eod = c[max(0, n - 4)]           # ~15:15 close
                sigs = day_signals(o, h, l, c, v, tm, prev)
                dstr = day.date().isoformat()
                for cell, i in sigs:
                    if i + 1 >= n - 4:
                        continue
                    e = o[i + 1]
                    if e <= 0:
                        continue
                    gross = eod / e - 1
                    slot = int(tm[i + 1] // 15)   # 15-min slot bucket
                    ftr.write(f"{sym},{cell},{dstr},{slot},{gross:.6f}\n")
                    n_tr += 1
                # baseline: slot-level to-EOD return, every 6th bar
                for i in range(1, n - 5, 6):
                    fsl.write(f"{sym},{dstr},{int(tm[i]//15)},{(eod/o[i]-1):.6f}\n")
            prev = {"H": dd["high"].max(), "L": dd["low"].min(), "C": c[-1]}
        ftr.flush()
        fsl.flush()
        with open(DONE, "a") as fd:
            fd.write(f"{sym}\n")
        log(f"[{k+1}/{len(uni)}] {sym}: {n_tr} trades ({time.time()-t0:.0f}s)")
    ftr.close()
    fsl.close()
    log("WAVE1 SIGNALS DONE")


if __name__ == "__main__":
    main()
