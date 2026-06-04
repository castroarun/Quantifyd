"""Adversarial kill / benchmark (playbook §6 placebo, §3 benchmark).

Is the positional edge ALPHA (from the volume+breakout signal) or just BETA
(holding large-caps ~N days in a bull market)? Compare identical positional
exits across three ENTRY arms on the same 8 names / period:

  SIGNAL     : first bar/day with vol > own-50d-MA AND close > prev-day high
  BREAK_ONLY : first bar/day with close > prev-day high (volume filter REMOVED)
  BASELINE   : enter every day at the 2nd 30-min bar open (unconditional drift)

If SIGNAL ~= BASELINE the volume+breakout entry has no alpha. Reports mean
net@6bp R + PF for HARD_SL, CHANDELIER_3ATR, SUPERTREND_D, MAXHOLD_10.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB = ROOT.parents[1] / "backtest_data" / "market_data.db"

UNIVERSE = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "SBIN", "INFY", "MARUTI", "TATASTEEL"]
START, END = "2018-01-01", "2026-06-01"
VOLMA_WIN, VOLMA_MINP = 600, 200
ATR_D_LEN, ST_PERIOD, ST_MULT, MAXHOLD = 14, 10, 3.0, 30
COST = 6.0 / 1e4
POLICIES = ["HARD_SL", "CHANDELIER_3ATR", "SUPERTREND_D_10_3", "MAXHOLD_10"]
ARMS = ["SIGNAL", "BREAK_ONLY", "BASELINE"]


def load(con, s, tf, a, b):
    df = pd.read_sql("SELECT date,open,high,low,close,volume FROM market_data_unified "
                     "WHERE symbol=? AND timeframe=? AND date>=? AND date<=? ORDER BY date",
                     con, params=(s, tf, a, b))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def to_30(df5):
    if df5.empty:
        return df5
    df = df5.copy(); df["_d"] = df.index.normalize(); rows = []
    for _, d in df.groupby("_d", sort=True):
        d = d.sort_index(); n = (len(d)//6)*6
        for i in range(0, n, 6):
            g = d.iloc[i:i+6]
            rows.append({"date": g.index[0], "open": float(g["open"].iloc[0]),
                         "high": float(g["high"].max()), "low": float(g["low"].min()),
                         "close": float(g["close"].iloc[-1]), "volume": float(g["volume"].sum())})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def atr_w(df, n):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def st_dir(df, p, m):
    atr = atr_w(df, p); hl2 = (df["high"]+df["low"])/2
    up = (hl2+m*atr); lo = (hl2-m*atr); n = len(df)
    fu = np.array(up, float); fl = np.array(lo, float)
    c = np.asarray(df["close"], float); un = np.asarray(up, float); ln = np.asarray(lo, float)
    for i in range(1, n):
        fu[i] = un[i] if (un[i] < fu[i-1] or c[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = ln[i] if (ln[i] > fl[i-1] or c[i-1] < fl[i-1]) else fl[i-1]
    d = np.ones(n, int)
    for i in range(1, n):
        d[i] = 1 if c[i] > fu[i-1] else (-1 if c[i] < fl[i-1] else d[i-1])
    return pd.Series(d, index=df.index)


def sim(entry, atrv, rem, dfut, sfut):
    R = atrv; stop0 = entry-R
    dayD = bool((rem["low"] <= stop0).any()) if len(rem) else False
    lastc = float(dfut["close"].iloc[-1]) if len(dfut) else float(rem["close"].iloc[-1])
    hi = dfut["high"].to_numpy(); lo = dfut["low"].to_numpy(); cl = dfut["close"].to_numpy(); sd = sfut.to_numpy()
    out = {}
    # HARD_SL
    px = stop0 if dayD else lastc
    if not dayD:
        for k in range(len(dfut)):
            if lo[k] <= stop0:
                px = stop0; break
    out["HARD_SL"] = px
    # CHANDELIER
    px = lastc; hh = max(entry, float(rem["high"].max()) if len(rem) else entry)
    for k in range(len(dfut)):
        tr = hh-3*R
        if lo[k] <= tr:
            px = tr; break
        hh = max(hh, hi[k])
    out["CHANDELIER_3ATR"] = px
    # SUPERTREND daily
    px = lastc
    for k in range(len(dfut)):
        if sd[k] == -1:
            px = float(cl[k]); break
    out["SUPERTREND_D_10_3"] = px
    # MAXHOLD_10 (+ initial stop)
    px = stop0 if dayD else lastc
    if not dayD:
        for k in range(len(dfut)):
            if lo[k] <= stop0:
                px = stop0; break
            if k+1 >= 10:
                px = float(cl[k]); break
    out["MAXHOLD_10"] = px
    return out


def run():
    con = sqlite3.connect(str(DB))
    rows = []
    for sym in UNIVERSE:
        df = to_30(load(con, sym, "5minute", START, END))
        if df.empty:
            continue
        df["volma"] = df["volume"].rolling(VOLMA_WIN, min_periods=VOLMA_MINP).mean().shift(1)
        df["day"] = df.index.normalize()
        daily = load(con, sym, "day", START, END).sort_index()
        if daily.empty:
            continue
        atrd = atr_w(daily, ATR_D_LEN); std = st_dir(daily, ST_PERIOD, ST_MULT)
        dhigh = daily["high"]; didx = daily.index
        for day, g in df.groupby("day", sort=True):
            ph = dhigh.loc[dhigh.index < day]
            if ph.empty:
                continue
            pdh = float(ph.iloc[-1])
            pa = atrd.loc[atrd.index < day].dropna()
            if pa.empty:
                continue
            atrv = float(pa.iloc[-1])
            if atrv <= 0:
                continue
            dfut = daily.loc[didx > day].iloc[:MAXHOLD]
            if dfut.empty:
                continue
            sfut = std.loc[dfut.index]; g = g.sort_index()
            picks = {}
            # BASELINE: first bar (enter 2nd bar)
            if len(g) > 1:
                picks["BASELINE"] = 0
            for i in range(len(g)-1):
                r = g.iloc[i]
                if "BREAK_ONLY" not in picks and r["close"] > pdh:
                    picks["BREAK_ONLY"] = i
                if "SIGNAL" not in picks and (not np.isnan(r["volma"])) and r["volume"] > r["volma"] and r["close"] > pdh:
                    picks["SIGNAL"] = i
            for arm, i in picks.items():
                entry = float(g.iloc[i+1]["open"]); rem = g.iloc[i+1:]
                ex = sim(entry, atrv, rem, dfut, sfut)
                for pol, xp in ex.items():
                    rows.append({"arm": arm, "policy": pol, "year": day.year,
                                 "net6R": ((xp-entry)-COST*entry)/atrv})
        print(f"{sym:11s} done")
    con.close()
    t = pd.DataFrame(rows)
    print("\n" + "="*78)
    print("PLACEBO / BENCHMARK — mean net@6bp R (PF) by entry arm. Alpha = SIGNAL >> BASELINE")
    print("="*78)
    print(f"{'policy':<18} " + " ".join(a.rjust(18) for a in ARMS))
    for pol in POLICIES:
        cells = []
        for arm in ARMS:
            s = t[(t.policy == pol) & (t.arm == arm)]["net6R"]
            pos = s[s > 0].sum(); neg = -s[s < 0].sum()
            pf = pos/neg if neg > 0 else float("inf")
            cells.append(f"{s.mean():+.3f}/{pf:.2f}(n{len(s)})")
        print(f"{pol:<18} " + " ".join(c.rjust(18) for c in cells))
    print("\n(read: meanR / PF (n). If SIGNAL ~ BASELINE -> volume+breakout adds no alpha.)")


if __name__ == "__main__":
    run()
