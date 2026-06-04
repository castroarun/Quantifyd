"""G1 cheap probe — POSITIONAL variant. Same entry as smoke_probe.py
(30-min: bar volume > own trailing 50-day MA AND close > prev-day high -> LONG),
but HELD ACROSS DAYS and managed on DAILY bars. This tests research/44's finding
that this breakout family only works as a SWING/positional hold, not intraday.

Exits (multi-day): HARD_SL(1 ATR), R_2R, R_3R, daily CHANDELIER(3 ATR),
daily SUPERTREND(10,3) flip, MAXHOLD_5, MAXHOLD_10. R = daily ATR(14) at entry
(appropriate for a multi-day hold -> cost-in-R is small).

Methodology: per-signal expectancy (event study; overlapping trades allowed) —
the standard cheap probe used in research/40 & 44, NOT a portfolio equity curve.
Causal: prev-day high known pre-open; volume-MA shifted 1; entry next 30-min bar
open; daily mgmt starts the day AFTER entry (entry-day tail also checked vs the
initial hard stop on the remaining 30-min bars, so an entry-day collapse is caught).

Run: venv/bin/python research/49_volbreak_pdh_30min/scripts/smoke_probe_positional.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB = ROOT.parents[1] / "backtest_data" / "market_data.db"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
TRADES_CSV = RESULTS / "smoke_trades_positional.csv"

UNIVERSE = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "SBIN",
            "INFY", "MARUTI", "TATASTEEL"]
START, END = "2018-01-01", "2026-06-01"

VOLMA_WIN, VOLMA_MINP = 600, 200
RSI_LEN, RSI_MIN = 14, 55.0
ATR_D_LEN = 14
ST_PERIOD, ST_MULT = 10, 3.0
MAXHOLD_BACKSTOP = 30          # hard cap on any trade (trading days)
COST_BPS = [0.0, 6.0, 12.0]
EXIT_POLICIES = ["HARD_SL", "R_2R", "R_3R", "CHANDELIER_3ATR",
                 "SUPERTREND_D_10_3", "MAXHOLD_5", "MAXHOLD_10"]
USE_RSI_VARIANTS = [False, True]


def load_5min(con, s, a, b):
    df = pd.read_sql("SELECT date,open,high,low,close,volume FROM market_data_unified "
                     "WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=? ORDER BY date",
                     con, params=(s, a, b + " 23:59:59"))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def load_daily(con, s, a, b):
    df = pd.read_sql("SELECT date,open,high,low,close,volume FROM market_data_unified "
                     "WHERE symbol=? AND timeframe='day' AND date>=? AND date<=? ORDER BY date",
                     con, params=(s, a, b))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def to_30min(df5):
    if df5.empty:
        return df5
    df = df5.copy()
    df["_day"] = df.index.normalize()
    rows = []
    for _, d in df.groupby("_day", sort=True):
        d = d.sort_index()
        n = (len(d) // 6) * 6
        for i in range(0, n, 6):
            g = d.iloc[i:i + 6]
            rows.append({"date": g.index[0], "open": float(g["open"].iloc[0]),
                         "high": float(g["high"].max()), "low": float(g["low"].min()),
                         "close": float(g["close"].iloc[-1]), "volume": float(g["volume"].sum())})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / \
        dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr_wilder(df, n):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def supertrend_dir(df, period=10, mult=3.0):
    atr = atr_wilder(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = (hl2 + mult * atr)
    lower = (hl2 - mult * atr)
    n = len(df)
    fu = np.array(upper, dtype=float)
    fl = np.array(lower, dtype=float)
    c = np.asarray(df["close"], dtype=float)
    upn, lon = np.asarray(upper, dtype=float), np.asarray(lower, dtype=float)
    for i in range(1, n):
        fu[i] = upn[i] if (upn[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lon[i] if (lon[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
    d = np.ones(n, dtype=int)
    for i in range(1, n):
        d[i] = 1 if c[i] > fu[i - 1] else (-1 if c[i] < fl[i - 1] else d[i - 1])
    return pd.Series(d, index=df.index)


def simulate_pos(entry, atrv, rem_dayD, dfut, stdir_fut):
    """dfut = daily bars AFTER entry day (cap applied by caller).
    Returns {policy: (exit_price, hold_days)}."""
    R = atrv
    stop0 = entry - R
    t2, t3 = entry + 2 * R, entry + 3 * R
    dayD_stop = bool((rem_dayD["low"] <= stop0).any()) if len(rem_dayD) else False
    last_close = float(dfut["close"].iloc[-1]) if len(dfut) else float(rem_dayD["close"].iloc[-1])
    last_hold = len(dfut)
    hi = dfut["high"].to_numpy() if len(dfut) else np.array([])
    lo = dfut["low"].to_numpy() if len(dfut) else np.array([])
    cl = dfut["close"].to_numpy() if len(dfut) else np.array([])
    sd = stdir_fut.to_numpy() if len(dfut) else np.array([])
    out = {}

    # HARD_SL
    if dayD_stop:
        out["HARD_SL"] = (stop0, 0)
    else:
        px, hd = last_close, last_hold
        for k in range(len(dfut)):
            if lo[k] <= stop0:
                px, hd = stop0, k + 1
                break
        out["HARD_SL"] = (px, hd)

    # R targets (stop checked before target intrabar = pessimistic)
    for tname, tgt in (("R_2R", t2), ("R_3R", t3)):
        if dayD_stop:
            out[tname] = (stop0, 0)
            continue
        px, hd = last_close, last_hold
        for k in range(len(dfut)):
            if lo[k] <= stop0:
                px, hd = stop0, k + 1
                break
            if hi[k] >= tgt:
                px, hd = tgt, k + 1
                break
        out[tname] = (px, hd)

    # CHANDELIER 3 ATR (daily trail off running high)
    px, hd = last_close, last_hold
    hh = max(entry, float(rem_dayD["high"].max()) if len(rem_dayD) else entry)
    done = False
    for k in range(len(dfut)):
        trail = hh - 3 * R
        if lo[k] <= trail:
            px, hd = trail, k + 1
            done = True
            break
        hh = max(hh, hi[k])
    out["CHANDELIER_3ATR"] = (px, hd)

    # SUPERTREND daily flip to down -> exit that day's close
    px, hd = last_close, last_hold
    for k in range(len(dfut)):
        if sd[k] == -1:
            px, hd = float(cl[k]), k + 1
            break
    out["SUPERTREND_D_10_3"] = (px, hd)

    # MAXHOLD caps (exit at close of Nth day, with initial hard stop still active)
    for mname, mh in (("MAXHOLD_5", 5), ("MAXHOLD_10", 10)):
        if dayD_stop:
            out[mname] = (stop0, 0)
            continue
        px, hd = last_close, last_hold
        for k in range(len(dfut)):
            if lo[k] <= stop0:
                px, hd = stop0, k + 1
                break
            if k + 1 >= mh:
                px, hd = float(cl[k]), k + 1
                break
        out[mname] = (px, hd)
    return out


def run():
    con = sqlite3.connect(str(DB))
    trades = []
    for sym in UNIVERSE:
        d5 = load_5min(con, sym, START, END)
        if d5.empty:
            print(f"{sym:11s} NO 5-min"); continue
        df = to_30min(d5)
        if df.empty:
            print(f"{sym:11s} no 30m"); continue
        df["rsi"] = rsi(df["close"], RSI_LEN)
        df["volma"] = df["volume"].rolling(VOLMA_WIN, min_periods=VOLMA_MINP).mean().shift(1)
        df["day"] = df.index.normalize()

        daily = load_daily(con, sym, START, END)
        if daily.empty:
            print(f"{sym:11s} no daily"); continue
        daily = daily.sort_index()
        atr_d = atr_wilder(daily, ATR_D_LEN)
        st_d = supertrend_dir(daily, ST_PERIOD, ST_MULT)
        dhigh = daily["high"]
        didx = daily.index

        nsig = 0
        for day, g in df.groupby("day", sort=True):
            ph = dhigh.loc[dhigh.index < day]
            if ph.empty:
                continue
            pdh = float(ph.iloc[-1])
            pa = atr_d.loc[atr_d.index < day].dropna()
            if pa.empty:
                continue
            atrv = float(pa.iloc[-1])
            if atrv <= 0:
                continue
            # daily bars strictly after entry day, capped
            fut_mask = didx > day
            dfut_all = daily.loc[fut_mask]
            if dfut_all.empty:
                continue
            dfut = dfut_all.iloc[:MAXHOLD_BACKSTOP]
            stfut = st_d.loc[dfut.index]
            g = g.sort_index()
            for use_rsi in USE_RSI_VARIANTS:
                for i in range(len(g) - 1):
                    row = g.iloc[i]
                    if np.isnan(row["volma"]):
                        continue
                    if not (row["volume"] > row["volma"]):
                        continue
                    if not (row["close"] > pdh):
                        continue
                    if use_rsi and not (row["rsi"] >= RSI_MIN):
                        continue
                    entry = float(g.iloc[i + 1]["open"])
                    rem = g.iloc[i + 1:]
                    exits = simulate_pos(entry, atrv, rem, dfut, stfut)
                    for pol, (xp, hd) in exits.items():
                        trades.append({"symbol": sym, "date": str(day.date()),
                                       "year": day.year, "use_rsi": use_rsi,
                                       "policy": pol, "entry": entry, "exit": xp,
                                       "hold_days": hd, "R_price": atrv,
                                       "gross_price": xp - entry,
                                       "gross_R": (xp - entry) / atrv})
                    if not use_rsi:
                        nsig += 1
                    break
        print(f"{sym:11s} bars={len(df):>6}  signals(no-rsi)={nsig}")
    con.close()

    if not trades:
        print("NO TRADES"); return
    t = pd.DataFrame(trades)
    t.to_csv(TRADES_CSV, index=False)
    print(f"\nWrote {len(t)} trade-rows -> {TRADES_CSV}\n")

    print("=" * 104)
    print("G1 POSITIONAL — per exit policy (LONG, multi-day hold). Bar: net +0.16R / PF 1.24 (research/44)")
    print("=" * 104)
    for use_rsi in USE_RSI_VARIANTS:
        sub = t[t["use_rsi"] == use_rsi]
        if sub.empty:
            continue
        print(f"\n--- filter: {'RSI>=55' if use_rsi else 'no-RSI '} ---")
        print(f"{'policy':<18} {'n':>5} {'WR%':>6} {'hold':>5} {'grossR':>8} " +
              " ".join(f'net{int(c)}bp'.rjust(8) for c in COST_BPS) + f" {'PF6bp':>7}")
        for pol in EXIT_POLICIES:
            s = sub[sub["policy"] == pol]
            if s.empty:
                continue
            net = {c: (s["gross_price"] - c / 1e4 * s["entry"]) / s["R_price"] for c in COST_BPS}
            net6 = net[6.0]
            pos = net6[net6 > 0].sum()
            neg = -net6[net6 < 0].sum()
            pf = pos / neg if neg > 0 else float("inf")
            print(f"{pol:<18} {len(s):>5} {100*(net6>0).mean():>6.1f} "
                  f"{s['hold_days'].mean():>5.1f} {s['gross_R'].mean():>8.3f} " +
                  " ".join(f"{net[c].mean():>8.3f}" for c in COST_BPS) + f" {pf:>7.2f}")

    print("\n--- per-year net@6bp (no-RSI) ---")
    s = t[t["use_rsi"] == False].copy()
    s["net6"] = (s["gross_price"] - 6 / 1e4 * s["entry"]) / s["R_price"]
    print(s.pivot_table(index="year", columns="policy", values="net6", aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    run()
