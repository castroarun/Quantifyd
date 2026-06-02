"""G1 cheap probe — Volume-MA + Prev-Day-High breakout, 30-min intraday LONG.

Playbook §12.5: smoke-test a few names BEFORE any sweep, to catch look-ahead /
cost bugs and to see if the signal has *any* life vs research/44's bar
(net +0.16R / PF 1.24). Self-contained (own 5min->30min resampler + daily
loader); reads only backtest_data/market_data.db. Reports GROSS and NET (cost
sensitivity 0/6/12 bps), per exit policy, with a per-year net break-down.

Run:  venv/bin/python research/49_volbreak_pdh_30min/scripts/smoke_probe.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # research/49_...
DB = ROOT.parents[1] / "backtest_data" / "market_data.db"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
TRADES_CSV = RESULTS / "smoke_trades.csv"

UNIVERSE = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "SBIN",
            "INFY", "MARUTI", "TATASTEEL"]
START, END = "2018-01-01", "2026-06-01"

VOLMA_WIN = 600          # ~50 trading days x 12 30-min bars ("own 50-day MA")
VOLMA_MINP = 200
RSI_LEN = 14
RSI_MIN = 55.0           # used only when use_rsi=True
ST_PERIOD, ST_MULT = 10, 3.0
ATR_DAILY_LEN = 14
COST_BPS = [0.0, 6.0, 12.0]   # round-trip, on entry notional
EXIT_POLICIES = ["EOD", "HARD_SL", "R_TARGET_2R", "CHANDELIER_3ATR", "SUPERTREND_10_3"]
USE_RSI_VARIANTS = [False, True]


# --------------------------- data ---------------------------
def load_5min(con, symbol, start, end):
    df = pd.read_sql(
        "SELECT date,open,high,low,close,volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=? ORDER BY date",
        con, params=(symbol, start, end + " 23:59:59"))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def load_daily(con, symbol, start, end):
    df = pd.read_sql(
        "SELECT date,open,high,low,close,volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' AND date>=? AND date<=? ORDER BY date",
        con, params=(symbol, start, end))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def to_30min(df5):
    """6 x 5min anchored per-session at 09:15; drop trailing partial."""
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
                         "close": float(g["close"].iloc[-1]),
                         "volume": float(g["volume"].sum())})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


# --------------------------- indicators ---------------------------
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


def supertrend(df, period=10, mult=3.0):
    atr = atr_wilder(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    n = len(df)
    fu = np.array(upper, dtype=float)
    fl = np.array(lower, dtype=float)
    c = np.asarray(df["close"], dtype=float)
    upn, lon = np.asarray(upper, dtype=float), np.asarray(lower, dtype=float)
    for i in range(1, n):
        fu[i] = upn[i] if (upn[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lon[i] if (lon[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
    direction = np.ones(n, dtype=int)        # 1 = up, -1 = down
    for i in range(1, n):
        if c[i] > fu[i - 1]:
            direction[i] = 1
        elif c[i] < fl[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return pd.Series(direction, index=df.index)


# --------------------------- exit sim ---------------------------
def simulate(entry, atrv, rem, st_dir):
    """rem = bars from entry bar (inclusive) to EOD; returns {policy: exit_price}."""
    R = atrv
    stop0 = entry - R
    target = entry + 2 * R
    last_close = float(rem["close"].iloc[-1])
    out = {}

    # EOD
    out["EOD"] = last_close
    # HARD_SL (1 ATR)
    px = last_close
    for low in rem["low"]:
        if low <= stop0:
            px = stop0
            break
    out["HARD_SL"] = px
    # R_TARGET_2R (stop checked before target within a bar = pessimistic)
    px = last_close
    for _, b in rem.iterrows():
        if b["low"] <= stop0:
            px = stop0
            break
        if b["high"] >= target:
            px = target
            break
    out["R_TARGET_2R"] = px
    # CHANDELIER_3ATR (trail off running high, prior-bar high only)
    px = last_close
    hh = float(rem["high"].iloc[0])
    for k in range(1, len(rem)):
        trail = hh - 3 * R
        if rem["low"].iloc[k] <= trail:
            px = trail
            break
        hh = max(hh, float(rem["high"].iloc[k]))
    out["CHANDELIER_3ATR"] = px
    # SUPERTREND flip to down -> exit at that bar close
    px = last_close
    dvals = st_dir.to_numpy()
    cvals = rem["close"].to_numpy()
    for k in range(len(rem)):
        if dvals[k] == -1:
            px = float(cvals[k])
            break
    out["SUPERTREND_10_3"] = px
    return out, R


# --------------------------- main ---------------------------
def run():
    con = sqlite3.connect(str(DB))
    trades = []
    for sym in UNIVERSE:
        d5 = load_5min(con, sym, START, END)
        if d5.empty:
            print(f"{sym:11s} NO 5-min data"); continue
        df = to_30min(d5)
        if df.empty or len(df) < VOLMA_MINP:
            print(f"{sym:11s} too few 30-min bars"); continue
        df["rsi"] = rsi(df["close"], RSI_LEN)
        df["volma"] = df["volume"].rolling(VOLMA_WIN, min_periods=VOLMA_MINP).mean().shift(1)
        df["stdir"] = supertrend(df, ST_PERIOD, ST_MULT)
        df["atr30"] = atr_wilder(df, 14)   # INTRADAY R unit so exits actually bind
        df["day"] = df.index.normalize()

        daily = load_daily(con, sym, START, END)
        if daily.empty:
            print(f"{sym:11s} no daily data"); continue
        dhigh = daily["high"]

        nsig = 0
        for day, g in df.groupby("day", sort=True):
            prior_h = dhigh.loc[dhigh.index < day]
            if prior_h.empty:
                continue
            pdh = float(prior_h.iloc[-1])
            g = g.sort_index()
            for use_rsi in USE_RSI_VARIANTS:
                for i in range(len(g) - 1):          # need i+1 for entry
                    row = g.iloc[i]
                    if np.isnan(row["volma"]) or np.isnan(row["atr30"]):
                        continue
                    atrv = float(row["atr30"])       # intraday R at signal bar (causal)
                    if atrv <= 0:
                        continue
                    if not (row["volume"] > row["volma"]):
                        continue
                    if not (row["close"] > pdh):
                        continue
                    if use_rsi and not (row["rsi"] >= RSI_MIN):
                        continue
                    entry = float(g.iloc[i + 1]["open"])
                    rem = g.iloc[i + 1:]
                    exits, R = simulate(entry, atrv, rem, rem["stdir"])
                    for pol, xp in exits.items():
                        gross_price = xp - entry
                        trades.append({
                            "symbol": sym, "date": str(day.date()),
                            "year": day.year, "use_rsi": use_rsi,
                            "policy": pol, "entry": entry, "exit": xp,
                            "R_price": R, "gross_price": gross_price,
                            "gross_R": gross_price / R})
                    if not use_rsi:
                        nsig += 1
                    break   # one trade per day per rsi-variant
        print(f"{sym:11s} bars={len(df):>6}  signals(no-rsi)={nsig}")

    con.close()
    if not trades:
        print("NO TRADES"); return
    t = pd.DataFrame(trades)
    t.to_csv(TRADES_CSV, index=False)
    print(f"\nWrote {len(t)} trade-rows -> {TRADES_CSV}\n")

    # ---- per-policy summary, gross + net at each cost ----
    print("=" * 96)
    print("G1 SMOKE — per exit policy (LONG, intraday). Bar to beat: net +0.16R / PF 1.24 (research/44)")
    print("=" * 96)
    for use_rsi in USE_RSI_VARIANTS:
        sub0 = t[t["use_rsi"] == use_rsi]
        if sub0.empty:
            continue
        tag = "RSI>=55" if use_rsi else "no-RSI "
        print(f"\n--- filter: {tag} ---")
        hdr = f"{'policy':<16} {'n':>5} {'WR%':>6} {'grossR':>8} " + \
              " ".join(f"net{int(c)}bp".rjust(8) for c in COST_BPS) + f" {'PF6bp':>7}"
        print(hdr)
        for pol in EXIT_POLICIES:
            s = sub0[sub0["policy"] == pol]
            if s.empty:
                continue
            n = len(s)
            net = {c: (s["gross_price"] - c / 1e4 * s["entry"]) / s["R_price"] for c in COST_BPS}
            net6 = net[6.0]
            wr = 100.0 * (net6 > 0).mean()
            pos = net6[net6 > 0].sum()
            neg = -net6[net6 < 0].sum()
            pf = pos / neg if neg > 0 else float("inf")
            line = (f"{pol:<16} {n:>5} {wr:>6.1f} {s['gross_R'].mean():>8.3f} " +
                    " ".join(f"{net[c].mean():>8.3f}" for c in COST_BPS) +
                    f" {pf:>7.2f}")
            print(line)

    # ---- per-year net6 for the best gross policy (no-rsi) ----
    print("\n--- per-year net@6bp (no-RSI) ---")
    s = t[(t["use_rsi"] == False)].copy()
    s["net6"] = (s["gross_price"] - 6 / 1e4 * s["entry"]) / s["R_price"]
    pv = s.pivot_table(index="year", columns="policy", values="net6", aggfunc="mean")
    print(pv.round(3).to_string())


if __name__ == "__main__":
    run()
