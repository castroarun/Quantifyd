"""G4 — DAILY compression->breakout on the FULL universe (incl. the actual
reference runners DATAPATTNS, TDPOWERSYS, which have NO 5-min data). This is the
honest home for the idea: your examples are multi-MONTH positional runs, daily
data is multi-regime (2018 selloff / 2020 crash / 2022 bear / 2024-25), 1600+
names, and the money is in HOLDING the run, not the 5-min tick.

Signal (LONG, all causal, decided at close of day D, entered at D+1 open):
  uptrend     : close > SMA50 (axis: + close>SMA200)
  contraction : 10-day range% < CONTRACT_PCT  (a coil) AND NR7 optional
  breakout    : close > prior-20-day high (Donchian-20 break of the range)
  volume      : volume >= k * SMA(volume,20)   (the visible spike)
  liquidity   : 20-day median turnover >= 3cr and price >= 30
Stop = consolidation low (min low, prior 10d) -> R = entry - stop (the box).
Exits (positional, daily mgmt, 60-day cap): HARD_SL(box), R_2R/3R, CHANDELIER_3ATR,
SUPERTREND_D(10,3), MAXHOLD_20.

Ablation arms (the alpha test): SIGNAL, VOL_SPIKE(k=3), NO_VOL, NO_CONTRACTION,
TREND_BASELINE(any uptrend day). Incremental/resumable. Per-year output.

Run: venv/bin/python research/55_.../scripts/g4_daily.py [--aggregate-only]
"""
from __future__ import annotations
import csv
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
DB = REPO / "backtest_data" / "market_data.db"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
TRADES = RESULTS / "g4_daily_trades.csv"

START, END = "2017-01-01", "2026-06-01"     # 2017 warmup -> trades from ~2018
SMA_TREND, DONCHIAN, CONTRACT_LB = 50, 20, 10
CONTRACT_PCT = 0.12          # 10-day high-low range / close below this = coil
NARROW_CPR_PCT = 0.6         # today's daily-CPR width% below this = narrow CPR (user's KMEW cue)
VOL_LB, VOL_K, VOL_SPIKE_K = 20, 1.5, 3.0
ATR_LEN, ST_P, ST_M, MAXHOLD = 14, 10, 3.0, 60
MIN_PRICE, MIN_TURN_CR = 30.0, 3.0
POLICIES = ["HARD_SL", "R_2R", "R_3R", "CHANDELIER_3ATR", "SUPERTREND_D_10_3", "MAXHOLD_20"]
ARMS = ["SIGNAL", "VOL_SPIKE", "NO_VOL", "NO_CONTRACTION", "TREND_BASELINE"]
FIELDS = ["symbol", "arm", "policy", "year", "entry", "R", "gross_price", "net6R"]


def load_daily(con, s):
    df = pd.read_sql("SELECT date,open,high,low,close,volume FROM market_data_unified "
                     "WHERE symbol=? AND timeframe='day' AND date>=? AND date<=? ORDER BY date",
                     con, params=(s, START, END))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


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


def sim(entry, stop0, atrv, dfut, sfut):
    R = entry - stop0
    if R <= 0:
        R = atrv
        stop0 = entry - R
    t2, t3 = entry+2*R, entry+3*R
    lastc = float(dfut["close"].iloc[-1])
    hi = dfut["high"].to_numpy(); lo = dfut["low"].to_numpy(); cl = dfut["close"].to_numpy(); sd = sfut.to_numpy()
    out = {}
    px = lastc
    for k in range(len(dfut)):
        if lo[k] <= stop0: px = stop0; break
    out["HARD_SL"] = px
    for nm, tgt in (("R_2R", t2), ("R_3R", t3)):
        px = lastc
        for k in range(len(dfut)):
            if lo[k] <= stop0: px = stop0; break
            if hi[k] >= tgt: px = tgt; break
        out[nm] = px
    px = lastc; hh = entry
    for k in range(len(dfut)):
        tr = hh-3*atrv
        if lo[k] <= tr: px = tr; break
        hh = max(hh, hi[k])
    out["CHANDELIER_3ATR"] = px
    px = lastc
    for k in range(len(dfut)):
        if sd[k] == -1: px = float(cl[k]); break
    out["SUPERTREND_D_10_3"] = px
    px = lastc
    for k in range(len(dfut)):
        if lo[k] <= stop0: px = stop0; break
        if k+1 >= 20: px = float(cl[k]); break
    out["MAXHOLD_20"] = px
    return out, R


def symbol_rows(con, sym):
    d = load_daily(con, sym)
    if d.empty or len(d) < 260:
        return []
    d = d.sort_index()
    c = d["close"]
    sma50 = c.rolling(SMA_TREND, min_periods=SMA_TREND).mean()
    sma200 = c.rolling(200, min_periods=200).mean()
    don_hi = d["high"].rolling(DONCHIAN, min_periods=DONCHIAN).max().shift(1)   # prior 20d high
    rng10 = (d["high"].rolling(CONTRACT_LB).max() - d["low"].rolling(CONTRACT_LB).min())
    contract = (rng10 / c) < CONTRACT_PCT
    rng = d["high"] - d["low"]
    nr7 = rng <= rng.rolling(7, min_periods=7).min()
    # today's daily CPR width% from PREV-day H/L/C (user's KMEW "narrow CPR" cue)
    H1, L1, C1 = d["high"].shift(1), d["low"].shift(1), d["close"].shift(1)
    P = (H1 + L1 + C1) / 3.0; bc = (H1 + L1) / 2.0; tc = 2 * P - bc
    narrow_cpr = ((tc - bc).abs() / P * 100.0) < NARROW_CPR_PCT
    volsma = d["volume"].rolling(VOL_LB, min_periods=VOL_LB).mean()
    box_low = d["low"].rolling(CONTRACT_LB, min_periods=CONTRACT_LB).min()
    atr = atr_w(d, ATR_LEN)
    std = st_dir(d, ST_P, ST_M)
    turn = (c * d["volume"]).rolling(20, min_periods=20).median() / 1e7  # cr
    idx = d.index
    rows = []
    arr = d.reset_index()
    for i in range(250, len(d) - 1):
        day = idx[i]
        price = float(c.iloc[i])
        if price < MIN_PRICE or pd.isna(turn.iloc[i]) or turn.iloc[i] < MIN_TURN_CR:
            continue
        if pd.isna(sma50.iloc[i]) or price <= sma50.iloc[i]:
            continue                                  # uptrend gate (all arms)
        atrv = float(atr.iloc[i])
        if pd.isna(atrv) or atrv <= 0:
            continue
        entry = float(d["open"].iloc[i + 1])          # next-day open (causal)
        dfut = d.iloc[i + 1:i + 1 + MAXHOLD]
        if len(dfut) < 2:
            continue
        sfut = std.iloc[i + 1:i + 1 + MAXHOLD]
        stop0 = float(box_low.iloc[i]) if not pd.isna(box_low.iloc[i]) else entry - atrv
        brk = (not pd.isna(don_hi.iloc[i])) and price > float(don_hi.iloc[i])
        # compression = multi-day coil AND today's narrow CPR (user's KMEW spec)
        comp = bool(contract.iloc[i]) and bool(narrow_cpr.iloc[i])
        vok = (not pd.isna(volsma.iloc[i])) and d["volume"].iloc[i] >= VOL_K * volsma.iloc[i]
        vspike = (not pd.isna(volsma.iloc[i])) and d["volume"].iloc[i] >= VOL_SPIKE_K * volsma.iloc[i]
        picks = set()
        picks.add("TREND_BASELINE")                   # any uptrend day = beta benchmark
        if brk and comp and vok:
            picks.add("SIGNAL")
        if brk and comp and vspike:
            picks.add("VOL_SPIKE")
        if brk and comp:
            picks.add("NO_VOL")
        if brk and vok:
            picks.add("NO_CONTRACTION")
        ex, R = sim(entry, stop0, atrv, dfut, sfut)
        for arm in picks:
            for pol, xp in ex.items():
                rows.append({"symbol": sym, "arm": arm, "policy": pol, "year": int(day.year),
                             "entry": round(entry, 2), "R": round(R, 3),
                             "gross_price": round(xp - entry, 3),
                             "net6R": round(((xp - entry) - 6 / 1e4 * entry) / R, 5)})
    return rows


def aggregate():
    if not TRADES.exists():
        print("no csv"); return
    t = pd.read_csv(TRADES)
    print(f"\n=== G4 DAILY — {len(t):,} rows, {t['symbol'].nunique()} symbols, 2018-2026 ===")
    for pol in POLICIES:
        print(f"\n### {pol}")
        print(f"{'arm':<16} {'n':>7} {'WR%':>6} {'net6R':>8} {'PF6':>6} {'grossR':>8}")
        for arm in ARMS:
            s = t[(t.policy == pol) & (t.arm == arm)]
            if s.empty:
                continue
            net = s["net6R"]; pos = net[net > 0].sum(); neg = -net[net < 0].sum()
            pf = pos/neg if neg > 0 else float("inf")
            print(f"{arm:<16} {len(s):>7} {100*(net>0).mean():>6.1f} {net.mean():>8.3f} {pf:>6.2f} {(s['gross_price']/s['R']).mean():>8.3f}")
    print("\n### ALPHA DELTA (arm - TREND_BASELINE) net6R")
    for pol in POLICIES:
        b = t[(t.policy == pol) & (t.arm == "TREND_BASELINE")]["net6R"].mean()
        line = f"  {pol:<16} BASE {b:+.3f} | "
        for arm in ["SIGNAL", "VOL_SPIKE", "NO_VOL", "NO_CONTRACTION"]:
            a = t[(t.policy == pol) & (t.arm == arm)]["net6R"].mean()
            if pd.notna(a):
                line += f"{arm}:{a-b:+.3f} "
        print(line)
    print("\n### per-year net6R — SUPERTREND_D, SIGNAL vs VOL_SPIKE vs BASELINE")
    s = t[t.policy == "SUPERTREND_D_10_3"]
    pv = s.pivot_table(index="year", columns="arm", values="net6R", aggfunc="mean")
    cols = [c for c in ARMS if c in pv.columns]
    print(pv[cols].round(3).to_string())
    print("\n### SIGNAL trades per year (SUPERTREND_D)")
    print(s[s.arm == "SIGNAL"].groupby("year").size().to_string())


def run():
    if "--aggregate-only" in sys.argv:
        aggregate(); return
    con = sqlite3.connect(str(DB))
    syms = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'")]
    done = set()
    if TRADES.exists():
        try:
            done = set(pd.read_csv(TRADES, usecols=["symbol"])["symbol"].unique())
        except Exception:
            done = set()
    todo = [s for s in sorted(syms) if s not in done]
    print(f"daily universe={len(syms)} done={len(done)} todo={len(todo)}", flush=True)
    new = not TRADES.exists()
    f = TRADES.open("a", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    tot = 0
    for n, sym in enumerate(todo, 1):
        try:
            rows = symbol_rows(con, sym)
        except Exception as e:
            print(f"[{n}/{len(todo)}] {sym:12s} ERR {e}", flush=True); continue
        for r in rows:
            w.writerow(r)
        f.flush()
        ns = sum(1 for r in rows if r["arm"] == "SIGNAL")//len(POLICIES) if rows else 0
        tot += ns
        if n % 50 == 0 or ns > 0:
            print(f"[{n}/{len(todo)}] {sym:12s} sig={ns:>3} cum={tot}", flush=True)
    f.close(); con.close()
    print("\nDONE — aggregating", flush=True)
    aggregate()


if __name__ == "__main__":
    run()
