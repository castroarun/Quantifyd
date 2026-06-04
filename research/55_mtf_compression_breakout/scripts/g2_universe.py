"""G2 broad-universe sweep — MTF compression breakout + ablation, ~218 liquid names.

Same signal/arms as g1_probe but across the turnover-filtered universe, with
per-year output and INCREMENTAL crash-safe writes (resumable: skips symbols
already in g2_trades.csv). Carries G1 decisions: keep the ablation arms so the
alpha-vs-beta (SIGNAL vs TREND_BASELINE) and volume-hurts findings are re-tested
at scale; emphasise tight-stop exits where the entry edge lives.

Run:            venv/bin/python research/55_.../scripts/g2_universe.py
Aggregate only: venv/bin/python research/55_.../scripts/g2_universe.py --aggregate-only
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
TRADES = RESULTS / "g2_trades.csv"
STATUS = ROOT / "MTF_COMPRESSION_BREAKOUT_MULTITF_SWEEP_STATUS.md"

sys.path.insert(0, str(REPO / "research/40_volsurge_pdr_break_weekly_cpr/scripts"))
import signal_lib as sl  # noqa: E402

START, END = "2018-01-01", "2026-06-01"
TRIGGER_TF = "15min"
NR_WIN = 7
NARROW_CPR_PCT = 0.6
VOL_K = 1.5
ATR_D_LEN, ST_P, ST_M, MAXHOLD = 14, 10, 3.0, 30
B_MIN, ZONE = sl.CLEAN_PRESETS["loose"]
POLICIES = ["HARD_SL", "R_2R", "R_3R", "CHANDELIER_3ATR", "SUPERTREND_D_10_3", "MAXHOLD_10"]
ARMS = ["SIGNAL", "NO_VOL", "NO_COMPRESSION", "TREND_BASELINE"]
FIELDS = ["symbol", "arm", "policy", "year", "entry", "R", "gross_price", "net6R"]


def universe():
    con = sqlite3.connect(str(DB))
    have5 = set(r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='5minute'"))
    con.close()
    tc = REPO / "research/34_nifty500_expansion/results/top_by_turnover.csv"
    sel = set()
    if tc.exists():
        with tc.open() as f:
            for r in csv.DictReader(f):
                try:
                    if float(r["avg_turnover_cr"]) >= 50.0:
                        sel.add(r["symbol"])
                except (ValueError, KeyError):
                    pass
    return sorted(sel & have5) if sel else sorted(have5)


def load(con, s, tf, a, b):
    df = pd.read_sql("SELECT date,open,high,low,close,volume FROM market_data_unified "
                     "WHERE symbol=? AND timeframe=? AND date>=? AND date<=? ORDER BY date",
                     con, params=(s, tf, a, b + (" 23:59:59" if tf != "day" else "")))
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


def daily_cpr_width_pct(daily):
    H = daily["high"].shift(1); L = daily["low"].shift(1); C = daily["close"].shift(1)
    P = (H+L+C)/3.0; bc = (H+L)/2.0; tc = 2*P-bc
    return ((tc-bc).abs()/P*100.0)


def sim_pos(entry, atrv, rem, dfut, sfut):
    R = atrv; stop0 = entry-R; t2 = entry+2*R; t3 = entry+3*R
    dayD = bool((rem["low"] <= stop0).any()) if len(rem) else False
    lastc = float(dfut["close"].iloc[-1]) if len(dfut) else float(rem["close"].iloc[-1])
    hi = dfut["high"].to_numpy(); lo = dfut["low"].to_numpy(); cl = dfut["close"].to_numpy(); sd = sfut.to_numpy()
    out = {}
    px = stop0 if dayD else lastc
    if not dayD:
        for k in range(len(dfut)):
            if lo[k] <= stop0: px = stop0; break
    out["HARD_SL"] = px
    for nm, tgt in (("R_2R", t2), ("R_3R", t3)):
        px = stop0 if dayD else lastc
        if not dayD:
            for k in range(len(dfut)):
                if lo[k] <= stop0: px = stop0; break
                if hi[k] >= tgt: px = tgt; break
        out[nm] = px
    px = lastc; hh = max(entry, float(rem["high"].max()) if len(rem) else entry)
    for k in range(len(dfut)):
        tr = hh-3*R
        if lo[k] <= tr: px = tr; break
        hh = max(hh, hi[k])
    out["CHANDELIER_3ATR"] = px
    px = lastc
    for k in range(len(dfut)):
        if sd[k] == -1: px = float(cl[k]); break
    out["SUPERTREND_D_10_3"] = px
    px = stop0 if dayD else lastc
    if not dayD:
        for k in range(len(dfut)):
            if lo[k] <= stop0: px = stop0; break
            if k+1 >= 10: px = float(cl[k]); break
    out["MAXHOLD_10"] = px
    return out


def symbol_rows(con, sym):
    d5 = load(con, sym, "5minute", START, END)
    daily = load(con, sym, "day", START, END)
    if d5.empty or daily.empty:
        return []
    daily = daily.sort_index()
    df15 = sl.resample_5m(d5, TRIGGER_TF)
    if df15.empty:
        return []
    df15["volma"] = df15["volume"].rolling(500, min_periods=100).mean().shift(1)
    df15["day"] = df15.index.normalize()
    trend = sl.daily_trend(daily, "sma50")
    tp = {d.normalize(): trend.shift(1).get(d) for d in trend.index}
    wcpr = sl.weekly_cpr(daily)
    rng = daily["high"]-daily["low"]
    nr = (rng <= rng.rolling(NR_WIN, min_periods=NR_WIN).min()).shift(1)
    nrp = {d.normalize(): (bool(nr.get(d)) if not pd.isna(nr.get(d)) else False) for d in daily.index}
    cprw = daily_cpr_width_pct(daily)
    nap = {d.normalize(): (float(cprw.get(d)) < NARROW_CPR_PCT if not pd.isna(cprw.get(d)) else False) for d in daily.index}
    atrd = atr_w(daily, ATR_D_LEN)
    ap = {d.normalize(): atrd.shift(1).get(d) for d in atrd.index}
    std = st_dir(daily, ST_P, ST_M)
    dhigh = daily["high"]; didx = daily.index
    rows = []
    for day, g in df15.groupby("day", sort=True):
        if tp.get(day) != "up":
            continue
        atrv = ap.get(day)
        if atrv is None or pd.isna(atrv) or atrv <= 0:
            continue
        ph = dhigh.loc[dhigh.index < day]
        if ph.empty:
            continue
        pdh = float(ph.iloc[-1])
        crow = sl.cpr_for_date(wcpr, day)
        if not crow:
            continue
        ctop = float(crow["top"]); pwh = float(crow["prev_week_high"])
        dfut = daily.loc[didx > day].iloc[:MAXHOLD]
        if dfut.empty:
            continue
        sfut = std.loc[dfut.index]; g = g.sort_index()
        comp = nrp.get(day, False) and nap.get(day, False)
        picks = {}
        if len(g) > 1:
            picks["TREND_BASELINE"] = 0
        for i in range(len(g)-1):
            r = g.iloc[i]
            o, h, l, c, v = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])
            base = (sl.is_clean_candle(o, h, l, c, "long", B_MIN, ZONE)
                    and sl.range_escape(c, pdh, 0, pwh, 0, "long") and c > ctop)
            vol_ok = (not np.isnan(r["volma"])) and sl.volume_surge(v, float(r["volma"]), VOL_K)
            if "NO_COMPRESSION" not in picks and base and vol_ok:
                picks["NO_COMPRESSION"] = i
            if comp:
                if "NO_VOL" not in picks and base:
                    picks["NO_VOL"] = i
                if "SIGNAL" not in picks and base and vol_ok:
                    picks["SIGNAL"] = i
        for arm, i in picks.items():
            entry = float(g.iloc[i+1]["open"]); rem = g.iloc[i+1:]
            ex = sim_pos(entry, float(atrv), rem, dfut, sfut)
            for pol, xp in ex.items():
                rows.append({"symbol": sym, "arm": arm, "policy": pol, "year": int(day.year),
                             "entry": round(entry, 2), "R": round(float(atrv), 3),
                             "gross_price": round(xp-entry, 3),
                             "net6R": round(((xp-entry)-6/1e4*entry)/float(atrv), 5)})
    return rows


def aggregate():
    if not TRADES.exists():
        print("no trades csv"); return
    t = pd.read_csv(TRADES)
    print(f"\n=== G2 AGGREGATE — {len(t):,} rows, {t['symbol'].nunique()} symbols ===")
    print("net@6bp R / PF / WR / n  by ARM x POLICY")
    for pol in POLICIES:
        print(f"\n### {pol}")
        print(f"{'arm':<16} {'n':>7} {'WR%':>6} {'net6R':>8} {'PF6':>6}")
        for arm in ARMS:
            s = t[(t.policy == pol) & (t.arm == arm)]
            if s.empty:
                continue
            net = s["net6R"]; pos = net[net > 0].sum(); neg = -net[net < 0].sum()
            pf = pos/neg if neg > 0 else float("inf")
            print(f"{arm:<16} {len(s):>7} {100*(net>0).mean():>6.1f} {net.mean():>8.3f} {pf:>6.2f}")
    print("\n### ALPHA DELTA (SIGNAL - TREND_BASELINE), net6R")
    for pol in POLICIES:
        a = t[(t.policy == pol) & (t.arm == "SIGNAL")]["net6R"]
        b = t[(t.policy == pol) & (t.arm == "TREND_BASELINE")]["net6R"]
        if len(a) and len(b):
            d = a.mean()-b.mean()
            print(f"  {pol:<18} SIGNAL {a.mean():+.3f}(n{len(a)})  BASE {b.mean():+.3f}  delta {d:+.3f}  -> {'ALPHA' if d>=0.10 else 'no-alpha'}")
    print("\n### per-year net6R — SIGNAL vs BASELINE (HARD_SL)")
    s = t[t.policy == "HARD_SL"]
    pv = s.pivot_table(index="year", columns="arm", values="net6R", aggfunc="mean")
    cols = [c for c in ["SIGNAL", "NO_VOL", "NO_COMPRESSION", "TREND_BASELINE"] if c in pv.columns]
    print(pv[cols].round(3).to_string())
    print("\n### SIGNAL trade count per year (HARD_SL)")
    print(s[s.arm == "SIGNAL"].groupby("year").size().to_string())


def run():
    if "--aggregate-only" in sys.argv:
        aggregate(); return
    done = set()
    if TRADES.exists():
        try:
            done = set(pd.read_csv(TRADES, usecols=["symbol"])["symbol"].unique())
        except Exception:
            done = set()
    uni = universe()
    todo = [s for s in uni if s not in done]
    print(f"universe={len(uni)}  done={len(done)}  todo={len(todo)}", flush=True)
    con = sqlite3.connect(str(DB))
    new_header = not TRADES.exists()
    f = TRADES.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_header:
        w.writeheader()
    total_sig = 0
    for n, sym in enumerate(todo, 1):
        try:
            rows = symbol_rows(con, sym)
        except Exception as e:
            print(f"[{n}/{len(todo)}] {sym:12s} ERROR {e}", flush=True)
            continue
        for r in rows:
            w.writerow(r)
        f.flush()
        nsig = sum(1 for r in rows if r["arm"] == "SIGNAL") // len(POLICIES) if rows else 0
        total_sig += nsig
        print(f"[{n}/{len(todo)}] {sym:12s} rows={len(rows):>5} sigdays={nsig:>3} cumSIG={total_sig}", flush=True)
    f.close(); con.close()
    print("\nALL SYMBOLS DONE — aggregating", flush=True)
    aggregate()


if __name__ == "__main__":
    run()
