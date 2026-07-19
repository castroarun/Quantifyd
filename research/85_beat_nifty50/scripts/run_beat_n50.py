"""research/85: beat Nifty by >=5pp within Nifty 50 — R1 momentum rotation,
R2 turtle-N50, R3 KC6-N50. Pre-registration: ../BEAT_NIFTY50_DAILY_SWEEP_STATUS.md
Reports IS (2005-2018) AND Val (2019-2023) per-year vs NIFTYBEES. OOS 2024+ untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R81 = HERE.parents[1] / "81_swing_edge_discovery"
R83S = HERE.parents[1] / "83_turtle_equities" / "scripts"
for p in (str(R81), str(R81.parents[1]), str(R83S)):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader, metrics                     # noqa: E402
from run_turtle_probe import turtle_sim                # noqa: E402

RESULTS = HERE.parent / "results"
START, END = "2005-01-01", "2023-12-31"          # IS+Val; OOS untouched
RT_COST = 0.0030                                  # delivery round-trip ~30bps
N50 = ["ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
       "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
       "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH",
       "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR",
       "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JSWSTEEL", "KOTAKBANK",
       "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
       "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
       "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN",
       "TRENT", "ULTRACEMCO", "WIPRO"]


def report(nav: pd.Series, bench: pd.Series, label: str):
    nav = nav.dropna()
    b = bench.reindex(nav.index).ffill()
    for win, lo, hi in (("IS 2005-18", "2005-01-01", "2018-12-31"),
                        ("Val 2019-23", "2019-01-01", "2023-12-31"),
                        ("FULL", START, END)):
        s = nav[(nav.index >= lo) & (nav.index <= hi)]
        bb = b[(b.index >= lo) & (b.index <= hi)]
        if len(s) < 100:
            continue
        cs = metrics.curve_stats(s)
        cb = metrics.curve_stats(bb / bb.iloc[0])
        exc = cs["cagr"] - cb["cagr"]
        print(f"{label} [{win}]: CAGR {cs['cagr']:.1%} (bench {cb['cagr']:.1%}, "
              f"EXCESS {exc:+.1%}) MaxDD {cs['max_dd']:.0%} (bench {cb['max_dd']:.0%}) "
              f"Sharpe {cs['sharpe']:.2f} Calmar {cs['calmar']:.2f}", flush=True)
    ry = metrics.per_year_table(nav.pct_change())["return"]
    by = metrics.per_year_table(b.pct_change())["return"]
    exc = ((ry - by) * 100).round(1)
    print(f"  yearly excess: {dict(exc)}  beat {int((exc > 0).sum())}/{len(exc)} yrs",
          flush=True)


def r1_momentum(closes: pd.DataFrame, bench: pd.Series):
    me = closes.resample("ME").last()
    gate = (bench.shift(1) > bench.rolling(200).mean().shift(1))
    gate_me = gate.resample("ME").last()
    for lb_name, lb, skip in (("6m", 126, 0), ("12-1", 252, 21)):
        mom_d = closes.shift(skip) / closes.shift(lb) - 1
        mom = mom_d.resample("ME").last()
        for topn in (5, 10):
            for gname, use_gate in (("nogate", False), ("gate200", True)):
                eq = 1.0
                navs = {}
                prev = set()
                for i in range(24, len(me) - 1):
                    m = me.index[i]
                    r = mom.loc[m].dropna()
                    picks = set(r.nlargest(topn).index) if len(r) >= topn else set()
                    if use_gate and not bool(gate_me.loc[m]):
                        picks = set()
                    # next month's return equal-weight
                    nxt = me.index[i + 1]
                    if picks:
                        rets = (me.loc[nxt, list(picks)] / me.loc[m, list(picks)] - 1)
                        gross = rets.mean()
                    else:
                        gross = 0.0
                    turn = len(picks - prev) / max(topn, 1)
                    eq *= (1 + gross - turn * RT_COST)
                    prev = picks
                    navs[nxt] = eq
                nav = pd.Series(navs)
                report(nav, bench.resample("ME").last(),
                       f"R1 mom{lb_name}_top{topn}_{gname}")


def r2_turtle(bars: dict, closes_d: dict, bench: pd.Series):
    gate = (bench.shift(1) > bench.rolling(200).mean().shift(1))
    trades = []
    for sym, df in bars.items():
        for (ni, no) in ((20, 10), (55, 20)):
            t = turtle_sim(df, ni, no, 1, sym, stop_mult=2.0)
            trades.append(t)
    tr = pd.concat(trades, ignore_index=True)
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    tr["exit_time"] = pd.to_datetime(tr["exit_time"])
    tr = tr[(tr.entry_time >= START) & (tr.entry_time <= END)]
    tr = tr.sort_values("entry_time").reset_index(drop=True)
    tr["eday"] = tr.entry_time.dt.normalize()
    cal = pd.DatetimeIndex(sorted({d for c in closes_d.values() for d in c.index
                                   if pd.Timestamp(START) <= d <= pd.Timestamp(END)}))
    eq, open_pos, ti = 1.0, [], 0
    nav = pd.Series(np.nan, index=cal)
    for d in cal:
        while ti < len(tr) and tr.loc[ti, "eday"] <= d:
            t = tr.loc[ti]
            ti += 1
            if t["eday"] != d or not bool(gate.get(d, False)) or len(open_pos) >= 8:
                continue
            px = closes_d[t["symbol"]].get(d, np.nan)
            if np.isnan(px):
                continue
            qty = (eq * 0.12) / px
            open_pos.append({"symbol": t["symbol"], "qty": qty, "mark": px,
                             "exit_day": t["exit_time"].normalize(),
                             "epx": px, "net_ret": t["net_ret"]})
        pnl, still = 0.0, []
        for p in open_pos:
            c = closes_d[p["symbol"]].get(d, np.nan)
            if d >= p["exit_day"]:
                pnl += p["qty"] * p["epx"] * p["net_ret"] \
                    - p["qty"] * (p["mark"] - p["epx"])
            else:
                if not np.isnan(c):
                    pnl += p["qty"] * (c - p["mark"])
                    p["mark"] = c
                still.append(p)
        open_pos = still
        eq += pnl
        nav[d] = eq
    report(nav.ffill(), bench, "R2 turtleN50")


def r3_kc6(bars: dict, bench: pd.Series):
    slots = 10
    campaigns = []
    for sym, df in bars.items():
        c, h, l = df["close"], df["high"], df["low"]
        ema6 = c.ewm(span=6, adjust=False).mean()
        pc = c.shift(1)
        tr_ = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr = tr_.rolling(10).mean()
        lower = (ema6 - 1.3 * atr).shift(0)
        sma200 = c.rolling(200).mean()
        sig = (c < lower) & (c > sma200)
        ci = c.to_numpy(float)
        oi = df["open"].to_numpy(float)
        mid = ema6.to_numpy(float)
        idx = df.index
        i, n = 210, len(df)
        while i < n - 1:
            if sig.iloc[i] and pd.Timestamp(START) <= idx[i] <= pd.Timestamp(END):
                e = oi[i + 1]
                if e <= 0:
                    i += 1
                    continue
                j, exit_px = i + 1, None
                while j < min(i + 1 + 15, n):
                    if ci[j] <= e * 0.95:
                        exit_px, reason = e * 0.95, "SL"
                        break
                    if ci[j] >= e * 1.15:
                        exit_px, reason = e * 1.15, "TP"
                        break
                    if ci[j] >= mid[j]:
                        exit_px, reason = ci[j], "MID"
                        break
                    j += 1
                if exit_px is None:
                    j = min(i + 15, n - 1)
                    exit_px, reason = ci[j], "MAXHOLD"
                ret = exit_px / e - 1 - RT_COST
                campaigns.append({"symbol": sym, "entry": idx[i + 1],
                                  "exit": idx[j], "ret": ret})
                i = j
            i += 1
    cs = pd.DataFrame(campaigns).sort_values("entry").reset_index(drop=True)
    eq, busy = 1.0, []
    navs = {}
    for _, r in cs.iterrows():
        busy = [b for b in busy if b > r.entry]
        if len(busy) >= slots:
            continue
        eq *= (1 + r.ret / slots)
        busy.append(r.exit)
        navs[r.exit] = eq
    nav = pd.Series(navs)
    nav = nav[~nav.index.duplicated(keep="last")].sort_index()
    print(f"R3 kc6N50: {len(cs)} signals, win {(cs.ret>0).mean():.0%}, "
          f"mean {cs.ret.mean()*100:.2f}%")
    report(nav, bench, "R3 kc6N50")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    bars = {}
    for sym in N50:
        df = loader.load_bars(sym, "day", start="2004-01-01", end=END)
        if len(df) >= 500:
            bars[sym] = df
    print(f"universe: {len(bars)}/{len(N50)} Nifty-50 names with data")
    closes = pd.DataFrame({s: b["close"] for s, b in bars.items()})
    closes_d = {s: b["close"] for s, b in bars.items()}
    bench = loader.load_bars("NIFTYBEES", "day", start="2004-01-01", end=END)["close"]
    print("\n--- R1 momentum rotation ---")
    r1_momentum(closes, bench)
    print("\n--- R2 turtle N50 ---")
    r2_turtle(bars, closes_d, bench)
    print("\n--- R3 KC6 N50 ---")
    r3_kc6(bars, bench)
    print("\nBEAT-N50 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
