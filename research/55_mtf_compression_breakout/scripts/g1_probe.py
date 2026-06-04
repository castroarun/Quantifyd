"""G1 probe — MTF compression breakout + alpha-vs-beta ABLATION.

Entry confluence (LONG): daily uptrend (SMA50) + price above weekly CPR top +
prior-day NR7 + narrow daily CPR + break of max(prev-day,prev-week) high +
clean directional bar + volume surge. Held positionally, managed on daily bars.

The point of G1 is NOT "is it positive" (a long breakout in an uptrend always is
in a bull) but "does SIGNAL beat TREND_BASELINE" (enter any uptrend day). Four
arms share identical exits so we can see which layers add alpha (playbook §6).

Reuses research/40 signal_lib.py. Reads only backtest_data/market_data.db.
Run: venv/bin/python research/55_mtf_compression_breakout/scripts/g1_probe.py
"""
from __future__ import annotations
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
TRADES = RESULTS / "g1_trades.csv"

sys.path.insert(0, str(REPO / "research/40_volsurge_pdr_break_weekly_cpr/scripts"))
import signal_lib as sl  # noqa: E402

UNIVERSE = ["TDPOWERSYS", "BHARATFORG", "ADANIENT", "TRENT", "BEL", "HAL",
            "RELIANCE", "TCS", "ICICIBANK", "SBIN", "MARUTI", "TATASTEEL"]
START, END = "2018-01-01", "2026-06-01"
TRIGGER_TF = "15min"
NR_WIN = 7
NARROW_CPR_PCT = 0.6          # daily CPR width% below this = "narrow CPR"
VOL_K = 1.5
ATR_D_LEN, ST_P, ST_M, MAXHOLD = 14, 10, 3.0, 30
COST_BPS = [0.0, 6.0, 12.0]
B_MIN, ZONE = sl.CLEAN_PRESETS["loose"]
POLICIES = ["HARD_SL", "R_3R", "CHANDELIER_3ATR", "SUPERTREND_D_10_3", "MAXHOLD_10"]
ARMS = ["SIGNAL", "NO_VOL", "NO_COMPRESSION", "TREND_BASELINE"]


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
    """Today's daily-CPR width% from PREV day H/L/C (causal, indexed by date)."""
    H = daily["high"].shift(1); L = daily["low"].shift(1); C = daily["close"].shift(1)
    P = (H+L+C)/3.0; bc = (H+L)/2.0; tc = 2*P-bc
    width = (tc-bc).abs()
    return (width/P*100.0)


def sim_pos(entry, atrv, rem, dfut, sfut):
    R = atrv; stop0 = entry-R; t3 = entry+3*R
    dayD = bool((rem["low"] <= stop0).any()) if len(rem) else False
    lastc = float(dfut["close"].iloc[-1]) if len(dfut) else float(rem["close"].iloc[-1])
    hi = dfut["high"].to_numpy(); lo = dfut["low"].to_numpy(); cl = dfut["close"].to_numpy(); sd = sfut.to_numpy()
    out = {}
    px = stop0 if dayD else lastc
    if not dayD:
        for k in range(len(dfut)):
            if lo[k] <= stop0: px = stop0; break
    out["HARD_SL"] = px
    px = stop0 if dayD else lastc
    if not dayD:
        for k in range(len(dfut)):
            if lo[k] <= stop0: px = stop0; break
            if hi[k] >= t3: px = t3; break
    out["R_3R"] = px
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


def run():
    con = sqlite3.connect(str(DB))
    rows = []
    for sym in UNIVERSE:
        d5 = load(con, sym, "5minute", START, END)
        daily = load(con, sym, "day", START, END)
        if d5.empty or daily.empty:
            print(f"{sym:12s} skip (5m={len(d5)} d={len(daily)})"); continue
        daily = daily.sort_index()
        df15 = sl.resample_5m(d5, TRIGGER_TF)
        if df15.empty:
            print(f"{sym:12s} skip (no 15m)"); continue
        df15["volma"] = df15["volume"].rolling(500, min_periods=100).mean().shift(1)
        df15["day"] = df15.index.normalize()

        trend = sl.daily_trend(daily, "sma50")
        trend_prev = {d.normalize(): trend.shift(1).get(d) for d in trend.index}
        wcpr = sl.weekly_cpr(daily)
        rng = daily["high"]-daily["low"]
        nr7_prev = (rng <= rng.rolling(NR_WIN, min_periods=NR_WIN).min()).shift(1)
        nr7_prev = {d.normalize(): bool(nr7_prev.get(d)) if not pd.isna(nr7_prev.get(d)) else False for d in daily.index}
        cprw = daily_cpr_width_pct(daily)
        narrow_prev = {d.normalize(): (float(cprw.get(d)) < NARROW_CPR_PCT) if not pd.isna(cprw.get(d)) else False for d in daily.index}
        atrd = atr_w(daily, ATR_D_LEN)
        atrd_prev = {d.normalize(): atrd.shift(1).get(d) for d in atrd.index}
        std = st_dir(daily, ST_P, ST_M)
        dhigh = daily["high"]; didx = daily.index

        nsig = 0
        for day, g in df15.groupby("day", sort=True):
            up = trend_prev.get(day) == "up"
            if not up:
                continue                       # all arms require uptrend
            atrv = atrd_prev.get(day)
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
            sfut = std.loc[dfut.index]
            g = g.sort_index()
            nr7 = nr7_prev.get(day, False)
            narrow = narrow_prev.get(day, False)
            compression = nr7 and narrow

            picks = {}
            # TREND_BASELINE: first 15m candle, enter next bar
            if len(g) > 1:
                picks["TREND_BASELINE"] = 0
            for i in range(len(g)-1):
                r = g.iloc[i]
                o, h, l, c, v = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])
                clean = sl.is_clean_candle(o, h, l, c, "long", B_MIN, ZONE)
                esc = sl.range_escape(c, pdh, 0, pwh, 0, "long")
                above = c > ctop
                vol_ok = (not np.isnan(r["volma"])) and sl.volume_surge(v, float(r["volma"]), VOL_K)
                base = clean and esc and above
                if "NO_COMPRESSION" not in picks and base and vol_ok:
                    picks["NO_COMPRESSION"] = i
                if compression:
                    if "NO_VOL" not in picks and base:
                        picks["NO_VOL"] = i
                    if "SIGNAL" not in picks and base and vol_ok:
                        picks["SIGNAL"] = i
            for arm, i in picks.items():
                entry = float(g.iloc[i+1]["open"]); rem = g.iloc[i+1:]
                ex = sim_pos(entry, float(atrv), rem, dfut, sfut)
                for pol, xp in ex.items():
                    rows.append({"symbol": sym, "arm": arm, "policy": pol,
                                 "year": day.year, "entry": entry, "R": float(atrv),
                                 "gross_price": xp-entry, "net6R": ((xp-entry)-6/1e4*entry)/float(atrv)})
                if arm == "SIGNAL":
                    nsig += 1
        print(f"{sym:12s} 15m={len(df15):>6} SIGNAL_days={nsig}")
    con.close()

    if not rows:
        print("NO TRADES"); return
    t = pd.DataFrame(rows); t.to_csv(TRADES, index=False)
    print(f"\nWrote {len(t)} rows -> {TRADES}\n")

    print("="*92)
    print("G1 — net@6bp R / PF / WR / n  by ARM x POLICY.  ALPHA = SIGNAL >> TREND_BASELINE")
    print("="*92)
    for pol in POLICIES:
        print(f"\n### {pol}")
        print(f"{'arm':<16} {'n':>6} {'WR%':>6} {'grossR':>8} {'net6R':>8} {'PF6':>6}")
        for arm in ARMS:
            s = t[(t.policy == pol) & (t.arm == arm)]
            if s.empty:
                continue
            net = s["net6R"]; pos = net[net > 0].sum(); neg = -net[net < 0].sum()
            pf = pos/neg if neg > 0 else float("inf")
            print(f"{arm:<16} {len(s):>6} {100*(net>0).mean():>6.1f} "
                  f"{(s['gross_price']/s['R']).mean():>8.3f} {net.mean():>8.3f} {pf:>6.2f}")

    print("\n### ALPHA DELTA (SIGNAL net6R - TREND_BASELINE net6R), per policy")
    for pol in POLICIES:
        a = t[(t.policy == pol) & (t.arm == "SIGNAL")]["net6R"].mean()
        b = t[(t.policy == pol) & (t.arm == "TREND_BASELINE")]["net6R"].mean()
        if pd.notna(a) and pd.notna(b):
            verdict = "ALPHA" if (a-b) >= 0.10 else "no-alpha"
            print(f"  {pol:<18} SIGNAL {a:+.3f}  BASELINE {b:+.3f}  delta {a-b:+.3f}  -> {verdict}")


if __name__ == "__main__":
    run()
