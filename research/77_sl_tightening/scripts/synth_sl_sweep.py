"""research/77 P2 — SYNTHETIC 09:16 ATM straddle SL sweep over the long NIFTY history, to ESTIMATE/CONFIRM
the real-premium finding (n=15) on ~800 days. Premiums = Black-Scholes on NIFTY 5-min spot + REAL India VIX
(5-min) as IV; theta from calendar time-to-Thursday-expiry. Same per-leg SL policies as research/77.
CAVEATS: BS synthetic (no skew, no weekly-vs-30d-VIX term structure, no IV-crush, no bid/ask beyond a
0.15%/leg cost assumption); Thursday weekly expiry approx. Directionally captures price-action-driven SL
triggering + theta, which is what the question is about. Per 1 lot=65. Read-only on market_data.db."""
import sqlite3, datetime as dt, os, sys
from math import log, sqrt, exp, erf
import numpy as np, pandas as pd

DB = "/home/arun/quantifyd/backtest_data/market_data.db"
OUT = "/home/arun/quantifyd/research/77_sl_tightening/results"
LOT, SLIP, R = 65, 0.0015, 0.065
START, END = "2023-01-01", "2026-03-26"
con = sqlite3.connect(DB)


def ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs(S, K, T, sig, call):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    d1 = (log(S / K) + (R + 0.5 * sig * sig) * T) / (sig * sqrt(T)); d2 = d1 - sig * sqrt(T)
    return (S * ncdf(d1) - K * exp(-R * T) * ncdf(d2)) if call else (K * exp(-R * T) * ncdf(-d2) - S * ncdf(-d1))


def hm(x):
    return int(x[:2]) * 3600 + int(x[3:5]) * 60


ENTRY, EXITT = hm("09:16"), hm("15:15")

print("loading NIFTY 5min + India VIX ...", flush=True)
nf = pd.read_sql(f"SELECT date, close FROM market_data_unified WHERE symbol IN ('NIFTY50','NIFTY 50','NIFTY') AND timeframe='5minute' "
                 f"AND date>='{START}' AND date<'{END}' ORDER BY date", con)
vx = pd.read_sql(f"SELECT date, close vix FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='5minute' "
                 f"AND date>='{START}' AND date<'{END}' ORDER BY date", con)
nf["dt"] = pd.to_datetime(nf["date"]).astype("datetime64[ns]"); vx["dt"] = pd.to_datetime(vx["date"]).astype("datetime64[ns]")
m = pd.merge_asof(nf.sort_values("dt"), vx[["dt", "vix"]].sort_values("dt"), on="dt", direction="nearest", tolerance=pd.Timedelta("6min"))
m["vix"] = m["vix"].ffill().fillna(13.0)
m["d"] = m["dt"].dt.date
m["sec"] = m["dt"].dt.hour * 3600 + m["dt"].dt.minute * 60
print("bars:", len(m), "days:", m["d"].nunique(), flush=True)


def thursday_expiry(d):
    wd = d.weekday()                 # Mon=0..Sun=6, Thu=3
    off = (3 - wd) % 7               # days to coming Thursday
    return d + dt.timedelta(days=off)


def sim_leg(times, prems, mode, X=0.0, T=None, Y=0.0):
    entry = prems[0]; cap = entry * 1.3; final = prems[-1]
    tT = hm(T) if T else None
    premT = None
    if tT is not None:
        i = np.searchsorted(times, tT); i = min(max(i, 0), len(times) - 1); premT = prems[i]
    rmin = entry
    for i in range(len(prems)):
        p = prems[i]; t = times[i]; rmin = min(rmin, p)
        if mode == "base": sl = cap
        elif mode == "trail_be": sl = entry if rmin < entry else cap
        elif mode == "ratchet": sl = min(cap, rmin * (1 + X))
        elif mode == "time": sl = cap if t < tT else min(cap, premT * 1.3)
        elif mode == "profit_lock": sl = entry if rmin <= entry * (1 - Y) else cap
        else: sl = cap
        if p >= sl and i > 0:
            return p, True, final
    return final, False, final


POLICIES = [
    ("BASELINE", dict(mode="base")), ("TRAIL_BE", dict(mode="trail_be")),
    ("RATCHET_0.30", dict(mode="ratchet", X=0.30)),
    ("TIME_11:15", dict(mode="time", T="11:15")), ("TIME_12:00", dict(mode="time", T="12:00")),
    ("TIME_13:00", dict(mode="time", T="13:00")), ("TIME_13:30", dict(mode="time", T="13:30")),
    ("TIME_14:00", dict(mode="time", T="14:00")), ("PLOCK_50", dict(mode="profit_lock", Y=0.50)),
]

rows = []
n = 0
for d, g in m.groupby("d"):
    g = g[(g["sec"] >= ENTRY - 300) & (g["sec"] <= EXITT + 60)].sort_values("sec")
    if len(g) < 20:
        continue
    s0 = g.iloc[0]["close"]
    if not (5000 < s0 < 40000):
        continue
    K = round(s0 / 50) * 50
    exp_d = thursday_expiry(d); dte = (exp_d - d).days
    secs = g["sec"].values; spots = g["close"].values; vols = g["vix"].values / 100.0
    ce = np.empty(len(g)); pe = np.empty(len(g))
    for i in range(len(g)):
        # calendar time-to-expiry in years, with intraday fraction (24h clock)
        mins_to_close = 930 - secs[i] / 60.0           # 15:30 = 930 min
        Tyr = max(dte * 1440 + mins_to_close, 3.0) / 1440.0 / 365.0
        ce[i] = bs(spots[i], K, Tyr, vols[i], True); pe[i] = bs(spots[i], K, Tyr, vols[i], False)
    row = {"day": str(d), "dte": dte, "vix": round(float(vols[0] * 100), 1)}
    for name, params in POLICIES:
        gross = 0.0; cost = 0.0
        for arr in (ce, pe):
            e = arr[0]; xp, _, _ = sim_leg(secs, arr, **params)
            gross += (e - xp) * LOT; cost += (e + xp) * SLIP * LOT
        row[name] = round(gross - cost, 1)
    rows.append(row); n += 1
    if n % 100 == 0:
        print("...%d days" % n, flush=True)

Rd = pd.DataFrame(rows)
Rd.to_csv(os.path.join(OUT, "synth_per_day.csv"), index=False)
base = Rd["BASELINE"].mean()
print("\n=== SYNTHETIC SL SWEEP (BS on NIFTY 5min + real VIX, net 0.15%/leg, per 1 lot=65) ===")
print("period %s..%s | n=%d days | DTE0-1 %d, DTE2+ %d | mean VIX %.1f" % (
    START, END, len(Rd), (Rd.dte <= 1).sum(), (Rd.dte >= 2).sum(), Rd.vix.mean()))
print("  %-14s meanNet  vsBASE  medNet  win%%  worst    std   | DTE0-1  DTE2+" % "policy")
summ = []
for name, _ in POLICIES:
    c = Rd[name]
    summ.append(dict(policy=name, meanNet=c.mean(), vsBASE=c.mean() - base, medNet=c.median(),
                     winpct=100 * (c > 0).mean(), worst=c.min(), std=c.std(),
                     dte01=Rd.loc[Rd.dte <= 1, name].mean(), dte2=Rd.loc[Rd.dte >= 2, name].mean()))
    print("  %-14s %+6.0f  %+6.0f  %+6.0f  %3.0f%%  %+6.0f  %5.0f  | %+6.0f  %+6.0f" % (
        name, c.mean(), c.mean() - base, c.median(), 100 * (c > 0).mean(), c.min(), c.std(),
        Rd.loc[Rd.dte <= 1, name].mean(), Rd.loc[Rd.dte >= 2, name].mean()))
pd.DataFrame(summ).to_csv(os.path.join(OUT, "synth_summary.csv"), index=False)
# per-year stability of the winner vs baseline
Rd["yr"] = Rd["day"].str[:4]
print("\n=== per-year: TIME_13:00 vs BASELINE (mean net/lot) ===")
for yr, gg in Rd.groupby("yr"):
    print("  %s  n=%3d  BASE %+6.0f  T13:00 %+6.0f  diff %+5.0f" % (
        yr, len(gg), gg["BASELINE"].mean(), gg["TIME_13:00"].mean(), gg["TIME_13:00"].mean() - gg["BASELINE"].mean()))
print("\nDONE")
