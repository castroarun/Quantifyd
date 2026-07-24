"""research/77 P1b — SL-tightening sweep, FIXED reconstruction over ALL usable option-chain days.
Prior version keyed ATM off the patchy underlying_spot TABLE -> only 15/56 days. This derives ATM from the
CHAIN itself (09:16 strike where |CE-PE| is smallest = the recorded ATM) -> recovers ~all 56 days. Real
option premiums (options_data.db). Same per-leg SL policies. All DTE. Per 1 lot=65, net 0.15%/leg. Read-only."""
import sqlite3, datetime as dt, os, sys
import numpy as np, pandas as pd

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = "/home/arun/quantifyd/research/77_sl_tightening/results"
LOT, SLIP = 65, 0.0015
con = sqlite3.connect(DB)


def tsec(s):
    return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def hm(x):
    return int(x[:2]) * 3600 + int(x[3:5]) * 60


ENTRY, EXITT = hm("09:16"), hm("15:15")
days = [r[0] for r in con.execute("SELECT DISTINCT date(snapshot_time) FROM option_chain ORDER BY 1").fetchall()]


def atm_from_chain(d, front):
    rows = con.execute(f"SELECT CAST(ROUND(strike) AS INT) k, instrument_type it, AVG(ltp) v FROM option_chain "
                       f"WHERE date(snapshot_time)='{d}' AND expiry_date='{front}' AND ltp>0 "
                       f"AND time(snapshot_time) BETWEEN '09:15:00' AND '09:18:30' AND instrument_type IN ('CE','PE') "
                       f"GROUP BY k, it").fetchall()
    ce = {k: v for k, it, v in rows if it == 'CE'}
    pe = {k: v for k, it, v in rows if it == 'PE'}
    common = [k for k in ce if k in pe]
    if not common:
        return None
    return min(common, key=lambda k: abs(ce[k] - pe[k]))


def leg_series(d, front, k, it):
    df = pd.read_sql(f"SELECT snapshot_time t, ltp FROM option_chain WHERE date(snapshot_time)='{d}' "
                     f"AND expiry_date='{front}' AND CAST(ROUND(strike) AS INT)={k} AND instrument_type='{it}' "
                     f"AND ltp>0 AND time(snapshot_time) BETWEEN '09:15:30' AND '15:16:00' ORDER BY t", con)
    if len(df) < 5:
        return None
    return df["t"].apply(tsec).values, df["ltp"].values.astype(float)


def load(d):
    exps = sorted([r[0] for r in con.execute(f"SELECT DISTINCT expiry_date FROM option_chain WHERE date(snapshot_time)='{d}' AND expiry_date IS NOT NULL").fetchall()])
    front = next((e for e in exps if e[:10] >= d), exps[0] if exps else None)
    if not front:
        return None
    k = atm_from_chain(d, front)
    if k is None:
        return None
    ce, pe = leg_series(d, front, k, "CE"), leg_series(d, front, k, "PE")
    if ce is None or pe is None:
        return None
    if ce[0][-1] < hm("14:30") or pe[0][-1] < hm("14:30"):   # series must reach late session
        return None
    dte = (dt.date.fromisoformat(front[:10]) - dt.date.fromisoformat(d)).days
    return dict(dte=dte, atm=k, ce=ce, pe=pe)


def prem_at(times, prems, sec):
    i = np.searchsorted(times, sec); i = min(max(i, 0), len(times) - 1); return float(prems[i])


def sim_leg(times, prems, mode, X=0.0, T=None, Y=0.0):
    entry = prems[0]; cap = entry * 1.3; final = prems[-1]
    tT = hm(T) if T else None
    premT = prem_at(times, prems, tT) if tT else None
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
    ("RATCHET_0.20", dict(mode="ratchet", X=0.20)), ("RATCHET_0.30", dict(mode="ratchet", X=0.30)),
    ("RATCHET_0.40", dict(mode="ratchet", X=0.40)),
    ("TIME_11:15", dict(mode="time", T="11:15")), ("TIME_12:00", dict(mode="time", T="12:00")),
    ("TIME_13:00", dict(mode="time", T="13:00")), ("TIME_14:00", dict(mode="time", T="14:00")),
    ("PLOCK_50", dict(mode="profit_lock", Y=0.50)),
]


def day_pnl(D, params, slip):
    gross = cost = premature = saved = 0.0; stops = 0
    for legname in ("ce", "pe"):
        times, prems = D[legname]; entry = prems[0]
        xp, stopped, final = sim_leg(times, prems, **params)
        gross += (entry - xp) * LOT; cost += (entry + xp) * slip * LOT
        if stopped:
            stops += 1
            if final < xp: premature += (xp - final) * LOT
            else: saved += (final - xp) * LOT
    return gross - cost, stops, premature, saved


rows = []; n = 0; fail = 0
for d in days:
    try:
        D = load(d)
    except Exception:
        D = None
    if D is None:
        fail += 1; continue
    row = {"day": d, "dte": D["dte"], "atm": D["atm"]}
    for name, params in POLICIES:
        net, stops, prem, saved = day_pnl(D, params, SLIP)
        row[name] = round(net, 1)
        if name != "BASELINE":
            row[name + "_prem"] = round(prem, 1); row[name + "_saved"] = round(saved, 1)
    rows.append(row); n += 1

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "real_v2_per_day.csv"), index=False)
base = R["BASELINE"].mean()
print("=== SL SWEEP on REAL premiums, FIXED reconstruction (n=%d days, dropped %d) ===" % (len(R), fail))
print("period %s..%s | DTE dist: %s | net 0.15%%/leg, per 1 lot=65" % (
    R.day.min(), R.day.max(), dict(R.dte.value_counts().sort_index())))
print("  %-14s meanNet  vsBASE  medNet  win%%  worst   avgStops | netProtection(saved-prem) | DTE0-1  DTE2+" % "policy")
for name, _ in POLICIES:
    c = R[name]; stops = 0
    prem = R[name + "_prem"].sum() if name + "_prem" in R else 0.0
    saved = R[name + "_saved"].sum() if name + "_saved" in R else 0.0
    d01 = R.loc[R.dte <= 1, name].mean(); d2 = R.loc[R.dte >= 2, name].mean()
    print("  %-14s %+6.0f  %+6.0f  %+6.0f  %3.0f%%  %+6.0f            | %+10.0f            | %+6.0f  %+6.0f" % (
        name, c.mean(), c.mean() - base, c.median(), 100 * (c > 0).mean(), c.min(), saved - prem, d01, d2))
print("\nDONE")
