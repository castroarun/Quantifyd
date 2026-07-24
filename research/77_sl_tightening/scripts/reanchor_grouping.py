"""research/77 P3 — WHEN does the SL re-anchor help? Group the re-anchor(12:00) benefit by day-type.
Reconstructs the 09:16 ATM straddle (real premiums, chain-derived ATM), computes BASELINE vs TIME_12:00
per day, plus day features (DTE, weekday, moves derived from put-call parity S ~ K + CE - PE), then groups
the per-day DIFF to see which day-types the re-anchor helps / is neutral / hurts. Read-only."""
import sqlite3, datetime as dt, os
import numpy as np, pandas as pd

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = "/home/arun/quantifyd/research/77_sl_tightening/results"
LOT, SLIP = 65, 0.0015
con = sqlite3.connect(DB)


def tsec(s): return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])
def hm(x): return int(x[:2]) * 3600 + int(x[3:5]) * 60


ENTRY, NOON, EXITT = hm("09:16"), hm("12:00"), hm("15:15")
days = [r[0] for r in con.execute("SELECT DISTINCT date(snapshot_time) FROM option_chain ORDER BY 1").fetchall()]


def atm_from_chain(d, front):
    rows = con.execute(f"SELECT CAST(ROUND(strike) AS INT) k, instrument_type it, AVG(ltp) v FROM option_chain "
                       f"WHERE date(snapshot_time)='{d}' AND expiry_date='{front}' AND ltp>0 "
                       f"AND time(snapshot_time) BETWEEN '09:15:00' AND '09:18:30' AND instrument_type IN ('CE','PE') "
                       f"GROUP BY k, it").fetchall()
    ce = {k: v for k, it, v in rows if it == 'CE'}; pe = {k: v for k, it, v in rows if it == 'PE'}
    common = [k for k in ce if k in pe]
    return min(common, key=lambda k: abs(ce[k] - pe[k])) if common else None


def leg_series(d, front, k, it):
    df = pd.read_sql(f"SELECT snapshot_time t, ltp FROM option_chain WHERE date(snapshot_time)='{d}' "
                     f"AND expiry_date='{front}' AND CAST(ROUND(strike) AS INT)={k} AND instrument_type='{it}' "
                     f"AND ltp>0 AND time(snapshot_time) BETWEEN '09:15:30' AND '15:16:00' ORDER BY t", con)
    if len(df) < 5: return None
    return df["t"].apply(tsec).values, df["ltp"].values.astype(float)


def at(times, prems, sec):
    i = np.searchsorted(times, sec); i = min(max(i, 0), len(times) - 1); return float(prems[i])


def sim_leg(times, prems, mode, T=None):
    entry = prems[0]; cap = entry * 1.3; final = prems[-1]
    tT = hm(T) if T else None
    premT = at(times, prems, tT) if tT else None
    for i in range(len(prems)):
        p = prems[i]; t = times[i]
        sl = cap if (mode == "base" or t < tT) else min(cap, premT * 1.3)
        if p >= sl and i > 0:
            return p, True
    return final, False


rows = []
for d in days:
    exps = sorted([r[0] for r in con.execute(f"SELECT DISTINCT expiry_date FROM option_chain WHERE date(snapshot_time)='{d}' AND expiry_date IS NOT NULL").fetchall()])
    front = next((e for e in exps if e[:10] >= d), None)
    if not front: continue
    k = atm_from_chain(d, front)
    if k is None: continue
    ce, pe = leg_series(d, front, k, "CE"), leg_series(d, front, k, "PE")
    if ce is None or pe is None or ce[0][-1] < hm("14:30"): continue
    # parity spot path: S ~ K + CE - PE  (on the common CE grid)
    ct, cp = ce; pt, pp = pe
    S = lambda sec: k + at(ct, cp, sec) - at(pt, pp, sec)
    s_open, s_noon, s_close = S(ENTRY), S(NOON), S(EXITT)
    # grid of parity spots to get the day's excursion
    grid = np.arange(ENTRY, EXITT, 300)
    path = np.array([k + at(ct, cp, s) - at(pt, pp, s) for s in grid])
    dte = (dt.date.fromisoformat(front[:10]) - dt.date.fromisoformat(d)).days
    res = {}
    for nm, params in [("BASE", dict(mode="base")), ("T12", dict(mode="time", T="12:00"))]:
        g = c_ = 0.0
        for (tt, pr) in (ce, pe):
            e = pr[0]; xp, _ = sim_leg(tt, pr, **params)
            g += (e - xp) * LOT; c_ += (e + xp) * SLIP * LOT
        res[nm] = g - c_
    rows.append(dict(day=d, dte=dte, wd=dt.date.fromisoformat(d).strftime("%a"),
                     base=round(res["BASE"], 0), t12=round(res["T12"], 0),
                     diff=round(res["T12"] - res["BASE"], 0),
                     mv_day=abs(s_close - s_open), mv_pm=abs(s_close - s_noon),
                     mv_am=abs(s_noon - s_open), rng=path.max() - path.min(),
                     excursion=max(abs(path.max() - s_open), abs(path.min() - s_open))))

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "reanchor_grouping.csv"), index=False)
print("n=%d days | mean diff (T12 - BASE) = %+.0f/lot | days helped %d, neutral %d, hurt %d" % (
    len(R), R["diff"].mean(), (R["diff"] > 50).sum(), (R["diff"].abs() <= 50).sum(), (R["diff"] < -50).sum()))

print("\n=== by DTE ===")
for v, g in R.groupby("dte"):
    print("  DTE%-2s n=%2d  BASE %+7.0f  T12 %+7.0f  diff %+7.0f  (helped %d/%d)" % (
        v, len(g), g.base.mean(), g.t12.mean(), g["diff"].mean(), (g["diff"] > 50).sum(), len(g)))

print("\n=== by weekday ===")
for v, g in R.groupby("wd"):
    print("  %-4s n=%2d  BASE %+7.0f  T12 %+7.0f  diff %+7.0f" % (v, len(g), g.base.mean(), g.t12.mean(), g["diff"].mean()))

print("\n=== by AFTERNOON move |close-noon| (the thing the re-anchor protects) ===")
R["pm_b"] = pd.cut(R.mv_pm, [-1, 30, 60, 100, 1e9], labels=["<30", "30-60", "60-100", ">100"])
for v, g in R.groupby("pm_b", observed=True):
    print("  pm move %-7s n=%2d  BASE %+7.0f  T12 %+7.0f  diff %+7.0f  (helped %d/%d)" % (
        v, len(g), g.base.mean(), g.t12.mean(), g["diff"].mean(), (g["diff"] > 50).sum(), len(g)))

print("\n=== by FULL-DAY excursion from entry (trend vs range) ===")
R["ex_b"] = pd.cut(R.excursion, [-1, 50, 100, 150, 1e9], labels=["<50", "50-100", "100-150", ">150"])
for v, g in R.groupby("ex_b", observed=True):
    print("  excursion %-8s n=%2d  BASE %+7.0f  T12 %+7.0f  diff %+7.0f  (helped %d/%d)" % (
        v, len(g), g.base.mean(), g.t12.mean(), g["diff"].mean(), (g["diff"] > 50).sum(), len(g)))

print("\n=== the days it HELPED most / HURT most ===")
for _, r in R.sort_values("diff", ascending=False).head(5).iterrows():
    print("  +HELP %s DTE%s %s  base %+7.0f -> t12 %+7.0f  (diff %+6.0f)  pm_move %.0f excursion %.0f" % (
        r.day, r.dte, r.wd, r.base, r.t12, r["diff"], r.mv_pm, r.excursion))
for _, r in R.sort_values("diff").head(5).iterrows():
    print("  -HURT %s DTE%s %s  base %+7.0f -> t12 %+7.0f  (diff %+6.0f)  pm_move %.0f excursion %.0f" % (
        r.day, r.dte, r.wd, r.base, r.t12, r["diff"], r.mv_pm, r.excursion))
print("\ncorr(diff, afternoon move) = %.2f | corr(diff, excursion) = %.2f" % (
    R["diff"].corr(R.mv_pm), R["diff"].corr(R.excursion)))
print("\nDONE")
