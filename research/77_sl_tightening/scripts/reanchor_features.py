"""research/77 P3b — WHICH days does the SL re-anchor(12:00) help, and is it PREDICTABLE ex-ante?
Adds day-type features: gap, CPR + CPR width (from prev day's H/L/C), open-vs-CPR, opening-range move,
VIX, DTE, weekday — and groups the per-day re-anchor benefit by each. Spot path from put-call parity on
the ATM legs (S ~ K + CE - PE), so it works on every chain day. Read-only."""
import sqlite3, datetime as dt, os
import numpy as np, pandas as pd

ODB = "/home/arun/quantifyd/backtest_data/options_data.db"
MDB = "/home/arun/quantifyd/backtest_data/market_data.db"
OUT = "/home/arun/quantifyd/research/77_sl_tightening/results"
LOT, SLIP = 65, 0.0015
con = sqlite3.connect(ODB)
mcon = sqlite3.connect(MDB)


def tsec(s): return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])
def hm(x): return int(x[:2]) * 3600 + int(x[3:5]) * 60


ENTRY, OR30, NOON, EXITT = hm("09:16"), hm("09:46"), hm("12:00"), hm("15:15")
days = [r[0] for r in con.execute("SELECT DISTINCT date(snapshot_time) FROM option_chain ORDER BY 1").fetchall()]

vix = {r[0][:10]: r[1] for r in mcon.execute("SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'").fetchall()}


def atm_from_chain(d, front):
    rows = con.execute(f"SELECT CAST(ROUND(strike) AS INT) k, instrument_type it, AVG(ltp) v FROM option_chain "
                       f"WHERE date(snapshot_time)='{d}' AND expiry_date='{front}' AND ltp>0 "
                       f"AND time(snapshot_time) BETWEEN '09:15:00' AND '09:18:30' AND instrument_type IN ('CE','PE') GROUP BY k,it").fetchall()
    ce = {k: v for k, it, v in rows if it == 'CE'}; pe = {k: v for k, it, v in rows if it == 'PE'}
    common = [k for k in ce if k in pe]
    return min(common, key=lambda k: abs(ce[k] - pe[k])) if common else None


def leg(d, front, k, it):
    df = pd.read_sql(f"SELECT snapshot_time t, ltp FROM option_chain WHERE date(snapshot_time)='{d}' AND expiry_date='{front}' "
                     f"AND CAST(ROUND(strike) AS INT)={k} AND instrument_type='{it}' AND ltp>0 "
                     f"AND time(snapshot_time) BETWEEN '09:15:30' AND '15:16:00' ORDER BY t", con)
    if len(df) < 5: return None
    return df["t"].apply(tsec).values, df["ltp"].values.astype(float)


def at(t, p, s):
    i = np.searchsorted(t, s); i = min(max(i, 0), len(t) - 1); return float(p[i])


def sim(times, prems, mode, T=None):
    e = prems[0]; cap = e * 1.3; fin = prems[-1]
    tT = hm(T) if T else None
    pT = at(times, prems, tT) if tT else None
    for i in range(len(prems)):
        p = prems[i]
        sl = cap if (mode == "base" or times[i] < tT) else min(cap, pT * 1.3)
        if p >= sl and i > 0: return p
    return fin


rows = []; prev = None
for d in days:
    exps = sorted([r[0] for r in con.execute(f"SELECT DISTINCT expiry_date FROM option_chain WHERE date(snapshot_time)='{d}' AND expiry_date IS NOT NULL").fetchall()])
    front = next((e for e in exps if e[:10] >= d), None)
    if not front: continue
    k = atm_from_chain(d, front)
    if k is None: continue
    ce, pe = leg(d, front, k, "CE"), leg(d, front, k, "PE")
    if ce is None or pe is None or ce[0][-1] < hm("14:30"): continue
    ct, cp = ce; pt, pp = pe
    grid = np.arange(ENTRY, EXITT + 1, 300)
    path = np.array([k + at(ct, cp, s) - at(pt, pp, s) for s in grid])
    s_open, s_close = float(path[0]), float(path[-1])
    s_noon = k + at(ct, cp, NOON) - at(pt, pp, NOON)
    s_or = k + at(ct, cp, OR30) - at(pt, pp, OR30)
    H, L, C = float(path.max()), float(path.min()), s_close
    dte = (dt.date.fromisoformat(front[:10]) - dt.date.fromisoformat(d)).days
    base = t12 = 0.0
    for arr in (ce, pe):
        e = arr[1][0]
        base += (e - sim(arr[0], arr[1], "base")) * LOT - (e + sim(arr[0], arr[1], "base")) * SLIP * LOT
        x = sim(arr[0], arr[1], "time", "12:00")
        t12 += (e - x) * LOT - (e + x) * SLIP * LOT
    r = dict(day=d, dte=dte, wd=dt.date.fromisoformat(d).strftime("%a"), base=round(base), t12=round(t12),
             diff=round(t12 - base), mv_pm=abs(s_close - s_noon), excursion=max(abs(H - s_open), abs(L - s_open)),
             or30=abs(s_or - s_open), vix=vix.get(d), H=H, L=L, C=C, O=s_open)
    if prev:
        pH, pL, pC = prev["H"], prev["L"], prev["C"]
        piv = (pH + pL + pC) / 3.0; bc = (pH + pL) / 2.0; tc = 2 * piv - bc
        cw = abs(tc - bc) / piv * 100.0
        r["gap"] = s_open - pC
        r["cpr_w"] = cw
        r["open_vs_cpr"] = "above" if s_open > max(tc, bc) else ("below" if s_open < min(tc, bc) else "inside")
    rows.append(r); prev = r

R = pd.DataFrame(rows).dropna(subset=["gap"])
R.to_csv(os.path.join(OUT, "reanchor_features.csv"), index=False)
print("n=%d days (with prev-day CPR) | mean diff (T12-BASE) = %+.0f/lot | helped %d, neutral %d, hurt %d" % (
    len(R), R["diff"].mean(), (R["diff"] > 50).sum(), (R["diff"].abs() <= 50).sum(), (R["diff"] < -50).sum()))


def grp(col, bins, labels, title):
    R["_b"] = pd.cut(R[col], bins, labels=labels)
    print("\n=== by %s ===" % title)
    for v, g in R.groupby("_b", observed=True):
        print("  %-12s n=%2d  BASE %+7.0f  T12 %+7.0f  diff %+7.0f  (helped %d/%d)" % (
            v, len(g), g.base.mean(), g.t12.mean(), g["diff"].mean(), (g["diff"] > 50).sum(), len(g)))


grp("cpr_w", [-1, 0.15, 0.30, 0.60, 99], ["ultra-narrow", "narrow", "medium", "wide"], "CPR WIDTH % (narrow=trend day)")
grp("gap", [-1e9, -60, -20, 20, 60, 1e9], ["gapdn>60", "gapdn20-60", "flat", "gapup20-60", "gapup>60"], "GAP")
grp("or30", [-1, 20, 40, 70, 1e9], ["<20", "20-40", "40-70", ">70"], "OPENING-RANGE move (09:16-09:46)")
grp("mv_pm", [-1, 30, 60, 100, 1e9], ["<30", "30-60", "60-100", ">100"], "AFTERNOON move |close-noon|")
grp("excursion", [-1, 50, 100, 150, 1e9], ["<50", "50-100", "100-150", ">150"], "DAY EXCURSION from entry")
if R.vix.notna().any():
    grp("vix", [-1, 12, 14, 16, 99], ["<12", "12-14", "14-16", ">16"], "INDIA VIX")
print("\n=== by open vs CPR ===")
for v, g in R.groupby("open_vs_cpr"):
    print("  %-8s n=%2d  BASE %+7.0f  T12 %+7.0f  diff %+7.0f" % (v, len(g), g.base.mean(), g.t12.mean(), g["diff"].mean()))
print("\n=== by DTE ===")
for v, g in R.groupby("dte"):
    print("  DTE%-2s n=%2d  BASE %+7.0f  T12 %+7.0f  diff %+7.0f" % (v, len(g), g.base.mean(), g.t12.mean(), g["diff"].mean()))
print("\ncorrelations of diff with: cpr_w %.2f | gap %.2f | or30 %.2f | mv_pm %.2f | excursion %.2f" % (
    R["diff"].corr(R.cpr_w), R["diff"].corr(R.gap.abs()), R["diff"].corr(R.or30), R["diff"].corr(R.mv_pm), R["diff"].corr(R.excursion)))
print("\nDONE")
