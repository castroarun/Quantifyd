"""research/77 — 09:16 ATM straddle: per-leg SL-tightening sweep, reconstructed from options_data.db.
Does an intraday-tightened per-leg stop beat the fixed SL=entry*1.3 (30% off morning premium)?
Each leg walks its own premium series; a leg stops when premium >= its (policy) SL, booking that leg;
the other runs to 15:15. Net of 0.15%%/leg cost. Per 1 lot=65. Read-only. Writes incrementally.
No re-enter after a stop (isolates SL-timing effect) — noted caveat. Verdict per QUANT_RESEARCH_PLAYBOOK."""
import sqlite3, datetime as dt, os, sys
import numpy as np, pandas as pd

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = "/home/arun/quantifyd/research/77_sl_tightening/results"
os.makedirs(OUT, exist_ok=True)
LOT = 65
SLIP = 0.0015
con = sqlite3.connect(DB)


def tsec(s):
    return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def hm(x):
    return int(x[:2]) * 3600 + int(x[3:5]) * 60


ENTRY, EXIT = hm("09:16"), hm("15:15")
days = pd.read_sql("SELECT DISTINCT date(snapshot_time) d FROM underlying_spot ORDER BY d", con)["d"].tolist()
DC = {}


def spot0(d):
    r = con.execute(f"SELECT AVG(spot_price) FROM underlying_spot WHERE snapshot_time BETWEEN "
                    f"'{d}T09:14:00' AND '{d}T09:19:00' AND spot_price BETWEEN 20000 AND 28000").fetchone()[0]
    return r


def leg_series(d, front, strike, it):
    df = pd.read_sql(f"SELECT snapshot_time t, ltp FROM option_chain WHERE date(snapshot_time)='{d}' "
                     f"AND expiry_date='{front}' AND CAST(ROUND(strike) AS INT)={strike} AND instrument_type='{it}' "
                     f"AND ltp>0 AND time(snapshot_time) BETWEEN '09:15:30' AND '15:16:00' ORDER BY t", con)
    if df.empty:
        return None
    return df["t"].apply(tsec).values, df["ltp"].values.astype(float)


def load(d):
    if d in DC:
        return DC[d]
    try:
        exps = pd.read_sql(f"SELECT DISTINCT expiry_date e FROM option_chain WHERE date(snapshot_time)='{d}' AND expiry_date IS NOT NULL", con)["e"].dropna().tolist()
        exps = sorted(exps)
        s0 = spot0(d)
        if not exps or not s0:
            DC[d] = None; return None
        front = next((e for e in exps if e[:10] >= d), exps[0])
        atm = int(round(s0 / 50) * 50)
        ce, pe = leg_series(d, front, atm, "CE"), leg_series(d, front, atm, "PE")
        if ce is None or pe is None:
            DC[d] = None; return None
        dte = (dt.date.fromisoformat(front[:10]) - dt.date.fromisoformat(d)).days
        DC[d] = dict(dte=dte, atm=atm, ce=ce, pe=pe)
    except Exception:
        DC[d] = None
    return DC[d]


def prem_at(times, prems, sec):
    i = np.searchsorted(times, sec)
    i = min(max(i, 0), len(times) - 1)
    return float(prems[i])


def sim_leg(times, prems, mode, X=0.0, T=None, Y=0.0):
    """Return (exit_prem, stopped, final_prem). SHORT leg: SL is an upper bound; stop when prem>=SL."""
    entry = prems[0]
    cap = entry * 1.3
    final = prems[-1]
    tsec_T = hm(T) if T else None
    premT = prem_at(times, prems, tsec_T) if tsec_T else None
    rmin = entry
    for i in range(len(prems)):
        p = prems[i]; t = times[i]
        rmin = min(rmin, p)
        if mode == "base":
            sl = cap
        elif mode == "trail_be":
            sl = entry if rmin < entry else cap
        elif mode == "ratchet":
            sl = min(cap, rmin * (1 + X))
        elif mode == "time":
            sl = cap if t < tsec_T else min(cap, premT * 1.3)
        elif mode == "profit_lock":
            sl = entry if rmin <= entry * (1 - Y) else cap
        else:
            sl = cap
        if p >= sl and i > 0:            # can't stop on the entry tick
            return p, True, final
    return final, False, final


POLICIES = [
    ("BASELINE", dict(mode="base")),
    ("TRAIL_BE", dict(mode="trail_be")),
    ("RATCHET_0.20", dict(mode="ratchet", X=0.20)),
    ("RATCHET_0.30", dict(mode="ratchet", X=0.30)),
    ("RATCHET_0.40", dict(mode="ratchet", X=0.40)),
    ("TIME_11:15", dict(mode="time", T="11:15")),
    ("TIME_12:00", dict(mode="time", T="12:00")),
    ("TIME_13:00", dict(mode="time", T="13:00")),
    ("PLOCK_40", dict(mode="profit_lock", Y=0.40)),
    ("PLOCK_50", dict(mode="profit_lock", Y=0.50)),
    ("PLOCK_60", dict(mode="profit_lock", Y=0.60)),
]


def day_pnl(D, params, slip):
    res = {}
    tot_gross = 0.0; tot_cost = 0.0; premature = 0.0; saved = 0.0; stops = 0
    for legname in ("ce", "pe"):
        times, prems = D[legname]
        entry = prems[0]
        exitp, stopped, final = sim_leg(times, prems, **params)
        tot_gross += (entry - exitp) * LOT
        tot_cost += (entry + exitp) * slip * LOT
        if stopped:
            stops += 1
            if final < exitp:            # would have decayed further -> stopping was premature
                premature += (exitp - final) * LOT
            else:                        # premium ended higher -> stop saved a give-back
                saved += (final - exitp) * LOT
    return tot_gross - tot_cost, stops, premature, saved


rows = []
n = 0
for d in days:
    D = load(d)
    if D is None:
        continue
    row = {"day": d, "dte": D["dte"]}
    ok = False
    for name, params in POLICIES:
        net, stops, prem, saved = day_pnl(D, params, SLIP)
        row[name] = round(net, 1)
        row[name + "_stops"] = stops
        if name != "BASELINE":
            row[name + "_prem"] = round(prem, 1)
            row[name + "_saved"] = round(saved, 1)
        ok = True
    if ok:
        rows.append(row); n += 1
        if n % 10 == 0:
            print("...%d days" % n); sys.stdout.flush()

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "per_day.csv"), index=False)
print("\n=== SL-TIGHTENING SUMMARY (net 0.15%%/leg, per 1 lot=65, n=%d days | DTE0-1 %d, DTE2+ %d) ===" % (
    len(R), (R.dte <= 1).sum(), (R.dte >= 2).sum()))
print("  %-14s meanNet  vsBASE  medNet  win%%  worst   avgStops | premature  saved(net) | DTE0-1  DTE2+" % "policy")
base_mean = R["BASELINE"].mean()
summ = []
for name, _ in POLICIES:
    col = R[name].dropna()
    stops = R[name + "_stops"].mean()
    prem = R[name + "_prem"].sum() if name + "_prem" in R else 0.0
    saved = R[name + "_saved"].sum() if name + "_saved" in R else 0.0
    d01 = R.loc[R.dte <= 1, name].mean(); d2 = R.loc[R.dte >= 2, name].mean()
    summ.append(dict(policy=name, meanNet=col.mean(), vsBASE=col.mean() - base_mean, medNet=col.median(),
                     winpct=100 * (col > 0).mean(), worst=col.min(), avgStops=stops,
                     premature_total=prem, saved_total=saved, netProtection=saved - prem, dte01=d01, dte2=d2))
    print("  %-14s %+6.0f  %+6.0f  %+6.0f  %3.0f%%  %+6.0f   %4.2f    | %8.0f  %+9.0f | %+6.0f  %+6.0f" % (
        name, col.mean(), col.mean() - base_mean, col.median(), 100 * (col > 0).mean(), col.min(), stops,
        prem, saved - prem, d01, d2))
pd.DataFrame(summ).to_csv(os.path.join(OUT, "policy_summary.csv"), index=False)

print("\n=== cost sensitivity (mean net/lot) ===")
print("  %-14s 0.10%%   0.15%%   0.20%%" % "policy")
for name, params in POLICIES:
    v = []
    for slp in (0.001, 0.0015, 0.002):
        xs = [day_pnl(load(d), params, slp)[0] for d in [r["day"] for r in rows]]
        v.append(np.mean([x for x in xs if x is not None]))
    print("  %-14s %+5.0f   %+5.0f   %+5.0f" % (name, v[0], v[1], v[2]))
print("\nDONE")
