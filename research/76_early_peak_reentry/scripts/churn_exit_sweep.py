"""research/76 — 09:16 ATM straddle: churn / exit-policy sweep, reconstructed from options_data.db.
Simulates the short straddle under HOLD / SL30 / move-stop bands / lull-hold / no-reenter / exit-15:00,
per day, net of 0.15%/leg-txn cost. Read-only. Per 1 lot = 65. Writes results incrementally.
Verdict per QUANT_RESEARCH_PLAYBOOK (gross AND net + cost sensitivity + per-DTE)."""
import sqlite3, datetime as dt, csv, os, sys
import numpy as np, pandas as pd

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = "/home/arun/quantifyd/research/76_early_peak_reentry/results"
os.makedirs(OUT, exist_ok=True)
LOT = 65
SLIP = 0.0015          # 0.15% per leg per transaction (default)
CAP = 5                # max re-centers/day
con = sqlite3.connect(DB)


def tsec(hhmmss):
    h, m = int(hhmmss[11:13]), int(hhmmss[14:16]); s = int(hhmmss[17:19])
    return h * 3600 + m * 60 + s


def hm_sec(hm):
    return int(hm[:2]) * 3600 + int(hm[3:5]) * 60


ENTRY, EXIT = hm_sec("09:16"), hm_sec("15:15")
LULL0, LULL1 = hm_sec("11:15"), hm_sec("13:00")
CKPTS = ["09:31", "09:46", "10:16", "11:15", "12:00", "13:00", "14:00", "15:00", "15:15"]

days = pd.read_sql("SELECT DISTINCT date(snapshot_time) d FROM underlying_spot ORDER BY d", con)["d"].tolist()
DCACHE = {}


def getD(d):
    if d not in DCACHE:
        try:
            DCACHE[d] = load_day(d)
        except Exception:
            DCACHE[d] = None
    return DCACHE[d]


def load_day(d):
    sp = pd.read_sql(f"SELECT snapshot_time t, spot_price s FROM underlying_spot WHERE date(snapshot_time)='{d}' "
                     f"AND time(snapshot_time) BETWEEN '09:15:00' AND '15:16:00' AND spot_price BETWEEN 20000 AND 28000 "
                     f"ORDER BY t", con)
    if sp.empty:
        return None
    sp["sec"] = sp["t"].apply(tsec)
    exps = pd.read_sql(f"SELECT DISTINCT expiry_date e FROM option_chain WHERE date(snapshot_time)='{d}' AND expiry_date IS NOT NULL", con)["e"].tolist()
    exps = sorted(exps)
    front = next((e for e in exps if e[:10] >= d), exps[0] if exps else None)
    if not front:
        return None
    lo, hi = sp.s.min() - 400, sp.s.max() + 400
    ch = pd.read_sql(f"SELECT snapshot_time t, CAST(ROUND(strike) AS INT) k, instrument_type it, ltp FROM option_chain "
                     f"WHERE date(snapshot_time)='{d}' AND expiry_date='{front}' AND instrument_type IN ('CE','PE') "
                     f"AND ltp>0 AND strike BETWEEN {lo} AND {hi} AND time(snapshot_time) BETWEEN '09:15:00' AND '15:16:00'", con)
    if ch.empty:
        return None
    ch["sec"] = ch["t"].apply(tsec)
    prem = {}
    for (k, it), g in ch.groupby(["k", "it"]):
        g = g.sort_values("sec")
        prem[(k, it)] = (g["sec"].values, g["ltp"].values)
    dte = (dt.date.fromisoformat(front[:10]) - dt.date.fromisoformat(d)).days
    return dict(spot_sec=sp["sec"].values, spot_v=sp["s"].values, prem=prem, dte=dte)


def spot_at(D, sec):
    i = np.searchsorted(D["spot_sec"], sec)
    i = min(max(i, 0), len(D["spot_sec"]) - 1)
    return float(D["spot_v"][i])


def leg(D, k, it, sec, tol=90):
    a = D["prem"].get((k, it))
    if a is None:
        return None
    times, vals = a
    i = np.searchsorted(times, sec)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(times) and abs(times[j] - sec) <= tol:
            if best is None or abs(times[j] - sec) < abs(times[best] - sec):
                best = j
    return float(vals[best]) if best is not None else None


def straddle(D, k, sec):
    ce, pe = leg(D, k, "CE", sec), leg(D, k, "PE", sec)
    return (ce + pe) if (ce is not None and pe is not None) else None


def atm(s):
    return int(round(s / 50) * 50)


def sim_hold(D, slip):
    k = atm(spot_at(D, ENTRY)); cred = straddle(D, k, ENTRY); ex = straddle(D, k, EXIT)
    if cred is None or ex is None:
        return None
    gross = (cred - ex) * LOT
    return dict(net=gross - (cred + ex) * slip * LOT, gross=gross, churn=0)


def sim_move(D, band, slip, reenter=True, lull_hold=False, hard_exit=None, cap=CAP):
    s0 = spot_at(D, ENTRY); k = atm(s0); cred = straddle(D, k, ENTRY)
    if cred is None:
        return None
    entry_spot = s0; realized = 0.0; cost = cred * slip * LOT; churn = 0
    active = True; cur_k = k; cur_cred = cred
    hx = hm_sec(hard_exit) if hard_exit else EXIT
    step = 60
    sec = ENTRY + step
    while sec <= min(hx, EXIT):
        if not active:
            break
        s = spot_at(D, sec)
        crossed = abs(s - entry_spot) >= entry_spot * band / 100.0
        in_lull = LULL0 <= sec <= LULL1
        if crossed and churn < cap and not (lull_hold and in_lull):
            closep = straddle(D, cur_k, sec)
            if closep is not None:
                realized += (cur_cred - closep) * LOT; cost += closep * slip * LOT
                if reenter:
                    nk = atm(s); nc = straddle(D, nk, sec)
                    if nc is not None:
                        cur_k, cur_cred, entry_spot = nk, nc, s; cost += nc * slip * LOT; churn += 1
                    else:
                        active = False
                else:
                    active = False
        sec += step
    if active:
        closep = straddle(D, cur_k, min(hx, EXIT))
        if closep is None:
            return None
        realized += (cur_cred - closep) * LOT; cost += closep * slip * LOT
    return dict(net=realized - cost, gross=realized - (cost - cred * slip * LOT) + 0, churn=churn)  # gross approx below


def sim_sl30(D, slip):
    k = atm(spot_at(D, ENTRY)); ce0 = leg(D, k, "CE", ENTRY); pe0 = leg(D, k, "PE", ENTRY)
    if ce0 is None or pe0 is None:
        return None
    legs = {"CE": [ce0, None], "PE": [pe0, None]}  # [entry, exit]
    cost = (ce0 + pe0) * slip * LOT
    for it in ("CE", "PE"):
        e = legs[it][0]
        sec = ENTRY + 300
        while sec <= EXIT:
            p = leg(D, k, it, sec)
            if p is not None and p >= 1.3 * e:
                legs[it][1] = p; break
            sec += 300
        if legs[it][1] is None:
            legs[it][1] = leg(D, k, it, EXIT)
    if legs["CE"][1] is None or legs["PE"][1] is None:
        return None
    gross = sum((legs[it][0] - legs[it][1]) * LOT for it in ("CE", "PE"))
    cost += sum(legs[it][1] for it in ("CE", "PE")) * slip * LOT
    return dict(net=gross - (cost - (ce0 + pe0) * slip * LOT), gross=gross, churn=0)


POLICIES = [
    ("HOLD", lambda D, s: sim_hold(D, s)),
    ("SL30_NORE", lambda D, s: sim_sl30(D, s)),
    ("MOVE_0.3", lambda D, s: sim_move(D, 0.3, s)),
    ("MOVE_0.4", lambda D, s: sim_move(D, 0.4, s)),
    ("MOVE_0.5", lambda D, s: sim_move(D, 0.5, s)),
    ("MOVE_0.6", lambda D, s: sim_move(D, 0.6, s)),
    ("MOVE_0.8", lambda D, s: sim_move(D, 0.8, s)),
    ("MOVE_1.0", lambda D, s: sim_move(D, 1.0, s)),
    ("MOVE_0.4_LULLHOLD", lambda D, s: sim_move(D, 0.4, s, lull_hold=True)),
    ("MOVE_0.4_NOREENTER", lambda D, s: sim_move(D, 0.4, s, reenter=False)),
    ("MOVE_0.4_EXIT1500", lambda D, s: sim_move(D, 0.4, s, hard_exit="15:00")),
]

perday = []
hold_path = []   # late-peak recheck on fuller sample
n_ok = 0
for d in days:
    D = getD(d)
    if D is None:
        continue
    row = {"day": d, "dte": D["dte"]}
    got = False
    for name, fn in POLICIES:
        r = fn(D, SLIP)
        if r is not None:
            row[name] = round(r["net"], 1)
            row[name + "_churn"] = r.get("churn", 0)
            got = True
    if not got:
        continue
    # HOLD checkpoint path
    k0 = atm(spot_at(D, ENTRY)); cred0 = straddle(D, k0, ENTRY)
    if cred0 is not None:
        pth = {"day": d, "dte": D["dte"]}
        for cp in CKPTS:
            p = straddle(D, k0, hm_sec(cp) if False else (int(cp[:2]) * 3600 + int(cp[3:5]) * 60))
            pth[cp] = round((cred0 - p) * LOT, 1) if p is not None else None
        hold_path.append(pth)
    perday.append(row); n_ok += 1
    if n_ok % 10 == 0:
        print("...%d days done" % n_ok); sys.stdout.flush()

PD = pd.DataFrame(perday)
PD.to_csv(os.path.join(OUT, "per_day.csv"), index=False)
pd.DataFrame(hold_path).to_csv(os.path.join(OUT, "hold_path.csv"), index=False)
print("\n=== POLICY SUMMARY (net of %.2f%%/leg, per 1 lot=65, n=%d days) ===" % (SLIP * 100, len(PD)))
print("  %-22s  meanNet  medNet  win%%  worst   std   avgChurn | DTE0-1  DTE2+" % "policy")
names = [p[0] for p in POLICIES]
summ = []
for nm in names:
    if nm not in PD:
        continue
    col = PD[nm].dropna()
    ch = PD.get(nm + "_churn", pd.Series([0] * len(PD))).mean()
    d01 = PD.loc[PD.dte <= 1, nm].mean(); d2 = PD.loc[PD.dte >= 2, nm].mean()
    summ.append(dict(policy=nm, meanNet=col.mean(), medNet=col.median(), winpct=100 * (col > 0).mean(),
                     worst=col.min(), std=col.std(), avgChurn=ch, dte01=d01, dte2=d2, n=len(col)))
    print("  %-22s  %+6.0f  %+6.0f  %3.0f%%  %+6.0f  %5.0f   %4.1f    | %+6.0f  %+6.0f" % (
        nm, col.mean(), col.median(), 100 * (col > 0).mean(), col.min(), col.std(), ch, d01, d2))
pd.DataFrame(summ).to_csv(os.path.join(OUT, "policy_summary.csv"), index=False)

# cost sensitivity on the top few move policies + hold
print("\n=== cost sensitivity (mean net/lot) ===")
print("  %-22s  0.10%%   0.15%%   0.20%%" % "policy")
for nm, fn in POLICIES:
    vals = []
    for slp in (0.001, 0.0015, 0.002):
        xs = [fn(getD(d), slp) for d in [r["day"] for r in perday]]
        xs = [x["net"] for x in xs if x is not None]
        vals.append(np.mean(xs) if xs else float("nan"))
    print("  %-22s  %+5.0f   %+5.0f   %+5.0f" % (nm, vals[0], vals[1], vals[2]))

# late-peak recheck
HP = pd.DataFrame(hold_path)
print("\n=== HOLD checkpoint path (fuller sample, mean net/lot, n=%d) ===" % len(HP))
for cp in CKPTS:
    c = HP[cp].dropna()
    print("  %-6s  mean %+6.0f  %%grn %3.0f%%" % (cp, c.mean(), 100 * (c > 0).mean()))
print("\nDONE")
