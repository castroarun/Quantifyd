"""research/54 Stage 3 — DEFINED-RISK structures vs naked straddle (real NIFTY chain).

Live pain = tail losses + premium-stop whipsaw. An iron-fly (short ATM straddle + long OTM
wings) caps the tail with a KNOWN max loss and needs no stop. Question: does it keep enough
of the 1-DTE edge while bounding the tail better than naked-straddle + ±0.4% move stop?

Engine reused from scan.py. Enter 09:20, exit 14:45. Net ₹80/leg (incl wing legs). Structures:
  naked_nostop, naked_move0.4, fly_W (long wings W pts OTM, no stop), fly_W_move0.4.
28-29 days => SIGNAL. Report all-DTE + 1-DTE (the edge), with worst-day (tail).
"""
import sqlite3
from pathlib import Path
from datetime import time as dtime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/arun/quantifyd"); BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"; OUT.mkdir(parents=True, exist_ok=True)
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; QTY = LOT * 2; BROK = 80
ENTRY = dtime(9, 20); TIMEEXIT = dtime(14, 45); EOD = dtime(15, 15)
oc = sqlite3.connect(str(OPT))
DAYS = [r[0] for r in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]

_c = {}
def load(day):
    if day in _c: return _c[day]
    df = pd.read_sql_query("SELECT snapshot_time,tradingsymbol,strike,instrument_type,ltp,expiry_date,underlying_spot "
                           "FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND ltp IS NOT NULL", oc, params=[day])
    if df.empty: _c[day] = None; return None
    df["t"] = pd.to_datetime(df["snapshot_time"])
    exps = sorted(df["expiry_date"].unique()); fut = [e for e in exps if e >= day]; exp = fut[0] if fut else exps[-1]
    df = df[df["expiry_date"] == exp]
    dte = (pd.to_datetime(exp).date() - pd.to_datetime(day).date()).days
    spot = df.groupby("t")["underlying_spot"].first().sort_index()
    chain = {ts: (g.sort_values("t")["t"].values, g.sort_values("t")["ltp"].values, g["strike"].iloc[0], g["instrument_type"].iloc[0])
             for ts, g in df.groupby("tradingsymbol")}
    strikes = {"CE": sorted({int(st) for _, (_, _, st, ty) in chain.items() if ty == "CE"}),
               "PE": sorted({int(st) for _, (_, _, st, ty) in chain.items() if ty == "PE"})}
    _c[day] = (dte, spot, chain, strikes); return _c[day]

def prem(chain, ts, t):
    if ts not in chain: return None
    ta, la, _, _ = chain[ts]; i = np.searchsorted(ta, np.datetime64(t), side="right") - 1
    if i < 0: return None
    v = la[i]; return float(v) if v and v > 0 else None

def tsym(chain, strike, typ):
    for ts, (_, _, st, ty) in chain.items():
        if int(st) == int(strike) and ty == typ: return ts
    return None

def nearest(strikes, target, typ):
    arr = strikes[typ]
    return min(arr, key=lambda k: abs(k - target)) if arr else None

def sim(day, wing=0, stop_mode="none", move_pct=0.4):
    d = load(day)
    if d is None: return None
    dte, spot_s, chain, strikes = d
    times = [t for t in spot_s.index if ENTRY <= t.time() <= EOD]
    if not times: return None
    t0 = times[0]; spot0 = float(spot_s.loc[t0]); atm = round(spot0 / 50) * 50
    legs = []
    for typ, sgn in (("CE", 1), ("PE", -1)):
        ts = tsym(chain, atm, typ); p = prem(chain, ts, t0) if ts else None
        if p: legs.append({"ts": ts, "entry": p, "sl": p * 1.3, "short": True, "open": True})
        if wing:
            wk = nearest(strikes, atm + sgn * wing, typ); wts = tsym(chain, wk, typ); wp = prem(chain, wts, t0) if wts else None
            if wp: legs.append({"ts": wts, "entry": wp, "short": False, "open": True})
    if sum(1 for l in legs if l["short"]) < 2: return None
    if wing and sum(1 for l in legs if not l["short"]) < 2: return None  # need both wings if fly
    pnl = 0.0
    for t in times:
        if not any(l["open"] for l in legs): break
        force = t.time() >= TIMEEXIT
        moved = abs(float(spot_s.loc[t]) - spot0) / spot0 * 100 >= move_pct
        for lg in legs:
            if not lg["open"]: continue
            p = prem(chain, lg["ts"], t)
            if p is None: continue
            hit = force or (stop_mode == "move" and moved)
            if hit:
                sign = 1 if lg["short"] else -1
                pnl += sign * (lg["entry"] - p) * QTY - BROK; lg["open"] = False
    lastt = times[-1]
    for lg in legs:
        if lg["open"]:
            p = prem(chain, lg["ts"], lastt) or lg["entry"]; sign = 1 if lg["short"] else -1
            pnl += sign * (lg["entry"] - p) * QTY - BROK
    return dict(day=day, dte=dte, pnl=pnl)

STRUCTS = [("naked_nostop", dict(wing=0, stop_mode="none")),
           ("naked_move0.4", dict(wing=0, stop_mode="move")),
           ("fly_300", dict(wing=300, stop_mode="none")),
           ("fly_400", dict(wing=400, stop_mode="none")),
           ("fly_500", dict(wing=500, stop_mode="none")),
           ("fly_400_move0.4", dict(wing=400, stop_mode="move"))]

print("running stage3 defined-risk structures over %d days..." % len(DAYS), flush=True)
res = {}
for name, kw in STRUCTS:
    rows = [sim(day, **kw) for day in DAYS]; rows = [r for r in rows if r]
    res[name] = pd.DataFrame(rows)
    print("  %s done (%d days)" % (name, len(rows)), flush=True)

def book(df):
    if df is None or df.empty: return (0, 0, 0, 0, 0)
    a = df["pnl"].values; return (len(a), round(a.sum()), round(a.mean()), round(a.min()), round(np.sort(a)[:3].mean()))

L = ["# research/54 Stage 3 — defined-risk structures vs naked straddle (real NIFTY chain)\n",
     "Short ATM straddle +/- long OTM wings. Enter 09:20, exit 14:45, net ₹80/leg. "
     "**28-29 days => SIGNAL.** Wings cap the tail with a KNOWN max loss, no stop needed.\n",
     "## ALL-DTE (n / total ₹ / ₹-day / worst-day / worst3-avg)",
     "| structure | n | total ₹ | ₹/day | worst ₹ | worst3 ₹ |", "|---|---|---|---|---|---|"]
for name, _ in STRUCTS:
    L.append("| %s | %d | %d | %d | %d | %d |" % ((name,) + book(res[name])))
L += ["\n## 1-DTE ONLY (the edge — Mondays, n~7)",
      "| structure | n | total ₹ | ₹/day | worst ₹ | worst3 ₹ |", "|---|---|---|---|---|---|"]
for name, _ in STRUCTS:
    d1 = res[name][res[name]["dte"] <= 1] if not res[name].empty else None
    L.append("| %s | %d | %d | %d | %d | %d |" % ((name,) + book(d1)))
L += ["\n## Read",
      "- Compare the naked-straddle+move-stop (current best) vs iron-flies on **worst-day / worst3** (tail) AND ₹/day.",
      "- A fly that keeps most of the 1-DTE ₹/day while cutting the worst-day = a better LIVE system (bounded, no whipsaw).",
      "- 28-29d SIGNAL, single regime; wing liquidity/slippage not modelled. Confirm as recorder grows."]
(OUT / "RESULTS_defined_risk.md").write_text("\n".join(L), encoding="utf-8")
print("DONE", flush=True)
for name, _ in STRUCTS:
    print("  %-16s all=%s  1DTE=%s" % (name, book(res[name]), book(res[name][res[name]['dte']<=1] if not res[name].empty else None)), flush=True)
oc.close()
