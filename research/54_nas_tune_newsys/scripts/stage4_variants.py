"""research/54 Stage 4 — NEW-system variants on the real NIFTY chain (read-only).

Cheap G0 kills around the confirmed winner (1-DTE short ATM straddle + ±0.4% move stop):
  AXIS 1 strike:  ATM straddle vs 1-OTM / 2-OTM strangle (less gamma, lower credit)
  AXIS 2 entry:   09:20 vs 11:00 vs 13:00 (theta vs gamma trade-off on 1-DTE)
All with ±0.4% underlying-move stop, exit 14:45, net ₹80/leg. Report 0/1-DTE (the edge) +
all-DTE, net + worst-day. 29 days => SIGNAL, read the gradient.
"""
import sqlite3
from pathlib import Path
from datetime import time as dtime
import numpy as np, pandas as pd

ROOT = Path("/home/arun/quantifyd"); BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"; OUT.mkdir(parents=True, exist_ok=True)
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; QTY = LOT * 2; BROK = 80
TIMEEXIT = dtime(14, 45); EOD = dtime(15, 15)
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
    _c[day] = (dte, spot, chain); return _c[day]

def prem(chain, ts, t):
    if ts not in chain: return None
    ta, la, _, _ = chain[ts]; i = np.searchsorted(ta, np.datetime64(t), side="right") - 1
    if i < 0: return None
    v = la[i]; return float(v) if v and v > 0 else None

def tsym(chain, strike, typ):
    for ts, (_, _, st, ty) in chain.items():
        if int(st) == int(strike) and ty == typ: return ts
    return None

def sim(day, strike_off=0, entry=dtime(9, 20), move_pct=0.4):
    d = load(day)
    if d is None: return None
    dte, spot_s, chain = d
    times = [t for t in spot_s.index if entry <= t.time() <= EOD]
    if not times: return None
    t0 = times[0]; spot0 = float(spot_s.loc[t0]); atm = round(spot0 / 50) * 50
    legs = []
    for typ, sgn in (("CE", 1), ("PE", -1)):
        k = atm + sgn * strike_off * 50
        ts = tsym(chain, k, typ); p = prem(chain, ts, t0) if ts else None
        if p: legs.append({"ts": ts, "entry": p, "open": True})
    if len(legs) < 2: return None
    pnl = 0.0
    for t in times:
        if not any(l["open"] for l in legs): break
        force = t.time() >= TIMEEXIT
        moved = abs(float(spot_s.loc[t]) - spot0) / spot0 * 100 >= move_pct
        for lg in legs:
            if not lg["open"]: continue
            p = prem(chain, lg["ts"], t)
            if p is None: continue
            if force or moved:
                pnl += (lg["entry"] - p) * QTY - BROK; lg["open"] = False
    lastt = times[-1]
    for lg in legs:
        if lg["open"]:
            p = prem(chain, lg["ts"], lastt) or lg["entry"]; pnl += (lg["entry"] - p) * QTY - BROK
    return dict(day=day, dte=dte, pnl=pnl)

def run(name, **kw):
    rows = [sim(d, **kw) for d in DAYS]; rows = [r for r in rows if r]
    df = pd.DataFrame(rows)
    return name, df

def book(df):
    if df is None or df.empty: return (0, 0, 0, 0)
    a = df["pnl"].values; return (len(a), round(a.sum()), round(a.mean()), round(a.min()))

VARIANTS = [
    ("ATM straddle 09:20", dict(strike_off=0, entry=dtime(9, 20))),
    ("1-OTM strangle 09:20", dict(strike_off=1, entry=dtime(9, 20))),
    ("2-OTM strangle 09:20", dict(strike_off=2, entry=dtime(9, 20))),
    ("ATM straddle 11:00", dict(strike_off=0, entry=dtime(11, 0))),
    ("ATM straddle 13:00", dict(strike_off=0, entry=dtime(13, 0))),
]
print("running stage4 variants...", flush=True)
res = [run(n, **kw) for n, kw in VARIANTS]
L = ["# research/54 Stage 4 — strike × entry-time variants (real NIFTY chain, ±0.4% move stop)\n",
     "All ±0.4% underlying-move stop, exit 14:45, net ₹80/leg. **29 days => SIGNAL.**\n",
     "## 0/1-DTE (the edge)", "| variant | n | total ₹ | ₹/day | worst ₹ |", "|---|---|---|---|---|"]
for name, df in res:
    L.append("| %s | %d | %d | %d | %d |" % ((name,) + book(df[df["dte"] <= 1] if not df.empty else None)))
L += ["\n## all-DTE", "| variant | n | total ₹ | ₹/day | worst ₹ |", "|---|---|---|---|---|"]
for name, df in res:
    L.append("| %s | %d | %d | %d | %d |" % ((name,) + book(df)))
L += ["\n## Read",
      "- Strike: does moving OTM (less gamma, lower credit) help or just cut the credit? Read 0/1-DTE ₹/day + worst.",
      "- Entry time: later entry = less theta captured but skips the morning move. 1-DTE theta is fast → earlier usually better.",
      "- 29d SIGNAL, ~13 obs 0/1-DTE. Gradient only; confirm as recorder grows."]
(OUT / "RESULTS_variants.md").write_text("\n".join(L), encoding="utf-8")
print("DONE", flush=True)
for name, df in res:
    print("  %-22s 0/1DTE=%s all=%s" % (name, book(df[df["dte"] <= 1] if not df.empty else None), book(df)), flush=True)
oc.close()
