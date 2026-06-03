"""research/54 Stage 5 — intraday RE-ENTRY + directional SKEW (real NIFTY chain, read-only).

Base = best so far: 1-DTE, ~100pt-OTM strangle (2 strikes), 09:20 entry, ±0.4% move stop, exit 14:45.
  AXIS A re-entry: after a move-stop, re-sell a fresh strangle at the stop time (0 / 1 / 3 max).
                   Does re-selling recover theta after a whipsaw-out, or just re-lose?
  AXIS B skew:     shift both strikes +/-1 in the opening-15min direction (lean with the morning move).
Net ₹80/leg per entry. 29 days => SIGNAL; report 0/1-DTE (the edge) + all-DTE, net + worst-day.
"""
import sqlite3
from pathlib import Path
from datetime import time as dtime
import numpy as np, pandas as pd

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

def open_strangle(chain, spot, off, skew):
    atm = round(spot / 50) * 50 + skew * 50
    legs = []
    for typ, sgn in (("CE", 1), ("PE", -1)):
        ts = tsym(chain, atm + sgn * off * 50, typ); p = prem(chain, ts, OPEN_T) if ts else None
    return atm  # placeholder, real open below

def sim(day, off=2, move_pct=0.4, max_reentry=0, skew_mode=False):
    d = load(day)
    if d is None: return None
    dte, spot_s, chain = d
    times = [t for t in spot_s.index if ENTRY <= t.time() <= EOD]
    if len(times) < 3: return None
    # opening direction for skew: spot at entry vs spot ~15min before (09:15)
    pre = spot_s[spot_s.index.time <= ENTRY]
    skew = 0
    if skew_mode and len(pre) >= 2:
        d0 = pre.iloc[-1] - pre.iloc[0]
        skew = 1 if d0 > 0 else (-1 if d0 < 0 else 0)
    pnl = 0.0; idx = 0; entries = 0
    while idx < len(times) and entries <= max_reentry:
        t0 = times[idx]; spot0 = float(spot_s.loc[t0]); atm = round(spot0 / 50) * 50 + skew * 50
        legs = []
        for typ, sgn in (("CE", 1), ("PE", -1)):
            ts = tsym(chain, atm + sgn * off * 50, typ); p = prem(chain, ts, t0) if ts else None
            if p: legs.append({"ts": ts, "entry": p, "open": True})
        if len(legs) < 2: break
        entries += 1; stopped_at = None
        for j in range(idx, len(times)):
            t = times[j]
            if not any(l["open"] for l in legs): break
            force = t.time() >= TIMEEXIT
            moved = abs(float(spot_s.loc[t]) - spot0) / spot0 * 100 >= move_pct
            for lg in legs:
                if not lg["open"]: continue
                p = prem(chain, lg["ts"], t)
                if p is None: continue
                if force or moved:
                    pnl += (lg["entry"] - p) * QTY - BROK; lg["open"] = False
            if force: stopped_at = None; break
            if moved and not any(l["open"] for l in legs): stopped_at = j; break
        if stopped_at is None: break          # closed at time-exit -> done
        idx = stopped_at + 1                    # re-enter after the move-stop bar
        if times[idx].time() >= TIMEEXIT if idx < len(times) else True: break
    return dict(day=day, dte=dte, pnl=pnl, entries=entries)

OPEN_T = None
VAR = [
    ("base (no re-entry)", dict(max_reentry=0, skew_mode=False)),
    ("re-entry x1", dict(max_reentry=1, skew_mode=False)),
    ("re-entry x3", dict(max_reentry=3, skew_mode=False)),
    ("directional skew", dict(max_reentry=0, skew_mode=True)),
    ("skew + re-entry x3", dict(max_reentry=3, skew_mode=True)),
]
def book(df):
    if df is None or df.empty: return (0, 0, 0, 0)
    a = df["pnl"].values; return (len(a), round(a.sum()), round(a.mean()), round(a.min()))

print("running stage5 re-entry/skew...", flush=True)
res = []
for name, kw in VAR:
    rows = [sim(d, off=2, **kw) for d in DAYS]; rows = [r for r in rows if r]
    res.append((name, pd.DataFrame(rows)))
    print("  %s done" % name, flush=True)

L = ["# research/54 Stage 5 — intraday re-entry + directional skew (real NIFTY chain)\n",
     "Base: 1-DTE focus, ~100pt-OTM (2-strike) strangle, 09:20 entry, ±0.4% move stop, exit 14:45, net ₹80/leg/entry. **29d SIGNAL.**\n",
     "## 0/1-DTE (the edge)", "| variant | n | total ₹ | ₹/day | worst ₹ |", "|---|---|---|---|---|"]
for name, df in res:
    L.append("| %s | %d | %d | %d | %d |" % ((name,) + book(df[df["dte"] <= 1] if not df.empty else None)))
L += ["\n## all-DTE", "| variant | n | total ₹ | ₹/day | worst ₹ |", "|---|---|---|---|---|"]
for name, df in res:
    L.append("| %s | %d | %d | %d | %d |" % ((name,) + book(df)))
L += ["\n## Read",
      "- Re-entry: if re-entry x1/x3 >> base on net WITHOUT a worse worst-day, re-selling recovers theta after a whipsaw-out.",
      "  If it just adds losses/worse tail, the move-stop should stay one-and-done.",
      "- Skew: leaning strikes with the morning move should help only if the open predicts the rest of the day.",
      "- 29d SIGNAL, ~13 obs 0/1-DTE. Gradient only."]
(OUT / "RESULTS_reentry_skew.md").write_text("\n".join(L), encoding="utf-8")
print("DONE", flush=True)
for name, df in res:
    print("  %-22s 0/1DTE=%s all=%s" % (name, book(df[df["dte"] <= 1] if not df.empty else None), book(df)), flush=True)
oc.close()
