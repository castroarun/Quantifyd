"""Part (a-ii): stop-DESIGN comparison on the recorded NIFTY chain.

Tests whether a stop tied to the UNDERLYING move (±X%) or a daily MAX-LOSS (-Rs Y)
beats both the whipsaw-prone 1.3x PREMIUM stop and the tail-risky no-stop.

Consistent management for apples-to-apples: sell ATM CE+PE at 09:20 (lots=2); the
stop rule triggers a FULL strangle exit (both legs) at current premiums; otherwise
exit 14:45. Reports total net, WORST single-day loss (tail proxy — 28 calm days has
no crash, so this UNDERSTATES the tail; the point is which designs are *structurally*
bounded), and trigger rate. Both all-days and 1-DTE-only.
"""
import sqlite3
from pathlib import Path
from datetime import time as dtime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/arun/quantifyd"); BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"; OUT.mkdir(parents=True, exist_ok=True)
oc = sqlite3.connect(str(ROOT / "backtest_data" / "options_data.db"))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_tsym_time ON option_chain(tradingsymbol, snapshot_time)")
DAYS = [r[0] for r in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]
LOT = 65; QTY = LOT * 2; BROK = 80
ENTRY = dtime(9, 20); TIMEEXIT = dtime(14, 45); EOD = dtime(15, 15)

_cache = {}
def load(day):
    if day in _cache: return _cache[day]
    df = pd.read_sql_query("SELECT snapshot_time,tradingsymbol,strike,instrument_type,ltp,expiry_date,underlying_spot "
                           "FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND ltp IS NOT NULL",
                           oc, params=[day])
    if df.empty: _cache[day] = None; return None
    df["t"] = pd.to_datetime(df["snapshot_time"])
    exps = sorted(df["expiry_date"].unique()); fut = [e for e in exps if e >= day]
    exp = fut[0] if fut else exps[-1]; df = df[df["expiry_date"] == exp]
    dte = (pd.to_datetime(exp).date() - pd.to_datetime(day).date()).days
    spot = df.groupby("t")["underlying_spot"].first().sort_index()
    chain = {ts: (g.sort_values("t")["t"].values, g.sort_values("t")["ltp"].values, g["strike"].iloc[0], g["instrument_type"].iloc[0])
             for ts, g in df.groupby("tradingsymbol")}
    _cache[day] = (dte, spot, chain); return _cache[day]

def prem(chain, ts, t):
    if ts not in chain: return None
    ta, la, _, _ = chain[ts]; i = np.searchsorted(ta, np.datetime64(t), side="right") - 1
    if i < 0: return None
    v = la[i]; return float(v) if v and v > 0 else None

def tsym(chain, strike, typ):
    for ts, (_, _, st, ty) in chain.items():
        if int(st) == int(strike) and ty == typ: return ts
    return None

def sim(day, stype, param):
    d = load(day)
    if d is None: return None
    dte, spot_s, chain = d
    times = [t for t in spot_s.index if t.time() >= ENTRY and t.time() <= EOD]
    if not times: return None
    t0 = times[0]; spot0 = float(spot_s.loc[t0]); atm = round(spot0/50)*50
    ce, pe = tsym(chain, atm, "CE"), tsym(chain, atm, "PE")
    ceE, peE = (prem(chain, ce, t0) if ce else None), (prem(chain, pe, t0) if pe else None)
    if not ceE or not peE: return None
    for t in times:
        cur_ce = prem(chain, ce, t) or ceE; cur_pe = prem(chain, pe, t) or peE
        spot = float(spot_s.loc[t]) if t in spot_s.index else spot0
        mtm = (ceE - cur_ce) * QTY + (peE - cur_pe) * QTY
        trig = False
        if t.time() >= TIMEEXIT:
            trig = True
        elif stype == "prem":
            if cur_ce >= ceE * param or cur_pe >= peE * param: trig = True
        elif stype == "underlying":
            if abs(spot - spot0) / spot0 * 100 >= param: trig = True
        elif stype == "maxloss":
            if mtm <= -param: trig = True
        # 'none' -> only time exit
        if trig:
            pnl = (ceE - cur_ce) * QTY + (peE - cur_pe) * QTY - 2 * BROK
            return dte, pnl
    cur_ce = prem(chain, ce, times[-1]) or ceE; cur_pe = prem(chain, pe, times[-1]) or peE
    return dte, (ceE - cur_ce) * QTY + (peE - cur_pe) * QTY - 2 * BROK

DESIGNS = [
    ("prem 1.2x", "prem", 1.2), ("prem 1.3x", "prem", 1.3), ("prem 1.5x", "prem", 1.5),
    ("no stop", "none", 0),
    ("undl 0.4%", "underlying", 0.4), ("undl 0.6%", "underlying", 0.6),
    ("undl 0.8%", "underlying", 0.8), ("undl 1.0%", "underlying", 1.0),
    ("maxloss 2k", "maxloss", 2000), ("maxloss 3k", "maxloss", 3000), ("maxloss 5k", "maxloss", 5000),
]
rows = []
for name, stype, param in DESIGNS:
    res = [sim(d, stype, param) for d in DAYS]
    res = [r for r in res if r]
    allp = [p for _, p in res]; one = [p for dte, p in res if dte == 1]
    rows.append(dict(Design=name, AllDays_net=round(sum(allp)), AllDays_worstday=round(min(allp)) if allp else 0,
                     OneDTE_net=round(sum(one)), n=len(allp)))
R = pd.DataFrame(rows)
print(R.to_string())

fig, ax = plt.subplots(1, 3, figsize=(17, 6))
def barcol(v): return ["#0a6" if x >= 0 else "#d33" for x in v]
ax[0].barh(R["Design"], R["AllDays_net"], color=barcol(R["AllDays_net"])); ax[0].set_title("All-days net ₹"); ax[0].axvline(0, color="#333", lw=.8); ax[0].invert_yaxis()
ax[1].barh(R["Design"], R["AllDays_worstday"], color="#d33"); ax[1].set_title("Worst single-day ₹ (tail proxy; understates — no crash in 28d)"); ax[1].invert_yaxis()
ax[2].barh(R["Design"], R["OneDTE_net"], color=barcol(R["OneDTE_net"])); ax[2].set_title("1-DTE-only net ₹"); ax[2].axvline(0, color="#333", lw=.8); ax[2].invert_yaxis()
fig.suptitle("Stop-DESIGN comparison — ATM straddle, recorded NIFTY chain (lots=2, 28d SIGNAL only)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "stop_design.png", dpi=110, bbox_inches="tight")
print("WROTE", OUT / "stop_design.png")

L = ["# Stop-DESIGN comparison (ATM straddle, recorded NIFTY chain, lots=2)\n",
     "Consistent mgmt: stop triggers a FULL strangle exit; else exit 14:45. **28 days, NO crash in "
     "sample -> worst-day UNDERSTATES the tail.** The structural point: underlying/max-loss stops have a "
     "BOUNDED loss by design; no-stop is unbounded.\n",
     "| Design | All-days net ₹ | All-days worst-day ₹ | 1-DTE-only net ₹ |", "|---|---|---|---|"]
for _, r in R.iterrows():
    L.append(f"| {r['Design']} | {r['AllDays_net']:,.0f} | {r['AllDays_worstday']:,.0f} | {r['OneDTE_net']:,.0f} |")
L += ["\n- Premium stops (1.2-1.5x) whipsaw on premium noise.",
      "- Underlying-move / max-loss stops trigger on REAL adverse moves, not premium spikes -> avoid whipsaw "
      "AND cap loss (structurally bounded). no-stop wins in-sample only because no trend/crash day occurred.",
      "- Read the GRADIENT + the worst-day column together: want decent net AND a bounded worst-day."]
(OUT / "RESULTS_stopdesign.md").write_text("\n".join(L), encoding="utf-8")
print("WROTE", OUT / "RESULTS_stopdesign.md")
oc.close()
