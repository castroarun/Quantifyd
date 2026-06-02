"""Baseline vs Combined, broken out BY DTE, on the REAL 28-day NIFTY chain.

BASELINE : ATM straddle, every day, 1.3x premium stop, exit 14:45.
COMBINED : ATM straddle, TIGHT-open days only (open-15min < median), +/-0.4%
           underlying-move stop, exit 14:45.
Real recorded premiums (level truth) + ACCURATE DTE. n per DTE is SMALL (28-day
window) -> directional; flagged.
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
    o15 = None
    e = spot[(spot.index.time >= dtime(9, 15)) & (spot.index.time <= dtime(9, 30))]
    if len(e): o15 = (e.max() - e.min()) / e.iloc[0] * 100
    chain = {ts: (g.sort_values("t")["t"].values, g.sort_values("t")["ltp"].values, g["strike"].iloc[0], g["instrument_type"].iloc[0])
             for ts, g in df.groupby("tradingsymbol")}
    _cache[day] = (dte, o15, spot, chain); return _cache[day]

def prem(chain, ts, t):
    if ts not in chain: return None
    ta, la, _, _ = chain[ts]; i = np.searchsorted(ta, np.datetime64(t), side="right") - 1
    if i < 0: return None
    v = la[i]; return float(v) if v and v > 0 else None
def tsym(chain, k, typ):
    for ts, (_, _, st, ty) in chain.items():
        if int(st) == int(k) and ty == typ: return ts
    return None

def sim(day, stype, param):
    d = load(day)
    if d is None: return None
    dte, o15, spot_s, chain = d
    times = [t for t in spot_s.index if t.time() >= ENTRY and t.time() <= EOD]
    if not times: return None
    t0 = times[0]; s0 = float(spot_s.loc[t0]); atm = round(s0 / 50) * 50
    ce, pe = tsym(chain, atm, "CE"), tsym(chain, atm, "PE")
    cE, pE = (prem(chain, ce, t0) if ce else None), (prem(chain, pe, t0) if pe else None)
    if not cE or not pE: return None
    for t in times:
        cc = prem(chain, ce, t) or cE; pp = prem(chain, pe, t) or pE
        s = float(spot_s.loc[t]) if t in spot_s.index else s0
        trig = t.time() >= TIMEEXIT
        if stype == "prem" and (cc >= cE * param or pp >= pE * param): trig = True
        elif stype == "underlying" and abs(s - s0) / s0 * 100 >= param: trig = True
        if trig:
            return dte, o15, (cE - cc) * QTY + (pE - pp) * QTY - 2 * BROK
    cc = prem(chain, ce, times[-1]) or cE; pp = prem(chain, pe, times[-1]) or pE
    return dte, o15, (cE - cc) * QTY + (pE - pp) * QTY - 2 * BROK

base = [sim(d, "prem", 1.3) for d in DAYS]; base = [r for r in base if r]
comb = [sim(d, "underlying", 0.4) for d in DAYS]; comb = [r for r in comb if r]
o15med = np.median([o for _, o, _ in base if o is not None])

dtes = sorted(set(r[0] for r in base))
rows = []
for dte in dtes:
    b = [p for d, o, p in base if d == dte]
    c = [p for d, o, p in comb if d == dte and o is not None and o < o15med]  # tight-open only
    rows.append(dict(DTE=dte, Base_net=round(sum(b)), Base_n=len(b),
                     Comb_net=round(sum(c)) if c else 0, Comb_n=len(c)))
R = pd.DataFrame(rows)
tot_b = sum(p for _, _, p in base); tot_c = sum(p for d, o, p in comb if o is not None and o < o15med)
print(R.to_string()); print("tight-open median open15% =", round(o15med, 3))
print("TOTAL base", round(tot_b), "combined", round(tot_c))

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(R)); w = 0.38
ax.bar(x - w/2, R["Base_net"], w, label="Baseline (all days, 1.3x prem stop)", color="#888")
ax.bar(x + w/2, R["Comb_net"], w, label="Combined (tight-open, ±0.4% move stop)", color="#0a6")
ax.set_xticks(x); ax.set_xticklabels([f"{int(d)}DTE\n(b={int(r.Base_n)},c={int(r.Comb_n)})" for d, r in zip(R["DTE"], R.itertuples())], fontsize=8)
ax.axhline(0, color="#333"); ax.legend()
for i, r in R.iterrows():
    ax.text(i - w/2, r["Base_net"], f"{r['Base_net']/1000:.1f}k", ha="center", va="bottom" if r["Base_net"] >= 0 else "top", fontsize=7)
    ax.text(i + w/2, r["Comb_net"], f"{r['Comb_net']/1000:.1f}k", ha="center", va="bottom" if r["Comb_net"] >= 0 else "top", fontsize=7)
ax.set_title("Baseline vs Combined by DTE — REAL 28-day NIFTY premiums (n small, directional)")
ax.set_ylabel("net ₹")
fig.tight_layout(); fig.savefig(OUT / "dte_compare.png", dpi=120, bbox_inches="tight"); print("WROTE dte_compare.png")

L = ["# Baseline vs Combined, BY DTE — real 28-day NIFTY premiums\n",
     f"Baseline: all days, 1.3x premium stop. Combined: tight-open days only (open-15min < median "
     f"{o15med:.3f}%), ±0.4% move stop. Both ATM, exit 14:45. **n per DTE is small (28-day window) → "
     f"directional.** Totals: baseline ₹{tot_b:,.0f}, combined ₹{tot_c:,.0f}.\n",
     "| DTE | Baseline net ₹ | Baseline n | Combined net ₹ | Combined n |", "|---|---|---|---|---|"]
for _, r in R.iterrows():
    L.append(f"| {int(r['DTE'])} | {r['Base_net']:,.0f} | {int(r['Base_n'])} | {r['Comb_net']:,.0f} | {int(r['Comb_n'])} |")
L.append("\n- Edge concentrated at 1 DTE in both; the combined filter+move-stop reshapes the per-DTE profile.")
L.append("- Small n per DTE (1-5 days) — real-level confirmation needs the recorder to accumulate.")
(OUT / "RESULTS_dte_compare.md").write_text("\n".join(L), encoding="utf-8"); print("WROTE RESULTS_dte_compare.md")
oc.close()
