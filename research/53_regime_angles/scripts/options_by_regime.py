"""Layer 2: how the ATM straddle ACTUALLY did on the 28-day real chain, sliced by
gap bucket and CPR width (and DTE). Prior-day H/L/C + gap computed from the chain's
own underlying_spot (market_data.db NIFTY ends 2026-03, no overlap with the options
window). n per bucket is tiny (28-day window) -> directional ONLY.
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
_c = {}
def load(day):
    if day in _c: return _c[day]
    df = pd.read_sql_query("SELECT snapshot_time,tradingsymbol,strike,instrument_type,ltp,expiry_date,underlying_spot "
                           "FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND ltp IS NOT NULL",
                           oc, params=[day])
    if df.empty: _c[day] = None; return None
    df["t"] = pd.to_datetime(df["snapshot_time"])
    exps = sorted(df["expiry_date"].unique()); fut = [e for e in exps if e >= day]; exp = fut[0] if fut else exps[-1]
    df = df[df["expiry_date"] == exp]; dte = (pd.to_datetime(exp).date() - pd.to_datetime(day).date()).days
    spot = df.groupby("t")["underlying_spot"].first().sort_index()
    chain = {ts: (g.sort_values("t")["t"].values, g.sort_values("t")["ltp"].values, g["strike"].iloc[0], g["instrument_type"].iloc[0])
             for ts, g in df.groupby("tradingsymbol")}
    daily = dict(o=float(spot.iloc[0]), c=float(spot.iloc[-1]), h=float(spot.max()), l=float(spot.min()))
    _c[day] = (dte, spot, chain, daily); return _c[day]
def prem(chain, ts, t):
    if ts not in chain: return None
    ta, la, _, _ = chain[ts]; i = np.searchsorted(ta, np.datetime64(t), side="right") - 1
    if i < 0: return None
    v = la[i]; return float(v) if v and v > 0 else None
def tsym(chain, k, typ):
    for ts, (_, _, st, ty) in chain.items():
        if int(st) == int(k) and ty == typ: return ts
    return None
def sim(day):
    d = load(day)
    if d is None: return None
    dte, spot_s, chain, _ = d
    times = [t for t in spot_s.index if t.time() >= ENTRY and t.time() <= EOD]
    if not times: return None
    t0 = times[0]; s0 = float(spot_s.loc[t0]); atm = round(s0 / 50) * 50
    ce, pe = tsym(chain, atm, "CE"), tsym(chain, atm, "PE")
    cE, pE = (prem(chain, ce, t0) if ce else None), (prem(chain, pe, t0) if pe else None)
    if not cE or not pE: return None
    for t in times:
        cc = prem(chain, ce, t) or cE; pp = prem(chain, pe, t) or pE
        if t.time() >= TIMEEXIT or cc >= cE * 1.3 or pp >= pE * 1.3:
            return (cE - cc) * QTY + (pE - pp) * QTY - 2 * BROK
    cc = prem(chain, ce, times[-1]) or cE; pp = prem(chain, pe, times[-1]) or pE
    return (cE - cc) * QTY + (pE - pp) * QTY - 2 * BROK

rows = []
prev = None
for day in DAYS:
    d = load(day)
    if d is None: continue
    dte, _, _, dl = d
    pnl = sim(day)
    if prev and pnl is not None:
        gap = (dl["o"] - prev["c"]) / prev["c"] * 100
        P = (prev["h"] + prev["l"] + prev["c"]) / 3; BC = (prev["h"] + prev["l"]) / 2; TC = 2 * P - BC
        cpr = abs(TC - BC) / prev["c"] * 100
        rows.append(dict(day=day, dte=dte, gap=gap, cpr=cpr, pnl=pnl))
    prev = dl
R = pd.DataFrame(rows)
R["gap_b"] = pd.cut(R["gap"], [-99, -0.3, 0.3, 99], labels=["gap dn", "flat", "gap up"])
R["cpr_b"] = pd.qcut(R["cpr"], 2, labels=["narrow", "wide"])
by_gap = R.groupby("gap_b", observed=True).agg(net=("pnl", "sum"), n=("pnl", "size"), avg=("pnl", "mean"))
by_cpr = R.groupby("cpr_b", observed=True).agg(net=("pnl", "sum"), n=("pnl", "size"), avg=("pnl", "mean"))
# gap x DTE pivot
piv = R.pivot_table(index="gap_b", columns="dte", values="pnl", aggfunc="sum", observed=True)
print(R.to_string()); print(by_gap); print(by_cpr)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].bar(by_gap.index.astype(str), by_gap["net"], color=["#d33" if v < 0 else "#0a6" for v in by_gap["net"]])
ax[0].axhline(0, color="#333"); ax[0].set_title("28d straddle net ₹ by gap");
for i, (idx, r) in enumerate(by_gap.iterrows()): ax[0].text(i, r["net"], f"n={int(r['n'])}", ha="center", va="bottom")
ax[1].bar(by_cpr.index.astype(str), by_cpr["net"], color=["#d33" if v < 0 else "#0a6" for v in by_cpr["net"]])
ax[1].axhline(0, color="#333"); ax[1].set_title("28d straddle net ₹ by CPR width")
for i, (idx, r) in enumerate(by_cpr.iterrows()): ax[1].text(i, r["net"], f"n={int(r['n'])}", ha="center", va="bottom")
fig.suptitle("28-day REAL straddle P&L by regime (tiny n — directional only)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "options_by_regime.png", dpi=120, bbox_inches="tight"); print("WROTE options_by_regime.png")

L = ["# 28-day REAL ATM straddle P&L by gap / CPR (1.3x stop, exit 14:45)\n",
     "**n per bucket is tiny (28-day window) -> directional only, NOT proof.** Prior H/L/C & gap from the "
     "chain's own underlying_spot.\n",
     "## By gap bucket\n| gap | net ₹ | n | avg ₹/day |\n|---|---|---|---|"]
for q in by_gap.index: L.append(f"| {q} | {by_gap.loc[q,'net']:,.0f} | {int(by_gap.loc[q,'n'])} | {by_gap.loc[q,'avg']:,.0f} |")
L.append("\n## By CPR width\n| CPR | net ₹ | n | avg ₹/day |\n|---|---|---|---|")
for q in by_cpr.index: L.append(f"| {q} | {by_cpr.loc[q,'net']:,.0f} | {int(by_cpr.loc[q,'n'])} | {by_cpr.loc[q,'avg']:,.0f} |")
L.append("\n## gap x DTE (net ₹)\n" + piv.fillna(0).round(0).to_string())
L.append("\n- Consistent with the years layer if gap-down days are the worst here too. CPR split here is tiny-n.")
(OUT / "RESULTS_options_regime.md").write_text("\n".join(L), encoding="utf-8"); print("WROTE RESULTS_options_regime.md")
oc.close()
