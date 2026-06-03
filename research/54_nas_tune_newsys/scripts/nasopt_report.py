"""NAS-OPT backtest performance report — the research/54 refined system on the recorded chain.

SYSTEM (NAS-OPT): on each 0/1-DTE day, 09:20 sell ~100pt-OTM strangle (2 strikes OTM each side),
±0.4% underlying-move stop (full exit, one-and-done), else time-exit 14:45. Net ₹80/leg.
Tag each trade calm (opening-15min range < median) vs wide. Real recorded NIFTY chain (29 days).

Outputs (research/54_nas_tune_newsys/results/):
  nasopt_trades.csv        per-trade ledger
  nasopt_perf.png          P&L curve + drawdown + per-day bars + KPI strip
  RESULTS_nasopt_report.md KPI summary
This is the durable past-days backtest record + performance report (PL curves).
"""
import sqlite3
from pathlib import Path
from datetime import time as dtime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/arun/quantifyd"); BASE = Path(__file__).resolve().parents[1] if "__file__" in dir() else ROOT / "research/54_nas_tune_newsys"
OUT = ROOT / "research/54_nas_tune_newsys/results"; OUT.mkdir(parents=True, exist_ok=True)
OPT = ROOT / "backtest_data" / "options_data.db"
LOT = 65; QTY = LOT * 2; BROK = 80
ENTRY = dtime(9, 20); TIMEEXIT = dtime(14, 45); EOD = dtime(15, 15); OFF = 2; MOVE = 0.4
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

def trade(day):
    d = load(day)
    if d is None: return None
    dte, spot_s, chain = d
    if dte > 1: return None  # NAS-OPT trades 0/1-DTE only
    times = [t for t in spot_s.index if ENTRY <= t.time() <= EOD]
    if not times: return None
    op = spot_s[(spot_s.index.time >= dtime(9, 15)) & (spot_s.index.time <= dtime(9, 30))]
    orpct = (op.max() - op.min()) / op.iloc[0] * 100 if len(op) >= 2 else np.nan
    t0 = times[0]; spot0 = float(spot_s.loc[t0]); atm = round(spot0 / 50) * 50
    legs = []
    for typ, sgn in (("CE", 1), ("PE", -1)):
        k = atm + sgn * OFF * 50; ts = tsym(chain, k, typ); p = prem(chain, ts, t0) if ts else None
        if p: legs.append({"ts": ts, "k": k, "typ": typ, "entry": p, "open": True})
    if len(legs) < 2: return None
    credit = sum(l["entry"] for l in legs)
    pnl = 0.0; exit_reason = "time1445"
    for t in times:
        if not any(l["open"] for l in legs): break
        force = t.time() >= TIMEEXIT
        moved = abs(float(spot_s.loc[t]) - spot0) / spot0 * 100 >= MOVE
        for lg in legs:
            if not lg["open"]: continue
            p = prem(chain, lg["ts"], t)
            if p is None: continue
            if force or moved:
                pnl += (lg["entry"] - p) * QTY - BROK; lg["open"] = False
                exit_reason = "move0.4%" if moved and not force else "time1445"
    lastt = times[-1]
    for lg in legs:
        if lg["open"]:
            p = prem(chain, lg["ts"], lastt) or lg["entry"]; pnl += (lg["entry"] - p) * QTY - BROK
    return dict(day=day, wd=pd.Timestamp(day).day_name()[:3], dte=dte, spot0=round(spot0),
                ce_k=int([l["k"] for l in legs if l["typ"]=="CE"][0]), pe_k=int([l["k"] for l in legs if l["typ"]=="PE"][0]),
                credit=round(credit, 1), orpct=round(orpct, 3) if orpct==orpct else None, exit=exit_reason, pnl=round(pnl))

print("building NAS-OPT report...", flush=True)
rows = [trade(d) for d in DAYS]; rows = [r for r in rows if r]
T = pd.DataFrame(rows)
ormed = T["orpct"].median()
T["calm"] = T["orpct"] < ormed
T["cum"] = T["pnl"].cumsum()
T["peak"] = T["cum"].cummax(); T["dd"] = T["cum"] - T["peak"]
T.to_csv(OUT / "nasopt_trades.csv", index=False)

a = T["pnl"].values
kpi = dict(trades=len(a), total=int(a.sum()), pertrade=int(a.mean()), win=round(100*(a>0).mean(),1),
           avgwin=int(a[a>0].mean()) if (a>0).any() else 0, avgloss=int(a[a<0].mean()) if (a<0).any() else 0,
           worst=int(a.min()), best=int(a.max()), maxdd=int(T["dd"].min()),
           sharpe=round(a.mean()/a.std()*np.sqrt(252),2) if a.std()>0 else 0)
calm_t = T[T["calm"]]["pnl"]; wide_t = T[~T["calm"]]["pnl"]

fig = plt.figure(figsize=(14, 9)); gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 0.7])
ax1 = fig.add_subplot(gs[0, :]); ax1.plot(range(len(T)), T["cum"], color="#0a6", lw=2, marker="o", ms=3)
ax1.fill_between(range(len(T)), T["cum"], color="#0a6", alpha=.08); ax1.axhline(0, color="#999", lw=.8)
ax1.set_title("NAS-OPT cumulative P&L — 0/1-DTE ~100pt-OTM strangle + ±0.4%% move-stop (real chain, %d trades)" % len(T), fontsize=11)
ax1.set_ylabel("cum ₹"); ax1.grid(alpha=.2)
ax2 = fig.add_subplot(gs[1, :]); cols = ["#0a6" if v>=0 else "#d33" for v in T["pnl"]]
ax2.bar(range(len(T)), T["pnl"], color=cols); ax2.axhline(0, color="#333", lw=.8)
ax2.set_title("Per-trade P&L (green=win)", fontsize=10); ax2.set_ylabel("₹")
ax2.set_xticks(range(len(T))); ax2.set_xticklabels([f"{r.day[5:]}\n{r.wd}" for r in T.itertuples()], fontsize=6, rotation=0)
ax3 = fig.add_subplot(gs[2, 0]); ax3.fill_between(range(len(T)), T["dd"], color="#d33", alpha=.5); ax3.axhline(0, color="#333", lw=.8)
ax3.set_title("Drawdown ₹ (maxDD %d)" % kpi["maxdd"], fontsize=10)
ax4 = fig.add_subplot(gs[2, 1]); ax4.axis("off")
txt = ("KPIs\n  trades %d   total ₹%s   per-trade ₹%s\n  win %% %s   avg win ₹%s   avg loss ₹%s\n"
       "  best ₹%s   worst ₹%s   maxDD ₹%s   Sharpe %s\n  calm-day ₹/t %s (n%d)  wide-day ₹/t %s (n%d)") % (
    kpi["trades"], f"{kpi['total']:,}", f"{kpi['pertrade']:,}", kpi["win"], f"{kpi['avgwin']:,}", f"{kpi['avgloss']:,}",
    f"{kpi['best']:,}", f"{kpi['worst']:,}", f"{kpi['maxdd']:,}", kpi["sharpe"],
    int(calm_t.mean()) if len(calm_t) else 0, len(calm_t), int(wide_t.mean()) if len(wide_t) else 0, len(wide_t))
ax4.text(0, .95, txt, va="top", fontsize=9, family="monospace")
fig.suptitle("NAS-OPT — backtest performance report (recorded NIFTY chain, %s → %s)" % (T["day"].iloc[0], T["day"].iloc[-1]), fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "nasopt_perf.png", dpi=110, bbox_inches="tight")

L = ["# NAS-OPT — backtest performance report (recorded NIFTY chain)\n",
     "System: 0/1-DTE · ~100pt-OTM strangle · 09:20 · ±0.4%% underlying-move stop (one-and-done) · exit 14:45. Net ₹80/leg. %s → %s.\n" % (T["day"].iloc[0], T["day"].iloc[-1]),
     "## KPIs", "| metric | value |", "|---|---|"]
for k, v in kpi.items(): L.append("| %s | %s |" % (k, v))
L += ["\n## Calm vs wide opening-range day", "| | n | ₹/trade |", "|---|---|---|",
      "| calm (OR<median) | %d | %s |" % (len(calm_t), int(calm_t.mean()) if len(calm_t) else 0),
      "| wide | %d | %s |" % (len(wide_t), int(wide_t.mean()) if len(wide_t) else 0),
      "\nArtifacts: `nasopt_trades.csv`, `nasopt_perf.png`. 29-day window = SIGNAL; paper-forward as recorder grows."]
(OUT / "RESULTS_nasopt_report.md").write_text("\n".join(L), encoding="utf-8")
print("DONE", kpi, flush=True)
oc.close()
