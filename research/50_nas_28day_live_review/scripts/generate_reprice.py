"""Phase 2: re-price every actual CLOSED leg against the recorded option_chain.

For each leg the systems actually traded (positions table, status=CLOSED, since
START), look up the REAL recorded premium at the leg's entry_time and exit_time
from options_data.db, and recompute P&L = (entry_real - exit_real) * qty (short).
This corrects the OTM trade-recorder bug (exit booked at Rs0) and captures roll
costs, using the systems' actual decisions (no strategy reimplementation, no
look-ahead — timestamps already happened).
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path("/home/arun/quantifyd")
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"
OUT.mkdir(parents=True, exist_ok=True)
START = "2026-04-20"
OPT_DB = ROOT / "backtest_data" / "options_data.db"
TOL_SEC = 600  # accept nearest chain snapshot within 10 min of the leg timestamp

SYSTEMS = [
    ("nas_trading.db",          "nas_positions",     "Squeeze OTM"),
    ("nas_atm_trading.db",      "nas_atm_positions", "Squeeze ATM"),
    ("nas_atm2_trading.db",     "nas_atm_positions", "Squeeze ATM2"),
    ("nas_atm4_trading.db",     "nas_atm_positions", "Squeeze ATM4"),
    ("nas_916_otm_trading.db",  "nas_positions",     "916 OTM"),
    ("nas_916_atm_trading.db",  "nas_atm_positions", "916 ATM"),
    ("nas_916_atm2_trading.db", "nas_atm_positions", "916 ATM2"),
    ("nas_916_atm4_trading.db", "nas_atm_positions", "916 ATM4"),
]
ORDER = [s[2] for s in SYSTEMS]

# index for fast tradingsymbol+time lookup
oc = sqlite3.connect(str(OPT_DB))
oc.execute("CREATE INDEX IF NOT EXISTS idx_oc_tsym_time ON option_chain(tradingsymbol, snapshot_time)")
oc.commit()

def chain_prem(tsym, ts, side):
    """Nearest recorded ltp for tradingsymbol around ISO time ts (<= preferred,
    else nearest after), within TOL_SEC. side: 'entry'/'exit' (just for logging)."""
    if not tsym or not ts:
        return None
    row = oc.execute(
        "SELECT ltp, snapshot_time FROM option_chain WHERE tradingsymbol=? AND snapshot_time<=? "
        "ORDER BY snapshot_time DESC LIMIT 1", (tsym, ts)).fetchone()
    after = oc.execute(
        "SELECT ltp, snapshot_time FROM option_chain WHERE tradingsymbol=? AND snapshot_time>? "
        "ORDER BY snapshot_time ASC LIMIT 1", (tsym, ts)).fetchone()
    cands = []
    for r in (row, after):
        if r and r[0] is not None:
            dt = abs((pd.to_datetime(r[1]) - pd.to_datetime(ts)).total_seconds())
            cands.append((dt, r[0]))
    if not cands:
        return None
    cands.sort()
    return cands[0][1] if cands[0][0] <= TOL_SEC else None

rows = []
for db, tbl, label in SYSTEMS:
    p = ROOT / "backtest_data" / db
    if not p.exists():
        continue
    cx = sqlite3.connect(str(p))
    try:
        legs = pd.read_sql_query(
            f"SELECT tradingsymbol, instrument_type, strike, qty, entry_time, exit_time, "
            f"entry_price, exit_price, strangle_id FROM {tbl} "
            f"WHERE status='CLOSED' AND entry_time >= ?", cx, params=[START])
    except Exception as e:
        print(f"skip {db}: {e}"); cx.close(); continue
    cx.close()
    for _, lg in legs.iterrows():
        qty = lg["qty"] or 0
        e_real = chain_prem(lg["tradingsymbol"], lg["entry_time"], "entry")
        x_real = chain_prem(lg["tradingsymbol"], lg["exit_time"], "exit")
        rec_pnl = ((lg["entry_price"] or 0) - (lg["exit_price"] or 0)) * qty
        repriced = None
        if e_real is not None and x_real is not None and qty:
            repriced = (e_real - x_real) * qty
        rows.append(dict(system=label, trade_date=str(lg["entry_time"])[:10],
                         qty=qty, entry_real=e_real, exit_real=x_real,
                         rec_pnl=rec_pnl, repriced_pnl=repriced))

legdf = pd.DataFrame(rows)
legdf.to_csv(OUT / "reprice_legs.csv", index=False)

# coverage + aggregates
summary = []
for s in ORDER:
    g = legdf[legdf["system"] == s]
    if g.empty:
        continue
    cov = g["repriced_pnl"].notna()
    summary.append(dict(
        System=s, Legs=len(g), Repriced=int(cov.sum()),
        CovPct=round(100 * cov.mean(), 0),
        Recorded=round(g["rec_pnl"].sum()),
        RepricedNet=round(g.loc[cov, "repriced_pnl"].sum()),
        ReprWin=round(100 * (g.loc[cov, "repriced_pnl"] > 0).mean(), 0) if cov.any() else 0,
    ))
summ = pd.DataFrame(summary)

# per-day per-system repriced (covered legs only)
cov = legdf[legdf["repriced_pnl"].notna()].copy()
cov["trade_date"] = pd.to_datetime(cov["trade_date"])
daily = cov.groupby(["system", "trade_date"])["repriced_pnl"].sum().reset_index()
dates = sorted(cov["trade_date"].unique())
present = [s for s in ORDER if s in set(cov["system"])]
pv = daily.pivot(index="system", columns="trade_date", values="repriced_pnl").reindex(present).reindex(columns=dates).fillna(0.0)
cum = pv.cumsum(axis=1)
comb = pv.sum(axis=0).cumsum()

# ---- figure ----
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white"})
fig = plt.figure(figsize=(15, 14))
gs = GridSpec(3, 2, height_ratios=[0.4, 1.2, 1.2], hspace=0.4, wspace=0.2)
ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
tot_rec = legdf["rec_pnl"].sum(); tot_rep = legdf.loc[legdf["repriced_pnl"].notna(), "repriced_pnl"].sum()
ax0.text(0, 0.7, "NAS 8-System — Phase 2: P&L re-priced from recorded option_chain", fontsize=16, fontweight="bold")
ax0.text(0, 0.25, f"Closed legs re-priced at REAL recorded premiums (entry+exit). "
                  f"Recorded-table net ₹{tot_rec:,.0f}  →  re-priced net ₹{tot_rep:,.0f}  "
                  f"(corrects OTM exit=0). Coverage = legs found in chain.", fontsize=10, color="#333")

# recorded vs repriced bar
ax1 = fig.add_subplot(gs[1, 0])
x = np.arange(len(summ)); w = 0.4
ax1.bar(x - w/2, summ["Recorded"], w, label="recorded-table", color="#bbb")
ax1.bar(x + w/2, summ["RepricedNet"], w, label="re-priced (real)", color="#0a6")
ax1.set_xticks(x); ax1.set_xticklabels(summ["System"], rotation=45, ha="right", fontsize=7)
ax1.axhline(0, color="#333", lw=0.8); ax1.legend(fontsize=8)
ax1.set_title("Recorded vs chain-re-priced net P&L (₹) per system")

# repriced per-system cumulative
ax2 = fig.add_subplot(gs[1, 1])
for s in present:
    ax2.plot(cum.columns, cum.loc[s].values, lw=1.4, label=s)
ax2.axhline(0, color="#999", lw=0.8); ax2.legend(fontsize=7, ncol=2)
ax2.set_title("Re-priced cumulative net P&L per system (₹)"); ax2.tick_params(axis="x", rotation=45)

# combined repriced + coverage table
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(comb.index, comb.values, color="#06c", lw=2); ax3.axhline(0, color="#999", lw=0.8)
ax3.set_title("Combined book — re-priced cumulative net P&L (₹)"); ax3.tick_params(axis="x", rotation=45)

ax4 = fig.add_subplot(gs[2, 1]); ax4.axis("off")
disp = summ.copy()
for cc in ["Recorded", "RepricedNet"]:
    disp[cc] = disp[cc].map(lambda v: f"{v:,.0f}")
disp["CovPct"] = disp["CovPct"].map(lambda v: f"{v:.0f}%")
disp["ReprWin"] = disp["ReprWin"].map(lambda v: f"{v:.0f}%")
t = ax4.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.7)
for j in range(len(disp.columns)):
    t[0, j].set_facecolor("#222"); t[0, j].set_text_props(color="white", fontweight="bold")
ax4.set_title("Coverage + recorded vs re-priced (₹)", y=0.9, fontweight="bold")

fig.savefig(OUT / "nas_28day_reprice.png", dpi=110, bbox_inches="tight")
print("WROTE", OUT / "nas_28day_reprice.png")

# ---- RESULTS_phase2.md ----
L = ["# NAS 8-System — Phase 2: chain-re-priced P&L\n",
     f"Closed legs re-priced at REAL recorded option_chain premiums (entry+exit), within {TOL_SEC//60}-min tolerance.\n",
     f"**Recorded-table net ₹{tot_rec:,.0f} → re-priced net ₹{tot_rep:,.0f}.**\n",
     "| " + " | ".join(summ.columns) + " |",
     "|" + "|".join(["---"] * len(summ.columns)) + "|"]
for _, r in summ.iterrows():
    L.append("| " + " | ".join(str(r[c]) for c in summ.columns) + " |")
L += ["\n**Read:** where re-priced << recorded, the recorder over-stated P&L (esp. OTM exit=0).",
      "Low coverage % = legs at strikes/times outside the recorded chain window (excluded, not zero).",
      "Still a 25-day single-regime sample — signal/audit, not validation."]
(OUT / "RESULTS_phase2.md").write_text("\n".join(L), encoding="utf-8")
print("WROTE", OUT / "RESULTS_phase2.md")
oc.close()
