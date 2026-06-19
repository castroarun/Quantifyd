"""Phase B improvement sweep: the only two levers research/54 left alive, + DTE-gating.

Axes per system (6):  STOP ∈ {premium_1.3x, move_0.4pct} × STRIKE ∈ {ATM, 100-OTM} = 4 configs.
For each config: full 41-day replay -> per-system net/day/MaxDD + a COMBINED 6-book, reported
both all-DTE and 0+1-DTE-gated (the one robust prior finding). Rank to find best performance.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import load_day, sim_loaded, SYSTEMS_6, days, oc  # noqa

OUT = Path(__file__).resolve().parents[1] / "results"; OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("premium", 0,   "prem_ATM"),
    ("premium", 100, "prem_OTM"),
    ("move",    0,   "move_ATM"),
    ("move",    100, "move_OTM"),
]


def mdd(s):
    return float((s - s.cummax()).min())


def sharpe(d):
    d = np.asarray(d, float)
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0


rows = []          # per-leg across all configs
for di, day in enumerate(days):           # day-outer: load each day ONCE, reuse across cells
    b = load_day(day)
    if b is None:
        continue
    for stop_mode, off, clabel in CONFIGS:
        for name, em, mg, live_lots in SYSTEMS_6:
            try:
                for r in sim_loaded(b, em, mg, stop_mode=stop_mode, strike_offset=off):
                    r.update(system=name, config=clabel, stop=stop_mode, strike=("OTM" if off else "ATM"))
                    rows.append(r)
            except Exception as e:
                print(f"ERR {clabel} {name} {day}: {e}")
    if di % 10 == 0:
        print(f"  day {di+1}/{len(days)} {day}", flush=True)
L = pd.DataFrame(rows)
L.to_csv(OUT / "improve_legs.csv", index=False)
L["day"] = pd.to_datetime(L["day"])
dates = sorted(L["day"].unique())

# ---- per-system × config grid ----
grid = []
for clabel in [c[2] for c in CONFIGS]:
    for name, *_ in SYSTEMS_6:
        g = L[(L["config"] == clabel) & (L["system"] == name)]
        if g.empty:
            continue
        bd = g.groupby("day")["pnl"].sum().reindex(dates).fillna(0)
        grid.append(dict(config=clabel, system=name, net=round(g["pnl"].sum()),
                         perday=round(g["pnl"].sum() / len(dates)),
                         maxdd=round(mdd(bd.cumsum())), sharpe=round(sharpe(bd), 2),
                         daywin=round(100 * (bd[bd != 0] > 0).mean())))
grid = pd.DataFrame(grid)
grid.to_csv(OUT / "improve_grid.csv", index=False)

# ---- combined 6-book per config: all-DTE and 0+1-DTE ----
comb_rows = []
equity = {}
for clabel in [c[2] for c in CONFIGS]:
    for gate, mask in [("all", L["dte"] >= 0), ("0+1DTE", L["dte"].isin([0, 1]))]:
        g = L[(L["config"] == clabel) & mask]
        bd = g.groupby("day")["pnl"].sum().reindex(dates).fillna(0)
        eq = bd.cumsum()
        equity[(clabel, gate)] = eq
        comb_rows.append(dict(config=clabel, gate=gate, net=round(bd.sum()),
                              perday=round(bd.sum() / len(dates)), maxdd=round(mdd(eq)),
                              sharpe=round(sharpe(bd), 2), daywin=round(100 * (bd[bd != 0] > 0).mean()),
                              calmar=round(bd.sum() / abs(mdd(eq)), 2) if mdd(eq) < 0 else float("inf")))
comb = pd.DataFrame(comb_rows)
comb.to_csv(OUT / "improve_combined.csv", index=False)

# ---- figure: equity curves of the combined books ----
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.facecolor": "white"})
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for (clabel, gate), eq in equity.items():
    ax = axes[0] if gate == "all" else axes[1]
    ax.plot(eq.index, eq.values, lw=1.6, label=clabel)
for ax, ttl in zip(axes, ["Combined 6-book — ALL DTE", "Combined 6-book — 0+1 DTE only"]):
    ax.axhline(0, color="#999", lw=.8); ax.legend(fontsize=8); ax.set_title(ttl)
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("NAS 6-system improvement sweep — combined equity by config (lots=2)", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "nas6_improve.png", dpi=110, bbox_inches="tight")
print("WROTE", OUT / "nas6_improve.png")
print("\n=== COMBINED 6-BOOK BY CONFIG ===")
print(comb.to_string(index=False))
print("\n=== PER-SYSTEM × CONFIG (net) ===")
print(grid.pivot(index="system", columns="config", values="net").to_string())
print("\n=== PER-SYSTEM × CONFIG (maxdd) ===")
print(grid.pivot(index="system", columns="config", values="maxdd").to_string())
oc.close()
