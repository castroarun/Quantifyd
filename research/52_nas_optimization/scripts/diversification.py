"""Part (b): diversification — correlation of the 8 NAS systems' daily P&L and the
best uncorrelated positive subset. Uses research/51 replay legs (ALL eligible DTEs,
lots=2). 28-day sample => correlations are NOISY; treat as directional.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"; OUT.mkdir(parents=True, exist_ok=True)
LEGS = BASE.parent / "51_nas_chain_replay" / "results" / "replay_legs.csv"
ANN = np.sqrt(252)

df = pd.read_csv(LEGS)
df["day"] = pd.to_datetime(df["day"])
daily = df.groupby(["system", "day"])["pnl"].sum().unstack(0)  # rows=day, cols=system
daily = daily.reindex(sorted(daily.columns), axis=1)
# union of all trading days; a system not trading that day contributes 0 to the book
alldays = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
daily = daily.reindex(alldays).fillna(0.0)

def stats(series):
    m, s = series.mean(), series.std(ddof=1)
    sharpe = (m / s * ANN) if s > 0 else 0.0
    eq = series.cumsum(); dd = (eq - eq.cummax()).min()
    return dict(total=series.sum(), perday=m, sharpe=sharpe, maxdd=dd)

# per-system
rows = []
for s in daily.columns:
    st = stats(daily[s]); st["System"] = s; rows.append(st)
per = pd.DataFrame(rows).set_index("System")

corr = daily.corr()

# greedy subset maximizing annualized Sharpe of the (equal-weight) book
def book_sharpe(cols):
    if not cols: return -9
    return stats(daily[list(cols)].sum(axis=1))["sharpe"]
chosen = []
remaining = list(daily.columns)
best = -9
while remaining:
    cand = max(remaining, key=lambda c: book_sharpe(chosen + [c]))
    sh = book_sharpe(chosen + [cand])
    if sh <= best + 1e-9:
        break
    chosen.append(cand); remaining.remove(cand); best = sh

all8 = stats(daily.sum(axis=1))
sub = stats(daily[chosen].sum(axis=1)) if chosen else {}

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.05, 1]})
im = ax[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax[0].set_xticks(range(len(corr))); ax[0].set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax[0].set_yticks(range(len(corr))); ax[0].set_yticklabels(corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax[0].text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if abs(corr.values[i,j]) > 0.5 else "black")
ax[0].set_title("Daily-P&L correlation (28d, all-DTE) — NOISY, directional only")
fig.colorbar(im, ax=ax[0], fraction=.046, pad=.04)
# equity curves: all-8 vs chosen subset
ax[1].plot(daily.index, daily.sum(axis=1).cumsum(), lw=2, label=f"All 8 (Sharpe {all8['sharpe']:.1f})", color="#888")
if chosen:
    ax[1].plot(daily.index, daily[chosen].sum(axis=1).cumsum(), lw=2,
               label=f"Subset {chosen} (Sharpe {sub['sharpe']:.1f})", color="#0a6")
ax[1].axhline(0, color="#999", lw=.8); ax[1].legend(fontsize=8); ax[1].tick_params(axis="x", rotation=45)
ax[1].set_title("Cumulative net ₹ — all-8 vs uncorrelated subset")
fig.tight_layout(); fig.savefig(OUT / "diversification.png", dpi=110, bbox_inches="tight")
print("WROTE", OUT / "diversification.png")

# ---- RESULTS ----
L = ["# Part (b) — Diversification / uncorrelated subset\n",
     "Daily P&L of the 8 replayed systems (all-DTE, lots=2). **28-day sample -> correlations "
     "and Sharpe are NOISY; directional only, not validation.**\n",
     "## Per-system (daily P&L stats)\n",
     "| System | Total ₹ | ₹/day | Sharpe(ann) | MaxDD ₹ |", "|---|---|---|---|---|"]
for s in per.sort_values("sharpe", ascending=False).index:
    r = per.loc[s]; L.append(f"| {s} | {r['total']:,.0f} | {r['perday']:,.0f} | {r['sharpe']:.2f} | {r['maxdd']:,.0f} |")
L.append("\n## Greedy max-Sharpe uncorrelated subset")
L.append(f"**Selected: {chosen}**")
if chosen:
    L.append(f"- Subset book: total ₹{sub['total']:,.0f}, Sharpe(ann) {sub['sharpe']:.2f}, MaxDD ₹{sub['maxdd']:,.0f}")
L.append(f"- All-8 book: total ₹{all8['total']:,.0f}, Sharpe(ann) {all8['sharpe']:.2f}, MaxDD ₹{all8['maxdd']:,.0f}")
L.append("\n## Correlation matrix")
L.append("| | " + " | ".join(corr.columns) + " |")
L.append("|" + "|".join(["---"]*(len(corr.columns)+1)) + "|")
for s in corr.index:
    L.append("| " + s + " | " + " | ".join(f"{corr.loc[s,c]:.2f}" for c in corr.columns) + " |")
L.append("\n- Non-trading days counted as ₹0 (portfolio view). 28d, ~16-28 obs -> noisy.")
(OUT / "RESULTS_diversification.md").write_text("\n".join(L), encoding="utf-8")
print("WROTE", OUT / "RESULTS_diversification.md")
