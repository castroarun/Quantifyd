"""Phase C — optimize / new systems for betterment. Driven by the baseline findings:
  - Squeeze family is robust across ALL DTEs (its ATR-compression filter self-selects calm
    days) -> should NOT be DTE-gated.
  - 916 family bleeds far-DTE (no vol filter on a blind 09:16 entry) -> near-expiry only.
  - A +-move-stop caps the trend-day tail (collapses ATM/ATM2/ATM4 into one system).

Tests:
  1. move-stop threshold sweep (param sensitivity / monotonicity): squeeze & 916, mp in
     {0.3,0.4,0.5,0.6%}, all-DTE and 0+1-DTE.
  2. Candidate BOOKS head-to-head (lots=2): equity curve + Net/PerDay/MaxDD/Sharpe/Calmar:
     A LiveCurrent  = premium, 6 sys, Fri/Mon/Tue (dte in 0,1,4)         [what arms Monday]
     B SmartGate    = premium, squeeze(all days) + 916(Mon+Tue only)     [no code, matrix only]
     C SqueezeOnly  = premium, squeeze 3 variants, all days              [drop 916]
     D MoveUnified  = squeeze-move(all days) + 916-move(Mon+Tue)         [move-stop code]
     E AllMove01    = move 2-sys, 0+1 DTE                                [move-stop + tight gate]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import load_day, sim_loaded, days, oc  # noqa

OUT = Path(__file__).resolve().parents[1] / "results"
SQ = ["Squeeze ATM", "Squeeze ATM2", "Squeeze ATM4"]
NN = ["916 ATM", "916 ATM2", "916 ATM4"]
MPS = [0.003, 0.004, 0.005, 0.006]


def mdd(s):
    return float((s - s.cummax()).min())


def sh(d):
    d = np.asarray(d, float)
    return round(d.mean() / d.std() * np.sqrt(252), 2) if d.std() > 0 else 0.0


# ---- move-stop threshold sweep (1 representative system per family) ----
mp_rows = []
for di, day in enumerate(days):
    b = load_day(day)
    if b is None:
        continue
    for mp in MPS:
        for fam, em in [("squeeze", "squeeze"), ("916", "t916")]:
            for r in sim_loaded(b, em, "SL_ST", stop_mode="move", move_pct=mp):
                r.update(fam=fam, mp=mp); mp_rows.append(r)
    if di % 10 == 0:
        print(f"  mp-sweep day {di+1}/{len(days)}", flush=True)
MP = pd.DataFrame(mp_rows); MP["day"] = pd.to_datetime(MP["day"])
dates = sorted(MP["day"].unique())
print("\n=== MOVE-STOP THRESHOLD SWEEP (per family, lots=2) ===")
print(f"{'fam':8} {'mp%':5} {'gate':7} {'net':>7} {'perday':>7} {'maxdd':>8} {'sharpe':>7} {'calmar':>7}")
for fam in ["squeeze", "916"]:
    for mp in MPS:
        for gate, mask in [("all", MP["dte"] >= 0), ("0+1DTE", MP["dte"].isin([0, 1]))]:
            g = MP[(MP["fam"] == fam) & (MP["mp"] == mp) & mask]
            bd = g.groupby("day")["pnl"].sum().reindex(dates).fillna(0); eq = bd.cumsum()
            cal = round(bd.sum() / abs(mdd(eq)), 2) if mdd(eq) < 0 else 99
            print(f"{fam:8} {mp*100:<5.1f} {gate:7} {bd.sum():7.0f} {bd.sum()/len(dates):7.0f} {mdd(eq):8.0f} {sh(bd):7} {cal:7}")
MP.to_csv(OUT / "optimize_mp_sweep.csv", index=False)

# ---- candidate books from existing premium legs + move legs ----
PREM = pd.read_csv(OUT / "improve_legs.csv"); PREM["day"] = pd.to_datetime(PREM["day"])
prem = PREM[PREM["config"] == "prem_ATM"]
BESTMP = 0.004   # confirmed/representative; sweep above shows sensitivity


def book_daily(legs):
    return legs.groupby("day")["pnl"].sum().reindex(dates).fillna(0)


books = {}
# A LiveCurrent: premium 6 sys, Fri/Mon/Tue = dte in {0,1,4}
books["A LiveCurrent (prem, Fri/Mon/Tue)"] = book_daily(prem[prem["dte"].isin([0, 1, 4])])
# B SmartGate: squeeze all-DTE + 916 Mon+Tue
sg = pd.concat([prem[prem["system"].isin(SQ)], prem[prem["system"].isin(NN) & prem["dte"].isin([0, 1])]])
books["B SmartGate (sq all + 916 Mon/Tue)"] = book_daily(sg)
# C SqueezeOnly all days
books["C SqueezeOnly (prem, all days)"] = book_daily(prem[prem["system"].isin(SQ)])
# D MoveUnified: squeeze-move all-DTE + 916-move Mon+Tue (1 sys per family)
md = pd.concat([MP[(MP["fam"] == "squeeze") & (MP["mp"] == BESTMP)],
                MP[(MP["fam"] == "916") & (MP["mp"] == BESTMP) & MP["dte"].isin([0, 1])]])
books["D MoveUnified (sq-move all + 916-move Mon/Tue)"] = book_daily(md)
# E AllMove 0+1DTE (2 sys)
ae = MP[(MP["mp"] == BESTMP) & MP["dte"].isin([0, 1])]
books["E AllMove 0+1DTE (2 sys)"] = book_daily(ae)

print("\n=== CANDIDATE BOOKS (lots=2) ===")
print(f"{'book':46} {'net':>7} {'perday':>7} {'maxdd':>8} {'sharpe':>7} {'calmar':>7} {'daywin':>6}")
summ = []
for name, bd in books.items():
    eq = bd.cumsum(); cal = round(bd.sum() / abs(mdd(eq)), 2) if mdd(eq) < 0 else 99
    dw = round(100 * (bd[bd != 0] > 0).mean())
    print(f"{name:46} {bd.sum():7.0f} {bd.sum()/len(dates):7.0f} {mdd(eq):8.0f} {sh(bd):7} {cal:7} {dw:6}")
    summ.append(dict(book=name, net=round(bd.sum()), perday=round(bd.sum()/len(dates)),
                     maxdd=round(mdd(eq)), sharpe=sh(bd), calmar=cal, daywin=dw))
pd.DataFrame(summ).to_csv(OUT / "optimize_books.csv", index=False)

# ---- figure: candidate book equity curves ----
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.facecolor": "white"})
fig, ax = plt.subplots(figsize=(13, 7))
for name, bd in books.items():
    ax.plot(dates, bd.cumsum().values, lw=1.8, label=name)
ax.axhline(0, color="#999", lw=.8); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("NAS Phase C — candidate book equity curves (lots=2, 41 days)", fontweight="bold")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout(); fig.savefig(OUT / "nas6_optimize.png", dpi=120, bbox_inches="tight")
print("WROTE", OUT / "nas6_optimize.png")
oc.close()
