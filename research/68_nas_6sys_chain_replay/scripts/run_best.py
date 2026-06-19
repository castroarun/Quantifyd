"""Phase D close-out: stack the ROBUST (train+test) mechanic wins on top of SmartGate and
compare to the current-mechanic SmartGate book. Robust wins kept:
  - naked-leg SuperTrend multiplier 2.0 -> 3.0 (SL_ST + ROLL_MATCH systems)
  - ATM2 cascade -> +-0.4% underlying move-stop
Rejected (hurt or no-edge OOS): profit-target, early exit, whole-position stop, OTM width,
wider SL. Gating = SmartGate (squeeze all days; 916 Mon+Tue).
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

OUT = Path(__file__).resolve().parents[1] / "results"
SQ = ["Squeeze ATM", "Squeeze ATM2", "Squeeze ATM4"]
NN = ["916 ATM", "916 ATM2", "916 ATM4"]

rows = []
for di, day in enumerate(days):
    b = load_day(day)
    if b is None:
        continue
    for name, em, mg, _ in SYSTEMS_6:
        # current mechanic
        for r in sim_loaded(b, em, mg):
            r.update(book="current", system=name); rows.append(r)
        # best-stack
        if mg == "CASCADE":
            kw = dict(stop_mode="move", move_pct=0.004)
        else:
            kw = dict(st_mult=3.0)
        for r in sim_loaded(b, em, mg, **kw):
            r.update(book="best", system=name); rows.append(r)
    if di % 10 == 0:
        print(f"  day {di+1}/{len(days)}", flush=True)

df = pd.DataFrame(rows); df["day"] = pd.to_datetime(df["day"])
dates = sorted(df["day"].unique())
train = set(dates[::2]); test = set(dates[1::2])


def smartgate(sub):
    return pd.concat([sub[sub["system"].isin(SQ)], sub[sub["system"].isin(NN) & sub["dte"].isin([0, 1])]])


def stats(daily):
    eq = daily.cumsum(); dd = float((eq - eq.cummax()).min())
    d = daily.values.astype(float)
    sh = round(d.mean() / d.std() * np.sqrt(252), 2) if d.std() > 0 else 0.0
    cal = round(daily.sum() / abs(dd), 2) if dd < 0 else 99
    return int(round(daily.sum())), int(round(dd)), sh, cal


curves = {}
print(f"{'book':30} {'net':>7} {'maxdd':>8} {'sharpe':>7} {'calmar':>6} {'daywin':>6} | {'TRAINnet':>8} {'TESTnet':>8} {'TESTsh':>6}")
for bk in ["current", "best"]:
    bd = smartgate(df[df["book"] == bk]).groupby("day")["pnl"].sum().reindex(dates).fillna(0)
    curves[bk] = bd.cumsum()
    a = stats(bd); tr = stats(bd[[d in train for d in bd.index]]); te = stats(bd[[d in test for d in bd.index]])
    dw = round(100 * (bd[bd != 0] > 0).mean())
    print(f"{('SmartGate '+bk):30} {a[0]:7} {a[1]:8} {a[2]:7} {a[3]:6} {dw:5}% | {tr[0]:8} {te[0]:8} {te[2]:6}")
# per-system delta
print("\nper-system net (lots=2): current -> best")
for s in SQ + NN:
    gate = (df["dte"].isin([0, 1])) if s in NN else (df["dte"] >= 0)
    c = df[(df["book"] == "current") & (df["system"] == s) & gate]["pnl"].sum()
    bb = df[(df["book"] == "best") & (df["system"] == s) & gate]["pnl"].sum()
    print(f"  {s:14} {c:8.0f} -> {bb:8.0f}  ({bb-c:+.0f})")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, curves["current"].values, lw=2, color="#888", label="SmartGate — current mechanic")
ax.plot(dates, curves["best"].values, lw=2.2, color="#06c", label="SmartGate + ST×3.0 + ATM2 move-stop (best stack)")
ax.axhline(0, color="#999", lw=.8); ax.legend(fontsize=10, loc="upper left")
ax.set_title("NAS Phase D — best mechanic stack vs current (SmartGate, lots=2, 41d)", fontweight="bold")
ax.tick_params(axis="x", rotation=45); fig.tight_layout()
fig.savefig(OUT / "nas6_best.png", dpi=120, bbox_inches="tight")
print("\nWROTE", OUT / "nas6_best.png")
oc.close()
