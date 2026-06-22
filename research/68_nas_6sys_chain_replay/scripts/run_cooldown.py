"""Backtest the move-stop RE-CENTER cooldown ('new system' per user 2026-06-22): does
exempting the 15-min re-entry cooldown (always re-center) beat throttling it — and do either
beat plain one-and-done? ATM2 move-stop book = squeeze (all days) + 916 (Mon+Tue), SmartGate,
lots=2, interleaved train/test. Churn proxy = total legs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import load_day, sim_loaded, days, oc  # noqa

OUT = Path(__file__).resolve().parents[1] / "results"
SYS = [("Squeeze ATM2", "squeeze"), ("916 ATM2", "t916")]
CONFIGS = [
    ("move_oneanddone", dict(stop_mode="move", move_pct=0.004, move_stop_reenter=False)),
    ("move_recenter_NOcooldown", dict(stop_mode="move", move_pct=0.004, move_stop_reenter=True, reenter_cooldown_min=0)),
    ("move_recenter_15mincool", dict(stop_mode="move", move_pct=0.004, move_stop_reenter=True, reenter_cooldown_min=15)),
]

rows = []
for di, day in enumerate(days):
    b = load_day(day)
    if b is None:
        continue
    for clabel, kw in CONFIGS:
        for name, em in SYS:
            for r in sim_loaded(b, em, "CASCADE", **kw):
                r.update(config=clabel, system=name); rows.append(r)
    if di % 10 == 0:
        print(f"  day {di+1}/{len(days)}", flush=True)
L = pd.DataFrame(rows); L["day"] = pd.to_datetime(L["day"])
dates = sorted(L["day"].unique())
train = set(dates[::2]); test = set(dates[1::2])


def stats(daily):
    eq = daily.cumsum(); dd = float((eq - eq.cummax()).min())
    d = daily.values.astype(float)
    sh = round(d.mean() / d.std() * np.sqrt(252), 2) if d.std() > 0 else 0.0
    cal = round(daily.sum() / abs(dd), 2) if dd < 0 else 99
    return int(round(daily.sum())), int(round(dd)), sh, cal


def book(sub):
    return pd.concat([sub[sub["system"] == "Squeeze ATM2"],
                      sub[(sub["system"] == "916 ATM2") & sub["dte"].isin([0, 1])]])


print("\n=== MOVE-STOP RE-CENTER COOLDOWN BAKE-OFF (SmartGate book, lots=2, 41d) ===")
print(f"{'config':26} {'legs':>5} {'net':>7} {'maxdd':>8} {'sharpe':>7} {'calmar':>7} | {'TRAINnet':>8} {'TESTnet':>8} {'TESTsh':>6}")
res = []
for clabel, _ in CONFIGS:
    g = book(L[L["config"] == clabel])
    bd = g.groupby("day")["pnl"].sum().reindex(dates).fillna(0)
    a = stats(bd); tr = stats(bd[[d in train for d in bd.index]]); te = stats(bd[[d in test for d in bd.index]])
    print(f"{clabel:26} {len(g):5} {a[0]:7} {a[1]:8} {a[2]:7} {a[3]:7} | {tr[0]:8} {te[0]:8} {te[2]:6}")
    res.append(dict(config=clabel, legs=len(g), net=a[0], maxdd=a[1], sharpe=a[2], calmar=a[3],
                    train_net=tr[0], test_net=te[0], test_sh=te[2]))
pd.DataFrame(res).to_csv(OUT / "cooldown_bakeoff.csv", index=False)
od = next(r for r in res if r["config"] == "move_oneanddone")
nc = next(r for r in res if r["config"] == "move_recenter_NOcooldown")
print(f"\nRe-center adds {nc['legs']-od['legs']} legs vs one-and-done; net {od['net']} -> {nc['net']} ({nc['net']-od['net']:+d})")
oc.close()
