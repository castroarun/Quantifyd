"""Phase D — optimize the MECHANICS across ALL meaningful angles, under SmartGate gating
(squeeze systems all days; 916 systems Mon+Tue only). Interleaved train/test split
(even/odd days) — a knob only 'wins' if it also holds on TEST. Everything = SIGNAL on a
41-day single regime; this MAPS the response surface, it does not pick a deployable optimum.

Angles:
  D1 SL multiplier        : per-leg premium stop x in {1.2..2.0}
  D2 SuperTrend naked     : period x mult x timeframe
  D3 Naked-trail method   : SuperTrend vs breakeven-lock vs %-trail-from-low
  D4 Cascade (ATM2) stop  : premium-cascade vs +-0.4% underlying move-stop
  D5 Profit target        : take whole-strangle profit at X% of credit (the big premium lever)
  D6 Exit time            : square-off at 14:00 / 14:30 / 15:00 / 15:15
  D7 Whole-position stop  : stop the strangle at X% of credit lost (per-leg SL OFF) vs per-leg
  D8 Structure / width    : ATM straddle (0) vs strangle offset 50/100/150
"""
import sys
from pathlib import Path
from datetime import time as dtime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import load_day, sim_loaded, SYSTEMS_6, days, oc  # noqa

OUT = Path(__file__).resolve().parents[1] / "results"
SYSMAP = {name: (em, mg) for name, em, mg, _ in SYSTEMS_6}
SQ = ["Squeeze ATM", "Squeeze ATM2", "Squeeze ATM4"]
NN = ["916 ATM", "916 ATM2", "916 ATM4"]
ATM_SYS = ["Squeeze ATM", "916 ATM"]
ATM2_SYS = ["Squeeze ATM2", "916 ATM2"]
SLMULTS = [1.2, 1.3, 1.4, 1.5, 1.75, 2.0]
ST_GRID = [(p, m, "5min") for p in (5, 7, 10) for m in (1.5, 2.0, 3.0)] + [(7, 2.0, "3min"), (7, 3.0, "3min")]
NAKED = [("breakeven", None), ("pct_trail", 0.3), ("pct_trail", 0.5)]
PTS = [0.35, 0.5, 0.65]
EXITS = [dtime(14, 0), dtime(14, 30), dtime(15, 0), dtime(15, 15)]
POSSTOPS = [0.5, 0.75, 1.0]
OFFSETS = [0, 50, 100, 150]

rows = []
for di, day in enumerate(days):
    b = load_day(day)
    if b is None:
        continue
    for name, (em, mg) in SYSMAP.items():
        for sl in SLMULTS:                                                  # D1
            for r in sim_loaded(b, em, mg, sl_mult=sl):
                r.update(test="sl", k=sl, system=name); rows.append(r)
        for pt in PTS:                                                      # D5
            for r in sim_loaded(b, em, mg, profit_target=pt):
                r.update(test="pt", k=pt, system=name); rows.append(r)
        for xt in EXITS:                                                    # D6
            for r in sim_loaded(b, em, mg, time_exit=xt):
                r.update(test="exit", k=xt.strftime("%H:%M"), system=name); rows.append(r)
        for ps in POSSTOPS:                                                 # D7 (per-leg SL off)
            for r in sim_loaded(b, em, mg, sl_mult=9.9, position_stop=ps):
                r.update(test="posstop", k=ps, system=name); rows.append(r)
        for off in OFFSETS:                                                 # D8
            for r in sim_loaded(b, em, mg, strike_offset=off):
                r.update(test="struct", k=off, system=name); rows.append(r)
    for (p, m, tf) in ST_GRID:                                             # D2
        for name in ATM_SYS:
            em, mg = SYSMAP[name]
            for r in sim_loaded(b, em, mg, st_period=p, st_mult=m, st_tf=tf):
                r.update(test="st", k=f"p{p}m{m}{tf}", system=name); rows.append(r)
    for (meth, pt) in NAKED:                                               # D3
        for name in ATM_SYS:
            em, mg = SYSMAP[name]
            kw = dict(naked_method=meth)
            if pt is not None:
                kw["pct_trail"] = pt
            for r in sim_loaded(b, em, mg, **kw):
                r.update(test="naked", k=f"{meth}{pt or ''}", system=name); rows.append(r)
    for sm in ("premium", "move"):                                        # D4
        for name in ATM2_SYS:
            em, mg = SYSMAP[name]
            for r in sim_loaded(b, em, mg, stop_mode=sm):
                r.update(test="cascade", k=sm, system=name); rows.append(r)
    if di % 5 == 0:
        print(f"  day {di+1}/{len(days)}", flush=True)

df = pd.DataFrame(rows); df["day"] = pd.to_datetime(df["day"])
df.to_csv(OUT / "phaseD_legs.csv", index=False)
dates = sorted(df["day"].unique())
train = set(dates[::2]); test = set(dates[1::2])


def stats(daily):
    eq = daily.cumsum(); dd = float((eq - eq.cummax()).min())
    d = daily.values.astype(float)
    sh = round(d.mean() / d.std() * np.sqrt(252), 2) if d.std() > 0 else 0.0
    cal = round(daily.sum() / abs(dd), 2) if dd < 0 else 99
    return int(round(daily.sum())), int(round(dd)), sh, cal


def smartgate(sub):
    return pd.concat([sub[sub["system"].isin(SQ)], sub[sub["system"].isin(NN) & sub["dte"].isin([0, 1])]])


res = []
def report(legs, label):
    bd = legs.groupby("day")["pnl"].sum().reindex(dates).fillna(0)
    tr = bd[[d in train for d in bd.index]]; te = bd[[d in test for d in bd.index]]
    a, t, e = stats(bd), stats(tr), stats(te)
    print(f"{label:24} ALL net {a[0]:7} dd {a[1]:7} sh {a[2]:5} cal {a[3]:6} | TRAIN {t[0]:6} sh {t[2]:5} | TEST {e[0]:6} sh {e[2]:5}")
    res.append(dict(lever=label, net=a[0], dd=a[1], sharpe=a[2], calmar=a[3], train_net=t[0], test_net=e[0], test_sh=e[2]))


print("\n=== D1 SL multiplier (SmartGate 6-book) — current=1.3 ===")
for sl in SLMULTS:
    report(smartgate(df[(df["test"] == "sl") & (df["k"] == sl)]), f"SL x{sl}")
print("\n=== D5 Profit-target (per-leg SL still on) — current=none ===")
for pt in PTS:
    report(smartgate(df[(df["test"] == "pt") & (df["k"] == pt)]), f"PT {int(pt*100)}% credit")
print("\n=== D6 Exit time — current=15:15 ===")
for xt in [x.strftime("%H:%M") for x in EXITS]:
    report(smartgate(df[(df["test"] == "exit") & (df["k"] == xt)]), f"exit {xt}")
print("\n=== D7 Whole-position stop (per-leg SL OFF) — vs per-leg ===")
for ps in POSSTOPS:
    report(smartgate(df[(df["test"] == "posstop") & (df["k"] == ps)]), f"posSL {int(ps*100)}% credit")
print("\n=== D8 Structure / width — current=ATM(0) ===")
for off in OFFSETS:
    report(smartgate(df[(df["test"] == "struct") & (df["k"] == off)]), f"offset {off}pt")
print("\n=== D2 SuperTrend naked trail (ATM SL_ST systems) — current p7 m2.0 5min ===")
st_res = []
for (p, m, tf) in ST_GRID:
    lab = f"ST p{p} m{m} {tf}"
    report(smartgate(df[(df["test"] == "st") & (df["k"] == f"p{p}m{m}{tf}")]), lab)
    st_res.append(res[-1])
print("\n=== D3 Naked method (ATM SL_ST systems) — vs ST above ===")
for (meth, pt) in NAKED:
    report(smartgate(df[(df["test"] == "naked") & (df["k"] == f"{meth}{pt or ''}")]), f"naked {meth}{pt or ''}")
print("\n=== D4 Cascade ATM2: premium vs move-0.4% ===")
for sm in ("premium", "move"):
    report(smartgate(df[(df["test"] == "cascade") & (df["k"] == sm)]), f"ATM2 {sm}")

pd.DataFrame(res).to_csv(OUT / "phaseD_summary.csv", index=False)
oc.close()
print("\nWROTE phaseD_summary.csv,", len(dates), "days, train", len(dates[::2]), "test", len(dates[1::2]))
