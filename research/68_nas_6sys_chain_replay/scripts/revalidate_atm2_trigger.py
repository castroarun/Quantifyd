"""research/68 — REVALIDATE the ATM2 exit trigger before arming the 9:16 systems live (2 lots).
Compares 30% per-leg PREMIUM SL (old) vs ±0.4% underlying MOVE-stop (v3, live now), both
cascade/re-enter with no cooldown (isolate the TRIGGER), on the 916 (+squeeze) ATM2 entry across
ALL recorded days; breaks P&L down by DTE and weekday. Read-only chain replay on options_data.db."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import load_day, sim_loaded, days, oc  # noqa

CONFIGS = [
    # A: old live behaviour — 30% SL closes+reopens at current ATM (can be the SAME strike = churn)
    ("premium30",   dict(stop_mode="premium", sl_mult=1.3, reenter_cooldown_min=0)),
    # C: user's fix — 30% SL breach AND price reached the NEXT ATM strike -> only then close+reopen
    #    at the new ATM; else HOLD (never same-strike churn)
    ("prem30_gate", dict(stop_mode="premium", sl_mult=1.3, strike_gate=True, reenter_cooldown_min=0)),
    # B: current v3 — ±0.4% underlying move -> re-center at new ATM
    ("move0.4",     dict(stop_mode="move", move_pct=0.004, move_stop_reenter=True, reenter_cooldown_min=0)),
]
ENTRY = [("916 ATM2", "t916"), ("squeeze ATM2", "squeeze")]

rows = []
for i, day in enumerate(days):
    b = load_day(day)
    if b is None:
        continue
    for elabel, em in ENTRY:
        for clabel, kw in CONFIGS:
            for r in sim_loaded(b, em, "CASCADE", **kw):
                r["config"] = clabel
                r["entry"] = elabel
                rows.append(r)
    if i % 10 == 0:
        print("day %d/%d" % (i + 1, len(days)), flush=True)

L = pd.DataFrame(rows)
L["day"] = pd.to_datetime(L["day"])
L["weekday"] = L["day"].dt.day_name()
daily = L.groupby(["entry", "config", "day", "dte", "weekday"])["pnl"].sum().reset_index()
daily.to_csv(Path(__file__).resolve().parents[1] / "results" / "atm2_trigger_revalidation.csv", index=False)


def summ(sub):
    if not len(sub):
        return (0, 0, 0, 0)
    return (int(sub.pnl.sum()), int(round(sub.pnl.mean())), int(round(100 * (sub.pnl > 0).mean())), len(sub))


for elabel, _ in ENTRY:
    d = daily[daily.entry == elabel]
    print("\n" + "=" * 70)
    print("ENTRY = %s  (per engine lot; win%% = up-days)" % elabel)
    print("-- OVERALL --")
    print("%-12s%10s%10s%7s%7s" % ("config", "totalRs", "mean/d", "win%", "nDays"))
    for cfg, _ in CONFIGS:
        t, m, w, n = summ(d[d.config == cfg])
        print("%-12s%10d%10d%6d%%%7d" % (cfg, t, m, w, n))
    print("-- BY DTE (win/mean Rs) --")
    for dte in sorted(d.dte.unique()):
        seg = " DTE%d: " % dte
        for cfg, _ in CONFIGS:
            t, m, w, n = summ(d[(d.config == cfg) & (d.dte == dte)])
            seg += "%s tot%d mean%d win%d%%(n%d)   " % (cfg, t, m, w, n)
        print(seg)
    print("-- BY WEEKDAY --")
    for wd in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        seg = " %s: " % wd[:3]
        for cfg, _ in CONFIGS:
            t, m, w, n = summ(d[(d.config == cfg) & (d.weekday == wd)])
            seg += "%s tot%d mean%d win%d%%(n%d)   " % (cfg, t, m, w, n)
        print(seg)
oc.close()
print("\nDONE")
