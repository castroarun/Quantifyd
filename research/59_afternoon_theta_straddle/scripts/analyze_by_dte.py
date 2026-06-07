"""Group the afternoon-straddle backtests by DTE and by weekday.
CSVs carry no expiry column, so DTE is RECONSTRUCTED from the NIFTY weekly-expiry
calendar: Thursday through 2025-08, Tuesday from 2025-09-01 (2024=Thu and 2026=Tue
are anchored; 2024 all-Thu known, 2026 Tue confirmed from our recorder). Expiry is
holiday-adjusted to the last trading day <= nominal, using the CSV's own date list.
DTE = trading-days from entry to the nearest expiry (expiry day = 0).
Weekday grouping is EXACT (no assumptions) and is the robust backbone."""
import csv, os
from datetime import datetime, timedelta

DL = r"C:\Users\arunc\Downloads"
FILES = {"baseline": os.path.join(DL, "Trades (9).csv"),
         "stop_0p4": os.path.join(DL, "Trades (10).csv")}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SWITCH = datetime(2025, 9, 1).date()  # Thu -> Tue

def load(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            idx = (r.get("Index") or "").strip()
            if not idx or "." in idx:
                continue
            try:
                pnl = float(r["P/L"])
            except (ValueError, KeyError):
                continue
            d = datetime.strptime(r["Entry Date"].strip(), "%Y-%m-%d").date()
            vix = None
            try:
                vix = float(r["Vix"])
            except Exception:
                pass
            rows.append({"date": d, "pnl": pnl, "vix": vix})
    return rows

# all trading dates (union of both files)
all_rows = {k: load(v) for k, v in FILES.items()}
T = sorted({r["date"] for r in all_rows["baseline"]} | {r["date"] for r in all_rows["stop_0p4"]})
Tset = set(T); pos = {d: i for i, d in enumerate(T)}

def exp_weekday(d):
    return 3 if d < SWITCH else 1  # Mon=0..Sun=6 ; Thu=3, Tue=1

# build expiry set: for each trade date's week, nominal expiry weekday date, rolled back to a trading day
def nearest_expiry(d):
    # nominal expiry in d's current week
    w = exp_weekday(d)
    delta = (w - d.weekday())
    nominal = d + timedelta(days=delta)
    if nominal < d:  # expiry weekday already passed this week -> go to next week's expiry
        nominal = nominal + timedelta(days=7)
        # next week's weekday may differ if we cross SWITCH; recompute weekday for nominal
        w2 = exp_weekday(nominal)
        nominal = nominal + timedelta(days=(w2 - nominal.weekday()))
    # holiday-adjust: last trading day <= nominal
    e = nominal
    guard = 0
    while e not in Tset and guard < 7:
        e -= timedelta(days=1); guard += 1
    # ensure e >= d; if rolled back before d, step to next week
    if e < d or e not in Tset:
        nxt = nominal + timedelta(days=7)
        w2 = exp_weekday(nxt); nxt = nxt + timedelta(days=(w2 - nxt.weekday()))
        e = nxt; guard = 0
        while e not in Tset and guard < 7:
            e -= timedelta(days=1); guard += 1
    return e

def dte(d):
    e = nearest_expiry(d)
    if e in pos and d in pos:
        return pos[e] - pos[d]
    return None

WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
lines = []
def P(s=""):
    lines.append(s); print(s)

def grp_stats(rows, keyfn):
    g = {}
    for r in rows:
        k = keyfn(r)
        if k is None:
            continue
        g.setdefault(k, []).append(r["pnl"])
    out = {}
    for k, v in g.items():
        n = len(v); tot = sum(v)
        out[k] = (n, tot, tot / n, 100 * sum(1 for x in v if x > 0) / n,
                  min(v), max(v))
    return out

P("=" * 86)
P("AFTERNOON STRADDLE 12:15->15:15  — performance grouped by DTE and weekday (10 lots)")
P("  DTE reconstructed: Thu-expiry <=2025-08, Tue-expiry >=2025-09 (2026 Tue confirmed).")
P("=" * 86)

for k in ("baseline", "stop_0p4"):
    rows = all_rows[k]
    P(f"\n############ {k.upper()}  ({len(rows)} days) ############")
    P("\n  --- by DTE (trading days to nearest weekly expiry; 0 = expiry day) ---")
    P("   DTE   days   total Rs       mean Rs    win%    worst Rs      best Rs")
    gd = grp_stats(rows, lambda r: dte(r["date"]))
    for d in sorted(gd):
        n, tot, mean, win, lo, hi = gd[d]
        P(f"    {d:>2}   {n:>4}   {tot:>11,.0f}   {mean:>9,.0f}   {win:>4.0f}   {lo:>10,.0f}  {hi:>10,.0f}")
    P("\n  --- by weekday (EXACT, no assumptions) ---")
    P("   day    days   total Rs       mean Rs    win%")
    gw = grp_stats(rows, lambda r: r["date"].weekday())
    for d in sorted(gw):
        n, tot, mean, win, lo, hi = gw[d]
        P(f"   {WD[d]:>4}   {n:>4}   {tot:>11,.0f}   {mean:>9,.0f}   {win:>4.0f}")
    # DTE x low-VIX (the real edge regime): DTE breakdown only on VIX<16 days
    P("\n  --- by DTE, VIX<16 days only (where the edge lives) ---")
    P("   DTE   days   total Rs       mean Rs    win%")
    gdv = grp_stats([r for r in rows if r["vix"] is not None and r["vix"] < 16], lambda r: dte(r["date"]))
    for d in sorted(gdv):
        n, tot, mean, win, lo, hi = gdv[d]
        P(f"    {d:>2}   {n:>4}   {tot:>11,.0f}   {mean:>9,.0f}   {win:>4.0f}")

# sanity: weekday vs DTE crosstab on baseline (should show expiry-weekday => DTE0)
P("\n" + "=" * 86)
P("SANITY: weekday -> modal DTE (Thu pre-2025-09 / Tue post should map to DTE 0)")
from collections import Counter
for period, lo, hi in [("2024 (Thu-exp)", datetime(2024,1,1).date(), datetime(2024,12,31).date()),
                       ("2026 (Tue-exp)", datetime(2026,1,1).date(), datetime(2026,12,31).date())]:
    P(f"  {period}:")
    cc = {}
    for r in all_rows["baseline"]:
        if lo <= r["date"] <= hi:
            cc.setdefault(WD[r["date"].weekday()], Counter())[dte(r["date"])] += 1
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        if wd in cc:
            modal = cc[wd].most_common(1)[0]
            P(f"     {wd}: modal DTE {modal[0]} ({modal[1]}/{sum(cc[wd].values())} days)")

with open(os.path.join(OUT, "by_dte_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n-> wrote results/by_dte_summary.txt")
