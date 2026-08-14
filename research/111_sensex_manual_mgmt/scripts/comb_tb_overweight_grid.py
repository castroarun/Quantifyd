"""Sec 18b: COMB x TB overweight grid. LIVE fixed 6L; COMB w in {2,4,6}, TB w in {2,4,6}.
Does COMB scale as gracefully as TB? (COMB is full-day exposure, corr~0.31-0.37 to LIVE;
TB is windowed, corr~0.18.) Live-first lab series, deployed 2L basis."""
import json
from datetime import datetime
d = json.load(open("/home/arun/quantifyd/static/app/straddles/portfolio_lab.json"))
comps = d["components"]
LIVE = dict((x[0], x[1]) for x in comps["LIVE_SUITE_6L"]["series"])
COMB = dict((x[0], x[1]) for x in comps["COMB_2L"]["series"])
TB = dict((x[0], x[1]) for x in comps["TBCSL_2L"]["series"])

def is_wed(dd): return datetime.strptime(dd, "%Y-%m-%d").weekday() == 2
ks = sorted(set(LIVE) & set(COMB) & set(TB))
ks_ex = [k for k in ks if not is_wed(k)]

def agg(f):
    c = pk = dd = 0
    for v in f: c += v; pk = max(pk, c); dd = min(dd, c - pk)
    n = len(f)
    return (round(sum(f)), round(dd), (round(sum(f) / abs(dd), 1) if dd < 0 else 99.0),
            round(100 * sum(1 for v in f if v > 0) / n), round(min(f)))

print("GRID: LIVE 6L fixed · COMB w x TB w (ex-Wed, n=%d) — NET / MAXDD / RATIO:" % len(ks_ex))
print("%-10s" % "" + "".join("%26s" % ("TB@%dL" % t) for t in (2, 4, 6)))
for cw in (2, 4, 6):
    cells = []
    for tw in (2, 4, 6):
        f = [LIVE[k] + COMB[k] * cw / 2.0 + TB[k] * tw / 2.0 for k in ks_ex]
        a = agg(f)
        cells.append("%+9d/%+7d/%5.1f" % (a[0], a[1], a[2]))
    print("%-10s" % ("COMB@%dL" % cw) + "".join("%26s" % c for c in cells))

print("\nSame grid, ALL days (n=%d):" % len(ks))
for cw in (2, 4, 6):
    cells = []
    for tw in (2, 4, 6):
        f = [LIVE[k] + COMB[k] * cw / 2.0 + TB[k] * tw / 2.0 for k in ks]
        a = agg(f)
        cells.append("%+9d/%+7d/%5.1f" % (a[0], a[1], a[2]))
    print("%-10s" % ("COMB@%dL" % cw) + "".join("%26s" % c for c in cells))

print("\nMARGINAL effect of each extra 2 lots (ex-Wed, from the 6/2/2 base):")
base = agg([LIVE[k] + COMB[k] + TB[k] for k in ks_ex])
for name, f in (("+2L TB   (6/2/4)", [LIVE[k] + COMB[k] + TB[k] * 2 for k in ks_ex]),
                ("+2L COMB (6/4/2)", [LIVE[k] + COMB[k] * 2 + TB[k] for k in ks_ex])):
    a = agg(f)
    print("  %s: net %+d (%+d) · maxDD %+d (%+d) · ratio %.1f (base %.1f) · worst day %+d" % (
        name, a[0], a[0] - base[0], a[1], a[1] - base[1], a[2], base[2], a[4]))

print("\nCOMB alone (2L): ", agg([COMB[k] for k in ks_ex]))
print("TB   alone (2L): ", agg([TB[k] for k in ks_ex]))
