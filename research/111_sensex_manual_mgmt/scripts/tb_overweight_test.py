"""TB-CSL overweight test: does loading the time-blocked sleeve (short exposure windows)
improve the stack? (a) additive TB lots, (b) same-total reallocation. Uses the lab's
live-first component series at deployed basis."""
import json
d = json.load(open("/home/arun/quantifyd/static/app/straddles/portfolio_lab.json"))
comps = d["components"]

def get(*names):
    for nm in names:
        if nm in comps: return nm, dict((x[0], x[1]) for x in comps[nm]["series"])
    raise KeyError(names)

n_live, LIVE = get("LIVE_SUITE_6L", "LIVE_SUITE_9L")
n_comb, COMB = get("COMB_2L", "COMB_3L")
n_tb, TB = get("TBCSL_2L", "TBCSL_3L")
BASE_TB_LOTS = 2 if n_tb.endswith("2L") else 3
BASE_LIVE_LOTS = 6 if n_live.endswith("6L") else 9
BASE_COMB_LOTS = 2 if n_comb.endswith("2L") else 3
from datetime import datetime

def is_wed(dd): return datetime.strptime(dd, "%Y-%m-%d").weekday() == 2

ks = sorted(set(LIVE) & set(COMB) & set(TB))
ks_ex = [k for k in ks if not is_wed(k)]

def agg(f):
    c = pk = dd = 0
    for v in f: c += v; pk = max(pk, c); dd = min(dd, c - pk)
    n = len(f)
    return (round(sum(f)), round(dd), (round(sum(f) / abs(dd), 1) if dd < 0 else 99.0),
            round(sum(f) / n))

print("Components: %s / %s / %s  ·  n=%d common days (%s -> %s), ex-Wed n=%d" % (
    n_live, n_comb, n_tb, len(ks), ks[0], ks[-1], len(ks_ex)))

print("\n(a) ADDITIVE: LIVE %dL + COMB %dL + TB-CSL at w lots (total grows):" % (BASE_LIVE_LOTS, BASE_COMB_LOTS))
print("%-28s %8s | %9s %9s %6s %8s | %s" % ("STACK", "TOTAL L", "NET", "MAXDD", "RATIO", "MEAN/d", "TB share of net"))
for scope, kk in (("all", ks), ("ex-Wed", ks_ex)):
    for w in (2, 4, 6, 8, 12):
        m = w / float(BASE_TB_LOTS)
        f = [LIVE[k] + COMB[k] + TB[k] * m for k in kk]
        tb_net = sum(TB[k] * m for k in kk)
        a = agg(f)
        print("%-28s %7dL | %+9d %+9d %6.1f %+8d | %d%%" % (
            "TB@%dL %s" % (w, scope), BASE_LIVE_LOTS + BASE_COMB_LOTS + w, a[0], a[1], a[2], a[3],
            round(100 * tb_net / a[0]) if a[0] else 0))
    print()

print("(b) SAME 10L BUDGET, reallocated (LIVE scales proportionally):")
print("%-34s | %9s %9s %6s" % ("SPLIT (LIVE/COMB/TB)", "NET", "MAXDD", "RATIO"))
for lv, cb, tb in ((6, 2, 2), (5, 2, 3), (4, 2, 4), (4, 1, 5), (3, 2, 5), (2, 2, 6)):
    for scope, kk in (("ex-Wed", ks_ex),):
        f = [LIVE[k] * lv / BASE_LIVE_LOTS + COMB[k] * cb / BASE_COMB_LOTS + TB[k] * tb / BASE_TB_LOTS for k in kk]
        a = agg(f)
        print("%-34s | %+9d %+9d %6.1f" % ("%dL/%dL/%dL %s" % (lv, cb, tb, scope), a[0], a[1], a[2]))

# TB risk profile on its own days
tb_days = [TB[k] for k in ks if k in TB]
neg = sorted(tb_days)[:3]
a = agg(tb_days)
print("\nTB-CSL alone (%dL): net %+d, maxDD %+d, ratio %s · worst 3 days: %s" % (
    BASE_TB_LOTS, a[0], a[1], a[2], neg))
print("TB exposure: ~1-2h windows/day vs ~6h for COMB/suite (time-in-market per lot ~3x lower)")
