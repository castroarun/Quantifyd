"""Does a PORTFOLIO-level SL / profit-trail on THE STACK's sleeves add value beyond
each book's own CSL? Test on synchronized per-minute intraday curves (backfill,
NAS_COMB20 + CSL_TIMEB_NIFTY scaled to 3L) over all common days.
Resolution caveat: minute-sampled curves of the 3-sec replay (fallback per the data
ladder; portfolio stop tested on minute marks). LIVE suite excluded (no intraday
history yet - nas_mtm snapshots only started accruing; revisit in a few weeks).
Arms: NONE / PSTOP at -X per lot (sweep) / TRAIL lock: after peak >= +A/lot, stop at
peak - B/lot / PSTOP+TRAIL combined."""
import json
from datetime import datetime

BF = json.load(open("/home/arun/quantifyd/static/app/csl_paper_backfill.json"))
LOTS = 6  # sleeve pair basis: COMB 3L + TB-CSL 3L(scaled)

def get(bk, scale=1.0):
    out = {}
    for r in BF["records"]:
        if r.get("book") == bk and r.get("series") and len(r["series"]) > 1:
            out[r["day"]] = [(t, v * scale) for t, v in r["series"]]
    return out

comb = get("NAS_COMB20")
tb = get("CSL_TIMEB_NIFTY", scale=3 / 12.0)
days = sorted(set(comb) & set(tb))

def merged(day):
    """minute -> combined MTM (each book forward-filled, frozen at its final after exit)"""
    a, b = comb[day], tb[day]
    times = sorted({t for t, _ in a} | {t for t, _ in b})
    out = []
    ia = ib = 0; va = vb = 0.0
    fa = a[-1][1]; fb = b[-1][1]
    for t in times:
        while ia < len(a) and a[ia][0] <= t: va = a[ia][1]; ia += 1
        while ib < len(b) and b[ib][0] <= t: vb = b[ib][1]; ib += 1
        ea = fa if t > a[-1][0] else va
        eb = fb if t > b[-1][0] else vb
        out.append((t, ea + eb))
    return out

def run_day(curve, pstop=None, trail=None):
    """pstop: rupees (negative). trail: (arm_at, giveback) rupees. Exit freezes P&L."""
    peak = -1e18
    for t, v in curve:
        peak = max(peak, v)
        if pstop is not None and v <= pstop:
            return v          # stop at this minute's mark
        if trail is not None:
            arm, give = trail
            if peak >= arm and v <= peak - give:
                return v
    return curve[-1][1]

def agg(f):
    c = pk = dd = 0
    for v in f: c += v; pk = max(pk, c); dd = min(dd, c - pk)
    n = len(f)
    return (round(sum(f)), round(sum(f) / n), round(100 * sum(1 for v in f if v > 0) / n),
            round(dd), (round(sum(f) / abs(dd), 1) if dd < 0 else 99.0))

curves = {d: merged(d) for d in days}
ex_wed = [d for d in days if datetime.strptime(d, "%Y-%m-%d").weekday() != 2]

print("SLEEVE-PAIR (COMB 3L + TB-CSL 3L = %dL) portfolio-overlay test, n=%d days (%s -> %s)" % (LOTS, len(days), days[0], days[-1]))
for scope, dset in (("ALL DAYS", days), ("EX-WED", ex_wed)):
    print("\n%s (n=%d):" % (scope, len(dset)))
    print("%-34s %9s %7s %5s %9s %6s" % ("OVERLAY", "TOTAL", "MEAN/d", "WIN%", "MAXDD", "RATIO"))
    rows = [("none (books' own CSLs only)", None, None)]
    for x in (500, 800, 1300, 2000):
        rows.append(("pstop -%d/lot (-%d)" % (x, x * LOTS), -x * LOTS, None))
    for a_, g_ in ((500, 250), (500, 500), (1000, 500), (1000, 1000)):
        rows.append(("trail: arm +%d/lot, give %d/lot" % (a_, g_), None, (a_ * LOTS, g_ * LOTS)))
    rows.append(("pstop -1300/lot + trail 1000/500", -1300 * LOTS, (1000 * LOTS, 500 * LOTS)))
    for name, ps, tr in rows:
        f = [run_day(curves[d], ps, tr) for d in dset]
        print("%-34s %+9d %+7d %4d%% %+9d %6.1f" % ((name,) + agg(f)))
