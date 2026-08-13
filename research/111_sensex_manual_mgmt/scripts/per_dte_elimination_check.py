"""Per-DTE validation: does eliminating a weak DTE change the system ranking?
Trading-day DTE (NIFTY Tue expiry): Mon=1 Tue=0 Wed=4 Thu=3 Fri=2."""
import json
rp = json.load(open("research/111_sensex_manual_mgmt/results/nas_suite_csl_replay.json"))
nb = json.load(open("static/app/nas_baseline.json"))["days"]
dte_of = rp["dte_of"]

def agg(f):
    if not f: return None
    c = pk = dd = 0
    for v in f:
        c += v; pk = max(pk, c); dd = min(dd, c - pk)
    n = len(f)
    return dict(total=round(sum(f)), mean=round(sum(f) / n), win=round(100 * sum(1 for v in f if v > 0) / n),
                maxdd=round(dd), n=n, ratio=(round(sum(f) / abs(dd), 1) if dd < 0 else 99.0))

# live actual suite, normalized to 3 lots/system
live = {}
for d in [d for d, _ in rp["arms"]["BASE_ATM"]["series"]]:
    t = 0; got = False
    for b in nb.get(d, []):
        if "SENSEX" in b["book"]: continue
        t += (b["pnl"] / b["lots"] * 3 if b.get("lots") else b["pnl"]); got = True
    if got: live[d] = round(t)

ARMS = {
    "LIVE_SUITE(9L,actual)": live,
    "BASE_ATM(3L,model)": dict(rp["arms"]["BASE_ATM"]["series"]),
    "BASE_ATM2(3L,model)": dict(rp["arms"]["BASE_ATM2"]["series"]),
    "BASE_ATM4(3L,model)": dict(rp["arms"]["BASE_ATM4"]["series"]),
    "COMB25(3L)": dict(rp["arms"]["COMB25"]["series"]),
    "COMB30(3L)": dict(rp["arms"]["COMB30"]["series"]),
    "HYB_ATM25(3L)": dict(rp["arms"]["HYB_ATM25"]["series"]),
    "HYB_ATM4_30(3L)": dict(rp["arms"]["HYB_ATM4_30"]["series"]),
}
WD = {0: "Tue", 1: "Mon", 2: "Fri", 3: "Thu", 4: "Wed"}

print("PER-DTE TOTALS (rupees; n days in brackets; trading-day DTE, NIFTY Tue expiry):")
hdr = "%-22s" % "ARM" + "".join("%14s" % ("DTE%d/%s" % (k, WD[k])) for k in range(5)) + "%12s" % "ALL"
print(hdr)
for name, series in ARMS.items():
    cells = []
    for k in range(5):
        f = [v for d, v in series.items() if dte_of.get(d) == k]
        a = agg(f)
        cells.append("%+10d(%d)" % (a["total"], a["n"]) if a else "          -")
    a_all = agg(list(series.values()))
    print("%-22s" % name + "".join("%14s" % c for c in cells) + "%+12d" % a_all["total"])

print("\nELIMINATE-WORST-DTE re-ranking (each arm minus its own worst DTE):")
print("%-22s %7s | %-10s %10s %8s %7s | %s" % ("ARM", "ALL", "worst-DTE", "ex-TOTAL", "ex-DD", "ex-RATIO", "ex-mean/day"))
rows = []
for name, series in ARMS.items():
    per = {}
    for k in range(5):
        f = [v for d, v in series.items() if dte_of.get(d) == k]
        if f: per[k] = sum(f)
    worst = min(per, key=per.get)
    ex = [v for d, v in sorted(series.items()) if dte_of.get(d) != worst]
    a = agg(ex); a_all = agg(list(series.values()))
    rows.append((name, a_all["total"], "DTE%d/%s" % (worst, WD[worst]), a["total"], a["maxdd"], a["ratio"], a["mean"]))
for r in sorted(rows, key=lambda r: -r[5]):
    print("%-22s %+7d | %-10s %+10d %+8d %7.1f | %+d" % r)

print("\nHEAD-TO-HEAD on the SAME retained DTEs (drop DTE4/Wed for everyone):")
for name, series in ARMS.items():
    ex = [v for d, v in sorted(series.items()) if dte_of.get(d) != 4]
    a = agg(ex)
    print("  %-22s ex-Wed: total %+9d  dd %+8d  ratio %5.1f  n=%d" % (name, a["total"], a["maxdd"], a["ratio"], a["n"]))

print("\nPORTFOLIO re-check ex-Wed: LIVE(9L)+COMB30(3L) on common days:")
ks = sorted(set(live) & set(ARMS["COMB30(3L)"]))
both_all = [live[k] + ARMS["COMB30(3L)"][k] for k in ks]
both_ex = [live[k] + ARMS["COMB30(3L)"][k] for k in ks if dte_of.get(k) != 4]
for lbl, f in (("all DTEs", both_all), ("ex-Wed", both_ex)):
    a = agg(f)
    print("  %-9s total %+9d  dd %+8d  ratio %5.1f  n=%d" % (lbl, a["total"], a["maxdd"], a["ratio"], a["n"]))
