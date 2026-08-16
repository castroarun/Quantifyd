"""STACK WEEKLY RE-ASSESSMENT -> static/app/straddles/reassessment.json
Asks "has anything CHANGED vs what we froze and sized?" - five checks with verdicts:
  1 CORR-DRIFT     rolling recent correlation vs full-window (the diversification basis)
  2 DTE-SHIFT      recent per-DTE behavior vs full-window per component (sign flips)
  3 TB-WINDOWS     frozen TB schedule vs the latest weekly sweep's best cells
  4 SIZING         is the deployed 6/2/6 grid cell still the right one on latest data?
  5 LIVE-TRACKING  sleeves' live (REAL/PAPER) days vs their backfill-model history
Cron: Fridays 16:35 IST (after the weekly sweep + daily regen). Manual:
  venv/bin/python3 research/111_sensex_manual_mgmt/scripts/stack_reassessment.py"""
import json
from datetime import datetime
from pathlib import Path

Q = Path("/home/arun/quantifyd")
OUTS = [Q / "static/app/straddles/reassessment.json", Q / "frontend/public/straddles/reassessment.json"]

def load(p):
    try: return json.load(open(p))
    except Exception: return None

lab = load(Q / "static/app/straddles/portfolio_lab.json") or {}
sweep = load(Q / "static/app/straddles/csl_best_configs.json") or {}
frozen = load(Q / "backtest_data/csl_paper_config.json") or {}
state = load(Q / "backtest_data/csl_paper_state.json") or {"records": []}

comps = lab.get("components", {})
def comp(prefix):
    for k, v in comps.items():
        if k.startswith(prefix): return k, dict((x[0], x[1]) for x in v["series"])
    return None, {}

n_live, LIVE = comp("LIVE_SUITE")
n_comb, COMB = comp("COMB")
n_tb, TB = comp("TBCSL")
WD2DTE = {0: 1, 1: 0, 2: 4, 3: 3, 4: 2}   # NIFTY weekday -> trading DTE

def dte_of(d):
    return WD2DTE.get(datetime.strptime(d, "%Y-%m-%d").weekday())

def corr(da, db, keys):
    ks = [k for k in keys if k in da and k in db]
    if len(ks) < 8: return None
    xa = [da[k] for k in ks]; xb = [db[k] for k in ks]
    ma = sum(xa) / len(xa); mb = sum(xb) / len(xb)
    num = sum((p - ma) * (q - mb) for p, q in zip(xa, xb))
    va = sum((p - ma) ** 2 for p in xa) ** 0.5; vb = sum((q - mb) ** 2 for q in xb) ** 0.5
    return round(num / (va * vb), 2) if va > 0 and vb > 0 else None

checks = []
def add(name, verdict, detail):
    checks.append({"name": name, "verdict": verdict, "detail": detail})

# ---- 1 CORR-DRIFT
all_keys = sorted(set(LIVE) & set(COMB) & set(TB))
recent = all_keys[-15:]
drift_notes = []; worst = 0.0
for lbl, a, b in (("LIVE~COMB", LIVE, COMB), ("LIVE~TB", LIVE, TB), ("COMB~TB", COMB, TB)):
    cf = corr(a, b, all_keys); cr = corr(a, b, recent)
    if cf is not None and cr is not None:
        dd = abs(cr - cf); worst = max(worst, dd)
        drift_notes.append("%s full %.2f vs recent15 %.2f" % (lbl, cf, cr))
add("CORR-DRIFT", "DRIFT" if worst > 0.25 else "OK",
    "; ".join(drift_notes) + (" | max drift %.2f (flag >0.25)" % worst))

# ---- 2 DTE-SHIFT (sign flips recent-15 vs full, n>=3 both)
flips = []
for nm, series in ((n_live, LIVE), (n_comb, COMB), (n_tb, TB)):
    for k in range(5):
        full = [v for d, v in series.items() if dte_of(d) == k]
        rec = [v for d, v in series.items() if d in recent and dte_of(d) == k]
        if len(full) >= 5 and len(rec) >= 3:
            mf = sum(full) / len(full); mr = sum(rec) / len(rec)
            if mf * mr < 0 and abs(mr) > 200:
                flips.append("%s DTE%d: full %+.0f/d -> recent %+.0f/d" % (nm, k, mf, mr))
add("DTE-SHIFT", "DRIFT" if flips else "OK",
    "; ".join(flips) if flips else "no sign flips recent-15d vs full window (per component per DTE)")

# ---- 3 TB-WINDOWS: frozen schedule vs latest sweep best
notes = []; flag = False
fro = (frozen.get("books", {}).get("CSL_TIMEB_NIFTY") or {})
cells = [c for c in (sweep.get("cells") or []) if c.get("sym") == "NIFTY"]
best = (sweep.get("best") or {}).get("NIFTY") or {}
for k, fc in sorted(fro.items()):
    b = best.get(k)
    mine = [c for c in cells if str(c["dte"]) == k and c["entry"] == fc["entry"] and c["exit"] == fc["exit"]
            and str(c["sl"]) == str(fc["sl"])]
    my_r = mine[0]["ratio"] if mine else None
    b_r = b.get("ratio") if b else None
    if my_r is not None and b_r:
        pct = round(100.0 * my_r / b_r) if b_r else None
        notes.append("DTE%s frozen %s->%s SL%s r%.1f vs sweep-best %s->%s SL%s r%.1f (%d%%)" % (
            k, fc["entry"], fc["exit"], fc["sl"], my_r, b["entry"], b["exit"], b["sl"], b_r, pct))
        if pct is not None and pct < 50: flag = True
    else:
        notes.append("DTE%s frozen %s->%s SL%s: not in latest sweep grid" % (k, fc["entry"], fc["exit"], fc["sl"]))
add("TB-WINDOWS", "DRIFT" if flag else "OK",
    "; ".join(notes) + " | flag if frozen cell <50%% of best ratio (informational - frozen config never auto-moves)")

# ---- 4 SIZING revalidation (ex-Wed grid on latest data)
ex = [k for k in all_keys if dte_of(k) != 4]
def ratio_of(f):
    c = pk = dd = 0
    for v in f: c += v; pk = max(pk, c); dd = min(dd, c - pk)
    return round(sum(f) / abs(dd), 1) if dd < 0 else 99.0
import re as _re
try:
    _src = open(str(Q / "research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py")).read()
    tb_deployed = int(_re.search(r'"CSL_TIMEB_NIFTY".*?"lots":\s*(\d+)', _src).group(1))
except Exception:
    tb_deployed = 6
tb_base = 2 if n_tb.endswith("2L") else tb_deployed   # component basis (lab normalizes to 2L)
grid = {}
for tw in (2, 4, 6):
    for cw in (2, 4):
        f = [LIVE[k] + COMB[k] * cw / 2.0 + TB[k] * tw / float(tb_base) for k in ex]
        grid["6/%d/%d" % (cw, tw)] = ratio_of(f)
deployed = "6/2/%d" % tb_deployed
best_cell = max(grid, key=grid.get)
add("SIZING", "OK" if grid.get(deployed, 0) >= 0.8 * grid[best_cell] else "DRIFT",
    "deployed %s r%.1f vs best %s r%.1f | %s" % (deployed, grid.get(deployed, 0), best_cell, grid[best_cell],
    " ".join("%s:%s" % kv for kv in sorted(grid.items()))))

# ---- 5 LIVE-TRACKING: sleeves' live days vs backfill-model means
notes = []
for bk in ("NAS_COMB20", "CSL_TIMEB_NIFTY"):
    lrec = [r for r in state.get("records", []) if r.get("book") == bk]
    if lrec:
        lots0 = lrec[0].get("lots") or 1
        lm = sum(r["pnl"] / (r.get("lots") or lots0) for r in lrec) / len(lrec)
        notes.append("%s: %d live day(s), mean %+.0f/lot" % (bk, len(lrec), lm))
    else:
        notes.append("%s: no live days yet" % bk)
add("LIVE-TRACKING", "INFO", "; ".join(notes) + " | full paper-vs-model checkpoint ~15-SEP")

out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
       "window": {"n": len(all_keys), "from": all_keys[0] if all_keys else None,
                  "to": all_keys[-1] if all_keys else None, "recent_n": len(recent)},
       "checks": checks,
       "overall": ("DRIFT" if any(c["verdict"] == "DRIFT" for c in checks) else "OK")}
for p in OUTS:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(p, "w"))
    except Exception as e:
        print("write", p, e)

print("STACK RE-ASSESSMENT %s  overall: %s" % (out["generated_at"], out["overall"]))
for c in checks:
    print("  %-13s %-6s %s" % (c["name"], c["verdict"], c["detail"][:150]))
