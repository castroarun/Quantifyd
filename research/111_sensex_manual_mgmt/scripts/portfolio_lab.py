"""OPTIONS PORTFOLIO LAB (research/111 sec 16) -> static/app/straddles/portfolio_lab.json

Tracks the NIFTY options-portfolio stack the research converged on:
    LIVE suite (9L, Mon/Tue-gated) + COMB sleeve (NAS_COMB20, 3L) + TB-CSL (3L-scaled)
plus the SHIFT candidate (HYB_ATM4_30 model until NAS_C20_SHIFT accrues live days).
Live-first: paper-book records override backfill; every component reports n-days by
source (PAPER vs BACKTEST vs MODEL vs REAL+shadow). All-days AND ex-Wednesday rows.
Runs daily in the 15:40 regen so the lab stays current as OOS paper days accrue."""
import json
from datetime import datetime
from pathlib import Path

Q = Path("/home/arun/quantifyd")
OUTS = [Q / "static/app/straddles/portfolio_lab.json", Q / "frontend/public/straddles/portfolio_lab.json"]

def load(p):
    try: return json.load(open(p))
    except Exception: return None

state = load(Q / "backtest_data/csl_paper_state.json") or {"records": []}
bf = load(Q / "static/app/csl_paper_backfill.json") or {"records": []}
nb = (load(Q / "static/app/nas_baseline.json") or {}).get("days", {})
rp = load(Q / "research/111_sensex_manual_mgmt/results/nas_suite_csl_replay.json")

def book_daily(bk, scale=1.0):
    """{day: (pnl, source)} - live paper records take precedence over backfill."""
    out = {}
    for r in bf.get("records", []):
        if (r.get("book") or r.get("sym")) == bk:
            out[r["day"]] = (round(r["pnl"] * scale), "BACKTEST")
    for r in state.get("records", []):
        if (r.get("book") or r.get("sym")) == bk:
            out[r["day"]] = (round(r["pnl"] * scale), r.get("source", "PAPER"))
    return out

# components (study basis: LIVE 9L, each sleeve 3L)
comp = {}
live = {}
for d, rows in nb.items():
    t = 0; got = False
    for b in rows:
        if "SENSEX" in b["book"]: continue
        t += (b["pnl"] / b["lots"] * 3 if b.get("lots") else b["pnl"]); got = True
    if got: live[d] = (round(t), "REAL+shadow")
comp["LIVE_SUITE_9L"] = live
comp["COMB_3L"] = book_daily("NAS_COMB20")
comp["TBCSL_3L"] = book_daily("CSL_TIMEB_NIFTY", scale=3 / 12.0)
shift = book_daily("NAS_C20_SHIFT")
if rp:  # model history until the paper book accrues
    for d, v in rp["arms"]["HYB_ATM4_30"]["series"]:
        if d not in shift: shift[d] = (v, "MODEL")
comp["SHIFT_CAND_3L"] = shift

def is_wed(d):
    return datetime.strptime(d, "%Y-%m-%d").weekday() == 2

def agg(f):
    if not f: return None
    c = pk = dd = 0
    for v in f:
        c += v; pk = max(pk, c); dd = min(dd, c - pk)
    n = len(f)
    return dict(total=round(sum(f)), mean=round(sum(f) / n), maxdd=round(dd), n=n,
                ratio=(round(sum(f) / abs(dd), 1) if dd < 0 else 99.0))

def corr(da, db):
    ks = sorted(set(da) & set(db))
    if len(ks) < 8: return None
    xa = [da[k][0] for k in ks]; xb = [db[k][0] for k in ks]
    ma = sum(xa) / len(xa); mb = sum(xb) / len(xb)
    num = sum((p - ma) * (q - mb) for p, q in zip(xa, xb))
    va = sum((p - ma) ** 2 for p in xa) ** 0.5; vb = sum((q - mb) ** 2 for q in xb) ** 0.5
    return round(num / (va * vb), 2) if va > 0 and vb > 0 else None

NAMES = ["LIVE_SUITE_9L", "COMB_3L", "TBCSL_3L", "SHIFT_CAND_3L"]
matrix = [[(1.0 if a == b else corr(comp[a], comp[b])) for b in NAMES] for a in NAMES]

comps_out = {}
for n2, dd in comp.items():
    ds = sorted(dd)
    src = {}
    for d in ds: src[dd[d][1]] = src.get(dd[d][1], 0) + 1
    comps_out[n2] = {"n": len(ds), "from": ds[0] if ds else None, "to": ds[-1] if ds else None,
                     "sources": src, "stats": agg([dd[d][0] for d in ds])}

PORTS = [
    ("LIVE suite alone", 9, ["LIVE_SUITE_9L"]),
    ("LIVE + COMB sleeve", 12, ["LIVE_SUITE_9L", "COMB_3L"]),
    ("THE STACK: LIVE + COMB + TB-CSL", 15, ["LIVE_SUITE_9L", "COMB_3L", "TBCSL_3L"]),
    ("LIVE + SHIFT-cand + TB-CSL", 15, ["LIVE_SUITE_9L", "SHIFT_CAND_3L", "TBCSL_3L"]),
    ("ALL-CSL (COMB+SHIFT+TB, no live)", 9, ["COMB_3L", "SHIFT_CAND_3L", "TBCSL_3L"]),
]
ports_out = []
for label, lots, parts in PORTS:
    ks = sorted(set.intersection(*[set(comp[p]) for p in parts]))
    if not ks: continue
    cs = [corr(comp[p1], comp[p2]) for i, p1 in enumerate(parts) for p2 in parts[i + 1:]]
    cs = [c for c in cs if c is not None]
    for scope, kk in (("all", ks), ("ex-Wed", [k for k in ks if not is_wed(k)])):
        a = agg([sum(comp[p][k][0] for p in parts) for k in kk])
        if a:
            ports_out.append({"label": label, "lots": lots, "scope": scope, **a,
                              "corr_parts": (round(sum(cs) / len(cs), 2) if cs else None)})

out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
       "basis": "study basis: LIVE suite 9L (3L/system, real+shadow as-traded normalized), sleeves 3L each; "
                "TB-CSL scaled 12L->3L; SHIFT candidate = model until NAS_C20_SHIFT accrues; live-first merge",
       "names": NAMES, "components": comps_out, "matrix": matrix, "portfolios": ports_out,
       "verdict": "Converged 2026-08-13: THE STACK (LIVE+COMB+TB-CSL, 15L) - high complementarity (avg corr ~0.26). "
                  "SHIFT-cand ex-Wed ratio is the single-arm standout but in-sample; paper book adjudicates."}
for p in OUTS:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(p, "w"))
    except Exception as e:
        print("write", p, e)

print("PORTFOLIO LAB - components:")
for n2, c2 in comps_out.items():
    print("  %-15s n=%3d %s->%s  src=%s  net %+d" % (n2, c2["n"], c2["from"], c2["to"], c2["sources"], c2["stats"]["total"]))
print("\nPORTFOLIOS (all-days | ex-Wed):")
for p2 in ports_out:
    print("  %-34s %2dL %-7s total %+9d dd %+8d ratio %5.1f corr %s" % (
        p2["label"][:34], p2["lots"], p2["scope"], p2["total"], p2["maxdd"], p2["ratio"], p2["corr_parts"]))
