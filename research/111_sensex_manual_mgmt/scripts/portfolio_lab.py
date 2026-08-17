"""OPTIONS PORTFOLIO LAB (research/111 sec 16) -> static/app/straddles/portfolio_lab.json

Tracks the NIFTY options-portfolio stack the research converged on:
    LIVE suite (6L = 2L/system, Mon/Tue-gated) + COMB sleeve (NAS_COMB20, 2L) + TB-CSL (2L)
plus the SHIFT candidate (HYB_ATM4_30 model until NAS_C20_SHIFT accrues live days).
Live-first: paper-book records override backfill; every component reports n-days by
source (PAPER vs BACKTEST vs MODEL vs REAL+shadow). All-days AND ex-Wednesday rows.
Runs daily in the 15:40 regen so the lab stays current as OOS paper days accrue.
THE STACK DEPLOYED 2026-08-13 at 2 lots/system, ex-Wednesday (docs/THE_STACK_NIFTY_EXWED_DEPLOY_STATUS.md)."""
import json
from datetime import datetime
from pathlib import Path

Q = Path("/home/arun/quantifyd")
OUTS = [Q / "static/app/straddles/portfolio_lab.json", Q / "frontend/public/straddles/portfolio_lab.json"]
TARGET_LOTS = 2   # 2026-08-13: THE STACK deployed at 2 lots/system; normalize all mixed-size history to 2L

def load(p):
    try: return json.load(open(p))
    except Exception: return None

state = load(Q / "backtest_data/csl_paper_state.json") or {"records": []}
bf = load(Q / "static/app/csl_paper_backfill.json") or {"records": []}
nb = (load(Q / "static/app/nas_baseline.json") or {}).get("days", {})
rp = load(Q / "research/111_sensex_manual_mgmt/results/nas_suite_csl_replay.json")

def book_daily(bk):
    """{day: (pnl, source)} normalized per-record to TARGET_LOTS - live paper records take precedence over backfill."""
    out = {}
    for r in bf.get("records", []):
        if (r.get("book") or r.get("sym")) == bk:
            lots = r.get("lots") or TARGET_LOTS
            out[r["day"]] = (round(r["pnl"] * TARGET_LOTS / lots), "BACKTEST")
    for r in state.get("records", []):
        if (r.get("book") or r.get("sym")) == bk:
            lots = r.get("lots") or TARGET_LOTS
            out[r["day"]] = (round(r["pnl"] * TARGET_LOTS / lots), r.get("source", "PAPER"))
    return out

# components (deployed basis: LIVE 6L = 2L/system, each sleeve 2L)
comp = {}
live = {}
for d, rows in nb.items():
    t = 0; got = False
    for b in rows:
        if "SENSEX" in b["book"]: continue
        t += (b["pnl"] / b["lots"] * TARGET_LOTS if b.get("lots") else b["pnl"]); got = True
    if got: live[d] = (round(t), "REAL+shadow")
comp["LIVE_SUITE_6L"] = live
comp["COMB_2L"] = book_daily("NAS_COMB20")
comp["TBCSL_2L"] = book_daily("CSL_TIMEB_NIFTY")
comp["TBCSL_4L"] = {d: (v * 2, src) for d, (v, src) in comp["TBCSL_2L"].items()}   # sec-18 overweight arms
comp["TBCSL_6L"] = {d: (v * 3, src) for d, (v, src) in comp["TBCSL_2L"].items()}
comp["TBCSL_8L"] = {d: (v * 4, src) for d, (v, src) in comp["TBCSL_2L"].items()}
shift = book_daily("NAS_C20_SHIFT")
if rp:  # model history until the paper book accrues (model basis 3L -> normalize to TARGET_LOTS)
    for d, v in rp["arms"]["HYB_ATM4_30"]["series"]:
        if d not in shift: shift[d] = (round(v * TARGET_LOTS / 3.0), "MODEL")
comp["SHIFT_CAND_2L"] = shift

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

NAMES = ["LIVE_SUITE_6L", "COMB_2L", "TBCSL_2L", "SHIFT_CAND_2L"]
matrix = [[(1.0 if a == b else corr(comp[a], comp[b])) for b in NAMES] for a in NAMES]

comps_out = {}
for n2, dd in comp.items():
    if n2 in ("TBCSL_4L", "TBCSL_6L", "TBCSL_8L"): continue   # virtual sec-18 arms, not real books
    ds = sorted(dd)
    src = {}
    for d in ds: src[dd[d][1]] = src.get(dd[d][1], 0) + 1
    comps_out[n2] = {"n": len(ds), "from": ds[0] if ds else None, "to": ds[-1] if ds else None,
                     "sources": src, "stats": agg([dd[d][0] for d in ds]),
                     "series": [[d, dd[d][0]] for d in ds]}

PORTS = [
    ("LIVE suite alone", 6, ["LIVE_SUITE_6L"]),
    ("LIVE + COMB sleeve", 8, ["LIVE_SUITE_6L", "COMB_2L"]),
    ("pre-18b reference — 10L (TB@2L)", 10, ["LIVE_SUITE_6L", "COMB_2L", "TBCSL_2L"]),
    ("LIVE + SHIFT-cand + TB-CSL", 10, ["LIVE_SUITE_6L", "SHIFT_CAND_2L", "TBCSL_2L"]),
    ("ALL-CSL (COMB+SHIFT+TB, no live)", 6, ["COMB_2L", "SHIFT_CAND_2L", "TBCSL_2L"]),
    ("prior step — 14L (6/2/6)", 14, ["LIVE_SUITE_6L", "COMB_2L", "TBCSL_6L"]),
    ("THE STACK (DEPLOYED 16L ex-Wed · 6/2/8): LIVE + COMB + TB-CSL", 16, ["LIVE_SUITE_6L", "COMB_2L", "TBCSL_8L"]),
]
ports_out = []
for label, lots, parts in PORTS:
    ks = sorted(set.intersection(*[set(comp[p]) for p in parts]))
    if not ks: continue
    cs = [corr(comp[p1], comp[p2]) for i, p1 in enumerate(parts) for p2 in parts[i + 1:]]
    cs = [c for c in cs if c is not None]
    for scope, kk in (("all", ks), ("ex-Wed", [k for k in ks if not is_wed(k)])):
        ser = [[k, sum(comp[p][k][0] for p in parts)] for k in kk]
        a = agg([v for _, v in ser])
        if a:
            ports_out.append({"label": label, "lots": lots, "scope": scope, **a,
                              "corr_parts": (round(sum(cs) / len(cs), 2) if cs else None),
                              "series": ser})

out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
       "basis": "DEPLOYED 2026-08-13: LIVE suite 6L (2L/system) + COMB 2L + TB-CSL 6L = 6/2/6, real+shadow normalized per-record; "
                "per-record lots-normalized to 2L; TB-CSL from 12L history; SHIFT candidate = model until NAS_C20_SHIFT accrues; live-first merge",
       "names": NAMES, "components": comps_out, "matrix": matrix, "portfolios": ports_out,
       "verdict": "DEPLOYED 2026-08-13, stepped 6/2/8 on 2026-08-18 (16L ex-Wed, sec-18b ladder): LIVE 6L + COMB 2L + TB-CSL 6L, Wednesday (DTE4) gated off. "
                  "High complementarity (avg component corr ~0.18 ex-Wed). SHIFT-cand ex-Wed ratio is the single-arm standout but in-sample; paper book adjudicates."}
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
    print("  %-45s %2dL %-7s total %+9d dd %+8d ratio %5.1f corr %s" % (
        p2["label"][:45], p2["lots"], p2["scope"], p2["total"], p2["maxdd"], p2["ratio"], p2["corr_parts"]))
