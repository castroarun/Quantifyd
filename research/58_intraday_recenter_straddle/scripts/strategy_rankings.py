"""Straddle STRATEGY LEADERBOARD — rates every system on /app/straddles by performance
to date (net, win%, maxDD, Calmar) and assigns a provisional risk-adjusted grade.
Reads all live/replay data sources; writes rankings.json. Run weekly by cron.
NOTE: single-regime (since 2026-04) => grades are PROVISIONAL SIGNALs, not validated."""
import json, sqlite3, datetime
from pathlib import Path

ROOT = Path("/home/arun/quantifyd")
PUB = ROOT / "frontend/public/straddles"
APP = ROOT / "static/app/straddles"
BT = ROOT / "backtest_data"


def stats(pnls):
    pnls = [p for p in pnls if p is not None]
    n = len(pnls)
    if not n:
        return None
    net = sum(pnls); wins = sum(1 for x in pnls if x > 0)
    cum = peak = mdd = 0
    for x in pnls:
        cum += x; peak = max(peak, cum); mdd = min(mdd, cum - peak)
    return dict(net=round(net), n=n, win=round(100 * wins / n), mean=round(net / n),
                best=round(max(pnls)), worst=round(min(pnls)), maxdd=round(mdd),
                calmar=round(net / abs(mdd), 2) if mdd < 0 else None)


def from_curve(curve):
    out, prev = [], 0
    for _, c in curve:
        out.append(c - prev); prev = c
    return out


def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


rows = []
DAILY = {}   # label -> {day: pnl} for correlation-vs-book

def reg(label, pairs):
    dd = {}
    for d, v in pairs:
        if d and v is not None:
            dd[str(d)[:10]] = dd.get(str(d)[:10], 0) + v
    if len(dd) >= 5:
        DAILY[label] = dd

def tdates(trs, *keys):
    out = []
    for t in trs:
        for k in keys:
            v = t.get(k)
            if v:
                out.append(str(v)[:10]); break
    return out

def add(label, kind, s, note="", dates=None):
    if s:
        if dates:
            ds = sorted(str(d)[:10] for d in dates if d)
            if ds: s = {**s, "from": ds[0], "to": ds[-1]}
        rows.append({"label": label, "kind": kind, "note": note, **s})

# V1 one-and-done (naked) — cum_curve
d = load_json(PUB / "v1.json")
if d and d.get("cum_curve"):
    add("V1 · intraday one-and-done (naked)", "backtest", stats(from_curve(d["cum_curve"])),
        f"trigger {d.get('trigger_pct')}% · daily", dates=[p[0] for p in d["cum_curve"]])
    reg("V1 · intraday one-and-done (naked)", zip([p[0] for p in d["cum_curve"]], from_curve(d["cum_curve"])))

# V1 daily re-enter — per_day finals
d = load_json(PUB / "v1_daily.json")
if d and d.get("per_day"):
    add("V1 · daily re-enter (naked)", "backtest",
        stats([v.get("final") for v in d["per_day"].values()]), f"trigger {d.get('trigger_pct')}% · daily", dates=list(d["per_day"].keys()))
    reg("V1 · daily re-enter (naked)", [(k, v.get("final")) for k, v in d["per_day"].items()])

# V2 iron-fly stop variants — trades pnl (already incl wings)
for m in ("1.5", "2.0"):
    d = load_json(PUB / f"v2_{m}.json")
    if d and d.get("trades"):
        add(f"V2 · positional iron-fly (stop {m}%)", "backtest",
            stats([t["pnl"] for t in d["trades"]]), "+wings · re-enter", dates=tdates(d["trades"], "exit_date", "exit", "date", "day", "entry_date", "entry"))
        reg(f"V2 · positional iron-fly (stop {m}%)", zip(tdates(d["trades"], "exit_date", "exit", "date", "day", "entry_date", "entry"), [t["pnl"] for t in d["trades"]]))

# V2 naked legacy (roll only) — exit_pnl
d = load_json(BT / "straddle_v2_trades.json")
if isinstance(d, list) and d:
    add("V2 · positional bi-weekly (naked · legacy)", "live-paper",
        stats([t.get("exit_pnl") for t in d]), "no move-stop · rolls to expiry", dates=tdates(d, "exit_date", "exit_time", "exit", "entry_date", "date"))
    reg("V2 · positional bi-weekly (naked · legacy)", zip(tdates(d, "exit_date", "exit_time", "exit", "entry_date", "date"), [t.get("exit_pnl") for t in d]))

# LIVE iron-fly executor + breakout sleeve — v2_ironfly db
try:
    c = sqlite3.connect(str(BT / "v2_ironfly_trading.db"))
    for sysname, lbl, note in [("v2", "LIVE · iron-fly executor (VIX-gated)", "real-time · combo skip-filter"),
                               ("breakout", "LIVE · inside-week breakout sleeve", "experimental")]:
        pr = [(r[0], r[1]) for r in c.execute("SELECT pnl, entry_time FROM v2_positions WHERE status='CLOSED' AND system=?", (sysname,)) if r[0] is not None]
        pnls = [p_ for p_, _ in pr]
        add(lbl, "live-paper", stats(pnls), note, dates=[d_ for _, d_ in pr if d_])
        reg(lbl, [(d_, p_) for p_, d_ in pr if d_])
    c.close()
except Exception as e:
    print("ironfly db:", e)

# Wed->Fri condor paper (if present)
d = load_json(APP / "condor_paper.json") or load_json(BT / "condor_paper_state.json")
if d and isinstance(d, dict):
    tr = d.get("trades") or []
    s = stats([t.get("pnl") for t in tr]) if tr else None
    add("Wed→Fri iron condor (research/80)", "live-paper", s, "2 lots", dates=tdates(tr, "day", "date", "entry_date", "entry"))
    reg("Wed→Fri iron condor (research/80)", zip(tdates(tr, "day", "date", "entry_date", "entry"), [t.get("pnl") for t in tr]))


# V1 + 30% combined-premium SL (from the sl30 backtest)
d = load_json(PUB / "v1_sl30.json")
if d and d.get("trades"):
    add("V1 + 30% combined-premium SL", "backtest",
        stats([t["final"] for t in d["trades"]]), "combined-premium 30% stop · recorded chain", dates=tdates(d["trades"], "day", "date"))
    reg("V1 + 30% combined-premium SL", zip(tdates(d["trades"], "day", "date"), [t["final"] for t in d["trades"]]))

# NAS live books (as-traded lots) — per-book rows so the whole short-vol book ranks in ONE table
import sqlite3
LOTS_BY_MONTH = {"2026-03": 1, "2026-04": 5, "2026-05": 1, "2026-06": 1, "2026-07": 2, "2026-08": 3}
for name, lbl in (("nas_916_atm", "NAS 916 ATM (NIFTY · live)"),
                  ("nas_916_atm2", "NAS 916 ATM2 (NIFTY · live)"),
                  ("nas_916_atm4", "NAS 916 ATM4 (NIFTY · live)")):
    try:
        c = sqlite3.connect(str(BT / (name + "_trading.db"))); c.row_factory = sqlite3.Row
        cols = [x[1] for x in c.execute("PRAGMA table_info(nas_atm_trades)")]
        pcol = "pnl" if "pnl" in cols else "net_pnl"
        dcol = "entry_time" if "entry_time" in cols else "created_at"
        daily = {}
        for r in c.execute("SELECT %s d,%s p FROM nas_atm_trades WHERE %s IS NOT NULL" % (dcol, pcol, pcol)):
            dd = str(r["d"])[:10]
            daily[dd] = daily.get(dd, 0) + r["p"]
        add(lbl, "live-real", stats([daily[k] for k in sorted(daily)]), "as-traded 1→3 lots · per-leg SL + trail", dates=sorted(daily))
        reg(lbl, daily.items())
        c.close()
    except Exception as e:
        print(name, e)
try:
    c = sqlite3.connect(str(BT / "sensex_atm2_trading.db"))
    daily = {}
    for dd, p in c.execute("SELECT trade_date, net_pnl FROM nas_atm_trades WHERE net_pnl IS NOT NULL"):
        daily[str(dd)[:10]] = daily.get(str(dd)[:10], 0) + p
    add("NAS ATM2 (SENSEX · live)", "live-paper", stats([daily[k] for k in sorted(daily)]), "as-traded lots · young book", dates=sorted(daily))
    reg("NAS ATM2 (SENSEX · live)", daily.items())
    c.close()
except Exception as e:
    print("sensex", e)

# TB-CSL (time-blocked CSL) best-config books — backtest rows from the Lab JSON
d = load_json(PUB / "csl_best_configs.json")
if d and d.get("best"):
    for sym, lbl, lots in (("NIFTY", "TB-CSL best-config book (NIFTY · backtest)", 10),
                            ("SENSEX", "TB-CSL best-config book (SENSEX · backtest)", 5)):
        daily = {}
        for k, b in d["best"].get(sym, {}).items():
            for dd, v in b.get("series", []):
                daily[dd] = daily.get(dd, 0) + v
        if daily:
            add(lbl, "backtest", stats([daily[k2] for k2 in sorted(daily)]),
                "%d lots · time-blocked windows · IN-SAMPLE optimized (see walk-forward caveat)" % lots, dates=sorted(daily))
            reg(lbl, daily.items())

# CSL paper books (frozen-config validation) — appear once they have records
d = load_json(BT / "csl_paper_state.json")
if d and d.get("records"):
    for sym, lbl, lots in (("NIFTY", "CSL Paper (NIFTY · 12 lots)", 12), ("SENSEX", "CSL Paper (SENSEX · 6 lots)", 6)):
        rs = [r for r in d["records"] if r.get("sym") == sym or r.get("book", "").endswith(sym)]
        if rs:
            add(lbl, "live-paper", stats([r["pnl"] for r in rs]), "frozen 13-AUG config · OOS validation", dates=[r["day"] for r in rs])
            reg(lbl, [(r["day"], r["pnl"]) for r in rs])

# correlation of each system's daily P&L vs the REST of the book (sum of all other systems)
def _pearson(da, db):
    ks = sorted(set(da) & set(db))
    if len(ks) < 8: return None
    xa = [da[k] for k in ks]; xb = [db[k] for k in ks]
    ma = sum(xa) / len(xa); mb = sum(xb) / len(xb)
    num = sum((p - ma) * (q - mb) for p, q in zip(xa, xb))
    va = sum((p - ma) ** 2 for p in xa) ** 0.5
    vb = sum((q - mb) ** 2 for q in xb) ** 0.5
    return round(num / (va * vb), 2) if va > 0 and vb > 0 else None

book = {}
for _lbl, _dd in DAILY.items():
    for _d, _v in _dd.items():
        book[_d] = book.get(_d, 0) + _v
for r in rows:
    dd = DAILY.get(r["label"])
    if dd:
        rest = {d2: book[d2] - dd.get(d2, 0) for d2 in dd if d2 in book}
        rest = {d2: v2 for d2, v2 in rest.items() if abs(v2) > 1e-9}
        r["corr"] = _pearson(dd, rest)

# links: card anchor + backtest report + tearsheet per system (rendered as icons on the page)
LINKS = {
    "V1 · intraday one-and-done (naked)": {"anchor": "live-box"},
    "V1 · daily re-enter (naked)": {"anchor": "live-box"},
    "V2 · positional iron-fly (stop 1.5%)": {"anchor": "variant-lab"},
    "V2 · positional iron-fly (stop 2.0%)": {"anchor": "variant-lab"},
    "V2 · positional bi-weekly (naked · legacy)": {"anchor": "live-box"},
    "LIVE · iron-fly executor (VIX-gated)": {"anchor": "v2-engine"},
    "LIVE · inside-week breakout sleeve": {"anchor": "v2-engine"},
    "V1 + 30% combined-premium SL": {"anchor": "sl30-card", "report": "/app/backtest/csl-best-config-straddles", "chart": "/app/csl30_vs_nas916.png"},
    "Wed→Fri iron condor (research/80)": {"anchor": "condor", "report": "/app/backtest/fardte-rescue"},
    "NAS 916 ATM (NIFTY · live)": {"page": "/app/nas", "report": "/app/backtest/sensex-nifty-stop-by-dte", "chart": "/app/nifty_csl_vs_nas.png"},
    "NAS 916 ATM2 (NIFTY · live)": {"page": "/app/nas", "report": "/app/backtest/sensex-nifty-stop-by-dte"},
    "NAS 916 ATM4 (NIFTY · live)": {"page": "/app/nas", "report": "/app/backtest/sensex-nifty-stop-by-dte"},
    "NAS ATM2 (SENSEX · live)": {"page": "/app/nas", "report": "/app/backtest/sensex-nifty-stop-by-dte", "chart": "/app/sensex_csl_vs_nas.png"},
    "CSL Paper (NIFTY · 12 lots)": {"anchor": "csl-paper", "report": "/app/backtest/csl-best-config-straddles", "chart": "/app/nifty_csl_vs_nas.png"},
    "CSL Paper (SENSEX · 6 lots)": {"anchor": "csl-paper", "report": "/app/backtest/csl-best-config-straddles", "chart": "/app/sensex_csl_vs_nas.png"},
    "NAS-COMB20 Paper (NIFTY · 3 lots)": {"anchor": "csl-paper", "report": "/app/backtest/sensex-nifty-stop-by-dte", "chart": "/app/perleg_vs_comb.png"},
    "TB-CSL best-config book (NIFTY · backtest)": {"anchor": "csl-lab", "report": "/app/backtest/csl-best-config-straddles", "chart": "/app/nifty_csl_vs_nas.png"},
    "TB-CSL best-config book (SENSEX · backtest)": {"anchor": "csl-lab", "report": "/app/backtest/csl-best-config-straddles", "chart": "/app/sensex_csl_vs_nas.png"},
}

# in-page anchors so the leaderboard links jump to each system's section/card
ANCHOR = {
    "V1 · intraday one-and-done (naked)": "live-box",
    "V1 · daily re-enter (naked)": "live-box",
    "V2 · positional iron-fly (stop 1.5%)": "variant-lab",
    "V2 · positional iron-fly (stop 2.0%)": "variant-lab",
    "V2 · positional bi-weekly (naked · legacy)": "live-box",
    "LIVE · iron-fly executor (VIX-gated)": "v2-engine",
    "LIVE · inside-week breakout sleeve": "v2-engine",
    "V1 + 30% combined-premium SL": "sl30-card",
    "Wed→Fri iron condor (research/80)": "condor",
}


def grade(r):
    if r["net"] <= 0:
        return "F" if r["net"] < -40000 else "D"
    c = r.get("calmar") or 0
    return "A" if c >= 3 else "B" if c >= 2 else "C" if c >= 1 else "D"

def confidence(r):
    n = r["n"]
    return "very low" if n < 6 else "low" if n < 12 else "medium"  # single-regime caps at medium

for r in rows:
    r["grade"] = grade(r)
    r["confidence"] = confidence(r)
    r["anchor"] = ANCHOR.get(r["label"])
    for kk, vv in LINKS.get(r["label"], {}).items():
        r[kk] = vv
# rank: positive-net first, by Calmar desc (None last), then net
rows.sort(key=lambda r: (r["net"] > 0, r.get("calmar") if r.get("calmar") is not None else -999, r["net"]), reverse=True)
for i, r in enumerate(rows, 1):
    r["rank"] = i

payload = {
    "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "cadence": "weekly (Sat) · daily data refresh",
    "caveat": "Single regime since 2026-04 · small samples · recorded-chain/paper (modeled fills). Grades are PROVISIONAL SIGNALs, not validated across regimes.",
    "systems": rows,
}
for o in (APP, PUB):
    o.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(o / "rankings.json", "w"))
print("wrote rankings.json:")
for r in rows:
    print(f"  #{r['rank']} [{r['grade']}] {r['label']:44s} net={r['net']:+9d} calmar={r.get('calmar')} maxDD={r['maxdd']:+9d} n={r['n']} conf={r['confidence']}")
