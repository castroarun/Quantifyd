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

def add(label, kind, s, note=""):
    if s:
        rows.append({"label": label, "kind": kind, "note": note, **s})

# V1 one-and-done (naked) — cum_curve
d = load_json(PUB / "v1.json")
if d and d.get("cum_curve"):
    add("V1 · intraday one-and-done (naked)", "replay", stats(from_curve(d["cum_curve"])),
        f"trigger {d.get('trigger_pct')}% · daily")

# V1 daily re-enter — per_day finals
d = load_json(PUB / "v1_daily.json")
if d and d.get("per_day"):
    add("V1 · daily re-enter (naked)", "replay",
        stats([v.get("final") for v in d["per_day"].values()]), f"trigger {d.get('trigger_pct')}% · daily")

# V2 iron-fly stop variants — trades pnl (already incl wings)
for m in ("1.5", "2.0"):
    d = load_json(PUB / f"v2_{m}.json")
    if d and d.get("trades"):
        add(f"V2 · positional iron-fly (stop {m}%)", "replay",
            stats([t["pnl"] for t in d["trades"]]), "+wings · re-enter")

# V2 naked legacy (roll only) — exit_pnl
d = load_json(BT / "straddle_v2_trades.json")
if isinstance(d, list) and d:
    add("V2 · positional bi-weekly (naked · legacy)", "live paper",
        stats([t.get("exit_pnl") for t in d]), "no move-stop · rolls to expiry")

# LIVE iron-fly executor + breakout sleeve — v2_ironfly db
try:
    c = sqlite3.connect(str(BT / "v2_ironfly_trading.db"))
    for sysname, lbl, note in [("v2", "LIVE · iron-fly executor (VIX-gated)", "real-time · combo skip-filter"),
                               ("breakout", "LIVE · inside-week breakout sleeve", "experimental")]:
        pnls = [r[0] for r in c.execute("SELECT pnl FROM v2_positions WHERE status='CLOSED' AND system=?", (sysname,)) if r[0] is not None]
        add(lbl, "live paper", stats(pnls), note)
    c.close()
except Exception as e:
    print("ironfly db:", e)

# Wed->Fri condor paper (if present)
d = load_json(APP / "condor_paper.json") or load_json(BT / "condor_paper_state.json")
if d and isinstance(d, dict):
    tr = d.get("trades") or []
    s = stats([t.get("pnl") for t in tr]) if tr else None
    add("Wed→Fri iron condor (research/80)", "live paper", s, "2 lots")


# V1 + 30% combined-premium SL (from the sl30 backtest)
d = load_json(PUB / "v1_sl30.json")
if d and d.get("trades"):
    add("V1 + 30% combined-premium SL", "backtest",
        stats([t["final"] for t in d["trades"]]), "combined-premium 30% stop · recorded chain")

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
