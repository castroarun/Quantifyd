#!/usr/bin/env python3
"""VPS nightly regen: travel page = replay sim for HISTORICAL weeks + the paper
engine's ACTUAL fills for LIVE weeks (>= 2026-07-27). research/90 item-5 fix
2026-07-29: live weeks previously came from the sim and diverged from the book."""
import csv
import json
import re

BASE = "/home/arun/quantifyd"
RES = BASE + "/research/90_nifty_strangle_rules/results"
TPL = BASE + "/research/90_nifty_strangle_rules/travel_template.html"
OUTS = [BASE + "/static/app/nsrw-travel-research90.html",
        BASE + "/frontend/public/nsrw-travel-research90.html"]
LIVE_FROM = "2026-07-27"
LOT_QTY = 650.0
BOOK_T = {"nsrw_paper.json": 30, "nsrw_paper_t20.json": 20}

# ---- historical weeks from the replay sim (pre-live only)
paths = json.load(open(RES + "/replay_paths.json"))
rows = {(r["entry_time"][:10], int(r["T"])): r
        for r in csv.DictReader(open(RES + "/replay_nsrw_cycles.csv"))}
cycles = []
for c in paths:
    if c["week"] >= LIVE_FROM:
        continue
    r = rows.get((c["week"], c["T"]))
    if r:
        c["net"] = float(r["net"])
        c["entry_time"] = r["entry_time"]
        c["exit"] = r["exit_time"]
        c["reason"] = r["exit_reason"]
    else:
        c["net"] = c["gross"]
    cycles.append(c)


def ev_label(tag, rest):
    if tag.startswith("PT"):
        return "PROFIT-TAKE"
    if tag.startswith("TIME"):
        return "TIME-EXIT"
    if tag.startswith(("RESTRANGLE", "RECENTER")):
        return f"ROLL {tag} {rest}".strip()
    return f"{tag} {rest}".strip()


def from_book(cyc, T, closed_reason=None, net_rs=None):
    """Engine cycle/history row -> template cycle dict (pts, template time format)."""
    events = []
    for e in cyc.get("events", []):
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) (\S+)(.*)", e)
        if m:
            events.append({"t": m.group(1).replace(" ", "T") + ":00",
                           "label": ev_label(m.group(2), m.group(3).strip())})
    legs = cyc.get("legs", [])
    pe = next((l for l in legs if l.get("side") == "PE"), {})
    ce = next((l for l in legs if l.get("side") == "CE"), {})
    path = [[p[0].replace(" ", "T") + ":00", round(p[1] / LOT_QTY, 2)]
            for p in cyc.get("path_week", [])]
    entry_t = (cyc.get("entry_time") or (events[0]["t"][:16].replace("T", " ") if events else cyc.get("entry_date", "") + " 15:14")).replace(" ", "T")
    if len(entry_t) == 16:
        entry_t += ":00"
    m2m = cyc.get("m2m_rs", 0) or 0
    is_open = closed_reason is None
    net_pts = round((m2m if is_open else (net_rs or 0)) / LOT_QTY, 2)
    if not path:  # pre-deploy weeks have no accumulated path: synthesize
        last_t = (cyc.get("closed") or cyc.get("mtm_date", cyc.get("entry_date", "")) + " 15:15").replace(" ", "T")
        if len(last_t) == 16:
            last_t += ":00"
        path = [[entry_t, 0.0], [last_t, net_pts]]
    exit_t = (cyc.get("closed") or (path[-1][0][:16].replace("T", " "))).replace(" ", "T")
    if len(exit_t) == 16:
        exit_t += ":00"
    return {
        "week": cyc.get("entry_date"), "T": T, "expiry": cyc.get("expiry"),
        "credit": cyc.get("credit0"), "spot0": cyc.get("spot0", 0),
        "pe": f"{int(pe.get('strike', 0))}PE@{pe.get('entry', 0)}",
        "ce": f"{int(ce.get('strike', 0))}CE@{ce.get('entry', 0)}",
        "events": events, "path": path,
        "entry_time": entry_t.replace("T", " ")[:16].replace(" ", "T"),
        "exit": exit_t, "reason": (closed_reason or "OPEN"),
        "gross": net_pts, "net": net_pts,
    }


for state_name, T in BOOK_T.items():
    try:
        st = json.load(open(BASE + "/backtest_data/" + state_name))
    except Exception:
        continue
    for h in st.get("history", []):
        if (h.get("week") or "") >= LIVE_FROM:
            cycles.append(from_book(h, T, closed_reason=h.get("reason"), net_rs=h.get("net_rs")))
    cur = st.get("cycle")
    if cur and (cur.get("entry_date") or "") >= LIVE_FROM:
        cycles.append(from_book(cur, T))

html = open(TPL, encoding="utf-8").read().replace(
    "/*__DATA__*/[]", json.dumps(cycles, separators=(",", ":"), default=str))
for o in OUTS:
    open(o, "w", encoding="utf-8").write(html)
print(f"deployed {len(cycles)} cycles ({sum(1 for c in cycles if (c.get('week') or '') >= LIVE_FROM)} live-book)")
