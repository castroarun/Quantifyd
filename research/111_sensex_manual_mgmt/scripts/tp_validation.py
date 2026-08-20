#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the venue take-profit rule (SENSEX Rs1,667/lot) against holding to the EOD
square-off, on every recorded session — the rule that closed the SENSEX book on
2026-08-20 had never been re-tested since research/91.

Method (no look-ahead): for each recorded suite session, rebuild the straddle's combined
premium minute by minute from the recorded chain, walk it forward, and ask:
  - when (if ever) would the book's combined P&L have crossed TP = 1,667 x lots?
  - what was the same book worth at the 15:15 square-off?
Reports mean uplift/drag of TP-vs-hold, hit rate, and the tail. Writes JSON for the
weekly reassessment to read.
"""
import json, sqlite3, sys
from datetime import date, datetime
from pathlib import Path

Q = Path("/home/arun/quantifyd")
CHAIN = Q / "backtest_data/options_data.db"
OUT = Q / "static/app/straddles/tp_validation.json"
LOT = {"SENSEX": 20, "NIFTY": 65}
TP_PER_LOT = {"SENSEX": 1667.0, "NIFTY": None}   # from services/nas_portfolio_stop.py
EOD_HM = "15:15"


def sessions(venue):
    """[(day, strike, expiry, credit, lots)] from the venue's suite DBs."""
    out = {}
    dbs = ["sensex_atm_trading.db", "sensex_atm2_trading.db", "sensex_atm4_trading.db"] \
        if venue == "SENSEX" else \
        ["nas_916_atm_trading.db", "nas_916_atm2_trading.db", "nas_916_atm4_trading.db"]
    for f in dbs:
        p = Q / "backtest_data" / f
        if not p.exists():
            continue
        c = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
        c.row_factory = sqlite3.Row
        for r in c.execute("SELECT date(entry_time) d, strike, expiry_date, entry_price, qty, leg, mode "
                           "FROM nas_atm_positions WHERE entry_price IS NOT NULL"):
            d = dict(r)
            if not d["d"] or not d["strike"]:
                continue
            key = (d["d"], int(d["strike"]), d["expiry_date"])
            e = out.setdefault(key, {"credit": 0.0, "qty": 0, "legs": 0})
            e["credit"] += float(d["entry_price"] or 0)
            e["qty"] = max(e["qty"], int(d["qty"] or 0))
            e["legs"] += 1
        c.close()
    # keep sessions with complete straddles
    return [(k[0], k[1], k[2], v["credit"] / max(1, v["legs"] // 2) if v["legs"] > 2 else v["credit"],
             v["qty"], v["legs"]) for k, v in sorted(out.items()) if v["legs"] >= 2]


def minute_comb(venue, day, strike, expiry):
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    rows = c.execute(
        "SELECT substr(snapshot_time,12,5) hm, instrument_type, ltp FROM option_chain "
        "WHERE symbol=? AND strike=? AND expiry_date=? AND snapshot_time LIKE ? "
        "AND ltp IS NOT NULL ORDER BY snapshot_time",
        (venue, strike, expiry, day + "%")).fetchall()
    c.close()
    per = {}
    for hm, it, ltp in rows:
        per.setdefault(hm, {})[it] = ltp
    return [(hm, d["CE"] + d["PE"]) for hm, d in sorted(per.items())
            if "CE" in d and "PE" in d and hm <= EOD_HM]


def run(venue):
    tp_lot = TP_PER_LOT.get(venue)
    if not tp_lot:
        return {"venue": venue, "note": "no fixed TP on this venue (trailing lock instead)"}
    res = []
    for day, strike, expiry, credit, qty, legs in sessions(venue):
        series = minute_comb(venue, day, strike, expiry)
        if len(series) < 30 or not qty:
            continue
        lots = qty / LOT[venue]
        n_str = max(1, legs // 2)                 # straddles that day (systems)
        tp_rs = tp_lot * lots * n_str
        book_qty = qty * n_str
        tp_hm, tp_pnl = None, None
        for hm, comb in series:
            pnl = (credit - comb) * book_qty
            if pnl >= tp_rs:
                tp_hm, tp_pnl = hm, pnl
                break
        eod_pnl = (credit - series[-1][1]) * book_qty
        res.append({"day": day, "strike": strike, "lots": round(lots * n_str, 1),
                    "tp_threshold": round(tp_rs), "tp_hit": tp_hm is not None,
                    "tp_time": tp_hm, "tp_pnl": round(tp_pnl) if tp_pnl else None,
                    "eod_pnl": round(eod_pnl),
                    "delta": round((tp_pnl if tp_pnl is not None else eod_pnl) - eod_pnl)})
    hits = [r for r in res if r["tp_hit"]]
    deltas = [r["delta"] for r in hits]
    summary = {
        "venue": venue, "sessions": len(res), "tp_days": len(hits),
        "tp_hit_pct": round(100 * len(hits) / len(res)) if res else None,
        "mean_uplift_on_tp_days": round(sum(deltas) / len(deltas)) if deltas else None,
        "total_uplift": round(sum(deltas)) if deltas else 0,
        "worst_giveup": min(deltas) if deltas else None,
        "best_save": max(deltas) if deltas else None,
        "verdict": None, "rows": res[-40:],
    }
    if deltas:
        summary["verdict"] = ("TP EARNS ITS KEEP" if sum(deltas) > 0 else
                              "TP COSTS MONEY vs holding to 15:15")
    return summary


if __name__ == "__main__":
    out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "rule": "venue take-profit, services/nas_portfolio_stop.py VENUES[*]['tp_per_lot']",
           "venues": [run("SENSEX"), run("NIFTY")]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for v in out["venues"]:
        print(json.dumps({k: val for k, val in v.items() if k != "rows"}))
    print("wrote", OUT)
