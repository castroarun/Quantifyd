#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/132 Stage B — entry-by-entry mis-strike + unintended-delta audit.

Two populations:
  (1) the CSL paper/live daemon records (backtest_data/csl_paper_state.json), which used
      round(index_ltp/step)*step until 019ae8f;
  (2) the NAS suite trading DBs, whose executor has carried the forward snap since
      57eb8c2 (2026-06-01) — these are the CONTROL and the natural before/after.

For each entry: the strike actually taken, the synthetic forward read off the chain at
the entry minute, the strike that forward rounds to, and the net delta of the SHORT
straddle that was actually sold, expressed as rupees per 100 index points.

READ-ONLY. Writes results/entry_audit.csv.
"""
import csv
import json
import os
import sqlite3
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common132 import (CHAIN, Q, RES, VENUE, WD, ro, load_day, read_forward, log_line,
                       hm2m, m2hm, trading_dte, tenor_years, implied_vol_straddle,
                       net_delta_short_straddle)

OUT = os.path.join(RES, "entry_audit.csv")
FG = ["src", "book", "venue", "day", "weekday", "dte_trd", "entry_hm", "entry_minute",
      "expiry", "k_actual", "spot_rec", "k_from_spot", "fwd", "k_fwd", "offset",
      "misstrike", "steps_off", "ce0", "pe0", "credit", "sigma", "T",
      "net_delta", "qty", "lots", "rs_per_100pt", "booked_pnl", "reason", "cfg", "mode"]

NAS_DBS = [
    ("nas_atm", "NIFTY"), ("nas_atm2", "NIFTY"), ("nas_atm4", "NIFTY"),
    ("nas_916_atm", "NIFTY"), ("nas_916_atm2", "NIFTY"), ("nas_916_atm4", "NIFTY"),
    ("sensex_atm", "SENSEX"), ("sensex_atm2", "SENSEX"), ("sensex_atm4", "SENSEX"),
]

_cache = {}


def day_chain(c, sym, day):
    key = (sym, day)
    if key not in _cache:
        if len(_cache) > 12:
            _cache.clear()
        _cache[key] = load_day(c, sym, day)
    return _cache[key]


def nearest_minute(chain, mi, span=6):
    """The recorded minute closest to mi (the recorder samples ~1/min but can gap)."""
    if mi in chain:
        return mi
    for d in range(1, span + 1):
        for m in (mi + d, mi - d):
            if m in chain:
                return m
    return None


def audit_one(c, sym, day, entry_hm, k_actual, qty, lots, ce_rec=None, pe_rec=None):
    """Return the audit dict for one entry, or None if the chain cannot serve it."""
    V = VENUE[sym]
    step = V["step"]
    d = day_chain(c, sym, day)
    if not d:
        return None
    fexp, spot, chain = d
    mi0 = hm2m(entry_hm)
    mi = nearest_minute(chain, mi0)
    if mi is None:
        return None
    sp = spot.get(mi)
    if sp is None:
        for dd in range(1, 7):
            sp = spot.get(mi + dd) or spot.get(mi - dd)
            if sp:
                break
    rf = read_forward(chain[mi], sp, step) if sp else None
    if rf is None:
        return None
    F, kref, spread = rf
    if spread > 0.25 * step:
        return None
    k_fwd = int(round(F / step) * step)
    dte_t = trading_dte(day, fexp)
    T = tenor_years(dte_t, mi)

    # legs actually sold, at the strike actually taken
    v = chain[mi].get(float(k_actual)) or chain[mi].get(k_actual)
    ce = pe = None
    if v:
        ce, pe = v
    if ce_rec is not None and pe_rec is not None:
        ce, pe = ce_rec, pe_rec
    credit = (ce + pe) if (ce and pe) else None
    sigma = implied_vol_straddle(F, float(k_actual), T, credit) if credit else None
    nd = net_delta_short_straddle(F, float(k_actual), T, sigma) if sigma else None
    return dict(
        venue=sym, day=day, weekday=WD[date.fromisoformat(day).weekday()], dte_trd=dte_t,
        entry_hm=m2hm(mi), entry_minute=mi, expiry=fexp, k_actual=int(k_actual),
        spot_rec=round(sp, 2) if sp else "", k_from_spot=int(round(sp / step) * step) if sp else "",
        fwd=round(F, 2), k_fwd=k_fwd, offset=round(F - sp, 2) if sp else "",
        misstrike=int(int(k_actual) != k_fwd),
        steps_off=int(round((k_fwd - int(k_actual)) / step)),
        ce0=round(ce, 2) if ce else "", pe0=round(pe, 2) if pe else "",
        credit=round(credit, 2) if credit else "",
        sigma=round(sigma, 4) if sigma else "", T=round(T, 6),
        net_delta=round(nd, 4) if nd is not None else "",
        qty=qty, lots=lots,
        rs_per_100pt=round(nd * qty * 100, 0) if nd is not None else "")


def main():
    os.makedirs(RES, exist_ok=True)
    c = ro(CHAIN)
    rows = []
    log_line("=== STAGE B: entry audit ===")

    # ---- (1) CSL daemon records ----------------------------------------------
    st = json.load(open(Q + "backtest_data/csl_paper_state.json"))
    recs = st["records"]
    ok = miss = 0
    for r in recs:
        sym = r["sym"]
        a = audit_one(c, sym, r["day"], r["entry_ts"][:5], r["strike"], r.get("qty"),
                      r.get("lots"), r.get("ce0"), r.get("pe0"))
        if a is None:
            log_line("  CSL %s %s %s: chain unusable" % (r["day"], r["book"], r["entry_ts"]))
            miss += 1
            continue
        a.update(src="CSL", book=r["book"], booked_pnl=r.get("pnl"),
                 reason=r.get("reason"), cfg=r.get("cfg"), mode=r.get("source"))
        rows.append(a)
        ok += 1
    log_line("CSL: %d audited, %d unusable (of %d records)" % (ok, miss, len(recs)))

    # ---- (2) NAS suite DBs ---------------------------------------------------
    for db, sym in NAS_DBS:
        path = Q + "backtest_data/%s_trading.db" % db
        if not os.path.exists(path):
            continue
        cc = ro(path)
        legs = {}
        for (sid, leg, tsym, qty, strike, exp, ep, et, xp, xt, xr, status, mode, espot) in cc.execute(
                "SELECT strangle_id,leg,tradingsymbol,qty,strike,expiry_date,entry_price,"
                "entry_time,exit_price,exit_time,exit_reason,status,mode,entry_spot "
                "FROM nas_atm_positions WHERE entry_time IS NOT NULL"):
            legs.setdefault(sid, {})[leg] = dict(
                tsym=tsym, qty=qty, strike=strike, exp=exp, ep=ep, et=et, xp=xp,
                xt=xt, xr=xr, status=status, mode=mode, espot=espot)
        ok = miss = 0
        for sid, d in sorted(legs.items()):
            if "CE" not in d or "PE" not in d:
                continue
            ce, pe = d["CE"], d["PE"]
            if ce["strike"] != pe["strike"]:
                continue                       # an adjusted/shifted pair, not a straddle
            et = ce["et"]
            day, hm = et[:10], et[11:16]
            if date.fromisoformat(day).weekday() >= 5:
                continue
            lot = VENUE[sym]["lot"]
            a = audit_one(c, sym, day, hm, ce["strike"], ce["qty"],
                          round(ce["qty"] / lot) if ce["qty"] else None,
                          ce["ep"], pe["ep"])
            if a is None:
                miss += 1
                continue
            # what the SPOT-only rule would have picked, from the executor's own entry_spot
            step = VENUE[sym]["step"]
            if ce["espot"]:
                a["spot_rec"] = round(ce["espot"], 2)
                a["k_from_spot"] = int(round(ce["espot"] / step) * step)
                a["offset"] = round(a["fwd"] - ce["espot"], 2)
            a.update(src="NAS", book=db, booked_pnl="", reason=ce["xr"] or "",
                     cfg="strangle_id=%s" % sid, mode=ce["mode"] or "")
            rows.append(a)
            ok += 1
        log_line("NAS %-14s: %d audited, %d unusable (%d pairs)" % (db, ok, miss, len(legs)))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FG, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log_line("STAGE B done: %d rows -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
