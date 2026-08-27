#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/132 Stage C — reconciliation + the forward-snapped counterfactual.

For every CSL record: replay the day on the 1-minute chain twice —
  ARM_ACTUAL : at the strike the daemon actually took  (this is the RECONCILIATION arm)
  ARM_FWD    : at the strike the synthetic forward rounds to (what 019ae8f would take)
each under that book's OWN rule (its window, its combined-SL shape, or the 50% backstop
for SL-none books), with the MEASURED outcome-aware cost model.

The dwell mechanic is bounded both ways: TOUCH (exit the first breaching bar) and
DWELL2 (breach on two consecutive bars, exit the third) — the live daemon polls every
5s and needs two confirming polls, so the truth sits between them on 1-minute bars.

The counterfactual is a genuine re-simulation, NOT a re-pricing: the forward-snapped
straddle collects a different credit, so its stop sits at a different level and it can
stop out on a day the real one held, and vice versa.

READ-ONLY. Writes results/replay.csv.
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common132 import (CHAIN, Q, RES, VENUE, ro, load_day, read_forward, log_line,
                       hm2m, m2hm, cost_per_lot, SESS_END_M, trading_dte)

OUT = os.path.join(RES, "replay.csv")
FG = ["book", "venue", "day", "dte_trd", "cfg", "entry_hm", "win_end", "sl_shape",
      "qty", "lots", "mode", "mgmt",
      "k_actual", "k_fwd", "misstrike", "steps_off", "fwd", "spot_rec",
      "booked_credit", "booked_exit", "booked_reason", "booked_exit_ts", "booked_pnl",
      "rep_a_credit", "rep_a_exit", "rep_a_reason", "rep_a_exit_hm", "rep_a_gross", "rep_a_net",
      "rep_f_credit", "rep_f_exit", "rep_f_reason", "rep_f_exit_hm", "rep_f_gross", "rep_f_net",
      "delta_net", "dwell_mode"]

BACKSTOP = 0.50


def parse_cfg(cfg):
    """'09:16->15:20 SL30' / 'SLnone' / 'SLrs1000' / '... SL20+trail' -> (entry, exit, shape)."""
    m = re.match(r"(\d\d:\d\d)->(\d\d:\d\d)\s+SL(\S+)", cfg or "")
    if not m:
        return None
    e0, e1, sl = m.group(1), m.group(2), m.group(3)
    sl = sl.split("+")[0]
    return e0, e1, sl


def threshold(credit, shape, lot):
    if shape == "none":
        return credit * (1.0 + BACKSTOP)
    if shape.startswith("rs"):
        return credit + float(shape[2:]) / float(lot)
    return credit * (1.0 + float(shape) / 100.0)


def replay(chain, K, m0, m1, shape, lot, dwell):
    """Sell the straddle at K at m0, cover at m1 or on the combined stop."""
    Kf = float(K)
    if m0 not in chain or Kf not in chain[m0]:
        return None
    ce0, pe0 = chain[m0][Kf]
    if not ce0 or not pe0 or ce0 <= 0 or pe0 <= 0:
        return None
    credit = ce0 + pe0
    thr = threshold(credit, shape, lot)
    streak = 0
    last_m, last_comb = m0, credit
    for mi in range(m0 + 1, m1 + 1):
        d = chain.get(mi)
        if not d or Kf not in d:
            continue
        ce, pe = d[Kf]
        if ce is None or pe is None:
            continue
        comb = ce + pe
        if streak >= dwell:                       # confirmed on prior bars -> exit HERE
            return dict(credit=credit, exit=comb, reason="SL", exit_m=mi)
        last_m, last_comb = mi, comb
        if comb >= thr:
            streak += 1
            if dwell == 1:                        # TOUCH: exit on the breaching bar
                return dict(credit=credit, exit=comb, reason="SL", exit_m=mi)
        else:
            streak = 0
    if last_m == m0:
        return None
    return dict(credit=credit, exit=last_comb, reason="TIME", exit_m=last_m)


def net_of(r, qty, lots, lot):
    gross = (r["credit"] - r["exit"]) * qty
    cost = cost_per_lot(r["credit"], r["exit"], lot, r["reason"]) * lots
    return gross, gross - cost


def main():
    os.makedirs(RES, exist_ok=True)
    c = ro(CHAIN)
    st = json.load(open(Q + "backtest_data/csl_paper_state.json"))
    recs = st["records"]
    rows = []
    cache = {}
    log_line("=== STAGE C: reconciliation + forward counterfactual ===")
    for dwell_mode, dwell in (("TOUCH", 1), ("DWELL2", 2)):
        for r in recs:
            sym = r["sym"]
            V = VENUE[sym]
            lot, step = V["lot"], V["step"]
            pc = parse_cfg(r.get("cfg"))
            if not pc:
                log_line("  SKIP unparsable cfg %r (%s %s)" % (r.get("cfg"), r["day"], r["book"]))
                continue
            _e0, e1, shape = pc
            key = (sym, r["day"])
            if key not in cache:
                if len(cache) > 12:
                    cache.clear()
                cache[key] = load_day(c, sym, r["day"])
            d = cache[key]
            if not d:
                log_line("  SKIP chain unusable %s %s" % (r["day"], r["book"]))
                continue
            fexp, spot, chain = d
            m0 = hm2m(r["entry_ts"][:5])
            if m0 not in chain:
                for dd in range(1, 7):
                    if m0 + dd in chain:
                        m0 = m0 + dd
                        break
                    if m0 - dd in chain:
                        m0 = m0 - dd
                        break
            m1 = min(hm2m(e1), SESS_END_M)
            sp = spot.get(m0)
            rf = read_forward(chain.get(m0, {}), sp, step) if sp else None
            if rf is None:
                log_line("  SKIP no forward %s %s" % (r["day"], r["book"]))
                continue
            F, _kref, spread = rf
            if spread > 0.25 * step:
                log_line("  SKIP PCP spread %.1f %s %s" % (spread, r["day"], r["book"]))
                continue
            k_act = int(r["strike"])
            k_fwd = int(round(F / step) * step)
            qty, lots = r["qty"], r["lots"]
            mgmt = "trail" if "TRAIL" in r["book"] else ("shift" if "SHIFT" in r["book"] else "")

            ra = replay(chain, k_act, m0, m1, shape, lot, dwell)
            rfw = replay(chain, k_fwd, m0, m1, shape, lot, dwell) if k_fwd != k_act else ra
            row = dict(
                book=r["book"], venue=sym, day=r["day"], dte_trd=trading_dte(r["day"], fexp),
                cfg=r.get("cfg"), entry_hm=m2hm(m0), win_end=e1, sl_shape=shape,
                qty=qty, lots=lots, mode=r.get("source"), mgmt=mgmt,
                k_actual=k_act, k_fwd=k_fwd, misstrike=int(k_act != k_fwd),
                steps_off=int(round((k_fwd - k_act) / step)),
                fwd=round(F, 2), spot_rec=round(sp, 2),
                booked_credit=r.get("credit"), booked_exit=r.get("exit_comb"),
                booked_reason=r.get("reason"), booked_exit_ts=r.get("exit_ts"),
                booked_pnl=r.get("pnl"), dwell_mode=dwell_mode)
            if ra:
                ga, na = net_of(ra, qty, lots, lot)
                row.update(rep_a_credit=round(ra["credit"], 2), rep_a_exit=round(ra["exit"], 2),
                           rep_a_reason=ra["reason"], rep_a_exit_hm=m2hm(ra["exit_m"]),
                           rep_a_gross=round(ga), rep_a_net=round(na))
            if rfw:
                gf, nf = net_of(rfw, qty, lots, lot)
                row.update(rep_f_credit=round(rfw["credit"], 2), rep_f_exit=round(rfw["exit"], 2),
                           rep_f_reason=rfw["reason"], rep_f_exit_hm=m2hm(rfw["exit_m"]),
                           rep_f_gross=round(gf), rep_f_net=round(nf))
            if ra and rfw:
                row["delta_net"] = round(nf - na)
            rows.append(row)
        log_line("  %s: %d rows" % (dwell_mode, len([x for x in rows if x["dwell_mode"] == dwell_mode])))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FG, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log_line("STAGE C done: %d rows -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
