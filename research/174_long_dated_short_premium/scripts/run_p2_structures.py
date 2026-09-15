#!/usr/bin/env python3
"""
research/174 â€” P2: structure bake-off. Straddle vs strangle vs iron condor vs winged
straddle, at each surviving tenor, ranked on return per rupee of MEASURED margin.

Two reasons a condor can win even on a smaller credit:
  1. the long wings cut the Kite margin requirement a long way (measured today: at
     196 DTE a 2.5%-body / 7%-wing condor blocks Rs 1.61L per lot against Rs 3.21L for
     the naked straddle), so the same rupee of capital carries more structures;
  2. it caps the tail, which is the thing a long-dated short premium book cannot
     otherwise control, because there is no intraday data to stop on.

Cost realism: slippage is charged per LEG on gross premium turnover
(`engine_lt.costs_points_legs`), not on the net credit. Charging a condor's slippage on
its net credit is the classic way to make a four-leg structure look cheap; it crosses
four spreads regardless of how small the net is.

Daily close only. research/127 prior carried over: body 2.5% out, wings 7%.

Writes results/p2_structures.csv â€” one row per (tenor, exit, structure, target, slip).
"""
import csv
import json
import math
import os
import statistics as st
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_lt as E                                        # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
TAG = os.environ.get("R174_TAG", "")
OUT = RES / ("p2_structures%s.csv" % TAG)

SYMBOL = os.environ.get("R174_SYMBOL", "NIFTY")
UNDER = {"NIFTY": "NIFTY50", "BANKNIFTY": "BANKNIFTY"}[SYMBOL]
START = os.environ.get("R174_START", "2015-01-01")
VOL_FLOOR = int(os.environ.get("R174_VF", "25"))
SLIPS = [0.0025, 0.0075, 0.015]
TARGETS = [0.50, 9.99]
ARMS = [tuple(int(x) for x in a.split(":")) for a in os.environ.get(
    "R174_ARMS", "45:21,60:27,90:40,120:54,180:81,240:108").split(",")]

STRUCTS = [("STR", dict(kind="STR"))]
for b in (0.015, 0.025, 0.035, 0.05):
    STRUCTS.append(("STG%.1f" % (b * 100), dict(kind="STG", body=b)))
for b in (0.025, 0.05):
    for w in (0.03, 0.05, 0.07):
        STRUCTS.append(("IC%.1fw%.0f" % (b * 100, w * 100), dict(kind="IC", body=b, wing=w)))
for w in (0.05, 0.07, 0.10):
    STRUCTS.append(("WS%.0f" % (w * 100), dict(kind="WS", wing=w)))

MJ = json.load(open(RES / "margin_by_tenor.json"))
MCOL = {"STR": "STR", "STG": "STG2.5", "IC": "IC2.5w7", "WS": "WS7"}


def margin_at(dte, col):
    pts = sorted((r["dte"], r.get(col)) for r in MJ["rows"] if r.get(col))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if not xs:
        return None
    if dte <= xs[0]:
        return ys[0]
    if dte >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if dte <= xs[i]:
            f = (dte - xs[i - 1]) / float(xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


FIELDS = ["tenor", "exit_dte", "struct", "kind", "target", "slip", "n", "n_indep",
          "first", "last", "win_rate", "avg_credit", "avg_gross_prem", "avg_days",
          "avg_net", "sd_net", "t_stat", "total_net", "worst", "best", "max_dd",
          "pts_per_lotyear", "entry_margin", "ret_pct_yr", "avg_cost", "n_target",
          "med_entry_vol"]


def indep_count(rows):
    n, last = 0, ""
    for r in sorted(rows, key=lambda x: x["exit_date"]):
        if r["entry_date"] >= last:
            n += 1
            last = r["exit_date"]
    return n


def main():
    con = E.connect()
    days = E.sessions(con, SYMBOL, "2011-01-01")
    cat = E.expiry_catalogue(con, SYMBOL)
    spot = E.spot_series(con, UNDER)
    print("arms=%s structs=%d" % (ARMS, len(STRUCTS)), flush=True)

    of = open(OUT, "w", newline="")
    w = csv.DictWriter(of, fieldnames=FIELDS)
    w.writeheader()

    for T, X in ARMS:
        cands = []
        for e in sorted(ex for ex in cat if ex >= START):
            if cat[e]["horizon"] < T + 5:
                continue
            ed = E.prev_session(days, E.dstr(E.dparse(e) - timedelta(days=T)))
            xd = E.prev_session(days, E.dstr(E.dparse(e) - timedelta(days=X)))
            if ed and xd and ed > "" and ed >= START and ed >= cat[e]["first_seen"] \
                    and xd > ed and ed in spot:
                cands.append((e, ed, xd))
        if not cands:
            print("tenor %d: no candidates" % T, flush=True)
            continue

        results = {}
        for e, ed, xd_t in cands:
            chain = E.expiry_chain(con, SYMBOL, e)
            if ed not in chain:
                continue
            s0 = spot[ed]
            for name, spec in STRUCTS:
                pos = E.build_position(chain[ed], s0, spec, VOL_FLOOR)
                if pos is None:
                    continue
                entry_px = []
                gross_in = 0.0
                for k, side, sign in pos["legs"]:
                    p = E.leg_price(chain[ed], k, side)
                    entry_px.append((p, sign))
                    gross_in += abs(p)
                credit = pos["credit"]
                evol = sum(chain[ed][k][side][1] for k, side, _ in pos["legs"])
                for target in TARGETS:
                    hit = None
                    for d in sorted(chain):
                        if d <= ed:
                            continue
                        m = E.mark(chain[d], pos["legs"], VOL_FLOOR)
                        if m is None:
                            continue
                        if target <= 5 and m <= target * credit:
                            hit = (d, m, "TARGET")
                            break
                        if d >= xd_t:
                            hit = (d, m, "TIME")
                            break
                    if hit is None:
                        continue
                    d, m, reason = hit
                    exit_px = [(E.leg_price(chain[d], k, side), sign)
                               for k, side, sign in pos["legs"]]
                    if any(p is None for p, _ in exit_px):
                        continue
                    results.setdefault((name, target), []).append(dict(
                        expiry=e, entry_date=ed, exit_date=d, credit=credit,
                        gross_in=gross_in, exit_cost=m, reason=reason,
                        entry_px=entry_px, exit_px=exit_px, evol=evol,
                        days=(E.dparse(d) - E.dparse(ed)).days))

        for (name, target), rows in sorted(results.items()):
            if len(rows) < 4:
                continue
            kind = ("IC" if name.startswith("IC") else
                    "WS" if name.startswith("WS") else
                    "STG" if name.startswith("STG") else "STR")
            # exact structure margin where Kite gave us one, else the family proxy
            em = margin_at(T, name) or margin_at(T, MCOL[kind])
            for slip in SLIPS:
                nets, costs = [], []
                for r in rows:
                    c = E.costs_points_legs(r["entry_px"], r["exit_px"], slip)
                    costs.append(c)
                    nets.append(r["credit"] - r["exit_cost"] - c)
                n = len(nets)
                mean = sum(nets) / n
                sd = st.stdev(nets) if n > 1 else 0.0
                eq = peak = mdd = 0.0
                for x in nets:
                    eq += x
                    peak = max(peak, eq)
                    mdd = min(mdd, eq - peak)
                ad = sum(r["days"] for r in rows) / n
                ply = mean * 365.0 / ad
                w.writerow(dict(
                    tenor=T, exit_dte=X, struct=name, kind=kind,
                    target=("none" if target > 5 else target), slip=slip, n=n,
                    n_indep=indep_count(rows), first=min(r["entry_date"] for r in rows),
                    last=max(r["entry_date"] for r in rows),
                    win_rate=round(100.0 * sum(1 for x in nets if x > 0) / n, 1),
                    avg_credit=round(sum(r["credit"] for r in rows) / n, 1),
                    avg_gross_prem=round(sum(r["gross_in"] for r in rows) / n, 1),
                    avg_days=round(ad, 1), avg_net=round(mean, 2), sd_net=round(sd, 1),
                    t_stat=round(mean / (sd / math.sqrt(n)), 2) if sd > 0 else 0,
                    total_net=round(sum(nets), 1), worst=round(min(nets), 1),
                    best=round(max(nets), 1), max_dd=round(mdd, 1),
                    pts_per_lotyear=round(ply, 1), entry_margin=int(em) if em else "",
                    ret_pct_yr=round(100.0 * ply * E.LOT / em, 1) if em else "",
                    avg_cost=round(sum(costs) / n, 2),
                    n_target=sum(1 for r in rows if r["reason"] == "TARGET"),
                    med_entry_vol=int(st.median([r["evol"] for r in rows]))))
        of.flush()
        print("tenor %d exit %d -> %d structure/target cells"
              % (T, X, len(results)), flush=True)
    of.close()
    print("DONE -> %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
