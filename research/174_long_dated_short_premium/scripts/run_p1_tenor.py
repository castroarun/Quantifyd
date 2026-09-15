#!/usr/bin/env python3
"""
research/174 — P1: the tenor bake-off. Short ATM straddle, NO STOP, daily close.

Question: at each entry tenor that survives the liquidity gate, what does a short ATM
straddle earn per trade, per lot-year, and per rupee of margin-year, after cost?

Slippage is applied in the SUMMARY, not in the path, because it changes no decision in a
no-stop / close-based-target book — so the four slippage settings are free.

Loop order is EXPIRY-outer so each expiry's chain is read from SQLite exactly once.

Writes:
  results/p1_trades.csv   one row per (arm, trade)
  results/p1_summary.csv  one row per (arm, slip)
"""
import csv
import math
import os
import statistics as st
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_lt as E                                        # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
RES.mkdir(exist_ok=True)
TAG = os.environ.get("R174_TAG", "")
TR = RES / ("p1_trades%s.csv" % TAG)
SM = RES / ("p1_summary%s.csv" % TAG)

SYMBOL = os.environ.get("R174_SYMBOL", "NIFTY")
UNDER = {"NIFTY": "NIFTY50", "BANKNIFTY": "BANKNIFTY"}[SYMBOL]

TENORS = [int(x) for x in os.environ.get(
    "R174_TENORS", "30,45,60,75,90,120,150,180,210,240,300,365").split(",")]
TARGETS = [0.35, 0.50, 0.65, 9.99]        # 9.99 == no target
VOL_FLOORS = [1, 25]
SLIPS = [0.0025, 0.0075, 0.015, 0.03]
START = os.environ.get("R174_START", "2015-01-01")

TR_FIELDS = ["arm", "tenor", "exit_dte", "target", "vol_floor", "expiry", "exp_cls",
             "entry_date", "exit_date", "days_held", "strike", "dist_pct", "spot_entry",
             "spot_exit", "credit", "exit_prem", "gross_pts", "reason",
             "entry_vol", "exit_vol", "entry_oi", "vix_entry", "vix_rank_entry", "rolled"]

SM_FIELDS = ["arm", "tenor", "exit_dte", "target", "vol_floor", "slip", "n", "n_indep",
             "first", "last", "win_rate", "avg_credit", "avg_days", "avg_gross",
             "avg_net", "sd_net", "t_stat", "total_net", "med_net", "avg_win", "avg_loss",
             "best", "worst", "max_dd", "pts_per_lotyear", "avg_cost", "breakeven_slip",
             "n_target", "n_time", "med_entry_vol", "med_exit_vol", "med_dist_pct"]


def exit_grid(t):
    xs = {21}
    for f in (0.10, 0.20, 0.30, 0.45, 0.60):
        xs.add(int(round(t * f)))
    return sorted(x for x in xs if 5 <= x <= t - 15)


def arm_key(T, X, p, vf):
    return "T%d_X%d_p%s_v%d" % (T, X, ("none" if p > 5 else int(round(p * 100))), vf)


def indep_count(trades):
    n, last = 0, ""
    for tr in sorted(trades, key=lambda r: r["exit_date"]):
        if tr["entry_date"] >= last:
            n += 1
            last = tr["exit_date"]
    return n


def main():
    con = E.connect()
    days = E.sessions(con, SYMBOL, "2011-01-01")
    cat = E.expiry_catalogue(con, SYMBOL)
    spot = E.spot_series(con, UNDER)
    vlvl, vrank = E.vix_series(con)
    print("sessions=%d expiries=%d tenors=%s" % (len(days), len(cat), TENORS), flush=True)

    tf = open(TR, "w", newline="")
    tw = csv.DictWriter(tf, fieldnames=TR_FIELDS)
    tw.writeheader()
    buckets = {}
    expiries = sorted(e for e in cat if e >= START)
    ntr = 0

    for ei, e in enumerate(expiries):
        tenors_here = [T for T in TENORS if cat[e]["horizon"] >= T + 5]
        if not tenors_here:
            continue
        entries = {}
        for T in tenors_here:
            ed = E.prev_session(days, E.dstr(E.dparse(e) - timedelta(days=T)))
            if ed and ed >= START and ed >= cat[e]["first_seen"] and spot.get(ed):
                entries[T] = ed
        if not entries:
            continue
        chain = E.expiry_chain(con, SYMBOL, e)
        if not chain:
            continue
        path_days = sorted(chain)

        for T, ed in entries.items():
            if ed not in chain:
                continue
            s0 = spot[ed]
            for vf in VOL_FLOORS:
                pos = E.build_position(chain[ed], s0, dict(kind="STR"), vf)
                if pos is None:
                    continue
                k = pos["legs"][0][0]
                credit = pos["credit"]
                ce_v, pe_v = chain[ed][k]["CE"][1], chain[ed][k]["PE"][1]
                oi0 = chain[ed][k]["CE"][2] + chain[ed][k]["PE"][2]
                fwd = [d for d in path_days if d > ed]
                marks = []
                for d in fwd:
                    m = E.mark(chain[d], pos["legs"], vf)
                    if m is not None:
                        marks.append((d, m))
                if not marks:
                    continue
                for X in exit_grid(T):
                    xd_t = E.prev_session(days, E.dstr(E.dparse(e) - timedelta(days=X)))
                    if not xd_t or xd_t <= ed:
                        continue
                    for p in TARGETS:
                        hit = None
                        for d, m in marks:
                            if p <= 5 and m <= p * credit:
                                hit = (d, m, "TARGET")
                                break
                            if d >= xd_t:
                                hit = (d, m, "TIME")
                                break
                        if hit is None:
                            continue
                        xd, xp, reason = hit
                        kd = chain[xd][k]
                        arm = arm_key(T, X, p, vf)
                        row = dict(
                            arm=arm, tenor=T, exit_dte=X,
                            target=("none" if p > 5 else p), vol_floor=vf,
                            expiry=e, exp_cls=cat[e]["cls"], entry_date=ed, exit_date=xd,
                            days_held=(E.dparse(xd) - E.dparse(ed)).days, strike=k,
                            dist_pct=round(100.0 * abs(k - s0) / s0, 3),
                            spot_entry=round(s0, 2), spot_exit=round(spot.get(xd, 0) or 0, 2),
                            credit=round(credit, 2), exit_prem=round(xp, 2),
                            gross_pts=round(credit - xp, 2), reason=reason,
                            entry_vol=ce_v + pe_v, exit_vol=kd["CE"][1] + kd["PE"][1],
                            entry_oi=oi0, vix_entry=vlvl.get(ed, ""),
                            vix_rank_entry=round(vrank.get(ed, -1), 1),
                            rolled=int(xd != xd_t and reason == "TIME"))
                        tw.writerow(row)
                        buckets.setdefault(arm, []).append(row)
                        ntr += 1
        if ei % 50 == 0:
            tf.flush()
            print("  expiry %d/%d (%s) trades=%d arms=%d"
                  % (ei, len(expiries), e, ntr, len(buckets)), flush=True)
    tf.close()
    print("trade build done: %d trades in %d arms" % (ntr, len(buckets)), flush=True)

    sf = open(SM, "w", newline="")
    sw = csv.DictWriter(sf, fieldnames=SM_FIELDS)
    sw.writeheader()
    for arm, rows in sorted(buckets.items()):
        T, X = rows[0]["tenor"], rows[0]["exit_dte"]
        tg, vf = rows[0]["target"], rows[0]["vol_floor"]
        for slip in SLIPS:
            nets, costs = [], []
            for r in rows:
                c = E.costs_points(r["credit"], r["exit_prem"], slip, n_legs=2)
                costs.append(c)
                nets.append(r["gross_pts"] - c)
            n = len(nets)
            mean = sum(nets) / n
            sd = st.stdev(nets) if n > 1 else 0.0
            eq = peak = mdd = 0.0
            for x in nets:
                eq += x
                peak = max(peak, eq)
                mdd = min(mdd, eq - peak)
            wins = [x for x in nets if x > 0]
            loss = [x for x in nets if x <= 0]
            days_avg = sum(r["days_held"] for r in rows) / n
            lo, hi = 0.0, 0.25
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                m2 = sum(r["gross_pts"] - E.costs_points(r["credit"], r["exit_prem"], mid, 2)
                         for r in rows) / n
                if m2 > 0:
                    lo = mid
                else:
                    hi = mid
            sw.writerow(dict(
                arm=arm, tenor=T, exit_dte=X, target=tg, vol_floor=vf, slip=slip, n=n,
                n_indep=indep_count(rows), first=min(r["entry_date"] for r in rows),
                last=max(r["entry_date"] for r in rows),
                win_rate=round(100.0 * len(wins) / n, 1),
                avg_credit=round(sum(r["credit"] for r in rows) / n, 1),
                avg_days=round(days_avg, 1),
                avg_gross=round(sum(r["gross_pts"] for r in rows) / n, 2),
                avg_net=round(mean, 2), sd_net=round(sd, 1),
                t_stat=round(mean / (sd / math.sqrt(n)), 2) if sd > 0 else 0,
                total_net=round(sum(nets), 1), med_net=round(st.median(nets), 2),
                avg_win=round(sum(wins) / len(wins), 1) if wins else 0,
                avg_loss=round(sum(loss) / len(loss), 1) if loss else 0,
                best=round(max(nets), 1), worst=round(min(nets), 1), max_dd=round(mdd, 1),
                pts_per_lotyear=round(mean * 365.0 / days_avg, 1),
                avg_cost=round(sum(costs) / n, 2), breakeven_slip=round(0.5 * (lo + hi), 4),
                n_target=sum(1 for r in rows if r["reason"] == "TARGET"),
                n_time=sum(1 for r in rows if r["reason"] == "TIME"),
                med_entry_vol=int(st.median([r["entry_vol"] for r in rows])),
                med_exit_vol=int(st.median([r["exit_vol"] for r in rows])),
                med_dist_pct=round(st.median([r["dist_pct"] for r in rows]), 3)))
    sf.close()
    print("DONE -> %s , %s" % (TR, SM), flush=True)


if __name__ == "__main__":
    main()
