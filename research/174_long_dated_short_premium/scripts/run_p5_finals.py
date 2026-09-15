#!/usr/bin/env python3
"""
research/174 — P5: the report engine. Takes a short list of finalist specs and produces
everything the verdict needs, on identical entry days so the comparison is paired:

  * per-trade economics net of cost, at three slippage settings (cost ladder)
  * per-YEAR table (return in points, and the intra-year peak-to-trough drawdown
    measured from the running peak of the FULL curve, never from the year's own first
    bar — that convention error is what r/154 had to retract)
  * TWO-WINDOW SPLIT: both halves must pass, or it is a regime artifact
  * VIX-rank entry filter as an AXIS, not an assumption (the live book runs rank > 25)
  * return on MEASURED Kite margin, and the equity curve in rupees at a stated lot count
  * trade-level correlation between specs, for the "run it alongside" question

Writes:
  results/p5_trades.csv     every finalist's trades
  results/p5_summary.csv    one row per (spec, slip, vix filter, window)
  results/p5_yearly.csv     one row per (spec, year)
"""
import csv
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_lt as E                                        # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
TAG = os.environ.get("R174_TAG", "")
SYMBOL = "NIFTY"
UNDER = "NIFTY50"
START = os.environ.get("R174_START", "2015-01-01")
VOL_FLOOR = int(os.environ.get("R174_VF", "25"))
SLIPS = [0.0025, 0.0075, 0.015]
VIX_FILTERS = [(-1, "off"), (25, "rank>25"), (50, "rank>50")]
LOTS = int(os.environ.get("R174_LOTS", "3"))       # the live 45-DTE book runs 3 lots

# spec = (label, tenor, exit_dte, structure spec, target)
SPECS = []
for item in os.environ.get(
        "R174_SPECS",
        "LIVE45:45:21:STR:0.50,STG5_45:45:21:STG0.05:0.50,STG35_45:45:21:STG0.035:0.50,"
        "LIVE60:60:27:STR:0.50,STG5_60:60:27:STG0.05:0.50").split(","):
    lab, T, X, sk, tg = item.split(":")
    if sk == "STR":
        spec = dict(kind="STR")
    elif sk.startswith("STG"):
        spec = dict(kind="STG", body=float(sk[3:]))
    elif sk.startswith("IC"):
        b, w = sk[2:].split("w")
        spec = dict(kind="IC", body=float(b), wing=float(w))
    else:
        spec = dict(kind="WS", wing=float(sk[2:]))
    SPECS.append((lab, int(T), int(X), spec, float(tg)))

MJ = json.load(open(RES / "margin_by_tenor.json"))


def margin_at(dte, col):
    pts = sorted((r["dte"], r.get(col)) for r in MJ["rows"] if r.get(col))
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if dte <= xs[0]:
        return ys[0]
    if dte >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if dte <= xs[i]:
            f = (dte - xs[i - 1]) / float(xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


def margin_col(sk_label):
    for c in (sk_label,):
        if any(r.get(c) for r in MJ["rows"]):
            return c
    return "STR"


TR_F = ["spec", "expiry", "entry_date", "exit_date", "days", "credit", "gross_prem",
        "exit_cost", "gross_pts", "reason", "vix_rank", "entry_vol", "strikes"]
SM_F = ["spec", "tenor", "exit_dte", "target", "slip", "vix_filter", "window", "n",
        "n_indep", "first", "last", "win_rate", "avg_credit", "avg_days", "avg_net",
        "sd_net", "t_stat", "total_net", "worst", "max_dd_pts", "pts_per_lotyear",
        "entry_margin", "ret_pct_yr", "rs_per_year_at_lots", "lots"]
YR_F = ["spec", "slip", "vix_filter", "year", "n", "net_pts", "intra_year_dd_pts",
        "win_rate"]


def structure_label(spec):
    k = spec["kind"]
    if k == "STR":
        return "STR"
    if k == "STG":
        return "STG%.1f" % (spec["body"] * 100)
    if k == "IC":
        return "IC%.1fw%.0f" % (spec["body"] * 100, spec["wing"] * 100)
    return "WS%.0f" % (spec["wing"] * 100)


def build(con, days, cat, spot, vrank, T, X, spec, target):
    out = []
    for e in sorted(ex for ex in cat if ex >= START):
        if cat[e]["horizon"] < T + 5:
            continue
        ed = E.prev_session(days, E.dstr(E.dparse(e) - timedelta(days=T)))
        xd_t = E.prev_session(days, E.dstr(E.dparse(e) - timedelta(days=X)))
        if not ed or not xd_t or ed < START or ed < cat[e]["first_seen"] or xd_t <= ed:
            continue
        if ed not in spot:
            continue
        chain = E.expiry_chain(con, SYMBOL, e)
        if ed not in chain:
            continue
        pos = E.build_position(chain[ed], spot[ed], spec, VOL_FLOOR)
        if pos is None:
            continue
        entry_px = [(E.leg_price(chain[ed], k, s), sg) for k, s, sg in pos["legs"]]
        if any(p is None for p, _ in entry_px):
            continue
        credit = pos["credit"]
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
        exit_px = [(E.leg_price(chain[d], k, s), sg) for k, s, sg in pos["legs"]]
        if any(p is None for p, _ in exit_px):
            continue
        out.append(dict(
            expiry=e, entry_date=ed, exit_date=d,
            days=(E.dparse(d) - E.dparse(ed)).days, credit=credit,
            gross_prem=sum(abs(p) for p, _ in entry_px), exit_cost=m,
            gross_pts=credit - m, reason=reason, vix_rank=vrank.get(ed, -1),
            entry_vol=sum(chain[ed][k][s][1] for k, s, _ in pos["legs"]),
            strikes="|".join("%s%s" % (int(k), s) for k, s, _ in pos["legs"]),
            entry_px=entry_px, exit_px=exit_px))
    return out


def dd_from_running_peak(seq):
    """Peak-to-trough of the cumulative curve. seq is (key, net) in time order."""
    eq = peak = 0.0
    dd_by_key = defaultdict(float)
    for key, x in seq:
        eq += x
        peak = max(peak, eq)
        dd_by_key[key] = min(dd_by_key[key], eq - peak)
    return dd_by_key


def main():
    con = E.connect()
    days = E.sessions(con, SYMBOL, "2011-01-01")
    cat = E.expiry_catalogue(con, SYMBOL)
    spot = E.spot_series(con, UNDER)
    _, vrank = E.vix_series(con)

    tf = open(RES / ("p5_trades%s.csv" % TAG), "w", newline="")
    tw = csv.DictWriter(tf, fieldnames=TR_F)
    tw.writeheader()
    sf = open(RES / ("p5_summary%s.csv" % TAG), "w", newline="")
    sw = csv.DictWriter(sf, fieldnames=SM_F)
    sw.writeheader()
    yf = open(RES / ("p5_yearly%s.csv" % TAG), "w", newline="")
    yw = csv.DictWriter(yf, fieldnames=YR_F)
    yw.writeheader()

    store = {}
    for lab, T, X, spec, target in SPECS:
        rows = build(con, days, cat, spot, vrank, T, X, spec, target)
        store[lab] = rows
        print("%-10s tenor %3d exit %3d %-8s -> %d trades"
              % (lab, T, X, structure_label(spec), len(rows)), flush=True)
        for r in rows:
            tw.writerow({k: r[k] for k in TR_F if k != "spec"} | {"spec": lab})
        if not rows:
            continue
        em = margin_at(T, margin_col(structure_label(spec)))
        allmid = sorted(r["entry_date"] for r in rows)[len(rows) // 2]

        for slip in SLIPS:
            for vcut, vname in VIX_FILTERS:
                sel = [r for r in rows if r["vix_rank"] > vcut]
                if len(sel) < 5:
                    continue
                for wname, sub in (("full", sel),
                                   ("H1", [r for r in sel if r["entry_date"] < allmid]),
                                   ("H2", [r for r in sel if r["entry_date"] >= allmid])):
                    if len(sub) < 4:
                        continue
                    nets = [r["gross_pts"] - E.costs_points_legs(r["entry_px"],
                                                                 r["exit_px"], slip)
                            for r in sub]
                    n = len(nets)
                    mean = sum(nets) / n
                    sd = st.stdev(nets) if n > 1 else 0.0
                    eq = peak = mdd = 0.0
                    for x in nets:
                        eq += x
                        peak = max(peak, eq)
                        mdd = min(mdd, eq - peak)
                    ad = sum(r["days"] for r in sub) / n
                    ply = mean * 365.0 / ad
                    nn, last = 0, ""
                    for r in sorted(sub, key=lambda z: z["exit_date"]):
                        if r["entry_date"] >= last:
                            nn += 1
                            last = r["exit_date"]
                    sw.writerow(dict(
                        spec=lab, tenor=T, exit_dte=X, target=target, slip=slip,
                        vix_filter=vname, window=wname, n=n, n_indep=nn,
                        first=min(r["entry_date"] for r in sub),
                        last=max(r["entry_date"] for r in sub),
                        win_rate=round(100.0 * sum(1 for x in nets if x > 0) / n, 1),
                        avg_credit=round(sum(r["credit"] for r in sub) / n, 1),
                        avg_days=round(ad, 1), avg_net=round(mean, 2),
                        sd_net=round(sd, 1),
                        t_stat=round(mean / (sd / math.sqrt(n)), 2) if sd > 0 else 0,
                        total_net=round(sum(nets), 1), worst=round(min(nets), 1),
                        max_dd_pts=round(mdd, 1), pts_per_lotyear=round(ply, 1),
                        entry_margin=int(em) if em else "",
                        ret_pct_yr=round(100.0 * ply * E.LOT / em, 1) if em else "",
                        rs_per_year_at_lots=int(ply * E.LOT * LOTS), lots=LOTS))
                # yearly, drawdown measured off the FULL curve's running peak
                if vname == "rank>25" or vname == "off":
                    ordered = sorted(sel, key=lambda r: r["exit_date"])
                    seq = [(r["exit_date"][:4],
                            r["gross_pts"] - E.costs_points_legs(r["entry_px"],
                                                                 r["exit_px"], slip))
                           for r in ordered]
                    ddk = dd_from_running_peak(seq)
                    agg = defaultdict(list)
                    for y, x in seq:
                        agg[y].append(x)
                    for y in sorted(agg):
                        v = agg[y]
                        yw.writerow(dict(
                            spec=lab, slip=slip, vix_filter=vname, year=y, n=len(v),
                            net_pts=round(sum(v), 1),
                            intra_year_dd_pts=round(ddk[y], 1),
                            win_rate=round(100.0 * sum(1 for x in v if x > 0) / len(v), 1)))
        sf.flush()
        yf.flush()

    # trade-level correlation between specs on shared entry dates
    print("\nTrade-level net-P&L correlation between specs (shared entry dates, slip 0.75%):")
    labs = [s[0] for s in SPECS if store.get(s[0])]
    series = {}
    for lab in labs:
        series[lab] = {r["entry_date"]:
                       r["gross_pts"] - E.costs_points_legs(r["entry_px"], r["exit_px"],
                                                            0.0075)
                       for r in store[lab]}
    print("%-12s" % "" + "".join("%10s" % l for l in labs))
    for a in labs:
        line = "%-12s" % a
        for b in labs:
            common = sorted(set(series[a]) & set(series[b]))
            if len(common) < 8:
                line += "%10s" % "-"
                continue
            xa = [series[a][d] for d in common]
            xb = [series[b][d] for d in common]
            ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
            num = sum((xa[i] - ma) * (xb[i] - mb) for i in range(len(xa)))
            den = math.sqrt(sum((x - ma) ** 2 for x in xa)
                            * sum((x - mb) ** 2 for x in xb))
            line += "%10.2f" % (num / den if den else 0)
        print(line)

    tf.close()
    sf.close()
    yf.close()
    print("\nDONE -> results/p5_*%s.csv" % TAG)


if __name__ == "__main__":
    main()
