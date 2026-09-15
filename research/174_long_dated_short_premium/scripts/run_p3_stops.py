#!/usr/bin/env python3
"""
research/174 — P3: the stop bake-off, PAIRED against no-stop on identical trades.

Arun's list: "sls can be combined premium, single side, underlying price move, vix,
relative vix or combinations". Two of those are already dead at 45 DTE and are carried
here only as controls:

  PREM  combined-premium stop      REFUTED by r/119 phase G (-130.9 pts, t -2.34)
  MOVE  underlying-move stop       REFUTED by r/119 phase E (0 of 63 cells beat holding)

The genuinely untested families, and why they might behave differently: every refuted
stop cuts the WHOLE position on a mark-to-market trigger, realising the loss and
forfeiting the theta that pays for it. A SINGLE-SIDE stop keeps the surviving leg's
theta and removes only the gamma that is hurting. A VIX stop fires on the price of risk
rather than on our own mark. Those are different mechanisms, not the same idea retuned.

  SIDE  buy back the threatened leg at c x its own entry price, other leg runs on
  SIDEK buy back the leg whose strike spot has crossed by >= m, other leg runs on
  VIXL  exit when India VIX closes above an absolute level
  VIXR  exit when VIX percentile rank (causal, 252d) crosses a threshold
  VIXD  exit when VIX >= entry-day VIX x d   (relative to the vol we sold into)

Pairing: for one (tenor, exit-DTE, target, vol-floor) arm the trade set is built ONCE
and every stop family is run over the SAME trades, so the comparison is paired by
construction. We report the median paired delta, the win count, and the t-stat of the
difference — never unpaired medians, which lie at small n (playbook 6.5).

Daily close only. No intraday claim is made anywhere in this file.

Writes:
  results/p3_stops.csv        one row per (arm, stop family, param)
  results/p3_paired.csv       one row per (arm, stop family, param, trade) for pairing
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
OUT = RES / ("p3_stops%s.csv" % TAG)
PAIR = RES / ("p3_paired%s.csv" % TAG)

SYMBOL = os.environ.get("R174_SYMBOL", "NIFTY")
UNDER = {"NIFTY": "NIFTY50", "BANKNIFTY": "BANKNIFTY"}[SYMBOL]
START = os.environ.get("R174_START", "2015-01-01")
SLIP = float(os.environ.get("R174_SLIP", "0.0075"))
VOL_FLOOR = int(os.environ.get("R174_VF", "25"))

# (tenor, exit_dte) pairs. Default set is refined from P1 before the real run.
ARMS = [tuple(int(x) for x in a.split(":")) for a in os.environ.get(
    "R174_ARMS", "45:21,60:27,90:40,120:54,180:81,240:108").split(",")]
TARGETS = [0.50, 9.99]

FAMILIES = {
    "NONE":  [None],
    "PREM":  [1.3, 1.5, 1.75, 2.0, 2.5],
    "MOVE":  [0.02, 0.03, 0.04, 0.05, 0.07],
    "SIDE":  [1.5, 2.0, 2.5, 3.0, 4.0],
    "SIDEK": [0.0, 0.01, 0.02, 0.03],
    "VIXL":  [16, 18, 20, 22, 25, 30],
    "VIXR":  [60, 70, 80, 90, 95],
    "VIXD":  [1.15, 1.25, 1.4, 1.6, 2.0],
}

OUT_FIELDS = ["arm", "tenor", "exit_dte", "target", "family", "param", "n", "win_rate",
              "avg_net", "sd_net", "t_stat", "total_net", "worst", "max_dd", "avg_days",
              "pts_per_lotyear", "n_fired", "pct_fired",
              "d_med", "d_mean", "d_t", "d_wins", "d_winpct"]
PAIR_FIELDS = ["arm", "family", "param", "expiry", "entry_date", "net", "base_net",
               "delta", "fired", "days_held"]


def build_trades(con, days, cat, spot, vlvl, vrank, T, X, vol_floor):
    """One trade per qualifying expiry: entry legs, and the full per-leg daily path."""
    trades = []
    for e in sorted(e for e in cat if e >= START):
        if cat[e]["horizon"] < T + 5:
            continue
        ed = E.prev_session(days, E.dstr(E.dparse(e) - timedelta(days=T)))
        if not ed or ed < START or ed < cat[e]["first_seen"] or ed not in spot:
            continue
        xd_t = E.prev_session(days, E.dstr(E.dparse(e) - timedelta(days=X)))
        if not xd_t or xd_t <= ed:
            continue
        chain = E.expiry_chain(con, SYMBOL, e)
        if ed not in chain:
            continue
        s0 = spot[ed]
        pos = E.build_position(chain[ed], s0, dict(kind="STR"), vol_floor)
        if pos is None:
            continue
        k = pos["legs"][0][0]
        ce0, pe0 = chain[ed][k]["CE"][0], chain[ed][k]["PE"][0]
        path = []
        for d in sorted(chain):
            if d <= ed:
                continue
            legs = chain[d].get(k)
            if not legs or "CE" not in legs or "PE" not in legs:
                continue
            ce, pe = legs["CE"], legs["PE"]
            path.append(dict(date=d, ce=ce[0], pe=pe[0],
                             ce_ok=(ce[0] > 0 and ce[1] >= vol_floor),
                             pe_ok=(pe[0] > 0 and pe[1] >= vol_floor),
                             spot=spot.get(d), vix=vlvl.get(d), vr=vrank.get(d)))
        if not path:
            continue
        trades.append(dict(expiry=e, entry_date=ed, exit_target_date=xd_t, strike=k,
                           ce0=ce0, pe0=pe0, credit=ce0 + pe0, spot0=s0,
                           vix0=vlvl.get(ed), path=path))
    return trades


def simulate(tr, target, family, param):
    """Return (net_gross_pts, exit_cost, days_held, fired)."""
    credit = tr["credit"]
    k, s0 = tr["strike"], tr["spot0"]
    v0 = tr["vix0"]
    ce_open = pe_open = True
    ce_cost = pe_cost = 0.0
    fired = 0
    last_day = tr["entry_date"]

    def close_ce(row):
        nonlocal ce_open, ce_cost
        if ce_open and row["ce_ok"]:
            ce_open, ce_cost = False, row["ce"]
            return True
        return False

    def close_pe(row):
        nonlocal pe_open, pe_cost
        if pe_open and row["pe_ok"]:
            pe_open, pe_cost = False, row["pe"]
            return True
        return False

    for row in tr["path"]:
        if not ce_open and not pe_open:
            break
        last_day = row["date"]
        cur = (ce_cost if not ce_open else row["ce"]) + (pe_cost if not pe_open else row["pe"])
        both_ok = row["ce_ok"] and row["pe_ok"]

        # --- time exit -------------------------------------------------------
        if row["date"] >= tr["exit_target_date"]:
            if ce_open and row["ce_ok"]:
                close_ce(row)
            if pe_open and row["pe_ok"]:
                close_pe(row)
            if not ce_open and not pe_open:
                break
            continue

        # --- profit target on the whole position -----------------------------
        if target <= 5 and cur <= target * credit and both_ok:
            close_ce(row)
            close_pe(row)
            break

        # --- stop families ---------------------------------------------------
        if family == "NONE":
            continue
        if family == "PREM":
            if cur >= param * credit and both_ok:
                close_ce(row)
                close_pe(row)
                fired = 1
                break
        elif family == "MOVE":
            if row["spot"] and abs(row["spot"] / s0 - 1.0) >= param and both_ok:
                close_ce(row)
                close_pe(row)
                fired = 1
                break
        elif family == "VIXL":
            if row["vix"] and row["vix"] >= param and both_ok:
                close_ce(row)
                close_pe(row)
                fired = 1
                break
        elif family == "VIXR":
            if row["vr"] is not None and row["vr"] >= param and both_ok:
                close_ce(row)
                close_pe(row)
                fired = 1
                break
        elif family == "VIXD":
            if row["vix"] and v0 and row["vix"] >= param * v0 and both_ok:
                close_ce(row)
                close_pe(row)
                fired = 1
                break
        elif family == "SIDE":
            if ce_open and row["ce"] >= param * tr["ce0"] and close_ce(row):
                fired = 1
            if pe_open and row["pe"] >= param * tr["pe0"] and close_pe(row):
                fired = 1
        elif family == "SIDEK":
            if row["spot"]:
                if ce_open and row["spot"] >= k * (1 + param) and close_ce(row):
                    fired = 1
                if pe_open and row["spot"] <= k * (1 - param) and close_pe(row):
                    fired = 1
        else:
            raise ValueError(family)

    # anything still open is marked at the last available price (expiry-ish)
    if ce_open:
        ce_cost = tr["path"][-1]["ce"]
    if pe_open:
        pe_cost = tr["path"][-1]["pe"]
    exit_cost = ce_cost + pe_cost
    days = (E.dparse(last_day) - E.dparse(tr["entry_date"])).days
    return credit - exit_cost, exit_cost, max(days, 1), fired


def main():
    con = E.connect()
    days = E.sessions(con, SYMBOL, "2011-01-01")
    cat = E.expiry_catalogue(con, SYMBOL)
    spot = E.spot_series(con, UNDER)
    vlvl, vrank = E.vix_series(con)
    print("arms=%s slip=%.4f vol_floor=%d" % (ARMS, SLIP, VOL_FLOOR), flush=True)

    of = open(OUT, "w", newline="")
    ow = csv.DictWriter(of, fieldnames=OUT_FIELDS)
    ow.writeheader()
    pf = open(PAIR, "w", newline="")
    pw = csv.DictWriter(pf, fieldnames=PAIR_FIELDS)
    pw.writeheader()

    for T, X in ARMS:
        trades = build_trades(con, days, cat, spot, vlvl, vrank, T, X, VOL_FLOOR)
        print("tenor %d exit %d -> %d trades" % (T, X, len(trades)), flush=True)
        if len(trades) < 8:
            print("  too few trades, skipping arm", flush=True)
            continue
        for target in TARGETS:
            arm = "T%d_X%d_p%s" % (T, X, "none" if target > 5 else int(target * 100))
            base = {}
            for tr in trades:
                g, xc, dd, _ = simulate(tr, target, "NONE", None)
                base[tr["expiry"]] = g - E.costs_points(tr["credit"], xc, SLIP, 2)
            for fam, params in FAMILIES.items():
                for prm in params:
                    nets, fires, dayl = [], 0, []
                    for tr in trades:
                        g, xc, dd, fired = simulate(tr, target, fam, prm)
                        net = g - E.costs_points(tr["credit"], xc, SLIP, 2)
                        nets.append(net)
                        dayl.append(dd)
                        fires += fired
                        pw.writerow(dict(arm=arm, family=fam, param=prm, expiry=tr["expiry"],
                                         entry_date=tr["entry_date"], net=round(net, 2),
                                         base_net=round(base[tr["expiry"]], 2),
                                         delta=round(net - base[tr["expiry"]], 2),
                                         fired=fired, days_held=dd))
                    n = len(nets)
                    mean = sum(nets) / n
                    sd = st.stdev(nets) if n > 1 else 0.0
                    eq = peak = mdd = 0.0
                    for x in nets:
                        eq += x
                        peak = max(peak, eq)
                        mdd = min(mdd, eq - peak)
                    deltas = [nets[i] - base[trades[i]["expiry"]] for i in range(n)]
                    dsd = st.stdev(deltas) if n > 1 else 0.0
                    dmean = sum(deltas) / n
                    ad = sum(dayl) / n
                    ow.writerow(dict(
                        arm=arm, tenor=T, exit_dte=X,
                        target=("none" if target > 5 else target), family=fam, param=prm,
                        n=n, win_rate=round(100.0 * sum(1 for x in nets if x > 0) / n, 1),
                        avg_net=round(mean, 2), sd_net=round(sd, 1),
                        t_stat=round(mean / (sd / math.sqrt(n)), 2) if sd > 0 else 0,
                        total_net=round(sum(nets), 1), worst=round(min(nets), 1),
                        max_dd=round(mdd, 1), avg_days=round(ad, 1),
                        pts_per_lotyear=round(mean * 365.0 / ad, 1),
                        n_fired=fires, pct_fired=round(100.0 * fires / n, 1),
                        d_med=round(st.median(deltas), 2), d_mean=round(dmean, 2),
                        d_t=round(dmean / (dsd / math.sqrt(n)), 2) if dsd > 0 else 0,
                        d_wins=sum(1 for x in deltas if x > 0),
                        d_winpct=round(100.0 * sum(1 for x in deltas if x > 0) / n, 1)))
            of.flush()
            pf.flush()
            print("  arm %s done" % arm, flush=True)
    of.close()
    pf.close()
    print("DONE -> %s , %s" % (OUT, PAIR), flush=True)


if __name__ == "__main__":
    main()
