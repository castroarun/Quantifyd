#!/usr/bin/env python3
"""
research/174 — P1b: WHERE in a contract's life does a short straddle actually earn?

The tenor question is usually argued as "longer tenor = more premium". That is true and
irrelevant; what matters is how much of the premium decays per day you are exposed, and
at what risk. This decomposes the contract's life into adjacent DTE windows and asks, for
each one: enter an ATM straddle at the START of the window, exit at the END of it, what
is the net result per day held?

Each window is measured independently with a strike chosen ATM at the window's own start,
because that is what a trader entering then would do. Windows do not share a strike, so
they do not compound into a single path — this is an attribution, not a backtest.

Writes results/p1b_decay_windows.csv
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
TAG = os.environ.get("R174_TAG", "")
OUT = RES / ("p1b_decay_windows%s.csv" % TAG)
SYMBOL = "NIFTY"
UNDER = "NIFTY50"
START = os.environ.get("R174_START", "2015-01-01")
VOL_FLOOR = int(os.environ.get("R174_VF", "25"))
SLIP = float(os.environ.get("R174_SLIP", "0.0075"))

LADDER = [365, 300, 240, 210, 180, 150, 120, 105, 90, 75, 60, 50, 45, 40, 35, 30, 25,
          21, 14, 7]

FIELDS = ["d_start", "d_end", "n", "first", "last", "avg_credit", "avg_days", "win_rate",
          "avg_gross", "avg_net", "sd_net", "t_stat", "net_per_day", "worst", "best",
          "med_entry_vol", "med_dist_pct"]


def main():
    con = E.connect()
    days = E.sessions(con, SYMBOL, "2011-01-01")
    cat = E.expiry_catalogue(con, SYMBOL)
    spot = E.spot_series(con, UNDER)

    of = open(OUT, "w", newline="")
    w = csv.DictWriter(of, fieldnames=FIELDS)
    w.writeheader()

    for i in range(len(LADDER) - 1):
        d0, d1 = LADDER[i], LADDER[i + 1]
        rows = []
        for e in sorted(ex for ex in cat if ex >= START):
            if cat[e]["horizon"] < d0 + 5:
                continue
            ed = E.prev_session(days, E.dstr(E.dparse(e) - timedelta(days=d0)))
            xd = E.prev_session(days, E.dstr(E.dparse(e) - timedelta(days=d1)))
            if not ed or not xd or ed < START or ed < cat[e]["first_seen"] or xd <= ed:
                continue
            if ed not in spot:
                continue
            chain = E.expiry_chain(con, SYMBOL, e)
            if ed not in chain:
                continue
            pos = E.build_position(chain[ed], spot[ed], dict(kind="STR"), VOL_FLOOR)
            if pos is None:
                continue
            k = pos["legs"][0][0]
            credit = pos["credit"]
            # first fillable session at or after the window end
            m = xdate = None
            for d in sorted(chain):
                if d < xd:
                    continue
                mm = E.mark(chain[d], pos["legs"], VOL_FLOOR)
                if mm is not None:
                    m, xdate = mm, d
                    break
            if m is None:
                continue
            held = (E.dparse(xdate) - E.dparse(ed)).days
            if held <= 0:
                continue
            cost = E.costs_points_legs([(credit / 2, -1), (credit / 2, -1)],
                                       [(m / 2, -1), (m / 2, -1)], SLIP)
            rows.append(dict(credit=credit, net=credit - m - cost, gross=credit - m,
                             days=held, entry=ed,
                             vol=chain[ed][k]["CE"][1] + chain[ed][k]["PE"][1],
                             dist=100.0 * abs(k - spot[ed]) / spot[ed]))
        if len(rows) < 5:
            print("window %4d -> %4d DTE : only %d trades, skipped" % (d0, d1, len(rows)),
                  flush=True)
            continue
        nets = [r["net"] for r in rows]
        n = len(nets)
        mean = sum(nets) / n
        sd = st.stdev(nets) if n > 1 else 0.0
        ad = sum(r["days"] for r in rows) / n
        rec = dict(
            d_start=d0, d_end=d1, n=n, first=min(r["entry"] for r in rows),
            last=max(r["entry"] for r in rows),
            avg_credit=round(sum(r["credit"] for r in rows) / n, 1),
            avg_days=round(ad, 1),
            win_rate=round(100.0 * sum(1 for x in nets if x > 0) / n, 1),
            avg_gross=round(sum(r["gross"] for r in rows) / n, 2),
            avg_net=round(mean, 2), sd_net=round(sd, 1),
            t_stat=round(mean / (sd / math.sqrt(n)), 2) if sd > 0 else 0,
            net_per_day=round(mean / ad, 3), worst=round(min(nets), 1),
            best=round(max(nets), 1),
            med_entry_vol=int(st.median([r["vol"] for r in rows])),
            med_dist_pct=round(st.median([r["dist"] for r in rows]), 3))
        w.writerow(rec)
        of.flush()
        print("%4d -> %4d DTE  n=%3d  credit=%7.1f  net=%8.2f  t=%5.2f  /day=%6.3f  "
              "win=%5.1f%%  worst=%8.1f  vol=%d"
              % (d0, d1, n, rec["avg_credit"], rec["avg_net"], rec["t_stat"],
                 rec["net_per_day"], rec["win_rate"], rec["worst"], rec["med_entry_vol"]),
              flush=True)
    of.close()
    print("DONE -> %s" % OUT)


if __name__ == "__main__":
    main()
