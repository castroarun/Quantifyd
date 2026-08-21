#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/121 S9 - the India VIX family, done properly.

DATA TRAP FOUND: `market_data_unified` INDIAVIX **daily** bars carry
`open(d) == close(d-1)` on the large majority of days - the daily series is derived,
not a real auction print. Any "overnight VIX change" computed from it is identically
zero for most of the sample, which would have produced a FALSE null ("VIX shocks
predict nothing"). This script rebuilds the whole VIX family from the INDIAVIX
**5-minute** series (2015-02-02 .. 2026-07-17, 211,684 bars), where the 09:15 bar is
a genuine opening print.

Signals rebuilt (all causal at the window start):
  vix_open5        first 5-min bar's open on the day
  vix_at_entry     VIX at the window's own start minute      <- the live-usable one
  vix_gap_pct/pts  vix_open5 vs the prior day's 15:25 VIX    <- the true overnight shock
  vix_cc_pct/pts   prior close-to-close change
  vix_intraday_pct vix_at_entry vs vix_open5 (how VIX has already moved today)

Swept, not fixed: the shock threshold is reported as a full response curve in both
percent and absolute points, because 5% off VIX 11 is not the same event as 5% off 22.
"""
import os, sqlite3, math
import numpy as np
import pandas as pd
from scipy import stats

MD = "/home/arun/quantifyd/backtest_data/market_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

CELL2WIN = {"MON_NIFTY_DTE1": ("MON_1300_1400", 0, "NIFTY50_5MIN", "13:00"),
            "WED_SENSEX_DTE1": ("WED_1030_1200", 2, "SENSEX_1MIN", "10:30"),
            "FRI_NIFTY_DTE2": ("FRI_1000_1200", 4, "NIFTY50_5MIN", "10:00")}


def hm2m(h):
    return int(h[:2]) * 60 + int(h[3:5])


def main():
    lines = []

    def P(s=""):
        lines.append(s); print(s, flush=True)

    c = sqlite3.connect("file:%s?mode=ro" % MD, uri=True)
    # ---- the trap, stated with numbers ----
    q = "SELECT date,open,close FROM market_data_unified WHERE symbol=? AND timeframe=? ORDER BY date"
    dd = list(c.execute(q, ("INDIAVIX", "day")))
    same = sum(1 for i in range(1, len(dd)) if abs(dd[i][1] - dd[i - 1][2]) < 1e-9)
    P("=" * 100)
    P("DATA TRAP - the INDIAVIX DAILY series is not usable for an overnight shock")
    P("=" * 100)
    P("  open(d) == close(d-1) on %d of %d daily bars (%.1f%%)."
      % (same, len(dd) - 1, 100.0 * same / (len(dd) - 1)))
    P("  Any close-to-open VIX change read off that series is identically zero for")
    P("  most of the sample. Rebuilding from the 5-minute series instead.")
    P("")

    # ---- rebuild from 5-min ----
    rows = list(c.execute("SELECT date,open,close FROM market_data_unified "
                          "WHERE symbol=? AND timeframe=? ORDER BY date", ("INDIAVIX", "5minute")))
    byday = {}
    for dt, o, cl in rows:
        d, t = dt[:10], dt[11:16]
        if not t or t < "09:15" or t > "15:30" or o is None or cl is None or cl <= 0:
            continue
        byday.setdefault(d, []).append((hm2m(t), o, cl))
    for d in byday:
        byday[d].sort()
    days = sorted(byday)
    P("INDIAVIX 5-minute: %d session days %s .. %s" % (len(days), days[0], days[-1]))
    vopen = {d: byday[d][0][1] for d in days}
    vclose = {d: byday[d][-1][2] for d in days}
    P("  open(d) == close(d-1) on %d of %d days (%.1f%%)  <- a genuine opening print"
      % (sum(1 for i in range(1, len(days)) if abs(vopen[days[i]] - vclose[days[i - 1]]) < 1e-9),
         len(days) - 1,
         100.0 * sum(1 for i in range(1, len(days)) if abs(vopen[days[i]] - vclose[days[i - 1]]) < 1e-9) / (len(days) - 1)))
    P("")

    feats = {}
    for i, d in enumerate(days):
        if i == 0:
            continue
        pdv = days[i - 1]
        vo, vpc = vopen[d], vclose[pdv]
        f = dict(vix_open5=vo, vix_prev_close=vpc,
                 vix_gap_pct=(vo - vpc) / vpc * 100.0, vix_gap_pts=vo - vpc)
        if i >= 2:
            v2 = vclose[days[i - 2]]
            f["vix_cc_pct"] = (vpc - v2) / v2 * 100.0
            f["vix_cc_pts"] = vpc - v2
        for cell, (win, dow, ser, start) in CELL2WIN.items():
            m0 = hm2m(start)
            cand = [x for x in byday[d] if x[0] <= m0]
            f["vix_at_entry_" + cell] = cand[-1][2] if cand else np.nan
        feats[d] = f
    fv = pd.DataFrame(feats).T
    fv.index.name = "day"
    for cell, (win, dow, ser, start) in CELL2WIN.items():
        fv["vix_intraday_pct_" + cell] = (fv["vix_at_entry_" + cell] - fv.vix_open5) / fv.vix_open5 * 100.0
    fv.to_csv(os.path.join(RES, "vix_features_5min.csv"))

    P("=" * 100)
    P("THE VIX FAMILY vs the window excursion, on the long sample")
    P("  exc_bp   = raw excursion")
    P("  exc_norm = excursion / VIX-implied 1-day sigma (premium-relative)")
    P("=" * 100)
    wo = {t: pd.read_csv(os.path.join(RES, "window_outcomes_%s.csv" % t))
          for t in ("SENSEX_1MIN", "NIFTY50_5MIN")}
    icrows = []
    for cell, (win, dow, ser, start) in CELL2WIN.items():
        s = wo[ser]
        s = s[(s.window == win) & (s.dow == dow)].copy()
        s = s.join(fv, on="day")
        sigs = ["vix_open5", "vix_gap_pct", "vix_gap_pts", "vix_cc_pct", "vix_cc_pts",
                "vix_at_entry_" + cell, "vix_intraday_pct_" + cell]
        P("### %s  (%s, n=%d)" % (cell, ser, len(s)))
        for out in ("exc_bp", "exc_norm"):
            y = pd.to_numeric(s[out], errors="coerce").values
            line = []
            for sig in sigs:
                x = pd.to_numeric(s[sig], errors="coerce").values
                m = ~(np.isnan(x) | np.isnan(y))
                if m.sum() < 60:
                    continue
                rho, p = stats.spearmanr(x[m], y[m])
                line.append("%s %+.3f%s" % (sig.replace("_" + cell, ""), rho,
                                            "*" if p < 0.05 else " "))
                icrows.append(dict(cell=cell, series=ser, outcome=out, signal=sig,
                                   n=int(m.sum()), spearman=round(float(rho), 4),
                                   p=round(float(p), 5)))
            P("  %-9s  %s" % (out, " | ".join(line)))
        # ---- swept shock thresholds, BOTH normalisations, full curve ----
        for sig, unit in (("vix_gap_pct", "%"), ("vix_gap_pts", "pts"),
                          ("vix_cc_pct", "%"), ("vix_cc_pts", "pts")):
            x = pd.to_numeric(s[sig], errors="coerce")
            yn = pd.to_numeric(s["exc_norm"], errors="coerce")
            yb = pd.to_numeric(s["exc_bp"], errors="coerce")
            m = x.notna() & yn.notna()
            if m.sum() < 60:
                continue
            cuts = [50, 60, 70, 80, 90, 95]
            parts = []
            for cu in cuts:
                thr = np.percentile(x[m], cu)
                sel = m & (x >= thr)
                if sel.sum() < 8:
                    continue
                parts.append("  >=p%d (%.2f%s, n=%d): exc %.1fbp / norm %.3f"
                             % (cu, thr, unit, int(sel.sum()), yb[sel].mean(), yn[sel].mean()))
            P("  shock sweep %s   [all days: exc %.1fbp / norm %.3f]"
              % (sig, yb[m].mean(), yn[m].mean()))
            for pt in parts:
                P(pt)
        P("")
    pd.DataFrame(icrows).to_csv(os.path.join(RES, "vix_family_ic.csv"), index=False)

    # ---- the shock as an actual SKIP RULE, against a random-skip null ----
    P("=" * 100)
    P("THE VIX SHOCK AS A SKIP RULE - does standing aside on jump days cut the tail?")
    P("  randPct = where the kept days' p90 premium-relative excursion sits vs skipping")
    P("  the same number of days at random. LOW = the filter is doing something.")
    P("=" * 100)
    P("%-16s %-13s %4s %6s %8s %8s %8s %9s %9s" % ("cell", "signal", "cut", "skip%",
      "p90_all", "p90_kept", "randPct", "max_all", "max_kept"))
    srows = []
    rng2 = np.random.default_rng(20260821)
    for cell, (win, dow, ser, start) in CELL2WIN.items():
        s = wo[ser]
        s = s[(s.window == win) & (s.dow == dow)].copy().join(fv, on="day")
        for sig in ("vix_gap_pts", "vix_gap_pct", "vix_cc_pts", "vix_cc_pct"):
            for cut in (80, 90, 95):
                x = pd.to_numeric(s[sig], errors="coerce")
                y = pd.to_numeric(s["exc_norm"], errors="coerce")
                m = x.notna() & y.notna()
                xv, yv = x[m].values, y[m].values
                n = len(yv)
                if n < 60:
                    continue
                thr = np.percentile(xv, cut)
                keep = xv < thr
                nk = int(keep.sum())
                yk = yv[keep]
                idx = rng2.random((4000, n)).argsort(axis=1)[:, :nk]
                r90 = np.percentile(yv[idx], 90, axis=1)
                p90k = float(np.percentile(yk, 90))
                pct = float((r90 <= p90k).mean() * 100)
                P("%-16s %-13s %4d %6.1f %8.3f %8.3f %8.1f %9.3f %9.3f"
                  % (cell, sig, cut, 100 * (1 - nk / n), np.percentile(yv, 90), p90k, pct,
                     yv.max(), yk.max()))
                srows.append(dict(cell=cell, signal=sig, cut=cut,
                                  skip_pct=round(100 * (1 - nk / n), 1),
                                  p90_all=round(float(np.percentile(yv, 90)), 4),
                                  p90_kept=round(p90k, 4), rand_pctile=round(pct, 1),
                                  max_all=round(float(yv.max()), 4),
                                  max_kept=round(float(yk.max()), 4)))
        P("")
    pd.DataFrame(srows).to_csv(os.path.join(RES, "vix_skiprules.csv"), index=False)


    with open(os.path.join(RES, "vix_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("DONE")


if __name__ == "__main__":
    main()
