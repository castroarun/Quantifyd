#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/121 S3 - LONG-SAMPLE FIT.

Question: does any pre-registered day-level regime signal predict the SIZE of the
intraday excursion inside the three live non-expiry TimeB windows?

This is deliberately answered on hundreds of days, NOT on the ~16 recorded option
days, because a filter fitted on 16 days will always look like it works.

Two outcomes, and the difference between them is the whole point:
  exc_bp    raw excursion in bp
  exc_norm  excursion / VIX-implied 1-day sigma  ("was the move big FOR WHAT THE
            OPTION MARKET CHARGED?")
A signal that moves exc_bp but not exc_norm is only detecting expensive days.
Skipping those days skips the premium along with the move - research/120's trap.

Outputs
  results/longfit_spearman.csv     every signal x window x series x outcome
  results/longfit_quintiles.csv    the full response curve (never just the best cut)
  results/longfit_skiprules.csv    skip-top-k% / skip-bottom-k% vs a random-skip null
  results/longfit_report.txt       printed summary
"""
import csv, os, math, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
rng = np.random.default_rng(20260821)

SERIES = ["SENSEX_1MIN", "NIFTY50_5MIN", "SENSEX_5MIN"]
WINDOWS = ["MON_1300_1400", "WED_1030_1200", "FRI_1000_1200"]
LIVE_DOW = {"MON_1300_1400": 0, "WED_1030_1200": 2, "FRI_1000_1200": 4}

# --- PRE-REGISTERED SIGNAL LIST (direction = hypothesised sign of corr with excursion) ---
SIGNALS = [
    ("cpr_today",      +1, "today's CPR width % (from prior-day OHLC); r/67 daily sign: narrow->calm"),
    ("cpr_prev",       +1, "previous day's CPR width %"),
    ("wcpr_this",      -1, "this week's CPR width %; r/67 weekly sign FLIPS: wide->contained"),
    ("wcpr_prev",      -1, "previous week's CPR width %"),
    ("gap_abs",        +1, "|opening gap| %"),
    ("gap_pct",         0, "signed opening gap % (direction test)"),
    ("pdr_pct",        +1, "previous day range %"),
    ("pdr_rel",        +1, "previous day range / trailing-20 mean range"),
    ("atr14_pct",      +1, "ATR(14) % - the ambient vol level"),
    ("vix_open",       +1, "India VIX at the open (LEVEL - guard: vol proxy vs vol outcome)"),
    ("vix_chg_oc_pct", +1, "overnight VIX change, % of prior close"),
    ("vix_chg_oc_pts", +1, "overnight VIX change, absolute points"),
    ("vix_chg_cc_pct", +1, "prior close-to-close VIX change, %"),
    ("vix_chg_cc_pts", +1, "prior close-to-close VIX change, points"),
    ("pre_move_bp",    +1, "|window-start price - day open| bp (how far today has already run)"),
    ("pre_range_bp",   +1, "session high-low up to the window start, bp"),
]
OUTCOMES = ["exc_bp", "exc_norm"]
CUTS = [50, 60, 70, 75, 80, 85, 90]      # skip the top k-th percentile and above
NBOOT = 2000


def load(tag):
    rows = []
    with open(os.path.join(RES, "window_outcomes_%s.csv" % tag)) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fnum(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def main():
    out_sp = open(os.path.join(RES, "longfit_spearman.csv"), "w", newline="")
    wsp = csv.writer(out_sp)
    wsp.writerow(["series", "window", "dow_filter", "outcome", "signal", "n",
                  "spearman", "p", "pearson", "hyp_sign", "sign_ok"])
    out_q = open(os.path.join(RES, "longfit_quintiles.csv"), "w", newline="")
    wq = csv.writer(out_q)
    wq.writerow(["series", "window", "dow_filter", "outcome", "signal", "quintile",
                 "lo", "hi", "n", "mean", "median", "p90", "p95", "max", "pct_gt_p80all"])
    out_s = open(os.path.join(RES, "longfit_skiprules.csv"), "w", newline="")
    ws = csv.writer(out_s)
    ws.writerow(["series", "window", "dow_filter", "outcome", "signal", "side", "cut_pct",
                 "thresh", "n_all", "n_kept", "pct_skipped",
                 "mean_all", "mean_kept", "p90_all", "p90_kept", "p95_all", "p95_kept",
                 "max_all", "max_kept", "tail_all", "tail_kept",
                 "rand_p90_mean", "rand_p90_pctile", "rand_tail_mean", "rand_tail_pctile"])

    lines = []

    def emit(s):
        lines.append(s)
        print(s, flush=True)

    ntests = 0
    for tag in SERIES:
        rows = load(tag)
        for win in WINDOWS:
            for dow_filter in ("LIVE", "ALL"):
                sub = [r for r in rows if r["window"] == win]
                if dow_filter == "LIVE":
                    sub = [r for r in sub if int(r["dow"]) == LIVE_DOW[win]]
                if len(sub) < 60:
                    continue
                for outcome in OUTCOMES:
                    y_all = np.array([fnum(r[outcome]) for r in sub])
                    ok0 = ~np.isnan(y_all)
                    if ok0.sum() < 60:
                        continue
                    tail_thresh = np.nanpercentile(y_all[ok0], 90)
                    nullcache = {}

                    def null_for(y, nk):
                        key = (len(y), nk)
                        if key in nullcache:
                            return nullcache[key]
                        idx = rng.random((NBOOT, len(y))).argsort(axis=1)[:, :nk]
                        yb = y[idx]
                        rp90 = np.percentile(yb, 90, axis=1)
                        rtail = (yb > tail_thresh).mean(axis=1) * 100.0
                        nullcache[key] = (rp90, rtail)
                        return nullcache[key]

                    for sig, hyp, _desc in SIGNALS:
                        x_all = np.array([fnum(r.get(sig, "")) for r in sub])
                        m = ok0 & ~np.isnan(x_all)
                        n = int(m.sum())
                        if n < 60:
                            continue
                        x, y = x_all[m], y_all[m]
                        rho, p = stats.spearmanr(x, y)
                        pr, _ = stats.pearsonr(x, y)
                        sign_ok = "" if hyp == 0 else ("YES" if np.sign(rho) == hyp else "no")
                        wsp.writerow([tag, win, dow_filter, outcome, sig, n,
                                      round(rho, 4), round(p, 6), round(pr, 4), hyp, sign_ok])
                        ntests += 1
                        qs = np.percentile(x, [0, 20, 40, 60, 80, 100])
                        for qi in range(5):
                            lo, hi = qs[qi], qs[qi + 1]
                            sel = (x >= lo) & (x <= hi) if qi == 4 else (x >= lo) & (x < hi)
                            if sel.sum() < 5:
                                continue
                            yy = y[sel]
                            wq.writerow([tag, win, dow_filter, outcome, sig, "Q%d" % (qi + 1),
                                         round(lo, 4), round(hi, 4), int(sel.sum()),
                                         round(yy.mean(), 2), round(np.median(yy), 2),
                                         round(np.percentile(yy, 90), 2),
                                         round(np.percentile(yy, 95), 2), round(yy.max(), 2),
                                         round(100.0 * (yy > tail_thresh).mean(), 1)])
                        for side in ("skip_high", "skip_low"):
                            for cut in CUTS:
                                thr = np.percentile(x, cut if side == "skip_high" else 100 - cut)
                                keep = (x < thr) if side == "skip_high" else (x > thr)
                                nk = int(keep.sum())
                                if nk < 30 or nk == n:
                                    continue
                                yk = y[keep]
                                p90a = float(np.percentile(y, 90)); p90k = float(np.percentile(yk, 90))
                                tail_a = float((y > tail_thresh).mean() * 100)
                                tail_k = float((yk > tail_thresh).mean() * 100)
                                rp90, rtail = null_for(y, nk)
                                ws.writerow([tag, win, dow_filter, outcome, sig, side, cut,
                                             round(float(thr), 4), n, nk,
                                             round(100.0 * (1 - nk / n), 1),
                                             round(float(y.mean()), 2), round(float(yk.mean()), 2),
                                             round(p90a, 2), round(p90k, 2),
                                             round(float(np.percentile(y, 95)), 2),
                                             round(float(np.percentile(yk, 95)), 2),
                                             round(float(y.max()), 2), round(float(yk.max()), 2),
                                             round(tail_a, 2), round(tail_k, 2),
                                             round(float(rp90.mean()), 2),
                                             round(float((rp90 <= p90k).mean() * 100), 1),
                                             round(float(rtail.mean()), 2),
                                             round(float((rtail <= tail_k).mean() * 100), 1)])
    out_sp.close(); out_q.close(); out_s.close()
    emit("Spearman tests written: %d" % ntests)
    with open(os.path.join(RES, "longfit_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("DONE")


if __name__ == "__main__":
    main()
