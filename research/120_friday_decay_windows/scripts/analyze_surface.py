#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/120 - build the window SURFACE, its controls and its multiple-testing haircut.

Reads results/stage_a_trades.csv (the pre-registered grid) and
results/stage_a_allstarts.csv (every possible start of the same duration - the control).

For every cell it reports:
  net P&L per lot           - does it decay well?
  MAE distribution          - does it decay well BECAUSE nothing happened, or in spite of it?
  excess vs same-day, same-duration ALL-START mean  - the only honest "is this window special"
  neighbour mean            - a real window is a plateau, an artefact is an isolated cell
  a Westfall-Young max-t bootstrap over the whole family - the multiple-testing haircut

Writes results/surface_cells.csv and prints the readable surface.
"""
import csv, os, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "surface_cells.csv")
TXT = os.path.join(RES, "surface_report.txt")
B = 5000
rng = np.random.default_rng(20260821)

lines = []
def P(s=""):
    lines.append(s)
    print(s, flush=True)


def load(fn):
    with open(os.path.join(RES, fn)) as f:
        return list(csv.DictReader(f))


grid = load("stage_a_trades.csv")
alls = load("stage_a_allstarts.csv")

# ---- baseline: mean net of ALL possible windows of that duration on that day ----
base = {}
for r in alls:
    base.setdefault((r["venue"], r["arm"], r["dur"], r["day"]), []).append(float(r["net"]))
basemean = {k: float(np.mean(v)) for k, v in base.items()}
basemae = {}
for r in alls:
    basemae.setdefault((r["venue"], r["arm"], r["dur"], r["day"]), []).append(float(r["mae_full_rs"]))
basemaemean = {k: float(np.mean(v)) for k, v in basemae.items()}

days = sorted({r["day"] for r in grid})
DI = {d: i for i, d in enumerate(days)}
ND = len(days)

cells = {}
for r in grid:
    k = (r["venue"], r["arm"], r["start"], r["dur"])
    cells.setdefault(k, {})[r["day"]] = r

rows = []
for k, byday in sorted(cells.items()):
    ven, arm, start, dur = k
    if len(byday) < ND:
        continue
    net = np.array([float(byday[d]["net"]) for d in days if d in byday])
    mae = np.array([float(byday[d]["mae_full_rs"]) for d in days if d in byday])
    und = np.array([float(byday[d]["und_exc_bp"]) for d in days if d in byday])
    cred = np.array([float(byday[d]["credit"]) for d in days if d in byday])
    exc = np.array([float(byday[d]["net"]) - basemean[(ven, arm, dur, d)]
                    for d in days if d in byday])
    excmae = np.array([float(byday[d]["mae_full_rs"]) - basemaemean[(ven, arm, dur, d)]
                       for d in days if d in byday])
    nsl = sum(1 for d in days if d in byday and byday[d]["reason"] == "SL")
    realdur = np.mean([(int(byday[d]["end"][:2]) * 60 + int(byday[d]["end"][3:5])) -
                       (int(byday[d]["start"][:2]) * 60 + int(byday[d]["start"][3:5]))
                       for d in days if d in byday])
    n = len(net)
    t = float(net.mean() / (net.std(ddof=1) / math.sqrt(n))) if net.std(ddof=1) > 0 else 0.0
    te = float(exc.mean() / (exc.std(ddof=1) / math.sqrt(n))) if exc.std(ddof=1) > 0 else 0.0
    rows.append(dict(venue=ven, arm=arm, start=start, dur=dur, real_min=round(realdur),
                     n=n, mean_net=round(net.mean()), med_net=round(np.median(net)),
                     total_net=round(net.sum()), win_pct=round(100 * (net > 0).mean()),
                     worst=round(net.min()), best=round(net.max()), t_net=round(t, 2),
                     mean_credit=round(cred.mean(), 1),
                     mean_mae=round(mae.mean()), p90_mae=round(np.percentile(mae, 90)),
                     max_mae=round(mae.max()), mean_und_bp=round(und.mean(), 1),
                     p90_und_bp=round(np.percentile(und, 90), 1),
                     sl_hits=nsl,
                     excess_vs_allstart=round(exc.mean()), t_excess=round(te, 2),
                     mae_vs_allstart=round(excmae.mean()),
                     ret_per_mae=round(net.mean() / mae.mean(), 3) if mae.mean() > 0 else None))

# ---- neighbour plateau score ----
DORD = ["45", "60", "90", "120", "HOLD"]
idx = {(r["venue"], r["arm"], r["start"], r["dur"]): r for r in rows}
starts = sorted({r["start"] for r in rows})
SI = {s: i for i, s in enumerate(starts)}
for r in rows:
    nb = []
    si, di = SI[r["start"]], DORD.index(r["dur"])
    for ds in (-1, 0, 1):
        for dd in (-1, 0, 1):
            if ds == 0 and dd == 0:
                continue
            s2 = starts[si + ds] if 0 <= si + ds < len(starts) else None
            d2 = DORD[di + dd] if 0 <= di + dd < len(DORD) else None
            if s2 and d2 and (r["venue"], r["arm"], s2, d2) in idx:
                nb.append(idx[(r["venue"], r["arm"], s2, d2)]["mean_net"])
    r["nbr_mean_net"] = round(float(np.mean(nb))) if nb else None
    r["nbr_n"] = len(nb)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ---- Westfall-Young max-t bootstrap, per venue x arm ----
P("=" * 100)
P("MULTIPLE-TESTING HAIRCUT  (Westfall-Young max-t, %d day-bootstraps, %d Fridays)" % (B, ND))
P("=" * 100)
wy = {}
for ven in ("NIFTY", "SENSEX"):
    for arm in ("SL20", "NOSTOP"):
        fam = [r for r in rows if r["venue"] == ven and r["arm"] == arm]
        if not fam:
            continue
        # matrix of per-day EXCESS vs all-start baseline
        M = []
        for r in fam:
            byday = cells[(ven, arm, r["start"], r["dur"])]
            M.append([float(byday[d]["net"]) - basemean[(ven, arm, r["dur"], d)]
                      for d in days if d in byday])
        M = np.array(M)                      # cells x days
        C = M - M.mean(axis=1, keepdims=True)   # centered => H0 true
        obs_t = np.array([r["t_excess"] for r in fam])
        maxnull = np.empty(B)
        for b in range(B):
            ii = rng.integers(0, M.shape[1], M.shape[1])
            S = C[:, ii]
            m, sd = S.mean(axis=1), S.std(axis=1, ddof=1)
            tt = np.where(sd > 0, m / (sd / math.sqrt(S.shape[1])), 0.0)
            maxnull[b] = np.abs(tt).max()
        crit = np.percentile(maxnull, 95)
        best = fam[int(np.argmax(np.abs(obs_t)))]
        pfw = float((maxnull >= abs(obs_t).max()).mean())
        wy[(ven, arm)] = (crit, pfw)
        P("%-7s %-7s cells=%3d   observed max|t(excess)| = %.2f  (%s %s)   "
          "null 95%% max|t| = %.2f   family-wise p = %.3f  -> %s"
          % (ven, arm, len(fam), abs(obs_t).max(), best["start"], best["dur"], crit, pfw,
             "SURVIVES" if pfw < 0.05 else "NOT SIGNIFICANT"))

# ---- readable surface ----
for ven in ("NIFTY", "SENSEX"):
    for arm in ("SL20", "NOSTOP"):
        fam = [r for r in rows if r["venue"] == ven and r["arm"] == arm]
        if not fam:
            continue
        P()
        P("=" * 100)
        P("SURFACE - %s %s : mean NET Rs/lot per Friday (n=%d)   [ | mean MAE Rs/lot ]" % (ven, arm, ND))
        P("=" * 100)
        hdr = "start  " + "".join("%16s" % d for d in DORD)
        P(hdr)
        for s in starts:
            line = "%-7s" % s
            for d in DORD:
                r = idx.get((ven, arm, s, d))
                line += "%16s" % ("%+5d|%5d" % (r["mean_net"], r["mean_mae"]) if r else "-")
            P(line)

P()
P("=" * 100)
P("TOP CELLS BY EXCESS OVER SAME-DURATION ALL-START BASELINE (the honest ranking)")
P("=" * 100)
for ven in ("NIFTY", "SENSEX"):
    for arm in ("SL20", "NOSTOP"):
        fam = sorted([r for r in rows if r["venue"] == ven and r["arm"] == arm],
                     key=lambda r: -r["excess_vs_allstart"])
        if not fam:
            continue
        P()
        P("%s %s  (WY 95%% crit |t| = %.2f)" % (ven, arm, wy[(ven, arm)][0]))
        P("  %-6s %-5s %5s %6s %6s %5s %7s %7s %7s %8s %6s %7s" %
          ("start", "dur", "net", "excess", "t_exc", "win%", "worst", "meanMAE", "p90MAE",
           "und_bp", "SL", "nbrNet"))
        for r in fam[:8]:
            P("  %-6s %-5s %5d %6d %6.2f %5d %7d %7d %7d %8.1f %6d %7s" %
              (r["start"], r["dur"], r["mean_net"], r["excess_vs_allstart"], r["t_excess"],
               r["win_pct"], r["worst"], r["mean_mae"], r["p90_mae"], r["mean_und_bp"],
               r["sl_hits"], r["nbr_mean_net"]))
        P("  ...worst 3:")
        for r in fam[-3:]:
            P("  %-6s %-5s %5d %6d %6.2f %5d %7d %7d %7d %8.1f %6d %7s" %
              (r["start"], r["dur"], r["mean_net"], r["excess_vs_allstart"], r["t_excess"],
               r["win_pct"], r["worst"], r["mean_mae"], r["p90_mae"], r["mean_und_bp"],
               r["sl_hits"], r["nbr_mean_net"]))

# ---- the deployed cell + reference cells ----
P()
P("=" * 100)
P("REFERENCE CELLS (what is actually deployed / proposed)")
P("=" * 100)
P("  %-7s %-7s %-6s %-5s %5s %6s %6s %5s %7s %7s %7s" %
  ("venue", "arm", "start", "dur", "net", "excess", "t_exc", "win%", "worst", "meanMAE", "p90MAE"))
for ven, arm, s, d, lab in [("NIFTY", "SL20", "10:00", "120", "LIVE TimeB Fri DTE2"),
                            ("NIFTY", "SL20", "13:00", "60", "TIMEB2 shape (Mon/Tue cell)"),
                            ("NIFTY", "SL20", "09:20", "HOLD", "COMB-like full day"),
                            ("SENSEX", "SL20", "10:30", "90", "TB-SX DTE1 shape"),
                            ("SENSEX", "SL20", "09:20", "HOLD", "CSL30F full day")]:
    r = idx.get((ven, arm, s, d))
    if r:
        P("  %-7s %-7s %-6s %-5s %5d %6d %6.2f %5d %7d %7d %7d   <- %s" %
          (ven, arm, s, d, r["mean_net"], r["excess_vs_allstart"], r["t_excess"],
           r["win_pct"], r["worst"], r["mean_mae"], r["p90_mae"], lab))

open(TXT, "w").write("\n".join(lines) + "\n")
print("\nwrote", OUT, "and", TXT)
