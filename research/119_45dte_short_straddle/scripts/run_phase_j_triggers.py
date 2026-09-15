#!/usr/bin/env python3
"""Phase J — the two management triggers NOT yet tested on the 45-DTE book.

Arun (2026-09-15): "lets say the price moves by x%, then we redeploy the straddle
or short a new straddle ... or say when vix increases further to some extent,
deploy or redeploy straddle/strangle".

Almost all of that is already on record and refuted - see the map in RESULTS.md
section 5e. What is NOT on record:

  J1  MOVE-triggered ADD:  spot has moved x% from entry -> keep the original,
      sell a SECOND straddle at the then-ATM.  (Phase E tested move-triggered
      EXIT and RECENTRE; Phase G tested PREMIUM-triggered ADD. Not this.)
  J2  VIX-triggered ADD / RECENTRE:  India VIX has risen r% above its level on
      the entry day -> add a second straddle, or close and re-centre.
      (Phase I tested VIX crossing the filter AFTER a failed entry day. Not a
      mid-campaign spike on a campaign already held.)

This is a PRE-REGISTERED kill test, not a search. r/174 found the mechanism:
the return is holding one strike through the drift, and every sub-window on its
own is ~zero. Both triggers intervene mid-hold, so the prediction is that both
lose. Falsification (declared now): an arm survives only if its PAIRED delta vs
HOLD on the fired campaigns is positive at t > 2 on BOTH the VIX>25 book and the
unfiltered 89. Anything less is the ninth kill.

Reuses Phase G's engine. Read-only.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_phase_g_addstraddle as G   # noqa: E402

RES = G.RES
STEP = G.STEP
MOVES = [1.5, 2.0, 3.0]          # % spot move from entry, either direction
VIXUPS = [15.0, 30.0, 50.0]      # % rise in India VIX level vs the entry day


def vix_level(vx, day):
    idx = [v for d, v in vx if d <= day]
    return idx[-1] if idx else None


def run(camp, arm, kind, level, chains, spot, vx):
    """One campaign. arm in HOLD / ADD / RECENTRE. kind in MOVE / VIX."""
    K0, C0 = float(camp["strike"]), float(camp["credit"])
    S0 = float(camp["entry_spot"])
    V0 = vix_level(vx, camp["entry_date"])
    days = sorted(d for d in chains if camp["entry_date"] < d <= camp["xd"])
    if not days or (kind == "VIX" and not V0):
        return None
    book = [dict(K=K0, credit=C0)]
    realised = cost = 0.0
    fired, peak = 0, 1
    for i, d in enumerate(days):
        ch = chains[d]
        last = (i == len(days) - 1)
        prices = [G.straddle(ch, p["K"]) for p in book]
        if any(p is None for p in prices):
            continue
        if arm != "HOLD" and not fired and not last:
            S = spot.get(d)
            if S is None:
                continue
            if kind == "MOVE":
                hit = abs(S / S0 - 1.0) * 100.0 >= level
            else:
                V = vix_level(vx, d)
                hit = V is not None and (V / V0 - 1.0) * 100.0 >= level
            if hit:
                Kn = round(S / STEP) * STEP
                pn = G.straddle(ch, Kn, need_liquid=True)
                if pn is None:
                    continue
                if arm == "RECENTRE":
                    for p, px in zip(book, prices):
                        realised += p["credit"] - px
                        cost += G.costs_points(p["credit"], px)
                    book = [dict(K=float(Kn), credit=pn)]
                else:                                   # ADD
                    book.append(dict(K=float(Kn), credit=pn))
                    peak = 2
                fired = 1
    for p in book:
        px = None
        for d in reversed(days):
            px = G.straddle(chains[d], p["K"])
            if px is not None:
                break
        if px is None:
            return None
        realised += p["credit"] - px
        cost += G.costs_points(p["credit"], px)
    return realised - cost, peak, fired


def main():
    con, spot, vx, sess = G.load_market()
    camps = list(csv.DictReader(open(G.TRADES)))
    for c in camps:
        c["vix_rank"] = G.vix_rank(vx, c["entry_date"])
        c["xd"] = G.true_exit(sess, c["expiry"]) or c["exit_date"]
    pre = {}
    for c in camps:
        k = (c["expiry"], c["entry_date"], c["xd"])
        if k not in pre:
            pre[k] = G.chains_for(con, c["expiry"], c["entry_date"], c["xd"])
    print("campaigns %d, chains preloaded" % len(camps))

    base = {}
    for c in camps:
        r = run(c, "HOLD", "MOVE", 0, pre[(c["expiry"], c["entry_date"], c["xd"])], spot, vx)
        if r:
            base[c["entry_date"]] = r[0]

    out = []
    print("\n%-10s %-9s %6s | %-7s %5s %9s %6s %8s | %-7s %5s %9s %6s %8s"
          % ("trigger", "arm", "level", "scope", "fires", "pairedD", "t", "helped",
             "scope", "fires", "pairedD", "t", "helped"))
    for kind, levels in (("MOVE", MOVES), ("VIX", VIXUPS)):
        for arm in ("ADD", "RECENTRE"):
            for lv in levels:
                line = "%-10s %-9s %5.1f%% |" % (kind, arm, lv)
                for scope, sel in (("VIX>25", lambda c: (c["vix_rank"] or 0) > 25),
                                   ("ALL", lambda c: True)):
                    d = []
                    for c in camps:
                        if not sel(c) or c["entry_date"] not in base:
                            continue
                        r = run(c, arm, kind, lv,
                                pre[(c["expiry"], c["entry_date"], c["xd"])], spot, vx)
                        if r and r[2]:
                            d.append(r[0] - base[c["entry_date"]])
                    if d:
                        mu, _, t = G.stats(d)
                        hp = "%d/%d" % (sum(1 for x in d if x > 0), len(d))
                        line += " %-7s %5d %+9.1f %6.2f %8s |" % (scope, len(d), mu, t, hp)
                        out.append(dict(trigger=kind, arm=arm, level=lv, scope=scope,
                                        fires=len(d), paired_delta=round(mu, 2),
                                        t=round(t, 2), helped=hp))
                    else:
                        line += " %-7s %5d %9s %6s %8s |" % (scope, 0, "-", "-", "-")
                print(line)
    with open(os.path.join(RES, "phase_j_triggers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    surv = [r for r in out if r["paired_delta"] > 0 and r["t"] > 2]
    print("\nSURVIVORS (paired delta > 0 AND t > 2): %s"
          % (", ".join("%s/%s@%s%% [%s]" % (r["trigger"], r["arm"], r["level"], r["scope"])
                       for r in surv) or "NONE"))
    print("wrote %s" % os.path.join(RES, "phase_j_triggers.csv"))


if __name__ == "__main__":
    sys.exit(main())
