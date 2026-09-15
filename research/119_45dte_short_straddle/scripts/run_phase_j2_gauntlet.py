#!/usr/bin/env python3
"""Phase J2 — the gauntlet for the one Phase J survivor: VIX-triggered ADD.

VIX/ADD@15% passed the pre-registered paired test on both scopes (t 3.04 / 4.01).
First management rule in nine phases to do so. Five things must hold before it
is anything more than an interesting cell:

  1. PLATEAU  - neighbouring trigger levels (10/12/15/20/25%) must also work.
                A lone spike at 15% is noise.
  2. ERA      - both halves of the sample (<=2022 vs 2023+) must agree in sign.
  3. SIZE     - "ADD at the new ATM" vs "ADD a second unit of the ORIGINAL strike
                at that day's price" on the SAME fired campaigns. If they match,
                the gain is just more size after a vol spike (not re-centring)
                and the honest comparison is against trading 2 lots always.
  4. RISK     - the ADD arm's campaign-level MaxDD and worst campaign, next to
                HOLD's. Doubling into a vol spike doubles exposure exactly when
                margin is dearest (31-Aug stress: 3 lots at +/-8% = 110% of the
                ring-fence). A P&L win that needs 2x reserved capital is judged
                on 2x capital.
  5. HOLD-2x  - the r/134 null: on the fired campaigns, what does simply running
                the ORIGINAL at 2x from ENTRY earn vs ADD? If HOLD-2x >= ADD,
                the rule is dominated by sizing.

Read-only, reuses Phase G / Phase J machinery.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_phase_g_addstraddle as G      # noqa: E402
import run_phase_j_triggers as J         # noqa: E402

LEVELS = [10.0, 12.0, 15.0, 20.0, 25.0]


def run_add_same(camp, level, chains, spot, vx):
    """VIX trigger -> add a second unit at the ORIGINAL strike, that day's price."""
    K0, C0 = float(camp["strike"]), float(camp["credit"])
    V0 = J.vix_level(vx, camp["entry_date"])
    days = sorted(d for d in chains if camp["entry_date"] < d <= camp["xd"])
    if not days or not V0:
        return None
    book = [dict(K=K0, credit=C0)]
    realised = cost = 0.0
    fired = 0
    for i, d in enumerate(days):
        ch = chains[d]
        last = (i == len(days) - 1)
        p0 = G.straddle(ch, K0)
        if p0 is None:
            continue
        if not fired and not last:
            V = J.vix_level(vx, d)
            if V is not None and (V / V0 - 1.0) * 100.0 >= level:
                pn = G.straddle(ch, K0, need_liquid=True)
                if pn is None:
                    continue
                book.append(dict(K=K0, credit=pn))
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
    return realised - cost, fired


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
    ch_of = lambda c: pre[(c["expiry"], c["entry_date"], c["xd"])]
    hold = {}
    for c in camps:
        r = J.run(c, "HOLD", "MOVE", 0, ch_of(c), spot, vx)
        if r:
            hold[c["entry_date"]] = r[0]
    scopes = (("VIX>25", lambda c: (c["vix_rank"] or 0) > 25), ("ALL", lambda c: True))

    # ---- 1. plateau -------------------------------------------------------
    print("=" * 84)
    print("1. PLATEAU  - VIX-triggered ADD at the new ATM, neighbouring levels")
    print("=" * 84)
    print("  %-6s | %-7s %5s %9s %6s %7s | %-7s %5s %9s %6s %7s"
          % ("level", "scope", "n", "pairedD", "t", "helped", "scope", "n", "pairedD", "t", "helped"))
    for lv in LEVELS:
        line = "  %5.0f%% |" % lv
        for sc, sel in scopes:
            d = []
            for c in camps:
                if not sel(c) or c["entry_date"] not in hold:
                    continue
                r = J.run(c, "ADD", "VIX", lv, ch_of(c), spot, vx)
                if r and r[2]:
                    d.append(r[0] - hold[c["entry_date"]])
            if len(d) >= 3:
                mu, _, t = G.stats(d)
                line += " %-7s %5d %+9.1f %6.2f %7s |" % (
                    sc, len(d), mu, t, "%d/%d" % (sum(1 for x in d if x > 0), len(d)))
            else:
                line += " %-7s %5d %9s %6s %7s |" % (sc, len(d), "-", "-", "-")
        print(line)

    # ---- 2. era split at 15% ----------------------------------------------
    print("\n" + "=" * 84)
    print("2. ERA SPLIT at 15%  (both halves must agree in sign)")
    print("=" * 84)
    for era, sel_e in (("<= 2022", lambda c: c["entry_date"] <= "2022-12-31"),
                       (">= 2023", lambda c: c["entry_date"] >= "2023-01-01")):
        d = []
        for c in camps:
            if not sel_e(c) or c["entry_date"] not in hold:
                continue
            r = J.run(c, "ADD", "VIX", 15.0, ch_of(c), spot, vx)
            if r and r[2]:
                d.append(r[0] - hold[c["entry_date"]])
        if len(d) >= 3:
            mu, _, t = G.stats(d)
            print("  %-8s n=%2d  pairedD %+7.1f  t %5.2f  helped %d/%d"
                  % (era, len(d), mu, t, sum(1 for x in d if x > 0), len(d)))
        else:
            print("  %-8s n=%2d  (too few)" % (era, len(d)))

    # ---- 3+4+5. size, risk, hold-2x on the fired campaigns ---------------
    print("\n" + "=" * 84)
    print("3-5. ON THE CAMPAIGNS WHERE VIX/ADD@15% FIRED: is it re-centring, or just size?")
    print("=" * 84)
    print("  %-7s %4s %10s %10s %10s %10s | %9s %9s | %9s %9s"
          % ("scope", "n", "HOLD", "ADD_ATM", "ADD_SAME", "HOLD-2x",
             "worstHOLD", "worstADD", "ddHOLD", "ddADD"))
    for sc, sel in scopes:
        H, A, S, fired_set = [], [], [], []
        for c in camps:
            if not sel(c) or c["entry_date"] not in hold:
                continue
            ra = J.run(c, "ADD", "VIX", 15.0, ch_of(c), spot, vx)
            if not (ra and ra[2]):
                continue
            rs = run_add_same(c, 15.0, ch_of(c), spot, vx)
            if not (rs and rs[1]):
                continue
            H.append(hold[c["entry_date"]]); A.append(ra[0]); S.append(rs[0])
        if not H:
            continue
        n = len(H)
        h2 = [2 * x for x in H]
        print("  %-7s %4d %+10.1f %+10.1f %+10.1f %+10.1f | %+9.1f %+9.1f | %+9.1f %+9.1f"
              % (sc, n, sum(H) / n, sum(A) / n, sum(S) / n, sum(h2) / n,
                 min(H), min(A), G.maxdd(H), G.maxdd(A)))
        da = [a - h for a, h in zip(A, H)]
        ds = [s - h for s, h in zip(S, H)]
        dx = [a - s for a, s in zip(A, S)]
        for lab, d in (("ADD_ATM - HOLD", da), ("ADD_SAME - HOLD", ds),
                       ("ADD_ATM - ADD_SAME (re-centring's own worth)", dx)):
            mu, _, t = G.stats(d)
            print("      %-46s %+7.1f  t %5.2f" % (lab, mu, t))
        d2 = [a - x for a, x in zip(A, h2)]
        mu, _, t = G.stats(d2)
        print("      %-46s %+7.1f  t %5.2f   <- the sizing null"
              % ("ADD_ATM - HOLD-2x (same peak capital)", mu, t))

    # ---- book-level: HOLD vs ADD@15 across ALL campaigns (ADD only fires on some)
    print("\n" + "=" * 84)
    print("BOOK LEVEL, all campaigns (ADD@15% fires on ~1 in 3; the rest are HOLD)")
    print("=" * 84)
    for sc, sel in scopes:
        H, A = [], []
        for c in camps:
            if not sel(c) or c["entry_date"] not in hold:
                continue
            ra = J.run(c, "ADD", "VIX", 15.0, ch_of(c), spot, vx)
            H.append(hold[c["entry_date"]]); A.append(ra[0] if ra else hold[c["entry_date"]])
        mh, _, th = G.stats(H); ma, _, ta = G.stats(A)
        print("  %-7s n=%2d  HOLD %+7.1f/camp (t %.2f, maxDD %+8.1f)   ADD@15 %+7.1f/camp (t %.2f, maxDD %+8.1f)"
              % (sc, len(H), mh, th, G.maxdd(H), ma, ta, G.maxdd(A)))


if __name__ == "__main__":
    sys.exit(main())
