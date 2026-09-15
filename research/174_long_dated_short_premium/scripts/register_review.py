#!/usr/bin/env python3
"""research/174 — register the dated reviews in the Ops & Review Centre.

Idempotent: re-running does nothing if the entries are already present.
"""
from pathlib import Path

OPS = Path("/home/arun/quantifyd/research/111_sensex_manual_mgmt/scripts/ops_center.py")
MARK = "research/174 - NIFTY long-dated"

ENTRIES = '''REVIEWS = [
    ('research/174 - NIFTY long-dated short premium: has NSE listing or liquidity changed enough to reopen tenors past 45 DTE?', '2027-09-15', 'PENDING', "RAISED 15-Sep-2026 by research/174, which CLOSED the tenor line. The kill rests on two measurable facts, both re-measurable in one command: (1) the at-the-money call at 365 DTE trades 54 contracts a day and at 730 DTE four, so the long contracts are untradeable at any size; (2) past ~105 DTE the strike grid widens to 1,000-1,500 points, so the nearest listed strike is 1.0-1.2% from spot and a long-dated ATM straddle cannot be placed at all. Both would change if NSE tightened the long-dated strike grid or long-dated volume grew. HOW TO CHECK: rerun research/174_long_dated_short_premium/scripts/probe_liquidity_v2.py then summarise_liquidity_v2.py; reopen ONLY if ATM volume at 180-365 DTE has risen above ~500 contracts/day AND the strike step has fallen below ~500 points. Otherwise re-date this review for another year. The economics kill is separate and stronger: entering at T and exiting at 21 DTE, net points a trade fall monotonically 45d +56.6 (t 2.96), 60d +46.3, 75d +30.5, 90d -5.5, 180d -70.4, 210d -164.3. Evidence: research/174_long_dated_short_premium/results/RESULTS.md."),
    ('research/174 - the SHORT-PREMIUM MANAGEMENT question is closed after EIGHT refutations. Is any new proposal being made to cite them?', '2027-03-15', 'PENDING', "RAISED 15-Sep-2026 by research/174. Stops and re-centring on a short-premium book have now been refuted eight independent times: research/119 phase E (move-triggered re-centring, 0 of 63 cells beat holding), research/119 phase G (premium stop, -130.9 pts at t -2.34), research/127 phase B3, research/128, research/129, research/130, research/135, and research/174 (seven families x every tenor, ALL paired, none with a positive delta at t above 1.4). research/174 also supplied the MECHANISM, which is the reusable part: chop the 45-to-21 DTE hold into five five-day pieces that each re-pick the ATM strike, and +68.5 gross / +56.6 net becomes +45.2 gross / MINUS 12.8 net. Re-centring gives up a third of the gross edge and pays four extra round trips. Every individual DTE window across the whole contract life is statistically indistinguishable from zero - the return comes from HOLDING ONE STRIKE THROUGH THE DRIFT, not from harvesting theta. Also settled there: the SINGLE-SIDE stop, the most promising untested idea, is indistinguishable from a whole-position stop at matched fire rates, so the theta-preservation hypothesis is dead. And: stops ARE real insurance at a real premium (VIX rank above 60 cuts the worst trade from -1,049 to -464 points) - if a smaller tail is ever wanted, size down instead, per research/119. CHECK at the due date: any short-premium management proposal raised since must cite this list and say why it is not a ninth."),
'''


def main():
    src = OPS.read_text(encoding="utf-8")
    if MARK in src:
        print("already registered; nothing to do")
        return
    assert "REVIEWS = [\n" in src, "REVIEWS block not found"
    src = src.replace("REVIEWS = [\n", ENTRIES, 1)
    OPS.write_text(src, encoding="utf-8")
    print("registered 2 reviews in %s" % OPS)


if __name__ == "__main__":
    main()
