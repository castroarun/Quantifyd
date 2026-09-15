#!/usr/bin/env python3
"""research/174 - insert the DONE block into TODO.md, after the header. Idempotent."""
from pathlib import Path

TODO = Path("/home/arun/quantifyd/TODO.md")
MARK = "2026-09-15 - NIFTY long-dated short premium"

BLOCK = """## DONE 2026-09-15 - NIFTY long-dated short premium: 2 / 3 / 6 / 12-month tenors are all worse than the live 45-DTE book, and no stop beats no stop

Arun: *"now that we are live with 45 DTE, can we test for more like 2 months away
straddles/strangles/condors with different DTE entries and DTE exits and different stop losses,
same for other liquid ones 3 months, 6 months away, 1 year away etc?"* and *"sls can be combined
premium, single side, underlying price move, vix, relative vix or combinations."*
research/174, published at `/app/backtest/longdated-short-premium-research174`.

**NO EDGE beyond 45 DTE - CONCLUDED. Nothing deployed, nothing live touched.**

**Every tenor Arun named loses, and the decay is monotone rather than a peak.** Fixed rule -
enter at tenor T, exit at 21 DTE, no stop, 0.75% slippage, 25-contract liquidity floor on both
legs, 2015-2026: 45d +56.6 pts a trade at t 2.96, 60d +46.3 at t 1.36, 75d +30.5 at t 0.78, 90d
**-5.5**, 105d -32.2, 180d -70.4, 210d -164.3. Win rate falls smoothly 73.6 / 66.7 / 59.0 / 48.0;
drawdown grows 1,243 / 3,671 / 4,795 points. **Of 624 cells exactly TWO clear t = 2, and both are
the live book.** Selling more premium earns less: the 365-day straddle collects 1,906 points
against 640, nets nothing, and blocks 2.7x the margin for twelve times as long.

**One year and beyond is killed on the data, not the P&L.** The at-the-money call at 365 DTE
trades **54 contracts a day**; the two-year contracts trade **four**. And past ~105 DTE the strike
grid widens to 1,000-1,500 points, so the nearest listed strike is 3% from spot - a long-dated ATM
straddle cannot be placed at all.

**The eighth refutation of stops, including the three genuinely new families.** All seven families
lose PAIRED against no-stop at every tenor, and the damage is monotone in how often the stop fires
(move stop: 86% fired -49.7 pts, 61% -68.5, 41% -65.2, 25% -58.6, 10% -22.7). **Single-side was the
most promising untested idea and is refuted specifically**: at matched fire rates it is
indistinguishable from a whole-position stop, so the theta-preservation hypothesis is dead. Stops
ARE real insurance at a real premium - VIX rank above 60 cuts the worst trade from -1,049 to -464
points - but research/119 already showed the cheaper way to buy that is fewer lots.

**The mechanism, which is the reusable part.** Chop the 45-to-21 DTE hold into five five-day pieces
that each re-pick the ATM strike: +68.5 gross / +56.6 net becomes +45.2 gross / **-12.8 net**.
Re-centring gives up a third of the gross edge and pays four extra round trips. Every individual
DTE window across the contract life is statistically indistinguishable from zero. **The return is
holding ONE STRIKE through the drift, not harvesting theta** - which is why research/119 phases E
and G failed, and why all seven stop families fail here.

**Also settled.** Condors and winged straddles are dead on the index at all six tenors (consistent
with research/128; NOT a contradiction of research/127, whose wings work on idiosyncratic stock
tails). The exit-DTE choice is **not resolvable** at n=140 - 9, 14, 21 and 27 DTE have overlapping
+-2SE intervals, so the live 21-DTE rule is defensible and unrefuted but should not be called
optimised. The live **VIX-rank entry filter is validated and monotone**: 27.8 / 37.3 / 41.5 percent
a year on measured margin at off / above-25 / above-50, both halves positive throughout.

**A correction I made to myself mid-study.** An interim table showed the 45-DTE 5% strangle beating
the live straddle on t-stat, worst trade and return on margin. Paired on the same 140 entry days it
**loses** - median -26.4 points, wins on 55 of 140. The advantage was lower variance and a smaller
margin block, not more earnings. Retracted. It remains open only as a capital-allocation question
(does freeing Rs 0.65L per lot and giving up 26 points a trade help the whole book?), which is a
Capital Desk question and would need its own study and its own deploy.

**By-product: a repo-wide data defect found and permanently repaired.** `nse_options_bhav` was
missing every NIFTY expiry beyond ~75 DTE for **2016-01 to 2024-02**, plus 2026-04/05/06 and
2026-09 - eight years - because `MAX_DTE = 75` in research/89's stock-bhav downloader resumed by
trade-date, marking those sessions done so the uncapped production downloader skipped them forever.
2,102 sessions re-downloaded with zero errors, **+2,836,719 rows merged**, every year 2015-2026 now
complete. The download ran during the session but wrote to a staging file; the merge was held until
15:40 IST so it never took a write lock on the database the live executors read.

**Registered:** a 15-Sep-2027 review that re-measures long-dated listing and liquidity before this
line may be reopened (with the exact thresholds that would justify it), and a 15-Mar-2027 review
requiring any future short-premium management proposal to cite the eight refutations and say why it
is not a ninth.

---

"""


def main():
    s = TODO.read_text(encoding="utf-8")
    if MARK in s:
        print("already present")
        return
    anchor = "Cross-session source of truth for pending work. Each item: what / why / when.\n\n"
    assert anchor in s, "TODO header not found"
    s = s.replace(anchor, anchor + BLOCK, 1)
    TODO.write_text(s, encoding="utf-8")
    print("TODO.md updated")


if __name__ == "__main__":
    main()
