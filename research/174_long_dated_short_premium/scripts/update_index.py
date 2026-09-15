#!/usr/bin/env python3
"""research/174 — append the INDEX.md row. Idempotent."""
from pathlib import Path

IDX = Path("/home/arun/quantifyd/research/INDEX.md")
ROW = (
    "| 174 | [**NIFTY long-dated short premium - 2 / 3 / 6 / 12-month straddles, strangles "
    "and condors, and every stop family Arun listed**]"
    "(174_long_dated_short_premium/results/RESULTS.md) - Arun 15-Sep-2026: *\"now that we "
    "are live with 45 DTE, can we test for more like 2 months away straddles/strangles/"
    "condors with different DTE entries and DTE exits and different stop losses, same for "
    "other liquid ones 3 months, 6 months away, 1 year away etc?\"* plus *\"sls can be "
    "combined premium, single side, underlying price move, vix, relative vix or "
    "combinations\"* | 2015-01 to 2026-09, real NSE bhavcopy, daily close only | "
    "**Tenor is a MONOTONE DECAY from 45 outward, not a peak.** Fixed rule (enter at T, "
    "exit 21 DTE, 0.75% slippage, 25-contract liquidity floor): 45d +56.6 pts/trade t 2.96 "
    "-> 60d +46.3 t 1.36 -> 75d +30.5 t 0.78 -> 90d -5.5 -> 105d -32.2 -> 180d -70.4 -> "
    "210d -164.3; win rate 73.6 / 66.7 / 59.0 / 48.0. Of **624 cells exactly TWO clear "
    "t = 2 and both are the live book**. 365 DTE and beyond are UNTRADEABLE (ATM call 54 "
    "contracts/day; 730 DTE four) and UNPLACEABLE (strike grid 1,000-1,500 pts past ~105 "
    "DTE, nearest strike 3% from spot). Selling more premium earns less: 365d collects "
    "1,906 pts vs 640, nets nothing, blocks 2.7x the margin for 12x as long. **Eighth "
    "refutation of stops** - all seven families (premium / move / single-side / "
    "single-side-on-strike / VIX level / VIX rank / VIX-vs-entry) lose PAIRED at every "
    "tenor, damage monotone in fire rate; single-side is indistinguishable from "
    "whole-position at matched fire rates, killing the theta-preservation hypothesis. "
    "**The mechanism, reusable:** chopping the 45->21 hold into five re-centred 5-day "
    "pieces turns +68.5 gross / +56.6 net into +45.2 gross / **-12.8 net** - every "
    "individual DTE window is indistinguishable from zero, so the return is HOLDING ONE "
    "STRIKE THROUGH THE DRIFT, not theta harvest. Condors and wings dead at all six "
    "tenors (consistent with r/128; not a contradiction of r/127, whose wings work on "
    "idiosyncratic STOCK tails). Exit DTE NOT RESOLVABLE at n=140 (9/14/21/27 overlap at "
    "+-2SE). VIX-rank entry filter validated and monotone (27.8 / 37.3 / 41.5 %/yr on "
    "measured margin at off / >25 / >50). A 5% strangle looked better unpaired and LOSES "
    "paired (-26.4 pts median, wins 55/140) - claim retracted mid-study. **By-product: "
    "repaired a repo-wide defect** - nse_options_bhav was missing every NIFTY expiry "
    "beyond ~75 DTE for 2016-01 to 2024-02 + 2026-04/05/06 + 2026-09 (cause: MAX_DTE=75 "
    "in the r/89 downloader, which resumed by trade-date so the uncapped production "
    "downloader skipped those sessions forever); 2,102 sessions re-downloaded, "
    "**+2,836,719 rows merged**, all years now complete. Published "
    "/app/backtest/longdated-short-premium-research174 | "
    "**NO EDGE beyond 45 DTE - CONCLUDED** |\n"
)


def main():
    s = IDX.read_text(encoding="utf-8")
    if "| 174 |" in s:
        print("row already present")
        return
    if not s.endswith("\n"):
        s += "\n"
    IDX.write_text(s + ROW, encoding="utf-8")
    print("appended r/174 row")


if __name__ == "__main__":
    main()
