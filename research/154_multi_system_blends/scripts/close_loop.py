"""research/154 - close the loop: research/INDEX.md row, TODO.md section, and the
Ops & Review Centre entries (one existing review resolved, one new one registered).
Idempotent."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/home/arun/quantifyd")
INDEX = ROOT / "research/INDEX.md"
TODO = ROOT / "TODO.md"
OPS = ROOT / "research/111_sensex_manual_mgmt/scripts/ops_center.py"

INDEX_ROW = (
    "| 154 | [Six-sleeve correlation & blend matrix](154_multi_system_blends/) - Arun: "
    "\"also find the correlations between each of these systems including our oa and tn in all "
    "possible combinations.\" Every pairwise correlation (15 pairs x daily/monthly x 3 panels), "
    "ALL 57 subsets of {OA, TN, VCP, MYB, IPO, GOLD} at equal weight, a weight sweep, and a full "
    "5%-grid frontier enumeration - all PAIRED across 360 paths (30 OA seeds x 12 TN "
    "rebalance-day offsets), after tax, net of 25 bps, monthly rebalanced, against three nulls "
    "(the deployed pair, a cash null, and an IPO BETA-MATCHED null because IPO is only 19.6% "
    "invested) | three explicitly separated panels: A 2010-01->2026-08 (all six), B "
    "2006-04->2026-08 (five; MYB cannot exist, contains 2008), C 2015-01->2026-08 (all six, no "
    "reconstructed data); **8,172 cells** | **STRATEGY (candidate) - the deployed pair is "
    "under-diversified and the fix is TWO satellites, not a third breakout sleeve.** "
    "**(1) A CORRECTION THAT CHANGES A STANDING BELIEF:** the deployed TN+OA pair's worst "
    "drawdown in 20 years is the 2008 crash at **-16.5%** (monthly marks) / -17.15% (daily), NOT "
    "the -2.4% r/146 and r/151 reported - those measured the window from 2008-01-01, AFTER the "
    "Dec-2007 peak. The claim that the TN gate + OA stops already stripped the crash tail is "
    "**withdrawn**; every per-window drawdown in r/146-153 needs re-auditing for the same "
    "artefact. **(2) THE BOOK OWNS ONE FACTOR:** OA<->VCP 0.749 daily / 0.767 monthly, and at "
    "POSITION level **87.0% of OA's signals are VCP signals** with 42-49% holding-day overlap - "
    "VCP is Open Alpha wearing another screen; MYB shares **90.2%** of its signals with VCP (the "
    "'75-93% overlap with OA' figure belonged to the raw family r/152 killed, not the adopted "
    "residual). Only two sleeves are different things: **IPO** (0.211 daily to OA, 0.220 to TN, "
    "and **0.0% signal AND 0.0% holding-day overlap - not one shared symbol-day in 16 years**) "
    "and **GOLD** (~0, negative monthly). **(3) THE FRONTIER:** 197 of 1,767 weight vectors clear "
    "the pre-registered bar on ALL THREE panels (CAGR >= the pair's, and beating the pair, the "
    "cash null AND the beta-matched null on >=288/360 paired paths each) - a broad CONTIGUOUS "
    "plateau, and every admitted vector holds gold while the best ones cut TN. RECOMMENDED with "
    "operational constraints (keep both live books, IPO <=20%, gold <=20%): **OA 40 / TN 25 / "
    "IPO 20 / GOLD 15 = 28.21% / -10.77% / Calmar 2.61** vs the pair's 27.74% / -17.01% / 1.68, "
    "360/360 vs the pair, 360/360 vs cash, 358/360 vs beta-matched; 2008 becomes +7.3% at -7.5%. "
    "DEPLOYABLE-TODAY (no unproven sleeve): **OA 60 / TN 15 / GOLD 25 = 28.02% / -13.31% / "
    "2.095**. r/147's 45/45/10 is **NOT admitted** (CAGR shortfall -1.13pp). **REGISTERED "
    "QUESTIONS ANSWERED:** r/152's MYB+OA-beats-TN+OA reproduces (+0.316 Calmar, 314/360) but is "
    "**not actionable** - MYB's 3-year pivot makes 2008 unreachable by construction, and every "
    "2006-testable substitute that wins does so on being uncorrelated, which MYB is not (0.412 "
    "to OA); r/152's 80/10/10 four-sleeve probe is **REFUTED** against a properly specified "
    "gold-only null at the same satellite weight (-0.094 Calmar, wins only 91/360). **DATA "
    "DEFECT FOUND AND FIXED:** r/147's cached gold-INR reference was missing **40 of 274 months** "
    "(Yahoo monthly candles drop months and their UTC-offset stamps collide across month "
    "boundaries); rebuilt at DAILY resolution with zero gaps, monthly correlation to real "
    "GOLDBEES 0.878 (was 0.788), daily 0.390 so daily gold correlations use real data only "
    "| 2026-09-05 | **STRATEGY candidate - two satellites; Arun decides; published at "
    "/app/backtest/multi-system-blends-research154** |\n"
)

TODO_BLOCK = """
### research/154 — Six-sleeve correlation & blend matrix — DONE 2026-09-05, verdict STRATEGY (candidate)

- **The deployed TN+OA pair's true 2008 drawdown is −16.5%, not −2.4%.** r/146 and r/151
  measured the 2008 window from 2008-01-01, which is after the Dec-2007 peak. 2008 is the
  pair's single deepest hole in twenty years. The standing claim that the TN gate plus OA's
  stops "already stripped the crash tail" is **withdrawn**.
  → **PENDING:** re-audit every per-window drawdown figure in r/146 through r/153 for the same
  window-start artefact.
- **VCP is Open Alpha.** 87.0% of OA's signals are VCP signals; 48.6% / 41.5% holding-day
  overlap; correlation 0.749 daily. MYB shares 90.2% of its signals with VCP. Both are retired
  from consideration permanently.
- **OA and IPO have never once held the same stock on the same day** (0.0% signal and 0.0%
  holding-day overlap, 2010–2026), at correlation 0.211 daily. Gold is ~0 to everything.
- **197 of 1,767 enumerated weight vectors** clear the pre-registered bar on all three panels
  against three nulls — a contiguous plateau. Recommended (constrained):
  **OA 40 / TN 25 / IPO 20 / GOLD 15 → 28.21% / −10.77% / Calmar 2.61** vs the pair's
  27.74% / −17.01% / 1.68 on 2006-04→2026-08.
  Deployable today without an unproven sleeve: **OA 60 / TN 15 / GOLD 25 → 28.02% / −13.31% /
  2.095**. r/147's 45/45/10 is NOT admitted (CAGR shortfall).
- **IPO is 80% cash** (19.6% invested; zero trades in 2013 and 2014). A cash null does not
  catch that, so a **beta-matched null** was built (IPO → 19.6% OA + 80.4% cash). Beyond ~20%
  IPO weight the extra Calmar is indistinguishable from de-levering on two of three panels.
- **Both r/152 open questions answered:** MYB+OA reproduces but is not actionable (2008 is
  unreachable by construction for a 3-year-high screen); the 80/10/10 four-sleeve probe is
  **REFUTED** against a gold-only null at the same satellite weight.
- **Data defect fixed:** r/147's gold-INR reference series was missing 40 of 274 months.
  Rebuilt at daily resolution, zero gaps, monthly correlation to real GOLDBEES 0.878.
  Lives in `research/154_multi_system_blends/results/gold_nav.csv` — never in market_data.db.
- Published at `/app/backtest/multi-system-blends-research154`.
- **ARUN DECIDES.** Nothing deployed; no live engine, crontab or spec touched.
  Dated obligation registered in the Ops & Review Centre (due 2026-10-15, merged with the
  r/153 adoption call; the r/152 four-sleeve review is marked DONE by this study).
"""


def main():
    # ---- INDEX.md
    t = INDEX.read_text(encoding="utf-8")
    if "154_multi_system_blends" not in t:
        if not t.endswith("\n"):
            t += "\n"
        INDEX.write_text(t + INDEX_ROW, encoding="utf-8")
        print("INDEX.md row appended")
    else:
        print("INDEX.md already has the row")

    # ---- TODO.md
    t = TODO.read_text(encoding="utf-8")
    if "research/154 — Six-sleeve" not in t:
        TODO.write_text(t.rstrip("\n") + "\n" + TODO_BLOCK, encoding="utf-8")
        print("TODO.md section appended")
    else:
        print("TODO.md already has the section")

    # ---- ops_center.py
    src = OPS.read_text(encoding="utf-8")
    if "research/154" in src:
        print("ops_center already updated")
        return
    # 1) the r/152 four-sleeve review is DELIVERED by this study
    old_status = ('     "2026-11-30", "SCHEDULED",\n'
                  '     "research/152 found the multi-year-breakout sleeve (MYB) and r/147\'s gold sleeve fail in "')
    new_status = ('     "2026-11-30", "DONE",\n'
                  '     "DELIVERED BY research/154 on 2026-09-05 - both questions in this review are answered, "\n'
                  '     "see the new research/154 entry below. Original text follows. "\n'
                  '     "research/152 found the multi-year-breakout sleeve (MYB) and r/147\'s gold sleeve fail in "')
    assert old_status in src, "could not locate the r/152 four-sleeve review status"
    src = src.replace(old_status, new_status, 1)

    # 2) register the new review right after the REVIEWS opening
    anchor = "REVIEWS = [\n"
    entry = (
        '    ("research/154 six-sleeve blend - allocation decision (2 satellites) + the r/146-153 '
        'drawdown re-audit",\n'
        '     "2026-10-15", "PENDING",\n'
        '     "research/154 enumerated every combination of the six sleeves (OA, TN, VCP, MYB, IPO, '
        'GOLD) - 8,172 cells, all PAIRED across 360 paths (30 OA seeds x 12 TN rebalance-day '
        'offsets) - and produced three things Arun must act on. (1) A RETRACTION THAT COMES FIRST: '
        'the deployed TN+OA pair\'s worst drawdown in twenty years is the 2008 crash at -16.5% '
        '(monthly marks) / -17.15% (daily), NOT the -2.4% that research/146 and /151 reported. Those '
        'studies measured the 2008 window starting 2008-01-01, i.e. AFTER the December-2007 peak, so '
        'the drawdown from that peak was invisible. ACTION OWED AT THIS REVIEW: re-audit every '
        'per-window drawdown figure in research/146 through /153 for the same window-start artefact, '
        'and re-open the crash-alpha candidates that r/146 rejected on the (now false) basis that the '
        'pair has no crash tail. (2) THE ALLOCATION DECISION: 197 of 1,767 enumerated weight vectors '
        'clear the pre-registered bar on ALL THREE panels (median CAGR at least the pair\'s, and '
        'beating the pair, a cash null AND an IPO beta-matched null on >=288/360 paired paths each) - '
        'a broad CONTIGUOUS plateau, not a peak. With operational constraints applied (keep both live '
        'books, cap the never-traded sleeve at 20%, cap gold at 20%) the best vector is OA 40 / TN 25 '
        '/ IPO 20 / GOLD 15 = 28.21% CAGR / -10.77% MaxDD / Calmar 2.61 against the deployed pair\'s '
        '27.74% / -17.01% / 1.68 on 2006-04 to 2026-08, winning 360/360 paths against the pair, '
        '360/360 against cash and 358/360 against the beta-matched null; its 2008 is +7.3% at a -7.5% '
        'drawdown. The DEPLOYABLE-TODAY step, using no unproven sleeve, is OA 60 / TN 15 / GOLD 25 = '
        '28.02% / -13.31% / Calmar 2.095. NOTE research/147\'s 45/45/10 is NOT admitted - it fails '
        'CAGR-at-least-the-pair by 1.13pp on the 2006+ window; gold pays for its drawdown cut with '
        'return unless Open Alpha\'s weight rises to fund it. (3) TWO KILLS TO RECORD: VCP is Open '
        'Alpha (87.0% of OA\'s signals are VCP signals, 48.6% holding-day overlap, correlation 0.749) '
        'and MYB shares 90.2% of its signals with VCP - retire both permanently. IPO is the opposite: '
        '0.0% signal AND 0.0% holding-day overlap with OA - not one shared symbol-day in sixteen '
        'years. PRE-CONDITION ON ANY IPO WEIGHT: it has never traded, live or paper, and it is '
        'invested only 19.6% of NAV (zero trades in 2013 and in 2014), so beyond about 20% weight its '
        'extra Calmar is indistinguishable from de-levering. The paper-soak criterion already '
        'registered under the research/153 review applies unchanged and must clear BEFORE any rupee. '
        'ALSO OWED BEFORE DEPLOYMENT: a quarterly-rebalance sensitivity, because blend-level '
        'rebalancing turnover and its tax are NOT modelled. Artifacts: '
        'research/154_multi_system_blends/results/RESULTS.md, p1_correlations.csv, p2_subsets.csv, '
        'p3_weights.csv, p5_windows.csv, p6_overlap.csv, p7_frontier_OA_TN_IPO_GOLD.csv, '
        'p8_daily_marked.csv, p8_yoy.csv, gold_nav.csv (a REBUILT daily gold-in-rupee reference, zero '
        'missing months, monthly correlation 0.878 to real GOLDBEES - replaces research/147\'s cached '
        'monthly series which was missing 40 of its 274 months). Published at '
        '/app/backtest/multi-system-blends-research154."),\n'
    )
    assert anchor in src
    src = src.replace(anchor, anchor + entry, 1)
    OPS.write_text(src, encoding="utf-8")
    print("ops_center.py: r/152 four-sleeve review marked DONE, research/154 review registered")


if __name__ == "__main__":
    main()
