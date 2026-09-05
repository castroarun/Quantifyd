# -*- coding: utf-8 -*-
"""research/153 — close the loop: research/INDEX.md row, TODO.md section, and the dated
review in the Ops & Review Centre. Run ON THE VPS. Idempotent."""
from pathlib import Path

ROOT = Path("/home/arun/quantifyd")

# ─────────────────────────────────────────────────────────── 1. research/INDEX.md
IDX = ROOT / "research" / "INDEX.md"
s = IDX.read_text(encoding="utf-8")
if "153_ipo_base" not in s:
    row = (
        '| 153 | [IPO Base breakout — the bananapatterns "IPO Base" screen](153_ipo_base/) — '
        "the first proper base a NEWLY LISTED stock builds, and the breakout out of it. "
        "r/142 engine extended with a vetted listing table, base depth/length dials, four RS "
        "policies, the site's +25% take-profit and risk-based sizing, and Indian-FY tax netting "
        "with loss carry-forward | daily 2006-01 -> 2026-09-04 (W2) and 2020-2025 (W1); "
        "**680 cells** (256 signal-geometry + 384 exits-x-book + ~40 controls); 10-seed scan / "
        "30-seed adoption; after tax, 25/40/60 bps, idle cash 5% | "
        "**THE STUDY WAS THE DATA PROBLEM.** No listing-date table exists and the naive proxy "
        "(first row per symbol) is only **70% accurate**: bulk DATA-ONBOARDING WAVES masquerade "
        "as IPOs (451 symbols start 2005-01-03, 95 on 2015-01-01, 45 on 2026-08-17, 41 on "
        "2026-04-20, **15 on 2025-05-26 including ABB, listed in the 1990s**), and PRE-LISTING "
        "JUNK ROWS sit on reused tickers (DELHIVERY carries 8 rows at Rs 5-11 from 2016 before "
        "its real Rs 536 listing — a **93x jump inside what a base window would measure**; also "
        "FUSION 97x, LATENTVIEW, SBICARD, STARHEALTH, MAZDOCK). A vetted table was built and "
        "**validated before any backtest ran: 48/48 known NSE IPOs accepted, listing date exact "
        "to +/-3 days for 47/48, 0/12 known onboardings leaked** -> 1,293 accepted listings, 786 "
        "tradeable. **RS >= 70 is IMPOSSIBLE on this screen** — the 252-day IBD score returns "
        "ZERO signals below a 12-month age band and every short-window substitute costs ~8pp of "
        "CAGR; the screen is pure price structure. **Two of the site's own dials are the WORST "
        'settings tested**: "Trail 30-week" (SMA-150) Calmar 0.49 vs 0.99 for SMA-20, and '
        '"Breakout close" costs **-14.08pp CAGR** vs a buy-stop at the pivot, losing 30/30 '
        "paired seeds — the entire edge lives in the entry price (the r/142 lesson again). Their "
        '"+25% take-profit" is the best dial found and wins in EVERY geometry. **207/256 and '
        "383/384 cells clear positive after-cost per-trade expectancy in BOTH windows** — a broad "
        "plateau, not a peak. Age band is a clean dose-response and decides what you own: <=3m "
        "24.10% / -13.8% / Calmar 1.74 (only 19.6% invested, biggest null edge +2.08pp/trade); "
        "**<=6m ADOPTED 31.03% [28.82..33.44] / -20.88% (worst seed -23.23) / Calmar 1.50**, 32.6 "
        "trades/yr, +4.89%/trade net, 49% win; <=24m 35.99% / -33.3% but by then it is no longer "
        "an IPO base and converges on Open Alpha. Beats a **date-matched random-entry null** by "
        "+0.96pp/trade on 29/30 paired seeds; survives deleting each seed's 10 best trades "
        "(5.39 -> 4.89%/trade, top-10 are only 11% of the summed return); survives 60 bps; the "
        "market gate LOSES on 0/30 seeds. **Complement test PASSES every pre-registered leg**: at "
        "20% weight beside TN+OA it gives **+1.13pp CAGR, -3.63pp drawdown, +0.56 Calmar** "
        "(27.14/-16.42/1.65 -> 28.27/-12.79/**2.21**) at correlation 0.16 daily to OA and 0.18 to "
        "TN — LOWER than OA<->TN at 0.42 — and beats a plain-cash sleeve by +5.60pp CAGR. On the "
        "common 2015+ window it beats r/147's gold as a complement on return (32.01 vs 27.03 at "
        "20%) at comparable Calmar: **gold buys Calmar by lowering return, this buys it while "
        "raising return**; the exploratory 4-sleeve 40/40/10g/10ipo = 29.05% / -11.55% / **2.52**. "
        "Capacity comfortable to ~Rs 10 cr, binding near Rs 50 cr. Survivorship MEASURED (the DB "
        "retains dead series: 41/334 traded names later go stale = 5.1% of trades at +2.70% mean "
        "vs +5.52%). Honest costs: **2013 and 2014 earned only the cash yield — no trades at "
        "all**, the IPO pipeline supplied 8-17 usable listings/yr in 2012-14 vs 80-182 in "
        "2021-25; and **no replication gate was run** (the panel dials were never legible) "
        "| 2026-09-05 | **STRATEGY candidate — third sleeve at 10-20%; Arun decides adoption; "
        "published at /app/backtest/ipo-base-breakout-research153** |\n"
    )
    IDX.write_text(s.rstrip("\n") + "\n" + row, encoding="utf-8")
    print("INDEX.md row added")
else:
    print("INDEX.md already has 153")

# ─────────────────────────────────────────────────────────── 2. TODO.md
TODO = ROOT / "TODO.md"
t = TODO.read_text(encoding="utf-8")
if "research/153 IPO Base" not in t:
    hdr = ("# Covered_Calls — TODO\n\nCross-session source of truth for pending work. "
           "Each item: what / why / when.\n")
    block = """
## ✅ 2026-09-05 — research/153 IPO Base breakout: STRATEGY CANDIDATE — the first third sleeve to clear every leg of the bar

Full verdict: `research/153_ipo_base/results/RESULTS.md` · study page
`/app/backtest/ipo-base-breakout-research153`.

**The study was really a data-integrity study.** We have no listing-date table, and the obvious
proxy (a symbol's first row in `market_data_unified`) is only **70% accurate**. Bulk
data-onboarding waves masquerade as IPOs — 451 symbols start on 2005-01-03, 15 on 2025-05-26
including **ABB, listed in the 1990s** — and pre-listing junk rows sit on reused tickers
(DELHIVERY carries 8 rows at ₹5–11 from 2016 before its real ₹536 listing: a **93× jump inside
what a base window would measure**). A vetted table now exists at
`research/153_ipo_base/results/listing_dates.csv` (1,293 accepted listings, 2006–2026) and was
**validated before any backtest ran: 48/48 known NSE IPOs accepted, date exact to ±3 days for
47/48, 0/12 known onboardings leaked.** Reusable by any future study that needs listing dates.

**Adopted spec (IPO-Base MID):** listed within 6 months · 25-day base, depth ≤ 30% · buy-stop at
the base high · −8% close stop · exit below SMA-20 · **+25% take-profit** · 8 slots @ 18.75% ·
no market gate. Standalone, 30 seeds, after tax and 25 bps a side, 2006→Sep-2026:
**31.03% CAGR [28.82..33.44], worst seed 28.82%, −20.88% drawdown, Calmar 1.50**, 32.6 trades a
year at +4.89% per trade net.

**As a third sleeve at 20% beside True North + Open Alpha: +1.13pp CAGR, −3.63pp drawdown,
+0.56 Calmar** (27.14 / −16.42 / 1.65 → **28.27 / −12.79 / 2.21**), correlation **0.16** daily
to OA and **0.18** to TN — lower than OA↔TN at 0.42 — and it beats a plain-cash sleeve at the
same weight by 5.60pp of CAGR. **Every leg of the pre-registered bar is met with room.** On the
common 2015+ window it beats r/147's gold on return at comparable Calmar: gold buys Calmar by
*lowering* return, this buys it while *raising* return.

### ⏳ PENDING — Arun's adoption call on IPO-Base MID at 10–20% of the book
Nothing was deployed. If adopted, the next step is a **G5 paper soak with a pre-registered fill
criterion** (modeled vs actual fill within 0.5% of the pivot, miss rate < 15%) and a dated
review — because **the entire edge lives in getting filled AT the pivot**: filling at the
signal-day close instead costs **−14.08pp of CAGR and loses on 30 of 30 paired seeds**.
Registered in the Ops Centre for 2026-10-15.

### ⏳ PENDING — send the site's IPO-Base panel dials + claimed numbers when legible
No replication gate was run (the screenshots did not come through). The engine is built so the
gate is a one-command run. Note in advance that two dials the site exposes — *Trail 30-week*
and *Breakout close* — are the **worst** settings we tested, so their published figures cannot
be assumed comparable.

### ⏳ PENDING — fold IPO into the four-sleeve study already owed from r/152
The exploratory cell 40% OA / 40% TN / 10% gold / 10% IPO scored **29.05% / −11.55% /
Calmar 2.52** on 2015+. That is one un-swept cell, not a finding. It joins the r/152 four-sleeve
question under the same Ops Centre review (2026-11-30) and must be run with a **gold-only null**
— the real question is what each candidate adds *on top of gold*.

**Operator caveats to carry forward:** the book earned only the idle-cash yield in **2013 and
2014** (no trades at all — the Indian IPO pipeline supplied 8–17 usable listings a year in
2012–14 against 80–182 in 2021–25); 2020–2026 supplies much of the record; capacity is
comfortable to about a ₹10 cr portfolio and binds near ₹50 cr.

Hand-off for the follow-on correlation study (r/154) is written:
`research/153_ipo_base/results/ipo_equity_seeds.csv` (30 seeds, daily, after-tax, cash 5%)
+ `results/ipo_adopted_spec.json`.
"""
    TODO.write_text(t.replace(hdr, hdr + block, 1), encoding="utf-8")
    print("TODO.md section added")
else:
    print("TODO.md already has 153")

# ─────────────────────────────────────────────────── 3. Ops & Review Centre
OPS = ROOT / "research" / "111_sensex_manual_mgmt" / "scripts" / "ops_center.py"
o = OPS.read_text(encoding="utf-8")
if "research/153 IPO Base" not in o:
    ENTRY = '''
    ("research/153 IPO Base - adoption call at 10-20%%, then paper-soak decision",
     "2026-10-15", "PENDING",
     "research/153 IPO-Base MID cleared EVERY leg of the pre-registered third-sleeve bar: at "
     "20%% weight beside TN+OA it adds +1.13pp CAGR, -3.63pp drawdown and +0.56 Calmar "
     "(27.14/-16.42/1.65 -> 28.27/-12.79/2.21), at correlation 0.16 daily to Open Alpha and "
     "0.18 to True North (LOWER than OA-to-TN at 0.42), and it beats a plain-cash sleeve at the "
     "same weight by +5.60pp of CAGR. Standalone 30 seeds 2006-2026 after tax: 31.03%% CAGR "
     "[28.82..33.44], worst seed 28.82%%, MaxDD -20.88%%, Calmar 1.50, 32.6 trades/yr at "
     "+4.89%%/trade net. ARUN TO DECIDE ADOPTION - nothing is deployed. PASS CRITERION IF "
     "ADOPTED: a G5 paper soak whose criterion is pre-registered BEFORE it starts - actual fills "
     "within 0.5%% of the modelled pivot and a miss rate under 15%% - because the ENTIRE edge "
     "lives in the entry price: filling at the signal-day close instead of the pivot buy-stop "
     "costs -14.08pp of CAGR and loses on 30 of 30 paired seeds. Two operator caveats to restate "
     "at the review: the book earned ONLY the idle-cash yield in 2013 and 2014 (no trades at all "
     "- the IPO pipeline supplied 8-17 usable listings a year in 2012-14 against 80-182 in "
     "2021-25), and capacity is comfortable to about a Rs 10 cr portfolio but binds near Rs 50 "
     "cr. IF NOT ADOPTED, record why and close. Artifacts: "
     "research/153_ipo_base/results/RESULTS.md; listing_dates.csv (a VETTED NSE listing-date "
     "table, 1,293 listings 2006-2026, validated at 48/48 recall and 0/12 leaks - REUSABLE by "
     "any future study that needs listing dates); ipo_equity_seeds.csv; ipo_adopted_spec.json. "
     "Published at /app/backtest/ipo-base-breakout-research153."),
'''
    i = o.index("REVIEWS = [") + len("REVIEWS = [")
    o = o[:i] + ENTRY.rstrip("\n") + o[i:]

    HOOK = ('the 4-sleeve study must justify overriding that or drop it. Artifacts: "')
    ADD = ('the 4-sleeve study must justify overriding that or drop it. THIRD CANDIDATE ADDED '
           '2026-09-05: research/153 IPO-Base sleeve is now the strongest complement measured - '
           'it PASSES the correlation leg MYB failed (0.16 daily to OA, 0.18 to TN) and its own '
           'exploratory 4-sleeve cell (40 OA / 40 TN / 10 gold / 10 IPO) scored 29.05%% / '
           '-11.55%% / Calmar 2.52 on 2015+. The weight grid for this review must therefore span '
           'TN / OA / GOLD / MYB / IPO, with the gold-only null as the binding comparison. '
           'Artifacts: "')
    if HOOK in o:
        o = o.replace(HOOK, ADD, 1)
        print("four-sleeve review extended with the IPO candidate")
    OPS.write_text(o, encoding="utf-8")
    print("ops_center.py review added")
else:
    print("ops_center.py already has 153")
