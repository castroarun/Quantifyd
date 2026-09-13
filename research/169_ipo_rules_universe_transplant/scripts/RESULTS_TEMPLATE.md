# research/169 — Why is IPO Base IPO-specific? Its rules transplanted to Nifty-50 / 100 / 200 / Midcap / 500 / Smallcap / all-stock universes

## VERDICT: **NO EDGE** — the rules do not transplant to any universe. IPO Base is IPO-specific because the return lives in a stock's first months after listing, not in the breakout rule. Two corrections to the live IPO book come with it.

IPO — the answer first, then the evidence. Every figure is after tax (20% STCG / 12.5% LTCG with
Indian FY loss netting), net of 25 bps a side, idle cash credited at 5.2% post-tax, 8 slots at
18.75% of a ₹10,00,000 book, median of 30 paired seeds, window W2 2006-01-01 → 2026-09-04 unless
stated. WA = 2006–2015, WB = 2016–2026. "Null" = date-matched random entry: the same days, the
same number of entries, names drawn at random from THAT universe's eligible set whose day high
reached their own pivot, with the same fill, stop, target, trail and gate. The size universes are
a **causal traded-value PROXY**, not historical index membership (Q7). Nothing live was changed.

---

**Q1. Why is the IPO system IPO-specific?**

IPO — because the money is made by owning stocks in their first months after listing, and the
age band is the only rule that puts the book there. Remove it and the identical rules become a
25-day closing-high breakout on seasoned stocks, which earns 2–9% a year at −34% to −54%
drawdown and does not beat buying random names on the same days.

For a stock listed under six months, "the highest close of the last 25 bars" is close to its
all-time high since listing, and the base is its first base. For a seasoned stock the same words
describe a 25-day Donchian close breakout — a different signal, and under these exits not a signal
at all. Three measurements say where the return actually lives:

1. **Widen the age band and the book decays steeply and smoothly** (all-stock universe):
   ≤6 months 22.39% → ≤12 months 16.28% → ≤24 months 14.68% → any vetted listing 7.55% → no age
   limit 6.97% → seasoned >6 months 5.86% → seasoned >24 months 4.63%.
2. **Inside the six-month band, most of the return is the very first base, 25–60 sessions after
   listing.** Requiring 60 bars instead of 25 takes Spec A from 21.80% to 11.57% (research/167's own
   engine, Q6). Entries that early happen before a 50-day average exists, so for their first weeks
   they are +25% / −10% bracket trades on brand-new listings — and they are the best trades in the
   book: +8.4% per trade against +5.7% for trail-managed holds.
3. **The breakout is a smaller part of the story than research/167 reported** (Q5). From 2016 to
   2026, random young, liquid names bought on the same days with the same exits and gate earned
   what Spec A earned: 28.27% against 28.37%.

**Q2. What happens on Nifty 50, 100, 200, Midcap, 500, Smallcap and all stocks?**

IPO — every one fails, with the age band removed and with seasoned names only. Spec A's rules,
identical except for the universe, each against its own null:

{{TRANSPLANT_TABLE}}

The pre-registered test for "the edge exists on universe X" was: W2 paired edge ≥ +1.0pp with the
real book beating its null on ≥ 25 of 30 seeds, AND ≥ 25/30 in both WA and WB. **0 of 16 transplant
cells pass, and none comes close**: the best W2 edge is +1.52pp (Midcap-like, no age limit) on 20
of 30 seeds, and 15 of 30 in WB.

Tradeability and capacity for the same cells (position size as a share of the name's 20-day
median traded value, at a ₹10 L book):

{{TRADE_TABLE}}

**Second check on true point-in-time market cap.** `fundamentals.db` `mcap_pit`, ranked on the
previous month's row, exists only from Aug-2018, so this runs 2018-09-01 → 2026-09-04:

{{MCAP_TABLE}}

Market-cap universes are no better than the traded-value proxies — worse in the mid and small
bands — and Spec A itself ties its null in this window. **The verdict does not depend on how
"Nifty-50-like" is defined.**

Cost ladder (a transplant trades about 36 times a year against Spec A's 20):

{{COST_TABLE}}

**Q3. Is the edge in the young-stock condition, or in the breakout + 50-day-trail mechanics?**

IPO — not in the mechanics. The seasoned-only arm (listed more than 6 months ago) fails on all 8
size universes, with W2 edges from −1.70 to +0.69pp. The pre-registered "the mechanics are the
edge" needed a majority of universes to pass: **0 of 8 did.**

The return is in the young-stock condition, but the honest wording is narrower than "the age band
is the edge". That pre-registered label also required Spec A itself to pass the null test, and on
the clean panel it does not: +2.25pp in W2 (29/30), +4.63pp in WA (30/30), −0.18pp in WB (14/30).
What the six-month band holds is a **cohort** that pays when it is held with a slow trail and an
index gate. The breakout selection added about +4.6pp in 2006–2015 and nothing since 2016.

{{AGE_TABLE}}

At ≤12 and ≤24 months the random null BEATS the breakout rule (18.04% vs 16.28%; 16.29% vs
14.68%): once the band is wider than six months, the rule picks worse young names than chance.

**Q4. Is a transplanted version just an existing book in disguise?**

IPO — partly, and a worse one. The broad transplants correlate 0.53–0.56 monthly with OA · Base
Age (the live Open Alpha book: all-time-high breakout after an aged, deep base), against IPO-A's
0.33, and every one of them makes the three-sleeve book worse on every path:

{{BLEND_TABLE}}

Base book: TN 37.5 / OA · Base Age 37.5 / IPO-A 25, monthly rebalanced, research/168's paths —
21.18% / −24.01% / Calmar 0.885. "vs cash" = the same book with cash in place of the candidate,
the cash weight solved to the same median drawdown (research/168's risk-matched null).

This converges with families already on file: research/71 (breakout exit bake-off — a trailing
stop beats a target), research/82 (10–15-day breakouts are real but cyclical and converge on the
breakout-paper book), research/142 (Blue-Sky ATH breakout), research/152 (multi-year breakout =
Open Alpha in disguise) and research/161 (OA · Base Age). A 25-bar breakout on seasoned stocks is
the high-turnover, shallow-base end of that family: it re-imports Open Alpha's breakout beta
without Base Age's base filter and without IPO Base's cohort. The pre-registered "worth a sleeve"
bar (+0.10 Calmar, or +2pp CAGR at no worse drawdown, on ≥ 20 of 30 paths; beats risk-matched cash
on ≥ 25 of 30; monthly correlation to OA · Base Age below 0.60) **is failed by every transplant, at
every weight, on 0 of 30 paths.**

**Q5. Does this change what we believed about IPO Base itself?**

IPO — yes, in two ways. Neither changes the blend value research/168 measured, because the real
curves it blended reproduce exactly.

**(a) research/167's "+4.8pp over a random null on 30 of 30 seeds" is about half a data artifact.**
research/167's panel computes rolling windows on a date index that is the union of every symbol's
dates. One missing row for a name — a partial-coverage special session, a phantom holiday row —
makes its 25-bar pivot and 50-bar trail NaN for weeks afterwards. Rebuilt with per-symbol windows
(the playbook's mandatory NaN-robust rule), the real book barely moves and the null rises:

{{NULL_ATTR_TABLE}}

research/167 ran its null on W2 only, so the WB collapse was never visible. The measured side
effect on the old panel: 32% of trades held through a stretch with no trail, and in those stretches
real breakouts earned +8.9% per trade against the null's +5.9% — the blackout turned breakouts into
bracket trades that suited them. **Read Spec A as a young-listing cohort harvest with a good exit
and gate, not as a stock-picking edge.**

**(b) research/167's capacity line quoted a median as a 90th percentile.** Its own
`stage9_adoption.csv`: at ₹10 L the **median** position is 1.56% of the name's 20-day traded value
and the **90th percentile is 9.05%** (9.64% on the clean panel). Its "~90% of a day's volume at
₹1 cr" was the p90 and is right. On a "p90 position ≤ 5% of traded value" yardstick Spec A is a
~₹5 L book; on a "median ≤ 5%" yardstick, ~₹32 L. The ₹20–25 L ceiling is optimistic on the tail.

**Q6. The Capital Desk: does anything here change whether IPO Spec A should go to 25%?**

IPO — it does not change **which** third sleeve: no transplant is a candidate, so IPO Base remains
the only third sleeve on file. It changes **what a 25% allocation buys today**, because
`services/ipo_paper.py` runs `MIN_BARS = 60` and Spec A was validated at 25.

The 6-Sep-2026 decision to run 60 rested on reading research/153's loader (`... where n >= 60`) as
"a stock is invisible until it has 60 bars". That count is taken over the WHOLE database today, not
at the signal date, so a name with 2,000 rows today is scanned from its 25th bar. Only listings in
the last ~60 sessions are invisible — exactly the window the 06-Sep reconciliation looked at, which
is why it saw 75% agreement. research/167's engine, unchanged, with its own null:

{{MINBARS_TABLE}}

What each version is worth inside the book Arun is about to fund:

{{DESK_TABLE}}

Standalone, the 60-bar book is 12.39% / −36.35% / Calmar 0.341, worst seed 11.30%, a 23-trade
losing streak, and it does not clear its own null (+0.79pp; WB 19 of 30). Inside the blend it has
the fingerprint that disqualified the old incumbent in research/168: it buys a little drawdown and
adds barely more than an arbitrage fund at equal risk.

**The line for the decision: 25% to IPO is supported only for the spec that was validated.** Until
the live book runs `MIN_BARS = 25`, the Capital Desk would be funding a ~12% sleeve that lowers the
blend's CAGR on every one of 30 paths. Changing `MIN_BARS` is a strategy change — its own STATUS
doc, the capacity question in Q5(b) answered for entries 25–60 sessions after listing (the
thinnest names the book will ever buy), and an after-15:40 deploy. **This study changed nothing
live.** A dated review is registered for 19-Sep-2026, ahead of the 26-Sep funding call.

**Q7. How honest are the universes?**

IPO — honest about survivorship, loose about exact membership. No point-in-time index constituent
history exists anywhere in the repo, only today's official lists, and using those back to 2006
hands the backtest the names that went on to become large. So on the first trading day of each
month every non-fund symbol with ≥ 60 bars was ranked by its trailing 126-bar median traded value
known at the prior close (the research/41 method) and held in that band for the month:

{{PROXY_TABLE}}

The top-200 / top-500 proxies are good; the Nifty-50-like set is fair; the Midcap- and
Smallcap-like bands are rough. That is why the `mcap_pit` re-run in Q2 exists, and it agrees.

---

## Reproduction gate and data defenses

- **research/167's engine reproduces Spec A exactly**: 21.80% / −26.63% (delta 0.000pp); all 30
  seed paths identical to research/168's `A_25bps_y52` (max abs diff 0.0); null 16.98%, edge
  +4.78pp, 30/30.
- **The new full-universe panel, set to research/167's conventions, reproduces 21.80% / −26.63%**
  with the same 1,545 signals, before any defense is switched on.
- Defenses switched on one at a time (Spec A, 5.0% cash): per-symbol NaN-robust windows → 22.24%
  (+162 signals); phantom holiday rows dropped (24-Apr-2014: 121 symbols, 15-Oct-2014: 129, all
  zero volume) → no change; split back-adjustment at one-day close ratios ≤ 0.60 or ≥ 1.80 → no
  change for IPO. Full universe: 273 adjustment events, 38 on names at ≥ ₹5 cr (e.g. ADANIENT 2015,
  ARVIND 2018, INFIBEAM 2018).
- Two process incidents, both recovered: a stage was killed out-of-memory when two panels were in
  memory at once (re-run alone, no partial results kept), and a progress-print bug aborted the
  first market-cap run before any cell was written.

## Per-year house table

Each cell is the annual return with the intra-year max drawdown beneath it, measured from the
running peak of the FULL curve. Columns are the median-CAGR path of each ensemble; the summary row
carries the 30-path medians. Best-of columns exclude NIFTYBEES. Common window 2006-04-03 →
2026-09-03.

{{YOY_TABLE}}

## Caveats — what would make this wrong

- **Proxy universes** (Q7): the Midcap- and Smallcap-like bands overlap the real bands only ~45%.
  The 2018+ market-cap re-run agrees, but it is eight years, not twenty.
- **Survivorship**: names never onboarded to Kite cannot be measured. Delisted names that are in the
  database are traded.
- **The ₹5 cr liquidity floor is nominal**, not inflation-adjusted, so it admits fewer names in 2006
  than in 2026 on every universe.
- **Split back-adjustment treats demerger drops as price adjustments** (the total-return
  convention). 38 events on ≥ ₹5 cr names in twenty years.
- **The re-fit of a survivor was pre-registered and therefore not run** — nothing survived. A
  seasoned-stock breakout with a different exit could behave differently; research/71 and
  research/161 already cover that ground, and OA · Base Age is that book.
- **Multiple testing**: ~100 backtest cells, each on 30 seeds. Nothing here is a discovery, so the
  concern cuts only against the corrections in Q5 and Q6 — and those rest on reproduced, paired,
  direct comparisons, not on a sweep peak.
- **Mixed panels in Q6**: TN, OA · Base Age and IPO-A are research/168's curves (research/167's
  panel); the 60-bar live book and the transplants are this study's clean panel. The clean-panel
  Spec A ties IPO-A inside the blend (+0.05pp CAGR, monthly correlation 0.925), so the two are
  commensurable for this purpose.
- **Not done**: an equity-curve chart pack (the family died at the null gate, before a tearsheet is
  owed); VIX gates; alternative entry mechanics for seasoned universes (the next-day buy-stop was
  held identical by design).

## Cells disclosed

S0 reproduction 3 checks · S1 panel equivalence 5 · S1b min_bars on research/167's engine 3 × (real
+ null) · S1c null attribution 4 × 2 · S1d mechanism 2 × 2 · S2 transplant 18 × 2 · S3 age axis 4 × 2
· S2b market-cap universes 13 × 2 · S4 re-fit 0 (no survivor) · S5 cost ladder 10 · S6 blends 5
candidates × (replace + 4 weights) each with a risk-matched cash null. About 100 backtest cells, 30
seeds each, most across three windows.

## Files

| File | What |
|---|---|
| `scripts/xpanel.py` | full-universe panel: causal traded-value ranks, split adjustment, NaN-robust windows, signal / null / runner |
| `scripts/run169.py` | stages repro, equiv, transplant, age, refit, costs, blend, proxy |
| `scripts/minbars_check.py`, `null_attrib.py`, `mechanism.py`, `mcap_check.py` | S1b, S1c, S1d, S2b |
| `scripts/close_loop.py` | builds this file's tables and the app entry from the result files |
| `results/s0_repro.json`, `s1_equiv.csv`, `s1b_minbars_r167engine.json`, `s1c_null_attribution.csv`, `s1d_mechanism.json` | reproduction, defenses, min_bars, null attribution, mechanism |
| `results/s2_transplant.csv`, `s3_age.csv`, `s2b_mcap_universes_2018.csv`, `s5_costs.csv` | cell tables |
| `results/s6_blend.json`, `s6_peryear.json`, `s6_peryear_table.md`, `carried.json` | portfolio fit |
| `results/s7_proxy.json`, `s1_split_events.csv` | proxy validation, adjustment events |
| `results/navs/*.npz` | 30-seed W2 NAV paths (gitignored) |

Reproduce: `venv/bin/python research/169_ipo_rules_universe_transplant/scripts/run169.py repro`,
then `equiv`, `grid`, `minbars_check.py`, `null_attrib.py`, `run169.py refit`, `costs`,
`mcap_check.py`, `run169.py blend`, `mechanism.py`, `close_loop.py`. One panel process at a time
(~3.5 GB each); about 40 minutes in total on the VPS.
