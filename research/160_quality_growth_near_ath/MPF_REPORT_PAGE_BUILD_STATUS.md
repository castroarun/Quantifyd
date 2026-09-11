# The Momentum Portfolio — ONE report page at `/app/mpf-report` — STATUS: DONE

**Task kind:** BUILD (frontend page + generator job). **Opened:** 11-Sep-2026 ~23:10 IST.
**Canonical host:** VPS `/home/arun/quantifyd`. The laptop folder is not a git checkout and its
`frontend/` is stale — every file was written locally, shipped to the VPS, and **built on the VPS**.
**No backend restart, no crontab edit, nothing under `services/`, no live DB touched.**
**Handover read in full before building:** `docs/MPF_UNIFIED_REPORT_HANDOVER_2026-09-11.md`.

---

## 1. Headline

`/app/mpf-report` is the single entry point to the Momentum Portfolio. The individual study
pages under `/app/backtest/<slug>` remain the archive and every section links back to them;
**nothing was removed anywhere, and the roster study page was NOT edited** — a second session
was editing it concurrently.

## 2. The Ask

**What Arun approved (11-Sep-2026, late evening IST):** one report page for the Momentum
Portfolio that replaces the confusing single-study roster page as the entry point, with no loss
of information.

**The binding editorial rules he raised more than once, all of which shape this page:**

| Rule | How it is honoured here |
|---|---|
| **POST-TAX ONLY** | Every figure on the page is after tax. Any figure that exists only pre-tax is OMITTED and the omission is stated — see §8.3 (4). |
| **Every table states WHICH SYSTEMS, WHICH WINDOW, WHICH BASIS** | Each table's title carries all three, verbatim. |
| **Name systems, don't version them** | "Open Alpha · Base Age" (r/161) and "Open Alpha · ATH + VIX" (r/159). No v2/v3 anywhere. |
| **Test period, average invested and average in cash ON the table** | Three dedicated columns on every summary table, plus a cash-yield caveat line beneath. |
| **Use the entire common testing period** | The headline is 2006-04-03 → 2026-09-03, 20.4 years. |
| **Answer in Q&A form** | Section 00 is five questions, each answered in one or two lines with the window stated. |
| **The correction leads** | A highlighted block sits above ANY return figure on the page. |
| **Visuals wherever possible** | Ten generated charts: curves, drawdowns, monthly heatmaps, yearly bars, correlation heatmaps, rolling 3-year CAGR, invested-vs-cash. |

## 3. The Base — what the page measures and how

| | |
|---|---|
| **Measurement standard** | after tax (20% STCG / 12.5% LTCG, Indian FY loss netting with carry-forward) · 25 bps a side · 5% p.a. on idle cash · **placeable entries only**: close-decided, next-open fills |
| **HEADLINE window** | **2006-04-03 → 2026-09-03, 20.4 years** — the full common period of every book actually being chosen between, set by the shortest series (True North). It contains BOTH 2008 and 2020. |
| **SECOND window** | **2018-08-01 → 2026-09-04** — the only window Quality Summit can exist in, with every system re-measured on it. Clearly labelled, visibly secondary, and never mixed with the headline. |
| **2016-2026** | appears ONLY as one labelled row of evidence inside the Open Alpha section, for the ATH + VIX variant, whose VIX gate needs INDIA VIX (2015+). |
| **Quality Summit path drawn** | the **median-CAGR rebalance offset** of twelve (`offset3`; median 21.05%, range 17.45 .. 23.50) — never an average of paths |
| **Intra-year drawdown** | from the running peak of the **FULL curve**, never from the year's first bar |
| **Blend row** | TN + Base Age, 50-50, **monthly rebalanced**, computed by this generator. **Labelled "computed here", not a study result.** |

## 4. Plan — what was built, in order

| Step | Artefact |
|---|---|
| 1 | this STATUS doc (sections 1-4 before anything was built) |
| 2 | `research/_utilities/mpf_report_build.py` — the generator, ten charts + one JSON |
| 3 | `static/app/mpf_report.json`, `frontend/public/mpf_report.json`, `frontend/public/mpf-report-*.png` |
| 4 | `frontend/src/data/mpf_report.ts` — per-system narrative, each study figure with its `source` |
| 5 | `frontend/src/pages/MpfReport.tsx` + `.module.css` |
| 6 | route in `App.tsx`, `active` union in `AppLayout.tsx`, sidebar entry in `Sidebar.tsx` (Holdings group, beside Momentum Portfolio) |
| 7 | `ops_center.py` GROUPS job + `docs/LABS_AND_JOBS_REFERENCE.md` mirror |
| 8 | VPS build, verify, `TODO.md`, commit |

**NOT done, deliberately:** `frontend/src/data/backtests.ts` was NOT touched (a second session
is editing the roster entry concurrently — a banner edit there would have collided), and
`frontend/src/data/strategies.ts` was NOT touched (register of record; no status, size or rule
changed tonight).

---

## 5. Status — event log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 11-Sep 23:10 | Context read complete | playbook §9, CLAUDE.md Q&A + YoY + register sections, backtests.ts roster entry, strategies.ts TN/OA/IPO, r/144, r/153, r/158, r/159, r/160, r/161, TODO.md, ops_center.py |
| 11-Sep 23:25 | STATUS doc v1 (sections 1-4) | before any build |
| 11-Sep 23:40 | Generator v1 run | single-window build |
| 11-Sep 23:55 | **REVISED BRIEF received** | headline window → 20.4 years; systems NAMED not versioned; post-tax only; invested/cash columns; correction leads; ten charts; Q&A block; do NOT edit backtests.ts |
| 12-Sep 00:05 | Handover doc read in full | `docs/MPF_UNIFIED_REPORT_HANDOVER_2026-09-11.md`, 276 lines |
| 12-Sep 00:15 | Generator v2 written and run | two windows, ten charts, after-tax evidence tables, JSON |
| 12-Sep 00:20 | Numbers verified against the handover §3 table | reproduce **exactly**: TN 19.5/−23.7/0.82/3787 · Base Age 20.3/−32.5/0.62/4334 · IPO 15.1/−35.9/0.42/1767 · index 10.6/−59.7/0.18/780 |
| 12-Sep 00:20 | 2018 table verified against `quality_summit_roster.json` | reproduces exactly: QS 20.9/−38.9/0.54/465, TN 20.64, Base Age 26.16, IPO 12.97, index 10.92 |
| 12-Sep 00:30 | Page, CSS, data TS shipped; route + sidebar + AppLayout patched | |
| 12-Sep 00:35 | `npm run build` on VPS | one TS error (AppLayout `active` union) fixed, then clean |
| 12-Sep 00:40 | Charts inspected; footnote overflow + one wrong caption fixed | footnotes hard-wrapped; the blend, not Base Age, ends highest |
| 12-Sep 00:45 | Verified: page 200, JSON 200, PNGs 200, bundle grep hit | |
| 12-Sep 00:50 | ops_center job registered + LABS doc mirrored | |
| 12-Sep 00:58 | **`after_tax_tables.csv` found COMPLETE (71 rows, 22:18)** — the other session finished the run | generator, page and §8.3 (4) updated: VIX gates now published after tax as their own 2016-2026 table |
| 12-Sep 01:05 | Regenerated, rebuilt, re-verified | |
| 12-Sep 01:10 | TODO.md + commit | |

### Findings that came out of building it

- **The 50-50 blend is the strongest thing on the page and it is nobody's study.** Over the full
  20.4 years it returns **20.42%** at **−25.24%** (Calmar 0.81) against Base Age's 20.27% at
  −32.45% (0.62) and True North's 19.48% at −23.67% (0.82). It keeps essentially all of Base
  Age's return and gives back seven points of drawdown. On the 2018 window it reads 24.11% at
  −19.93%, **Calmar 1.21 — the best number anywhere on this page.** This is exactly the work the
  handover lists as owed item 7 and NOT STARTED; this row is a first look, not an allocation.
- **Nothing reaches 25% after tax over the long window.** The best single book is 20.27% and the
  blend is 20.42%. Only the 2018 window produces a 26% figure, and that window throws away 2008
  and 2020.
- **Window choice is the single biggest lever on how this book looks**, exactly as the handover
  warned: Base Age reads 20.3% on 20.4 years and 26.2% on the 2018 window. Both are true.

---

## 6. Crash recovery — how Arun resumes without Claude

**The page is static.** If it renders, nothing is running and nothing can be half-done.

```bash
# 1. Is it served?
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:5000/app/mpf-report            # 200
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:5000/app/mpf_report.json       # 200
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:5000/app/mpf-report-curves-20y.png

# 2. Regenerate every number and all ten charts (safe any time; reads research CSVs only)
cd /home/arun/quantifyd && venv/bin/python3 research/_utilities/mpf_report_build.py

# 3. Rebuild the frontend after ANY .tsx/.ts/.css edit, and after step 2 so the PNGs
#    reach static/app/ (frontend-only: NO backend restart)
export PATH=$HOME/.nvm/versions/node/v20.20.2/bin:$PATH
cd /home/arun/quantifyd/frontend && npm run build
# then hard-refresh the browser — the bundle hash changes
```

**Safe to inspect:** everything in §7. **Do NOT touch:** anything under `services/`, the
crontab, `backtest_data/*.db`. **If the JSON is missing** the page shows a plain "numbers have
not been generated yet" line with the command — it degrades, it does not crash.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `research/160_quality_growth_near_ath/MPF_REPORT_PAGE_BUILD_STATUS.md` | this file | yes |
| `research/_utilities/mpf_report_build.py` | the generator — every number and every chart | yes |
| `static/app/mpf_report.json` | what the page fetches | yes (~60 KB) |
| `frontend/public/mpf_report.json` | survives a full `emptyOutDir` rebuild | yes |
| `frontend/public/mpf-report-curves-20y.png` | log growth + drawdown, 20.4 years | yes |
| `frontend/public/mpf-report-curves-2018.png` | the same on the Quality Summit window | yes |
| `frontend/public/mpf-report-yearly-bars.png` | yearly returns, systems side by side | yes |
| `frontend/public/mpf-report-rolling3y.png` | rolling 3-year CAGR vs the 25% bar | yes |
| `frontend/public/mpf-report-corr-20y.png` / `-2018.png` | weekly-return correlation heatmaps | yes |
| `frontend/public/mpf-report-invested.png` | average invested vs cash, measured | yes |
| `frontend/public/mpf-report-heat-{truenorth,baseage,ipobase,qualitysummit}.png` | monthly-return heatmaps | yes |
| `frontend/src/data/mpf_report.ts` | per-system narrative + study figures with `source` | yes |
| `frontend/src/pages/MpfReport.tsx` / `.module.css` | the page | yes |
| `frontend/src/App.tsx`, `components/Layout/AppLayout.tsx`, `components/Sidebar/Sidebar.tsx` | route + nav | yes |
| `research/111_sensex_manual_mgmt/scripts/ops_center.py` | generator registered as a job | yes |
| `docs/LABS_AND_JOBS_REFERENCE.md` | mirror of that job | yes |
| `TODO.md` | today's entry | yes |

---

## 8. PROVENANCE — every number on the page and the file it came from

### 8.1 Computed by `mpf_report_build.py` — nothing typed

| Figure group | Source file(s) |
|---|---|
| **HEADLINE** CAGR / MaxDD / Calmar / growth-of-100 for True North, Open Alpha · Base Age, IPO Base, NIFTYBEES, 2006-04-03 → 2026-09-03 | `research/159_oa_honest_reoptimization/results/full_period_after_tax.csv` |
| Average invested / average in cash | `research/159_oa_honest_reoptimization/results/full_period_summary.json` + `scripts/full_period.py` `INVESTED` dict |
| **2018 WINDOW** rows for TN, Base Age, IPO Base, NIFTYBEES | `research/159_oa_honest_reoptimization/results/all_systems_after_tax.csv` (the roster file, so the section reproduces the published roster page exactly) |
| **2018 WINDOW** Quality Summit | `research/160_quality_growth_near_ath/results/F_Bb7_equity.csv`, 12 offsets, drawn path = median-CAGR offset (`offset3`) |
| Year-by-year return + intra-year drawdown (full-curve peak), BEST CAGR / LEAST DD / BEST OVERALL, both windows | computed from the curve frames above |
| Weekly-return correlation matrices, both windows | same, resampled W-FRI |
| **TN + Base Age 50-50 monthly blend**, both windows | computed here from the two daily curves — **not a study result** |
| Entry-mechanic surface (table A), null control (B), PRICE-gate bake-off (C), all AFTER TAX, 2006-2026 | `research/159_oa_honest_reoptimization/results/after_tax_tables.csv` (71-row complete version, 11-Sep 22:18) |
| VIX-gate bake-off, AFTER TAX, **2016-2026 only** — rendered as its own table, never merged with the price gates | same file, `window == '2016-2026'` rows |
| Open Alpha · ATH + VIX summary row (19.23% / −34.15% / 0.56, 2016-2026) | `research/159_oa_honest_reoptimization/results/all_systems_summary.json` |
| All ten PNGs | drawn from the same assembled frames |

### 8.2 Typed into `frontend/src/data/mpf_report.ts`, each quoted with its source in the prose

| Figure | Value | Source file |
|---|---|---|
| True North own study window | 20.9% / −23.7% / Calmar 0.88, WA 2012 → 2026-09-03, net of cost AND tax | `research/144_truenorth_reassessment/results/RESULTS.md` §(b), incumbent row |
| True North offset ensemble | median 20.7% [14.9 .. 25.1], DD median −25.1%, worst offset −28.3% | same, §(c) |
| True North without the gate | 23.9% at −46.5%, Calmar 0.51 | same, §(b) NO GATE row |
| Open Alpha · Base Age own window | 21.26% [19.87 .. 21.89] / −34.80% / Calmar 0.618, 2005-01-03 → 2026-09-11, 30-seed median | `research/161_ath_base_age_breakout/results/RESULTS.md` §1 |
| Base Age decomposition | exit ST(14,4) worth +11.85pp (6.81% → 18.66%); base age + depth worth +2.60pp and −6.79pp DD | same, §2 |
| Base Age null control | rule 21.26% vs date-matched random 12.11% → +9.15pp | same, §1 criterion 4 |
| Base Age cost ladder / no-cash / outliers | 21.26 / 20.34 / 19.35%; 19.27% with no cash carry; 133,000× collapse on removing 10 of 687 trades | same, §4 and §5 |
| Quality Summit own window | 21.19% / −37.1% / Calmar 0.58, 91% invested, 2018-08-01 → 2026-09-10, 12-offset median (W1 20.11%, W2 20.95%; 19.10% at 60 bps) | `research/160_quality_growth_near_ath/results/RESULTS.md` |
| QS "screen as written" | 10.77% / −29.1% / 0.40 at 43% invested; 7.74% with no cash yield | same |
| QS screen value / null | −1.32pp CAGR, +0.106 Calmar vs unscreened; random from the same universe 14.82%; ranking worth +7.70pp | same, decomposition + paired-test tables |
| QS blend value | Calmar 2.37 → 2.23 → 2.02 → 1.69 at 10 / 20 / 33%; cash wins 360 of 360 paths | same |
| IPO Base own window | **15.00% / −37.55% / Calmar 0.40**, 2006 → 2026-09, 30 seeds, after tax | `research/158_oa_arming_width/OA_ARMING_WIDTH_AND_POKE_FILL_DAILY_SWEEP_STATUS.md` §3 IPO table, "LIVE engine: next-day stop at the broken pivot" |
| IPO published vs honest | reproduced 31.48%, published 31.03%, close-fill control 17.49%, honest 15.00%; 98.5% of signals survive as reachable fills | same |
| Open Alpha as published | 40.8% → −1.7%; 49 of 50 closed above the level bought; 7.0 failed resting fills per published entry; HCLTECH 20 prior touches | same §3 + handover §1 |
| The two extra defects (221 funds; renamed symbols stale) | as stated | handover §5.1 and §5.2, from `research/158_oa_arming_width/scripts/build_etf_list.py` |
| System rules, sizes, dashboards, the paused-entry detail | as in the register | `frontend/src/data/strategies.ts` (read-only; **not edited**) |
| The nine owed items and the dated reviews | as stated | handover §7 + `research/111_sensex_manual_mgmt/scripts/ops_center.py` REVIEWS |

### 8.3 Where sources disagreed — and which one wins

1. **THE THREE-WINDOW PROBLEM — the hardest editorial call, and how it is resolved.** The same
   system reads three different ways: Open Alpha · Base Age is **20.3%** on 2006-2026, **23.0%**
   on 2016-2026 and **26.2%** on 2018-2026. None is wrong. The page makes the **20.4-year window
   the headline** (it contains 2008 and 2020, the two falls that matter), gives Quality Summit a
   **separate, clearly labelled 2018 section in which every system is re-measured**, and admits
   the 2016-2026 window only as one labelled evidence row for the ATH + VIX variant. **No table
   mixes windows**, and every table title states which systems, which window and which basis.

2. **The two sections are built from DIFFERENT curve files for the same systems.**
   `full_period_after_tax.csv` (headline) and `all_systems_after_tax.csv` (2018 section) are
   separate runs. The 2018 section deliberately uses the roster file so that it reproduces the
   published roster page to the decimal — verified: it does. This is stated on the page itself
   (`notes.two_curve_files`), not buried here.

3. **IPO Base own-window drawdown.** `research/158` STATUS says **−37.55%** (Calmar 0.40);
   `full_period_summary.json` says −35.86% (Calmar 0.42) at 15.1% CAGR. Both honest, different
   runs over slightly different spans. The page prints **r/158's 15.00% / −37.55% / 0.40** as the
   own-window figure because that is the audit that established the number, and the curve-file
   figure is what the generated 20.4-year table shows — so **both appear, each labelled with its
   window and basis**, and the difference is stated in IPO Base's caveats.

4. **The after-tax re-run FINISHED while this page was being built, and the page was rebuilt on
   it.** An earlier read of `after_tax_tables.csv` (41 rows, 22:08) caught it mid-flight after a
   `ValueError` in `scripts/aftertax_all.py`, and a first draft of this page said the VIX rows
   were unavailable. The other session completed the run at **22:18** (71 rows, commits
   `c92a4749` / `eb3c9f71`), so **every table in the evidence section is after tax and no pre-tax
   figure appears anywhere on the page**. What the completed file does carry is a **window split
   inside the gate bake-off**: the PRICE gates ran on the full 2006-2026 window, every VIX
   construction only on 2016-2026 because INDIA VIX begins in 2015. They are rendered as **two
   separate tables** with their windows in the titles and a line saying they must not be read
   across. The adopted 1-year 70th-percentile VIX gate reads **19.07% / −36.39% / Calmar 0.522**
   after tax there, which corroborates the 19.23% / −34.15% / 0.56 summary row from
   `all_systems_summary.json` rather than contradicting it.

5. **Average invested for Open Alpha · Base Age is a GAP, not a number.** The handover's §3 table
   prints 67% / 33%. No file on disk carries it: `scripts/full_period.py` records `None` with the
   comment *"to be measured in its own harness"*, and `curves161.npz` holds NAV series only. The
   page therefore renders **"not measured"** in that cell and the chart renders a grey bar saying
   so. A page that prints a number it cannot point at is the thing this whole evening was about.

6. **Cash yield is not consistent across the books and it is not a small term.** Idle cash is
   credited at **5%** everywhere except True North, whose curve comes from research/144's own
   after-tax NAV file at **6.5%** — worth roughly **0.9 points a year** to it, since it holds cash
   57% of the time. At 5%, roughly **2.8 of True North's points** and **3.4 of IPO Base's** are the
   sweep rather than the strategy, while NIFTYBEES is fully invested and gets none. Stated as a
   caveat line directly beneath every summary table, per Arun's instruction.

7. **The r/144 blend figure of 27.4% / −16.4% / Calmar 1.68 is superseded and is NOT shown.** It
   blended True North against the *pre-audit* Open Alpha curve, which carries the same-bar
   look-ahead fill. The page shows instead a TN + Base Age 50-50 monthly blend computed by this
   generator, explicitly tagged "computed here".

### 8.4 "Not on disk" — things that may exist only in another session's chat

- **Average invested for Open Alpha · Base Age (~67%).** Asserted in the handover; no file
  carries it. Rendered as a visible gap.
- **The weekly correlations quoted in handover §3** are the 2016-2026 ones. The page computes its
  own on each of its two windows rather than reusing that table, so the figures differ slightly
  and are labelled with their window.
- **Any pre-registered soak criterion for an Open Alpha · Base Age paper book.** None exists
  anywhere. If the 26-Sep answer is yes, it must be written before the book starts.
- **A Quality Summit paper-book spec.** None exists; r/160 explicitly did not paper it.
- **A True North ensemble curve** to put it on the same footing as the others. Owed.
- **A daily invested-fraction series for any book.** No engine writes one, so the "when is each
  book in cash" chart is a bar of measured averages rather than a strip over time. Owed.
- **An IPO Base re-optimisation on the honest entry.** NOT STARTED — top of the owed list.
- **A blend / allocation study across TN + Base Age + IPO.** NOT STARTED. The 50-50 row on the
  page is this generator's arithmetic, and it is the only structure that plausibly clears 25%.

---

## 9. What the page says, in one paragraph

Two of the three live books were justified by an entry no order can place, and the page says so
before it shows anyone a return. What survives, after tax, over the full 20.4 years: Open Alpha ·
Base Age 20.3% at a −32.5% fall, True North 19.5% at −23.7%, IPO Base 15.1% at −35.9%, against
NIFTYBEES 10.6% at −59.7%. All three beat the index; **none reaches 25%**. Half True North and
half Base Age, rebalanced monthly, returns 20.4% at −25.2% — essentially all of Base Age's return
with True North's shallower ride — and on the 2018 window it posts the best Calmar on the page at
1.21. That blend is this generator's own arithmetic, not a study, and the study that would
settle it has not been started. Quality Summit, measured on the only window it can exist in,
lands on True North's return with double the fall and 0.66 correlation to Base Age: it is a
weaker sampling of a family the book already trades, which is why research/160 left it unpapered.
