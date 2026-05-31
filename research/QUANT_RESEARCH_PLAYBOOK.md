# Quantifyd Quant-Research Playbook — the Constitution for Backtesting

> Written as the operating doctrine of a fund that **promises and delivers superior
> risk-adjusted returns to clients**, never stops improving its systems, and treats
> every result as something a regulator, an investor, and a future version of itself
> will audit. If you are an agent (or human) about to run *any* investment/trading
> backtest in this repo, **read this first and follow it**. It is the umbrella over
> the operational conventions already in `.claude/CLAUDE.md` (STATUS-MD discipline,
> research-folder layout, VPS-canonical-host, no-restart-during-market-hours).

---

## 0. Prime directives (read every time)

1. **Truth over hope.** The job is to find what is *real and tradeable after costs*,
   not to confirm an idea. A clean negative result is a win — it saves capital. Report
   failures as plainly as successes. If new evidence contradicts an earlier conclusion,
   **correct it loudly** (we did this with the call-spread overlay — flattered by a
   2-year view, killed by the all-years view).
2. **Net-of-cost or it didn't happen.** Gross edges are abundant and mostly fake once
   you pay to trade them. Every result carries gross *and* net, and a cost-sensitivity.
3. **A SIGNAL is not a STRATEGY.** A positive expectancy per trade (signal) is
   necessary but not sufficient. A *strategy* must also survive portfolio construction,
   correlation, drawdown, capacity, and execution. Many of our signals were real and
   still uninvestible. Always state which one you have.
4. **Reproducible or it's a rumour.** Any number must be regenerable from a committed
   script + a known data snapshot, by someone else, with no hidden state.
5. **Assume you will crash mid-run.** Every long task is checkpointed, resumable, and
   narrated in a live STATUS-MD that is the *sole* recovery source.
6. **Kill cheap, scale dear.** Spend the least compute to disprove an idea. Only ideas
   that survive cheap probes earn expensive sweeps.
7. **Economic rationale first.** If you cannot say *why* an edge should exist and
   *who is on the other side*, treat any backtest of it as data-mining until proven.

---

## 1. The research lifecycle (stage gates)

Run ideas through a funnel. **Each gate has explicit pass/fail criteria; do not spend
the next stage's compute until the current gate passes.** Record the gate decision in
the STATUS-MD.

| Stage | Question | Cheap test | PASS gate → next |
|---|---|---|---|
| **G0 Hypothesis** | Why should this edge exist? Who loses? | 1 paragraph: mechanism + counterparty + decay risk | Plausible economic story |
| **G1 Probe** | Does *any* signal exist in *our* data? | Cross-sectional/sweep, gross, t-stat, monotonicity | Gross edge with t≳3 or clean monotonic deciles |
| **G2 Mechanics** | Does a tradeable rule capture it? | Full backtest of entry/exit variants, **gross vs net** | Net edge after realistic cost on a meaningful sample |
| **G3 Robustness** | Is it real, not overfit? | OOS/walk-forward, per-year, param sensitivity, causal-only features | Stable across time & params; survives adversarial attack |
| **G4 Portfolio** | Is it *investable*? | Equity curve, sizing, correlation, MaxDD, capacity | Acceptable Sharpe/Calmar & DD at realistic size |
| **G5 Paper** | Does it work live (no money)? | Live paper soak on VPS | Tracks backtest within tolerance over weeks |
| **G6 Live** | Scale with real capital | Small size → ramp on conviction | Live matches paper; kill-switch ready |

> Most ideas die at G1–G4. That is the system working. Our research/43–47 all died at
> G2–G4 and that was the correct, money-saving outcome.

---

## 2. Pre-flight — frame the question before you run anything

Write STATUS-MD **sections 1–4 before launching** (per `.claude/CLAUDE.md`). Lock:

- **The Ask, restated more precisely than asked** — universe, period, signal definition,
  the single success metric, and the gates a result must clear.
- **Economic hypothesis** — the mechanism, the counterparty, why it should persist, how
  it might decay. (Short-term reversal = illiquidity premium; momentum = under-reaction;
  breakout = late-comer flow. Name it.)
- **The base mechanics** — bar-by-bar signal, exits, costs, direction handling.
- **The grid** — axes × values × cell count; what's deliberately held fixed and why.
- **Falsification plan** — what result would make you *abandon* this idea. Decide it now,
  before you're attached to a number.

---

## 3. Data integrity & provenance (where most "edges" are born as bugs)

Before trusting any result, verify the data can support the claim:

- **Coverage check, always.** Print min/max date, symbol count, rows per year *for the
  exact series you use*. We were bitten twice: NIFTY50 daily **only starts 2023** (voided
  a whole run via idle-cash compounding), and broad-universe **5-min only densifies from
  ~2024** (collapsed a "12-year" test to 2 bullish years). Never assume history depth.
- **Survivorship bias.** A universe defined by *today's* liquid names, applied to the
  past, is biased. Use point-in-time universe reconstruction where it matters; otherwise
  **state the bias and report a modern sub-period**.
- **Look-ahead / lag discipline.** A feature at time `t` may use only data ≤ `t`. Forward
  returns are `t→t+h`. Rank on the close you'd actually have; trade on the *next* bar/open.
  Full-sample statistics (beta, vol, normalization) computed over all time and applied to
  the past are look-ahead — use trailing/expanding windows (we re-ran beta causally and it
  *held*, which is the only reason we trusted it).
- **Benchmark validity.** RS/relative metrics need a benchmark with full history and the
  right scale. Prefer a long-history ETF (`NIFTYBEES` from 2005) or a **synthetic
  equal-weight index** (median daily return across the universe) when the index series is
  short. Exclude the benchmark from the investable universe.
- **Corporate actions / data hygiene.** Watch splits/bonus gaps, bad prints, session
  filtering (drop 18:00+ junk bars), price floors for penny names.
- **Canonical host & snapshot.** `backtest_data/market_data.db` on the **VPS is canonical**;
  the laptop holds a frozen snapshot. Record *which* snapshot a result used. All Kite data
  writes are VPS-only.

---

## 4. The seven deadly sins (failure taxonomy — with our scars)

| Sin | How it fakes an edge | Guard | Where it bit us |
|---|---|---|---|
| **Look-ahead** | Future info leaks into the signal | Causal features only; trade next bar; trailing-window stats | Fade entry at whole-flag max (research/43); full-period beta (44) |
| **Survivorship** | Dead names silently dropped | PIT universe; report modern sub-period | 5-min universe = today's liquid names (44/45) |
| **Overfitting / data-snooping** | Best of N configs ≈ luck | Hold-out, walk-forward, fewer params, **multiple-testing haircut** (deflated Sharpe), monotonicity > peak-picking | Every grid sweep risks this |
| **Cost neglect** | Gross edge eaten by turnover | Always gross *vs* net; cost-sensitivity; break-even cost | Reversal break-even ~13 bps (46); H-pattern costs > edge (43) |
| **Regime dependence** | Works only in one regime | Per-year/sub-period; stress the opposite regime | Breakout long-beta thrived 2024 bull, died in pullbacks (44) |
| **Correlation / single-factor** | Many positions = one bet | Factor/sector caps; market-neutral; measure DD under cluster stress | Long high-beta breakout = −70%+ correlated DD (44) |
| **Capacity / liquidity / shortability** | Untradeable at size | Liquidity-filtered universe; impact model; India SLB/F&O short limits | Reversal short book not executable in cash (46) |

If you cannot name how each of these is controlled for, the result is not yet credible.

---

## 5. Cost & execution realism

- **Model costs explicitly and report gross + net side by side.** Make cost a parameter,
  not a constant buried in code.
- **Cost units matter.** Express per-trade cost in the *same unit as the edge* (R, bps).
  Tight intraday stops make a flat bps cost enormous in R-terms (research/43: 6 bps on a
  25 bps stop = 0.24R/trade of pure drag). 
- **Turnover is destiny for high-frequency edges.** Report average turnover and a
  **break-even cost**. If break-even is below realistic all-in cost, it's not a strategy
  (reversal: break-even ~13 bps vs ~10–15 bps reality → dead).
- **Slippage & impact** scale with size and inverse liquidity; model at least a fixed
  slippage and flag capacity limits. **Gaps** through stops fill at the next open, not the
  stop — model overnight gap risk for swing books.
- **Shortability (India-specific):** cash equities can't be held short overnight without
  SLB; only ~F&O names are shortable via futures. A long-short stock book must respect this.
- **Taxes** (STCG/LTCG) materially change net for holding-period-sensitive systems — model
  when comparing intraday vs swing vs positional.

---

## 6. Robustness protocol (mandatory before any "it works")

No strategy graduates G3 without **all** of these, recorded in RESULTS:

1. **Out-of-sample / walk-forward.** Reserve data the optimizer never saw; or roll
   train→test windows. In-sample-only "edges" are presumed overfit.
2. **Per-year / sub-period stability.** Show every year. A good average hiding 2 monster
   years and 5 losers is fragile. (We always tabulate per-year now.)
3. **Parameter sensitivity.** The edge must survive ±perturbation of every threshold.
   Prefer **monotonic dose-response** (e.g. PF rising with beta cutoff) over a lone peak.
4. **Causal-only features.** Re-run any normalization/beta/vol on trailing windows; if the
   edge evaporates, it was look-ahead.
5. **Adversarial kill attempt.** Actively try to break it: different universe, different
   period, costs +50%, exclude top contributors (super-winner guard — does the edge survive
   never holding its 3 best names?), shuffle/placebo tests.
6. **Multiple-testing honesty.** If you tried N configs, the best one's t-stat is inflated.
   Discount it (deflated Sharpe / Bonferroni intuition). Report how many configs were tried.
7. **Direction split.** Report long and short separately from the start — edges are often
   one-sided (43 short-only, 44 long-only).

---

## 7. Portfolio construction & risk (turning a signal into a book)

- **From R to equity curve.** Per-trade expectancy in R is the signal; convert to a real
  NAV with explicit position sizing (fixed-fractional risk, vol-targeting), concurrency,
  and capital.
- **Correlation is the silent killer.** Sum of positive-expectancy trades can still be a
  catastrophic book if they're the same bet. Measure drawdown under *clustered* stress.
  Prefer market-neutral / de-correlated construction (sector caps, one-per-theme, pairing a
  long sleeve with a short sleeve).
- **Drawdown budget is a hard constraint, not an output.** Define the max tolerable DD up
  front; a 45% CAGR at 70% DD is uninvestible. Calmar & MaxDD rank above raw CAGR.
- **Regime overlays** help *books that already work* (cut MQ's 27% DD) far more than they
  rescue broken ones. But a regime gate's value is *being flat* — don't bolt a directional
  options/ETF overlay onto risk-off periods expecting free return (research/47: nothing beat
  plain cash).
- **Capacity.** State the AUM at which impact erodes the edge. A retail-only edge ≠ a fund edge.

---

## 8. Metrics & reporting standards

**Always report** (per strategy, gross & net, full-sample & modern sub-period):

`CAGR · Sharpe · Sortino · MaxDD · Calmar · win% · profit factor · expectancy (R) ·
avg win/avg loss · trade count · avg holding · turnover · t-stat · per-year table ·
long/short split · cost-sensitivity · capacity note`

**Verdict vocabulary (use these exact labels):**
- **NO EDGE** — gross is flat/negative.
- **SIGNAL (not investable)** — net-positive per trade but fails portfolio/robustness/capacity.
- **STRATEGY (candidate)** — survives G4; ready for paper.
- **LIVE** — in paper/live with monitoring.
- **CONCLUDED / SHELVED** — dead; *say why* so it isn't re-litigated.

**Every RESULTS.md ends with a mandatory `Honest caveats` section** (biases, modeled
assumptions, sample limits) and a **`Next levers`** section. State the IV/data caveat
loudly when results are modeled (research/47 options were modeled — "directional, not
decision-grade").

---

## 8.5 Client Tearsheet — the presentation standard (visual-first)

The internal `RESULTS.md` is the *honest research verdict* (warts and all). The **client
tearsheet** is the *presentation layer* a fund hands a prospective investor: comprehensive
yet clean, **visual-first (charts > tables > text)**, benchmark-relative, scannable in 30
seconds, with the detail available below the fold. Generate it with
`research/_utilities/tearsheet.py` (`generate_tearsheet(strat_nav, bench_nav, name, meta)`)
which emits a one-page factsheet (PNG) + self-contained HTML. **Never hand a client a wall
of text.** A tearsheet is mandatory at G4+ and whenever a result is shown to Arun as a
"this is what we'd pitch" artifact.

**Top-of-page KPI strip (the 30-second story — big numbers, color-coded):**
`CAGR · Total return (×) · Excess CAGR vs index · Sharpe · Sortino · Max Drawdown ·
Calmar · Volatility · % of years beating the index`

**Mandatory visuals (in this order):**
1. **Growth of ₹100 — strategy vs benchmark (log scale).** The hero chart; the gap *is*
   the pitch. Linear inset optional.
2. **Underwater (drawdown) curve.** Clients fear losses more than they crave gains — show
   the pain honestly, with the max-DD depth and recovery time annotated.
3. **Yearly returns — strategy vs index, side-by-side bars.** Green/red, with the excess
   labelled. This answers "does it beat the index, every year?".
4. **Monthly-returns heatmap** (year × month, diverging color). Shows consistency & the bad
   patches at a glance.
5. **Rolling 12-month return** (strategy vs index) — demonstrates persistence, not one lucky
   run.
6. **Return distribution / histogram** with mean & worst month marked.

**Mandatory tables (compact, highlighted):**
- **Headline stats** (strategy vs benchmark, two columns): CAGR, vol, Sharpe, Sortino,
  MaxDD, Calmar, best/worst year, % positive months.
- **Versus-benchmark**: excess return, beta, correlation, up-capture / down-capture,
  tracking error, hit-rate (% periods > index).
- **Year-by-year**: strategy %, index %, excess % — color the excess column.

**Tone & honesty rules (a credible fund, not a brochure):**
- Lead with **risk-adjusted** numbers (Sharpe/Calmar), not just CAGR. Anyone can show CAGR;
  clients pay for *return per unit of drawdown*.
- **Always show the benchmark and the drawdown.** Hiding either is a red flag to a real
  allocator.
- **Footer disclosures, plainly:** backtest vs live, period, costs assumed, modeled
  assumptions (e.g. "modeled IV — directional"), survivorship/sample caveats, "past
  performance…". The internal caveats from RESULTS must be represented, not buried.
- **One page.** If it doesn't fit, it isn't a tearsheet.
- Net-of-cost, net-of-tax where relevant; state which.

> A factsheet that shows a beautiful equity curve and hides the drawdown is how funds lose
> credibility (and clients). Show the hole *and* how it recovered.

### Publish every completed study to the app (binding)

A study isn't "done" until it is **viewable in the app**, not just as files on disk. The
React SPA already has a **data-driven Backtest section** (`/app/backtest` →
`/app/backtest/<slug>`) that renders a uniform 8-section study layout (system rules,
metrics tiles, comparison tables with heatmaps, **finished-figure PNGs**, winner callouts,
caveats, links) from one registry. **Every COMPLETE study must be published there.**

**How to publish (frontend-only — safe any time, no backend restart):**
1. Generate the tearsheet PNG (`research/_utilities/tearsheet.py`) and any key charts.
2. Copy the PNG(s) into **`frontend/public/`** (they serve under `/app/<name>.png`).
3. Add/append a `BacktestStudy` object to **`frontend/src/data/backtests.ts`**
   (`BACKTEST_STUDIES`): `slug`, `title`, `verdict` (use the playbook verdict label),
   `status`, `cardStats` (the KPI strip), `systemRules`, `comparisons`, `results.metrics`
   + `results.tables` + **`results.charts: [{ src: '/app/<name>.png', caption }]`** (put the
   client factsheet first), `winners`, `caveats`, `githubLinks`, `projectPaths`.
4. `cd frontend && npm run build`; hard-refresh. No backend restart (Flask serves the new
   static bundle on next request) — so this is safe even during market hours.

The study page **is** the durable, shareable report — the factsheet PNG embeds as a
full-width figure with caption. Keep `results.charts[0]` = the client tearsheet so the
visual story leads.

## 9. Record-keeping & re-referencing (so future-you finds it in 30 seconds)

- **One folder per study:** `research/NN_<kebab-short-name>/` with `scripts/`, `results/`,
  and the STATUS-MD + RESULTS.md. `NN` is the next sequential number.
- **STATUS-MD naming:** `<DESCRIPTIVE>_<TIMEFRAME>_<TASK_KIND>_STATUS.md`, ALL-CAPS, per
  `.claude/CLAUDE.md`. It carries sections 1–8 and is the crash-recovery source.
- **RESULTS.md** = the honest verdict (+ `RESULTS_P2/P3/...md` for phases). Lead with a
  one-paragraph **verdict in bold**, then tables, then caveats, then next levers.
- **`research/INDEX.md` is the registry** — keep it current: one row per study with
  period, what, key result, and **verdict label**. (It is currently stale at phase 12;
  bringing it to 47 with verdict labels is a standing chore.)
- **`TODO.md` is the cross-session source of truth** — move work between Pending/In-Progress/
  Done; tag shelved studies CONCLUDED with the why; never accumulate stale lines.
- **Heavy artifacts** (multi-hundred-MB CSVs, logs) are gitignored; small rankings/STATUS/
  RESULTS are committed. Commit each study when it concludes.
- **Reproducibility stamp** in RESULTS: data snapshot date, key params, script path, cost
  assumption. No `Date.now()`/random seeds that break re-runs.

---

## 10. Resiliency & backup protocols (assume the worst)

- **Live STATUS-MD before launch**, updated at every state transition; a human must be able
  to resume from it alone (per `.claude/CLAUDE.md`).
- **Incremental, crash-safe writes.** Append per-symbol/per-cell results immediately; never
  batch to the end. **Resumable**: re-running skips completed work (track a done-set).
- **Background long runs + progress cadence.** Run sweeps detached; post a concise progress
  line ~every few minutes; never go silent for long stretches.
- **Aggregate-only recovery path.** If detection finished but aggregation crashed, support
  re-aggregating from the incremental files without re-running the sweep.
- **VPS is canonical** for data + long sweeps (laptop sweeps die unpredictably). Build/smoke
  on laptop snapshot → push → run on VPS. **Never restart `quantifyd` during market hours
  (09:15–15:30 IST).**
- **Version control as backup.** Commit scripts + STATUS + RESULTS at every milestone; push.
  Secrets never inlined (credential manager). The `claude-state` repo holds recovery docs.
- **Data backup/refresh.** The market DB is the crown jewel — keep the VPS copy authoritative
  and snapshot deliberately; log the snapshot date in results.

---

## 11. Continuous improvement & meta-learning

- **After every study, write the lesson down** (RESULTS + a line in this playbook's failure
  taxonomy if it's a new failure mode). Patterns compound: across 43–47 we learned
  *"directional intraday dies to cost+correlation; robust cross-sectional anomalies are
  eaten by turnover or already harvested."* That meta-insight is worth more than any single
  backtest.
- **Improve what already works before hunting new alpha.** The highest-EV move is usually to
  de-risk/extend a proven book (MQ 32–48% CAGR, KC6 PF 1.70) — a regime overlay, a long-short
  tilt, better sizing — not a fifth from-scratch signal that costs will eat.
- **Re-open shelved ideas when the inputs change** — more data (5-min backfill, real options/
  IV history once the live collector matures), lower costs, or a new construction (a shelved
  signal may live as a *hedged sleeve* even if it failed standalone).
- **Periodically re-validate live strategies** against fresh data; markets decay edges
  (reversal has visibly decayed since ~2021).

---

## 12. Agentic execution rules (how the agent must operate)

1. **STATUS-MD sections 1–4 before any launch.** No exceptions.
2. **Ask the human only on genuine forks** that change the design (strategy direction,
   universe, the success metric) — not on things you can default sensibly. State the data
   reality (coverage, modeling assumptions) *before* asking.
3. **Surface partial findings live**; don't wait for the full run.
4. **Be adversarially honest.** Lead with whether it's NO EDGE / SIGNAL / STRATEGY. Never
   oversell. If a chart/table contradicts your prior claim, retract it explicitly.
5. **Smoke-test on a few names first** to catch look-ahead/cost bugs (the fade look-ahead and
   the cost-domination were both caught in smoke tests — do this).
6. **Prefer cheap probes and analytical isolation** (e.g. isolate one overlay's marginal P&L
   rather than re-running everything) to answer "does X help?" precisely.
7. **Close the loop:** RESULTS.md verdict → update INDEX.md → update TODO.md → commit →
   propose the next highest-EV step.

---

## 13. Checklists (run these literally)

**Pre-launch (G0–G1):**
- [ ] Economic hypothesis + counterparty written
- [ ] STATUS-MD sections 1–4 written; falsification criterion set
- [ ] Data coverage printed (dates, names, rows/yr) for the *exact* series
- [ ] Universe survivorship & benchmark validity addressed
- [ ] Cost model & success metric + gates defined

**"Is it REAL?" (G2–G3):**
- [ ] Gross *and* net reported; cost-sensitivity table
- [ ] Causal-only features (trailing-window stats re-run)
- [ ] Per-year/sub-period table; survives the weak regime
- [ ] Parameter sensitivity / monotonicity (not a lone peak)
- [ ] OOS/walk-forward; adversarial kill attempt; super-winner guard
- [ ] Multiple-testing discount noted; long/short split shown

**"Is it INVESTABLE?" (G4):**
- [ ] Equity curve with real sizing; MaxDD within budget; Calmar/Sharpe acceptable
- [ ] Correlation/cluster-stress drawdown measured; de-correlation applied
- [ ] Capacity & shortability/liquidity feasible at target size

**Reporting & close-out:**
- [ ] RESULTS.md: bold verdict label + tables + `Honest caveats` + `Next levers`
- [ ] STATUS-MD → DONE; INDEX.md row added with verdict; TODO.md updated
- [ ] Reproducibility stamp (snapshot, params, script); committed & pushed

---

*Living document. When a study teaches a new failure mode or a better practice, add it
here. The fund that audits itself hardest keeps its clients longest.*
