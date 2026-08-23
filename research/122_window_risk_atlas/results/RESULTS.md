# research/122 — The Window Risk Atlas: decay earned vs violent-move probability, every intraday straddle window

## VERDICT (per deployed cell, in one line each)

| Cell | Verdict |
|---|---|
| **MON NIFTY DTE1 13:00–14:00 SL20** | **DOWNSIZE (or drop)** — thinnest profit of the five, R:R@p95 **1 : 11.8**, breakeven terminal move just **11.5 bp**; re-confirms r/121's "cut Monday" by an independent route |
| **TUE NIFTY DTE0 09:30–11:00 SL25** | **KEEP** — best R:R of the whole book (**1 : 1.5 @p95**), 81% win, the deployed stop trips on ~**1.1%** of expiry-morning days, and the cell already took its tail day (−₹27,040) and stayed comfortably net-positive |
| **WED SENSEX DTE1 10:30–12:00 SL20** | **KEEP** — R:R@p95 **1 : 2.9**, P(loss) ≈ 29% by both routes, and the calmest long-sample window of the five (p95 excursion 66.7 bp) |
| **THU SENSEX DTE0 13:00–15:20 NO-STOP** | **KEEP — but size for the tail it actually has.** The biggest earner (median +₹15,070/day, 58% of the five-cell take) carries a bridged p95/p99 adverse of **₹47k / ₹70k at 10 lots**, has already printed **−₹54,100** inside 17 recorded Thursdays, and a stop cannot fix it: an SL20-style stop would fire on **85% of expiry afternoons** (r/114/118's result, reproduced). Size is the only dial. |
| **FRI NIFTY DTE2 10:00–12:00 SL20** | **KEEP** — 93% win, worst observed −₹3,440, R:R@p95 1 : 6.9; r/120 already adjudicated this cell and nothing here overturns it |

**No alternative window is recommended.** TUE and THU have no dominators at all. MON, WED and
FRI have cells that *formally* dominate (higher median, lower p95 adverse), but every one of
them is either (a) a last-hour window — the family r/120 already showed is 0.58–0.81 correlated
with COMB, which is in the market over exactly those minutes, (b) an overlap/extension of the
deployed window itself, or (c) r/120's already-adjudicated "start Friday earlier" marginal call
— and all of them are 16–17-day medians inside a ~220-comparison family, which r/120's
Westfall–Young precedent says cannot be distinguished from the search maximum. They are logged
as highlights for the 2026-11 re-run, not as recommendations.

---

## 1. What was asked, and what was actually built

> **Arun:** "did you look at other windows? I also asked for the probability of the worst-case
> scenarios. For the decay look at the options data we have; for the violent market move, look
> at the price action data we have for years (the least timeframe, 1-min or whatever) and come
> up with your report/recommendations in a table clearly with R:R, 90th percentile, 95th,
> probability of losses, other relevant stats."

Built: **one row per candidate window** — the 5 deployed TimeB cells plus a pre-registered
grid of 12 starts (09:20→14:30, 30-min steps) × 4 durations (60/90/120/hold-to-15:20), per
venue per DTE — marrying:

- **Stage A (decay, rupee truth):** every recorded day of the real 1-minute option chain
  (`options_data.db :: option_chain`, 2026-04-20 → 2026-08-21, READ-ONLY), both venues, ATM
  straddle sold at the window start, covered at the end or on the stop, net of 0.5/1.0 pt
  slippage per leg-side + ₹30/leg-side/lot (₹250/lot NIFTY, ₹200/lot SENSEX round trip).
- **Stage B (violent moves, the long clock):** `market_data.db :: market_data_unified` —
  SENSEX **1-minute** 2021-01→now (1,354 days) and NIFTY50 **5-minute** 2015-02→now
  (2,754 days), maximum adverse excursion and terminal move inside every window, in bp of the
  window's own entry price.
- **The bridge** (§4): 2026-observed premium-rise slopes and a **credit ladder**
  (p25/median/p75 of observed credits) convert Stage-B bp percentiles into rupees at 10 lots.

**Days are labelled by trading-day DTE, never by weekday name** (r/118: the SENSEX weekly
expiry moved Fri→Tue→Thu inside our own sample; NIFTY moved Thu→Tue on 2025-09-01; weekly
NIFTY options exist only from 2019-02, earlier days carry no label and are excluded from
DTE-matched samples). Frozen-chain holidays (2026-05-01, 05-28, 06-26; <50 distinct spot
prints) and partial sessions are rejected by data rule. 2026-08-21 IS included — its session
was complete (last snapshot 15:40).

## 2. Reconciliation before publication (required, passed)

| Check | This study | Prior | Match |
|---|---|---|---|
| FRI NIFTY DTE2 10:00–12:00 SL20, days ≤ 08-14 | n=14, mean **+399/lot**, worst **−344**, 13/14 | r/120: +400 / −344 / 13-14 | **exact** |
| MON NIFTY DTE1 13:00–14:00 SL20, days ≤ 08-17 | n=17, median **+1,240**, mean +999, worst −4,840, 71% (@10L) | r/121 §3 | **exact** |
| WED SENSEX DTE1 10:30–12:00 SL20, days ≤ 08-19 | n=17, median +3,370, mean +1,403, worst −11,440, 71% | r/121: +3,830 / +1,922 / −11,440 | **resolved** |

The WED gap is one holiday-shifted expiry week: 2026-05-28 (Thu) was a market holiday, so that
week's expiry was **Wednesday 05-27 (DTE0)** and the true DTE1 day was **Tuesday 05-26**.
r/121 selected by weekday (includes 05-27, excludes 05-26); this study selects by DTE —
selecting by weekday reproduces r/121 to the rupee (median +3,830, mean +1,923). The DTE
selection is kept: it is what the DTE-keyed live config actually trades.

## 3. THE ATLAS — deployed cells in full (₹ at 10 lots; books actually run 8)

| | MON NIFTY DTE1 13:00–14:00 | TUE NIFTY DTE0 09:30–11:00 | WED SENSEX DTE1 10:30–12:00 | THU SENSEX DTE0 13:00–15:20 | FRI NIFTY DTE2 10:00–12:00 |
|---|---|---|---|---|---|
| arm | SL20 | SL25 | SL20 | **NO STOP** | SL20 |
| n_opt / n_px | 17 / 353 | 16 / 358 | 17 / 124 | 17 / 122 | 15 / 354 |
| credit p25/med/p75 (pts) | 144 / 185 / 203 | 92 / 115 / 127 | 553 / 621 / 682 | 201 / 237 / 295 | 233 / 262 / 281 |
| **median net P&L** | **+1,240** | **+9,525** | **+3,370** | **+15,070** | **+3,120** |
| mean net / total | +999 / +16,980 | +7,552 / +120,840 | +1,403 / +23,850 | +15,571 / +264,710 | +3,871 / +58,070 |
| win % | 70.6 | 81.2 | 70.6 | 82.4 | 93.3 |
| **P(loss day)** obs \| modelled | 29.4 \| **52.1** | 18.8 \| 15.9 | 29.4 \| 29.0 | 17.6 \| 22.1 | 6.7 \| 4.8 |
| exc p90/p95/p99 (bp, long) | 60 / 78 / 133 | 69 / 81 / 121 | 56 / 67 / 111 | 76 / 87 / 133 | 69 / 92 / 129 |
| **p90 adverse ₹** (c25/cmed/c75) | 9.8k / **11.9k** / 12.8k | 10.7k / **12.7k** / 13.8k | 7.7k / **8.5k** / 9.1k | 34.9k / **40.8k** / 50.4k | 15.2k / **16.8k** / 17.8k |
| **p95 / p99 adverse ₹** (cmed) | 14.6k / 23.2k | 14.6k / 20.5k | 9.7k / 14.8k | **46.9k / 70.0k** | 21.7k / 29.4k |
| max-rung adverse ₹ (cmed) | 32.3k | 48.0k | 19.0k | **102.7k** | 68.6k |
| P(move > deployed SL cap) | 0.6% (n 353) | 1.1% (n 358) | 0.0% (n 124) | n/a (no stop; SL20 would trip **85.2%**) | 0.3% (n 354) |
| SL cap ₹ (cmed) | 26.5k | 21.2k | 26.8k | — | 36.6k |
| worst observed (options) | −4,840 (07-20) | **−27,040** (05-19, stopped) | −11,440 (06-10) | **−54,100** (06-11) | −3,440 (07-17) |
| **R:R @p90 / @p95** | **1:9.6 / 1:11.8** | **1:1.3 / 1:1.5** | 1:2.5 / 1:2.9 | 1:2.7 / 1:3.1 | 1:5.4 / 1:6.9 |
| breakeven terminal move | **11.5 bp** | 43.1 bp | 23.8 bp | 34.9 bp | 68.6 bp |
| exc p95/p99 ex-2021-22 | 52 / 78 | 78 / 86 | 67 / 111 | 87 / 133 | 67 / 115 |
| **verdict** | **DOWNSIZE / DROP** | **KEEP** | **KEEP** | **KEEP (size for tail)** | **KEEP** |

Denominators: P(loss) observed is over n_opt recorded 2026 days; P(loss) modelled and all
P(move>…) are frequencies over the n_px DTE-matched long-sample days shown in row 2. Adverse
percentiles are **uncapped** bridge values (what the move implies if the stop does not save
you); R:R uses the **stop-capped** p90/p95 (min of bridge value and SL cap) because the stop
is deployed — for THU nothing caps. The full surface (1,590 rows: every grid cell × venue ×
DTE × 3 stop arms) is `results/atlas.csv`; long-sample percentile tables incl. the
with/without-2021-22 regime split are `results/percentiles_long.csv`.

### How to read the two P(loss) columns honestly

The observed column is 16–17 benign-quarter days; the modelled column is 122–358 days through
a 2026-fitted P&L line. Where they agree (WED 29/29, TUE 19/16, FRI 7/5, THU 18/22) the cell's
loss frequency is probably real. Where they diverge — **MON 29 observed vs 52 modelled** — the
model is saying the recorded quarter was kind: Monday's median profit is so thin that a
terminal move of just 11.5 bp (roughly the *median* one-hour NIFTY move) erases it. Half of
all history-shaped Mondays lose. That is the same arithmetic that made r/121 call Monday
unreachable for 1:2.5, arrived at from the move side instead of the cost side.

## 4. THE BRIDGE — assumptions, stated plainly (the weakest link)

Stage B measures **bp of spot**; rupees require the option book. Conversion:

> adverse ₹(percentile) = credit(pts) × **b** × exc_bp(percentile) × lot × 10 lots + costs

- **b** = the 2026-observed premium-rise slope: median over recorded days (excursion ≥ 20 bp)
  of (max combined-premium rise ÷ credit) ÷ (max underlying excursion in bp). Fitted per cell
  where n ≥ 8, else pooled per venue-DTE. Full map with the bp level at which each stop trips:
  `results/bridge_map.csv`. Sanity: it puts SL20 at 154 bp (MON), 215 bp (WED), 164 bp (FRI)
  vs r/121's per-cell Theil-Sen 133/132/224 bp — same order, different fit; both agree the
  deployed stops are **decorative on non-expiry days** (fire ≲1% of days) and would be
  **ruinous on expiry afternoons** (85% of days).
- **Credit ladder**, not one number: the p25/median/p75 of the credits the market actually
  paid for that exact window in 2026. The p90-adverse row shows the sensitivity — spanning
  the ladder moves the answer by roughly ±10–25%. A high-VIX regime sits **above the p75
  rung**: the ladder spans 2026's pricing, not 2020's.
- **What the bridge assumes:** the premium's adverse extreme is proportional to the
  underlying's adverse extreme at the 2026-observed gamma/vega mix; the rest of the premium is
  carried at entry value (decay earned before the spike is not credited, spike-driven IV rise
  is not charged). The second omission dominates on true tails, so **bridged tail rupees are
  floors, not ceilings**. Proof inside the sample: TUE's own recorded worst (−₹27,040) already
  exceeds its bridged p99 (−₹20,470) — an SL25 fill on an above-median-credit day beat the
  model's tail. P(loss) uses a separate Theil-Sen line of booked net vs |terminal move| per
  cell — grounded in real P&L, but still a straight line through 16 points.
- **The 5-min licence (NIFTY):** no NIFTY 1-minute series exists. The NIFTY long sample is
  5-minute under r/121's proof that for the **max excursion inside a fixed window** 5-min
  equals 1-min *exactly* (max of bar highs is resolution-invariant; 0 differing rows in 4,068
  window-days). The no-5-min rule bites on the *path* — which minute a stop fires — and no
  path statistic here uses 5-min data. Entry price is the first bar's close at/after the
  start minute (≤ one bar of imprecision).

## 5. Alternatives — the highlights the atlas surfaces (and why none is a recommendation)

Scan rule: a grid cell "dominates" a deployed cell when its median net ≥ deployed **and** its
p95 adverse ≤ deployed, same venue-DTE, same stop family; it must then pass r/120's plateau
rule (agreeing neighbours). Full output: `results/alternatives_report.txt`.

- **TUE NIFTY DTE0, THU SENSEX DTE0: no dominators.** The deployed expiry-day cells are the
  best-shaped windows on their own days. (Thursday's 09:50-hold cell earns a bigger median —
  +35,100 — but at p95 adverse ₹60k it dominates nothing; it is the r/114 morning-entry HOLD,
  a different and riskier trade.)
- **MON NIFTY DTE1:** the real signal in the noise is that **every afternoon extension beats
  the deployed hour** — 13:20–15:20 median +4,940 (p95 adverse 9.8k), 13:50–15:20 +4,160
  (4.5k), all plateau-PASS, against the deployed +1,240. Read with §3: this is less "move the
  window" than "the deployed Monday hour is the thinnest slice of a mediocre afternoon". It
  overlaps the deployed window, sits in the COMB-correlated last-hour family (r/120 measured
  r = 0.58–0.81 on the Friday equivalents), and is a 17-day median in a ~220-comparison
  family. Logged for the 2026-11 re-run; the action that is already defensible on Monday is
  r/121's: cut size.
- **WED SENSEX DTE1:** only the 14:30–15:20 sliver (+4,800, p95 6.2k, one window counted four
  times by duration clamping) — pure last-hour/COMB family, not pursued.
- **FRI NIFTY DTE2:** 09:50–11:50 (+3,770 vs +3,120, p95 8.4k) — this is r/120's "start ~25
  minutes earlier" marginal call again (t = 1.99 there); r/120 already ruled: leave it, re-check
  with ~28 Fridays. Nothing new.

Family-wise haircut: ~44 grid cells × 5 deployed rows ≈ 220 ordered comparisons on 15–17-day
medians. r/120 ran Westfall–Young on the directly comparable Friday surface and **nothing**
survived at n = 14; nothing here is claimed either.

## 6. What the atlas says about the book as a whole

1. **The five-cell book's income is one Thursday cell.** THU contributes +₹2.65L of the
   +₹4.84L five-cell total at 10 lots. Its no-stop design is *correct* (any stop fires on
   85% of expiry afternoons and r/114/116/121 all showed stops convert decay into booked
   losses) — but that means its risk is managed **only by size**. At 10 lots the long sample
   prices a p95 afternoon at ₹47k, a p99 at ₹70k, and 2026 has already delivered −₹54k. A
   1-in-20 Thursday costs ~3 median Thursdays. That is the honest trade being run.
2. **Tuesday is the best-priced risk in the book** — the only cell whose p95 adverse is
   roughly its own 1.5×-median (R:R 1:1.5), because the expiry-morning credit is huge
   relative to the move the morning actually delivers (be_term 43 bp vs median morning move
   well below that).
3. **Monday is the worst-priced** — R:R@p95 1:11.8 and a modelled coin-flip loss rate. Third
   study in a row (r/121 cost-arithmetic, r/120 window-surface, now move-probability) to
   conclude the same thing by different routes.
4. **The regime split matters for tails**: NIFTY exc-p95 on the Monday window is 78 bp over
   2019→now but 52 bp over 2023→now. The atlas prices risk off the full sample deliberately —
   2021-22-style regimes return. SENSEX DTE samples only exist 2024→now (n 122–124); their
   p99s rest on ~1 day and should be read as "at least".
5. **The stops are decorative where deployed and lethal where not deployed** — non-expiry
   SL20/SL25 caps sit at moves the day delivers ≤1% of the time; the same construction on
   expiry afternoon would fire almost daily. The config is on the right side of both facts.

## 7. Sins accounting

| Sin | Control |
|---|---|
| Look-ahead | Strike from the spot at the window's own start minute; stop evaluated minute-forward; expiry derived per day from the chain; DTE from data-derived era tables |
| Survivorship / selection | Every recorded day used; holidays/partials removed by data rule (<50 spot prints, last snap <15:15), never by P&L; long sample = full index history |
| Overfitting / multiple testing | Whole 1,590-row surface reported; dominance scan pre-specified (median AND p95, plateau rule); ~220-comparison family named; no new window crowned; r/120's max-t precedent cited as the bar nothing here clears |
| Cost neglect | Net everywhere (₹250/₹200 per lot RT); cost is per-trade constant so 2× cost subtracts exactly ₹2,500/₹2,000 @10L from every median |
| Regime dependence | Tail percentiles from 2015→/2021→ samples, NOT the 2026 quarter; with/without-2021-22 split published per cell |
| Correlation / single factor | Late-window "dominators" explicitly discounted for COMB overlap (r/120's r = 0.58–0.81); the five cells are acknowledged as one short-gamma family, never summed as diversification |
| Capacity / shortability | Unchanged from r/120's margin work; no size increase proposed anywhere |
| Bridge honesty | Assumptions in §4; bridged tails labelled floors; in-sample counter-example (TUE) shown rather than hidden |

## 8. Caveats

1. **n_opt = 15–17 per cell, one benign quarter.** Observed win rates and worsts are not
   estimates of anything; the long-sample columns exist because of that. Size from §3's
   bridged percentiles, not from the observed worsts.
2. **The bridge is a straight line through 2026.** IV-pop days will beat it (floors, not
   ceilings). The THU p99 of ₹70k should be treated as "₹70k or worse".
3. **SENSEX DTE-matched history is short** (2024→, n≈122) because weekly-expiry DTE labels
   don't exist before; its p99 is one day deep.
4. **DTE-era labelling ignores intra-week holiday shifts** in the long sample (a ~1-day label
   error on holiday weeks; the 2026 options sample derives expiry per day from the chain and
   has no such error).
5. **Single-entry model** — one straddle at the start minute, 1-min LTP, fixed slippage; no
   5-second polling, no dwell, no 50% disaster backstop. Stop fills optimistic by construction.
6. **P(loss) modelled** rests on a 16-point Theil-Sen per cell; treat ±10 percentage points.

## 9. Files

| File | Purpose | Committed? |
|---|---|---|
| `scripts/stage_a_alldays.py` | options-chain replay, all days × grid × 3 arms | yes |
| `scripts/stage_b_allweekday_clock.py` | long-sample excursion clock, DTE-labelled | yes |
| `scripts/build_atlas.py` | bridge + atlas assembly + reconciliation | yes |
| `scripts/analyze_alternatives.py` | dominance/plateau scan | yes |
| `results/atlas.csv` | THE ATLAS — 1,590 rows, every window × venue × DTE × arm | yes |
| `results/percentiles_long.csv` | long-sample exc/terminal percentiles + regime splits | yes |
| `results/bridge_map.csv` | premium-rise slopes; bp level each stop trips at | yes |
| `results/alternatives_report.txt` | full dominance scan output | yes |
| `results/reconciliation.txt` | r/120 + r/121 baseline checks | yes |
| `results/stage_a_alldays.csv` (3.7 MB) | per-day per-window options replays | no (gitignored) |
| `results/stage_b_window_days.csv` (15 MB) | per-day per-window long-sample moves | no (gitignored) |
| `results/RESULTS.md` | this report | yes |

**Reproducibility stamp.** Data snapshot 2026-08-21 evening (`options_data.db`,
`market_data.db`, both opened `mode=ro`). All runs `nice -n 10` on the VPS. Costs ₹250/lot
NIFTY, ₹200/lot SENSEX round trip. Deployed cells read from
`backtest_data/csl_paper_config.json` (refrozen 2026-08-20), not from memory. No live config,
service, engine, order path, or frontend was touched.
