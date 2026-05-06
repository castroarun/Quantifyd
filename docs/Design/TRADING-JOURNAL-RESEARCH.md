# Trading Journal — Research & Feature Survey

**Date:** 2026-05-06
**Purpose:** Survey best-of-breed trading-journal tools and academic
literature on trade-review practice; synthesise into a feature matrix
that informs the design of a Quantifyd-native journal.
**Author:** Quantifyd / Design

---

## 1. Why a journal at all?

Across Brett Steenbarger's *The Daily Trading Coach*, Mark Douglas's
*Trading in the Zone*, and the trader-development literature in general
(Lo, Tharp, Kahneman applied), the single highest-leverage activity for
a discretionary or semi-systematic trader is **structured review of
their own trades**. The journal is the substrate for that review.

For a multi-strategy, semi-systematic Indian-equity setup like
Quantifyd, the journal does four jobs:

1. **Single source of truth** for what actually happened — survives
   broker downtime, app restarts, mid-trade context loss.
2. **System scoreboard** — which strategy made money this month, which
   bled? Per-strategy P&L attribution is non-negotiable when 6+ systems
   run in parallel.
3. **Edge audit** — backtest expected EV vs realised EV per strategy.
   When realised drifts more than X% from expected, that system is
   degrading and needs review.
4. **Behavioural log** — overrides, missed signals, manual exits. The
   biggest leak in any semi-automated book is the human bypassing the
   system; the journal is where that leak gets measured.

---

## 2. Tool survey — what the leaders ship

Surveyed: **TraderSync**, **Tradervue**, **Edgewonk 2.0**,
**Chartlog**, **Trademetria**, **TradeZella**, plus open-source
**Profit-Loss-Reporter** and the journal modules of **TraderEvolution**
and **Quantower**.

### Common feature surface (every tool ships these — table-stakes)

| Feature | What it does | Notes |
|---|---|---|
| **Auto-import from broker** | CSV / API ingest of executed trades | Tradervue + TraderSync support 90+ brokers; in India, Zerodha Console export is the standard |
| **Round-trip aggregation** | Multiple fills → one trade row | Critical for partial-exit strategies (NAS legs, ATH-trail topups) |
| **Per-trade P&L (gross + net)** | After brokerage, STT, GST, exchange fees | Indian context = SEBI/STT charges very material for intraday |
| **Daily / weekly / monthly P&L summary** | Calendar heatmap, totals | Calendar heatmap is the most-recognised journal artefact |
| **Tagging system** | Free-form + structured tags per trade | Strategy, setup, mistake-type, conviction level |
| **Notes per trade** | Markdown, screenshots | Edgewonk pioneered the "trade thesis vs outcome" pairing |
| **Equity curve** | Cumulative net P&L over time | Needs to be filterable by tag/strategy |
| **Win-rate, PF, expectancy** | Standard performance metrics | Per-strategy + overall |
| **Search & filter** | By symbol, date, tag, side, strategy | Journal becomes useless without this past ~200 trades |

### What separates the elite tools (differentiators)

| Feature | Tools that have it | Why it matters |
|---|---|---|
| **MAE / MFE distribution** | Edgewonk, TraderSync, TradeZella | Maximum Adverse / Favourable Excursion. Tells you whether your stops are in the right place — if MAE rarely hits SL but trade exits at SL frequently, your fills are bad |
| **R-multiple reporting** | Edgewonk, Tharp's TradeMill | Trades expressed as multiples of initial risk. Win-rate is misleading; expectancy in R is the truth |
| **Trade replay / chart playback** | TradeZella, Quantower | Re-watch the candles around your entry/exit. Transformative for discretionary review |
| **Drawdown windows** | Tradervue, Edgewonk | Largest peak-to-trough sequences with date ranges. Identify regime breaks |
| **Mistake / rule-violation tagging** | Edgewonk (gold standard) | Pre-defined mistake categories (early entry, moved stop, oversized, FOMO entry) → leak analysis |
| **Trade-grade scoring** | Edgewonk, TradeZella | Score the trade *process* 1-5 independent of outcome. Decouples luck from skill |
| **Psychology / state-of-mind tracking** | Edgewonk, TraderSync | Pre-trade and post-trade emotion log. Correlate state with P&L |
| **Goal tracking** | TradeZella | Daily / monthly P&L and discipline goals |
| **Slippage tracker** | Quantower, custom | Expected fill vs actual fill. Massive in Indian intraday where bid-ask is wide on smaller-cap F&O |
| **Reverse-engineering analytics** | Edgewonk | "Best 10 trades vs worst 10 — what's different?" feature |
| **Public sharing / accountability** | Tradervue, TradeZella | Share annotated trades; coaching loop |

### Killer features specific to a multi-strategy, Indian-equity, intraday-+-swing book

These are the things no off-the-shelf tool has, but matter most for
Quantifyd:

| Feature | Why it matters here |
|---|---|
| **Per-strategy P&L attribution** | 6 systems running in parallel (ORB, MQ, KC6, NAS, Diamond Short, Long-MR, Long-TC). Needs to be one click to ask "How is the new Diamond Short doing this month?" |
| **NIFTY-regime tagging captured at trade time** | Auto-stamp every trade with: NIFTY 5m trend, daily ADX bucket, India VIX level, gap tier. Lets you slice "Long-MR in low-VIX regimes" vs "in high-VIX regimes" |
| **Expected EV vs realised EV** | Each system has a backtest expected R-per-trade. Plot rolling-window realised vs that. When realised drifts by >2σ, the system is regime-broken — pause it |
| **Kite tradebook auto-import** | Zerodha publishes a daily tradebook CSV (downloadable from Console) and a real-time Kite Connect `/orders` + `/trades` API. Auto-poll → reconcile against `orb_positions` etc. → flag any unmatched fill |
| **Slippage tracker (entry + SL + target)** | For each ORB trade, compare `signal.entry_price` (what we wanted) to `position.entry_price` (what we got) and the same for SL. India-specific because liquidity in 5-min OR breakouts can be thin in mid-caps |
| **Missed-signal log** | Backtest engine fires a signal but live engine didn't take it (e.g. Kite rejected, capital blocked, manual override). Logged with reason. This is where the biggest leaks live in semi-automated systems |
| **Override / manual-action log** | Every time the user clicks "kill switch" or manually exits a position, log it with a reason field. Track cost of overrides over time |
| **Post-trade-review checklist** | Daily 5-question review at session close: did I follow the system? Any rule violations? What surprised me? What's tomorrow's priority? Persisted as `daily_review` row |
| **Weekly system-grading** | End-of-week review per system: Green / Amber / Red. Trades, win-rate, PF, drift vs backtest, drawdown. Drives the "should I size up / down / pause" decision |
| **Pre-market intent log** | Before open, log expected behaviour: "ORB long bias today, NAS premium 8-12 acceptable, Diamond Short universe of 3 candidates flagged". Compare to what actually happened |
| **Integrated with Pre-Market Brief** | The Pre-Market Brief (`premarket_brief.py`) already produces a daily snapshot. Auto-attach today's brief to today's journal day-page so context is preserved |
| **Tax-pack export (India)** | At year-end, generate the FY P&L statement broken into intraday speculative vs delivery vs F&O — the three Indian tax buckets. Saves hours at filing time |

---

## 3. Academic & practitioner reading — what the literature says

- **Steenbarger, Brett — *The Daily Trading Coach* (2009).** Chapter on
  journaling: structure beats volume; review the *process* not the
  outcome; track patterns of mistakes, not isolated mistakes.
- **Douglas, Mark — *Trading in the Zone* (2000).** "The probabilistic
  mindset" — every trade is a sample from a distribution. The journal
  is what lets you see the distribution rather than overfit on the
  last trade.
- **Tharp, Van — *Trade Your Way to Financial Freedom*.** Introduced
  R-multiples to retail. Win-rate without R is meaningless.
- **Lo, Andrew & Hasanhodzic, Jasmina — *The Heretics of Finance*.**
  Across 13 elite traders, a common thread: a written record of *why*
  each trade was taken, reviewed weekly.
- **Kahneman — *Thinking, Fast and Slow*.** Hindsight bias means you
  cannot accurately remember why you took a trade unless you wrote it
  down at the time. The journal is the antidote.
- **Mauboussin, Michael — *The Success Equation*.** The skill / luck
  continuum. Outcome ≠ skill. Score process, not result.
- **Soltes, Eugene — *Why They Do It*.** On the slow drift of decision
  quality when not measured. Applies to traders bypassing their own
  system.

**Synthesis of the literature:**

1. **Process > outcome.** Score the trade independent of the P&L.
2. **Patterns > anecdotes.** Single trades don't matter; the
   distribution does.
3. **Pre-commitment > recall.** Write the thesis *before* entry, not
   after exit.
4. **Review cadence matters.** Daily (close), weekly (Sunday), monthly
   (1st of month), annual (April for Indian FY).

---

## 4. Synthesised feature matrix — what Quantifyd's journal should ship

Three tiers. Tier 1 = MVP. Tier 2 = Phase 2. Tier 3 = Phase 3.

### Tier 1 (Must-have — MVP, week 1)

1. **Auto-import from existing strategy DBs** — read `orb_positions`,
   `kc6_trades`, `nas_*`, `maruthi_*` and project them into a unified
   `journal_trades` table. No double-entry; the journal is a *view*
   over execution data first.
2. **Calendar heatmap (P&L by date)** — month grid, green/red
   intensity by net P&L, click date → day-page.
3. **Per-trade detail page** — entry, exit, P&L, R, hold-time, tags,
   notes, screenshot slot, links back to source strategy page.
4. **Per-strategy P&L attribution** — strategy chip on every trade,
   filter the equity curve by strategy.
5. **Tag system** — free-form + structured (strategy, setup,
   conviction, mistake-type).
6. **Notes per trade + per day** — markdown, attached screenshots.
7. **Equity curve + drawdown** — cumulative net P&L; current and
   max drawdown windows.
8. **Insights dashboard** — win-rate, PF, expectancy, R-distribution,
   per-strategy table.

### Tier 2 (Differentiators — Phase 2, weeks 2-3)

9. **Kite tradebook reconciliation** — pull `/orders` and `/trades`
   from Kite, match against journal trades, flag unmatched.
10. **MAE / MFE per trade** — for trades with intraday data, compute
    max-adverse and max-favourable excursion during hold.
11. **R-multiple reporting** — every trade expressed in R; R-curve;
    R-distribution histogram.
12. **Slippage tracker** — `signal.entry_price` vs `position.entry_price`,
    aggregated per-strategy.
13. **NIFTY-regime auto-tagging** — at trade-open, snapshot NIFTY 5m
    trend, daily ADX, India VIX, gap tier; persist to `trade_context`.
14. **Mistake-type taxonomy** — predefined list (early entry, moved
    stop, oversized, FOMO, override-against-system, …).
15. **Daily review form** — 5-question post-close checklist persisted
    per day.
16. **Trade-grade scoring** — 1-5 process score per trade,
    independent of outcome.

### Tier 3 (Killer / long-tail — Phase 3, months later)

17. **Expected EV vs realised EV** — per-strategy backtest baseline
    plotted against rolling realised; drift alerts.
18. **Missed-signal log** — diff `orb_signals` (or equivalent)
    against `orb_positions`; any signal with no matching position is
    a miss with a reason field.
19. **Override log** — every kill-switch click, manual exit, or
    paper→live toggle, with cost-of-override calculation.
20. **Pre-market intent log** — before-open form per session;
    auto-attach Pre-Market Brief.
21. **Weekly system-grading** — Sunday-night auto-generated card per
    system, Green/Amber/Red.
22. **Trade replay** — render the 5-min candles around each trade
    with entry/exit markers (uses `market_data_unified`).
23. **Tax-pack export** — FY P&L split by speculative / delivery /
    F&O for ITR filing.
24. **Public-share annotated trade** — for external coaching /
    discussion.

---

## 5. References

- TraderSync — <https://tradersync.com>
- Tradervue — <https://tradervue.com>
- Edgewonk 2.0 — <https://www.edgewonk.com>
- TradeZella — <https://tradezella.com>
- Chartlog — <https://chartlog.com>
- Trademetria — <https://trademetria.com>
- Zerodha Console (Indian broker journal export) — <https://console.zerodha.com>
- Kite Connect API (orders + trades) — <https://kite.trade/docs/connect/v3/orders/>
- Steenbarger blog — <https://traderfeed.blogspot.com>
- Van Tharp R-multiple primer — <https://vantharp.com/r-multiples>

---

## 6. Recommended scope for Quantifyd

Given Arun's actual setup (one trader, six live systems, mostly
automated, Kite as the only broker), the recommended Tier-1 + Tier-2
shape is small but pointed:

- **MVP (week 1):** items 1-8 above, with auto-import from
  `orb_positions` and `kc6_trades` driving the table from day one.
  Real ORB trades from April-May 2026 already exist — that's the
  test data.
- **Phase 2 (weeks 2-3):** items 9-16. Kite reconciliation is the
  single most impactful Phase-2 item — it closes the loop on "did
  the broker actually do what I think it did?".
- **Phase 3 (later):** items 17-24, prioritised by what hurts most.
  Missed-signal log probably first — that's where unknown leaks
  hide.

The next document (`TRADING-JOURNAL-DESIGN.md`) translates this into
schema, endpoints, pages, and a phasing plan.
