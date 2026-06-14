# GTAA ETF Rotation — Validate Upstox "Strategy 1" & Beat It on Its Own Terms

STATUS: DONE — verdict STRATEGY (candidate); see results/RESULTS.md

---

## 1. The Ask

**What you asked:** (from an Upstox / Darshan Rathod "Zero to Masterclass" slide,
"Strategy 1: Long only Multi Assets") — *"validate and assess this, see if we can
[build] long only portfolios better than this."* You chose **Both**: run the proper
study AND frame it against our existing higher-octane books.

**What we're actually testing:** A monthly multi-asset momentum rotation over Indian
ETFs. Faithfully **replicate** the slide's 3-ETF top-1 strategy to confirm its claim
(15.45% CAGR / 16.62% MaxDD / **Calmar 0.93**), then try to **beat it on its own terms**
— same beginner-simple, low-turnover, tax-efficient, low-DD profile — by adding the
defensive mechanics the slide omits. Finally, place the result on a risk/return map
against research/41 (regime-gated midcap RS, Calmar ~1.7) and research/62 (Momentum-30
ETF sub-select, Calmar ~1.7).

**Success metric:** **Calmar** (CAGR / MaxDD), net of cost, over the full common ETF
history. Secondary: Sharpe, MaxDD depth, per-year stability, turnover/tax friendliness.

**Gates a result must clear to be called "better":**
- Net Calmar > 0.93 (the bar), AND
- MaxDD ≤ the baseline's (don't buy Calmar by levering return into deeper holes), AND
- Per-year stable (no single-year carry), AND
- Survives ±lookback perturbation (monotonic-ish, not a lone peak), AND
- Turnover low enough that it stays the "simple ETF" archetype (not a churn machine).

---

## 2. Economic hypothesis

- **Cross-asset (relative) momentum** — capital chases the strongest trend; under-reaction
  + herding makes the recent 6–12m winner keep winning over a 1-month horizon. Counterparty:
  late rebalancers / mean-reversion sellers. Decay risk: crowding into the same ETFs.
- **Absolute momentum / trend filter (close > MA)** — the *defensive* half. Being in cash
  (LIQUIDBEES / 1-day-rate) when nothing is above trend is what historically cuts the
  drawdown of a buy-and-hold rotation. This is the lever the slide under-specifies and the
  most likely source of its mediocre 0.93 Calmar.
- **Diversification across uncorrelated sleeves** (Indian equity / gold / US tech) — gold &
  Nasdaq carry the book when Indian equity is in drawdown and vice-versa.

**Why it should persist:** momentum is the most-replicated anomaly across asset classes
(AQR/Antonacci/Faber); the trend filter is a risk-management overlay, not an alpha claim.

**Falsification (decided now):** if, after a real cash leg + abs-momentum gate, NO config
beats Calmar 0.93 **with MaxDD ≤ 16.6%** and per-year stability, we declare the Upstox
top-1 design already near-optimal for its archetype → verdict NO IMPROVEMENT, and the only
"better" is the different-profile equity books (41/62). Conversely if costs eat the cash-leg
gains we say so loudly.

---

## 3. The Base (mechanics being tested)

- **Bar:** daily ETF closes → **resample to month-end** (last trading day of month).
- **Signal (per month-end t, using data ≤ t):**
  - `ROC(12)` = 12-month rate of change (relative-momentum rank score).
  - `MA(6)` = 6-month simple MA of monthly close; "bullish" = `close > MA(6)`
    (absolute-momentum / trend filter).
- **Selection:** rank eligible assets by ROC(12), take **top-N** (baseline N=1).
- **Cash rule (the key variant):**
  - *Baseline (slide-faithful):* always hold the top-N risk assets regardless of trend.
  - *Improved:* an asset only enters if it is **bullish** (close>MA6) **and** ROC>0;
    any unfilled slot goes to **LIQUIDBEES** (1-day-rate / cash proxy).
- **Weighting:** top-N equal-weight (baseline N=1 → 100% one asset).
- **Rebalance:** monthly, on the month-end close; trade modeled at that close (or next-month
  open in a robustness pass).
- **Costs:** modeled explicitly as a per-side bps on traded notional (ETF reality: brokerage
  ~0 on delivery at discount brokers, but STT 0.1%/side + exchange + slippage). Test
  **0 / 10 / 20 / 40 bps per side**; report gross AND net. ETFs held in delivery → equity
  LTCG/STCG noted but low turnover keeps it light (report turnover).
- **Universe (core):** NIFTYBEES, GOLDBEES, MON100 (the slide's 3).
  **Universe (extended):** + JUNIORBEES (Next-50), BANKBEES, SILVERBEES, MIDCAP/MAFANG where
  history allows, with LIQUIDBEES as the safe asset.
- **Period:** full common history of the 3 core ETFs (MON100 lists ~2011 → that's the binding
  constraint; NIFTYBEES 2005, GOLDBEES 2007). Report a modern sub-period too.
- **Direction:** long-only (it's an ETF rotation; no shorting).

---

## 4. Plan (variant grid)

Cheap-first (G1→G2). Each axis multiplies; we kill early.

| Axis | Values | Why |
|---|---|---|
| Universe | {3-core} , {extended} | does a wider menu help or just add noise? |
| Top-N | 1, 2, 3 | concentration vs smoothing (DD lever) |
| Cash leg | off (faithful) , on (abs-mom gate → LIQUIDBEES) | the main DD/Calmar lever |
| Momentum lookback | ROC12 (faithful), ROC6, blended(3/6/12) | robustness, not peak-picking |
| Trend filter | MA6 (faithful), none, MA10 | sensitivity |
| Cost | 0 / 10 / 20 / 40 bps per side | net-of-cost honesty |

**Cell count:** start with the **faithful replication** (1 cell) to confirm ~0.93, then a
focused grid ≈ {2 universes × 3 N × 2 cash × 3 lookback × 2 trend} = 72 logical configs ×
cost-sensitivity applied post-hoc on the survivors. Most die at G2; only Calmar>0.93 +
DD≤16.6% survivors get the per-year / lookback-sensitivity / next-open robustness pass (G3)
and a tearsheet (G4).

**Deliberately fixed:** monthly frequency (the archetype's whole point); long-only; equal-
weight within top-N (risk-parity is a later lever if equal-weight already wins).

---

## 5. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-06-14 ~10:30 | Framing done; STATUS sections 1-4 written | Data gap identified: need GOLDBEES + MON100 |
| 2026-06-14 ~11:00 | VPS Kite token was stale (Sunday, no auto-login) | Fixed via `services.kite_auth.auto_login()` (TOTP) — no user action |
| 2026-06-14 ~11:10 | ETF download done (7 syms) | **Kite serves these only from 2015-01** (SILVERBEES from 2022). Common core3 window = 2016-02→2026-06 after 12m warmup |
| 2026-06-14 ~11:20 | Engine built + common-window trim fix | pre-2015 single-asset artifact (held NIFTYBEES thru 2008, fake -55% DD) removed |
| 2026-06-14 ~11:30 | Full 108-config sweep DONE | `results/gtaa_sweep.csv` |
| 2026-06-14 ~11:35 | Independent verification of winner | equal-weight confirmed real (not a bug) |

**Live findings:**
- **Slide top-1 momentum rotation is WEAK in our window:** core3_top1_roc12_ma6 = Calmar **0.30** (nocash) / **0.44** (cash-gated), DD −34%/−25%. Cannot reproduce the slide's 0.93 — almost certainly a longer/older window (Kite history starts 2015).
- **WINNER = naive equal-weight (hold all 3, monthly reb):** Calmar **1.73** (engine, 2016-26) / **1.55** (independent, 2015-26), CAGR ~17-19%, **DD only −11.3%**, turnover ~0, no momentum selection. Diversification beats rotation.
- **Why:** correlations Nifty/Gold −0.08, Nifty/Nasdaq +0.25, Gold/Nasdaq +0.04 — three uncorrelated sleeves. Equal-weight harvests the diversification; top-1 throws it away by concentrating into the hot asset then eating the reversal.
- **Caveats logged:** (1) MON100 INR CAGR 24.6% (Nasdaq bull + INR depreciation) carries much of the absolute return — won't repeat at that rate; the *low-DD/diversification* property is the robust part. (2) LIQUIDBEES price-return ≈0% (daily-dividend ETF) → cash-leg yield understated ~6%/yr; winner uses no cash leg so unaffected. (3) Window never saw an all-3 simultaneous crash.

---

## 6. Crash Recovery (resume without Claude)

- **Host:** ALL work on VPS `arun@94.136.185.54:/home/arun/quantifyd`. Laptop has NO DB.
- **Python:** use `venv/bin/python` (NOT system python3 — missing dotenv).
- **Data:** `backtest_data/market_data.db`, table `market_data_unified`. Verify ETF presence:
  `venv/bin/python -c "import sqlite3;print(sqlite3.connect('backtest_data/market_data.db').execute('select symbol,min(date),max(date),count(*) from market_data_unified where timeframe=\"day\" and symbol in (\"GOLDBEES\",\"MON100\",\"NIFTYBEES\") group by symbol').fetchall())"`
- **Download (if ETFs missing):** `venv/bin/python research/63_gtaa_etf_rotation/scripts/download_etfs.py`
- **Run sweep:** `venv/bin/python research/63_gtaa_etf_rotation/scripts/run_gtaa_sweep.py`
  (writes incremental rows to `results/gtaa_sweep.csv`; re-running skips done labels).
- **Safe to inspect:** everything under `research/63_gtaa_etf_rotation/results/`.
- **Do NOT** restart `quantifyd` during market hours (09:15-15:30 IST Mon-Fri) — not needed
  for this (research only, no service change).

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/download_etfs.py` | One-shot Kite download of the ETF menu (VPS) | yes |
| `scripts/gtaa_engine.py` | Monthly GTAA backtest engine (signal, cash leg, costs) | yes |
| `scripts/run_gtaa_sweep.py` | Variant grid runner (incremental CSV) | yes |
| `GTAA_..._STATUS.md` | This file | yes |
| `results/gtaa_sweep.csv` | Per-config metrics | yes (small) |
| `results/*_equity.csv` | Equity curves of finalists | yes if small |
| `results/RESULTS.md` | Final verdict | yes |
| `results/*.png` | Tearsheet / charts | yes |

---

## 8. Findings (during + final)

_(to be filled)_
