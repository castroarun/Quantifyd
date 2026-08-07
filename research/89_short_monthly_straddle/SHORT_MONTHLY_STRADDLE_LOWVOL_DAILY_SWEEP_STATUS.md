# Short Monthly Straddle in Predicted Calm Regimes — NIFTY/BANKNIFTY (real IV) + F&O-Liquid Stock Basket (modeled IV)

**STATUS: DONE — VERDICT: NO ROBUST TRADEABLE EDGE (confirmed on REAL option data + liquidity test). See `results/RESULTS.md`.**
Research folder: `research/89_short_monthly_straddle/` · Started 2026-07-22 09:51 IST · Runs on VPS (canonical DB host)

---

## 1. Headline

Can we *predict*, causally, when a liquid Indian underlying is entering a low-volatility
phase — and does selling a ~monthly ATM straddle into that phase harvest a real, net-of-cost
edge, or does the thin premium plus the calm-ending-in-a-jump tail eat it? Deliverables: a
low-vol **detector**, an empirically **calibrated probability the calm persists** over the
holding window (the "confidence score"), a **recommended holding length + exit rule**, and an
honest **verdict** graded against the seven deadly sins.

---

## 2. The Ask

**What you asked (verbatim intent):** "Identify periods in any liquid stock likely to go
through a subtle / least-volatile phase so that deploying a monthly straddle makes sense.
Assess the probability of such a period sticking to a low-volatility level, our confidence
score, how many days recommended, when to exit. Do an elaborate study and backtest, test
extensively, report collectively."

**What we are actually testing (cleaned up):**
A short ATM straddle profits when the **realized move over the holding month comes in below the
premium collected** — i.e. it is short gamma / short vega / long theta. So the ask decomposes
into three measurable questions:

1. **Detection** — using only past data, can we flag that an underlying is in / entering a
   low realized-vol regime? (Vol clusters, so this should carry signal.)
2. **Persistence + confidence** — given the flag fires, what is the empirical probability the
   underlying *stays* calm over the next ~20 trading days, and is that probability well
   calibrated? (Pure daily-data question — the most robust deliverable, independent of any
   option-pricing assumption.)
3. **Harvest** — does conditioning a short straddle on the detector beat an unconditional
   straddle, **net of cost**, and does it survive the tail (vol-expansion) and the fact that
   calm regimes pay thin premium?

**Success criterion:** ranked by risk-adjusted per-trade edge (gross → net), with the gates
in §3. A positive per-trade result is a **SIGNAL**; it only becomes a **STRATEGY** if it
survives portfolio sizing, correlation, drawdown and capacity (playbook §8).

---

## 3. The Base — what is being tested

**Underlying universe (two evidence tiers):**
- **Tier A — decision-grade anchor:** `NIFTY50`, `BANKNIFTY`. Real 30-day IV via `INDIAVIX`
  (in `market_data.db`, ~2024-03→2026-03); a 2-month real NIFTY option chain
  (`options_data.db`, 2026-04→06) for a final realism spot-check. No earnings gaps.
- **Tier C — modeled extension ("any liquid stock"):** ~20–30 F&O-liquid, Kite-split-adjusted
  large caps (RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, SBIN, ITC, HINDUNILVR, KOTAKBANK,
  BHARTIARTL, LT, AXISBANK, MARUTI, BAJFINANCE, HCLTECH, ASIANPAINT, TITAN, SUNPHARMA,
  ULTRACEMCO, TATAMOTORS, TATASTEEL, M&M, NTPC, POWERGRID, WIPRO, NESTLEIND …). No recorded
  stock option history exists anywhere → **premiums are MODELED (directional, not
  decision-grade)**, and these names carry earnings-gap and SLB-shortability caveats.

**Signal / detector (entry trigger), all causal:**
- `RV20` = trailing 20-day annualized realized vol (log returns × √252).
- **Regime rank** = percentile of `RV20` within its own trailing 252-day window (causal, no
  look-ahead). Low-vol flag = rank ≤ 0.30 (bottom third). Decile of rank is the graded score.
- Supporting features (used as decile refinements / robustness): Bollinger-band-width pct,
  ATR14/close pct, trend flatness |close/SMA20−1|, volume-decline pct.

**Synthetic IV model (Tier C) / real IV (Tier A):**
- Tier A: `IV_t = INDIAVIX_t / 100` (real 30-day IV; term matches a monthly straddle).
- Tier C: `IV_t = max(RV20_t × VRP_mult, 0.08)`, where `VRP_mult` is calibrated from the
  index `INDIAVIX / NIFTY-RV20` ratio (the average volatility risk premium) and
  **sensitivity-swept {1.00, 1.15, 1.30}**. IV *moves with realized vol along the trade path*,
  so short-vega bite on a vol expansion is captured (realistic tail).

**Structures (both):**
- **Naked short straddle** — sell ATM call + ATM put. Pure thesis, uncapped tail.
- **Iron fly** — sell ATM C+P, buy OTM wings at K ± W (W swept: 1×premium, fixed %). Tail-capped, deployable.

**Pricing:** stdlib Black-Scholes (repo `cpr_st_theta.py` pricer), r = 6.5%. Daily
mark-to-market repricing legs with `IV_t` path and decaying `T`.

**Exit policies tested (the "how many days / when to exit" axis):**
- Hold to expiry (T→0).
- Time exit at DTE = {7, 5, 3}.
- Profit target: close at {40, 50, 60}% of premium captured.
- Stop: MTM loss = {150, 200}% of premium, OR underlying move > {1.0, 1.5}× breakeven, OR
  `RV20` re-enters high regime (vol-expansion stop).

**Period:** daily 2010-01-01 → 2026-07-15 (Tier C stocks); Tier A real-IV window
2024-03 → 2026-03 (INDIAVIX span) with modeled-IV backfill 2010→ for comparison.

**Costs (modeled explicitly, playbook §5):** per-leg bid/ask + brokerage + STT + slippage.
Baseline ≈ 0.17% of premium (measured NIFTY, research/60) for Tier A; **wider band swept for
stocks (0.5–1.5% of notional-equivalent)**. Reported gross AND net + cost sensitivity.

**Seven-deadly-sins control (must hold or the result isn't credible):**
| Sin | Control |
|---|---|
| Look-ahead | RV percentile uses only trailing 252d; IV path uses only past RV; entry decided at close t, outcome measured t+1…t+H. |
| Survivorship | Tier C uses *today's* liquid names → biased for cross-sectional claims; headline leans Tier A + reports Tier C as directional only. |
| Overfitting / multi-test | Decile **monotonicity** required, not a single lucky cell; OOS split + param-sensitivity in G3. |
| Cost neglect | Gross AND net always; cost swept; edge expressed in premium units. |
| Regime dependence | Per-year table; 2010-26 spans calm+COVID+2022+2024-25. |
| Correlation / single-factor | Short-vol book is one factor; Tier A vs Tier C correlation reported; not double-counted. |
| Capacity / shortability | Tier C = F&O-shortable names only; margin + capacity noted; Tier A deepest. |

---

## 4. Plan — stage gates + grid

Staged per playbook (kill cheaply; don't build the next gate's machinery until this one passes):

- **G0 Hypothesis** (this doc): vol clusters → low-vol regimes are partly predictable; the
  volatility risk premium makes short straddles positive-EV on average; conditioning on the
  detector should improve the realized-move-vs-premium spread. Counterparty = hedgers/buyers
  of convexity paying the VRP. Decay risk = crowding / vol-regime shifts.
- **G1 Probe** (`run_g1_probe.py`): across all entries × universe × history —
  (a) **persistence table**: P(stays calm over next H) by detector decile → must be monotonic;
  (b) **gross straddle EV** by decile, conditional vs unconditional, with t-stat + block bootstrap.
  *Gate: monotonic persistence AND detector-conditioned gross EV materially > unconditional.*
- **G2 Mechanics** (`run_g2_backtest.py`): full entry/exit backtest, both structures, exit-policy
  grid, **gross vs net**. *Gate: net edge survives on a meaningful sample.*
- **G3 Robustness**: OOS (train ≤2021 / test 2022-26) + walk-forward, per-year stability,
  param sensitivity (regime threshold, H, VRP_mult, cost), adversarial kill. *Gate: stable.*
- **G4 Portfolio**: equity curve, sizing, Tier A/C correlation, MaxDD, capacity, tearsheet.
- **G5 realism check**: modeled-IV vs real INDIAVIX + 2-month real NIFTY chain reconciliation.

**Grid (G2/G3):** regime_threshold {0.20, 0.30, 0.40} × H {15, 20, 25} × VRP_mult {1.0, 1.15, 1.3}
× exit-policy {~10} × structure {naked, iron-fly} × cost {3 levels}. Tier A run with real IV
(VRP_mult axis collapses). G1 first on a thin grid to decide if any of this is worth running.

---

## 5. Status (live log)

**State:** DESIGN → building engine + G1 probe. Nothing launched yet.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-07-22 09:51 | Folder + STATUS created; design locked | Universe both Tier A + C; naked + iron fly; run on VPS |
| 2026-07-22 09:59 | G1 probe done (107s, 103,715 entries) | Monotonic but INVERTED: calm nets −121 bps, loud +440 (modeled artifact); real-IV index calm ≈ 0 (t 0.4); VRP=1.28 |
| 2026-07-22 10:02 | G2-lite exit + VRP probe done (27s) | No exit rescues calm; index all ≈0 (|t|<1), stocks all neg (t −4…−10); rich-IV −25 bps. |
| 2026-07-22 10:05 | RESULTS.md written; STATUS → DONE | **VERDICT: NO EDGE, thesis inverted.** Died G1/G2. |
| 2026-07-22 10:14 | G3 FLIP probe done (sell after spikes, defined-risk) | Real-IV index NOT robust: loud/spike both flip NEGATIVE OOS (train≤2020 +→ test≥2021 −). Stock +247-340bps = modeled-IV artifact. Flip needs real stock IV to decide. |
| 2026-07-22 10:30 | Kicked off REAL stock option IV history download (2016→now, NSE bhavcopy) | See NSE_STOCK_OPTION_BHAV_DAILY_DOWNLOAD_STATUS.md; enables real-premium daily-mark active management. |
| 2026-07-22 11:58 | Calm-persistence by horizon (user reframe: shorter hold, not full month) | P(calm persists): index 73%(3d)→66%(20d), stocks 67%(3d)→53%(20d). Confirms shorter hold = better calm odds; stocks jumpier than index. |
| — | PLAN: G5 real-IV ACTIVE-MANAGEMENT bake-off (pending download) | Daily-mark straddle at REAL option closes; exit grid = profit-target 25/50%, stop 1.5/2x credit, time/21-DTE, 5% move-stop; entry filters IV-rank/ADX/calm; naked vs iron fly; per-year + OOS. |

---

## 6. Crash Recovery — how to resume without Claude

Nothing has run yet. Once the G1 probe is deployed:
- Code lives on VPS at `/home/arun/quantifyd/research/89_short_monthly_straddle/scripts/`.
- Run: `cd /home/arun/quantifyd/research/89_short_monthly_straddle && nohup python3 scripts/run_g1_probe.py > results/g1.log 2>&1 &`
- Check progress: `tail -f results/g1.log`; outputs land in `results/*.csv`.
- DB read-only at `/home/arun/quantifyd/backtest_data/market_data.db`; nothing is written to it.
- Safe to inspect all `results/*`. Do NOT edit the DB.

## 7. Files (output map)

| File | Purpose | Committable? |
|---|---|---|
| `SHORT_MONTHLY_STRADDLE_LOWVOL_DAILY_SWEEP_STATUS.md` | This doc | yes |
| `scripts/engine.py` | Loader + BS pricer + detector + IV model + straddle/iron-fly simulator | yes |
| `scripts/run_g1_probe.py` | G1 persistence table + gross-EV-by-decile probe | yes |
| `scripts/deploy.py` | paramiko SFTP deploy + remote run of the folder | yes |
| `results/*.csv` | Persistence table, per-entry outcomes, decile EV | yes if small |
| `results/RESULTS.md` | Final verdict | yes |

## 8. Findings (during + final)

_None yet — pending G1._
