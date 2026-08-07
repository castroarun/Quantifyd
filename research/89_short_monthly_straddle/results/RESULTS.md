# RESULTS — Short Monthly Straddle in Predicted Calm Regimes (research/89)

## VERDICT: **NO ROBUST TRADEABLE EDGE** (confirmed on REAL option data + adversarial liquidity test)

**Final program verdict (2026-07-22, after building real stock option IV history):** there is no
robust, net-of-realistic-frictions short-straddle edge on Indian stocks or indices today.
- Original thesis (sell *into* calm): **inverted / no edge** (below).
- Flip (sell *after* spikes, defended): index edge was real pre-2021 but **decayed to ≈0** post
  the 2022+ retail options-selling boom; the huge *stock* numbers were a **phantom of illiquid,
  stale option prices** — on genuinely tradeable options (ATM volume ≥ 50) the capped iron fly is
  **NEGATIVE (−82 bps, t −7.9)** and **105% of the apparent profit came from untraded options**.
- What *does* stand: the management best-practices (take 25–50% profit, shorter hold, iron fly for
  tail) genuinely beat hold-to-expiry — but management can't manufacture an edge that isn't there.

See the "REAL-IV ESCALATION" section at the bottom for the G4/G5/G6 real-data work. The original
modeled-IV analysis (still valid and now corroborated) follows.

---

## (Original modeled-IV verdict) NO EDGE — and the original thesis is *inverted*

Selling a ~monthly ATM straddle **into a low-volatility period is the worst time to do it**,
not the best. There is no harvestable net edge on realistic (real-IV) index data, and no exit
rule or "sell-when-IV-is-rich" refinement rescues it. Died at G1/G2 per playbook — kill cheaply.

Study: 2010→2026 daily, NIFTY50 + BANKNIFTY (REAL IV via INDIAVIX) and 26 F&O-liquid large caps
(MODELED IV = realized × 1.15, calibrated from INDIAVIX/NIFTY-RV). 103,715 entry-days.
Holding H = 20 trading days (~1 month). Costs 0.17% of premium turned over. Runs on VPS.

---

## Why — the mechanism (volatility mean reversion)

Vol clusters, but at the 1-month horizon it **mean-reverts**. Conditioning on the low-vol
detector (decile 0 = calmest) is cleanly monotonic — the *wrong way* for selling premium:

| Regime decile | Entry RV | Forward realized RV | Premium | Net bps | Win% | **P(stays calm)** |
|---|---|---|---|---|---|---|
| 0 calmest | 16% | **23% ↑** | 4.2% | **−121** | 48% | **35%** |
| 5 mid | 26% | 26% | 6.7% | +65 | 63% | 79% |
| 9 loudest | 42% | **31% ↓** | 10.7% | **+440** | 80% | 93% |

- **Calm does not stick:** from the calmest regime, forward realized vol *rises* (16→23%); only
  **~35%** of the time does the quiet persist through the month. You collect thin premium, then
  the stock moves more than you were paid → you lose.
- **Answering the user's literal question:** the probability of a calm period sticking to a low
  vol level over ~1 month is **~35%** (calmest decile), rising monotonically with how loud the
  starting regime is. Low starting vol = LOW confidence the calm holds.

## The decision-grade anchor (NIFTY/BANKNIFTY, REAL IV)

The stock "+440 bps selling into loud vol" is a **modeled-IV artifact** (a constant IV=RV×1.15
multiplier mechanically wins when you sell high current RV, because payoff depends on the lower
*future* RV — real IV already prices that in). The real-IV index leg is the honest number:

- **Calm entries (rv_rank ≤ 0.30): net +2.4 bps, t = 0.4 (n=92 independent) → statistically zero.**
- Unconditional monthly NIFTY straddle: **net −20 bps** over 2015–26. The volatility risk
  premium is real (mean INDIAVIX/RV = **1.28**, ~2.6 vol points), but it **did not cover the
  2020 / 2022 tail losses** — short-straddle P&L is negatively skewed (>56% win, negative mean).

## No exit or refinement rescues it (G2)

- **Exit bake-off on calm entries** (expiry / move-stop 1.0× / move-stop 1.5× / 50% target / DTE-5):
  index all ≈ 0 (|t|<1); stocks all significantly negative (naked −63…−73 bps t −7…−10). Iron fly
  *caps the tail* (−15 vs −73 bps) but stays negative. Move-stops whipsaw (win → ~25%) without net gain.
- **Correctly-specified VRP version** (index, sell only when real IV rich vs realized): unconditional
  +7.5 bps (t 0.26), **rich-IV −25 bps**, cheap-IV −12 bps — richness did not help.

## Seven deadly sins

| Sin | Control / note |
|---|---|
| Look-ahead | RV percentile trailing-252d only; IV path uses past RV; outcome measured strictly forward. Clean. |
| Survivorship | Tier C = today's liquid names → stock numbers directional only; **headline leans real-IV index**. |
| Overfitting | Result is monotonic across 10 deciles, not a lucky cell; independent (non-overlapping) t-stats reported. |
| Cost neglect | Gross + net everywhere; 0.17% modeled; result robust to cost (edge ~0 gross too on index). |
| Regime dependence | 2010–26 incl. COVID + 2022; tail losses are the whole story. |
| Correlation | Single short-vol factor; not double counted. |
| Capacity/shortability | Moot — no positive edge to size. |

## Honest caveats (loud)

- **Stock premiums are MODELED** (no recorded stock option IV exists). Their magnitudes are
  directional, not decision-grade. Every stock number above inherits the IV=RV×1.15 assumption.
- The **real-IV evidence (index) is the trustworthy leg** and it says: **no edge, ~zero net,
  calm-timing does not help.** This is the conclusion to act on.
- A short 2-month real NIFTY chain (`options_data.db`) exists for a finer realism check but was
  not needed — the daily + INDIAVIX evidence is already decisive.

## ADDENDUM (2026-07-22) — the FLIP hypothesis: sell AFTER vol spikes, defined-risk

Tested selling premium in the LOUD regime / after a vol spike (spiked-then-cooling), iron fly +
naked, exit grid, per-year, OOS. Decision-grade real-IV index leg:

- **LOUD (rv_rank≥0.70), real-IV index:** naked expiry +8 bps (t 0.19), iron fly +0.5 bps (t 0.03)
  — essentially zero. **OOS FLIPS: train≤2020 +23 bps (t 0.84) → test≥2021 −20 bps (t −1.15).**
- **SPIKE-then-cooling, real-IV index:** +21–56 bps in-sample but n=36 and **dominated by 2020's
  COVID vol collapse (+489 bps that year)**; t 0.8–1.5 (not significant). **OOS FLIPS: train≤2020
  +86 bps (t 1.72) → test≥2021 −25 bps (t −0.84).**
- **Stocks (modeled) print +247…+340 bps, t 15–18 — but this is the SAME modeled-IV artifact**
  (IV=RV×1.15 mechanically wins selling after high RV since payoff uses lower future RV). NOT trustworthy.

**Flip verdict: NOT a robust edge on trustworthy (real-IV) data** — the index short-vol premium was
harvestable pre-2018 and right after the 2020 spike, but compressed/crowded from 2022 and the signal
flips negative out-of-sample. The decisive question — does the flip work on real STOCK IV (where the
VRP may be structurally richer / less crowded than the index)? — **cannot be answered until real stock
option IV history exists.** That is exactly why the NSE-bhavcopy option-history build (below) matters.

## Next levers (if ever revived)

1. Short vol is a *mean-reversion* trade → the interesting (opposite) hypothesis is selling
   premium **after** vol spikes with a defined-risk structure; but on real index IV even that
   was only ~breakeven net over 2015–26. Low priority.
2. Would need **real per-stock option IV history** (years, not 2 months) before any stock
   short-vol claim is decision-grade. Re-open when the live chain collector matures (playbook §11).
3. If the user wants a *long*-vol / breakout angle, that's the natural flip and belongs to the
   existing breakout/momentum books, not here.

**Bottom line for the user:** deploying a monthly straddle into a calm stock is backed by
intuition but not by the data — the calm only holds ~1-in-3 months, and the thin premium loses
to the vol that follows. Don't trade it.

---

## REAL-IV ESCALATION (2026-07-22) — built real stock option history, then killed the edge properly

**New data asset:** downloaded NSE F&O bhavcopy EOD option history 2016→2026 into
`nse_options_bhav` (30.3M rows, 83 symbols) — real premiums, no more modeled IV. IV computed by
BS inversion. Permanent asset for any future options study. (Downloader: `download_nse_bhav_stocks.py`.)

**Management reframe (user, correct):** don't hold a month — take profit / cut on criteria for a
better-probability-of-calm shorter hold. Confirmed two ways:
- Calm PERSISTENCE by horizon: index 73%(3d)→66%(20d), stocks 67%(3d)→53%(20d). Shorter hold =
  higher calm odds; stocks jumpier than index.
- Exit bake-off (real daily marks): take-25/50%-profit BEATS hold-to-expiry everywhere; ~18-day
  median hold; iron fly caps the tail. (Best practice per tastytrade/OptionAlpha/Zerodha Varsity.)

**G4/G5 real-IV daily-marked results (naked & iron fly, exit grid, per-year, OOS):**
- **INDEX (NIFTY/BANKNIFTY):** short-vol WAS strongly profitable through 2021 (TP50 naked
  +315 bps/trade OOS train≤2021, t 6.2) but **decayed to ≈0 in 2022-26** (naked +47 t 1.6; iron
  fly +48 t 2.6 but worst-year negative). Retail options-selling boom compressed the VRP. Not
  deployable today. (Revises the earlier modeled "≈0" — historically real, now gone.)
- **STOCKS (raw):** looked spectacular — iron fly +146 bps OOS≥2022 (t 16), EVERY year positive,
  no decay. **Too clean → adversarial test.**

**G6 adversarial kill (cost sweep + LIQUIDITY filter + concentration):**
- Cost sweep (0.2%→5% of premium) barely dents it — gross is huge, so cost isn't the killer.
- **LIQUIDITY filter is the killer.** Requiring entry ATM options to have actually traded
  (volume ≥ 50 contracts) collapses it: TP50 naked +517→**+67 bps (t 3.2)**; **iron fly +140→
  −82 bps (t −7.9)**; at volume ≥ 1000, both negative. Only ~22% of the 7,400 "trades" had ≥50
  contracts of ATM volume.
- **105% of the apparent iron-fly profit came from UNTRADED (vol<50) options** — stale settlement
  marks you could never transact at. Per-name on the liquid subset: only **9 of 39 names** positive,
  no economic persistence (JINDALSTEL −608, M&M −225 vs BANKBARODA +235 — noise).
- Survivorship (today's F&O list) not even corrected yet — it would only worsen this.

**Seven-sins note:** the whole stock illusion was **cost/liquidity neglect + survivorship**. LESSON
(now binding for this repo): **any options backtest here MUST filter on real traded volume/OI** —
EOD "close" on illiquid contracts is a phantom price.

**Final:** short vol is a mean-reversion/VRP trade that was harvestable historically (pre-2021
index) but is **not a robust, tradeable, net-of-frictions edge today** — on stocks or indices.
Active management is the right *how*, but there is no *edge* to manage on liquid instruments.

**Reusable engine:** `engine.py` (BS + detector + IV model), `run_g4/g5/g6_*` (real-IV daily-mark,
management bake-off, adversarial stress), `run_persistence_horizons.py`. Data: `nse_options_bhav`.
