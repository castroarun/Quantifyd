# RESULTS — GTAA ETF Rotation (research/63)

**Verdict: the Upstox "Strategy 1" top-1 momentum rotation is a WEAK design in all
testable history (Calmar 0.30–0.44, not the slide's claimed 0.93 — which we cannot
reproduce because Kite only serves these ETFs from 2015). We BEAT it decisively, and
the winner is almost embarrassingly simple: hold all three uncorrelated ETFs
EQUAL-WEIGHT, rebalanced monthly — Calmar ≈ 1.73, CAGR ≈ 19.5%, MaxDD only −11.3%,
turnover ≈ 0, and completely cost-insensitive. The momentum *selection* in the slide
actively destroys value here; the edge is diversification, not rotation. Label:
STRATEGY (candidate) for a simple low-DD multi-asset mandate — with a loud
period-dependence caveat.**

Period: **2016-02 → 2026-06 (125 months, ~10.4y)** net of 20 bps/side. Snapshot:
`market_data.db` @ 2026-06-12. Universe: NIFTYBEES, GOLDBEES, MON100 (+LIQUIDBEES cash).

---

## 1. Did we reproduce the slide? No — and here's why

| Slide claim | Our measurement (2016-26) |
|---|---|
| top-1, ROC12 rank, MA6 "bullish", monthly | same rules, faithfully coded |
| 15.45% CAGR / 16.62% MaxDD / **Calmar 0.93** | top-1 raw: 10.5% / −34.4% / **0.30**; top-1 cash-gated: 10.9% / −24.9% / **0.44** |

The slide's 0.93 is almost certainly a **longer/older window** (GOLDBEES from 2007, MON100
from ~2011) that included gold's 2007–2012 super-run — Kite's historical API only serves
these ETFs from **2015-01**, so we physically cannot replay their period. In the *modern*
decade the concentrated top-1 design is fragile: it piles into the hottest asset and eats
the reversal (−34% DD).

## 2. Finalists (net 20 bps/side, 2016-26)

| Strategy | CAGR | MaxDD | **Calmar** | Sharpe | Turnover/yr |
|---|---|---|---|---|---|
| **EqualWeight 3-asset (WINNER)** | **19.5%** | **−11.3%** | **1.73** | 1.55 | **0.05** |
| EqualWeight + trend filter (defensive) | 11.8% | −8.5% | 1.40 | 1.38 | 2.16 |
| Momentum top-2 (gated) — best *tactical* | 16.4% | −12.5% | 1.31 | 1.34 | 2.38 |
| Slide top-1 (gated) | 10.9% | −24.9% | 0.44 | 0.62 | 3.31 |
| Slide top-1 (raw) | 10.5% | −34.4% | 0.30 | 0.57 | 2.83 |
| Benchmark: NIFTYBEES buy&hold | 10.0% | −28.8% | 0.35 | — | 0 |

**The winner doubles the return of the benchmark at ~40% of its drawdown.** Every
momentum-selection variant we tested underperformed plain equal-weight.

## 3. Why diversification beats rotation here

Monthly-return correlations (2015-26): **Nifty/Gold −0.08, Nifty/Nasdaq +0.25,
Gold/Nasdaq +0.04** — three genuinely uncorrelated sleeves. Per-asset INR CAGR:
NIFTYBEES 10.1%, GOLDBEES 14.6%, MON100 **24.6%** (Nasdaq bull + INR depreciation).
Equal-weighting harvests the diversification + rebalancing premium and never concentrates
into a single drawdown; top-1 throws the diversification away.

## 4. Robustness

- **Cost-insensitive (the standout):** winner Calmar = 1.73 at **0/10/20/40 bps** — it
  barely trades (5% turnover/yr = small annual rebalance trims). Momentum variants decay
  with cost (top-2: 1.53→1.13 as cost 0→40bps).
- **Per-year (winner vs Nifty):** beats Nifty in **7/11 years**; worst year **−3.5%
  (2022)** vs benchmark's deeper holes. 2020 +32%, 2024 +28.6%, 2025 +28.7%, 2026 +22%
  excess. Underperformed (but stayed positive/flat) in the equity-only bull years 2017/2021.
- **Parameter-insensitive:** the winner is the SIMPLEST config — ROC/MA settings are moot
  once top-N ≥ universe size, so there is **no knife-edge parameter to overfit** (108
  configs tried; the winner is the one with the *fewest* degrees of freedom — the opposite
  of a data-mined peak).
- **Look-ahead:** signals use closes ≤ t; returns realized t→t+1. No same-bar leak.

## 5. Frame vs our other long-only books (the "both" ask)

| Book | CAGR | MaxDD | Calmar | Turnover/Tax | Complexity |
|---|---|---|---|---|---|
| **GTAA EqualWeight 3-ETF (this)** | ~19.5% | **−11.3%** | **1.73** | tiny / light | trivial (3 ETFs) |
| research/62 Momentum-30 ETF sub-select | ~33% (net 29) | −17% | ~1.7 | higher | medium |
| research/41 regime-gated midcap RS | ~28–30 net | −18 to −20 | 1.66–1.75 | high / STCG churn | high |

**Same Calmar tier (~1.7) as our best equity books — but at the lowest drawdown,
lowest turnover, lowest tax, and lowest complexity of any of them.** The equity books
win on *absolute return*; this wins on *return-per-unit-of-pain and simplicity*. For a
beginner / core-allocation / "set-and-forget" mandate, this is the better product. For a
return-maximiser who can stomach −18%+ DD and tax churn, 62/41 dominate on CAGR.

## 6. Honest caveats

1. **Period dependence (biggest).** 2016-26 was a golden decade for this trio —
   MON100's +24.6% INR CAGR (Nasdaq mega-bull + ~3%/yr INR depreciation) carries much of
   the absolute return and **will not repeat at that rate**. The *low-DD / diversification
   / Calmar* property is the robust takeaway; treat ~19% CAGR as an upper bound, not a
   forecast. A forward-realistic equal-weight is plausibly ~10–13% CAGR with −15 to −20%
   DD in a less benign regime.
2. **No all-3 simultaneous crash in sample.** 2008 isn't testable (no data); COVID-2020
   was V-shaped. True tail risk (a global risk-off that hits equity *and* gold *and* tech)
   is under-represented → real MaxDD could exceed −11%.
3. **MON100 capacity/regulatory:** overseas-ETF flows hit RBI/SEBI caps in 2022 (creation
   halted, premium to NAV). At size, the Nasdaq sleeve has tracking/capacity risk;
   substitute (e.g. direct US ETF via LRS) may be needed.
4. **LIQUIDBEES price-return ≈ 0%** (daily-dividend ETF) → cash-leg yield understated
   ~6%/yr. The WINNER uses no cash leg, so unaffected; but the "defensive trend-filter"
   variant's CAGR is understated by ~the cash-fraction × 6%.
5. **Single 11-year window, no true OOS / walk-forward.** Mitigated only by the winner's
   zero parameter-fit.
6. **Tax:** equal-weight rebalancing trims ~5%/yr → light STCG, not zero. Held in ETFs
   (delivery), LTCG-eligible after 1y.

## 7. Next levers

1. **Lengthen history** with non-Kite gold/Nasdaq proxies (GOLDBEES NAV from AMFI 2007;
   ^NDX in INR via FX) to test 2008/2011/2013 — does Calmar survive a real bear?
2. **Add a market-level trend overlay** (go partial-cash when ALL three < MA) with a
   *proper* cash yield (T-bill 6%) — the defensive variant already cut DD to −8.5%; with
   correct cash return it may dominate on Calmar.
3. **Risk-parity weights** (inverse-vol) instead of equal-weight — MON100's 24% vol vs
   Nifty's 16% means equal-weight overweights risk; inverse-vol may lift Calmar further.
4. **Wider menu probe** once SILVERBEES/midcap have ≥7y (silver added 2022) — but ext5
   already showed more assets ≠ better (Calmar 0.83) without the gold/Nasdaq diversification.
5. **Paper-forward** the winner on the VPS as a monthly cron (it's trivial to run live).

---

*Reproducibility: `scripts/{download_etfs,gtaa_engine,run_gtaa_sweep,finalists}.py`;
snapshot 2026-06-12; cost 20 bps/side; VPS `venv/bin/python`. Verdict: **STRATEGY
(candidate)** — simple low-DD multi-asset core; period-dependence caveat binding.*
