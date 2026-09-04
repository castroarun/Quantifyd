# Research 150 — Options-Structure Overlay on the High-WR Killed Signals — RESULTS

**VERDICT: NO EDGE. None of the 9 signal × structure cells passes the pre-registered bar
(positive mean return-on-risk at the 10%-premium haircut with t ≥ 2 and ≥2 of 3 years
positive). Connors- and pullback-triggered structures are SIGNIFICANTLY NEGATIVE net
(t = −2.7 to −5.3); KC6-triggered structures have positive point estimates but n ≈ 80 and
t ≈ 0.2-1.0 — indistinguishable from zero.** Arun's payoff-restructuring intuition half-works
and that is the interesting part: the structures DO achieve the high win rates the signals
promised (57-73% WR across cells — the bull put spread wins on flat-or-up exactly as
designed). What they do not do is create expectancy: **the payoff shape was never the
problem — the signal's left tail is.** A short-put structure opened on a buy-weakness signal
concentrates precisely the tail the cash version suffered: every cell that looks mildly
positive in 2024-2025 gives it back multiplied in the 2026 correction (per-year RoR −9% to
−26% in 2026, uniformly). r/129's index-level kill of regime credit spreads extends to
stock-level SIGNAL-TRIGGERED entry — a genuinely different construction (stated up front,
evidence decided), same graveyard: this is the 5th kill of the premium-selling-on-a-timing-
signal family.

> 7,913 structures priced from REAL NSE bhavcopy closes at entry (traded strikes only,
> contracts>0 — the r/89 binding filter), settled at expiry intrinsic vs the underlying's
> real close (no mid-life marks on illiquid strikes needed). **Window 2024-01→2026-09
> (~2.7y) — stock-option bhav is dense only from 2024** (RELIANCE 2019: 140 rows vs 2025:
> 48,300). Short window, includes one up-cycle and one correction. 80 F&O underlyings.

## The table (mean return-on-risk per structure, held to expiry; h = credit haircut for spreads/slippage)

| Signal | Structure | n | WR | RoR gross | RoR h=5% | **RoR h=10%** | t @h10 | 2024 / 2025 / 2026 @h10 | med. leg volume |
|---|---|---|---|---|---|---|---|---|---|
| Connors RSI2 | S1 BPS 0.97/0.90 | 1,414 | 73% | −1.1% | −2.2% | **−3.3%** | **−3.0** | −0.1 / +0.7 / −21.0 | 53 |
| Connors RSI2 | S2 BPS ATM/0.95 | 1,445 | 61% | −0.7% | −3.8% | **−6.3%** | **−3.7** | −0.3 / −5.2 / −26.5 | 108 |
| Connors RSI2 | S3 skewed IC | 1,383 | 62% | +0.6% | −1.6% | **−3.6%** | **−2.7** | −1.2 / +2.6 / −23.4 | 29 |
| KC6 | S1 BPS | 81 | 72% | +4.6% | +3.2% | +1.7% | 0.4 | +6.3 / −0.1 / −11.4 | 108 |
| KC6 | S2 BPS ATM | 85 | 57% | −5.1% | −7.6% | −9.9% | −1.4 | +1.1 / −13.0 / −46.9 | 151 |
| KC6 | S3 skewed IC | 80 | 62% | +5.6% | +3.1% | +0.8% | 0.2 | −3.9 / +10.7 / −5.6 | 102 |
| Pullback-50SMA | S1 BPS | 1,152 | 72% | −1.8% | −2.9% | **−4.0%** | **−3.3** | −0.4 / −3.4 / −12.3 | 45 |
| Pullback-50SMA | S2 BPS ATM | 1,160 | 58% | −5.1% | −7.7% | **−10.1%** | **−5.3** | −5.9 / −13.3 / −14.0 | 78 |
| Pullback-50SMA | S3 skewed IC | 1,113 | 61% | −1.7% | −3.7% | **−5.6%** | **−3.8** | −2.8 / −0.8 / −18.2 | 25 |

## What the experiment actually established

1. **The mechanism works as Arun described — and it isn't enough.** WR jumps to 57-73%
   (S1 cells hit ~72-73%, the promised profile). But mean RoR is negative or zero because
   the rare full-width loss (stock keeps falling through both put strikes after a
   buy-weakness signal) outweighs the many small credits — the same asymmetry as the cash
   version, relocated into option strikes.
2. **The 2026 column is the tell.** Every cell is deeply negative in 2026 (−5 to −47% RoR):
   the overlay is short-put beta on weak stocks. 2024-25's mild positives are bull-tape
   premium, not signal alpha.
3. **Costs decide the marginal cells.** Gross, three cells are slightly positive (Conn-S3
   +0.6, KC6-S1 +4.6, KC6-S3 +5.6); at a 10% premium haircut — mild for stock-option
   monthlies with median leg volume 25-150 contracts — only the two tiny-n KC6 cells stay
   above water, at t ≈ 0.2-0.4. Capacity note: median MINIMUM leg volume 25-108 contracts/
   day means even a small book would be a large fraction of the traded market in the wings.
4. **KC6 cells (n≈80) are the only "maybe" and they are unbettable**: the signal fires ~30×/
   year across 80 names with fillable structures; even if the +2-5% gross RoR were real it
   would take years to reach significance, and the 2026 rows are already negative.

## Recommendations

- **REJECT the overlay family.** Do not paper-trade it: the Connors/pullback cells are
  significantly negative, and the KC6 cells cannot accumulate evidence at a useful rate.
- The constructive reading for Arun: high win rate is a payoff-shape choice, not an edge —
  any structure can buy WR by selling tail. The only thing that creates expectancy is a
  signal whose UNDERLYING distribution is mispriced, and none of these three is (r/146,
  r/149, and now real option prices agree).
- If premium-selling capital wants deployment, the validated venues remain the index/stock
  books already running (C1 winged strangles r/127, the NAS/CSL stack) — not
  signal-triggered stock structures.

## What was NOT tested, and why

- Early exits / profit-taking mid-life (requires trustworthy daily marks on illiquid stock
  options — bhav closes on zero-volume days are stale; hold-to-expiry avoids fabricating
  them; a TP variant would only reduce the already-negative tail capture further in the
  losing cells).
- Call-side structures on SHORT signals (no validated short signal exists — r/81/82/83).
- Pre-2024 history (bhav coverage scraps), IV-conditioning, delta-targeted strikes
  (bhav has no greeks; the %-of-spot scheme is the honest EOD-implementable proxy).
- Weekly stock options (do not exist), index options versions (r/129 already killed).

## Seven sins

Look-ahead: entries at same-day bhav closes on signal close; no mid-life data used.
Survivorship: today's F&O list, 2024+ window (mild). Multiple-testing: 9 pre-declared
cells, no grid — and nothing passed anyway. Costs: traded-strike entries + 5/10% premium
haircuts (stock-option spreads are wide; 10% is not conservative overkill). Regime: 2.7y
window with per-year splits — the 2026 correction is the informative slice. Capacity:
reported (25-150 contracts median min-leg volume — thin). Correlation: moot (no survivor).

## Reproducibility

`research/150_signal_options_overlay/scripts/overlay.py` (single pass ~3 min);
`results/overlay_trades.csv` (7,913 rows), `results/overlay_summary.csv`. Data:
market_data.db `nse_options_bhav` (33.1M rows; 9.56M in-window) snapshot 2026-09-04.
