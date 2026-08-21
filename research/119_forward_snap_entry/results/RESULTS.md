# research/119 — Forward-ATM vs Spot-ATM entry for the live CSL books

**Verdict: NO EDGE — leave the live CSL entry rule alone.**

The forward snap does exactly what it claims mechanically (it removes the entry skew,
decisively and significantly), and that buys the CSL books **nothing**: net P&L is
neutral-to-slightly-worse in 12 of 14 cells, the tail is worse, the effect does not
scale with the size of the spot-forward gap, and the directional exposure it was
supposed to remove **is not measurably there in the first place**. Two of the three
success criteria fail. Under the stated rule — *anything less than all three, leave the
live rule alone* — the recommendation is to **make no change to `csl_paper_exec.py`**.

- Period: **2026-04-20 → 2026-08-21**, 86 recorded days, real 1-min `option_chain`.
- Constructions: the live COMB (09:16→15:20, per-DTE combined-SL) and TimeB (per-DTE
  windows) books, NIFTY + SENSEX, config frozen from `backtest_data/csl_paper_config.json`.
- 241 paired book-days (each replayed twice — arm A and arm B — on the identical minute series).
- Costs: 0.5 pt/leg-side NIFTY, 1.0 pt SENSEX, plus ₹30/leg-side/lot. **All figures are
  net Rs per LOT** (NIFTY lot 65, SENSEX lot 20). Live sizes are 2–8 lots per book.

---

## 1. The two rules do differ — often, and predictably

This is *not* a non-issue. Arm B picks a different strike on **31% of NIFTY entries and
48% of SENSEX entries**.

| venue | n | mean \|gap\| | median | p90 | max | strike changed |
|---|---|---|---|---|---|---|
| NIFTY | 121 | 16.4 | 10.3 | 42.0 | 69.2 | 37/121 = **31%** |
| SENSEX | 120 | 54.4 | 40.0 | 126.7 | 266.4 | 58/120 = **48%** |

The gap is a **cost-of-carry basis**: the forward sits *above* spot (NIFTY mean +9.8,
SENSEX mean +43.3) and it scales with time to expiry exactly as theory says it should.
That monotone DTE ladder is the strongest single confirmation that the quantity being
measured is real and not a data artefact:

| venue | DTE0 | DTE1 | DTE2 | DTE3 | DTE4 |
|---|---|---|---|---|---|
| NIFTY mean gap | **−2.2** | +7.9 | +21.3 | +14.7 | — |
| NIFTY changed % | **3%** | 42% | 41% | 41% | — |
| SENSEX mean gap | +16.3 | +30.4 | +32.6 | +67.2 | **+108.1** |
| SENSEX changed % | 24% | 50% | 41% | 72% | **76%** |

On expiry day the basis collapses to zero and the two rules agree almost always
(NIFTY DTE0: 3% changed). The divergence is a far-DTE phenomenon.

**The 2026-08-21 anchor case reproduces exactly** in the replay — spot 24,261.15,
spot-ATM 24250 (CE 118.40 / PE 61.05, skew **57.35**), forward 24,307.35 → forward-ATM
24300 (CE 88.95 / PE 81.40, skew **7.55**). Same 50-point disagreement Arun saw live.

## 2. The mechanism is confirmed: B removes the skew

| book | arm | n | mean \|skew\| | median \|skew\| | p90 | mean signed skew |
|---|---|---|---|---|---|---|
| COMB_NIFTY | A | 69 | 21.4 | 18.9 | 45.3 | +10.1 |
| COMB_NIFTY | **B** | 69 | **12.8** | 12.6 | 22.6 | **−0.1** |
| COMB_SENSEX | A | 86 | 65.9 | 55.0 | 129.5 | +45.8 |
| COMB_SENSEX | **B** | 86 | **27.8** | 31.1 | 45.9 | **0.0** |
| TIMEB_NIFTY | A | 52 | 19.4 | 15.1 | 42.2 | +6.0 |
| TIMEB_NIFTY | **B** | 52 | **12.3** | 11.2 | 22.0 | **−0.6** |
| TIMEB_SENSEX | A | 34 | 48.4 | 38.0 | 97.5 | +24.9 |
| TIMEB_SENSEX | **B** | 34 | **22.5** | 22.3 | 38.9 | **−9.4** |

Paired: NIFTY mean Δ\|skew\| **−7.98 (t −5.74)**, SENSEX **−34.63 (t −7.46)**. The mean
*signed* skew goes to ~0 under B in every book. **Criterion 2 passes decisively.**

## 3. …and it earns nothing for it

Net Rs per lot, same days both arms:

| cell | arm | n | total | mean | median | win% | worst | p05 |
|---|---|---|---|---|---|---|---|---|
| ALL | A | 241 | 141,427 | **587** | 620 | 71% | −6,817 | −3,734 |
| ALL | B | 241 | 125,818 | **522** | 569 | 69% | **−8,755** | −3,690 |
| | **B−A** | 241 | **−15,609** | **−65** | 0 | — | — | **t −1.79** |
| NIFTY | A | 121 | 86,488 | 715 | 562 | 73% | −6,269 | −2,177 |
| NIFTY | B | 121 | 82,152 | 679 | 605 | 69% | −8,755 | −2,177 |
| | **B−A** | 121 | −4,336 | −36 | 0 | — | — | t −1.00 |
| SENSEX | A | 120 | 54,939 | 458 | 723 | 69% | −6,817 | −4,127 |
| SENSEX | B | 120 | 43,666 | 364 | 558 | 68% | −7,690 | −4,188 |
| | **B−A** | 120 | −11,273 | −94 | 0 | — | — | t −1.49 |

Per book: COMB_NIFTY −16/lot (t −0.28), COMB_SENSEX −122 (t −1.48), TIMEB_NIFTY −62
(t −1.85), TIMEB_SENSEX −23 (t −0.29). Across the **14 book × DTE cells, 11 are
negative for B, 3 positive**, and the largest B-favourable t-stat anywhere in the study
is **+0.49** (COMB_NIFTY DTE1).

**Restricted to the days where B actually picked a different strike** (the only days the
rule can matter):

| venue | n | A mean | B mean | diff | t | B better on |
|---|---|---|---|---|---|---|
| NIFTY | 37 | 442 | 323 | **−119** | −1.02 | 41% |
| SENSEX | 58 | 111 | −84 | **−195** | −1.50 | 41% |

Nothing here is significant. The honest statement is *"B does not beat A, and the point
estimate says it costs about ₹65/lot/day"* — not *"B loses"*. But the success criterion
required B to **beat or match**, and a rule change that is at best a coin-flip is not a
reason to invalidate a validated book (see §7).

Stop-out frequency barely moves (A: 11% SL_DWELL, B: 12%), and **B's worst day is worse
in both venues** (NIFTY −6,269 → −8,755; SENSEX −6,817 → −7,690). No tail improvement.

## 4. The directional exposure B was supposed to remove is not measurable

Regressing each day's net P&L (Rs/lot) on the index move over the hold:

| cell | arm | n | slope (Rs per %) | r | t |
|---|---|---|---|---|---|
| ALL | A | 241 | +439 | 0.076 | 1.18 |
| ALL | B | 241 | +770 | 0.126 | 1.97 |
| NIFTY | A | 121 | +564 | 0.103 | 1.13 |
| NIFTY | B | 121 | +991 | 0.170 | 1.89 |
| SENSEX | A | 120 | +361 | 0.061 | 0.66 |
| SENSEX | B | 120 | +640 | 0.102 | 1.12 |

**Arm A shows no significant directional tilt (t 1.13 / 0.66), and arm B's slope is
larger, not smaller, in every cut.** There is nothing to remove.

What actually drives these books is **absolute** move — they are short gamma, and that
exposure is an order of magnitude stronger than any delta term:

| venue | arm | slope on \|move\| | r | t |
|---|---|---|---|---|
| NIFTY | A | −3,667 | −0.488 | −6.10 |
| NIFTY | B | −4,270 | −0.535 | **−6.90** |
| SENSEX | A | −6,150 | −0.714 | −11.08 |
| SENSEX | B | −6,654 | −0.733 | **−11.70** |

B is *more* short-gamma, which is exactly right economically — the balanced straddle is
the maximally short-gamma one — and it is the wrong direction for a risk-reduction argument.

The one cut that mildly supports the delta story is the up-day/down-day asymmetry, and
only on NIFTY: A earns 633 on up days vs 875 on down days (asymmetry −242, i.e. a net
short-delta tilt, consistent with selling a strike below the forward); B halves it to
−129. On SENSEX the asymmetry does not shrink, it flips sign (−107 → +109). Neither is
significant. **Criterion 3 fails.**

## 5. No monotonicity — the small negative is noise

If the forward snap mattered, (B−A) would grow with the size of the spot-forward gap.
It does not:

| venue | Q1 \|gap\| | Q2 | Q3 | Q4 | regression |
|---|---|---|---|---|---|
| NIFTY (B−A) | +0 | −50 | −50 | −44 | slope −3.3 Rs/pt, r −0.128, **t −1.41** |
| SENSEX (B−A) | +0 | −220 | −104 | −54 | slope −0.3 Rs/pt, r −0.022, **t −0.24** |
| ALL (B−A) | +1 | −71 | −84 | −107 | slope −0.7 Rs/pt, r −0.055, **t −0.85** |

(Q1 is by construction ~0: at a tiny gap the strike does not change and the arms are
identical.) On SENSEX the difference **shrinks** as the gap grows — the opposite of a
dose-response. The skew reduction over the same quartiles is strongly monotone
(SENSEX Δ\|skew\| −0.0 / −14.7 / −28.7 / −95.1), so the mechanism scales cleanly while
the P&L consequence does not. That is the signature of a real mechanic with no economic
consequence for this holding period.

## 6. Cost of switching: liquidity is fine, but B collects less premium

| venue | arm | n | mean bid-ask (% of mid) | median | p90 |
|---|---|---|---|---|---|
| NIFTY | A | 116 | 0.220% | 0.227% | 0.285% |
| NIFTY | B | 116 | 0.217% | 0.225% | 0.280% |
| SENSEX | A | 116 | 0.262% | 0.247% | 0.378% |
| SENSEX | B | 116 | 0.253% | 0.242% | 0.348% |

Paired Δ(bid-ask): NIFTY −0.0034 pp (t −1.13), SENSEX −0.0092 pp (t −1.87). **B is not
the less liquid strike** — if anything it is marginally tighter, because the forward-ATM
strike is the one with two near-the-money legs rather than one ITM-ish and one OTM leg.
So the liquidity objection to switching does not hold.

The real, *statistically solid* cost of switching is premium:

| venue | paired Δ(credit) B−A | median | t |
|---|---|---|---|
| NIFTY | **−0.73 pts** | +0.00 | **−4.38** |
| SENSEX | **−2.35 pts** | +0.00 | **−4.98** |

Straddle premium is minimised at the forward, so the balanced strike always collects
less. That is the trade the forward snap makes: give up measured, certain credit
(t ≈ −4.4 / −5.0) to remove a delta tilt that is not measurably costing anything
(t 1.13 / 0.66). Over a same-day hold the extra credit at the spot strike appears to be
worth at least as much as the delta risk it compensates for — which is why (B−A) is
mildly negative rather than zero.

## 7. The blocker, stated plainly

**research/111 validated COMB and TimeB with spot-ATM entries.** Every per-DTE window,
every combined-SL level in `csl_paper_config.json` was fitted and frozen against straddles
entered at `round(spot/step)*step`. Changing the entry rule changes the credit (−0.73 /
−2.35 pts, significant), which changes the SL threshold `(1+sl)·credit`, which changes
when the stop fires — so it does not merely re-centre the strike, it moves the whole
calibrated exit stack off its basis. That is only worth doing if forward-ATM is
neutral-or-better. It is not neutral-or-better on P&L, and it is not better on risk.
**Do not change it.**

## 8. Robustness

- **Exit-model sensitivity.** The live daemon polls every ~5 s with a 2-poll dwell; a
  1-min replay makes that dwell coarse. Re-run with the dwell removed (exit on the first
  breach minute): NIFTY A 663 / B 643 (diff −20, t −0.64), SENSEX A 456 / B 378
  (diff −78, t −1.30). **Same conclusion under both exit models.**
- **In-flight day.** 2026-08-21 was still trading when the replay ran (its exit is a
  mid-day mark). Excluding it: ALL n=238, A 592 / B 526, diff −65, t −1.79 — unchanged
  to the rupee.
- **Per-month stability of (B−A):** NIFTY +227 / −171 / −36 / −47 / +85 (Apr→Aug);
  SENSEX −101 / −353 / −41 / +65 / −60. Sign flips month to month in both venues. There
  is no persistent effect either way — which is itself the finding.

## 9. The seven deadly sins

| Sin | How it is controlled here |
|---|---|
| **Look-ahead** | The forward is computed **only** from the CE/PE quotes at the entry minute at the spot-ATM strike — exactly the two numbers the live executor already fetches before it decides. No future bar is touched. Exits use each minute's own quotes. |
| **Survivorship** | Not applicable — a fixed 2-instrument construction on 2 indices, no universe selection. Every recorded day is used; skipped days are logged (`results/run.log`). |
| **Overfitting / multiple testing** | **Nothing was fitted.** Arm B is the *existing* production rule from `nas_atm_executor.py`, ported verbatim (including the no-quote fallback); everything else is frozen from `csl_paper_config.json`. Cuts examined: ~21 P&L cuts (3 pooled + 4 per-book + 14 book×DTE), 14 directional regressions, 4 abs-move regressions, 4 up/down splits, 15 monotonicity buckets + 3 regressions, 6 liquidity/credit cuts, 10 per-month cells, 2 robustness re-runs — call it **~80 looks**. **Zero reached \|t\| ≥ 2 in B's favour**; the best was +0.49. With ~80 looks you would expect ~4 at \|t\|≥2 by chance alone, so the *absence* is the meaningful result — it is very hard to make this rule look good even by cherry-picking. |
| **Cost neglect** | Gross and net both carried in `fs_detail.csv`; net uses 0.5/1.0 pt slippage + ₹30/leg-side/lot, applied identically to both arms. The credit difference (§6) is **measured from the chain**, not assumed. Bid-ask at the chosen strike is measured, not modelled. |
| **Regime dependence** | **This is the study's main weakness.** One 4-month window (2026-04→08), a calm-ish period: n=17–18 per book×DTE cell. Per-month (B−A) flips sign in both venues. A large trending regime could in principle make a delta tilt matter more than it did here — but note the delta term was not significant in *either* arm, and the gamma term dominates by 5–10×, so a regime that hurts A hurts B slightly *more*. |
| **Correlation / single factor** | The two arms are the same trade on adjacent strikes — by construction ~0.99 correlated, which is *why* the paired test (same day, same book, same minute series) is the right statistic and is what is reported throughout. |
| **Capacity / liquidity** | Measured directly (§6): bid-ask at the B strike is equal-or-tighter, so the switch has no capacity penalty. Sizes here are 2–8 lots, far inside the top-of-book of front-week NIFTY/SENSEX ATM. |

## 10. Other caveats

- **DTE keying.** The replay keys the config off the live `wd2dte` weekday map, exactly
  as `csl_paper_exec.py` does. On 7 of 172 venue-days the actual front expiry disagreed
  (recorder gaps 2026-04-21..24, and a SENSEX expiry that fell on Wed 2026-05-27); those
  days are logged. Both arms are affected identically, so the A/B comparison is unaffected —
  but it is a live-daemon fragility worth its own look someday.
- **Entry-minute slippage.** Entry is the first recorded snapshot ≥ the configured time;
  on some days the recorder's first tick is 09:20 rather than 09:16. Identical for both arms.
- **Single entry per day**, no re-entry, no `mgmt` arms (trail/shift), no per-leg stop —
  faithful to the COMB/TimeB constructions as configured, but it means this study says
  nothing about the mgmt A/B books.
- The absolute P&L levels here are a *replay* of the frozen config on recorded quotes, not
  the live books' realised P&L. Only the **A-vs-B difference** should be read as a result.

## 11. Recommendation

1. **Leave `csl_paper_exec.py`'s entry rule as-is** (`K = round(spot/step)*step`). No code
   change, no re-freeze of `csl_paper_config.json`, no re-validation of research/111.
2. **Do not "fix" the 916-suite either.** The suite's forward snap is not wrong — it is
   a different, defensible choice that costs a little premium to buy a cleaner delta.
   This study finds no reason to move *either* family toward the other, and moving the
   suite would invalidate *its* basis for the same reason (§7).
3. **Stop treating the strike disagreement as a bug.** The two families will keep landing
   on different strikes on ~1/3 of NIFTY and ~1/2 of SENSEX days, systematically more at
   high DTE, and that is now a documented, expected, harmless property — worth a line in
   the ops notes so the next morning's divergence does not restart this investigation.
4. **If anything is revisited later**, revisit it as a *sizing/gamma* question, not an
   entry-strike question: the regressions say these books live or die on \|move\|
   (r −0.49 to −0.73, t up to −11.7), not on direction (r 0.06–0.17, n.s.).

---

**Files:** `scripts/run_forward_snap.py` (A/B replay), `scripts/analyze_forward_snap.py`
(all six measurements), `results/fs_detail.csv` (964 rows — every replayed day, both arms,
both exit models), `results/analysis.txt` (full output), `results/run.log`.
