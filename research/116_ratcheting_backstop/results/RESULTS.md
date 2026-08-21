# research/116 — Static vs Ratcheting Backstop — RESULTS

**Verdict: NO EDGE — do not ratchet the defence. Leave the live rule exactly as it is.**

Stage: **G2 (mechanics, real data, net of cost).** 17 defence variants replayed on the real
1-minute option chain over 236 construction-days (85 recorded sessions, 2026-04-20 → 2026-08-20,
NIFTY + SENSEX, four live constructions). **Not one variant beat STATIC on total P&L, and not one
variant improved the median give-back.** The prior I stated up front — that the breakeven clamp
would be the strongest candidate — is **contradicted by the data**.

---

## 1. The question, and the honest answer

Arun's premise: *"we reach some decent profit levels, now the distance from this point up to that
original 50% backstop is a lot — should we still hold the same defence which is now way extended?"*

**The premise is factually correct.** At the moment of peak open profit the stop is a median
**₹5,185/lot away** (p90 ₹8,294, max ₹13,019). The defence really is far out of the money
exactly when you have the most to lose.

**The feared event does not happen.** Of 236 construction-days, 38 (16%) ever went "deep"
(open profit ≥ 40% of the entry credit). Of those 38, **exactly 2 later came back and touched the
static stop** — 0.85% of all days. Both were COMB_NIFTY expiry-day (DTE0) sessions:

| Day | Construction | Credit | Peak open | Final |
|---|---|---|---|---|
| 2026-06-02 | COMB_NIFTY DTE0 | 122.05 | +₹3,484/lot | −₹4,426/lot |
| 2026-06-30 | COMB_NIFTY DTE0 | 111.90 | +₹3,429/lot | −₹1,420/lot |

A further 6 deep days round-tripped to ≤ ₹0 open without reaching the stop. That is the entire
population of the event the ratchet exists to catch, in four months of live-shaped data.

**And there is almost nothing to trail.** The peak open profit of a decaying straddle arrives at
the **90th percentile of the window (median)** and lands in the **last 10% of the window on 50% of
days**. Theta accrues monotonically; on half of all days the peak *is* the exit. Median give-back
peak→close is **₹289/lot** — roughly one round-trip's costs. There is no pot of gold being handed
back on the typical day.

---

## 2. P&L and give-back, side by side (pooled, net ₹ per lot, n = 236 construction-days)

`gb_*` = give-back = peak open profit (gross) minus realised (gross), on the 215 days that showed a
peak worth at least one round-trip. `resc`/`cut` = days better/worse than STATIC by at least one
round-trip cost (₹250 NIFTY, ₹200 SENSEX). `exit%` = share of days the defence fired before the
window closed.

| variant | total | mean | median | win% | worst | **gb_p50** | **gb_p90** | gb_max | resc | cut | uplift vs STATIC | exit% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NO_DEFENCE | **141,596** | 600 | 775 | 74 | **−16,527** | 299 | 4,014 | 18,138 | 13 | 14 | **+4,913** | 0 |
| **STATIC (live)** | **136,683** | 579 | 628 | 71 | −6,667 | **289** | 3,401 | 7,644 | — | — | **0** | 12 |
| BE_CLAMP_50 | 134,234 | 569 | 628 | 71 | −6,667 | 289 | 3,749 | 7,644 | 0 | 2 | −2,449 | 13 |
| BE_CLAMP_60 | 132,300 | 561 | 616 | 70 | −6,667 | 289 | 3,749 | 7,644 | 1 | 4 | −4,383 | 14 |
| BE_CLAMP_70 | 126,211 | 535 | 566 | 69 | −6,667 | 299 | 3,464 | 7,644 | 3 | 6 | −10,472 | 14 |
| RATCHET_K1.3 | 85,521 | 362 | 490 | 67 | −6,667 | 428 | 3,030 | 7,644 | 21 | 29 | −51,162 | 29 |
| RATCHET_K1.5 | 78,882 | 334 | 510 | 67 | −6,667 | 373 | 3,399 | 7,644 | 11 | 24 | −57,801 | 24 |
| RATCHET_K1.75 | 112,401 | 476 | 590 | 68 | −6,667 | 324 | 3,417 | 7,644 | 7 | 12 | −24,282 | 20 |
| RATCHET_K2.0 | 132,648 | 562 | 620 | 71 | −6,667 | 302 | 3,401 | 7,644 | 5 | 6 | −4,035 | 17 |
| RATCHET_K2.5 | 135,334 | 573 | 640 | 71 | −6,667 | 289 | 3,417 | 7,644 | 1 | 1 | −1,349 | 14 |
| GIVEBACK_30 | 79,327 | 336 | 535 | 74 | −6,667 | 381 | 3,030 | 7,644 | 13 | 36 | −57,356 | 33 |
| GIVEBACK_50 | 78,625 | 333 | 422 | 73 | −6,667 | 404 | 3,030 | 7,644 | 9 | 31 | −58,058 | 29 |
| RS_GB_1000 | 55,163 | 234 | 294 | 64 | −5,040 | 1,022 | **1,590** | 5,532 | 26 | 68 | **−81,520** | 51 |
| RS_GB_2000 | 110,696 | 469 | 566 | 69 | −6,667 | 351 | 3,030 | 7,644 | 8 | 18 | −25,987 | 22 |
| RS_GB_3000 | 129,728 | 550 | 566 | 71 | −6,667 | 299 | 3,459 | 7,644 | 4 | 6 | −6,955 | 17 |
| TIME_RATCHET_MID | 111,539 | 473 | 566 | 69 | −6,667 | 356 | 3,369 | 7,644 | 9 | 19 | −25,144 | 22 |
| HYBRID_BE60_GB50 | 78,625 | 333 | 422 | 73 | −6,667 | 404 | 3,030 | 7,644 | 9 | 31 | −58,058 | 29 |

### The two things this table says

1. **Every ratchet loses money.** Best non-static variant (RATCHET_K2.5) is −₹1,349; the shapes
   that actually engage cost −₹51k to −₹82k, i.e. **up to 60% of the book's entire profit.**
2. **No ratchet improves the give-back that Arun is worried about.** The median give-back gets
   **worse**, not better, as the rule tightens: 289 (static) → 302 → 324 → 373 → 428. This is not
   noise, it is mechanical — **a trailing rule can only fire after a retrace, so it manufactures the
   very give-back it is meant to prevent**, and on a decaying straddle it fires on ordinary
   theta-noise wobbles rather than on the rare crack.

The p90 give-back tail is flat across every variant (3,030–3,749) except **RS_GB_1000**, which
genuinely halves it (3,401 → 1,590) — and charges **₹81,520**, sixty percent of the book's profit,
for the privilege. That is the honest cost of "safety" here.

---

## 3. Monotonicity — a real effect is monotone, a fitted one peaks

The k-ladder is clean from k = 1.5 upward and points in one direction: **looser is better, and the
limit of "better" is not ratcheting at all.**

| k | total | give-back p50 | early exits |
|---|---|---|---|
| 1.30 | 85,521 | 428 | 68 |
| 1.50 | 78,882 | 373 | 57 |
| 1.75 | 112,401 | 324 | 47 |
| 2.00 | 132,648 | 302 | 40 |
| 2.50 | 135,334 | 289 | 33 |
| ∞ (STATIC) | **136,683** | **289** | 28 |

The single inversion (k1.3 above k1.5) is two SENSEX-DTE0 days, not a structure. There is **no
interior optimum** — the gradient runs monotonically to the boundary, which is the live rule.

---

## 4. Per-construction and per-DTE stability (net ₹/lot, uplift vs STATIC)

| construction | n | STATIC total | BE_CLAMP_60 | RATCHET_K1.5 | GIVEBACK_50 | RS_GB_1000 |
|---|---|---|---|---|---|---|
| COMB_NIFTY (NAS_COMB20, 09:16–15:20) | 68 | 61,030 | −185 | −12,363 | −17,657 | −32,789 |
| COMB_SENSEX (CSL30F_SENSEX) | 84 | 31,142 | −1,749 | −27,271 | −21,651 | −37,745 |
| TIMEB_NIFTY (windows) | 50 | 18,721 | 0 | 0 | −2,686 | −2,961 |
| TIMEB_SENSEX (windows) | 34 | 25,790 | −2,449 | −18,167 | −16,064 | −8,025 |

Per venue × DTE (9 slices, `results/ratchet_by_dte.csv`): the ratchets are negative or exactly zero
in 8 of 9. The only slices with a positive ratchet uplift are **SENSEX DTE2 (n=17)** and
**SENSEX DTE4 (n=15)** — the two thinnest, most-parked cells, at 1–3 differing days each. That is
what noise looks like; it is not a finding.

The one directionally-consistent slice is **NIFTY DTE0**, where BE_CLAMP_70 is +₹1,729 on 2 rescues
vs 1 cut-short. Two events. It is the same two days listed in §1. **Not actionable.**

---

## 5. My stated prior, contradicted

I predicted the **breakeven clamp** would be the strongest candidate: asymmetric in the right way
(can only turn a winner into a scratch, never cut a winner short while it is working), the shape the
naked-survivor ST trail already uses, and unlikely to fire on ordinary decay noise.

**The mechanics of the prior held; the economics did not.** The clamp is indeed nearly inert —
it changes 0–6 days out of 236 — but across every trigger level it produced **more cut-shorts than
rescues** (0/2, 1/4, 3/6) and a **negative** uplift (−2,449 / −4,383 / −10,472), and it did not move
the give-back distribution at all (gb_p50 identical to static; gb_p90 slightly *worse*).

The reason is §1: the clamp waits for a state — deep profit that then reverses all the way to the
stop — that occurs twice in four months. When it does fire, it usually fires on the days that were
about to recover. "Asymmetric in the right way" is only worth something if the bad tail it truncates
actually exists at meaningful frequency. Here it does not.

---

## 6. Byproduct worth its own decision (NOT a conclusion of this study)

This study's one varied axis was the *ratchet*. But the NO_DEFENCE control incidentally prices the
existing defence, and the picture is uneven enough to record:

| slice | stops fired | STATIC net | HOLD net | cost of the stop |
|---|---|---|---|---|
| COMB_NIFTY DTE0 | 6 | −13,400 | −34,412 | **stop SAVED +21,012** |
| COMB_SENSEX DTE1 | 6 | −18,524 | −29,834 | **stop SAVED +11,310** |
| COMB_SENSEX DTE2/3 | 5 | −18,618 | −21,779 | stop saved +3,161 |
| **COMB_SENSEX DTE0 (SENSEX expiry)** | **4** | **−18,005** | **+10,054** | **stop COST −28,059** |
| TIMEB_SENSEX DTE0 | 3 | −9,917 | −6,851 | stop cost −3,066 |
| TIMEB_NIFTY DTE0 | 1 | −2,704 | +1,476 | stop cost −4,180 |

(Remaining slices — COMB_NIFTY DTE1/DTE2, COMB_SENSEX DTE4 — are ±₹2,700 or less.)

Pooled, the defence costs ₹4,913 and buys a worst day of −₹6,667 instead of −₹16,527. **That is a
sane insurance premium and it should stay.** But it is paid almost entirely on **SENSEX expiry
Thursday**, where 4 firings of the 50% disaster backstop cost ₹28,059 while improving the worst day
by only ₹370 (−5,040 vs −5,410). This independently reproduces research/114 ("on SENSEX expiry every
stop tested lost to holding") and the 2026-08-19 config note ("every pct-stop fires on Thu expiry
gamma noise") on a *different* construction and a *different* stop level.

**Do not act on this here.** It is a level question, not a ratchet question; n = 4 firings; and the
backstop is unbounded-loss insurance on a naked short. It belongs in its own study with its own
STATUS-MD, alongside the 11-SEP SX-Thu review already on the books.

---

## 7. The seven deadly sins — how each is controlled

| Sin | Control in this study |
|---|---|
| **Look-ahead** | Every bar tests the stop level **carried in from prior bars**, then updates the ratchet/peak with that bar's data. A bar can never trigger a stop it just set. Peak-giveback likewise arms and tests off the prior running peak. Entry strike is chosen from `underlying_spot` at the entry minute only. |
| **Survivorship** | None available to exploit: every recorded session in the window is replayed; the only exclusion is 2026-08-21 (today, partial series, market still open). Skips are logged in `results/run.log` (1 day: TIMEB_NIFTY 2026-04-20, recorder started mid-session). |
| **Overfitting / multiple testing** | 17 variants × 4 constructions × 9 venue-DTE slices ≈ 600 cells inspected. **No haircut is needed because nothing beat the null**: among the 16 defended variants STATIC is top-1 pooled, top-1 in COMB_SENSEX / TIMEB_SENSEX, tied-top in TIMEB_NIFTY, and beaten in COMB_NIFTY only by ≤₹2,010 (3%) on 1–2 differing days. The two positive venue-DTE slices are n = 15–17 with 1–3 differing days. Monotonicity was required in advance and the ladder runs to the boundary. |
| **Cost neglect** | Net of 0.5 pt/leg-side NIFTY, 1.0 SENSEX, plus ₹30/leg-side/lot = ₹250/lot (NIFTY) and ₹200/lot (SENSEX) round trip on 2 legs. Ratchets add no trades unless they fire, so the cost delta is purely exit-frequency — and the tighter rules exit on up to 51% of days vs 12% for STATIC, which is part of why they lose. Materiality thresholds for rescue/cut-short are set at exactly one round trip. |
| **Regime dependence** | **This is the binding weakness.** One regime: 2026-04-20 → 2026-08-20, four months, benign for short vol. Reported per venue, per DTE, per construction; the conclusion is directionally identical in all four constructions, which is the best cross-check available in a 4-month window. |
| **Correlation / single-factor** | All four constructions are the same short-gamma factor; their agreement is corroboration of consistency, **not** four independent tests. Effective sample is closer to 85 sessions than 236 cells. |
| **Capacity / liquidity** | ATM front-expiry options only — the most liquid strikes on both venues. No capacity constraint at 2–8 lots. Fills modelled at observed LTP plus the venue slippage above. |

---

## 8. Caveats a reader must carry

- **One regime, 85 sessions.** Four months, no VIX shock, no gap-down cluster. A ratchet's whole
  case rests on tail days; four months is thin evidence about tails. This study can say
  "ratcheting does not pay for itself in a normal regime"; it cannot say "a ratchet would not have
  helped in March 2020".
- **1-minute granularity understates intrabar breaches.** All variants see the same series, so the
  comparison is fair — and the bias runs **in the ratchets' favour**: a ratcheted stop sits closer
  to the current premium, so finer data would make it fire *more* often, not less. The ratchets lose
  anyway. (Per the binding data rule: sub-minute ticks do not exist for these contracts —
  `option_chain` is a 1-minute full-chain recorder and `option_ohlc` is empty.)
- **Single entry per day** per construction. Live books may re-enter or cascade; that is a separate
  axis, deliberately frozen here.
- **Give-back is measured gross** (peak gross minus realised gross); costs are identical across
  variants and cancel.
- **Per-lot figures.** NIFTY lot 65, SENSEX lot 20. At live size multiply: COMB NIFTY 2 lots,
  TB-SENSEX Thursday 8 lots, NAS_COMB20 DTE3 5 lots.
- The `sl: "none"` cells (CSL30F_SENSEX DTE0, CSL_TIMEB_SENSEX DTE0) are modelled with the **50%
  disaster backstop (1.5 × credit)** as their STATIC defence — that is exactly the level Arun's
  question is about.

---

## 9. Recommendation

**Leave the defence alone. Do not deploy any ratchet, clamp, or trailing give-back rule to any
combined-SL book.**

Concretely, for the position that prompted the question — TimeB NIFTY, credit 175.13, 8 lots,
20% combined-SL at 210.2, ₹18,214 max loss:

- The ₹18,214 does **not** grow as the trade wins; it is a fixed ceiling. What grows is the
  *unrealised* profit sitting above it, and this study shows that profit is, on the typical day,
  still growing at the moment the window closes.
- Moving that stop down to protect the open profit would, on this evidence, have cost the book
  between 2% (loosest clamp) and 60% (₹1,000 giveback) of its total P&L while leaving the median
  give-back unchanged or worse.
- The right lever for "I am uncomfortable with the distance to my stop" is **size**, not stop
  placement — and on the one cell where the defence measurably misfires (SENSEX expiry Thursday)
  it is a **level** question for a separate, properly-powered study, not a ratchet.

**Re-open this question if and only if** (a) we accumulate a genuinely stressed regime in the
recorder, or (b) the deep-then-reverse event rate rises materially above the 2-in-236 observed here.
A dated re-check belongs in the Ops & Review Center alongside the 11-SEP SX-Thu item.

---

## 10. Files

| File | Purpose |
|---|---|
| `scripts/run_ratchet_sweep.py` | the bake-off — 17 defence variants × 4 frozen constructions |
| `scripts/analyse_ratchet.py` | pooled / per-construction / per-DTE P&L + give-back tables |
| `scripts/diagnose_giveback.py` | the anatomy: where the peak sits, how far it retraces, deep-then-stop counts |
| `results/ratchet_detail.csv` | 4,012 rows — every variant × construction × day (gitignored per repo convention; regenerate with `run_ratchet_sweep.py`, ~20 min) |
| `results/ratchet_summary.csv` | pooled + per-construction summary |
| `results/ratchet_by_dte.csv` | venue × DTE stability |
| `results/giveback_anatomy.csv` | 236 construction-days of peak timing / retrace / stop-distance |
| `results/run.log` | run log incl. skipped days |
