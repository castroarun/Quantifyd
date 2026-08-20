# research/115 — PCR / OI-change adjustment signals for our short-straddle books

## Verdict: **NO EDGE**

Across 85 trading days of 1-minute full-chain data on both venues, **not one** PCR,
ΔOI, OI-wall or max-pain construction carries information about where a short straddle
is going, once you control for things that are not option information at all. The
signals that looked strongest were the most contaminated: `pcr_oi_all` vs the 60-minute
forward return has a raw rank-IC of **−0.39**, and **3.6% of it survives** the controls.

The single cleanest way to say it: a placebo signal containing **zero option
information** — literally minus the intraday price deviation, `placebo_negspot` — scores
a raw IC of **+0.51**, larger than every real signal in the study, and collapses to
**−0.0006** under the same controls. The raw IC table was measuring the shape of a price
path, not open interest.

In rupees: of **96 pre-registered adjustment arms** tested against hold-to-time-exit,
**zero** beat holding significantly, and **67 of 96 did worse than pulling the same
trigger at a random minute.**

> **Recommendation: drop this line of enquiry.** Do not build a PCR/OI adjustment layer
> on the live books. Details and the one out-of-scope byproduct in §7.

---

## 1. What was actually asked

Our live books are short-volatility: a 09:16 ATM straddle carried to a time exit. So the
question was never "does PCR predict direction" in the abstract. It was:

> Does any PCR / OI-change signal, computable in real time from our recorded chain, warn
> early enough that a short straddle is about to go wrong — and does acting on that
> warning beat doing nothing, net of costs?

Gates, pre-registered in `PCR_OI_ADJUSTMENTS_1MIN_SWEEP_STATUS.md` before any run:
G0 data trustworthy · G1 |IC| ≥ 0.05 sign-stable across venues and months · G2 an
adjustment rule beats hold-to-time-exit net of costs · G3 robust.

**Outcome: G0 PASS (with a caveat) → G1 FAIL → G2 run anyway for a rupee answer, also
FAIL → G3 not reached.**

---

## 2. G0 — is the OI data trustworthy? (PASS, with one caveat)

`backtest_data/options_data.db :: option_chain`, opened read-only. 85 trading days per
venue, 2026-04-20 → 2026-08-20, NIFTY + SENSEX, 1-minute snapshots (verified: stamps are
exactly on the minute; the DB uses a `T` separator, which silently returns zero rows if
you range-query with a space).

| | NIFTY | SENSEX |
|---|---|---|
| Days | 85 | 85 |
| Minutes/day (median) | 373 | 371 |
| Expiries recorded/snapshot | 4 | 4 |
| Strike window | ATM ±20 (step 50) | ATM ±20 (step 100) |
| OI null, near expiry | 0.0% | 0.0% |
| OI **zero** (illiquid strikes) | 8.0% | 29.5% |
| IV null | 24.0% | 13.4% |
| **Minutes where OI actually changed** (ATM±5) | **p50 97.3%** | **p50 36.9%** |
| Median run-length of constant OI | 1 | **exactly 3** |

**The caveat, and it is a real one: SENSEX OI is published on a strict 3-minute cadence.**
The median run of an unchanged OI value is exactly 3 minutes on every one of the 85 days,
and the change rate sits in a tight 35.7–38.1% band (i.e. ≈1/3). NIFTY OI, by contrast, is
genuinely live minute-to-minute (97% of minute-pairs change).

**How this was handled:** every ΔOI signal in the study uses a horizon of ≥5 minutes, so
the 3-minute cadence does not invalidate any of them — but SENSEX carries roughly one
third the independent OI observations NIFTY does, and any *sub-3-minute* SENSEX OI signal
is structurally impossible. Zero-OI strikes were kept (they contribute 0 to the OI sums,
which is the correct treatment) rather than dropped, to avoid a liquidity-selection bias.

Note also that "total PCR" here means PCR over the recorded ATM±20 window, not the
literal full chain — the recorder does not store the far tails.

---

## 3. G1 raw — the result that looked spectacular and was not

Feature build: 60,092 minute rows over **168 day-venue units** (84 NIFTY + 84 SENSEX),
26 pre-registered signals × 3 outcomes × 4 horizons = **312 tests**. The day is the
independent unit: Spearman IC computed *within* each day across minutes, then t-tested
across days — minutes overlap massively, so a pooled t-stat would be a fiction.

44 of 312 tests "passed" the pre-registered G1 bar. The top of the table:

| Signal | Outcome | h | mean IC | t |
|---|---|---|---|---|
| pcr_oi_all | fret | 60 | **−0.4076** | −21.0 |
| pcr_vol_atm5 | fret | 60 | +0.3757 | +17.2 |
| pcr_oi_all | fret | 30 | −0.3188 | −22.1 |
| dist_wall_ce | fret | 60 | +0.3041 | +12.5 |
| wall_squeeze | fret | 60 | +0.2840 | +10.5 |
| mp_dev | fret | 60 | +0.2557 | +10.8 |

An IC of −0.41 with t = −21 on a liquid index does not exist. Two mechanical explanations
had to be eliminated:

1. **Price-anchor artifact.** `dist_wall_ce`, `mp_dev`, `wall_squeeze` and (empirically)
   the PCR levels are all monotone functions of *where spot sits relative to a sticky
   intraday anchor*. Regressing forward returns on a persistent, contemporaneously
   price-linked variable inside one finite path produces a systematically signed sample
   correlation **even for a pure random walk** (the finite-sample mean-reversion /
   Stambaugh bias). Its sign is the same every day — so "sign-stable across venues and
   months", the very thing our G1 gate asked for, is exactly what the *artifact* predicts.
2. **Time-of-day artifact.** Straddle premium decays monotonically through the session, so
   any signal with an intraday time trend correlates with `fdprem` for that reason alone.

---

## 4. G1b/G1c — the controls, and the placebo that beat everything

Controls applied (`g1b_controls.py`, then `g1c_controls2.py`), each signal projected off
the ranks of: **intraday spot deviation · minute-of-day · current premium level ·
trailing 15-min return · trailing 30-min return**, within the same day. Plus a
**shuffled-day** test (the signal path from a different day of the same venue against this
day's outcomes — preserves shape and persistence, destroys any real link) and five
**placebos** carrying no option information whatsoever.

### The money exhibit

| Signal | Outcome | h | raw IC | partial IC | % of raw surviving |
|---|---|---|---|---|---|
| **placebo_negspot** *(zero option info)* | fret | 60 | **+0.5074** | −0.0006 | **0.1%** |
| pcr_oi_all | fret | 60 | −0.3932 | −0.0141 | 3.6% |
| pcr_oi_all | fret | 30 | −0.3126 | −0.0229 | 7.3% |
| pcr_vol_atm5 | fret | 60 | +0.3907 | +0.0314 | 8.0% |
| dist_wall_ce | fret | 60 | +0.3212 | −0.0120 | 3.7% (sign flip) |
| wall_squeeze | fret | 60 | +0.3064 | +0.0022 | 0.7% |
| mp_dev | fret | 60 | +0.2862 | −0.0074 | 2.6% (sign flip) |
| dist_wall_pe | fret | 60 | +0.2488 | −0.0354 | 14.2% (sign flip) |
| pcr_oi_atm10 | fret | 60 | −0.2243 | −0.0010 | 0.4% |

The placebo out-scores every genuine signal on raw IC. Several real signals **flip sign**
once controlled, which is the signature of a variable whose entire apparent relationship
was borrowed from the price path.

**Best partial IC anywhere in the whole PCR / ΔOI / OI-wall / max-pain family:**
`pcr_vol_all` vs 60-min forward return, **+0.0773, t = 3.55** — below the multiple-testing
bar (§6), and its shuffled-day twin scores **−0.0699**, the same magnitude with the
opposite sign. That is shape artifact, not signal.

### The two things that looked like survivors, and why they are not

- **`atm_iv` → forward premium change.** Partial IC −0.39 (t −18) after the *first* round
  of controls. But ATM IV is a proxy for the current premium *level*, and the outcome is
  that premium's own forward increment in rupees — the same finite-sample bias again, plus
  a big rupee premium mechanically decaying by a bigger rupee amount. Adding
  `rank(prem_t)` to the control set and normalising the outcome by entry premium collapses
  it to **−0.063**, against a `placebo_noise` (a random walk!) scoring **+0.049** on the
  same outcome. Dead.
- **The ΔOI family** (`doi_ce_*`, `doi_pe_*`, `doi_net_15`, `d_pcr_oi_*`) sat at partial IC
  ≈ 0.08–0.10 after round one. OI accumulates where price has just been, so a 15-minute
  ΔOI is contaminated by the trailing 15-minute return, making IC(ΔOI, forward return)
  partly plain intraday price reversal — an effect **research/109 already found
  un-tradeable after costs**. Controlling for trailing returns drops the whole family to
  |IC| ≤ 0.03. Dead.

**G1 verdict: FAIL.** Zero PCR/OI constructions alive.

---

## 5. G2 — the rupee answer (run despite the G1 failure)

Because the question was asked in rupees, it is answered in rupees. Book modelled: short
1 ATM straddle at 09:16 on the nearest expiry, exit 15:15. NIFTY lot 75, SENSEX lot 10.

**Costs**, per leg-side per lot: NIFTY 0.5 pt × 75 = ₹37.5 + ₹30 = **₹67.5**; SENSEX
1.0 pt × 10 = ₹10 + ₹30 = **₹40**. Every arm pays exactly 4 leg-sides — exiting early, or
exiting one leg early, adds **no** leg-sides. **This gives the adjustment rules the most
favourable possible treatment: zero incremental cost.** They still lose.

**Baseline (HOLD), net of costs:**

| | n | mean ₹/lot/day | win% | p05 | worst |
|---|---|---|---|---|---|
| NIFTY | 84 | **+282.9** | 65.5% | −5,374 | −18,866 |
| SENSEX | 84 | **+193.7** | 67.9% | −2,384 | −8,311 |
| Both | 168 | **+238.3** | 66.7% | −4,305 | −18,866 |

22 pre-registered triggers × 4 actions (EXIT_ALL · EXIT_LEG · STOP@1.2× · STOP@1.3×)
= **96 pooled arms**, each with a matched **random-trigger twin** (20 deterministic draws
at random minutes) — the random-entry control this repo requires.

| Trigger | Action | ₹/lot vs HOLD | t | random twin | fire% |
|---|---|---|---|---|---|
| d_atm_iv_15>2.0 | STOP12 | +85.7 | 1.26 | +76.1 | 6.0% |
| doi_net_15<−0.10 | STOP13 | +71.2 | 0.55 | +15.6 | 86.9% |
| d_pcr_oi_15<−0.10 | STOP13 | +70.2 | 0.54 | +25.0 | 82.7% |
| d_atm_iv_15>1.0 | EXIT_ALL | +59.0 | 0.59 | **+160.4** | 23.8% |
| pcr_oi_all<0.8 | EXIT_LEG | +55.3 | 0.21 | −375.3 | 79.8% |
| MOVE>0.3% *(pure price ref)* | STOP13 | +45.2 | 0.34 | +33.8 | 82.7% |

- **Arms beating HOLD at all: 27/96. Beating HOLD at t ≥ 3.9 *and* beating their own
  random twin: 0/96.**
- **67 of 96 arms do worse than pulling the same trigger at a random minute.** The best
  NIFTY arm (`doi_net_15>0.05` + STOP13, +₹381.9, t 1.70) has a random twin at +₹288 —
  three quarters of its "improvement" is reproduced by a coin flip. What little the arm
  contributes comes from the *stop mechanic*, not from the signal that armed it.
- **Anti-monotonic in threshold**, the clearest tell of a worthless signal. `pcr_oi_all>X`
  + EXIT_ALL: −₹184.6 (fires 57.7%) → −₹127.8 (25.6%) → −₹113.5 (15.5%). The rule improves
  strictly as it acts less often; the optimum is never acting.

This independently reproduces **research/114**'s finding from a different direction:
holding beats the adjustments.

---

## 6. The seven deadly sins — how each was controlled

| Sin | Control applied here |
|---|---|
| **Look-ahead** | Every signal uses only data stamped ≤ that minute; forward outcomes strictly t→t+h; all lags trailing. The 09:16 entry strike is fixed from the entry minute forward. |
| **Survivorship** | N/A — index options, no cross-sectional universe. Zero-OI strikes deliberately *kept* to avoid liquidity selection. |
| **Overfitting / multiple testing** | **Pre-registered** signal, threshold and action lists before any run. Total hypotheses tried ≈ **824** (312 G1 + 416 G1c + 96 G2 arms). Bonferroni 5% ⇒ per-test α ≈ 6×10⁻⁵ ⇒ **\|t\| ≥ 4.0**. Applied throughout; the best PCR/OI partial IC reaches only t 3.55. Monotonicity checked and found *anti*-monotonic. |
| **Cost neglect** | Costs explicit per leg-side per venue; gross and net both computed. Adjustment arms deliberately given a zero-incremental-cost handicap and still lose, so the negative result is *not* a cost artifact. |
| **Regime dependence** | **NOT controlled — the binding limitation.** One 4-month window (2026-04→08), a single volatility regime. See caveats. |
| **Correlation / single-factor** | Both venues reported separately throughout; NIFTY and SENSEX agree, but they are ~1 bet, not 2 independent confirmations. |
| **Capacity / liquidity** | Not reached (nothing to size). SENSEX 29.5% zero-OI strikes flagged as a liquidity reality of the recorded window. |

Additional controls beyond the standard seven, and the ones that actually decided this
study: **placebo signals** (5, carrying no option information), **shuffled-day** tests, and
**random-trigger twins** on every G2 arm.

---

## 7. The one byproduct — out of scope, flagged not claimed

`skew` (IV of the PE 3 strikes below ATM minus IV of the CE 3 strikes above ATM) is the
only construction that survived every control: partial IC **−0.138 (t −13.9)** vs the
5-minute forward return, −0.089 at 60 min, sign-stable, NIFTY −0.128 / SENSEX −0.149,
shuffled-day twin −0.008, placebo ceiling 0.011. It also survives adding the trailing
5-minute return to the control set (IC −0.1375), killing the short-horizon-reversal
objection.

**But it does not answer this study's question, and it is not a result we are claiming:**

| Outcome | h=5 | h=15 | h=30 | h=60 |
|---|---|---|---|---|
| forward return (`fret`) | −0.138 | −0.119 | −0.096 | −0.089 |
| **\|forward move\| (`fabs`)** | **+0.000** | −0.013 | +0.001 | +0.031 |
| **forward premium change (`fdprem`)** | **+0.005** | +0.005 | +0.010 | +0.016 |

Skew is a **pure direction** signal with **precisely zero** information about whether a
short straddle is about to go wrong — which is the only thing our books need. A
delta-neutral book cannot bank a directional IC without a delta-hedging or directional
overlay, which is a different study with its own cost problem. Further caveats: single
4-month regime; recorded IV is broker-derived from LTP so a stale wing quote mechanically
moves skew; skew is computable on only ~80% of minutes.

Treat it as a **lead for a future directional study, not a finding.**

---

## 8. Honest caveats

- **One regime, four months.** 85 days, 2026-04-20 → 2026-08-20. This is the study's
  weakest dimension by far. A negative result on 4 months is not proof of a permanent
  absence of edge — it is proof that nothing here is strong enough to detect in 4 months,
  which is a different and weaker claim. It is, however, enough to refuse to build on.
- **SENSEX OI is 3-minute data.** All ΔOI horizons used were ≥5 min, but SENSEX carries
  ~⅓ the independent OI observations of NIFTY.
- **"Total" PCR is ATM±20**, not the literal full chain; the recorder does not keep the
  far tails. A far-tail PCR could in principle behave differently — untested.
- **Single entry construction.** One straddle per day, entered 09:16, nearest expiry.
  Adjustment value could differ for other entries or for strangles.
- **ROLL and BUY-WING actions were not tested** — the recorded window supports them, but
  both cost strictly *more* leg-sides than EXIT, and since exits with zero incremental
  cost already fail to beat HOLD on an uninformative trigger, they were not worth the
  compute. This is a scope choice, stated rather than hidden.
- **The IC statistics are honest about n** (day-level units, 168), but 168 units is a
  modest sample for chasing IC ≈ 0.05 effects.

## 9. Next levers

1. **Stop building adjustment logic on PCR/OI.** The highest-EV action here is not to
   spend more compute — it is to not spend it. This closes the question.
2. If revisited, revisit **after ~12 months of recorded chain**, across at least one
   volatility regime change, and only for far-tail PCR (which this window cannot see).
3. The genuinely open lever for these books remains **sizing and per-DTE participation**
   (research/103, 104, 113, 114), not intra-day adjustment.
4. `skew` → a separate directional study, if and only if someone wants a directional
   sleeve. It pays nothing to a delta-neutral book.

---

## Reproducibility stamp

- Data snapshot: `backtest_data/options_data.db` on the VPS as of **2026-08-20**
  (85 trading days from 2026-04-20; the final day is partial — recording was live).
  Opened **read-only** (`file:...?mode=ro`, `uri=True`, `PRAGMA query_only=ON`).
  No database was written by this study.
- Scripts (run in this order, all re-runnable, all self-contained):

  | Script | Produces |
  |---|---|
  | `scripts/g0_data_recon.py` | `g0_day_coverage.csv`, `g0_oi_staleness.csv`, `g0_summary.json` |
  | `scripts/build_features.py` | `features_master.csv` (60,092 minute rows) |
  | `scripts/g1_ic.py` | `g1_ic_table.csv` (raw, uncontrolled) |
  | `scripts/g1b_controls.py` | `g1b_controls.csv` (round-1 controls + placebos) |
  | `scripts/g1c_controls2.py` | `g1c_controls2.csv` (full controls) |
  | `scripts/g1d_skew_probe.py` | `g1d_skew_probe.csv` (targeted probe of the survivor) |
  | `scripts/g2_rules.py` | `g2_rule_bakeoff.csv`, `g2_summary.json` |

- Costs assumed: NIFTY ₹67.5/leg-side/lot (0.5 pt × 75 + ₹30), SENSEX ₹40/leg-side/lot
  (1.0 pt × 10 + ₹30); 4 leg-sides per day per arm.
- Randomness: all seeded deterministically from the date/venue string — re-runs reproduce
  exactly.
- `features_master.csv` is ~46 MB and is **not** committed; regenerate it with
  `build_features.py` (about 4 minutes).
