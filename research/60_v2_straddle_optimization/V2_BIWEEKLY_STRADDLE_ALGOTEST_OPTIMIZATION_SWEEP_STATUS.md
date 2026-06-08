# V2 Positional Bi-Weekly Short Straddle — algotest.in Multi-Year Optimization (2019→now)

STATUS: SETUP — baseline reproduction pending (algotest run by user)

## 1. The Ask

**What you asked:** "for V2, I have algotest.in — we can test variances and optimize for a
better/wider period than what we have backtested."

**What we're actually testing:** Our own recorded-chain V2 (research/57/58) is only ~6 weeks
(2026-04-20→now), one regime → a **SIGNAL, not validated**. Take the locked V2 recipe to
algotest.in over the **longest available history (~2019→now)**, (a) confirm it survives OOS
across every regime, then (b) optimize its levers — ranked by **consistency (net-positive
every year) + risk-adjusted (Calmar = CAGR/MaxDD)**, monotonic response preferred over a
single peak.

## 2. The Base — locked V2 recipe (baseline to reproduce first)

- **Underlying:** NIFTY. **Structure:** short ATM straddle (SELL ATM CE + SELL ATM PE).
- **Expiry:** 2nd-nearest weekly (~8–12 DTE at entry) — "bi-weekly".
- **Entry:** 09:20, ATM at entry spot (round to 50). Size: 1 lot for the sweep (scale later).
- **Intraday crash-stop:** exit immediately if |spot − entry_spot| ≥ **2.0%** → re-enter next 09:20 (don't chase the crash).
- **EOD check (15:20):** exit if |spot − entry_spot| ≥ **1.5%** (move-stop) OR profit ≥ **40%** of credit (PT) OR **DTE ≤ 1** (roll). Else carry overnight.
- **Re-entry:** on an EOD exit, re-enter immediately at 15:20 (stay short → capture overnight theta).
- **Overnight wings:** while carrying past 15:20, BUY ±500pt wings (≈2% OTM, K±500) at 15:20, SELL next 09:20. Caps overnight gap. Tracked separately.
- **Costs:** model ₹ per leg (algotest brokerage+slippage). **Net-of-cost or it didn't happen.**

Reference engine (our recorded-chain implementation, already live as paper):
`research/57_positional_straddle_biweekly/scripts/biweekly_paper.py` → `paper_straddle.db`.

## 3. Success criterion + gates

- **Rank by:** net-positive in **every calendar year** AND best **Calmar**. Tie-break: lower MaxDD, lower worst-day.
- **Gates:** baseline must clear OOS (positive most years, no single-regime dependence) BEFORE testing variants. A variant only "wins" if it's monotonic-ish across the grid (not a lone peak) and holds per-year.
- **Guard the sins:** look-ahead (none — entry/exit on observable spot), cost neglect (always net), regime dependence (per-year table is the whole point), overfitting (one lever at a time, monotonic preferred), capacity (1 lot for now).

## 4. Plan — Step 0 then one-lever-at-a-time

**Step 0 (DO FIRST):** reproduce the locked baseline over 2019→now. Output: per-year net,
MaxDD, Calmar, win%, worst-day. Decision gate: is it positive across regimes?

| # | Lever | Grid | Baseline |
|---|---|---|---|
| 1 | EOD move-stop | 1.0 / 1.5 / 2.0 % | 1.5% |
| 2 | Profit target | 30 / 40 / 50 % of credit | 40% |
| 3 | Intraday crash-stop | 1.5 / 2.0 / 2.5 % | 2.0% |
| 4 | Overnight wings | none / ±300 / ±500 / ±700 | ±500 |
| 5 | Expiry | nearest weekly / 2nd-weekly / monthly | 2nd-weekly |
| 6 | Adjust vs hold | hold-to-stop / re-center to ATM on skew | hold |
| 7 | Entry time | 09:20 / 09:30 / 09:45 | 09:20 |

Test order: 0 → 1 → 2 → 4 → 5 → 3 → 6 → 7 (highest-leverage / most-uncertain first).
One lever at a time; hold the rest at baseline.

## 5. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-06-08 ~11:40 | Folder + STATUS created; period=2019→now, objective=consistency+Calmar | algotest runs by user |
| | Step 0 baseline | _pending — awaiting algotest capability confirmation + first run_ |

## 6. Results (fill as algotest runs come in)

### Step 0 — baseline per-year
| Year | Net ₹ (1 lot) | MaxDD | Win% | Worst-day | Notes |
|---|---|---|---|---|---|
| 2019 | | | | | |
| 2020 | | | | | |
| 2021 | | | | | |
| 2022 | | | | | |
| 2023 | | | | | |
| 2024 | | | | | |
| 2025 | | | | | |
| 2026 YTD | | | | | |
| **All** | | | | | |

### Lever sweeps (one table per lever as we go)
_to be added_

## 7. algotest capability notes (confirm before Step 0)

The recipe has parts algotest may/may not express directly. Confirm what your algotest plan supports:
- [ ] Positional / overnight carry (multi-day hold), not intraday-only
- [ ] Underlying-move SL (|spot−entry| %), as opposed to premium % SL
- [ ] Separate intraday crash-stop AND an EOD-only evaluated move-stop (two different triggers)
- [ ] Immediate re-entry on exit (stay short)
- [ ] 2nd-nearest weekly expiry selection
- [ ] Extra long wing legs (±500pt) as a separate hedge

Where algotest can't express a part exactly, we test the closest proxy and note the gap here.

## 8. Crash recovery / how to resume without Claude

This is a USER-DRIVEN backtest (algotest.in, external). To resume: open this file, see which
lever rows in §6 are filled vs blank. Next action = the first blank in the §4 test order. The
locked baseline recipe is §2. Our own recorded-chain reference numbers (6-week SIGNAL) live in
`research/58/results/v2_positional.json` and the live paper engine `research/57/.../paper_straddle.db`.
Once a config wins, wire it into `biweekly_paper.py` (engine) + the `/app/straddles` V2 card.

---

## ALGOTEST-ADAPTED RECIPE (2026-06-08) — supersedes §2 overnight-wing + two-tier-stop details

User confirmed algotest capabilities: (1) positional carry YES · (2) underlying-move SL YES ·
(3) CANNOT separate an intraday 2% crash-stop from the EOD move-stop · (4) re-entry YES ·
(5) 2nd-nearest weekly YES · (6) wings YES but **bought at entry together, NOT overnight-only**.

**Adapted structure tested on algotest = short IRON FLY, positional, bi-weekly:**
- SELL ATM CE + SELL ATM PE, **2nd-nearest weekly**, entry **09:20**.
- BUY +500 CE wing + BUY −500 PE wing **at entry**, held to exit (defined-risk throughout).
- **Single continuous underlying-move SL** (fires any time, intraday or EOD) — baseline **1.5%**, swept as lever #1.
- **Profit target: 40%** of net credit.
- **Roll** at DTE ≤ 1.
- **Re-enter** immediately on any exit (stay short / positional carry overnight).
- Costs on (brokerage + slippage per leg).

**Gaps vs our recorded-chain engine (note when comparing):**
- Wings held continuously (iron fly) vs our engine's overnight-only wings → more wing theta cost,
  but full-time tail protection. This is a *different, arguably more robust* structure.
- Single continuous SL vs our two-tier (intraday 2.0% + EOD-only 1.5%) → fewer overnight gaps held,
  more intraday stop-outs. Tested as the SL-level lever.

**Revised lever grid (one at a time, hold rest at baseline):**

| # | Lever | Grid | Baseline |
|---|---|---|---|
| 0a | Baseline iron fly (above) | — | run first, per-year |
| 0b | Same but **no wings** (naked straddle) | — | quantifies the wing's cost/benefit |
| 1 | Continuous underlying SL | 1.0 / 1.5 / 2.0 / 2.5 % | 1.5% |
| 2 | Profit target | 30 / 40 / 50 % of net credit | 40% |
| 3 | Wing width | none / ±300 / ±500 / ±700 | ±500 |
| 4 | Expiry | nearest weekly / 2nd-weekly / monthly | 2nd-weekly |
| 5 | Entry time | 09:20 / 09:30 / 09:45 | 09:20 |
| 6 | Adjust vs hold | hold-to-stop / re-center to ATM on skew | hold |

Test order: 0a → 0b → 1 → 2 → 3 → 4 → 5 → 6.

---

## ALGOTEST CONFIG MAPPING (2026-06-08) — how to enter the recipe in the AlgoTest backtest builder

AlgoTest positional model = "Positional · Weekly Expiry basis · Entry N trading-days-before-expiry ·
Exit M trading-days-before-expiry". This maps cleanly:
- Exit "1 trading day before expiry" = our **roll at DTE≤1**.
- Entry "N trading days before expiry" = entry DTE. **8 TD ≈ 2nd-nearest weekly (~12 calendar DTE)**.

**0b — naked straddle (configure first, 2 legs):**
- Instrument: NIFTY, Underlying from = **Cash** (SL reads spot).
- Strategy Type **Positional**; Positional expire on **Weekly Expiry**.
- Entry Time **09:20**, Exit Time **15:15**; Entry **8 TD before expiry**; Exit **1 TD before expiry**.
- Leg1 SELL CALL ATM 1 lot; Leg2 SELL PUT ATM 1 lot.
- Per-leg SL/Target **OFF**. Re-entry on SL **ON**, Re-entry on Tgt **ON**.
- Strategy-level **Overall SL = Underlying movement 1.5%** (NOT premium %); **Overall Target = 40% of premium received** (combined).

**0a — iron fly (duplicate 0b, add wings):**
- Leg3 BUY CALL **OTM 500 pts (ATM+10 strikes)** 1 lot; Leg4 BUY PUT **OTM 500 pts (ATM−10 strikes)** 1 lot.

**Verify in UI:** (1) Entry=8TD lands on ~12-DTE 2nd weekly (use leg "Next Weekly 2" if offered);
(2) SL is **Underlying %** at overall level, per-leg SL OFF; (3) Target is **combined 40%** of premium, not per-leg.

**Known AlgoTest gaps vs our engine** (already noted above): single continuous SL (no separate
intraday 2% crash-stop); wings held entry→exit (iron fly) not overnight-only.

---

## STEP 0a RESULT (2026-06-08) — front-weekly ATM iron fly (±250 wings), 10 lots, 2019→2026

**Config actually tested (drifted from spec, both noted):** ATM short straddle + **±250** long wings
(user chose tighter than the ±500 plan) · **front weekly**, entry **4 TD before expiry** (AlgoTest
max; not the 2nd-weekly 12-DTE carry) · exit 1 TD before / underlying-SL / target-40% · re-enter ·
positional · costs on. 273 trades. Source: AlgoTest export (Trades CSV + PDF).

| Year | Trades | Net ₹ | Win% | Worst trade |
|---|---|---|---|---|
| 2019 | 31 | +61,555 | 55% | −27,495 |
| 2020 (COVID) | 39 | +117,618 | 59% | −59,865 |
| 2021 | 38 | +110,792 | 63% | −22,458 |
| 2022 (bear) | 40 | +284,732 | 65% | −45,630 |
| 2023 (low-vol) | 37 | +24,278 | 46% | −17,062 |
| 2024 | 37 | +133,640 | 65% | −17,550 |
| 2025 | 38 | +122,492 | 68% | −13,065 |
| 2026 YTD | 13 | +83,720 | 62% | −8,158 |
| **All** | **273** | **+938,828** | **60%** | **−59,865** |

**Verdict: PASSES the gate.** Net-positive every year incl. COVID + 2022 bear + dead-vol 2023.
Max DD ₹96k (trade-close) / ₹1.27L (PDF MTM). Calmar ≈ 1.0–1.34. ₹129k/yr on 10 lots.
Wings cap the tail (worst −₹60k vs −₹2-3L naked). Weak spot: 2023 low-vol (+₹24k, 46% win).
Reconciles to PDF total ₹949,723 (rounding). PDF win% 55.68 (leg/charge basis) vs trade-level 60%.

**Next:** 0b (no wings) to value the wings → then sweep wing width / SL / target / VIX filter.

---

## STEP 0b RESULT + 0a-vs-0b MARGIN-EQUALIZED (2026-06-08)

**0b = naked ATM straddle (NO wings), front-weekly, 10 lots, 2019→2026, 271 trades.**
CSV P/L is net of taxes&charges but EXCLUDES brokerage + slippage (both 0 in AlgoTest).

0b net-of-tax, slip 0: **+₹1,890,720** (vs fly 0a +₹938,828). Win 55%. MaxDD ₹318,792.
Worst trade −₹129,578 (2024-08-02 gap). 0b NEGATIVE 2026-YTD (−₹116,610). 2021 win% only 39%.

### Cost stress — slippage hits the fly 2× harder (8 fills/trade vs 4)
| slip/fill | fly 0a (10lot) | naked 0b (10lot) |
|---|---|---|
| 0 | +895k | +1,869k |
| 0.25pt | +540k | +1,693k |
| 0.5pt | +185k | +1,517k |
| 1.0pt | **−524k** | +1,164k |

### Margin-equalized (fly ~₹1.5L vs naked ~₹11L per 10 lots → ~7.3×; CONFIRM w/ broker)
At equal capital run ~7.3× more fly lots:
| slip/fill | fly ×7.3 | naked 10lot |
|---|---|---|
| 0 | +6.56M | +1.87M |
| 0.5pt | +1.36M | +1.52M (≈ dead heat) |
| 1.0pt | −3.85M | +1.16M |

- DD at equal capital: **fly ₹703k > naked ₹319k** → fly's low DD was just small sizing.
- Return-on-margin: slip0 fly **82%/yr** vs naked 23%/yr; slip0.5 fly 17% vs naked 19%.
- TRUE catastrophe: fly CAPPED ~₹1.2M (73lot) | naked **UNBOUNDED** (~−₹15.6L on 10% gap = account-ender).
- Consistency FAILS for both @0.5pt slip: naked neg 2021+2026; fly neg 2019+2023.

### VERDICT (interim)
Decision hinges on TWO measured numbers, not guesses:
1. **Real slippage** — pull from the live NAS book (trades ATM NIFTY daily). Crossover ≈ 0.5pt/fill.
2. **Real margins** — confirm fly-vs-naked SPAN in AlgoTest (drives the 7.3× equal-capital table).
Lean: positional overnight carry + UNBOUNDED naked tail = disqualifying → stay defined-risk (fly)
IF real slip ≤ ~0.4pt; else test WIDER wings (±500/±700 = cheaper insurance) before going naked.
**Next levers:** wing width sweep (none/±250/±500/±700), then SL/target, then VIX filter for 2023-type low-vol.

---

## REAL-INPUT VERDICT (2026-06-08) — Zerodha margins + 0.4% slippage

Margins (1 lot, qty 65, from Zerodha SPAN): naked straddle **Rs2,17,412** ; iron fly ±500 **Rs95,802**
-> ratio **2.27x** (NOT the 7.3x earlier estimated). Slippage 0.4% = % of premium (~1-2 pts/fill on ATM).

### Equal capital (~Rs21.7L), net after slippage (premium turnover ESTIMATED; confirm via AlgoTest Re-calculate)
| slippage | fly x2.27 | naked 10lot |
|---|---|---|
| 0.1% (realistic ATM) | 1.74M | 1.76M (dead heat) |
| 0.2% | 1.34M | 1.63M |
| 0.4% (assumed) | 0.55M | 1.38M |
| DD equal-cap | 218k | 319k |
| catastrophe | CAPPED ~738k | UNBOUNDED (~-15.6L/10% gap) |

Return-on-margin @0.4%: fly 3.5%/yr vs naked 8.7%/yr.

### Reads
- Fly capital edge is only 2.27x -> can't out-lever naked.
- **Slippage is the whole decision; crossover ~0.1-0.15%.** 0.4% is pessimistic for liquid ATM NIFTY
  (real bid-ask ~0.05-0.15%). Below crossover fly ties naked on return + wins on DD/tail.
- Naked tail UNBOUNDED = disqualifying for overnight carry (one 2020-style gap = account-ender).
- CAVEAT favoring fly: ±500 margin pairs with a ±500 fly = LESS wing drag than the ±250 0a (Rs938k)
  -> real ±500 fly nets MORE than shown. Wing-width sweep will quantify.

### VERDICT: lean DEFINED-RISK FLY if real slippage <= ~0.15%; naked only wins if slip truly 0.4%
### AND you accept an unbounded overnight tail (not advised for real money).
### TODO to lock: (1) real slippage from live NAS ATM fills; (2) AlgoTest Re-calculate @0.4% exact 0a+0b.
### NEXT: Step 1 wing-width sweep — run fly at ±500 and ±700 (have ±250=0a), find cheapest tail cap.

---

## WING-WIDTH SWEEP RESULT (2026-06-08) — ±500 WINS

All net of taxes + Rs20/order + measured slippage. 10 lots, 2019→2026.
| width / slip | Net P&L | MaxDD | Calmar | RoM/yr | worst trade | +ve yrs |
|---|---|---|---|---|---|---|
| ±250 @0.5% | −1.3L | 2.6L | neg | neg | −63k | 2/8 |
| **±500 @0.25%** | **+8.0L** | **1.85L** | **0.59** | 11.5% | −53k | 6/8 |
| ±700 @0.25% | +8.6L | 2.35L | 0.50 | 12.3% | −74k | 5/8 |

**LOCKED BASE: ±500 wings.** ±700 earns +₹64k more raw but worse on every risk metric
(deeper DD, worse Calmar, −74k vs −53k tail, one more losing year). ±250 dies on slippage.
±500 = Goldilocks: near-max return, tightest DD, best Calmar, smallest tail, 6/8 yrs green.

**Final structure:** short ATM straddle + long ±500 wings, positional (front/2nd weekly, 4TD
entry / 1TD roll), underlying-SL 1.5% + target 40% + re-enter, realistic slip 0.25%.
~+₹8L/7yr/10lots, Calmar 0.59, ~11.5%/yr on Rs9.58L margin, defined-risk capped tail.

**Slippage settled empirically:** ATM NIFTY half-spread median 0.169% (3.47M recorder quotes),
mean 0.336% — 0.25% is the prudent full-period blend. The earlier 0.5% was pessimistic (~p85).

**Soft spots = dead-vol years:** 2023 (−91k), 2021 (+667 marginal), 2026-YTD (−5k). Root cause:
thin ATM premium vs wing cost. NEXT LEVER = VIX filter (enter only VIX ≥ 12/13/14), should lift
those years. Figure: research/60 wing_sweep.png. Then: SL/target sweep, entry-time sweep.

---

## VIX FILTER RESULT + WING-% CONFOUND (2026-06-08)

### VIX filter (assessed on ±700 @0.25%, entry VIX from CSV; ±500 effect directionally identical)
Avg P&L/trade by entry VIX: <12 +1,933 | **12-14 −4,749 (THE BLEED)** | 14-16 +7,930 | 16-20 +2,744 | 20+ +9,713.
The loser is the 12-14 BAND, not the lowest VIX. VIX floor cuts it.
| filter | trades | total | 2021 | 2023 | 2026 |
| base | 270 | +8.9L | −19k | −1k | −51k |
| **VIX>=12** | 228 | +8.1L | +8k | +25k | +10k → **8/8 yrs POSITIVE** |
| VIX>=13 | 203 | +10.8L | +9k | +97k | −46k → Calmar 0.83, DD 179k |
| VIX>=14 | 168 | +10.9L | +14k | +87k | −2k |
12/13/14 = robust PLATEAU (not a peak). **Add VIX>=12 floor** (consistency) or >=13 (max Calmar).
Figure: research/60/results/vix_filter.png.

### WING-% CONFOUND (user caught this — important, unresolved)
±500 is FIXED points but NIFTY ran 10,700→26,000, so ±500 drifted from **~4.4% (2019) to ~2.0% (2026)**.
The whole ±250/500/700 sweep is confounded by index level — different structures in different years.
"±500 wins" really = "~2% wings in the recent high-index years that drove most P&L."
**NEXT (supersedes fixed-point sweep): re-run wings as % of spot — ±2.0% / ±2.5% / ±3.0% @0.25% slip + VIX>=12.**
Even better if AlgoTest supports: premium- or delta-based wing (auto-normalizes index AND vol regime).
This becomes the genuinely locked, regime-consistent base. ±2% ≈ today's ±500 (forward-relevant).

---

## WING-% SWEEP RESOLVED — 2.0% of ATM LOCKED (2026-06-08)

User ran %-of-ATM wings on AlgoTest: 2.0% / 2.5% / 3.0%, @0.25% slip + Rs20/order + taxes,
10 lots (qty 650), 272 trades each, 2019-02 → 2026-05. Resolves the index-drift confound.
Net results (Claude computed year-wise from the master-row trade lists):

| Wing %ATM | Net total | Calmar | MaxDD | RoM/yr | Neg years | Worst |
|---|---|---|---|---|---|---|
| **2.0%** (=±500 today) | +6.24L raw / **+7.64L ex-COVID** | 0.51 / **0.70** | -1.67L / -1.50L | 8.9 / 10.9% | 2020*,23,26 / 23,26 | -1.39L* |
| 2.5% | +4.84L | 0.29 | -2.28L | 6.9% | 2019,20,23,26 | -85k |
| 3.0% | +6.38L raw / +5.96L ex-COVID | 0.34 / 0.31 | -2.59L | 9.1 / 8.5% | 2020,21,23,26 | -81k |
| (prior fixed ±500) | +8.0L | 0.59 | -1.85L | 11.5% | 6/8 yrs+ | -53k |

CAVEATS: (*) March-2020 COVID circuit week is an ARTIFACT in all 3 — AlgoTest couldn't place the
short straddle at gap strikes, leaving stray wing-only legs (2%: -1.39L single PE; 3%: +42k strangle;
2.5%: skipped). "ex-COVID" cols strip 2020-03-13/03-20 — trust those. Also the 2.5% run's 2025-26
came out identical to the 2% run (cross-transcription suspected, ~50 trades); 2.5% 2019-24 is clean
and already worst, so ranking unaffected.

VERDICT:
- %-normalization CONFIRMS ±500. 2.0%-of-ATM (= ±500 at today's NIFTY) is best on Calmar and lands
  +7.64L ex-COVID — within noise of the confounded fixed-±500 (+8.0L). The confound did NOT move the answer.
- Wider wings DO NOT help: 2.5% & 3.0% strictly worse on return, Calmar, DD, and # red years. Wing
  width in 2-3% is second-order noise; credit given up buying closer protection isn't recovered by
  tail reduction in this universe.
- Consistency killers are 2023 (whipsaw) + 2026 (5-mo stub), red at EVERY width = a REGIME problem,
  not a wing problem → VIX filter territory.

**LOCKED BASE: wing = 2.0% of ATM (rebuild as ±500 pts live; re-spec as % if NIFTY moves materially).
Width sweep CLOSED.**

NEXT: apply VIX>=12 floor to the 2.0% base (proven on fixed-±500: all 8 yrs green, +8.1L). Claude to
pull daily VIX from Kite and filter the 2% entry dates directly — no AlgoTest re-run needed. Then
SL/target sweep + entry-time sweep on the locked 2%+VIX base.

---

## VIX OVERLAY ON LOCKED 2% BASE (2026-06-08) — VIX>=13 floor locked

Claude pulled India VIX (token 264969) daily history 2019-2026 from Kite (1841 days) and applied a
VIX floor to the 2.0%-wing base (ex-COVID, 270 trades). Entry-VIX proxy = daily VIX OPEN (09:20 entry).
Margin Rs9.58L, 7.3yr span, net of 0.25% slip + Rs20/order + taxes.

| 2% base + filter | Trades | Net total | Calmar | MaxDD | RoM/yr | Red years |
|---|---|---|---|---|---|---|
| no filter | 270 | +7.64L | 0.70 | -1.50L | 10.9% | 2023, 2026 |
| VIX>=12 | 231 | +6.15L | 0.56 | -1.50L | 8.8% | 2023, 2026 |
| **VIX>=13** | 203 | **+8.53L** | 0.76 | -1.54L | **12.2%** | only 2026 |
| **VIX>=14** | 172 | +7.79L | **0.94** | **-1.13L** | 11.1% | only 2026 |

READ:
- Plateau shifted to 13-14 (was 12-14 on the intraday-CSV +-700 run) ONLY because daily-VIX-open runs
  ~1pt below the true 09:20 entry VIX. ==> VIX>=13 here ~= VIX>=12 live. Same finding, re-confirmed.
- VIX>=13 flips 2023 GREEN (-74k -> +50k) and lifts total to +8.53L. The whipsaw year was a low-VIX-band
  problem, as predicted.
- VIX>=14 = risk-adjusted winner: Calmar 0.94 (vs 0.70 unfiltered), DD -1.13L, at -1/3 trades.
- 2026 marginally red at EVERY floor (-10..-20k) = a 5-month, 8-11-trade stub dominated by two -40k Jan
  trades. Too small a sample to drive the lock; not a strategy flaw. Discount it.

**LOCKED BASE: 2.0% wings (=+-500 today) + VIX>=13 entry floor.** Best balance: +8.5L, Calmar 0.76,
12.2% RoM, every FULL year green. Use VIX>=14 for max risk-adjusted (Calmar 0.94, smallest DD). Live
filter reads VIX at 09:20 -> >=13 is the right live threshold.
Script: research/60_v2_straddle_optimization/scripts/vix_overlay_2pct.py

NEXT: SL/target sweep + entry-time sweep on this locked 2%+VIX>=13 base. Then wire the live V2 page
(research/57 engine) to run these rules and show entry/exit time + reason.

---

## SL SWEEP DONE — LOCKED BASE = 2.0% wings + 2.0% move-stop + VIX>=13 (2026-06-08)

Phase-1 stop-loss sweep on the 2.0%-wing base, run on AlgoTest (per-trade entry-VIX in CSV, exact).
Net of costs, ex-COVID, 273->271 trades, margin Rs9.58L, 7.3yr.

STOP-LOSS @ VIX>=13:
| stop | net | Calmar | MaxDD | neg |
| 1.0% | +6.51L | 0.58 | -1.53L | 2019 |
| 1.5%* | +8.53L | 0.76 | -1.54L | 2026 |
| **2.0%** | **+8.80L** | **1.03** | **-1.17L** | only 2026 |
| 2.5% | +6.29L | 0.62 | -1.39L | 2026 |
| no-SL | +8.85L | 0.97 | -1.25L | 2021, 2026 |
(* 1.5% VIX rows = daily-open proxy; all else exact CSV VIX.)

STOP-LOSS no-filter: 1.0% +6.73L/0.44 | 1.5% +7.64L/0.70 | 2.0% +8.50L/0.68 | 2.5% +6.60L/0.49 | no-SL +8.98L/0.89.
VIX floor on 2.0% base: none +8.50L/0.68 | >=13 +8.80L/1.03 | >=14 +8.16L/0.89 (8/8 GREEN).

KEY FINDING: Calmar PEAKS at a 2.0% stop (0.76->1.03->0.62 across 1.5/2.0/2.5% @VIX>=13) — NOT monotonic.
My interim "drop the stop entirely / monotonic" read (before 2.0/2.5 were in) was WRONG and is retracted.
The DEFINED-RISK WINGS are the real risk control; the stop is a sweet-spot, not a plateau -> treat live
rule as "~2% wide move-stop", not a precise value. 1.0% over-stops (choppy -2.1L DD); 2.5% dips.

**LOCKED BASE: 2.0% wings + 2.0% underlying move-stop + VIX>=13** (Balanced: Calmar 1.03, +8.80L,
DD -1.17L, 7/8 green, only 2026 stub red). Conservative alt = VIX>=14 (Calmar 0.89, +8.16L, 8/8 green).
This REPLACES the old V2 spec's 1.5% stop.

PUBLISHED to the app: /app/backtest/v2-nifty-ironfly-sl-vix (BacktestStudy in frontend/src/data/backtests.ts;
factsheet PNG frontend/public/v2_ironfly_factsheet.png; PNG generator one-off via matplotlib on VPS venv).
Standalone HTML factsheet also at (laptop) research_v2_locked_factsheet.html.

NEXT — Phase 2: profit-target sweep on the 2.0%-stop + VIX>=13 base. PT in {25%, 55%, 70%, none}
(40% = the 2.0% run already in hand). Then entry-time sweep. Then wire live V2 card to research/57 engine.

---

## CPR COMPRESSION OVERLAY — WALK-FORWARD VALIDATED (2026-06-08, candidate)

Conditional-attribution study on the locked 2%-wing + 2%-stop + VIX>=13 book (Claude pulled NIFTY 50
daily OHLC from Kite, built causal entry-time features, sliced P&L). Losses concentrate in volatility
COMPRESSION, best flagged by a NARROW PRIOR-DAY DAILY CPR = |TC-BC| from the prior day's H/L/C
(TC=2*pivot-BC, pivot=(H+L+C)/3, BC=(H+L)/2) / entry-day open, in %. Causal (known before 09:20).

OVERLAY (VIX>=13, ex-COVID; CPR needs only 1 prior day):
- baseline ~203t Calmar 0.95
- + skip CPR width < 0.10%: 147t, +11.0L, Calmar 1.59, 7/8 green
- + skip CPR<0.10% & Jan/Aug/Sep: 116t, +11.85L, Calmar 1.71, 8/8 GREEN
CPR filter RAISES return AND CUTS drawdown simultaneously (signature of a real conditioning var).

WALK-FORWARD (pick CPR threshold by Calmar on TRAIN half, apply blind to TEST half):
- train 2019-22 -> t*=0.12 -> TEST 2023-26: Calmar 1.13->2.81, DD -117k->-51k
- train 2023-26 -> t*=0.12 -> TEST 2019-22: Calmar 1.11->2.08, DD -102k->-72k
- fixed t=0.10 helps BOTH halves (H1 1.11->1.75, H2 1.13->1.83)
- skipped bucket (CPR<0.10) NEGATIVE in BOTH halves (-90k, -130k) -> robust, not one era
Same threshold both directions; improves OOS on return AND drawdown = genuine edge, not overfit.

MECHANISM: narrow CPR = compression -> expansion/breakout in days ahead -> short-gamma fly run over.
Seasonal Jan/Aug/Sep also a persistent drag (neg 5/8 yrs) -> secondary, same regime story.
DIRECTIONAL SKEW NOT SUPPORTED (prior-momentum sign/strength not a clean P&L driver) -> regime SKIP, not tilt.

CAVEATS: in-sample (one instrument/engine); threshold ~0.10-0.12% = "skip bottom-quartile CPR width",
not a precise constant; ~9 features tested (CPR earned it on mechanism+monotonicity+per-year+walk-forward);
trade count drops 191->147; needs FORWARD/paper confirmation before live.

STATUS: STRONG CANDIDATE (walk-forward passed). NOT yet folded into the locked base or the app study.
Factsheet (laptop) research_v2_locked_factsheet.html has a collapsible CPR-overlay section (curves + monthly
tables). Scripts: research/60_v2_straddle_optimization/scripts/attrib*.py + factsheet2.py.
NEXT: (1) forward paper-validate; (2) AlgoTest native CPR filter else compute CPR in live engine, skip narrow
days; (3) test WEEKLY-CPR variant; (4) once forward-confirmed, fold into locked base + app study.
