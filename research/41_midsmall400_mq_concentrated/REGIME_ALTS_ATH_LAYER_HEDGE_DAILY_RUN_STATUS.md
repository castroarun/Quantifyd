# Regime-Filter Alternatives + ATH-Proximity Layer + Drawdown Hedge

**STATUS: COMPLETE — Phases 09/10/11 all done. New best system locked; two recommended endpoints. See §7.**

Extends the locked mid-cap winner (`mid_120d_N15` + q0.5 + SMA200
regime). Companion to `MIDCAP_RS120_REGIME_MOMENTUM_DETAILED_REPORT.md`.

## 1. The Ask (user, condensed)

> (a) The 200-SMA regime gate is lagging / whipsaws — try other regime
> filters (ATR / volatility, momentum, drawdown-based, faster MAs).
> (b) On top of RS, also layer the earlier MQ idea: pick stocks within
> ~10% of all-time-high and exit on a 20% trailing drawdown — what
> happens combined/layered?
> (c) Without the gate CAGR is higher but drawdown is the pain — find a
> way to cut DD via hedging (covered calls / short Nifty OTM / opportunities).

## 2. Honest scope decisions

- **Covered calls on the 15 holdings: REJECTED (not built).** Caps the
  right-tail that *is* the CAGR; and the rotating mid-cap holdings mostly
  have no liquid options (only ~22 of the whole mid band is F&O). Would
  look clever and silently destroy the edge.
- Regime alts + ATH layer = Phase 09 (price-only, no new data).
- Hedge = Phase 10: **regime-triggered Nifty beta-hedge** (in risk-off,
  hold the stocks but short a Nifty notional → keep RS alpha, kill market
  beta) — needs no options data. Index-put overlay only if
  `options_data.db` has real Nifty option/IV history (else flagged as an
  assumed-premium estimate, not a result).

## 3. The Base (held constant)

Core = mid PIT band, RS-120 vs NIFTYBEES, N=15, monthly, top-22 buffer,
q0.5 positive-month quality screen, 0.4% RT cost, 6.5% bear-cash,
2014-2026. Only the regime rule / ATH layer / hedge changes, so every
delta is attributable.

Reference anchors: `SMA200(base)` gross 35.3% / net20 ~29% / DD −24.6% /
Calmar 1.44 ; `OFF` 37.0% / DD −29.6% / Calmar 1.25.

## 4. Plan / grid

**Phase 09A — regime filters (9):** OFF, SMA200(base), SMA100, SMA50,
cross50/200, DDfrom1yHi>10%, mom3m<0, volspike(ATR), SMA200+vol.
Each: gross + post-tax@20% + MaxDD/Sharpe/Calmar + per-year.

**Phase 09B — ATH layer (4)**, under the best-Calmar regime from 09A:
base / +ATH≤10% entry / +20% trail exit / +both.

**Phase 10 — hedge overlay:** regime-triggered Nifty beta-hedge vs
cash-out gate vs no-gate; + permanent partial hedge; + (data-permitting)
index-put overlay. Separate runner, after 09.

## 5. Status (live log)

| Time IST | Event | Notes |
|---|---|---|
| 2026-05-16 ~12:4x | STATUS written, Phase 09 launched (bg `bjwo0lp8q`) | 9 regime + 4 ATH configs, monitor `b51kjd4xi` |
| 2026-05-16 ~13:0x | **Phase 09 DONE** | SMA100 wins; ATH-entry additive; trail inert; ATR failed |

## 6. Crash recovery

Re-run: `python research/41_midsmall400_mq_concentrated/scripts/09_regime_alts_ath_layer.py`
(idempotent; reads canonical `backtest_data/market_data.db`). Outputs →
`results/phase09_regime_alts.csv`, `phase09_ath_layer.csv`,
`phase09_*_peryear.csv`, log `results/phase09_run.log`. Phase 10 runner:
`scripts/10_*` (written when 09 closes).

## 7. Findings

**Phase 09 — DONE.** Core held constant (mid_120d_N15 + q0.5).

Part A (regime filters): **SMA100 is the clear winner** — 35.1% gross /
29.5% post-tax / **MaxDD −16.4%** / Sharpe 1.66 / **Calmar 2.14** vs
SMA200's −24.6% / 1.44 at ~same CAGR. It still protects the 2025 bear
(−1.1% vs no-gate −19.7%) without SMA200's whipsaw drag in 2016/2019/
2022. Honest cost: SMA100 has a worse 2018 (−13.0% vs −1.9%); not
strictly year-dominant but full-period DD/Calmar far better.
Losers: cross50/200, DDfrom1yHi, **volspike(ATR) (user idea — FAILED,
−33% DD/Cal 1.02; NIFTYBEES has no true OHLC so ATR is a c2c proxy,
flagged)**; SMA200+vol decent (Cal 1.61) but < SMA100; SMA50 over-
whipsaws (29.7%).

Part B (ATH layer on SMA100): **+ATH≤10% entry = additive win** →
35.2% / 29.3% post-tax / **−15.1% DD / Sharpe 1.78 / Calmar 2.33**.
**+20% trailing exit = INERT** (no change; the monthly top-22 RS buffer
already rotates losers before −20% from peak — honest: don't bother).

**NEW BEST SYSTEM (supersedes the SMA200 lock):**
`mid_120d_N15 + q0.5 + SMA100 regime + ATH≤10% entry` →
**35.2% gross / 29.3% post-tax / −15.1% MaxDD / Sharpe 1.78 / Calmar
2.33** (vs old SMA200 lock Calmar 1.44). Trailing-stop dropped (inert).
→ Pending: fold into DETAILED_REPORT + app; run Phase 10 (hedge) vs
this stronger baseline.

**Phase 10 — hedge overlay:** RUNNING (bg `bbri7s5ov`). SMA100→cash vs
SMA100→beta-hedge (keep stocks, short Nifty) vs permanent partial short
vs OFF. Covered calls rejected (caps CAGR tail; holdings illiquid).

**Phase 10 — DONE.** Permanent hedge = dead (constant short bleeds the
bull; CAGR 28→21%, Cal <1 — rejected). **Regime-triggered beta-hedge
hr1.0 = standout: 42.8% gross / 34.0% post-tax (highest of ANY config
in the project) / DD −22.7% / Cal 1.89** — in gated months it's long
top-RS / short 1× Nifty, harvesting the RS spread as market-neutral
alpha instead of dead cash. NOT a DD reducer (−22.7 vs cash −15.1;
mid-cap β>1 under-hedged) — a *return amplifier with contained DD vs
OFF (−33%)*. hr0.5 dominated. → Frontier (all SMA100): cash =
smoothest (29.3% / −15% / Cal 2.32); beta-hedge = max return (34% /
−23% / Cal 1.89).

**Phase 11 — DONE.** Stock-level controls **alone cannot replace** the
market gate: OFF+trail{15,12,10} → DD ~−32% / Cal ~1.0; OFF+perStockSMA
→ −30% / Cal 1.10 (bottom-up signals fire only after each name falls —
too late in a broad bear). **On TOP of the gate they add a small, free
gain:** SMA100+perStockSMA+trail12 → 35.6% gross / 29.6% post-tax /
−15.1% DD / **Calmar 2.36** (vs SMA100+ATH 2.33; same DD, +0.3pp CAGR).

## 8. FINAL CONSOLIDATED VERDICT (Phases 09–11)

Core constant: `mid_120d_N15 + q0.5`. Journey (post-tax @20% STCG):

| System | post-tax CAGR | MaxDD | Calmar |
|---|---|---|---|
| Original lock — SMA200 | 29.4% | −24.6% | 1.44 |
| Ph09 — **SMA100** + ATH≤10% | 29.3% | −15.1% | 2.33 |
| Ph11 — + per-stock-SMA + trail12 | **29.6%** | **−15.1%** | **2.36** |
| Ph10 — SMA100→**beta-hedge** hr1.0 | **34.0%** | −22.7% | 1.89 |

**Two recommended endpoints (risk choice):**
- **SMOOTHEST (best risk-adj):** `mid_120d_N15 + q0.5 + SMA100 regime +
  ATH≤10% entry + per-stock-SMA100 + 12% trail` → 29.6% post-tax /
  −15.1% DD / Sharpe 1.80 / **Calmar 2.36**. Supersedes the SMA200 lock
  (was 1.44 — biggest single project improvement, from the user's
  SMA100 + ATH instincts).
- **MAX RETURN:** `… + SMA100→beta-hedge hr1.0` (short 1× Nifty in
  risk-off instead of cash) → **34.0% post-tax** / −22.7% DD /
  Calmar 1.89. The RS-spread alpha harvest.

Rejected honestly: covered calls (caps the CAGR tail; holdings
illiquid), permanent hedge (bleeds the bull), hr0.5 (dominated), ATR/
vol-spike regime (failed), 20% trail (inert), stock-level-only control
(can't replace the market gate). Caveats from the parent study still
bind (price-path quality not fundamentals; PIT liquidity proxy; LTCG
not netted; no live wiring).

| 2026-05-16 ~13:3x | **Phases 10 & 11 DONE; consolidated verdict locked** | new best Calmar 2.36; beta-hedge 34% post-tax |
| 2026-05-16 ~18:0x | **Phase 13 (COMBINED) DONE — dominated middle** | beta-hedge + per-stock-SMA + 12% trail = 43.0 gross / 33.7 net20 / −22.3 DD / Cal 1.92. Fails both ways: < Smoothest Calmar 2.36, < Max-Return net20 34.0 (extra stop churn adds tax drag, barely moves DD). Keep the two clean parents. |
| 2026-05-16 ~18:0x | **Phase 14 — equity overlay chart** | `results/final_systems_pl_overlay.png` (log-scale NAV + drawdown panel). Final: SMOOTHEST 40.4× / MAX-RETURN 75.2× / Nifty50 4.7× over 2014–2026. |

## 9. Phase 13/14 addendum

**Phase 13 — combining is NOT worth it.** Adding Smoothest's stock-level
exits onto Max-Return: 43.0% gross / **33.7% post-tax** / −22.3% DD /
Calmar 1.92. vs SMOOTHEST 35.6/29.6/−15.1/2.36 and MAX-RETURN
42.8/34.0/−22.7/1.89. The combined is a **dominated middle** — doesn't
beat Smoothest's risk-adjusted return nor Max-Return's post-tax CAGR
(the extra per-stock-SMA/12%-trail churn realizes more STCG and barely
changes drawdown — same inertia finding as the 20% trail). Decision:
keep the two clean parents; do not ship the hybrid.

**Phase 13 system NAMED = `FORTIFIED`** (user keeps it as a candidate
despite ~wash vs Max-Return — values the explicit per-stock exits as a
discipline comfort at near-zero cost).

### THREE CANDIDATE SYSTEMS (decision pending — user will pick later)

| System | Risk-off | Stock stops | CAGR | net20 | MaxDD | Sharpe | Calmar |
|---|---|---|---|---|---|---|---|
| **SMOOTHEST** (Ph11) | →cash | perSMA+12%tr | 35.6 | 29.6 | **−15.1** | 1.80 | **2.36** |
| **MAX-RETURN** (Ph10) | short 1×Nifty | none | 42.8 | **34.0** | −22.7 | 1.83 | 1.89 |
| **FORTIFIED** (Ph13) | short 1×Nifty | perSMA+12%tr | **43.0** | 33.7 | −22.3 | 1.84 | 1.92 |

Shared core: PIT mid-cap band, RS-120 vs NIFTYBEES, q0.5, ATH≤10% entry,
N=15, top-22 buffer, monthly, 0.4% RT, 6.5% cash, 2014-2026. Final
multiples: SMOOTHEST 40.4× / MAX-RETURN 75.2× / FORTIFIED ~75× / Nifty50
4.7×. **Open**: pick 1 of 3; optional Phase 15 (daily/weekly regime
gate to cut the month-end lag); optional protective-put hedge (needs
real options/IV data).

**Phase 14 — equity overlay** (`final_systems_pl_overlay.png`): both
track to ~2020 then Max-Return separates (75.2× vs 40.4×; Nifty50
4.7×); drawdown panel shows Max-Return consistently deeper (2018-19
~−21% vs ~−14%; 2025 ~−19% vs ~−9%). Visual of the return-for-drawdown
trade-off.

**Phase 15 — decoupled regime clock — DONE.** Daily-marked engine
(stricter daily MaxDD; compare WITHIN table only, not vs Ph09-13).

| System | M (Cal/DD) | W (Cal/DD) | D (Cal/DD) | Best |
|---|---|---|---|---|
| SMOOTHEST | 1.52 / −24.1 | **1.65 / −22.2** | 1.53 / −23.1 | **WEEKLY** |
| MAX-RETURN | 1.00 / −32.6 | 1.04 / −30.5 | 0.98 / −34.0 | ~flat (M) |
| FORTIFIED | 1.02 / −31.7 | 1.06 / −29.6 | 1.00 / −33.2 | ~flat (M) |

flips: M 29 / W 49 / D 105 (all systems). **Verdict:** the month-end
lag is real **and worth fixing only for SMOOTHEST → use WEEKLY** (DD
−24.1→−22.2, Calmar 1.52→1.65, CAGR flat; DAILY over-whipsaws — 105
flips, CAGR 36.7→35.3, no DD gain). For MAX-RETURN/FORTIFIED the
regime clock is ~irrelevant (Calmar pinned ~1.0 across M/W/D) — their
drawdown is driven by mid-cap β>1 under-hedging in bears, NOT regime
timing; faster only adds flip noise. → SMOOTHEST spec updated to a
**weekly** regime check; MAX/FORT stay month-end.
`scripts/15_decoupled_regime.py`, `results/phase15_decoupled_regime.csv`.

**Phase 18 — FORTIFIED-B (emergent-cash, no market gate) — RUNNING.**
User design: NO Nifty regime gate. Sell a holding ONLY if (close < its
own 100-day SMA) OR (price ≤ 0.88 × peak-since-entry). Refill freed
slots from fresh RS-ranked names passing q0.5 + ATH≤10% + above own
100-SMA. NO forced monthly RS rotation (a name is held until it breaks,
not dropped on RS-rank slip). Emergent de-risk: in a crash, holdings
hit stops AND nothing is near ATH → cash accumulates organically (6.5%).
Equal-weight new buys = total/15; survivors not trimmed (winners run).
Daily-marked engine; cadence swept **monthly vs weekly** (assess best).
vs SMOOTHEST/MAX-RETURN/FORTIFIED. Runner `scripts/18_fortified_b.py`.

**Phase 18 — FORTIFIED-B — DONE. REJECTED (dominated).** Emergent-cash,
no market gate; tested R(forced-RS-rot)/H(hold-til-break) × M/W.

| Variant | CAGR | net20 | dailyDD | Calmar | sells |
|---|---|---|---|---|---|
| R-M | 32.1 | 21.0 | −34.4 | 0.94 | 831 |
| R-W | 31.4 | **17.5** | −27.1 | 1.16 | 1749 |
| H-M | 33.7 | 26.6 | −35.3 | 0.95 | 473 |
| H-W | 29.8 | 20.7 | −30.8 | 0.97 | 781 |

vs SMOOTHEST(weekly) Cal 1.65 / DD −22.2 / post-tax ~30 (same daily
engine). Every FORTIFIED-B variant dominated: Calmar 0.94–1.16,
daily-DD −27 to −35%, post-tax 17–27% (best-Calmar R-W destroyed by
1749-sell STCG churn → 17.5% post-tax). Per-year: monthly variants bled
2025 (−20/−21%); weekly tamed it only via tax-ruinous churn. **Core
reason:** in a crash the damage is in names already held — not buying
fresh near-ATH names doesn't stop the held book bleeding; only an
explicit action on the held book (cash / short Nifty) controls DD.
No good cadence — the mechanism is the flaw, not its timing. **Verdict:
drop FORTIFIED-B; keep the explicit-gate trio (SMOOTHEST/MAX-RETURN/
FORTIFIED).** Confirms Phase 11 (stock-level alone can't replace the
market gate).

**Open (user, not started):** protective-put hedge (5% OTM / ITM,
regime-triggered) — conceptually a better fit than the futures short
(keeps upside) but needs REAL Nifty options/IV history to backtest
honestly; pending an `options_data.db` coverage check. No fabricated
IV.
User Qs: (a) the 20% trail was INERT — so test *binding* tighter
stock-level trailing stops (15/12/10%); can a binding stop REPLACE the
market SMA100 gate? (b) per-stock SMA100 trend filter (hold only while a
name is above its OWN 100d SMA) — instead of / alongside the market
regime. Core = mid_120d_N15 + q0.5 + ATH≤10% entry. Configs: OFF+trail
{15,12,10}, OFF+perStockSMA, OFF+perStockSMA+trail12, SMA100+perStockSMA,
SMA100+trail12, perStockSMA-only, vs Ph09 winner (SMA100+ATH). Each
gross + post-tax@20% + per-year. Runner `scripts/11_stocklevel_risk.py`.

**niftyindices extra-index fetch (bg `banij5nyh`):** slow/flaky (4-retry
×18s on stalling years). De-prioritised; will ship heatmap with reliable
data + honest blanks if it doesn't converge. Not blocking.

| 2026-05-16 ~13:1x | Phase 10 launched (bg `bbri7s5ov`, mon `bvpfc56od`); Phase 11 building | hedge + stock-level risk |
