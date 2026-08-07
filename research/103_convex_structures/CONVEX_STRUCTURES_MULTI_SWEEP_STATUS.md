# Convex & Defined-Risk Option Structures — Batman / Ratio-Spreads / Backspreads, NIFTY·BNF·SENSEX, Weekly/Bi-weekly/Monthly

STATUS: RUNNING (G1 — NIFTY structure bake-off)

## 1. The Ask

**What Arun asked (verbatim):** "ok this returns is not good. even nifty buy and hold gives me better returns. we need to try more, completely different this. Batman, ratio spreads etc, try and optimize for weekly, bi weekly. monthly series - in nifty, bnf, and sensex with historical EOD data. You shud not only look at exits, but also adjustments."

**What we're actually testing:** Short-premium strangles top out ~10% CAGR and lose to NIFTY buy-&-hold (~12–13%). To BEAT B&H we need structures with positive convexity or leverage, not just theta harvest. Test a family of defined-risk / convex option structures — **ratio backspreads (put & call), front ratio spreads, iron fly, Batman (double OTM butterfly), broken-wing butterfly** — plus the short-strangle baseline, across **weekly / bi-weekly / monthly** tenors on **NIFTY, BankNifty, (SENSEX cross-check)**, using EOD bhav data. Optimize **entry structure × tenor × adjustments × exits**, unbiased, and conclude whether any family beats NIFTY B&H on a risk-adjusted AND absolute basis.

## 2. The Base — what's being tested

- **Strike selection:** ATM = nearest strike to entry-day OPEN spot (no look-ahead); offset = nearest to spot×(1±pct); premium = nearest close to ₹target.
- **Structures (G1, naked / hold-to-expiry):**
  - `SHORT_STRANGLE` — sell CE+PE ~₹20 (baseline, known)
  - `IRON_FLY` — sell ATM CE+PE, buy ±2% wings (defined-risk short vol)
  - `PUT_BACKSPREAD` — sell 1 near put, buy 2 further OTM puts (long crash convexity)
  - `CALL_BACKSPREAD` — sell 1 near call, buy 2 further OTM calls (long melt-up convexity)
  - `PUT_RATIO` (front) — buy 1 put, sell 2 further OTM (credit, range profit, downside tail)
  - `BATMAN` — OTM call butterfly + OTM put butterfly (twin-peak, cheap, defined risk, profits on a move either way)
  - `BWB_PUT` — broken-wing put butterfly (skewed, near-zero cost, directional-down convexity)
- **Direction:** each structure treated as specified (some directional, some neutral).
- **P&L:** MTM = Σ(sign×ratio×price); net of ₹400/leg-transaction + 0.3% slippage; 10 lots (NIFTY qty 750).
- **Success criterion:** total & CAGR vs **NIFTY buy-&-hold** over the same window, plus Calmar / max single-trade loss / win%. A family must show a pulse (positive expectancy and a path to beat B&H) to advance to G2.

## 3. Plan — stage gates

| Gate | Scope | Kill criterion |
|---|---|---|
| **G1** | NIFTY, 7 structures × {weekly, monthly}, naked hold-to-expiry | drop families with negative expectancy or that can't plausibly beat B&H |
| **G2** | Survivors + **adjustments** (roll tested side, convert-to-fly, re-center, delta-hedge on breach) + exits + bi-weekly | drop adjustments that don't lift risk-adjusted return |
| **G3** | Replicate best on **BankNifty** (15y); SENSEX 2024–26 cross-check | drop if edge doesn't transfer |
| **G4** | Walk-forward / per-year stability / cost sensitivity; client tearsheet; publish `/app` page | — |

## 4. Grid (G1)

- 7 structures × 2 tenors (weekly 2019–26, monthly 2015–26) × 1 underlying (NIFTY) = 14 cells, naked.
- Adjustments deferred to G2 (per stage-gate — don't spend G2 compute before G1 passes).

## 5. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-08-06 | Data audit done | NIFTY/BNF 15y (nse_options_bhav); SENSEX/BANKEX only 2024→26 (bse_options_bhav) |
| 2026-08-06 | STATUS-MD written, G1 launching | structure_bakeoff.py — NIFTY 7×2 naked |

## 6. Crash Recovery

- G1 runner: `research/103_convex_structures/scripts/structure_bakeoff.py` (self-contained; reads market_data.db on VPS).
- Rerun: `ssh arun@94.136.185.54 'cd /home/arun/quantifyd && ./venv/bin/python3 research/103_convex_structures/scripts/structure_bakeoff.py'`
- Output: prints a ranked table + writes `research/103_convex_structures/results/g1_nifty.csv`. Idempotent — safe to re-run.
- Do NOT touch market_data.db (read-only use).

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/structure_bakeoff.py` | G1 NIFTY structure engine + sweep | yes |
| `CONVEX_STRUCTURES_MULTI_SWEEP_STATUS.md` | This file | yes |
| `results/g1_nifty.csv` | Per-structure×tenor G1 results | yes (small) |
| `results/RESULTS.md` | Final verdict (written at end) | yes |

## 8. Findings

### G1 (NIFTY, naked hold-to-expiry) — DONE 2026-08-06

**Verdict: no naked structure beats NIFTY B&H risk-adjusted. The convex/long families BLEED; the credit families make money but with account-killing tails.**

Monthly (2015–26), 10 lots, ₹20L, net:
| Structure | Total | avg/trade | Win | Worst | eqMaxDD |
|---|---|---|---|---|---|
| PUT_RATIO_FRONT | +₹155L | +37.9k | 78% | −₹13.9L | **−₹39.5L** (mirage — wiped in DD) |
| SHORT_STRANGLE | +₹51.6L | +12.9k | 88% | −₹8.3L | −₹18.3L |
| BWB_PUT | −₹12.7L | −3.1k | 31% | | −₹28L |
| CALL_BACKSPREAD | −₹17L | −4.1k | 38% | | −₹65L |
| BATMAN | −₹45.6L | −11.4k | 36% | | −₹54L |
| IRON_FLY | −₹54.9L | −13.7k | 31% | | −₹71L |
| PUT_BACKSPREAD | −₹208L | −50.9k | 17% | | −₹243L |
| **NIFTY B&H** | **+₹33.9L (170%)** | | | | (index DD only) |

Weekly (2019–26) same shape: only PUT_RATIO_FRONT (+₹23.9L, 29% win, −₹14.2L DD) and SHORT_STRANGLE (+₹7.5L, 82%, −₹10L DD) positive; all convex/long families negative.

**Reads:**
1. **Naked long-convexity (put/call backspread, Batman, BWB, iron fly) all lose held to expiry** — pure theta bleed. They can only work with a TIMING signal (deploy cheap convexity *before* an expansion), never unconditionally.
2. **Credit structures (front ratio, short strangle) make money but carry fat tails** (−₹14L to −₹40L). The front-ratio's +₹155L total is a MIRAGE — the −₹39.5L drawdown wipes the ₹20L account; you'd be margin-called out.
3. **Nothing naked beats B&H risk-adjusted.** To beat B&H you must either TIME the convex structures or ADJUST the credit-structure tails — which is exactly G2.

### G2 focus (next)
- **Signal-timed convexity** — deploy backspread/Batman ONLY on a vol-expansion / breakout / low-IV-squeeze signal (buy cheap convexity before the move). The real "beat B&H" thesis.
- **Credit-family tail adjustments** — roll/convert the tested short wing on the front ratio; the short-strangle hold+redeploy we already validated.

STATUS: G1 DONE — pivot to G2 (signal-timed convexity + tail adjustments)
