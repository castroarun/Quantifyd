# NIFTY straddle + BANKNIFTY wings (cross-instrument hedge) — STATUS

**STATUS: G1 PASS (beta thesis validated) → ₹ test = AlgoTest two-backtest (prepared 2026-06-19).**

## The thesis
SHORT NIFTY ATM straddle (collect premium) + LONG BANKNIFTY OTM wings (the hedge). BANKNIFTY is
higher-beta, so in a big move/collapse it moves MORE than NIFTY → the BNF wings over-cover the NIFTY
straddle's tail while the wings can sit further OTM (cheaper) than same-index NIFTY wings.

## G1 — beta validation (real BNF daily 2015-2026, downloaded 2026-06-19)
| measure | value |
|---|---|
| corr(NIFTY,BNF) daily | 0.889 |
| beta (all) | **1.19** |
| beta big-down (<−1%) | **1.28** |
| weekly beta / down-beta | 1.24 / 1.22 |
| weekly \|move\| p90 | BNF **4.42%** vs NIFTY 3.02% |

Crash days: NIFTY<−2% → BNF falls 1.25× (more 79% of the time); NIFTY<−3% → 1.27× (more 86%).
**Validated:** BNF over-covers in collapses. CAVEATS: ~21% basis risk (corr 0.889); down-beta dipped
<1 in 2018 (0.83) & 2025 (0.92); the 2024-26 window was anomalously low-beta (0.98) — do NOT judge on it.

## ₹ test — AlgoTest CANNOT combine NIFTY+BNF in one backtest → run TWO, combine offline
Run each leg as its own AlgoTest backtest with IDENTICAL timing, export both Trades CSVs, Claude aligns by
entry-date and sums → net per-cycle P&L, per-year, MaxDD, crash-day behaviour.

### BT1 — SHORT NIFTY ATM straddle (the income leg)
- NIFTY · Cash underlying · Positional · **[run twice: Weekly AND Monthly]**
- Weekly: Entry **8 TD before expiry** (≈2nd-nearest weekly), Exit **1 TD before**. Monthly: monthly expiry.
- Entry 09:20, Exit 15:15 · Leg1 SELL CALL ATM · Leg2 SELL PUT ATM
- **NO underlying stop** (the BNF wings ARE the protection — that is the whole test) · PT OFF
- Re-entry ON · costs ON · **10 lots**

### BT2 — LONG BANKNIFTY wings (the hedge leg) — SAME timing as BT1
- BANKNIFTY · Cash underlying · Positional · **same expiry basis + same Entry/Exit TD as BT1**
- Entry 09:20, Exit 15:15 · Leg1 **BUY CALL OTM** · Leg2 **BUY PUT OTM**
- Re-entry ON · costs ON · lots = per the SIZING below
- **WING-WIDTH GRID (run 3×):** BNF wings at **±1.5% / ±2.0% / ±2.5%** of the BNF ATM
  (≈ ±800 / ±1050 / ±1300 pts at BNF ~52,000 — use the nearest 100-pt strike)

### SIZING (notional-match, so the beta does the over-cover)
Match the BNF notional to the NIFTY straddle notional:
  qty_BNF ≈ qty_NIFTY × (NIFTY_spot / BNF_spot) ≈ qty_NIFTY × (24,000 / 52,000) ≈ **0.46 × qty_NIFTY**
Worked (NIFTY lot 75, BNF lot 15): 10 NIFTY lots = 750 qty → BNF qty ≈ 345 → **≈ 23 BNF lots** (×15).
At notional-match, BNF's 1.25× crash-beta gives ~**25% over-cover** on a collapse. (Confirm live lot sizes.)
Sizing is a lever — also worth a 0.5× and 0.75× notional run to see how little BNF you need.

### Combine (Claude does this)
Per cycle: net = (NIFTY straddle P&L) + (BNF wings P&L). Build per-year net, MaxDD, and the crash-cycle
table (did the BNF wings actually cover the straddle's worst NIFTY-move cycles?). Compare vs the finalized
same-index NIFTY ±500 fly (research/60) — does the cross-hedge beat it on tail / net / capital?

### Export & hand back
BT1 Trades CSV + BT2 (×3 widths) Trades CSVs (+ PDFs). Save them in research/69_nifty_straddle_bnf_wings/.
