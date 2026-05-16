# Combo (Mid+Small) RS(blend) + Regime Momentum — Results

> **STATUS: EXPLORED — NOT PREFERRED. MID is the locked system.**
> Combo is strictly dominated by the mid config (lower post-tax CAGR
> *and* deeper drawdown *and* lower Calmar). Kept for completeness /
> apples-to-apples comparison only. See the 3-way verdict in
> `MIDCAP_RS120_REGIME_MOMENTUM_DETAILED_REPORT.md` §9.

Phase 07. Identical drawdown-control overlay + OOS + post-tax pipeline
as mid (Phase 03/04) and small (Phase 06), applied to the combined
mid+small band (rank 101–500).

## 1. Core & baseline

- **Combo core:** `combo_blend_6m12m_N20` (strong robust RS-alone combo;
  ex-top-3 37.9%). RS-alone baseline 2014–2026: **39.9% CAGR / −40.3%
  MaxDD / Sharpe 1.35 / Calmar 0.99**.
- Same axes (quality pos-frac, own-DD cap, SMA200 regime), monthly,
  top-22 buffer, 0.4% RT, 6.5% bear-cash, NIFTYBEES benchmark.

## 2. Overlay sweep (selected)

| Config | CAGR | MaxDD | Sharpe | Calmar |
|---|---|---|---|---|
| COMBO baseline (RS-alone) | 39.9% | −40.3% | 1.35 | 0.99 |
| q0.5_dd__REG | 37.1% | −34.7% | 1.61 | 1.07 |
| q0.5_dd-0.5_REG | 34.7% | −31.4% | 1.55 | 1.11 |
| **q0.58_dd-0.4_REG ★ champion** | **33.3%** | **−29.6%** | **1.49** | **1.13** |

(Full: `phase07_combo_overlay.csv`.)

## 3. Champion `q0.58_dd-0.4_REG` — OOS + post-tax

**OOS:** H1 2014–2019 32.0% / −29.6% / Sh 1.40 · H2 2020–2026 33.8% /
−16.2% / Sh 1.54 (stable both halves).

**Post-tax:**

| | CAGR | MaxDD | Sharpe | Drag |
|---|---|---|---|---|
| Gross | 33.3% | −29.6% | 1.49 | — |
| Net STCG @15% | 29.3% | −30.4% | 1.36 | −4.0pp |
| Net STCG @20% (current) | 28.1% | −30.6% | 1.31 | −5.2pp |

## 4. Verdict

Combo, properly gated, clears the ~20% hurdle and is OOS-stable — but
vs the **mid** champion it is **worse on every axis that matters**:
post-tax CAGR 28.1% (*below* mid's 28.9%), MaxDD −30.6% (vs −24.6%),
Sharpe 1.31 (vs 1.53), Calmar 1.13 (vs 1.44). It is **strictly
dominated by mid** — there is no investor preference under which combo
beats mid. Combining the universes just dilutes mid's quality with
small's drawdown. **Recommendation: do not use combo; use mid.**

Same binding caveats as the mid study apply.

## 5. Files
| File | Purpose |
|---|---|
| `scripts/07_combo_overlay_oos.py` | this pipeline |
| `results/phase07_combo_overlay.csv` | full combo overlay sweep |
| `results/phase07_run.log` | run log |
| `results/COMBO_RSBLEND_REGIME_MOMENTUM_RESULTS.md` | this document |
