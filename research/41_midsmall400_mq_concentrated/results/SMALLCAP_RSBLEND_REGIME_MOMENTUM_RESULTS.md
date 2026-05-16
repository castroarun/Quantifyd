# Small-Cap RS(blend) + Regime Momentum — Results & MID-vs-SMALL Verdict

> **STATUS: EXPLORED — NOT PREFERRED. MID is the locked system.**
> Small clears the hurdle (30.2% post-tax) but at deeper drawdown
> (−28/−30% vs mid −24.6%), lower Calmar (1.27 vs 1.44), needs an extra
> junk-filter, and has near-zero F&O liquidity (1 vs 22 names). A
> higher-pain alternative only. 3-way verdict:
> `MIDCAP_RS120_REGIME_MOMENTUM_DETAILED_REPORT.md` §9.

Phase 06. Applies the **identical** drawdown-control overlay + OOS +
post-tax pipeline used for the mid-cap winner, to a small-cap core, so
the comparison is apples-to-apples. Companion to
`MIDCAP_RS120_REGIME_MOMENTUM_RESULTS.md`.

## 1. Why this run

Mid was chosen for the headline because, RS-alone, small had ~equal/
higher CAGR but far deeper drawdown (−40% to −66%) and worse Calmar
(0.95 vs mid 1.29). But Phase-03's regime gate is a *drawdown lever*,
and small has more bear beta to remove — so the gate could help small
disproportionately. This phase tests exactly that.

- **Small core:** `small_blend_6m12m_N15` — best risk-adjusted robust
  small RS-alone config. RS-alone baseline (full 2014–2026): **39.5%
  CAGR / −41.5% MaxDD / Sharpe 1.28 / Calmar 0.95**.
- Same overlay axes (quality pos-frac, own-DD cap, SMA200 regime),
  monthly rotation, top-22 buffer, 0.4% RT cost, 6.5% bear-cash,
  NIFTYBEES benchmark. Volume-confirm omitted (Phase-03 proved poison).

## 2. Small-cap overlay sweep (selected rows)

| Config | CAGR | MaxDD | Sharpe | Calmar |
|---|---|---|---|---|
| SMALL baseline (RS-alone) | 39.5% | −41.5% | 1.28 | 0.95 |
| q__dd__REG (regime only) | 34.9% | −35.7% | 1.53 | 0.98 |
| q0.5_dd__REG | 36.1% | −33.4% | 1.57 | 1.08 |
| q0.58_dd-0.4_REG | 36.2% | −31.6% | 1.55 | 1.15 |
| **q0.5_dd-0.4_REG ★ champion** | **35.9%** | **−28.2%** | **1.56** | **1.27** |

Regime gate alone cuts small's DD −41.5%→−35.7%; adding the quality
screen + a −0.4 own-DD cap takes it to −28.2% at ~36% CAGR. (Full
table: `phase06_smallcap_overlay.csv`.)

## 3. Small champion `q0.5_dd-0.4_REG` — OOS + post-tax

**OOS sub-period (fixed config):**

| Window | CAGR | MaxDD | Sharpe |
|---|---|---|---|
| H1 2014–2019 | 35.0% | −28.2% | 1.42 |
| H2 2020–2026 | 35.1% | −16.6% | 1.65 |

Remarkably stable — H1≈H2≈35% (even tighter than mid's 32/37 split).

**Post-tax (STCG):**

| | CAGR | MaxDD | Sharpe | Drag |
|---|---|---|---|---|
| Gross | 35.9% | −28.2% | 1.56 | — |
| Net STCG @15% | 31.6% | −29.3% | 1.41 | −4.4pp |
| Net STCG @20% (current) | 30.2% | −29.8% | 1.36 | −5.7pp |

## 4. MID vs SMALL — honest verdict

| Metric (gated, post-tax @20% STCG) | MID `q0.5_dd__v__REG` | SMALL `q0.5_dd-0.4_REG` |
|---|---|---|
| Post-tax CAGR | 28.9% | **30.2%** |
| MaxDD (gross / net) | **−24.6% / −25.3%** | −28.2% / −29.8% |
| Sharpe (gross) | 1.53 | **1.56** |
| Calmar (gross) | **1.44** | 1.27 |
| OOS H1 / H2 CAGR | 32.2% / 37.3% | 35.0% / 35.1% |
| Robust ex-top-3 (RS-alone) | yes | yes |
| Extra filter needed | none | −0.4 own-DD cap |

**Verdict:** Small, properly gated, **clears the ~20% hurdle by even
more than mid on raw return** and is OOS-stable — it is *not* inferior.
But vs the mid champion it delivers only **+1.3pp post-tax CAGR for
~4–5pp deeper drawdown** and a lower Calmar (1.27 vs 1.44), and it
needs an extra junk filter to get there. Per the pre-set rule (small
wins only if it *clearly* beats mid's CAGR **and** DD is not materially
deeper), **MID `q0.5_dd__v__REG` remains the recommended risk-adjusted
system.**

Small is best understood as a **higher-return / higher-pain point on
the same efficient frontier**: appropriate only for an investor who
will genuinely tolerate a ~−30% equity hole in exchange for ~1–2pp
extra post-tax CAGR. Same binding caveats as the mid study apply
(price-path "quality" not fundamentals; PIT liquidity-proxy universe;
LTCG not modelled; no live wiring; not a guarantee).

## 5. Files

| File | Purpose |
|---|---|
| `scripts/06_smallcap_overlay_oos.py` | this pipeline (overlay + OOS + post-tax) |
| `results/phase06_smallcap_overlay.csv` | full small overlay sweep |
| `results/phase06_run.log` | run log |
| `results/SMALLCAP_RSBLEND_REGIME_MOMENTUM_RESULTS.md` | this document |
