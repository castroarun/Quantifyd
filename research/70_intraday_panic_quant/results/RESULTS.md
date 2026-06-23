# Intraday PANIC / Spike Quantification for NIFTY — a desk number that says "act now"

**STATUS: DONE** · 2026-06-23 · verdict: **TOOL BUILT & WIRED INTO THE GUARDIAN** (calibrated, live).

## The ask
> "How do we quantify today's deep spike move based on something? Let the news be news — within the
> trading desk there should be a quantification that says *beyond this, act urgently*. There was a sudden
> fall ~1–2 months ago around 3 PM IST; consider that and other scenarios, assess comprehensively."

Build a news-independent intraday panic gauge with calibrated "act now" thresholds.

## Data
- **Recent intraday events:** `options_data.db` `underlying_spot` — per-minute NIFTY since 2026-04-20.
- **Long-history tail / thresholds:** `market_data.db` `market_data_unified` NIFTY50 5-minute, **2,706 days, 2015-02 → 2026-03**.

## Metrics (per day, intraday)
| Metric | Definition | Axis |
|---|---|---|
| **V15** | worst (most negative) rolling **15-min** return | velocity |
| V30 | worst rolling 30-min return | velocity (persistence) |
| **DDhi** | max drop from the running intraday **high** | depth |
| **Z15** | \|V15\| / that day's 15-min-return sigma | abnormality (regime-normalised) |

## Calibration (2015–2026, worst-15-min-drop of the day)
| Pctile | V15 | DDhi | Z15 |
|---|---|---|---|
| p50 | −0.28% | −0.56% | 2.6σ |
| p75 | −0.39% | — | — |
| p90 | −0.55% | −1.34% | 3.6σ |
| p95 | −0.69% | −1.69% | 4.0σ |
| **p99** | **−1.13%** | **−2.89%** | **4.6σ** |
| p99.9 | −2.10% | −5.36% | 5.4σ |

Worst 10 days are **all 2020-COVID** (V15 −2 to −3.65%, DDhi −4 to −6.9%).

## The "~3 PM fall" — identified: **2026-05-29**
Single worst intraday spike in the recorder window, hit at **15:10**:
**V15 −1.11% · V30 −1.20% · DDhi −1.97% · Z15 5.3σ · day −1.48%.** This is the **PANIC archetype**.

## Today (2026-06-23) for contrast — a GRIND, not a spike
**V15 −0.44% @ 11:59 · DDhi −0.91% · Z15 3.7σ · day −0.62%** → only **~p75–p80** of all days.
Despite the KOSPI −8% headline, NIFTY's actual intraday velocity is unremarkable → tier **ELEVATED**, no urgency.

## The PANIC SCORE (wired into `weekly_cpr_guardian.py` → `panic_metrics()`)
Tier = the worse of velocity / depth / abnormality:

| Tier | Trigger (any one) | Action |
|---|---|---|
| NORMAL | V15 > −0.40% | ignore |
| ELEVATED | V15 ≤ −0.40% or Z ≥ 3.5σ | note, check level proximity |
| SPIKE | V15 ≤ −0.70% (p95) / DDhi ≤ −1.5% / Z ≥ 4.0σ | defense-ready + news-check |
| **🔴 PANIC / ACT URGENT** | **V15 ≤ −1.10% (p99) / DDhi ≤ −2.0% / Z ≥ 4.6σ** | **act now — skip the 30m/S1 ladder** |

The monitor prints `panic <TIER> V15 … DDhi … …σ` on every line (CALM and ALERT). On **PANIC** the
velocity *itself* is the act-now signal — defend/exit immediately rather than waiting for a 30-min close
confirmation. Symmetric: tracks up-moves too (call-side risk) via the run-from-low axis.

## Caveats
- Recorder window (2 months) is calm; the **thresholds come from the 11-year 5-min history**, not the 2 months.
- Z15 needs ≥8 intraday points (early-session it's noisy) — velocity/depth axes carry it before ~10:00.
- This sizes *urgency*, not direction — pair with the CPR/S1 level ladder for the trade decision.
