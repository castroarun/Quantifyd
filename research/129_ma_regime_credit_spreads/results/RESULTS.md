# research/129 — MA/EMA/RSI/Stochastic-Regime Directional Credit Spreads — RESULTS

**VERDICT: NO EDGE — family CONCLUDED at G1 (price-only kill test). Do not build
options machinery on directional MA/EMA/RSI/stochastic regime states.**

The test: a credit spread sells one tail. For the regime to be monetizable, the
in-state probability of the SOLD tail over the 24-session hold must shrink vs
unconditional (gate: >=20% relative, per-year stable). On 198,582 stock-days,
81 F&O names, 2016-2026:

- Unconditional: P(fwd<-2.5%)=30.8%, P(fwd>+2.5%)=45.5%, mean fwd +1.92%.
- EVERY MA/EMA level and cross GROWS its sold tail in-state (+0.4% to +5.4% rel).
- RSI and stochastic states are noise (-0.5% to +4.1% rel; best = 0.5% shrink,
  unstable 7/11 years). Nothing approaches the gate.
- **The premise inverts: BEAR states carry HIGHER forward drift than BULL states**
  (+2.1-2.25% vs +1.72-1.94%) — 24-day mean reversion. Selling put spreads only
  above the MA picks the weaker state; bear call spreads get overrun MORE below it.

Consistent with r/91 (20/200 SMA = survivor drift), MQ technical filters (all
subtract value), r/56 (directional options NO NET EDGE). Fourth kill of this
family — treat as settled.

Honest caveats: overlapping windows (t downscaled by sqrt(24)); today-universe
survivorship inflates the unconditional drift but affects both states equally;
event-driven (cross-day) entries not separately tested — cross-day subsets are a
strict subset of states this weak, and r/91 tested crosses directly (dead).

Next levers: none for this family. The queued idea from the same conversation —
IV-breach EXIT on the r/127 book — is a separate stop-family test (prior:
skeptical, B3 twin test) using results/iv_daily.csv.

Reproducibility: scripts/run_g1_probe.py (27s, read-only), results/g1_probe.txt.
