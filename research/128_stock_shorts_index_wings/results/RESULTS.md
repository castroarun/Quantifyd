# research/128 — Stock Shorts + INDEX Wings (NIFTY/BNF) — RESULTS

**VERDICT: CONCLUDED — index wings REFUTED. Stock-side tails are IDIOSYNCRATIC,
so index wings charge premium (0.7-1.9% of spot/trade) and deliver ~zero tail
protection. r/127 C1 keeps its per-stock 7% wings.**

Matched 604 entries (same stock+expiry), 2016-2026, net of costs:
NAKED +0.839%/tr (t 6.8) p01 -11.2%, worst -56.8% (ADANIENT Jan-23);
NW7 +0.645 p01 -11.8; NW5 +0.576 p01 -11.7; NW3 +0.446 p01 -12.1;
C1 stock wings +0.276 p01 -3.1%, worst -4.6%.

Findings:
1. Index-wing tails == naked tails: the killer moves (ADANIENT -57%, TATAMOTORS
   -24%, HAL -19%) were single-stock gaps on non-crash index days — the hedge
   never pays. Every index-wing variant is strictly dominated by naked.
   Weekly/biweekly tenors moot: the flaw is WHAT is hedged, not how long.
2. Naked out-earns C1 3x per trade but is uninvestable: one trade can exceed
   the entire slot capital several times over (-57% of notional ~ -57% of a
   Rs20L book at paper sizing), and naked stock strangles carry ~2x margin
   (no same-underlying hedge benefit). Stock wings 2.45%/trade premium buys
   p01 from -11.6% to -3.1% — fairly-priced idiosyncratic-gap insurance.
3. Mechanism now PROVEN for r/127: its risk is idiosyncratic, strengthening
   the earnings-skip filter as the top remaining tail lever.

Honest caveats: fractional index units (lot-rounding unmodeled — immaterial to
the verdict); NAKED/NW books trade 1083 entries vs C1 628 (no stock-wing
liquidity gate) — matched-set comparison used for the verdict; margins modeled
not measured for naked.

Next levers: none for index wings. Earnings-calendar acquisition -> skip-test
on r/127 (queued). Reproducibility: scripts/run_g1.py, results/g1_analysis.txt.
