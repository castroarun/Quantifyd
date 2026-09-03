# True North Rules on the Open Alpha Universe — Adversarial Revalidation of the r/62 Universe Rejection

STATUS: DONE — **VERDICT: CONCLUDED — full-universe TN REJECTED; r/62's rejection REVALIDATED.** See `results/RESULTS.md`.

## 1. The Ask

**What Arun asked:** "take up TN as is, extend to the same universe of stocks and same criteria
like the OA's and come up with the numbers, assessments."

**What we're actually testing:** keep True North mechanics EXACTLY as deployed (rsblend 6m+12m
RS vs NIFTYBEES, top-8 equal-weight, keep-buffer rank 22, monthly no-trim rebalance, NIFTYBEES
SMA-100 weekly liquidate-all gate, per-stock 15-day-low Donchian daily stop, 0.3% RT cost,
cash 5-6.5%, FY-netted 20% STCG / 12.5% LTCG) and swap ONLY the universe:

- **U-200 (control):** PIT top-200 by traded value — the r/144 incumbent baseline.
- **U-500:** PIT top-500 by traded value (the size gradient midpoint).
- **U-OA (the ask):** the Open Alpha universe — ALL NSE dailies with 20-day-median traded
  value ≥ ₹5cr as of t-1 (`bluesky_replay.TV_FLOOR`), ETFs excluded (`ETF_RE` + our ETF set),
  NO market-cap floor.

Then: does full-universe TN add anything the TN+OA 50-50 pair doesn't already have?
(OA already harvests the small-cap breakout side — full-universe TN may just be a worse OA
with monthly lag. Measured via holding-name overlap with OA positions + correlations + blends.)

## 2. The prior to confront (stated UP FRONT — this is an adversarial revalidation)

**research/62 CONCLUSIVELY REJECTED wider/lower-cap momentum** (`results/p2_universe_bands.csv`,
participation-based impact model at AUM ₹1cr/10cr/50cr):

- top500: gross 36.6% CAGR but −25.4% DD (Calmar **1.44 vs top200's 2.21** — MORE return, much
  worse risk-adjusted) and at even **₹1cr AUM it collapses to 11.7% / −66.3%** with
  participation up to **1081% of daily traded value**. At ₹10cr: −17% CAGR / −97.6% DD.
- small band (250-500): gross 33.5%/−30.9%; at ₹1cr: **0.1% / −82.3%**.
- Capture analysis (p2c): the mega-runners (SIGIND +5996%, BCG +4991%, DUCON, VIVIDHA) were
  never in top-200 and had **capture_ratio ≈ 0.00-0.02 even when held** — monthly momentum
  ranking catches lottery small-caps too late to matter.

**Null hypothesis H0: r/62 was right — full-universe TN is worse after costs, and uninvestable
at size.** The burden of proof is on the wider universe. If it confirms the rejection, that is
the expected and welcome outcome.

Differences from r/62's test (stated): r/145 uses the deployed-faithful r/144 engine (no-trim,
FY-netted tax, exact costs) with FLAT cost tiers 0.3/0.5/0.75% RT plus an explicit
capacity/participation note, instead of r/62's AUM-scaled impact model; and TN keeps its gate
(r/62 P2 bands were tested in that engine's own config).

## 3. Pre-registered metric + verdict rules (BINDING, declared before any run)

- **Primary metric:** net-of-cost net-of-TAX CAGR on WA (2012-01→now), DD constraint within
  3pp of the U-200 control, judged at 0.3% RT — **and the story must survive 0.5% RT**
  (small-caps do not fill at large-cap cost).
- **Robustness:** 12 rebalance-day-offset bands (median [min..max]) per universe; W1
  (2016-06→2019-12) and W2 (2020→now) reported; per-year table.
- **Capacity gate:** report held-name 20d-median traded value (p50/p10/min over time) and the
  implied max book size at ≤10% participation per name; a "winner" that cannot absorb the
  live book size (₹7.7L today, ₹20L design) at sane participation is NOT a winner.
- **Blend-value test:** U-OA TN earns a portfolio slot ONLY if replacing standard TN in the
  50-50 OA pair improves the pair's after-tax Calmar (or materially lowers correlation).
  Overlap with OA holdings measured (daily |TN∩OA|/8 across OA seeds).
- Gross AND net reported everywhere; verdict label from {NO EDGE / SIGNAL / STRATEGY /
  CONCLUDED} with the H0 outcome stated plainly.

## 4. Plan — grid

| Phase | Cells |
|---|---|
| sweep | 3 universes × (12 offsets × tax{0,1} + rt {0.5%,0.75%} × tax{0,1} + cash5% tax1) ≈ 87 runs |
| capacity | per-universe held-name TV distributions + universe-size time series (no new backtests) |
| blend | OA adopted spec 10 seeds after-tax × {U-OA TN, U-200 TN} blends + corr matrix + OA-holdings overlap (3 seeds) |

Seven-sins controls inherited from r/144 (same engine, same STATUS discipline); the specific
new risks here are **survivorship in the small-cap tail** (Kite lists current instruments — the
pre-2015 small-cap universe is survivor-flattered, ~528 syms in 2006 vs 2,300 now: stated on
every result) and **cost realism** (cost tiers + capacity gate above).

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-03 22:3x | r/62 prior read + quantified; STATUS written (sections 1-4) before any run | — |
| 2026-09-03 22:4x | `scripts/tn_universe.py` written (UCtx universe override on the r/144 engine); sweep launched | /tmp/tn145_sweep.log |

## 6. Crash recovery

- VPS `/home/arun/quantifyd/research/145_truenorth_full_universe/`; engine reused from
  `research/144_truenorth_reassessment/scripts/tn_sweep.py` (imported, not copied).
- `ps aux | grep tn_universe`; logs `/tmp/tn145_*.log`.
- Incremental CSV `results/universe_sweep.csv` (one row per run, label-keyed, reruns skip done);
  `results/blend_universe.csv`, `results/capacity.csv`, `results/peryear.csv`.
- Resume: `cd /home/arun/quantifyd && setsid nohup venv/bin/python -u
  research/145_truenorth_full_universe/scripts/tn_universe.py <sweep|capacity|blend>
  > /tmp/tn145_<phase>.log 2>&1 &`
- Rank caches `results/ranks_*.pkl` (safe; delete to force recompute). Do NOT touch anything
  deployed. Safe to inspect everything under research/145.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| this STATUS md | live status | yes |
| `scripts/tn_universe.py` | universe override + all three phases | yes |
| `results/universe_sweep.csv` | per-run results | yes |
| `results/capacity.csv`, `results/blend_universe.csv`, `results/peryear.csv`, `results/nav_*.csv` | analytics | yes |
| `results/ranks_*.pkl` | rank caches | NO — gitignored |
| `results/RESULTS.md` | final verdict | yes |

## 8. Findings

Full write-up in `results/RESULTS.md`. Headlines (all net-of-tax, 12-offset medians, 2012→now):

1. **U-OA (full universe): 22.9% CAGR but −35.6% DD** vs U-200's 20.7%/−25.1% — fails the
   pre-registered DD constraint by ~7.5pp; Calmar 0.66 vs 0.88; W1 (2016-19) collapses to
   7.9%; worst-offset DD −41.7%. **U-500 strictly dominated** (19.3%/−34.9%).
2. **Offset-0 single paths overstate wider universes by 4-8pp** (U-OA 27.5 vs median 22.9);
   U-200 sits on its median — wider books are far more rebalance-day-dependent.
3. **Capacity:** U-OA executable at ₹20L (~0.5% participation) but ceiling ~₹7.5cr book vs
   ₹17cr for U-200; U-500 capacity-broken (worst p10 held-name TV ₹0.2cr). The nominal ₹5cr
   floor makes the early universe tiny (112 names in 2006 → 954 in 2026): the 20-year run
   mixes two different universes.
4. **Blend-value test FAILED:** OA + U-OA TN = 30.4%/−20.6/Calmar 1.47 vs OA + U-200 TN
   27.2%/−16.4/**1.65**. The U-OA leg re-imports OA's small-cap beta (name overlap median 20%,
   p90 50%; TN U-200 vs U-OA corr 0.75 daily). More return from the pair = weight OA up, not
   blur TN into a slower OA.
5. Flat cost tiers graceful (~1.1pp per +25bps to 0.75%) — the kill is risk/regime/capacity,
   not flat cost. H0 (r/62 was right) CONFIRMED.

Closing log: 2026-09-03 22:5x sweep DONE (87 runs) → capacity DONE → blend DONE;
23:0x RESULTS.md written, committed + pushed.
