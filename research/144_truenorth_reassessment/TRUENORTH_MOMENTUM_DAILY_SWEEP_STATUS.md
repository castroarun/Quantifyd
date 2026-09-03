# True North (Momentum-30 Sub-Selection) Re-Assessment — Gate Bake-off + Slots + Exit Sweep, After-Tax

STATUS: DONE — **VERDICT: CONCLUDED, INCUMBENT STANDS (no change adopted).** See `results/RESULTS.md`.

## 1. The Ask

**What Arun asked:** "all the optimizations we did or studied today [on the Open Alpha
ATH-breakout system, research/142], let's revisit our momentum system and do the same."

**What we're actually testing:** Apply the research/142 (Open Alpha) re-assessment program to
the deployed True North momentum book (`services/momentum_paper.py`, research/62 winner):

1. **Data hygiene** — verify the momentum engine's indicators (gate SMA, Donchian lows, RS
   ranks) are NaN-robust and its input series (esp. the NIFTYBEES gate series) are current,
   post the 2026-01-15 phantom-holiday-row purge.
2. **Gate bake-off** — the deployed NIFTYBEES-100SMA weekly liquidate-all gate was inherited
   from research/41, never swept. Sweep gate SERIES × CONSTRUCTION × ACTION × CHECK-FREQUENCY.
3. **Holding-count (slots) sweep** — n_hold 5/8/10/12/16, buffer = round(2.75 × n).
4. **Exit/stop combos** — Donchian 10/15/20/25/none, ATR-trail, per-stock SMA-trail, jointly
   with the gate variants (OA lesson: exits tuned under one gate can flip under another).
5. **After-tax adoption** — any recommended change must win AFTER 20% STCG / 12.5% LTCG,
   with idle-cash yield 5–6.5% p.a. modeled (the gate parks the whole book in cash).

The incumbent stands unless beaten by the pre-declared margin. "No change" is a valid outcome.

## 2. The Base — the incumbent (deployed spec, `services/momentum_paper.py` CFG)

- **Universe:** Nifty-200. Backtest uses the survivorship-free PIT proxy: top-200 by
  trailing-6-month (202-session) median traded value, ≥75 sessions of data, ETFs/index
  series excluded. (Deployed live uses the OFFICIAL constituent list — stated divergence;
  the official list is not reconstructable historically.)
- **Score:** 6m (126d) + 12m (252d) price-relative-strength vs NIFTYBEES, 50/50 blend
  ("rsblend"). Rank descending.
- **Hold:** top-8 equal-weight, 100% invested when risk-on.
- **Buffer:** keep a holding while it stays within rank 22 (= round(2.75×8)); else evict and
  replace with best un-owned.
- **Rebalance:** MONTHLY, last trading day, at close. NO trimming of kept winners (deployed
  `live_rebalance_trim=False`): new names sized toward NAV/n from available cash only.
  (NOTE: research/62's backtest re-equalized all names monthly — this engine follows the
  DEPLOYED convention instead, so the incumbent is re-baselined here; all comparisons are
  within-engine.)
- **Gate (incumbent):** WEEKLY (last trading day of week, at close) — NIFTYBEES < 100-day
  SMA → liquidate ALL to cash; redeploy at the next month-end once back above.
- **Stop:** DAILY — holding closes below its own prior-15-day low (Donchian-15, excludes
  today) → exit that stock to cash.
- **Cash:** idle/risk-off cash earns 6.5% p.a. (liquid fund). Sensitivity at 5%.
- **Costs:** 0.3% round-trip (0.15% per side on traded value).
- **Tax:** on realization — 20% STCG if held <365 days, 12.5% LTCG if ≥365 days, deducted
  from cash at sale (LTCG ₹1.25L exemption ignored = conservative). Unrealized terminal
  gains untaxed in both arms.

**Windows:** W0 full 2006-04→2026-09 · WA 2012-01→2026-09 (all gate series defined; primary
comparison window) · W1 2016-06→2019-12 · W2 2020-01→2026-09. Window stats are slices of the
single full-period NAV (positions carry across boundaries).

## 3. Pre-registered ranking metric + adoption rule (BINDING, declared before any run)

- **Primary metric:** net-of-cost, net-of-TAX CAGR on window WA, subject to net MaxDD on WA
  no worse than the incumbent's by >3pp. Tie-break: net-tax Calmar on WA.
- **Adoption rule:** a challenger is recommended ONLY if (a) its median net-tax CAGR across
  the 12 rebalance-offset runs beats the incumbent's median by >1.0pp; (b) its net-tax
  Calmar ≥ incumbent's on BOTH W1 and W2; (c) it sits on a parameter plateau (its grid
  neighbours also beat the incumbent), not a lone peak.
- Gross figures reported alongside net everywhere.

## 4. Plan — variant grid

| Phase | Axes | Cells |
|---|---|---|
| A Gate bake-off | series {NIFTYBEES, NIFTY50, NIFTY500, NIFTYMIDCAP150, NIFTYSMLCAP250} × construction {SMA100/150/200, EMA100/150/200, XO 50-200, XO 20-100, DD from 252d-high 8/10/12/15%, 63d-mom<0, 126d-mom<0} + NO-gate; action=cash, freq=weekly, N8/Donch15 fixed | 71 |
| B Action × frequency | top-2 gates from A + incumbent gate × action {cash, block-new, halve} × freq {daily, weekly, monthly} | ≤27 |
| C Slots × exits | ≤3 gate variants × n {5,8,10,12,16} (buffer=round(2.75n)) × exit {Donch 10/15/20/25, none, ATR(20)×3 trail, SMA50-trail, SMA100-trail} | ≤120 |
| D Robustness | finalists (≤8, incumbent always included) × 12 rebalance-day offsets (k trading days before month-end, k=0..11 — the deterministic analogue of OA's seed ensemble) + sensitivities (cash 5%, cost 0.5%) | ≤110 |

Every cell runs twice (tax-off = gross-of-tax net-of-cost, and tax-on). ~660 runs, ~1–4 s
each after precomputation (rank lists per rebalance date are gate/exit-independent and cached;
gate booleans and exit matrices vectorized).

**Seven deadly sins — controls:**
- Look-ahead: all ranks/indicators use data ≤ d only; Donchian/SMA/ATR matrices shift(1);
  gate + fills act on the same close the live engine acts on (15:15/14:45 pre-close).
- Survivorship: PIT traded-value universe (not today's constituent list).
- Overfitting/multiple testing: metric + adoption margin pre-registered above; plateau
  required; 12-offset distribution reported (median [min..max]); incumbent default-wins.
- Cost neglect: 0.3% RT baked into every figure; 0.5% stress on finalists; taxes modeled.
- Regime dependence: W1/W2 validation both required; W0 includes 2008 for NIFTYBEES-series
  gates; per-year table for finalists.
- Correlation/single-factor: single momentum book — concentration risk reported via n-sweep
  DD; no new claim.
- Capacity/shortability: long-only Nifty-200 large caps at ₹20L — not binding; stated.

**Falsification plan:** if no challenger clears the adoption rule, the verdict is
"incumbent stands — no change", and that is a welcome outcome.

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-03 20:15 | Grounding read: playbook, momentum_paper.py, r62/r75 RESULTS | rules locked |
| 2026-09-03 20:28 | Data hygiene on VPS DB | 2026-01-15 phantom purge CONFIRMED (0 rows). Residual phantom-signature rows: 2025-03-18 (766 syms), 2024-01-15 (348 syms) — genuinely-untraded small-caps on normal trading days (RELIANCE/TCS/NIFTYBEES fine) → benign for Nifty-200. NIFTYBEES current to 2026-09-03; NIFTY50/500/MIDCAP150/SMLCAP250 exist 2011→2026-08-28 (4 sessions stale, acceptable for gate research). Engine audit: `_gate_risk_off` and `_donchian_low` dropna() before rolling (NaN-robust); `_rs_basket` ffills (robust); OK. |
| 2026-09-03 20:35 | STATUS doc written (this file), sections 1–4 locked | before any backtest run |
| 2026-09-03 20:50 | Engine `scripts/tn_sweep.py` on VPS; smoke run | Incumbent (2012+): gross-of-tax 25.3% CAGR / −17.1% DD (DD matches r62's −17.0); no-gate-no-stop: 26.7% / −52.8% (gate value confirmed). avg_inv 0.43 investigated — NOT a bug: gate risk-off ≈32% of days + month-end-only redeploy + Donchian idle slots. |
| 2026-09-03 20:58 | TAX MODEL FIX before any ranking run | v1 taxed every winner, never offset losses → biased against stop-heavy configs (723 stop-outs are mostly losses). Replaced with Indian FY netting: STCL offsets STCG then LTCG, LTCL offsets LTCG only, settled each Apr 1, no carry-forward (mildly conservative). Incumbent net-tax WA CAGR 20.9% (drag −4.4pp, consistent with r62). |
| 2026-09-03 21:02 | Phase A LAUNCHED (71 gate cells × gross/tax) | /tmp/tn_phaseA.log |
| 2026-09-03 21:10 | Phase A DONE — **incumbent gate WINS the bake-off** | NIFTYBEES-SMA100 weekly/cash is the top net-tax Calmar (0.88) of all 71 cells; only NIFTY50-SMA100 (near-identical twin) and NIFTYBEES-XO20/100 (lower CAGR) also pass the DD constraint. DD8-15 / momentum-negative / crossover gates and NO-gate all fail DD (−33..−52%). Only NIFTYBEES-series gates protect 2008 (index series start 2011). |
| 2026-09-03 21:20 | Phase B DONE (54 runs) | Net-tax WA: block-new-only 21.12/−22.05/Cal 0.96 and cash+monthly-check 21.24/−21.61/Cal 0.98 both edge the incumbent (20.90/−23.67/0.88) on every metric incl. W1+W2 — but by +0.2-0.3pp CAGR, below the 1pp adoption margin. Halve-exposure: more CAGR, DD −33 → fails. Check-frequency: daily liquidation is WORSE (whipsaw); block is frequency-invariant as expected. |
| 2026-09-03 21:30 | Phase C DONE (240 runs) | Donch15 = best-DD exit at every n (r62 confirmed); ATR×3 and SMA-trails clearly worse. **n5+donch15 challenger: net-tax 25.44/−25.52/Cal 1.00, beats incumbent on W1 AND W2.** Wider books n≥10 lose CAGR and W1 collapses (2016-19 dead zone). |
| 2026-09-03 21:35 | ARUN PRIORITY (via coordinator): (1) deep-dive block-new-only vs liquidate-all incl. tax churn + offset bands; (2) 50-50 blend with OA adopted spec (30 seeds) for both gate actions, corr + blend stats | folded into Phase D + new `tn_blend.py` |
| 2026-09-03 21:40 | Phase D (6 finalists × 12 offsets × gross/tax + sensitivities) + blend LAUNCHED | /tmp/tn_phaseD.log, /tmp/tn_blend.log |

## 6. Crash recovery (resume without Claude)

- Everything runs on the VPS in `/home/arun/quantifyd/research/144_truenorth_reassessment/`.
- Check what's running: `ps aux | grep tn_sweep` ; logs at `/tmp/tn_phase*.log` (tail them).
- Results accumulate incrementally in `results/phaseA_gates.csv`, `results/phaseB_actions.csv`,
  `results/phaseC_slots_exits.csv`, `results/phaseD_robustness.csv` — one row per completed
  cell; reruns skip completed labels automatically.
- Resume any phase: `cd /home/arun/quantifyd && setsid nohup venv/bin/python -u
  research/144_truenorth_reassessment/scripts/tn_sweep.py <A|B|C|D> > /tmp/tn_phase<X>.log 2>&1 &`
- Rank-list caches: `results/ranks_off*.pkl` (safe to keep; delete only to force recompute).
- Do NOT touch: `services/momentum_paper.py`, `backtest_data/momentum_paper.db`, crontab,
  any live engine. This research changes NOTHING deployed.
- Safe to inspect: everything under `research/144_truenorth_reassessment/`.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `TRUENORTH_MOMENTUM_DAILY_SWEEP_STATUS.md` | this live-status doc | yes |
| `scripts/tn_sweep.py` | engine + all phase runners | yes |
| `results/phase[A-D]_*.csv` | incremental per-cell results | yes (small) |
| `results/ranks_off*.pkl` | monthly ranking caches | NO — gitignored |
| `results/RESULTS.md` | final findings + verdict | yes |

## 8. Findings

Full write-up in `results/RESULTS.md`. Headlines:

1. **Incumbent gate WON its own bake-off** (71 cells) — best net-tax Calmar of all; OA-style
   drawdown gates are wrong for this book (−34..−47% DD).
2. **Offset-robustness reversed offset-0 rankings** — block-new-only and cash-monthly looked
   better at offset 0 but are a wash across the 12 rebalance-day offsets; the incumbent has
   the BEST worst-offset DD (−28.3) of all finalists.
3. **Two challengers beat the +1pp after-tax margin** (n5/D15 +2.0pp, n8/D20 +1.9pp) but both
   fail the W2-Calmar condition and carry −32..−34% worst-offset DDs → NOT adopted.
4. **Block-new-only tax saving quantified: only ~0.22pp/yr** (FY loss netting absorbs the
   mass-realization hit).
5. **OA 50-50 blend: 27.4% after-tax CAGR at −16.4% DD (Calmar 1.68)**; the incumbent
   liquidate-all TN leg is the (slightly) better blend partner — corr 0.40 vs 0.43 daily.
6. Data hygiene: 2026-01-15 purge confirmed; residual benign phantom rows 2025-03-18 /
   2024-01-15 (small-caps only); live gate on the SMA100 knife-edge this week (expected
   behavior, weekly check).

Closing log: 2026-09-03 22:0x Phase D DONE (156 runs) + blend DONE (30 seeds × 2 TN legs);
22:1x RESULTS.md written, verdict CONCLUDED — incumbent stands; committed + pushed.
