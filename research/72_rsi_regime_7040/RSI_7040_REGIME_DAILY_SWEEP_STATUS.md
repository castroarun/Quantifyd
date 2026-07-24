# RSI 70/40 Momentum-Regime Timing — RELIANCE base → Nifty universe expansion

**STATUS: DONE** — Verdict **SIGNAL, not a clean STRATEGY** (see `results/RESULTS.md`). Single-name RSI 70/40 = NO EDGE; filters don't rescue it; diversified it's a real OOS-robust momentum-breadth signal but faces a return/DD frontier (2.8× index at ~index-level DD, OR 1.5× at lower DD — not both), flattered by survivorship + small-cap capacity, and converges on the fund's existing regime-gated momentum book (research/41/62) with a cruder entry.

_(originally RUNNING)_ — orchestrator (main session) + fan-out optimization agents. Runs on VPS `94.136.185.54` (canonical host + `market_data.db`). Snapshot: VPS DB 5.24 GB, RELIANCE daily 2000-01-03 → 2026-07-07, benchmark NIFTYBEES 2005 → 2026.

Master orchestrator: this session. Recovery doc: THIS FILE. Findings: `results/RESULTS.md`.

---

## 1. The Ask

**What you asked (verbatim intent):** "Whenever RSI breaches and closes at/above 70 on the daily timeframe, enter that stock at that day's closing price; hold until RSI breaches and closes below 40 on a daily closing basis, then exit at that day's close. Fully RSI-based; fine-tune the numbers; add filters (CPR, MAs, stochastics, Donchian, weekly-RSI conjunction, ADX for trend, Supertrend). Deploy multiple agents with a master orchestrator that maintains a recovery doc + a results MD. Aim: a *reliable* system beating the Nifty index return by **at least 50%** while keeping **lower drawdowns**. Base on RELIANCE, expandable to Nifty 200 / midcap / smallcap."

**What we are actually testing:**
A long-only, close-basis, daily momentum-**regime** timing system. Signal = RSI(14) close ≥ `ENTRY` → go long at that close; hold until RSI(14) close < `EXIT` → sell at that close; flat (cash) otherwise. Base instrument = **RELIANCE**. The success bar is **risk-adjusted outperformance of a real, tradeable Nifty benchmark (NIFTYBEES)**, net of costs:

> **Success gate:** net-of-cost strategy total-return (or CAGR) ≥ **1.5×** NIFTYBEES over the same window, **AND** MaxDD strictly **lower** than NIFTYBEES buy-&-hold. Rank by **Calmar** (return per unit drawdown) and Sharpe, not raw CAGR. A config that beats the index only by taking *more* drawdown FAILS.

We also always compare vs **RELIANCE buy-&-hold** — the timing overlay must justify itself against simply owning the stock, or the "edge" is just single-name beta.

## 2. Economic hypothesis (G0)

- **Mechanism:** momentum persistence / under-reaction. A daily RSI ≥ 70 marks a decisive momentum thrust; behavioural under-reaction + trend-following flow tend to extend such moves. Exiting only when RSI decisively breaks (< 40) rides the trend and steps aside when momentum has clearly rolled over, sidestepping the worst of mean-reverting drawdowns.
- **Counterparty:** disposition-effect sellers who cut winners too early, and anchoring investors who fade strength. Their early supply is the trend we ride.
- **Why it might persist / decay:** momentum is one of the most robust cross-market anomalies, but crude fixed RSI thresholds are widely known → any edge is likely modest and regime-dependent (momentum crashes in sharp V-reversals). **Falsification (decided up front):** if on RELIANCE the *net* result does not beat RELIANCE buy-&-hold on a risk-adjusted basis AND the threshold response is a lone spike (not a monotonic/plateau region), the single-name timing idea is **NO EDGE** — pivot to portfolio-breadth only or shelve.

## 3. The Base — mechanics (locked)

- **Indicator:** RSI, Wilder's smoothing (matches TradingView), on daily **close**. Default length 14.
- **Entry:** the day RSI closes ≥ `ENTRY` (default 70) and we are flat → buy at that day's close.
- **Exit:** the day RSI closes < `EXIT` (default 40) and we are long → sell at that day's close.
- **Direction:** long-only. Flat = 100% cash (idle-cash return modelled at 0% by default; a cash-yield sensitivity is a later lever).
- **Sizing:** all-in single name for the RELIANCE base (100% of NAV when in position). Portfolio construction (equal-weight, position caps) is the universe-expansion phase.
- **Costs:** parametric round-trip cost in bps (default **15 bps** = 0.15% per full round trip: STT sell 0.1% + slippage + charges for CNC delivery). Report **gross AND net**; run a cost-sensitivity (0 / 15 / 30 / 50 bps). STCG-tax note where holds < 1yr dominate.
- **Universe/period:** RELIANCE, primary window **2015-01-01 → 2026-07-07** (matches NIFTY50 index history) + a full **2005 → 2026** robustness window on NIFTYBEES.
- **Benchmark:** **NIFTYBEES** buy-&-hold (long history, tradeable, playbook-preferred). Secondary: RELIANCE buy-&-hold (does timing beat owning the name?).
- **Success criterion:** see §1 gate. Rank metric = **Calmar**, tiebreak Sharpe; hard constraint MaxDD < benchmark.

## 4. Plan — variant grid + phases (stage-gated)

**Phase A — baseline + threshold/length sweep (G1/G2, CHEAP, run first):**
- Base: RELIANCE RSI14 e70/x40 (the literal ask), gross + net, per-year, vs both benchmarks.
- Sweep axes: `ENTRY ∈ {60,65,70,75,80}` × `EXIT ∈ {30,35,40,45,50}` × `RSI_LEN ∈ {7,14,21}` = **75 cells**. Keep only EXIT < ENTRY. Look for monotonic/plateau response, not a lone peak.
- **Gate G1→B:** does *any* cell beat RELIANCE B&H risk-adjusted (Calmar) with lower DD? If nothing clears, single-name timing is weak → go straight to portfolio breadth (Phase C) which may still work.

**Phase B — filter overlays (agents, marginal-add analysis on the best base):**
Each filter is a *gate on new entries* (and optionally an alternate exit), tested as a marginal add so we can isolate its value:
- SMA/EMA trend filter (enter only if close > SMA200 / 200-vs-50 regime).
- ADX(14) > {20,25} trend-strength gate.
- Weekly-RSI conjunction (weekly RSI > 50 / > 60 to confirm higher-TF momentum).
- Supertrend(10,3) regime agreement (only long when Supertrend bullish; Supertrend-flip as alt exit).
- Donchian(20/55) breakout confirmation / Donchian exit.
- Daily/weekly CPR width or position (narrow-CPR skip, per research/67 sign findings).
- Stochastic confirmation.

**Phase C — universe expansion + portfolio (agents):**
- Apply best base(+filter) across **Nifty 50** then **Nifty 200** constituents.
- Portfolio: equal-weight across names currently long, cap N positions, cash when few signals. Measure portfolio CAGR/DD/Calmar vs NIFTYBEES. This is where "beat the index by 50% with lower DD" is truly decided (diversification cuts single-name DD).
- Optional: midcap/smallcap extension if the large-cap portfolio clears G4.

**Phase D — robustness (agents, G3/G4):** OOS/walk-forward split, per-year stability, parameter sensitivity/monotonicity, cost-stress (+50%), adversarial kill (drop top contributors, alt window), multiple-testing discount, capacity/liquidity note. Then tearsheet + publish to `/app/backtest/<slug>` + RESULTS.md.

**Grid cell counts:** Phase A = 75; Phase B ≈ 6 filter families × ~4 settings each (marginal) ≈ 30–40 runs on best base; Phase C = |universe| × best-configs (Nifty50 ≈ 50 names × few configs; Nifty200 ≈ 200). All single-name daily → milliseconds/run; compute is trivial, the discipline (robustness, honesty) is the work.

---

## 5. Status (live log)

**State header:** Phase A DONE — single-name RELIANCE RSI-regime = **NO EDGE** (fails gate hard). Pivoting to Phase C portfolio (the real lever) + Phase B filters via fan-out agents.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-07-07 | Study framed; folder + STATUS-MD created (§1-4) | research/72; success gate = net ≥1.5× NIFTYBEES & lower DD, rank Calmar |
| 2026-07-07 | Verified VPS data | RELIANCE 6591d 2000-2026; NIFTYBEES 5330d 2005-2026; NIFTY50 idx 2015+ |
| 2026-07-07 | Baseline RELIANCE 70/40 run | net CAGR **4.19%** vs NIFTYBEES 10.95% & RELIANCE B&H 17.13%; DD −23% (lower, but from sitting in cash). FAIL. |
| 2026-07-07 | Phase A sweep 75 cells DONE | `results/phaseA_sweep.csv`. **Best beat-ratio = 0.97× — NOT ONE cell beats the index**; all lose to RELIANCE B&H; response noisy (no monotonic plateau). **Verdict: single-name RSI-regime timing NO EDGE.** |
| 2026-07-07 | Fan-out launched | Agent-P (Nifty50/200 slot-portfolio), Agent-F (filter overlays MA/ADX/wRSI/ST/Donchian). Lists: nifty50_official.csv, nifty200_official.csv on VPS. |
| 2026-07-07 | Agent-F DONE — filters NO EDGE | Only SMA200/wRSI50 add ~+1pp CAGR; ADX/ST/Donchian just cut exposure. Best filtered basket 2.74% vs index 10.95%. `results/phaseB_filters.csv` |
| 2026-07-07 | Agent-P DONE — portfolio | 1/40 cells pass gate: nifty50 N20 60/30 net 16.76% / −23.85% / Calmar 0.70 / 1.53×. DD-half easy, return-half fails. `results/phaseC_portfolio.csv` |
| 2026-07-07 | Phase D robustness DONE | Broad-533 universe: net 29.3% but DD −45%; edge STRONGER OOS (2021-26 net 51.8%, not overfit); param plateau. Return/DD tension structural. `results/phaseD_robustness.json` |
| 2026-07-07 | Phase E regime-gate DONE | 200DMA exit-all gate: broad net 30.4% / DD −35.3% → 1 config technically passes both goals but razor-thin DD, fails at 30bps & OOS. `results/phaseE_regime_gate.json` |
| 2026-07-07 | **CONCLUDED — RESULTS.md written** | Verdict SIGNAL not STRATEGY; converges on research/41/62 momentum book. INDEX + TODO + memory updated. |

## 6. Crash Recovery — resume without the assistant

- **Everything runs on the VPS** `arun@94.136.185.54:/home/arun/quantifyd/research/72_rsi_regime_7040/`.
- Engine: `scripts/rsi_regime_engine.py` (self-contained; reads `backtest_data/market_data.db`). Sweep runner: `scripts/run_phaseA_sweep.py`.
- To re-run baseline: `ssh arun@94.136.185.54 'cd /home/arun/quantifyd && python3 research/72_rsi_regime_7040/scripts/rsi_regime_engine.py --symbol RELIANCE --entry 70 --exit 40 --rsi_len 14'`.
- To re-run/resume the sweep: `... run_phaseA_sweep.py` — it **skips cells already in** `results/phaseA_sweep.csv` (done-set on `label`), so re-running continues safely.
- Check progress: `wc -l results/phaseA_sweep.csv`; `tail results/*.log`.
- Safe to inspect: all `results/*.csv`, `*.md`, `*.log`. Do NOT edit the CSVs by hand mid-run.

## 7. Files (output map)

| File | Purpose | Committable? |
|---|---|---|
| `scripts/rsi_regime_engine.py` | Core parametric RSI-regime backtest engine | yes |
| `scripts/run_phaseA_sweep.py` | Threshold × length sweep runner (incremental CSV) | yes |
| `RSI_7040_REGIME_DAILY_SWEEP_STATUS.md` | This recovery doc | yes |
| `results/baseline_reliance.json` | Baseline RELIANCE 70/40 full metrics + per-year | yes |
| `results/phaseA_sweep.csv` | Per-cell sweep results | yes (small) |
| `results/*.log` | Run logs | yes (small) |
| `results/RESULTS.md` | Final honest verdict | yes |

## 8. Findings (final)

**Verdict: SIGNAL, not a clean STRATEGY for the stated dual goal.** Full detail in `results/RESULTS.md`.

1. **Single-name RSI 70/40 on RELIANCE = NO EDGE.** Net 4.2% CAGR vs index 10.9% / stock B&H 17.1%.
   0/75 threshold cells beat the index; RSI≥70 enters late, RSI<40 exits after the drop.
2. **Filters don't rescue it.** Only SMA200/weekly-RSI add ~1pp; ADX/Supertrend/Donchian just park you in
   cash (Calmar illusion). 0 filtered configs beat the index.
3. **Diversified it becomes a REAL momentum-breadth signal** — OOS-robust (stronger 2021-26), param
   plateau — but hits a **return/DD frontier**: broad universe = 2.8× index CAGR at ~index-level DD (−45%);
   blue chips = 1.5× at lower DD (−24%). Not both at once from pure RSI.
4. **200DMA regime gate** gets one config to *technically* pass both (broad exit-all: 2.78×, −35.3% <
   −36.3%) but the DD margin is razor-thin, it fails at 30bps cost and out-of-sample.
5. **Dominant caveats:** survivorship (both universes are survivors) + capacity (high return = illiquid
   small/midcaps; research/62 already showed lower-cap momentum is a gross-only mirage at size).
6. **Convergence insight:** pushed to where it works, this *is* the fund's existing regime-gated momentum
   book (research/41 midcap RS, research/62 Momentum-30, Calmar ~1.7, live paper) — with a **cruder entry**.
   RSI-regime adds no new alpha over what we already run.

**Recommended next levers:** (a) liquidity-floored capacity-aware universe (honest test — likely collapses
the 30% toward ~16%); (b) vol-targeting sizing to force DD below index; (c) improve the existing momentum
book rather than productionise RSI-regime.
