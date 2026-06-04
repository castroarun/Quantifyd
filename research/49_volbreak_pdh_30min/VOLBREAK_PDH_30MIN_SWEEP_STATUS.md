# Volume-MA + Prev-Day-High Breakout — 30-min Intraday Long (research/49)

STATUS: **DONE — VERDICT: NO EDGE.** Intraday dies to cost; positional looked good
(net +0.70R / PF 1.54) but the placebo kill shows it is **pure beta, not alpha**
(SIGNAL ≈ random-day BASELINE). See `results/RESULTS.md`.

> Follows `research/QUANT_RESEARCH_PLAYBOOK.md`. STATUS sections 1–4 written
> BEFORE any run, per `.claude/CLAUDE.md`.

---

## 1. The Ask

**What you asked (verbatim):** "Volume breakout stocks directional system — 30 min
timeframe. If volume of a stock 30 min candle exceeds its own 50 day MA and if the
same volume candle breaks the previous day's high, go long. You can consider RSI
levels as additional filter. For the exit criteria… assess all possible combinations
— fixed target or trailing stoplosses like supertrend variations."

**What we're actually testing (cleaned up):** On 30-min bars (built by resampling the
rich 5-min stock data, 2018–2026), does a LONG entry triggered when a single 30-min
candle's volume exceeds its own trailing 50-day volume MA **and** that same candle's
close breaks the previous day's high produce a net-of-cost positive expectancy,
across a representative liquid universe — and does any exit policy (fixed R-target,
ATR/Chandelier trail, **Supertrend** trail, hard SL, EOD) make the *intraday* version
tradeable? RSI is an optional gating axis.

## 1b. Prior art (DO NOT re-litigate — playbook §1, §11)

- **research/44 (prev-day range breakout):** prev-day-HIGH break long is **breakeven
  raw** across 374 names; reaches only **PF ~1.24 / +0.16R** with NR7-compression +
  high-beta + HTF-trend AND **swing (~3-day) holds. Intraday was WORSE than swing.**
- **research/40 (volsurge + PDR break):** vol>k×avg + PDR break (incl. 30-min) →
  top configs are tiny-n per-stock cherry-picks (n=15–22): SIGNAL, not STRATEGY.

**Implication:** this study's burden is to show the *specific* combination
(own-50d-volume-MA test + 30-min + intraday exits) does something 40/44 did not.

## 1c. Falsification criterion (decided BEFORE results — playbook §2)

ABANDON the intraday version if, after costs, no exit policy clears **either**:
(a) net expectancy ≥ **+0.16R/trade** AND PF ≥ **1.24** (research/44's bar), on
n ≥ 500 trades across ≥5 names; or (b) a clean monotonic dose-response in the
volume-MA multiple. If it fails both → verdict **NO EDGE / SIGNAL**, and the only
follow-up worth compute is a *swing-hold* variant (since 44 says swing ≫ intraday).

## 2. Economic hypothesis (G0)

Heavy volume confirming a break of the prior day's high = late/continuation flow
(under-reaction → momentum) and short-covering above a visible level. Counterparty:
prior-day-high sellers / stops; mean-reverters fading the gap. Decay risk: HIGH —
intraday breakout edges are heavily competed and eaten by cost+slippage (44/43 scars).

## 3. The Base (mechanics locked for G1)

- **Bars:** 30-min, resampled 6×5-min anchored 09:15 IST (reuse `33/data30.to_30min`);
  12 full bars/day, trailing 15:15–15:30 partial dropped.
- **Volume-MA test:** bar volume > `VOLMA` where `VOLMA` = trailing mean of 30-min
  bar volume over the last `W` bars, **shifted 1 bar (causal)**. Default W = 600
  (≈ 50 trading days × 12 bars) = the literal "own 50-day MA". (ASSUMPTION — flagged;
  becomes a grid axis: also test same-time-of-day seasonal MA.)
- **Breakout test:** bar close > previous *trading day's* daily high (known pre-open
  → no look-ahead).
- **Entry:** first qualifying bar per day; enter LONG at **next 30-min bar open**
  (trade next bar — no signal-bar look-ahead). Skip if no next bar (EOD trigger).
- **Initial risk R:** entry − (entry − 1.0×ATR14_daily). Stop variants tested below.
- **Direction:** LONG only (per spec). Short logged for reference only.
- **Costs:** 6 bps round-trip (consistent w/ 44), reported as gross AND net; also a
  cost-sensitivity (0 / 6 / 12 bps).
- **Universe (G1 smoke):** RELIANCE, TCS, HDFCBANK, ICICIBANK, SBIN, INFY, MARUTI,
  TATASTEEL (liquid; mix of Cohort-A long history + high-beta). Period: 2018→2026
  for Cohort-A names, else from first 5-min data.
- **Success metric:** net expectancy (R) per exit policy; PF; WR; per-year stability.

## 4. Plan — exit-policy grid (G1 subset → full sweep if it passes)

G1 smoke exits (representative): `EOD` (exit 15:15 close), `HARD_SL` (1×ATR),
`R_TARGET_2R`, `CHANDELIER_3ATR` (trail off highest close), `SUPERTREND_10_3` (flip).
Optional axis: RSI filter OFF / ON (RSI14 ≥ 55 on 30-min at entry).

If G1 clears the falsification bar → full sweep adds the rest of the 13 policies
(ATR_SL 0.3/0.5/1.0, Chandelier 1.0/1.5/2.0, R 1/1.5/2/3, STEP_TRAIL) + Supertrend
(7,3 / 10,3 / 14,2) + volMA window {300,600,900} + RSI thresholds, across the full
liquid universe on the VPS.

---

## 5. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-06-01 ~22:?? | STATUS sections 1–4 written | pre-launch, per playbook |
| 2026-06-01 ~22:?? | G1 smoke probe authored | `scripts/smoke_probe.py`, 8 names |

(updated as the probe runs)

## 6. Crash recovery

- Smoke runner: `venv/bin/python research/49_volbreak_pdh_30min/scripts/smoke_probe.py`
- Self-contained (own resampler + daily loader); reads only `backtest_data/market_data.db`.
- Writes `results/smoke_trades.csv` incrementally + prints a per-policy summary.
- Safe to re-run (overwrites smoke outputs). No live-trading state touched.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `VOLBREAK_PDH_30MIN_SWEEP_STATUS.md` | this file | yes |
| `scripts/smoke_probe.py` | G1 cheap probe | yes |
| `results/smoke_trades.csv` | per-trade × exit-policy | small → yes |
| `results/RESULTS.md` | verdict (after G1) | yes |

## 8. Findings

**G1 KILLED IT. NO EDGE (intraday, net of cost).** n=3,486 trades, 8 names,
2018–2026. Gross faintly positive (best HARD_SL +0.081R); every exit policy
net-negative at 6bps (best −0.029R, PF 0.95) — cost ≈0.08–0.11R/trade wipes the
edge. RSI no help. No per-year persistence. Fails the falsification bar
(net ≥ +0.16R AND PF ≥ 1.24). Did NOT proceed to the 30k-cell sweep — that would
burn hours re-confirming a dead intraday signal. Full table + caveats: RESULTS.md.

Pivot options (all low-EV, see RESULTS §Next levers): volume dose-response probe;
swing-hold variant (but research/44 already owns that ground); else CONCLUDED.

**POSITIONAL test (user request) — DONE.** Multi-day hold flipped the headline
numbers positive (daily-Supertrend net +0.701R/PF1.54, Chandelier +0.490/1.53,
several policies clear the bar). **But the placebo/benchmark kill ended it:**
SIGNAL ≈ BREAK_ONLY ≈ BASELINE(any random day) for every exit → the positional
return is **unconditional large-cap drift (beta), not signal alpha.** Volume filter
adds nothing (slightly hurts); prev-day-high break adds nothing over random entry.
**FINAL: NO EDGE / CONCLUDED on both timeframes.** Files: scripts/smoke_probe.py
(intraday), smoke_probe_positional.py, placebo_benchmark.py; results/*.csv + RESULTS.md.
