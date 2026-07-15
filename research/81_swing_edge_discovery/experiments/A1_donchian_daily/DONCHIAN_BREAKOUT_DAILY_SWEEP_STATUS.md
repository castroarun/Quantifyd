# A1 Donchian Breakout Swing (2–4d) — F&O Universe Daily, IS Coarse Screen

STATUS: DONE — NO EDGE (see Verdict)

Experiment **EXP-A1** of research/81 Swing Edge Discovery
(`EDGE_DISCOVERY_81_STUDY_STATE.md`). Family A: momentum / trend continuation.

---

## 1. The Ask

**What we're testing:** Does an N-day high/low breakout on DAILY bars, held
2–4 sessions with an ATR stop, show a gross+net edge on the F&O-liquid
universe in-sample (2005–2017)? This is the G1 coarse probe for family A —
survivors go to the full validation pipeline (walk-forward, param plateau,
Monte Carlo, regime split, nulls).

## 2. Economic hypothesis (G0)

Breakouts of multi-day ranges attract late-comer/momentum flow and forced
short-covering; institutional execution is spread over days, giving a
few-day drift after the break. Counterparty: mean-reversion traders fading
the break + resting limit orders at the old range edge. Decay risk: widely
known; costs and false breaks in chop are the killer — hence a net-of-cost,
time-capped test. Failure would look like: positive gross eaten by costs, or
edge concentrated in 2007/2009-type regimes only.

## 3. The Base — locked mechanics

- **Data:** daily bars, canonical loader (audit-clean), F&O universe
  (`FNO_LOT_SIZES` keys, ~86 names) + NIFTY50/BANKNIFTY dropped (index daily
  starts 2015 — insufficient IS depth; index gets its own 5-min experiment).
- **Period (IS only):** signals 2005-01-01 → 2017-12-31 (chronological 60%).
  Val 2018-01→2022-06 and OOS 2022-07→2026-07 are NOT touched by this screen.
- **Signal (long):** close[t] > max(high[t-N..t-1]) → enter next open.
  **Short:** close[t] < min(low[t-N..t-1]) → enter next open. Sides
  independent (playbook: report split).
- **Stop:** long = close[t] − k·ATR14[t]; short = close[t] + k·ATR14[t]
  (ATR14 = SMA-14 of true range, causal). No profit target (research/71:
  targets hurt trend trades; also bounds the grid).
- **Time-stop:** exit at close of session entry+(ts−1). Hard ≤ 4.
- **Corporate-action guard:** no entries within 3 sessions after a >25%
  overnight-gap date; symbols with >5 such flags excluded from the run.
- **Costs:** FUTURES_PROXY (user decision), slippage 3 bps/side. Cells that
  pass the gate get an exact 2× slippage re-run (fragility check).
- **Engine:** `engine/` @ git 6656e92+patch, 32 unit assertions passing.

## 4. Plan — pre-registered grid (LOCKED before first run)

| Axis | Values |
|---|---|
| N (breakout lookback, days) | 10, 20, 55 |
| Direction | long, short |
| ATR stop multiple k | 1.5, 2.5 |
| Time-stop sessions | 2, 4 |

**24 cells** × ~86 symbols. Experiment-count ledger +24 (t-stats to be
Bonferroni-discounted across the family).

**G1 pass gate (decided now):** pooled NET expectancy > 0 with t ≥ 3, AND
≥55% of symbols net-positive, AND effect not a lone spike across N
(monotonic or plateau). Cells failing all → family A daily variant recorded
as NO EDGE and we move on. **Falsification:** if no cell passes gross, the
family fails at G1 — no threshold shopping, no grid extensions without a NEW
pre-registered experiment ID.

## 5. Status (live log)

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-15 ~19:45 IST | Pre-registered | grid locked, runner being written |

## 6. Crash Recovery

- Runner: `scripts/run_a1_donchian_daily.py` (repo-rooted paths); resume =
  re-run same command; per-(cell,symbol) rows in
  `results/a1_cells.csv` are skipped if present.
- Trades detail: `results/a1_trades.csv` (slim, appended per symbol-cell).
- Log: `/tmp/a1_donchian.log` on VPS.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/run_a1_donchian_daily.py` | Runner | yes |
| `results/a1_cells.csv` | Per-cell-per-symbol aggregates | yes |
| `results/a1_trades.csv` | Slim per-trade rows | if <10 MB |
| `results/A1_RESULTS.md` | Verdict | yes |

## 8. Findings

(after run)
| 2026-07-15 19:51 IST | Sweep complete: 72 symbols run, 2 CA-excluded (BAJFINANCE,VEDL...), 7 too thin, 1.2 min | |
| 2026-07-15 19:51 IST | Aggregation done — ranking in results/a1_ranking.csv | |

## VERDICT (2026-07-15 ~20:00 IST): NO EDGE

**All 24 cells net-negative; 22/24 negative even gross.** Longs strongly
negative gross (pooled t -8..-10). Best cells = wide-stop SHORT breakdowns,
+4-6 bps gross, killed by ~9.6 bps round-trip cost. G1 gate (net t>=3,
>=55% symbols positive) not approached. Pre-registered falsification met ->
family A daily variant closed, NO grid extensions. 187,283 trades, 72 F&O
symbols, IS 2005-2017. Consistent with research/71: breakout pays only with
multi-week trailing exits - a 2-4 day cap structurally kills it.
5-min variants (A2 ORB->swing) remain queued post-backfill as a SEPARATE
pre-registered experiment.

STATUS: DONE
