# NAS Portfolio Bracket — Daily ±TP/±SL on the Combined 3-System 916 Book, 1-min Replay over 64 Recorded Days

STATUS: DONE

## 2. The Ask

**What Arun asked:** "n=10 is too small. We've recorded options data every day for
60–70 days — use it. And it need not be 5k both sides; check variable combinations
and find the sweet spot."

**What we're actually testing:** Take the three LIVE 916 systems (ATM = per-leg SL +
ST-trail survivor; ATM2 = ±0.4% move-stop re-center; ATM4 = roll-to-match, max 1),
replayed faithfully at 2 lots each against the recorded per-minute NIFTY option chain
(`options_data.db`, 64 trading days, 2026-04-20 → 2026-07-22). Build the COMBINED
intraday portfolio P&L path each day. Then overlay a daily portfolio bracket: the first
minute combined P&L ≥ TP → flatten the whole book (book the breach value); ≤ SL →
flatten. Sweep TP × SL and find the configuration that best improves total P&L AND
risk (drawdown, worst day), then stress it for robustness (per-half stability, is the
sweet spot a broad plateau or a lucky spike).

Success criterion: a bracket that beats the ride-to-EOD baseline on BOTH net P&L and
Calmar-style return/drawdown, on a config that sits on a smooth plateau (not a peak) and
holds in both halves of the sample.

## 3. The Base

- **Systems (each 2 lots, QTY=130):**
  - 916 ATM: entry 09:16 ATM straddle; per-leg SL 1.3× entry; on SL, survivor trails on
    SuperTrend(7,2) of its premium (naked_method=st).
  - 916 ATM2: entry 09:16 ATM straddle; ±0.4% underlying move-stop, one-and-done, then
    re-center one fresh straddle at the new ATM (move_stop_reenter).
  - 916 ATM4: entry 09:16 ATM straddle; per-leg SL 1.3×; on SL, roll the stopped leg once
    to a premium-matched strike; else survivor trails.
- **Force squareoff:** 15:15 (live).
- **Engine:** research/68 faithful replay (`engine.py`), instrumented to emit per-minute
  MTM (realized + unrealized − brokerage) per system. Brokerage ₹80/leg round-trip.
- **Universe/period:** NIFTY weekly front-expiry, 64 recorded days.
- **Portfolio path:** sum of the 3 systems' per-minute MTM, forward-filled to 15:15.
- **Baseline:** ride every day to 15:15 (no bracket) = (TP None, SL None).
- **Known optimism (state honestly):** LTP fills, no slippage, 1-min resolution → exits
  (incl. the bracket) are optimistic; real gap-through makes an SL leak past its level
  (seen live: a −5k stop booked −7.5k to −8k). Replaying the CURRENT config over all 64
  days (earlier days actually ran different lots/configs) — this isolates the bracket
  effect on today's system, which is the question, but is not the historical P&L.

## 4. Plan — the grid

- TP ∈ {None, 3k, 4k, 5k, 6k, 8k, 10k, 12k, 15k}
- SL ∈ {None, −3k, −4k, −5k, −6k, −8k, −10k, −12k, −15k}
- 9 × 9 = 81 configs. Baseline = (None, None).
- For each config, over the 64 daily paths: total net, daily win-rate, worst day, P&L
  std, max drawdown of the daily-cumulative curve, Calmar = total/|maxDD|.
- Robustness: first-32 vs last-32-day totals for the top configs; neighbourhood check
  around the best (plateau vs spike). Multiple-testing (81 cells) is the #1 risk — the
  verdict must lean on smoothness + per-half stability, not the single best cell.

## 5. Status (event log)

| Time (IST) | Event |
|---|---|
| 2026-07-22 ~14:2x | Folder + STATUS created; engine copied + instrumented; driver written |

## 6. Crash Recovery

- Replay + sweep driver: `research/90_nas_portfolio_bracket/scripts/run_bracket_sweep.py`
  (VPS). Instrumented engine: `scripts/engine_mtm.py`. Run:
  `cd /home/arun/quantifyd && ./venv/bin/python3 research/90_nas_portfolio_bracket/scripts/run_bracket_sweep.py`
- Intermediate per-day paths cached to `results/day_paths.json` — if present, the sweep
  re-runs instantly without re-replaying the chain.
- Outputs: `results/sweep_results.csv` (81 rows), `results/RESULTS.md` (verdict).
- Safe to inspect anything under results/. Do not edit engine_mtm.py mid-run.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| scripts/engine_mtm.py | research/68 engine + per-minute MTM recording | yes |
| scripts/run_bracket_sweep.py | replay 3 systems × 64 days, sweep TP×SL | yes |
| results/day_paths.json | cached per-minute portfolio paths | maybe (small-ish) |
| results/sweep_results.csv | 81-cell sweep metrics | yes |
| results/RESULTS.md | final verdict | yes |
| THIS FILE | live status | yes |

## 8. Findings

(pending run)
