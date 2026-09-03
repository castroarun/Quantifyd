# Open Alpha Weak-Market Gate Bake-Off — Index × MA-Type × Length on the ATH-Breakout Book

STATUS: RUNNING

## 1. The Ask

**What you asked (Arun, 2026-09-03):** "are u sure ab the correct gate with which u hv
done the backtestting? is 200 SMA the optimal one? did u try 200 dma or 100 dma or so?
... instead of nifty, how ab trying gates on small cap index?"

**What we're actually testing:** The adopted Open Alpha spec uses a NIFTYBEES 200-day
SMA gate inherited from the BananaPatterns spec — we validated gate ON vs OFF (ON won)
but never swept the gate's *series*, *type*, or *length*. Since the book's breadth
driver is smallcap participation (established in the r/142 study), a smallcap-index
gate may time entries better than large-cap NIFTYBEES. Which gate construction
maximizes the risk-adjusted outcome of the adopted trail-20 spec?

**Context that forced this now:** the 2026-09-03 gate audit found the backtest/seeding
gate had been silently DISABLED since ~28-Apr-2026 — Kite phantom holiday rows on
2026-01-15 (O=H=L=C=prev close, vol 0, 526 symbols) NaN-poisoned `rolling(200).mean()`
on the union-index-aligned NIFTYBEES series (pandas returns NaN if any window value is
NaN; `close < NaN` = False = "not weak"). Fixed: gate now computed on the dropna'd
series then re-aligned (`bluesky_replay.py`, `seed_paper_state.py`; the live
`bluesky_paper.py` engine already used dropna and was never affected). The 5 seeded
open positions (entries Aug-4/Aug-13/Sep-2, all true-gate-WEAK days) are spec-invalid
and the book will be RE-SEEDED after this sweep fixes the final gate choice.
Phantom rows themselves still in DB — purge blocked by permission classifier, needs
Arun's go-ahead (they also slightly pollute per-stock TV/SMA/ATH series).

## 2. The Base — what's being tested

Engine: `scripts/bluesky_replay.py` `simulate()` — the adopted spec, all constants
frozen: ATH-close breakout signal, setup within 20% of ATH-close, RS>=70 (IBD-weighted
percentile), TV floor ₹5cr/day 20d-median (t-1), fill at pivot (realistic), 25bps/side,
−8% stop on close (gap-aware), trail-SMA20 exit at signal close, 8 slots, 18.75%
sizing, random selection among candidates, 10-seed ensemble.

**Only the weak-market gate varies.** Gate rule: entries blocked on day i when
gate-series close < its N-day MA on day i−1 (shift-1, causal). NaN-robust dropna
construction.

## 3. Plan — variant grid

| Axis | Values |
|---|---|
| Gate series | NIFTYBEES (ETF, current), NIFTY50, NIFTY500, NIFTYMIDCAP150, NIFTYSMLCAP250 (indices) |
| Price-vs-MA | SMA100/150/200, EMA100/150/200 (blocked while close < MA) |
| MA crossover | SMA50<200, EMA20<100, SMA20<100 (blocked while fast < slow) |
| Drawdown | blocked while close < 95% / 90% of rolling 252d high (per series) |
| Momentum | blocked while 63d / 126d return < 0 (per series) |
| Breadth (universe) | blocked while % of stocks above own SMA200 < 40/50%, SMA50 < 40/50% |
| Volatility | blocked while NIFTY50 20d realized vol (ann.) > 18% / 25% |
| + baseline | gate OFF |

= 5×(6+3+2+2) + 4 + 2 + 1 = **72 cells × 10 seeds × 2 windows** (per Arun: "try
different emas, dmas, smas, MA crossovers... consider even nifty 500, anything other
than MAs also"). All cells run BOTH windows: W1 2020→now primary, W2 2016-06→2019-12
regime validation. Ranking metric: median terminal × with median maxDD alongside; a
winner must beat NIFTYBEES-SMA-200 on BOTH windows to displace it, not just the bull
window (regime-dependence + multiple-testing guard: 72 cells ⇒ expect several lucky
bull-window winners; the two-window AND-rule plus construction-family coherence is the
overfit control).

Data caveats stated up front: index series (NIFTY50/MIDCAP150/SMLCAP250) end
2026-08-27 in the DB (5 days stale — ffill'd weak value for the tail; if an index gate
wins, the nightly refresh must add that series before the live engine adopts it).
r/64 found some Kite INDEX series corrupt — sanity ranges printed per series at load.

## 4. Status log

| Date/time | Event |
|---|---|
| 2026-09-03 ~16:15 IST | Gate bug found + fixed (dropna); STATUS written; sweep launching |
| 2026-09-03 16:2x IST | Sweep launched on VPS (nohup), results/gate_bakeoff.csv incremental |

## 5. Crash recovery

- Progress: `tail /tmp/gate_bakeoff.log`; results accumulate in
  `research/142_bananapatterns_replication/results/gate_bakeoff.csv` (one row per cell,
  written incrementally — safe to inspect mid-run).
- Alive? `pgrep -f gate_bakeoff.py`
- Resume: re-run `venv/bin/python research/142_bananapatterns_replication/scripts/gate_bakeoff.py`
  — it skips cells already in the CSV.
- After sweep: adopt winner → re-seed paper book via `scripts/seed_paper_state.py`
  (WARNING: re-seed wipes state — must re-apply Arun's ₹2.5L deposits (2026-09-03
  fund_flows: 2,00,000 + 50,000) and re-run `services/dividend_engine.py --init`
  so the HWM re-anchors on the corrected NAV).

## 6. Files

| File | Purpose | Committable |
|---|---|---|
| `scripts/gate_bakeoff.py` | sweep runner | yes |
| `results/gate_bakeoff.csv` | per-cell results | yes |
| this file | status | yes |

## 7. Findings

(pending)
