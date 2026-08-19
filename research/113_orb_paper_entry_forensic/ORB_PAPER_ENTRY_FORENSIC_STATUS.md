# ORB Cash — paper entries never booked since 2026-05-05 (one-word regression)

STATUS: **DIAGNOSED — fix NOT applied (engine code; needs explicit approval + after-15:40 deploy)**

## 1. The Ask

**What Arun asked:** "why is this not paper trading?" (about /app/breakout-paper, then ORB Cash,
N500M, MST, I75WR, Pairs) — followed by "orb cash also shud be paper trading".

**What we actually investigated:** ORB Cash shows enabled + PAPER + 15 stocks scanning, yet Day P&L
Rs 0, 0 positions, 0 trades — every day. Is it (a) not enabled, (b) filtered out by design, or
(c) broken? Answer: **(c)**, and it has been broken for 3.5 months.

## 2. The Base — what the system is meant to do

ORB Cash: OR15 breakout on a 15-stock NSE cash universe, 09:14-15:20, filters = direction, VWAP,
RSI(15m) 60/40, CPR direction + width, gap <=1%, signal age <=30 min, signal drift <=0.5%,
entry cutoff, restart cooldown. Risk-based sizing, SL at the opposite OR boundary, 1.5R target,
EOD square-off 15:16. Paper mode simulates orders; live places MIS orders on Kite.

## 3. Finding — root cause

`place_entry_order()` (services/orb_live_engine.py:1870) calls:

```python
self._verify_order(kite, order_id_str, instrument, 'entry')   # <- 'kite' is undefined here
```

There is no local `kite`, no module-level `kite`, and it is not `self.kite`. Every entry therefore
raises `NameError: name 'kite' is not defined`, which the enclosing `except Exception` catches. It
logs "Entry order FAILED", writes a REJECTED order row, and **returns None** — so the caller hits
`continue` and **no position is ever recorded**.

The exit path at line 2284 does it correctly:

```python
if not self._is_paper():
    kite = self._get_kite()
    self._verify_order(kite, kite_exit_order_id, instrument, 'exit')
```

**Regression point:** commit `03fc917` (2026-05-05) introduced the `_kite_place_order` paper-mode
wrapper. Before it, the entry function did `kite = self._get_kite()` inline, which happened to
define the name used two lines later. The wrapper removed that local; the exit path got the
paper guard, the entry path did not.

## 4. Evidence

| Month | REAL orders PLACED | PAPER orders PLACED | REJECTED rows | positions recorded |
|---|---|---|---|---|
| 2026-04 | 80 | 0 | 6 | 30 |
| 2026-05 | 49 (to the 5th) | 116 | 116 | 16 (last: 05 May 13:05) |
| 2026-06 | 0 | 225 | 225 | **0** |
| 2026-07 | 0 | 200 | 200 | **0** |
| 2026-08 | 0 | 83 | 83 | **0** |

PAPER-placed == REJECTED, one for one, every month since the regression — the NameError signature.

Live log, 2026-08-19 (market open):
```
[ORB] GRASIM sizing: mode=risk R=34.6 qty=86 risk=Rs 2976 notional=Rs 278993
[ORB] Entry order FAILED for GRASIM: name 'kite' is not defined
```

Signals log `action_taken='ENTERED'` (all filters passed) 50x on 18 Aug and 33x by 11:05 on 19 Aug —
the label only means "filters passed", not "position taken". Aggregate ENTERED rows with zero
positions: May 334, Jun 225, Jul 836, Aug 317.

**Corollary — the filters are NOT the reason the book is flat.** The filter-cost study is moot for
this question. For the record, the stored counterfactuals (backtest_data/orb_backtest.db,
2025-08-18 -> 2026-08-18, 186 run days) say the filters earn their keep: TAKEN 675 trades
+Rs 34,260; BLOCKED 376 signals would-be **-Rs 25,104**.

## 5. Severity — why this matters beyond paper

In **live** mode the same path places a REAL Kite order first, then raises, then returns None. The
engine would hold **an untracked real intraday position with no SL placed and no monitoring** — the
exact failure class the guardian exists to catch. Nothing is at risk today (the book is PAPER since
May), but the page has a Live button.

## 6. Proposed fix (NOT applied — engine code)

Mirror the exit path, one line:

```python
if not self._is_paper():
    kite = self._get_kite()
    self._verify_order(kite, order_id_str, instrument, 'entry')
```

Deploy after 15:40 IST with a restart; verify next session that orb_positions gets rows and that
PAPER-placed no longer pairs 1:1 with REJECTED.

## 7. Status log

| Date/time | Event |
|---|---|
| 2026-08-19 11:07 IST | Diagnosed during the paper-book liveness sweep; market OPEN, so read-only only |
| 2026-08-19 11:20 IST | Root cause + regression commit identified; fix written up, awaiting approval |
