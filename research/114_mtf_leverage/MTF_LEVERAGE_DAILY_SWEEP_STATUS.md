# MTF Leverage on the Live Momentum Book — Is 2.5x Survivable?

STATUS: DONE

## 1. The Ask

**What Arun asked:** "run a study what if we take the mtf option for our running momentum
portfolio, how does the numbers, returns, drawdowns etc look like... the mtf factor itself varies,
we can take the min side like 2.5x".

**What we are actually testing:** research/104 already ran the leverage frontier on THIS EXACT book
(rsblend, top-8, top-22 buffer, Donchian-15, weekly NIFTYBEES-100SMA gate) over 2006-2026 including
2008 and 2020. It found 0 margin calls up to 2.0x and CAGR rising from 32.7% to ~57%. Two gaps make
it unable to answer today's question:

1. **It stopped at 2.0x.** Arun is asking about 2.5x. Untested.
2. **It financed at 10.5%.** `docs/MOMENTUM_GOLIVE_RUNBOOK_3L.md` puts real Zerodha MTF at
   **~14.6%/yr (0.04%/day)**. At 2.5x you borrow 1.5x equity, so a 4.1pp rate error is ~6pp of
   equity per year — largest exactly where the question is being asked.

So: re-run the identical engine over a wider grid at BOTH rates, and report margin calls.

## 2. The Base (unchanged from r/104 — this is not a new strategy)

- Engine: `research/104_momentum_leverage/scripts/run_lev62.py::run_lev62`, imported not copied.
- Rules: rsblend score, N=8, buffer 22, Donchian 15, weekly index gate at 100-SMA, monthly rotate.
- Period 2006-01-01 -> 2026, daily-marked, net 0.3% round-trip, idle cash at 6.5%.
- Leverage applied only while the gate is risk-on.
- **Margin call: MAINT = 0.25** — if equity/gross < 25% the engine liquidates and counts a call.
  At 2.5x you START at 40% equity, so roughly a 37% fall in the holdings triggers liquidation.
  The book's own unlevered drawdown is -22 to -30%, so this is NOT a remote scenario.

## 3. Plan

| Axis | Values |
|---|---|
| Leverage | 1.0, 1.3, 1.6, 2.0, **2.5**, 3.0 |
| Borrow rate | 10.5% (r/104), **14.6% (real Zerodha MTF)** |

12 runs. Report CAGR, MaxDD, Sharpe, Calmar, margin calls, plus an explicit rate-cost table.

**Success criterion:** 2.5x is only interesting if it clears 1.0x on Calmar AND takes zero margin
calls across the full cycle. Higher CAGR at a worse Calmar is leverage, not edge — the book already
tells us that above 1.6x.

## 4. What this study does NOT do

- It does not derive per-stock Zerodha MTF factors. Kite does not expose them and the token was
  invalid at write time. 2.5x is taken as Arun's stated conservative floor; if a held name has a
  lower factor the achievable leverage is lower, never higher.
- It does not model intraday gap risk, MTF pledge haircuts, or the broker's discretionary right to
  square off early. Real margin calls can fire before a daily close does.

## 5. Status

| Time | Event |
|---|---|

## 6. Crash Recovery

- Progress: `tail -f research/114_mtf_leverage/results/run.log`
- Alive? `pgrep -af mtf25`
- Resume: `cd /home/arun/quantifyd && nohup ./venv/bin/python3 research/114_mtf_leverage/scripts/mtf25.py > research/114_mtf_leverage/results/run.log 2>&1 &`
- Partial rows land in `results/mtf_frontier.csv` (flushed per run). Re-running overwrites cleanly.
- READ-ONLY: places no orders, touches no live state.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/mtf25.py` | Runner | yes |
| `results/mtf_frontier.csv` | 12-run grid | yes |
| `results/run.log` | Progress | yes |
| `results/RESULTS.md` | Verdict | yes |

## 8. Findings

See results/RESULTS.md. 2.5x = 67.3% CAGR / -52.0% DD / Calmar 1.30 / 0 margin calls at the real 14.6% rate. Zero calls is REAL: on 5 of the 6 worst days in 20 years the gate had already moved the book to cash (28 days before the 2020 low, 73 before 2008). But 3.0x survives on a 0.2pp margin on the one exposed day (2006-05-19) and is NOT recommended. Calmar falls 1.50 -> 1.30: leverage, not new edge.
