# AlgoTest Test Card — Confirm the Compression-Entry Edge on Real Premiums

Goal: validate on AlgoTest's **real option premiums** what the price-only study (research/64) found —
that compressed-volatility entries lift the net edge — and settle the one thing the proxy can't:
the **low-VIX premium** question and the exact ₹.

## Fixed base (do NOT change between runs)
- Instrument: **NIFTY weekly iron fly** — SELL ATM CE+PE, BUY ATM ±2% wings (the locked V2 base).
- Entry **09:20**, 2nd-nearest weekly (~4 TD-before-expiry on AlgoTest), exit 1-TD-before-expiry/roll.
- Stop **2% underlying move**; target **+40% of credit**; **10 lots (qty 650)**.
- Costs: ₹20/order, taxes on, **0.25% slippage**. Report **net**, ex-COVID (strip 2020-03-13/03-20).
- Metric per run: net P&L, **Calmar**, MaxDD, win-rate, per-year P&L (positive-every-year is the bar).

## What AlgoTest CAN gate directly → run these
| # | Run | Entry filter to set | Tests |
|---|---|---|---|
| 0 | **Baseline** | VIX ≥ 13 (current live) | reference |
| 1 | VIX band 13–16 | VIX ≥ 13 AND VIX ≤ 16 | is the calm-rich 13–16 zone better than open-ended ≥13? |
| 2 | VIX band 13–20 | VIX ≥ 13 AND VIX ≤ 20 | does capping at 20 (avoid the low-calm 20+ tail) help? |
| 3 | **Hard-skip high VIX** | VIX ≥ 13 AND VIX ≤ 22 | study says >22 is the only EV-negative regime |
| 4 | Drop the floor | VIX ≤ 16 (no lower floor) | the low-VIX question: does thin premium kill the high calm-rate? **(key)** |
| 5 | Low-VIX only | VIX ≤ 13 | direct read on whether low-VIX flies are net-positive on REAL premiums |

Runs 4–5 are the decisive ones: the price-only proxy says low-VIX = highest calm/EV, but real premium
is thin there. AlgoTest's actual fills settle it. If 4–5 are net-negative → keep the VIX *floor*; if
positive → the floor is leaving money on the table.

## What AlgoTest likely CANNOT gate (needs the forward shadow log / Python on recorder data)
- **Daily CPR width < 0.10–0.16%**, **ATR% < 1.1%**, **Stochastic %K > 65** — arbitrary indicator
  conditions on a positional book aren't expressible on AlgoTest. These are validated forward by the
  **shadow logger** (engine logs would-skip daily) and, once the options recorder has enough history,
  a Python fly-backtest on real recorded premiums.

## Read-out template (fill per run)
| Run | net P&L | Calmar | MaxDD | win% | +years | verdict |
|---|---|---|---|---|---|---|
| 0 baseline | | | | | | |
| 1 VIX 13–16 | | | | | | |
| 2 VIX 13–20 | | | | | | |
| 3 VIX ≤22 | | | | | | |
| 4 VIX ≤16 (no floor) | | | | | | |
| 5 VIX ≤13 | | | | | | |

Decision: pick the VIX rule with the best Calmar that's **positive every year**; hand the winner back to
research/64 to fold into the live entry alongside the (shadow-validated) compression filter.
