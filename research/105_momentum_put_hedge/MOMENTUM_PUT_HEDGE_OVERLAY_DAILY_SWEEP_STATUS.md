# Momentum Book — Put-Hedge Overlay Instead of the Cash-Exit Gate (NIFTY EOD options, 2011–2026)

STATUS: DONE — G1+G2+G3 complete. VERDICT: SIGNAL (weekly tenor only), NOT validated through a grinding bear. See results/RESULTS.md

## 1. The Ask

**What Arun asked:** keep the monthly rebalance as-is, but at the **weekly NIFTY gate check**, instead of
exiting all stocks to cash, **buy NIFTY puts sized to the portfolio** as a partial hedge. Sweep ATM/ITM/OTM,
weekly/bi-weekly/monthly tenors, and different exits (gate reversal, SuperTrend trail, take-profit). Also
test **bear put spreads / ratio structures** rather than pure long puts. Aim: *not* 100% downside
mitigation but a **meaningful partial hedge** that **cuts the exit/re-entry churn and its STCG bill**.

**Correction on the base rule:** the live gate is **weekly** (correct) but on a **100-day SMA** of
NIFTYBEES (not an EMA).

**What we're actually testing:** does replacing the gate's *liquidate-to-cash* with a *stay-invested +
put-hedge* overlay produce a better **after-tax, risk-adjusted** outcome — i.e. keep most of the gate's
drawdown control while eliminating the full-book exits (and the short-term capital-gains tax they realize)?

## 2. The Base — what's held constant

Existing live book, unchanged except for what happens at a risk-off gate:
- Universe: official Nifty-200 proxy; Score: 6m/12m relative strength vs NIFTYBEES (rsblend)
- Hold **top-8 equal-weight**, **top-22 anti-churn buffer**, **monthly rebalance (rotate-only, let winners run)**
- Per-stock **15-day Donchian EOD stop** → to cash, redeployed at next rebalance
- **Weekly gate**: NIFTYBEES < 100-day SMA = risk-off
- Costs: 0.15%/leg equity; idle cash 6.5%; **STCG 20% tracked on gains realized < 365 days**

### Arms
| Arm | Behaviour at weekly risk-off |
|---|---|
| **A0 CASH_EXIT** (baseline = current live) | liquidate ALL stocks to cash; redeploy at next rebalance when risk-on |
| **A1 HOLD_NAKED** (control) | stay fully invested, no hedge — isolates what the gate is actually worth |
| **A2 HEDGE** (the study) | stay invested + buy a NIFTY put structure sized to hedge_ratio × equity |

## 3. Plan — the sweep grid

- **Structure:** `LONG_PUT` · `BEAR_PUT_SPREAD` (long m, short m−width; width 5% / 10%) · `PUT_RATIO_1x2` (flagged: naked tail)
- **Moneyness m:** +2% ITM · ATM · −2% OTM · −5% OTM
- **Tenor:** monthly (~30 DTE, 2011–2026) · weekly (~7 DTE, 2019–2026 only)
- **Hedge ratio:** 0.5 · 0.75 · 1.0 × equity notional
- **Hedge exit:** `GATE_ON` (close when gate flips risk-on) · `+ST_TRAIL` (also exit if NIFTY SuperTrend(10,3) flips bullish) · `+PT100` (also exit if the hedge doubles)
- Rolling: while the gate stays risk-off, roll to the next expiry at expiry.

### Mechanics
- Option prices = **EOD close from `nse_options_bhav`, OI > 0** (binding liquidity rule).
- units = hedge_ratio × equity ÷ NIFTY spot at entry; premium paid in cash, **funded by trimming holdings pro-rata when cash is short** (honest — the book runs ~fully invested); slippage 0.3% of premium.
- Hedge marked daily; P&L = (premium_now − premium_in) × units.

### Success criterion (the decision metric)
Ranked on **net-of-STCG CAGR** and **Calmar**, with **max drawdown** as a guard and **STCG tax paid** +
**number of full-book liquidations** as the churn evidence. A hedge arm must beat A0 on after-tax
risk-adjusted return *without* materially deepening the drawdown.

## 4. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-08-07 | Data audit | monthly opts 2011→2026; weekly only 2019→2026 → monthly = full-cycle arm |
| 2026-08-07 | STATUS-MD written; G1 launching | `run_hedge_sweep.py` |

## 5. Crash Recovery

- Runner: `research/105_momentum_put_hedge/scripts/run_hedge_sweep.py` (self-contained; reads market_data.db on VPS).
- Resume: writes `results/hedge_sweep.csv` incrementally and **skips already-completed configs** — safe to re-run.
- Relaunch: `ssh arun@94.136.185.54 'cd /home/arun/quantifyd && nohup ./venv/bin/python3 research/105_momentum_put_hedge/scripts/run_hedge_sweep.py > /tmp/hedge.log 2>&1 &'` then `tail -f /tmp/hedge.log`.
- Do NOT modify market_data.db (read-only use).

## 6. Files

| File | Purpose | Commit? |
|---|---|---|
| `scripts/run_hedge_sweep.py` | G1 sweep runner | yes |
| `MOMENTUM_PUT_HEDGE_OVERLAY_DAILY_SWEEP_STATUS.md` | This file | yes |
| `results/hedge_sweep.csv` | Per-config results | yes (small) |
| `results/RESULTS.md` | Final verdict | yes |

## 7. Findings

- (pending G1)

## 8. Known caveats to carry into the verdict

- **Fractional lots:** hedge sizing is modelled continuously. On a ₹20L book one NIFTY lot ≈ ₹18L notional,
  so real hedge ratios are coarsely quantized (~1 lot). The winner must be re-checked at integer lots.
- Weekly-tenor arms cover only 2019–2026 (no 2011/2013/2015 stress) — not comparable head-to-head with the
  monthly arms on the full cycle.
- EOD-only decisions and EOD option marks throughout (matches how the book actually trades).
