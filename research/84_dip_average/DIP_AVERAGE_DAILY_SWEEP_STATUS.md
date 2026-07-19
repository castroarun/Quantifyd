# research/84 — Drawdown Dip-Buy with Averaging-Down (x/y/z), Nifty-50 class

STATUS: DONE — NO EDGE (see Verdict)

User idea 2026-07-17: buy a stock once it drops x% from its top; sell once it
rises y% (from average cost); if it keeps falling, average down per a z% plan.
Optimize x/y/z, add variants, verdict vs Nifty/market returns.

## 1. G0 — hypothesis & the known trap

Mechanism: large-cap drawdowns mean-revert (quality names get bought);
counterparty: capitulating holders. The KNOWN killer: averaging-down
concentrates capital in the names that DON'T come back — and a backtest on
today's index members (survivors) structurally hides that. Controls:
(a) three universe arms — TODAY'S Nifty-50 (user ask, biased), F&O ~80
(broader), and an ADVERSARIAL blow-up set (YESBANK, RCOM, SUZLON, RPOWER,
JPASSOCIAT, IDEA, PCJEWELLER, UNITECH, IBULHSGFIN, RELCAPITAL) — what the
plan does when reversion never comes; (b) no-averaging control cells (z off);
(c) campaign DURATION and open-campaign marks reported, not just win rate
(this system's losses hide in never-closed positions).

## 2. Locked mechanics

- Daily closes, causal (signals on close t → fills next open t+1).
- Reference top: {ATH close, 252d rolling high} (causal, shift-1).
- ENTER: close ≤ ref×(1−x). AVERAGE: equal tranche each further z% drop from
  the LAST fill, up to max_adds=3 (4 tranches total, capital reserved
  upfront — no infinite-capital fantasy). EXIT: close ≥ avg_cost×(1+y), sell
  all. No stop-loss (that IS the system); campaigns open at data end are
  marked to last close and flagged.
- Costs: CASH_DELIVERY preset (STT 0.1% both sides, DP, 3bp slippage) —
  this is a delivery strategy, holds can span months/years.
- Portfolio NAV: 10 concurrent campaign slots, slot = equity/10 reserved at
  campaign start, idle cash earns 0. Benchmark NIFTYBEES B&H.

## 3. Grid (LOCKED, 24 cells, ledger +24)

ref {ATH, 252d} × x {15, 25, 35}% × y {10, 20}% × averaging {OFF, z=15%×3}
Splits: IS 2005-2018 (campaigns assigned by entry date), Val 2019-2023,
OOS 2024+ untouched. G1 on IS.

**Gate:** portfolio NAV must beat NIFTYBEES on Calmar AND CAGR on the F&O arm
(not just the survivor arm), with open-campaign marks included, AND the
blow-up arm must not show catastrophic slot loss (>60% of a slot) in >20% of
its campaigns — else the system is a survivorship artifact / tail bomb.
Falsification: survivor-arm-only outperformance = NOT an edge.

## 4. Files

Runner `scripts/run_dip_probe.py` (campaign sim + sanity check), results in
`results/`, log `/tmp/dip_probe.log`. Ledger +24. OOS: nothing consumed.

## Status

| Date/time | Event |
|---|---|
| 2026-07-17 ~15:35 IST | Pre-registered |

## Findings

(after run)

## VERDICT (2026-07-17 ~16:20 IST): NO EDGE vs market — pre-registered gate failed

Campaign level (the seductive numbers): 93-98% win rates, +4-9% mean per
campaign on survivor universes. Portfolio level (the truth): best cell ~10.8%
CAGR vs NIFTYBEES 12.6% B&H (2005-18) — UNDERPERFORMS by 1.8-8pp on the
SURVIVOR universe, before the blow-up reality. Mechanisms:
1. Slot demand is 3-4x capacity (skipped 300-1100 campaigns vs taken 116-303)
   — dips cluster in crashes, exactly when all slots jam; the per-campaign
   stats cannot be harvested.
2. Tail burial: p95 campaign duration 1-4 YEARS; 2-11% never close.
3. Adversarial arm: on blow-up names the same rules give ~0 to NEGATIVE mean
   despite 75-93% win rates; averaging makes it WORSE (slot-wipes 2-7%,
   capital dead up to a decade) — and our DB is missing the worst blowups, so
   this UNDERSTATES the damage. Today's Nifty 50 is a survivor set by
   construction.
The system is structurally short-volatility: many small wins, rare
catastrophic burials — an illusion of win rate created by take-profit +
no-stop asymmetry. Gate (beat NIFTYBEES on F&O arm + survive blowup arm)
decisively failed. STATUS: DONE — NO EDGE.
