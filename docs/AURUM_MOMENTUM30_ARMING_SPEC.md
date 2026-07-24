# AURUM — Arm Dad's Kite Account with the Momentum-30 Strategy (handoff spec)

**Source of truth / reference implementation (already built + tested):** quantifyd
`services/momentum_paper.py` — has a real Kite **CNC MARKET** order layer, gated by a `live_mode`
flag (default OFF), 20/20 simulated-live tests pass. Aurum should MIRROR this logic, per-account.
Strategy provenance: research/62 momentum30-subselect — validated **33.4% gross / 29.0% net-tax CAGR,
−17.0% MaxDD, Calmar ~1.7, Sharpe 1.78** (2013–2026, net of ~0.3% RT cost + 20% STCG).

---

## 1. The strategy rules (implement EXACTLY)

- **Universe:** the OFFICIAL NSE **Nifty 200** (market-cap defined; refresh the current 200 from
  niftyindices.com CSV, cache it). **Exclude ALL ETFs** (GOLDBEES, SILVERBEES, LIQUIDCASE, MON100,
  SETFNIF50, etc. — keep an explicit exclusion list). Equities only.
- **Factor score:** rank the 200 by momentum = **6-month & 12-month relative strength** → take the
  **top-30**. (This reconstructs the "Nifty 200 Momentum 30" from methodology.)
- **Hold:** the **top-8** of the 30, **equal-weight**, 100% invested when risk-on (~12.5% each).
- **Buffer:** keep a held name while it stays inside the **top-22** of the 30 (low churn). Only drop it
  at a rebalance if it falls out of the top-22.
- **Macro gate (weekly):** on the **last trading day of each week (~15:15 IST)**, check **NIFTYBEES vs
  its 100-day SMA**. If NIFTYBEES < 100-DMA → **RISK-OFF: liquidate ALL 8 to cash**. It only redeploys
  at the **next month-end** once NIFTYBEES reclaims the 100-DMA.
- **Per-stock stop (daily):** every day **~15:15 IST**, if a held name **closes below its prior-15-day
  low (15-day Donchian)** → **sell it**. The freed cash sits in the liquid bucket **until the next
  month-end** — it does NOT redeploy mid-month. (This "drift to cash" is a *deliberate defensive
  feature* — tested: immediately redeploying doubles the drawdown.)
- **Rebalance (monthly):** on the **last trading day of the month (~14:45 IST)**, IF the gate is
  risk-on, rebalance to the current top-8. **ROTATE-ONLY** (see §3).
- **Idle / risk-off cash:** park in a liquid fund (**LIQUIDCASE / LIQUIDBEES**, ~6.5%) so it earns while
  waiting. (Optional but recommended.)

**Key behavioural rule (do not "improve" this):** between month-ends the book can only **shed**
(Donchian → cash, or gate → all-cash). It **re-arms only at the month-end rebalance**. No mid-month buys.

---

## 2. Order execution (per Kite account)

- **Product = CNC (delivery). Order type = MARKET.** Poll each order to a COMPLETE fill (timeout ~90s);
  read back `average_price` as the real fill. Raise/alert on rejection or timeout.
- **Integer shares only** (Kite has no fractional): `qty = int(rupees // price)`.
- **Cash-aware:** never overspend into negative cash. When buying N new names, split available cash
  `budget_each = cash / len(new_names)` so no order can push cash negative.
- **Slippage alert:** if `|fill − expected| / expected` exceeds a threshold, log a SLIPPAGE ALERT.
- **Market-hours guard:** only place orders 09:15–15:30 IST, Mon–Fri (block otherwise).
- **Per-order value cap** (safety) + a global **kill switch** that flips `live_mode` OFF (leaves open
  positions untouched — square off manually or let the next risk-off gate exit them).

---

## 3. Month-end rebalance = ROTATE-ONLY (do NOT churn the whole book)

Given the new target top-8:
1. **Exit** any currently-held name **NOT** in the new target (full sell).
2. **Buy** the brand-new target names, cash-aware & equal-weight (split available cash across them).
3. **Kept** names (still in target) **RIDE AS-IS** — no top-up/trim. (Avoids needless brokerage + 20%
   STCG on winners every month; lets winners run. Weight-drift top-up is deferred — flag for later.)

Reference: `momentum_paper._rebalance_live_delta(target, per, live, close, asof, d)`.

---

## 4. ★ ONBOARDING Dad's account — the new bit (existing ETFs → strategy)

Dad's Kite account currently holds ETFs. Arming = a one-time **onboarding rebalance**:

1. **Reconcile** — pull `kite.holdings()` for Dad's account; list everything currently held.
2. **Liquidate the existing ETFs** — CNC MARKET **sell** the existing ETF holdings to cash.
   ⚠ **Tax:** selling realises LTCG/STCG on those ETFs (depends on holding period). Surface the est.
   tax to the user before executing; this is the user's call, not automatic.
3. **Deploy per the gate, at the (first) month-end:**
   - **Gate RISK-ON** (NIFTYBEES ≥ 100-DMA) → deploy the cash into the **top-8 momentum basket**,
     equal-weight (the normal seed/rebalance).
   - **Gate RISK-OFF** (NIFTYBEES < 100-DMA) → **stay in cash / LIQUIDCASE**; do NOT buy. Deploy at the
     **next month-end** once the gate flips risk-on. (You are right — deployment is gated + month-end
     timed; that IS the system's balancing method.)
4. From then on it runs on the normal schedule (§1).

**Timing decision (confirm with user):** two clean options —
- **(a) Liquidate-now, deploy-at-month-end:** sell the ETFs immediately (stop the old exposure), park
  in LIQUIDCASE, then deploy at the first month-end if gate risk-on. *Recommended* — decouples "get out
  of the old ETFs" from "get into the strategy," and matches the month-end deploy discipline.
- **(b) All-at-once at the next month-end:** sell ETFs + deploy in a single rebalance. Fewer taxable
  events clustered but leaves the old ETF exposure on until month-end.

Reference: `momentum_paper.seed()` handles fresh-capital seeding; Aurum's onboarding = **reconcile +
sell-non-strategy-holdings FIRST**, then seed. (seed() assumes an empty account, so add the liquidation
step.)

---

## 5. Aurum app — what to build/configure

- **Per-account Kite auth** (Dad's account): api_key/secret + daily access-token refresh, stored per
  account (Aurum is multi-account; keep Dad's creds isolated).
- **Strategy engine** = port the momentum_paper rules (§1–§3): universe refresh, momentum ranking,
  gate, Donchian, rotate-only rebalance. (Postgres for state: positions, orders, nav, gate history.)
- **Scheduler (APScheduler / cron, IST):**
  - Daily 15:15 — Donchian stop + EOD mark/NAV.
  - Weekly (last trading day of week) ~15:15 — NIFTYBEES 100-DMA gate.
  - Monthly (last trading day of month) ~14:45 — rebalance (rotate-only), guarded by
    `is_last_trading_day_of_month`.
- **Order layer** (§2): CNC MARKET, integer qty, cash-aware, fill-poll, slippage alert, market-hours
  guard, per-order cap.
- **Onboarding flow** (§4): reconcile → liquidate existing ETFs → deploy per gate at month-end.
- **Reconcile job:** book (Postgres) vs Kite holdings — alert on mismatch (`momentum_paper.reconcile_
  holdings()` is the model: alert-only, don't auto-correct ambiguous cases).
- **Controls:** `live_mode` toggle **default OFF (paper)** + capital amount set at flip; **kill switch**;
  a LIVE/PAPER badge in the Aurum UI.

---

## 6. Safety checklist / decisions to confirm BEFORE going live

- [ ] **Capital amount** for Dad's book (set at the live flip).
- [ ] **Onboarding timing:** option (a) liquidate-now vs (b) all-at-month-end (§4). *Rec: (a).*
- [ ] **Gate-off handling at arming:** cash vs LIQUIDCASE (rec: LIQUIDCASE to earn ~6.5% while waiting).
- [ ] **Keep any existing ETF?** (e.g. if some are already the liquid parking fund — don't sell those.)
- [ ] **Tax awareness:** liquidating Dad's ETFs realises CGT — show the estimate, get explicit sign-off.
- [ ] **Dry-run first:** run the whole flow on Dad's account in **PAPER (live_mode OFF)** for one full
      cycle (a gate check + a simulated rebalance) before flipping live.
- [ ] **Reconcile after the first live fills** (book vs Kite) and confirm the 8 positions match.
- [ ] **Kill switch tested** and reachable.
- [ ] First live action is the onboarding rebalance — **fund the account + flip live before the target
      month-end.**

---

## 7. Validated numbers to show the client (Dad) — the honest pitch

- **33.4% gross / 29.0% net-tax CAGR, −17.0% max drawdown, Calmar ~1.7, Sharpe 1.78** (2013–2026, net of
  ~0.3% RT cost + 20% STCG), vs NIFTYBEES 12.3% / −36.3%.
- Caveats to state plainly: survivorship-free PIT universe used in backtest but a **modern sub-period**;
  monthly turnover → real STCG; a G5 paper soak is running live (research/62 book) — Dad's account would
  be the first real-money instance. Past performance ≠ future results.
