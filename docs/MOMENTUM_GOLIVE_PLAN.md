# Momentum-30 — Go-Live Plan (wife's Zerodha account)

**Status:** PLANNING · real money, family account → every phase gated, soak before scale.
**Owner:** Arun · **Prepared:** 2026-08-07

The book: relative-strength momentum, **top-8 of Nifty-200, top-22 anti-churn buffer, 15-day Donchian
per-stock EOD stop, NIFTYBEES>100-SMA index cash gate, monthly rebalance (rotate-only, let winners run)**.
Live-validated backtest (full cycle 2006–2026): **32.7% CAGR, −22% max DD, Calmar 1.48** at 1.0× (own capital).

---

## Decisions locked (2026-08-07)

| # | Decision | Choice |
|---|---|---|
| 1 | Leverage at go-live | **1.0× (own capital only)** — no MTF until the live plumbing is proven |
| 2 | Order type | **MARKET (CNC)** — liquid Nifty-200, executed in the 15:15 EOD window |
| 3 | Notifications v1 | **Email only** (reuse `services/notifications.py` email provider); WhatsApp deferred |
| 4 | Email transport | **Gmail SMTP** already configured in `config.py` (`smtp.gmail.com:587`, `arun.castromin@gmail.com`, `GMAIL_APP_PASSWORD` in `.env`) — same infra as NAS |
| 5 | Wife's Kite account | **To be set up** — needs its own Kite Connect API app (key/secret) + daily login token; blocks go-live |

## Scope corrections (from Arun's review — these are NOT gaps)

- **Whole-share integer qty is CORRECT** — Indian equity delivery (CNC) is whole-shares-only; "skip if budget < 1 share" is right. (Paper books fractions for accounting; live is integer.)
- **Kill-switch = flip to PAPER (stop new automation), KEEP positions** — the existing `_kill_switch()` is exactly the desired behaviour for a long equity book you hold through. No position-flatten needed (that's a NAS-options requirement, not ours).
- **No exchange-side SL-M needed** — the Donchian stop is evaluated at **EOD (15:15) on the daily close**, and the backtest models exactly that. An intraday SL-M would *deviate* from tested logic. Software EOD stop is correct by design.
- **No 30-sec / 5-min polling / live-guardian** — momentum is EOD-only. A **once-daily morning reconciliation** is sufficient (vs NAS's 5-min guardian for intraday options).

---

## Readiness assessment — EXISTS vs BUILD

| Capability | Status | Action |
|---|---|---|
| Real Kite CNC MARKET order + fill-poll (`_place_cnc_market`) | ✅ EXISTS | reuse |
| `_is_live()` single-flag arming, all orders funnel through `_buy/_sell` | ✅ EXISTS | reuse |
| Guardrails: ₹15L order cap, market-hours guard, whole-share, slippage log | ✅ EXISTS | reuse; tune caps |
| Live rebalance = rotate-only, let winners run | ✅ EXISTS (just fixed) | reuse |
| Kite auto-login cron 08:50 + token `access_token.json` | ✅ EXISTS (shared) | point at wife's account |
| Kill-switch = flip to paper (keep positions) | ✅ EXISTS = desired | reuse as-is |
| `reconcile_holdings()` DB vs Kite (alert-only) | ✅ EXISTS | wire to email alert + daily cron |
| Notification service (email/WhatsApp/in-app), `send_alert` + `send_eod_report` | ✅ EXISTS (`services/notifications.py`) | reuse (email) |
| **Two-key live safety** (mode flag + explicit live-capital arm) | 🔨 BUILD (small) | require live-capital > 0 AND live_mode=1 to place orders |
| **EOD email report** (entries/exits + positions + P&L, once/day) | 🔨 BUILD (medium) | `momentum_eod_report.py` à la `nas_eod_report.py` |
| **Monthly email report** (NAV, trades, holdings, benchmark) | 🔨 BUILD (small) | month-end scheduled report |
| **Order-failure / reconcile-mismatch email alert** | 🔨 BUILD (small) | `send_alert()` on REJECTED/timeout + daily reconcile diff |
| **Frontend LIVE controls** (badge, toggle, kill button, live positions, reconcile panel) | 🔨 BUILD (medium) | on `/app/momentum-paper`, NAS.tsx pattern |
| **Cash-management UI + logic** (deposit / withdraw) | 🔨 BUILD (medium) | see logic below |
| Live capital amount set | 🔨 SET | via `/toggle-mode` payload |

---

## Cash-management logic

### Deposit (add funds)
- **Default: equal-rupee top-up of current holdings when gate is ON** (buy `+deposit/8` of each held name). Deploys money immediately, sells nothing (respects let-winners-run), simple. If gate is OFF → **park in liquid (LIQUIDCASE ~6.5%)** until gate turns on / next rebalance. Never deploy new cash into a downtrend.
- **Alternative (UI option): wait for next monthly rebalance** — simplest, tiny yield drag; best for small top-ups.
- **Reject: proportional-to-winners** — chases extended names, concentrates risk.
- **Backtest owed (light):** confirm "deploy-now equal top-up" ≈ "wait-for-rebalance" within noise → lock the default. (Second-order timing effect; expect negligible.)

### Withdraw (remove funds)
- **Sell from the WEAKEST momentum rank upward** — fully liquidate the lowest-ranked held name, then the next, until the amount is raised. Keeps the strongest names (edge-preserving), momentum-consistent, tax-efficient (weak names carry the smallest gains; losses get harvested). Tie-break: within similar rank, sell the lower-gain lot first.
- **Reject: even / proportional selling** — trims winners = the churn we just removed.
- Always **show the sell plan and require confirmation** before placing orders.

---

## Phased checklist

### Phase 0 — Readiness gate (before ANY live order)
- [ ] Wife's Kite Connect API app created (key/secret); login flow + daily token wired to `access_token.json` path for her account
- [ ] Two-key arm: orders place ONLY when `live_mode=1` AND `live_capital>0` (add the second key)
- [ ] Confirm guardrail caps for her capital (order-value cap, per-name max %)
- [ ] Daily **morning reconciliation** cron (~09:20) → `reconcile_holdings()` → email alert on any mismatch
- [ ] Order-failure alert: `send_alert()` on REJECTED / timeout in `_place_cnc_market`
- [ ] Dry-run: flip live with tiny capital (e.g. ₹1–2L), one rebalance cycle, verify fills + reconcile + emails
- [ ] **Soak at 1.0× / small capital for ≥1 month** before scaling

### Phase 1 — Frontend live controls (`/app/momentum-paper`, NAS.tsx pattern)
- [ ] LIVE/PAPER badge + mode toggle (`confirm()` "real money at risk" guard) → `POST /toggle-mode`
- [ ] Kill button (→ flip to paper) → `POST /kill-switch`
- [ ] Live positions view (qty, avg, LTP, MTM, days held) — same layout as paper
- [ ] Reconcile panel (DB vs Kite) → `GET /reconcile`
- [ ] Live-capital input on arm

### Phase 2 — Notifications (reuse `services/notifications.py`)
- [ ] `momentum_eod_report.py` — assemble today's entries/exits + positions + P&L; `send_eod_report()` at ~15:35 (after the 15:15 EOD job), **once/day**
- [ ] Order/reconcile **alert emails** (Phase 0 wiring surfaces here)
- [ ] **Monthly report email** — month-end: NAV curve, trades, holdings, benchmark vs NIFTYBEES, STCG realized
- [ ] (Deferred) WhatsApp via existing Twilio provider — enable `whatsapp_enabled` + twilio config when wanted

### Phase 3 — Cash management
- [ ] Deposit UI: show Kite available cash → deploy full/custom → equal top-up (gate-on) / park (gate-off); confirm before orders
- [ ] Withdraw UI: input amount → weakest-rank-first sell plan → show plan → confirm → execute
- [ ] Light deposit-timing backtest to lock the deposit default

### Phase 4 — Ops hardening
- [ ] EOD-job watchdog: alert if the 15:15 job didn't run (missed stop/rebalance risk)
- [ ] Extend the existing runbook (`docs/MOMENTUM_LIVE_RUNBOOK.md`) with the wife-account specifics

---

## Open items / blockers
1. **Wife's Kite API app + token** — hard blocker; nothing goes live without it.
2. **Live capital amount** for the soak (suggest ₹1–2L to start).
3. Deposit-default backtest (light) — before shipping the deposit UI.
4. Confirm `notifications.py` email path works end-to-end with a test send before relying on it for live alerts.

## Reference
- `services/momentum_paper.py` — the book (live path already wired)
- `services/notifications.py` — email/WhatsApp/in-app, `send_alert` / `send_eod_report`
- `services/nas_eod_report.py` — EOD report template to mirror
- `docs/MOMENTUM_LIVE_RUNBOOK.md` — existing flip/kill/reconcile procedure
- `docs/AURUM_MOMENTUM30_ARMING_SPEC.md` — arming spec (review before go-live)
- `frontend/src/pages/Nas.tsx` — live-controls UI template
