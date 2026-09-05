# Cash Handling Unification — Capital Ledgers, Fund Flows and the Capital Desk

**STATUS: IN PROGRESS** (started 05-Sep-2026, Saturday — no market open, restart safe)

## 1. The Ask

**What Arun asked:**
> "build the IPO page like True north and OA ... however we have a 50-50 sleeve page handling cash
> deposits, but withdrawal sits only in TN, not even in OA. we need to make things uniform... we can
> move all the cash handling to 50-50 sleeve and call it by a different name ... We will go with
> TN 40 / OA 40 / IPO 20 for now... however for me to brng in sufficuent cash to make this
> proporation will take a while, so whenever i bring cash, we shud be putting into OA and IPO to
> match the prportions wrt TN as the base."

**What this document tracks:** the *backend* half — giving every book a truthful capital ledger so
that cash can be handled uniformly at all. Exploration found the fragmentation is not cosmetic;
three defects touch real money. The UI half (Capital Desk page, OA parity, IPO page) is tracked in
the approved plan and depends on this landing first.

**Explicitly NOT in scope:** no entry, exit, stop, trail, sizing, gate or selection rule is changed
on any book. Per `.claude/CLAUDE.md`, this is raised as its own backend change rather than smuggled
into a UI task.

## 2. The Base — defects being fixed

| ID | Defect | Evidence |
|---|---|---|
| **D1** | Open Alpha's REAL money is untracked and unwithdrawable. `sleeves_api.py:36` points `STATE` at `bluesky_paper_state.json` — the retired paper book. `oa_real_state.json` (₹4,46,348) had no `capital`, `cash` or `fund_flows`; `mark()` hardcoded `cash=0` so NAV ignored cash and returns were `pnl/invested` | `sleeves_api.py:35-42`, `oa_real.py:122` |
| **D2** | The dividend engine can declare a dividend on Arun's own deposit. `TrueNorthBook.net_flows_since()` reads `mp_state['fund_flows']`; `momentum_paper.py` never writes that key, so a TN deposit is counted as **profit** | `dividend_engine.py:195-200`; live `mp_state` has no such key |
| **D3** | The sleeves page's TN leg is a fake — raw `INSERT OR REPLACE` on `mp_state` instead of the hardened `cash_deposit()`/`cash_withdraw()`. No deploy plan, no CASHIETF unsweep, no weakest-first sell, no `mp_fills` audit row | `sleeves_api.py:224-260` |
| **D4** | Consequence of D3: a ₹1L TN withdrawal 409s on `/sleeves` and executes cleanly on `/momentum-paper` | `_tn_flow` caps at ledger cash ₹27,607; CASHIETF holds ₹3.3L |
| **D5** | Two non-atomic legs behind one `window.confirm` — a partial failure leaves a half-executed flow | `Sleeves.tsx:140-153` |
| **D6** | No target-allocation concept exists anywhere. Only cross-book ratio in the repo is `Math.round(n/2)` | `Sleeves.tsx:118` |
| **D7** | Locking inconsistent: the 2026-08-05 hardening reached `bluesky_paper.py` only. `oa_real.py` wrote state unlocked and in-place from a **per-minute** cron; `momentum_paper._conn()` has no busy timeout and no WAL | `oa_real.py:132`, `momentum_paper.py:197` |

## 3. Plan

| # | Change | File | State |
|---|---|---|---|
| 1 | Capital ledger (`capital`/`cash`/`fund_flows`), idempotent migration, lock + atomic save, NAV = value + cash, flow-neutral returns, `deposit()`/`withdraw()`/`status()` (alert-and-ledger only, never places orders) | `services/oa_real.py` | **DONE, tested, deploying** |
| 2 | Write `fund_flows` on every TN deposit/withdraw; backfill known flows since inception | `services/momentum_paper.py` | pending |
| 3 | `timeout=` + `PRAGMA journal_mode=WAL` | `momentum_paper._conn()` | pending |
| 4 | Delete `_tn_flow`; call the real `cash_deposit`/`cash_withdraw`. Repoint the OA leg at `oa_real`. Fix the two false docstrings | `services/sleeves_api.py` | pending |
| 5 | Repoint `OpenAlphaBook` at `oa_real_state.json` | `services/dividend_engine.py` | pending |
| 6 | Target allocation store `{TN:.40, OA:.40, IPO:.20}` + router + drift | new `backtest_data/allocation_targets.json`, `sleeves_api.py` | pending |

## 4. Status log

| Time (IST) | Event | Notes |
|---|---|---|
| 05-Sep 20:5x | Exploration complete, plan approved | 3 agents; VPS canonical, laptop repo stale |
| 05-Sep 21:0x | `oa_real.py` rewritten | capital ledger + lock + atomic save |
| 05-Sep 21:1x | Migration dry-tested on a COPY of live state | first pass gave `cash = -0.68` (stored `invested` was rounded below true position cost) — fixed to take `max(invested, cost)`; re-tested: capital ₹4,46,348.68, cash ₹0.00, idempotent |

## 5. Crash recovery

**Nothing is destructive yet.** The live state file is backed up before every write:

```bash
ssh arun@94.136.185.54 'ls -la /home/arun/quantifyd/backtest_data/oa_real_state.json*'
# restore:
ssh arun@94.136.185.54 'cp /home/arun/quantifyd/backtest_data/oa_real_state.json.bak-<ts> \
                           /home/arun/quantifyd/backtest_data/oa_real_state.json'
```

To check what landed:
```bash
ssh arun@94.136.185.54 'cd /home/arun/quantifyd && venv/bin/python -c "
import json; s=json.load(open(\"backtest_data/oa_real_state.json\"))
print({k: s.get(k) for k in (\"capital\",\"cash\")}); print(len(s.get(\"fund_flows\",[])), \"flows\")"'
```

The migration is **idempotent** — re-running it on an already-migrated file is a no-op, so a partial
run can simply be repeated. `git -C /home/arun/quantifyd diff` shows uncommitted code changes.

Crons are weekday-only (`* 9-15 * * 1-5`), so no `mark` can race this work on a Saturday.

**Files safe to inspect:** anything under `services/`, `backtest_data/*.json`.
**Do NOT hand-edit:** `backtest_data/oa_real_state.json` while a `mark` may be running; take
`oa_real_state.lock` first.

## 6. Files

| File | Purpose | Committable |
|---|---|---|
| `services/oa_real.py` | OA real book + capital ledger | yes |
| `services/sleeves_api.py` | money-flow endpoints | yes |
| `services/momentum_paper.py` | TN engine + hardened cash flows | yes |
| `services/dividend_engine.py` | quarterly dividend policy | yes |
| `backtest_data/oa_real_state.json` | live book state | NO — data |
| `backtest_data/allocation_targets.json` | TN/OA/IPO targets | yes (small) |
| `services/_oa_real_new.py` | staging copy used for dry-tests | NO — delete after cutover |

## 7. Findings

- The **D2 dividend defect is the most dangerous** of the seven: it is silent, it favours paying out
  capital, and it gets worse precisely as Arun funds the book toward 40/40/20. It must land before
  the next quarterly declaration.
- `oa_real.py`'s stored `invested` (₹4,46,348) was **rounded below** the true position cost
  (₹4,46,348.68). Anything deriving cash as `capital − cost` from the stored value goes negative.
- The seeded OA book has **no `cash` concept at all**, so its published return has been
  `pnl / invested` — correct only while capital never moves, which is about to stop being true.
