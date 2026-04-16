# ORB Paper Trading — Claude Code Handoff Instructions

## Files in this package

1. `ORB_PAPER_TRADING_PRD_v2.md` — Build spec (the brain dump)
2. `ORB_VWAP_RSI_CPR_Filter_v2.1.pine` — Pine Script source of truth (the reference implementation)
3. `CLAUDE_CODE_HANDOFF.md` — This file (the kickoff instructions)

## Step 1 — Get the files into your repo

In your VS Code project (the NAS app repo), create a `docs/orb-paper-trading/` folder and drop both files there:

```bash
mkdir -p docs/orb-paper-trading
cp ~/Downloads/ORB_PAPER_TRADING_PRD_v2.md docs/orb-paper-trading/
cp ~/Downloads/ORB_VWAP_RSI_CPR_Filter_v2.1.pine docs/orb-paper-trading/
git add docs/orb-paper-trading/
git commit -m "Add ORB paper trading build spec and Pine Script reference"
```

## Step 2 — Open Claude Code in the repo root

```bash
cd /path/to/your/nas-app
claude
```

## Step 3 — Paste this exact prompt to Claude Code

```
I want you to build a new strategy module — ORB paper trading — and integrate 
it into this existing app.

Read these two files in order:
1. docs/orb-paper-trading/ORB_PAPER_TRADING_PRD_v2.md
2. docs/orb-paper-trading/ORB_VWAP_RSI_CPR_Filter_v2.1.pine

Then explore this codebase, focusing specifically on the existing ATM/OTM 
strategy page. I need you to understand:
- How Kite Connect auto-login works
- How live prices are fetched (WebSocket? polling? what cadence?)
- The page UI structure and styling
- Which database tables exist

After reading and exploring, write a 1-page implementation plan as a new 
markdown file at docs/orb-paper-trading/IMPLEMENTATION_PLAN.md. The plan 
should cover:
- Which existing files you'll touch
- Which new files you'll create  
- Which dependencies you'll add (and why)
- How you'll integrate with the existing Kite live-price infrastructure
- Your Phase 1 acceptance test plan (matching signals against TradingView)

DO NOT write any implementation code yet. Just read, explore, and produce 
the plan. I'll review and approve before you start Phase 1.

Per the PRD's Section 14, ask me before:
- Choosing between web frameworks if multiple are present
- Adding any new Python dependency
- Any UI/design decision that diverges from the existing ATM/OTM page

Start now.
```

## Step 4 — Review the plan

Claude Code will produce `IMPLEMENTATION_PLAN.md`. Read it carefully. Things to verify:

- [ ] Plan correctly identifies the existing Kite live-price mechanism (not creating a duplicate WebSocket)
- [ ] Plan creates new files in `strategies/orb_paper/` (or whatever your existing strategies folder is called)
- [ ] Plan adds new SQLite tables, doesn't replace existing ones
- [ ] Plan acknowledges the lookahead-safe CPR fetch pattern from PRD Section 2.4
- [ ] Plan describes the Phase 1 acceptance test (run engine on 30 days of historical data, compare signal count to TradingView)

If anything is wrong, push back: "Re-read PRD Section X — your plan doesn't match it."

## Step 5 — Approve and start Phase 1

Once the plan looks good, reply:

```
Plan approved. Start Phase 1. Remember:
- Phase 1 ends when all tests in test_signals.py and test_cpr.py pass AND 
  the engine produces signals matching TradingView for the last 30 days of 
  Bank Nifty 5-min data.
- Do NOT proceed to Phase 2 until I've reviewed the Phase 1 acceptance 
  test results.
- Show me the test output after each test file is complete.
```

## Step 6 — Iterate phase by phase

Phases are deliberately small so you can catch issues early:
- **Phase 1** — Core engine + tests (2 days)
- **Phase 2** — Paper trading + DB (1 day)
- **Phase 3** — Live data integration in observation mode (1 day)
- **Phase 4** — Dashboard UI (2 days)
- **Phase 5** — Polish + alerts (0.5 day)
- **Phase 6** — 5 days of live paper trading

After each phase, Claude Code should report what got built, what tests pass, and what it needs from you. Don't let it skip phases.

## Common things Claude Code might get wrong (watch for these)

1. **Reusing your existing Kite setup.** If it tries to write its own KiteConnect login, stop it. Point at the file in your repo that already does this.

2. **CPR data fetch.** If you see code like `prev_day = today - timedelta(days=1)`, this is wrong (breaks on Mondays and post-holidays). The PRD's Section 2.4 has the right pattern.

3. **Trying to do too much at once.** If Claude Code starts Phase 4 (UI) before Phase 1 acceptance gate passes, stop it. The signal correctness MUST be validated before anything else.

4. **Ignoring Pine Script reference.** If Claude Code's signal output diverges from TradingView, it'll be tempted to "improve" the logic. Don't let it. The Pine Script is the spec.

5. **Skipping tests.** Section 7 of the PRD lists 12 tests that MUST pass before live deployment. If Claude Code says "the test pattern is just a suggestion," it's wrong.

## When something breaks during the live paper trading phase

The most likely failure modes during Phase 6 are:

1. **CPR mismatch with TradingView** — check `orb_daily_state.prev_day_date` column. Should match what TradingView used. If not, the Section 2.4 fetch pattern is broken.

2. **No signals firing** — check the signal eval logs (PRD Section 11.1). The structured log will show which filter blocked.

3. **EOD square-off didn't fire** — check the scheduler. Verify timezone is `Asia/Kolkata` not UTC.

4. **Position MTM looks wrong** — check `orb_mtm_history` table. Verify LTP being captured correctly.

For any of these, the audit trail in the DB (Section 11.2) should let you debug without re-running the strategy.

## Going from paper to live (someday, after weeks of paper validation)

When you're ready to go live, the change is small:
- Add a new flag `is_live_mode` to config
- In `paper_trader.py`, branch: if live mode, call `kite.place_order()`; if paper, just record virtually
- Everything else stays the same

DO NOT enable live mode until:
- 30+ trading days of paper trading completed
- Paper P&L is positive (or at least matches the v2.1 backtest)
- All edge cases (gap days, holidays, broker disconnects) handled in paper without errors
- You've personally manually executed at least 5 of the system's signals via Kite to validate execution feasibility (slippage, liquidity)

Good luck. Send the IMPLEMENTATION_PLAN.md back to Claude (the chat) for review before you start Phase 1 if you want a second pair of eyes.
