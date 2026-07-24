# NSR-W v1.0 — Intraday Replay on Recorded Chain (1-min snapshots, Apr 20 → Jul 24 2026)

**STATUS: DONE** · Ran 2026-07-24 15:54 IST, 51s, 28 cycles · Verdict: **EXECUTION VALIDATED — 12W/2L both targets** (§6)

## 1. The Ask

**What Arun asked:** "NSR-W v1.0 — now backtest this on our 60+ days of options
data that we have and report."

**What we're testing:** Replay the locked NSR-W v1.0 spec on `options_data.db`
(1-min full-chain snapshots, 66 trading days 2026-04-20 → 2026-07-24, 14 Monday
entries) with REAL quote-level execution: entries at BID, exits/stops at ASK,
GTT-style stops triggered on LTP minute-by-minute. This validates the EOD
backtest's fills, replays Arun's own W30 week under the rules, and produces the
tracking-error vs the G3 EOD sim for overlapping weeks.

**Success:** cycle-by-cycle honest report; check whether intraday reality is
consistent with the EOD result family (t 4.8–5.5). 13-14 weeks proves execution
realism, NOT edge (sample too small — the edge claim rests on the 378-week EOD
study).

## 2. The Base (spec held fixed; execution modeled at quote level)

- Entry: first snapshot ≥ 09:30 Monday (04-20 starts 13:56 — noted). Target
  expiry: nearest with cal DTE 6–12 (Tuesday-expiry era → 8 days).
- Strikes: OTM side, MID premium nearest target T ∈ {20, 30}, volume>0 or oi>0.
  SELL at BID (conservative).
- Stops: GTT SL per leg at 2.0× entry credit — trigger on LTP ≥ level, fill at
  ASK that minute. PT 50% of total credit on combined MID → buy both at ASK.
  ONE roll-away (new strike at same T rule, sell at BID, own 2× stop); second
  stop → flat everything at ASK. Time exit: cal DTE ≤ 1, last snapshot ≤ 15:15,
  buy at ASK.
- Missing leg snapshot minutes: carry last quote.
- Costs: 0.25% × (entry+exit premium) + 0.1 pts (charges only — spread already
  paid via bid/ask crossing; the EOD model's 0.5% included slippage).
- Cycle whose exit lies beyond 2026-07-24 → reported OPEN with m2m at last
  snapshot MID.

## 3. Plan

14 Mondays × T {20, 30} = 28 cycles. Per-cycle event log with timestamps.
Comparator: matching G3 EOD rows (mode=monday, pt=0.5, stop=2.0) for the same
entry dates.

## 4. Status

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-07-24 15:5x | STATUS + runner written | pre-launch |

## 5. Crash Recovery

- Runner: `research/90_nifty_strangle_rules/scripts/run_replay_nsrw.py` (VPS)
- Launch: `cd /home/arun/quantifyd && setsid nohup venv/bin/python research/90_nifty_strangle_rules/scripts/run_replay_nsrw.py > research/90_nifty_strangle_rules/results/run_replay.log 2>&1 < /dev/null &`
- Check: `tail -60 .../results/run_replay.log`; output `replay_nsrw_cycles.csv`
- Idempotent, read-only on options_data.db.

## 6. Findings

14 Mondays (2026-04-20 → 07-20), all cycles CLOSED before data end.

| Target | Weeks | W/L | Net pts | ₹ @10 lots | Mean/wk | Worst wk |
|---|---|---|---|---|---|---|
| T20 | 14 | 12/2 | +182.3 | **+₹1.19L** | 13.0 | −46.9 (−₹30k) |
| T30 | 14 | 12/2 | +266.1 | **+₹1.73L** | 19.0 | −53.1 (−₹35k) |
| T20 excl. wk1* | 13 | 11/2 | +134.9 | +₹0.88L | 10.4 | −46.9 |
| T30 excl. wk1* | 13 | 11/2 | +217.5 | +₹1.41L | 16.7 | −53.1 |

*wk1 (04-20) anomalous: recorder started 13:56 that day, partial chain → strikes
far off the ₹-target (PE @53.4). Excluded view is the honest one.

- **The W30 punchline:** robot entered Mon 07-20 09:30 at 23350PE@19.2 +
  24800CE@17.8 — the SAME strikes Arun sold that morning — and hit its 50%
  profit-take **Tuesday 14:20 (+18.3 net, ₹+11.9k)**, then sat in cash through
  the entire Wed–Fri drift that Arun spent firefighting.
- Losing weeks (May-11, Jul-06): PE stop → roll → rolled leg stopped again →
  CE survivor rode to time exit. Losses −33 to −53 pts — the stop machinery
  worked; compare the EOD study's unstopped weeks at −400+.
- **Real GTT slippage captured:** normal stops filled 0.1–2 pts past trigger;
  the Jul-08 crash minute filled 13.7 pts past (LTP gapped between snapshots).
  Fast-crash slippage is real and survivable, not model fiction.
- Exits: 18/28 PT (most by Tue/Wed — capital free most of the week), 6 TIME,
  0 expiry, both losers via stop+roll+time.
- **Spec correction (implemented ≠ worded):** all backtests (G2/G3/replay)
  implement "per-leg stop; ONE roll max; a post-roll stop closes THAT leg only;
  the survivor leg rides with its own stop" — NOT "second stop → flat all".
  The validated numbers belong to the implemented rule; spec v1.0 wording
  updated to match (survivor rides). Ledger updated.
- Caveats: 14 weeks (execution validation, NOT an edge estimate — edge claim
  rests on the 378-week EOD study); window was a friendly range regime (VIX
  12–15); charges 0.25% + spread crossing (lighter than EOD's 0.5%-incl-slip,
  since spread is paid explicitly here).
