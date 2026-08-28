# N500M CCRB Rules Have Never Fired — Half the Book Dead Since May

STATUS: **DIAGNOSED — fix pending, needs after-15:40 deploy**
Opened 2026-08-28 12:55 IST · market open, nothing deployed

---

## 1. The Ask

**What you asked:** "make this live paper trading and record daily performances
like the other live paper trading pages" — then "proceed", after I offered to
find out why N500M has not traded since 20 Aug.

**What was actually investigated:** N500M is enabled and in PAPER mode, its four
scheduler jobs run on time, and its data refresh succeeds 27/27 — yet it has
produced 32 trades in four months and none in eight days. Is the signal simply
rare, or is something silently blocking it?

**Answer: something is silently blocking it.** Half the book — all 15 CCRB
rules — has never produced a single trade since deployment in May, and cannot,
by construction. The 32 trades in the book's entire history are all vol-BO.

---

## 2. The Base — what the book is meant to do

`/app/n500m` advertises: *"Per-stock CCRB + vol-BO portfolio · 27 stocks ·
30 rules"*. Two signal families, 15 rules each:

| Family | Setup gate | Where the gate is evaluated |
|---|---|---|
| **vol-BO** | volume multiple + close past PDH/PDL | intraday, off 5-minute bars |
| **CCRB** | today's CPR narrow vs yesterday's context | `precompute_setup`, 09:10 IST, off **daily** bars |

Pipeline: `precompute_setup` (09:10) writes a daily-state row per config with
`setup_qualifies`; `scan_for_signals` (every minute, 09:20–14:00) skips any
config whose stored state does not qualify.

---

## 3. Evidence

**a. Every CCRB row ever written was skipped, for one reason.**

```
setup_reason distribution, all history (n500m_daily_state):
  ccrb    qual=0   n=1230   skip:no_setup_row
  volbo   qual=1   n=1230   candidate (volume + PDH/PDL gate runs intraday)
```

1,230 rows, 100% skipped, one reason. Not a market outcome — a structural one.

**b. Every trade the book has ever taken is vol-BO.**

```
signal_type: {'volbo': 32}          by month: May 9 · Jun 11 · Jul 9 · Aug 3
```

**c. The skip comes from one branch** (`services/n500m_scanner.py:209`):

```python
setup_tbl = daily_setup_table(daily)
today_ts  = pd.Timestamp(today).normalize()
today_setup = setup_tbl.loc[today_ts] if today_ts in setup_tbl.index else None
...
if cfg.signal == "ccrb":
    if today_setup is None:
        reason = "skip:no_setup_row"      # <- always taken
```

**d. Today's daily bar does not exist — not at 09:10, and not mid-session.**

My first hypothesis was the 09:10 schedule: precompute runs five minutes before
the 09:15 open, so today's bar cannot exist yet. Re-running the identical
computation at 12:47, well into the session, refuted the *timing* explanation
and produced a worse one — the bar is still missing:

```
today's daily bar present : 0/15
setup row available now   : 0/15
would qualify right now   : 0/15
```

```
symbol       latest 'day' bar   latest '5minute' bar
DLF          2026-08-27         2026-08-28 12:40:00
HDFCBANK     2026-08-27         2026-08-28 12:40:00
ITC          2026-08-27         2026-08-28 12:40:00
```

**e. Why: the refresher only maintains one timeframe.**
`services/market_data_refresh.py` calls the downloader with
`timeframe="5minute"`. The `day` timeframe is never refreshed during the
session, so `market_data.db` holds no daily bar for today at any hour of the
trading day.

**Root cause:** CCRB's setup gate needs a row keyed on *today* in the daily
setup table. That row requires today's daily bar. Nothing writes today's daily
bar while the market is open. So `today_setup` is always `None`, every CCRB
config is skipped every day, and the 15 CCRB rules have been dead since the
book was deployed. vol-BO is unaffected because its gate reads 5-minute bars,
which are current.

---

## 4. The fix (NOT YET APPLIED)

`daily_setup_table` needs exactly **one** field from today's bar:

```python
today = df.iloc[i]
"today_open": float(today["open"]),
"today_cpr_width_pct": today_w / today["open"]      # today_w comes from PREV day's HLC
```

Everything else — today's CPR width, previous range, previous CPR — is derived
from prior days, which are present. So the missing input is today's **open**,
and that is already in the database: it is the open of today's first 5-minute
bar, refreshed every five minutes.

Candidate fixes, cheapest first:

1. **Synthesise today's daily row from the first 5-minute bar** and move the
   CCRB precompute to ~09:20 (after the first bar closes). Smallest change,
   uses data already maintained, no new download load.
2. Add the `day` timeframe to the intraday refresher for these 27 symbols.
   More data, more API calls, same effect.
3. Re-run precompute lazily inside the scan when the state row is missing.
   Most robust to restarts, largest change.

**Preference: (1).**

### Before deploying

This turns on 15 rules that have never traded. They came from a bake-off, but
they have no live or paper record at all, so treat the switch-on as a change to
the strategy, not a bug fix:

- [ ] Confirm the synthesised open matches the true daily open on history
      (compare first 5-min open vs the daily bar's open across ~30 sessions).
- [ ] Replay the CCRB gate over the last 3 months to see how often it *would*
      have qualified, and sanity-check that against the bake-off's expectation.
- [ ] Deploy **after 15:40 IST**, then watch the first CCRB signals closely.

---

## 5. Status log

| Date/time (IST) | Event |
|---|---|
| 2026-08-28 12:47 | Read-only diag: 15/30 configs qualify, 0 would fire; clean split — every CCRB fails setup, every vol-BO passes |
| 2026-08-28 12:49 | All 32 historical trades are vol-BO; CCRB has never fired |
| 2026-08-28 12:50 | `setup_reason` = `skip:no_setup_row` on all 1,230 CCRB rows |
| 2026-08-28 12:52 | Hypothesis "09:10 is before the open" REFUTED — bar missing mid-session too |
| 2026-08-28 12:53 | Confirmed: `day` timeframe stale at 2026-08-27; only `5minute` is refreshed |
| 2026-08-28 12:55 | Root cause established; fix deferred to a post-15:40 change with its own evidence |

## 6. Crash recovery / how to re-check without Claude

```bash
cd /home/arun/quantifyd

# Is CCRB still being skipped for the same reason?
venv/bin/python3 -c "
import sqlite3
c=sqlite3.connect('file:backtest_data/n500m_trading.db?mode=ro',uri=True)
print(dict(c.execute('select setup_reason, count(*) from n500m_daily_state where signal_type=\"ccrb\" group by 1')))"

# Is today's daily bar present yet?
venv/bin/python3 -c "
import sqlite3
c=sqlite3.connect('file:backtest_data/market_data.db?mode=ro',uri=True)
print(c.execute(\"select max(date) from market_data_unified where symbol='DLF' and timeframe='day'\").fetchone())"
```

Fixed when the first query stops returning `skip:no_setup_row` for every row.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `services/n500m_scanner.py` | holds the bug at line ~209 (`precompute_setup`) | yes — untouched so far |
| `services/market_data_refresh.py` | refreshes `5minute` only | yes — untouched so far |
| `research/31_cpr_compression_breakout/scripts/signals_ccrb.py` | `daily_setup_table`, needs `today["open"]` | yes |
| this file | the forensic record | yes |
