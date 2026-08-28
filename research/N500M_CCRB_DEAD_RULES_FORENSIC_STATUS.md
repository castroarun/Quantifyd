# N500M CCRB Rules Have Never Fired — Half the Book Dead Since May

STATUS: **FIXED IN CODE — activates at the next pre-open restart (Mon 09:00). No manual restart taken.**
Opened 2026-08-28 12:55 IST · fix committed 13:20 IST · market was open throughout, nothing restarted

---

## 1. The Ask

**What you asked:** "make this live paper trading and record daily performances
like the other live paper trading pages", then "proceed" — find out why N500M
has not traded since 20 Aug, and prepare the fix.

**What was investigated:** N500M is enabled and in PAPER mode, its four
scheduler jobs run on time, and its data refresh succeeds 27/27 — yet it had
produced 32 trades in four months and none in eight days.

**Answer: half the book has never worked.** All 15 CCRB rules have been skipped
every day since deployment in May and could not have fired. Every one of the 32
trades in the book's history is vol-BO.

---

## 2. The Base

`/app/n500m` advertises *"Per-stock CCRB + vol-BO portfolio · 27 stocks ·
30 rules"* — two families, 15 rules each:

| Family | Setup gate | Evaluated |
|---|---|---|
| **vol-BO** | volume multiple + close past PDH/PDL | intraday, off 5-minute bars |
| **CCRB** | today's CPR narrow vs yesterday's context | `precompute_setup`, off **daily** bars |

`precompute_setup` writes a daily-state row per config with `setup_qualifies`;
`scan_for_signals` (every minute, 09:20–14:00) skips anything that does not
qualify.

---

## 3. Evidence

**a. Every CCRB row ever written was skipped, for one reason.**

```
n500m_daily_state, all history:
  ccrb    qual=0   n=1230   skip:no_setup_row
  volbo   qual=1   n=1230   candidate (volume + PDH/PDL gate runs intraday)
```

**b. Every trade the book has taken is vol-BO.**
`{'volbo': 32}` — May 9 · Jun 11 · Jul 9 · Aug 3.

**c. The skip comes from one branch** (`n500m_scanner.py` ~line 209): if the
setup table has no row keyed on today, the config is written off as
`skip:no_setup_row`.

**d. Today's daily bar is absent — and not because of the 09:10 schedule.**
First hypothesis was that precompute at 09:10 runs before the 09:15 open.
Re-running the identical computation at 12:47, mid-session, **refuted it** —
the bar was still missing. `market_data.db` holds `day` bars only after the
close (`day` latest 2026-08-27 while `5minute` was current to 12:40), because
`market_data_refresh.py` refreshes `timeframe="5minute"` only.

**Root cause:** CCRB's gate is keyed on today; the stored daily series has no
today during the session; so the gate is never evaluated. vol-BO is unaffected
because its gate reads 5-minute bars.

---

## 4. Choosing the fix — and rejecting two of them

**Rejected: substitute the first 5-minute open for the daily open.**
Measured first. Against the stored database the two matched exactly on only
**52.7%** of 1,480 sessions (worst gap 2.66%) — against a gate threshold of
0.5%, an error larger than the thing being measured. That looked fatal.

**But the 52.7% was measuring a different defect.** Against the live Kite API
the same two values are identical — 30/30 sessions on each of three symbols.
Pulling both series fresh from the API and comparing them to what is stored:

| check | DLF | HDFCBANK | ITC |
|---|---|---|---|
| API daily open == API first-5min open | 30/30 | 30/30 | 30/30 |
| DB daily open == API daily open | 29/30 | 29/30 | 29/30 |
| **DB first-5min == API first-5min** | **14/30** | **17/30** | **19/30** |

So the relationship holds perfectly; it is the **stored 5-minute series** whose
first bar of the day is wrong on roughly half of sessions. Stored daily bars are
fine. See §6 — this is a separate and larger problem.

**Rejected: fetch today's daily bar and store it.**
Actively dangerous. `data_manager._store_data` inserts only timestamps it does
not already hold —

```python
df_new = df_insert[~df_insert['date'].isin(existing_dates)].copy()
```

— so a partial daily bar written at 09:20 would keep its intraday high/low/close
forever and corrupt the daily series that ATR, the setup table and every
backtest read.

**Chosen: fetch today's daily candle at precompute time, in memory only.**
`daily_setup_table` needs exactly one field from today's row — `today["open"]`
— and today's CPR is derived from *yesterday's* H/L/C, which is already stored.
Kite serves the forming daily candle intraday with the open already fixed by the
first trade (verified live: DLF 680.00, HDFCBANK 710.85, ITC 269.00, HAL
4884.10, ASIANPAINT 2645.00 — all matching the first 5-min open exactly).
Nothing is persisted, so the stored daily series stays end-of-day and correct.

Precompute moves **09:10 → 09:20**: at 09:10 there is no open to fetch.

---

## 5. Verification

**The gate now evaluates** (patched code, today's live data, nothing written):

```
today's daily bar available — before fix: 0/15   after fix: 15/15
CCRB configs whose setup qualifies today: 0/15
this morning's 09:10 run stored: {'skip:no_setup_row': 15}
```

Zero qualifying today is the expected outcome, not a failure: the gate is now
being *asked* rather than skipped, and the historical rate is ~11.7%.

**Expected firing rate** — replaying the gate over the last 120 days, where the
daily bars exist and the computation is exact:

| | |
|---|---|
| config-sessions evaluated | 1,245 |
| setups qualifying | 146 (**11.7%**) |
| days with at least one qualifying config | 72 |
| mean qualifying configs on such a day | 2.0 |
| busiest | 2026-05-14 (6), 2026-07-21 (5) |

Per-config rates run 4.8% (HAL) to 25.3% (CHENNPETRO). Qualifying is the setup
gate only — the intraday trigger still has to fire on top, so actual trades will
be a fraction of these. Expect a trickle, not a flood.

---

## 6. ⚠️ Separate finding: the stored 5-minute first bar of the day is unreliable

Turned up while testing the rejected fix, and it is the more consequential
finding of the two.

**What:** the first 5-minute bar of each session, as stored, disagrees with
Kite's final value on roughly half of sessions (14–19 of 30 match, across three
symbols). Differences are small — 0.1% to 0.6% — but systematic.

**Why, most likely:** the refresher runs every 5 minutes and captures the 09:15
candle while it is still forming; the writer then never corrects a timestamp it
already holds, so the partial values are frozen permanently. The module's own
docstring claimed "Idempotent — uses INSERT OR REPLACE", which is not what the
code does; corrected in this commit (docstring only, no behaviour change).

**Who reads this data:** vol-BO (the half of N500M that does trade), ORB, and
every intraday backtest run off `market_data.db`.

**Not fixed here.** It is a data-layer change affecting every consumer and needs
its own testing and its own deploy. Logged in TODO.md.

---

## 7. Status log

| Date/time (IST) | Event |
|---|---|
| 08-28 12:47 | Read-only diag: clean split — every CCRB fails setup, every vol-BO passes |
| 08-28 12:49 | All 32 historical trades are vol-BO; CCRB has never fired |
| 08-28 12:50 | `setup_reason` = `skip:no_setup_row` on all 1,230 CCRB rows |
| 08-28 12:52 | "09:10 is before the open" REFUTED — bar missing mid-session too |
| 08-28 12:53 | Confirmed `day` timeframe never refreshed intraday; only `5minute` is |
| 08-28 13:00 | First-5min substitute measured at 52.7% — fix rejected |
| 08-28 13:05 | Live API: daily open == first-5min open 30/30 → the 52.7% is a STORED-DATA defect |
| 08-28 13:08 | Store-the-bar fix rejected: append-only writer would freeze a partial daily bar |
| 08-28 13:12 | Fix applied in code: fetch today's candle in memory; precompute 09:10 → 09:20 |
| 08-28 13:15 | Verified 0/15 → 15/15 setup rows; expected rate 11.7% from a 120-day replay |
| 08-28 13:20 | Committed. **No restart** — the 09:00 Mon-Fri pre-open cron activates it before Monday |

---

## 8. Deployment

**No manual restart was taken and none is needed.** The market was open for the
whole of this work (fix committed 13:20, market closes 15:30). The existing
`0 9 * * 1-5 preopen_restart.sh` cron restarts the service before Monday's
session, and precompute now runs at 09:20 — so the fix is live for Monday
without anyone touching the trading process.

**Monday morning check:**

```bash
sudo journalctl -u quantifyd --since 09:20 | grep 'precompute_setup'
# want: "... N configs, M qualify today (15 CCRB configs had today's bar)"

cd /home/arun/quantifyd && venv/bin/python3 -c "
import sqlite3
c=sqlite3.connect('file:backtest_data/n500m_trading.db?mode=ro',uri=True)
print(dict(c.execute('select setup_reason, count(*) from n500m_daily_state '
                     'where signal_type=\"ccrb\" and trade_date=date(\"now\") group by 1')))"
# want: skip:setup_gate and/or qualifies — NOT skip:no_setup_row
```

If it still says `skip:no_setup_row` for all 15, the Kite fetch inside
`append_today_daily` is failing; the warning line names the symbol and reason.

**Rollback:** revert the commit and restart after 15:40. The change is confined
to `n500m_scanner.py` (one new function plus one call) and the precompute cron
minute; nothing else reads either.

---

## 9. Files

| File | Change | Committable |
|---|---|---|
| `services/n500m_scanner.py` | `append_today_daily()` + call in `precompute_setup` | yes |
| `app.py` | precompute cron 09:10 → 09:20 | yes |
| `services/market_data_refresh.py` | docstring corrected (no behaviour change) | yes |
| this file | the forensic record | yes |
