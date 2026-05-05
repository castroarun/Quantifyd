# MST Index Strategy — Design & Implementation Spec

**For: Claude Code, future implementation session**
**Status:** SPEC ONLY — do not start coding until user gives explicit go-ahead.
**Source research:** `research/35_nifty_bnf_master_child_supertrend/` (RESULTS.md has the full backtest evidence)
**Target app:** Quantifyd, hosted on Contabo VPS at `http://94.136.185.54:5000/app`
**Final page URL:** **`http://94.136.185.54:5000/app/mst`**
**Page route within SPA:** `/app/mst` (React component `Mst.tsx`)
**Reference page (mirror this exactly):** `/app/orb` ([frontend/src/pages/Orb.tsx](../../frontend/src/pages/Orb.tsx)) — structure, layout, conventions, rules block, metrics, sections all match ORB

---

## 1. What this strategy is

An always-on, live-trading options strategy on **NIFTY 30-min** that builds long call condors when the trend is up and long put condors when the trend is down.

- **MST (Master SuperTrend)** sets the directional bias — long or short. On activation, the system enters a **debit spread** (bull call when long, bear put when short) for the next NIFTY weekly **Tuesday** expiry with at least 6 DTE.
- **CST (Child Stochastic)** signals exhaustion within the active MST trend. On the **first** CST trigger inside an active weekly expiry cycle, the system adds a **contra credit spread** (bear call when long, bull put when short) to convert the position into a **long condor**. Subsequent CSTs within the same expiry cycle are informational only.
- All four legs squared off at **T-1 EOD (Monday 15:25 IST, or Friday 15:25 if Monday is a market holiday)** — never carried to expiry day.
- **Phase 1 is LIVE TRADING with 1 lot.** Not paper. The user wants to validate the system with real money at minimum size from day one.

> **Expiry day clarification:** As of 2026-05-05 NIFTY weekly options expire on **Tuesday** (per user, who is the live trader on this account). The implementation MUST query Kite at startup (`kite.instruments('NFO')` + filter `name='NIFTY'`, `instrument_type='CE'/'PE'`) to fetch actual exchange-published expiry dates rather than hardcoding "Tuesday" — these are subject to SEBI/NSE regulatory changes and the exchange already shifts expiry dates around holidays.

---

## 2. The signal rules (locked from research/35)

### 2.1 MST — SuperTrend(ATR=21, multiplier=5.0) on NIFTY50 30-min

- Compute SuperTrend on 30-min OHLC using Wilder ATR with period=21, multiplier=5.0
- Direction `+1` (long bias) or `-1` (short bias)
- A direction flip occurs at the **close of bar `i`** when close crosses the active band

### 2.2 MST entry filter — break-of-extreme (mandatory)

When MST flips at close of bar `i`:

- Record `flip_high = high[i]` and `flip_low = low[i]`
- For LONG flip: bias becomes "armed long" but does NOT activate until a subsequent bar prints `high > flip_high` (intra-bar break is enough — a stop-buy at `flip_high` would fill)
- For SHORT flip: bias becomes "armed short" until `low < flip_low`
- If MST flips again before the breakout occurs, the prior flip is **discarded** — no trade was triggered

This is the validated edge: lifts MFE/MAE from 2.13× → 3.62× at the cost of ~2 hours entry lag and ~6% of flips filtered.

### 2.3 CST — Stochastic(14, 3, 3) on the same 30-min bars

- `%K_raw[i] = 100 × (close[i] − min(low, k=14)) / (max(high, k=14) − min(low, k=14))`
- `%K = SMA(%K_raw, 3)` (smoothing)
- `%D = SMA(%K, 3)`
- All computed on close of completed 30-min bars

### 2.4 CST trigger rule (the original cross-from-extreme rule)

| MST state | CST trigger condition (at bar close) |
|---|---|
| **LONG (active)** | `%K[i-1] >= %D[i-1]` AND `%K[i] < %D[i]` AND `%K[i-1] >= 80` |
| **SHORT (active)** | `%K[i-1] <= %D[i-1]` AND `%K[i] > %D[i]` AND `%K[i-1] <= 20` |

**Notes:**
- The `%K_prev >= 80` (or `<= 20`) gate is what makes this lead the MAE peak (+8.8 bars on NIFTY 30-min) rather than confirm it. Tested alternative ("exit-zone" variant — requires K to also close back below 80) was rejected: lead time turned to −18 bars.
- **K does NOT need to be below 80 on the trigger bar.** The cross can fire while K is still at 85. That is by design.
- If MST is "armed but not active" (waiting for breakout), CST events are ignored.
- If MST flips, any pending CST state and any open position is reset.

### 2.5 Multi-CST policy (one condor per weekly expiry cycle, with pyramiding)

Empirically (research/35, NIFTY 30-min p21,m5.0): **median 4 CSTs per MST trend, mean 5.4, max 24**.
Empirically (research/36): **67% of FIRST CSTs are false alarms — trend continues after the hedge**.

Single-CST-per-trend policy would over-trade. Plain "build condor on first CST" policy would build hedges at the wrong moment 2/3 of the time. Policy combines both findings:

| Condition | Action |
|---|---|
| First CST in an active MST trend, in current weekly expiry cycle | Add contra credit spread → CONDOR_OPEN_L1 |
| Subsequent CSTs (level 1, no pyramid yet) | Informational — log to events, do nothing to position |
| **Pyramid trigger fires (D AND B; see §2.6)** | **Add SECOND debit spread (current spot ATM-anchored) → DEBIT_OPEN_L2** |
| First CST after pyramid (in level 2) | Add SECOND contra credit spread → CONDOR_OPEN_L2 |
| Subsequent triggers/CSTs after level 2 | Informational only — pyramid capped at level 2 |
| MST flip (any direction, any level) | Close all open legs, reset state |
| First CST in a NEW expiry cycle (because we rolled over to next week) | Add contra credit spread for that new week's condor (back to level 1) |

### 2.6 Pyramid trigger — D AND B (research/36)

**Trigger fires when BOTH conditions are true at the same 30-min bar close:**

| Condition | LONG MST | SHORT MST |
|---|---|---|
| **D** — price action | Two consecutive 30-min closes ABOVE the most recent CST bar's `high` | Two consecutive 30-min closes BELOW the most recent CST bar's `low` |
| **B** — momentum return | %K has been below 70 since the last CST (left the OB zone) AND has now returned to ≥ 80 | %K has been above 30 since the last CST (left the OS zone) AND has now returned to ≤ 20 |

Validated (research/36 on 6.3-year extended period: 302 trends, 1,495 CSTs):
- **Coverage:** 80% of trend continuations correctly flagged
- **False positive rate:** **19%** (1 in 5 fires when trend was actually exhausting — extended-period estimate; the 2-year sample suggested 13%, but the larger sample tightens this to ~19% with 95% CI of 16-21%)
- **Median lead time:** 36 bars (~18 hours) before the trend's MFE peak after the CST

The single-trigger D alone catches 99% but with 31% FP rate — too aggressive for pyramiding.
The single-trigger B alone catches 80% with 20% FP — close to the combo but slightly worse.
**The AND combination is the right rule for pyramiding** because false-pyramid cost > missed-pyramid cost.

---

## 3. Position state machine (with pyramid)

```
              ┌────────────────────────────┐
              │ NO_POSITION                │
              └─────┬──────────────────────┘
                    │ MST flip + break-of-extreme confirmed
                    ▼
              ┌────────────────────────────┐
              │ ARMED                      │
              │ Waiting for break of       │
              │ flip-bar high/low          │
              └─────┬──────────────────────┘
                    │ break confirmed
                    ▼
              ┌────────────────────────────┐
              │ DEBIT_OPEN_L1              │
              │ 1× bull call (or put)      │
              └─────┬──────────────────────┘
                    │ first CST in active week
                    │ (credit ≥ ₹1,000/lot → §4.3)
                    ▼
        ┌───────────┴────────────┐
        │                        │ credit too low
        │ credit OK              ▼
        │             ┌──────────────────────────┐
        │             │ ROLL_PENDING             │
        │             │ Close debit, open fresh  │
        │             │ next-week reset condor   │
        │             │ (per §4.3, back to L1)   │
        │             └─────┬────────────────────┘
        ▼                   │
  ┌──────────────┐          │
  │ CONDOR_      │          ▼
  │ OPEN_L1      │     ┌─────────────────────┐
  │ 1×deb+1×cred │     │ DEBIT_OPEN_L1       │
  └──────┬───────┘     │ or CONDOR_OPEN_L1   │
         │             │ (week N+1)          │
         │             └─────────────────────┘
         │ pyramid trigger D AND B (§2.6)
         ▼
  ┌──────────────────────────────────┐
  │ DEBIT_OPEN_L2                    │
  │ 2×deb (level 1 + new at current  │
  │ ATM) + 1×cred                    │
  └──────┬───────────────────────────┘
         │ next CST (credit ≥ ₹1,000/lot)
         ▼
  ┌──────────────────────────────────┐
  │ CONDOR_OPEN_L2                   │
  │ 2×deb + 2×cred — MAX LEVEL       │
  │ Further triggers/CSTs: log only  │
  └──────────────────────────────────┘

Forks (apply to ANY state):
  MST flips opposite     → close ALL legs → ARMED in new direction
  Kill switch toggled    → close ALL legs → halted (no new entries)
  T-1 EOD               → close ALL legs → if MST still active, immediately
                          rollover to next weekly expiry (≥6 DTE rule, fresh
                          ATM, L1) → DEBIT_OPEN_L1; else → NO_POSITION/ARMED.
                          T-1 = previous trading day before each held
                          position's expiry (typically Monday for Tue expiry,
                          shifts on holidays; see §4.6).
```

The pyramid is **capped at level 2**. After CONDOR_OPEN_L2:
- Further D-AND-B triggers → logged, ignored
- Further CSTs → logged, ignored
- Position runs to T-1 EOD or MST flip

Maximum exposure at level 2:
- 8 open legs total (4 debit + 4 credit)
- ~2× margin requirement vs standard condor
- 2× directional risk vs standard condor (compensated by 2× hedges)

---

## 4. Spread structure (live, 1 lot per leg)

NIFTY weekly options · 50-point strike interval · 1 lot = 75 contracts (read from `FNO_LOT_SIZES['NIFTY']`).

### 4.1 DTE rule at entry — ≥6 DTE, NIFTY weekly Tuesday expiry

The ≥6 calendar-DTE rule applies to **every new entry** (initial activation, post-MST-flip re-arm, weekly rollover at T-1, pyramid level-2 entry).

For Tuesday weekly expiry (no holidays):

| Activation day | DTE to this Tue | DTE to next Tue | DTE to Tue after | Expiry used | DTE at entry |
|---|---|---|---|---|---|
| Monday | 1 | 8 | 15 | **next Tuesday** (skip current week) | **8** |
| Tuesday (post-15:30) | — | 7 | 14 | **next Tuesday** | **7** |
| Wednesday | — | 6 | 13 | **next Tuesday** | **6** |
| Thursday | — | 5 (fails) | 12 | **Tuesday after** (skip a week) | **12** |
| Friday | — | 4 (fails) | 11 | **Tuesday after** | **11** |

Median entry DTE ~7-8 across the week. After median CST lag of ~2.2 days, 4-6 DTE remain on the bear call spread — enough theta for meaningful credit (~₹14-22/share).

**Note on bumpy entry days:** Thursday and Friday entries skip ahead by a week because of the weekend gap. This is the cost of the strict ≥6 DTE rule. Practical impact is small: with ~3.9 MST flips/month, only a few entries per month land on Thu/Fri.

### 4.2 Standard spread structure (default — used when credit at CST time meets threshold)

**Long MST → Long Call Condor (level 1) → optional Pyramid (level 2)**

| Step | Trigger | Action | Strike | Notes |
|---|---|---|---|---|
| 1a | MST activates LONG | BUY 1 lot CE | ATM (entry-time spot rounded to 50 = `entry_atm`) | DTE rule §4.1 |
| 1b | MST activates LONG | SELL 1 lot CE | `entry_atm + 200` | same expiry |
| 2a | First CST in active week (credit ≥ ₹1,000/lot) | SELL 1 lot CE | `entry_atm + 400` | same expiry |
| 2b | First CST in active week (credit ≥ ₹1,000/lot) | BUY 1 lot CE | `entry_atm + 600` | same expiry |
| **3a** | **Pyramid (D AND B fires)** | **BUY 1 lot CE** | **current spot rounded to 50 = `pyramid_atm`** | **same expiry** |
| **3b** | **Pyramid (D AND B fires)** | **SELL 1 lot CE** | **`pyramid_atm + 200`** | **same expiry** |
| 4a | Next CST after pyramid (credit ≥ ₹1,000/lot) | SELL 1 lot CE | `pyramid_atm + 400` | same expiry |
| 4b | Next CST after pyramid (credit ≥ ₹1,000/lot) | BUY 1 lot CE | `pyramid_atm + 600` | same expiry |

Level-1 strikes anchored to **MST-entry-time ATM**.
Level-2 strikes anchored to **spot at pyramid-trigger time** (price has moved up by then; level-2 condor is positioned at the new operative range).

**Short MST → Long Put Condor** (mirror)

| Step | Trigger | Action | Strike |
|---|---|---|---|
| 1a | MST activates SHORT | BUY 1 lot PE | `entry_atm` |
| 1b | MST activates SHORT | SELL 1 lot PE | `entry_atm - 200` |
| 2a | First CST (credit ≥ ₹1,000/lot) | SELL 1 lot PE | `entry_atm - 400` |
| 2b | First CST (credit ≥ ₹1,000/lot) | BUY 1 lot PE | `entry_atm - 600` |
| **3a** | **Pyramid (D AND B fires)** | **BUY 1 lot PE** | **`pyramid_atm`** |
| **3b** | **Pyramid (D AND B fires)** | **SELL 1 lot PE** | **`pyramid_atm - 200`** |
| 4a | Next CST after pyramid (credit ≥ ₹1,000/lot) | SELL 1 lot PE | `pyramid_atm - 400` |
| 4b | Next CST after pyramid (credit ≥ ₹1,000/lot) | BUY 1 lot PE | `pyramid_atm - 600` |

### 4.3 Reset structure — when current-week credit is too low

If at the first CST in the current expiry cycle, the bear call (or bull put) credit at standard strikes is **< ₹1,000/lot total**:

1. **Close** the existing debit spread at market (lock in current week's P&L)
2. **Open a fresh condor on next week's expiry** (Tuesday week N+1, ≥ 6 DTE again)
3. The fresh condor uses **Reading D — narrow spot-centered structure**:

```
At reset, current spot = S. Round S to nearest 50 → centered_atm.
Use 100/100/100 strike spacing (NOT 200/200/200) so the deep-ITM bull call
cost is bounded.

Long Call Condor strikes (long MST):
  K1 = centered_atm − 50  (long, slightly ITM)
  K2 = centered_atm        (short, ATM)
  K3 = centered_atm + 50   (short, OTM by 50)
  K4 = centered_atm + 150  (long, OTM by 150)

Long Put Condor strikes (short MST):
  K1 = centered_atm − 150
  K2 = centered_atm − 50
  K3 = centered_atm
  K4 = centered_atm + 50
```

Properties of this reset structure (NIFTY at ~22,750 with 8 DTE, IV ~14%):
- Bull call (K1/K2) cost: ~₹50/share = ~₹3,750/lot
- Bear call (K3/K4) credit: ~₹20/share = ~₹1,500/lot ✓
- Profitable zone width: 50 (between K2 and K3) — narrow but spot-centered
- Spot at midpoint of profitable zone (K2 ≤ S ≤ K3 with S at centered_atm)
- Risk-reward: max profit ~₹4,875/lot vs max loss ~₹2,250/lot (≈ 1:0.46 against — better than standard structure during reset)

**Recursive guard:** if even the new week's bear call credit is < ₹1,000/lot at standard strikes, the fresh condor is built using the reset structure above — same spot-centered logic, sized for the new week's DTE.

> **Open question for the user:** the reset structure above is "Reading D" from prior conversation — narrow 100/100/100 spot-centered. Confirm before live deploy. Alternatives are Reading A (200/200/200 spot-centered, deep-ITM bull call → bad economics) and Reading C (ATM bull, shifted bear call → asymmetric 200/150/200). Default = Reading D.

### 4.4 Order placement

| Order property | Value |
|---|---|
| Type | LIMIT at mid-price |
| Mid calculation | (bid + ask) / 2 from option chain quote |
| Fallback | If LIMIT not filled in 30 seconds → cancel & re-place at MARKET |
| Per-leg quantity | 1 lot = 75 contracts |
| Total legs at full condor | 4 (2 debit + 2 credit) |
| Total legs at reset | 6 in sequence (close 2 + open 4) |
| Order tag | `MST_<state>_<bar_dt>` for traceability |

### 4.5 Exit rules + weekly rollover

| Trigger | Action |
|---|---|
| **T-1 EOD (typically Monday 15:25 IST for Tue expiry; shifts to Friday 15:25 if Monday is a holiday — see §4.6)** | Close all open legs of expiring position at market. Mandatory. THEN: if MST is still active long/short with break confirmed, **immediately open NEW debit spread** for next weekly expiry (≥6 DTE rule), anchored to current spot's ATM. **Pyramid level resets to L1.** |
| MST flips opposite (any time, any state) | Close all open legs at market. Re-arm in new direction. New entry uses ≥6 DTE rule (i.e., this Tuesday's expiry is too close → skip to next Tuesday). |
| Kill switch toggled | Close all open legs at market. Halt entries. |

**No profit-target or stop-loss-based exits.** The condor structure self-caps loss; we let it run until T-1, MST flip, or kill switch.

#### 4.5.1 Rollover semantics — the universal ≥6 DTE rule

**The ≥6 DTE rule is universal.** Every new entry — initial activation, re-arm after MST flip, weekly rollover at T-1, pyramid level-2 entry — must use a weekly Tuesday expiry that is at least 6 calendar days away. There are no exceptions.

Concrete example illustrating the user's "MST flips right before T-1" edge case (now restated for Tuesday expiry):

```
Scenario: MST flips opposite on Monday at 14:30 IST.
  Today is Monday. T-1 of THIS Tuesday's expiry is today at 15:25.

  14:30  MST flip detected (long → short)
         → Action: CLOSE all open legs of this Tuesday's positions at market
         → State: ARMED (short), waiting for break of flip-bar low

  14:30 onwards  System monitors for break-of-extreme.

  15:25  T-1 EOD scheduled fire.
         → If we're still ARMED: nothing to close (already closed at flip)
         → Rollover trigger fires regardless. But ARMED means no rollover entry yet.

  Suppose break of extreme confirms at 14:50 IST:
         → New entry needed. Apply ≥6 DTE rule:
           - this Tuesday: 1 DTE → fail
           - next Tuesday: 8 DTE → use this
         → Place bear put spread for NEXT Tuesday expiry, anchored to current spot ATM.
         → State: DEBIT_OPEN_L1 (new direction, new week)
         → T-1 of THIS new position is the FOLLOWING Monday 15:25 (one week from now).

  Suppose break of extreme does NOT confirm before 15:30 close:
         → State stays ARMED (short) overnight.
         → Tuesday morning (current week's expiry day): if break confirms, apply
           ≥6 DTE rule: this Tuesday is 0 DTE → fail; next Tuesday is 7 DTE → use.
         → New position opens for next Tuesday's expiry.
```

Key takeaway: the ≥6 DTE rule **always** wins. If a flip or rollover would otherwise force a sub-6-DTE entry, the system skips ahead to the next valid expiry. There's never a scenario where the engine opens a position with <6 DTE.

#### 4.5.2 Pyramid level resets at rollover

When the T-1 close happens with MST still active and immediate rollover fires, the new position is opened at **level 1**, regardless of the previous week's level. To reach level 2 again, the D AND B trigger must fire fresh in the new week.

Rationale: each weekly expiry is an independent validation. Carrying L2 forward would assume continued strong momentum without any new signal in the new week's price action — too aggressive for Phase 1.

### 4.6 Trading-day & holiday handling

NSE has ~10-15 holidays per year. Some fall on Tuesday (the weekly expiry day) and some on Monday (T-1 day). Handling:

#### 4.6.1 Expiry day shifts (NSE rule)

If the regular Tuesday expiry falls on an NSE holiday, the exchange **shifts expiry to the previous trading day** (Monday, or earlier if Monday is also a holiday). The instruments() API returns the actual exchange-published expiry dates, which are already holiday-adjusted.

The implementation must:

```python
# Query at startup AND when computing next entry's expiry
instruments = kite.instruments('NFO')
nifty_weekly_expiries = sorted(set(
    i['expiry'] for i in instruments
    if i['name'] == 'NIFTY' and i['instrument_type'] in ('CE', 'PE')
    and (i['expiry'] - i['expiry'].replace(day=1)).days < 28  # weekly, not monthly
))
# Use these dates as the source of truth for "next expiry".
```

DO NOT hardcode "next Tuesday from today's date" — that would miss exchange shifts.

#### 4.6.2 T-1 calculation (own logic)

T-1 = the **previous trading day before the position's expiry date**. NSE doesn't publish "T-1" — we compute it from the trading-day calendar.

```python
def t_minus_1(expiry_date: date, calendar: TradingCalendar) -> date:
    return calendar.previous_trading_day(expiry_date)
```

Examples:

| Expiry | T-1 (normal) | T-1 (with Monday holiday) |
|---|---|---|
| Tuesday May 12 | Monday May 11 | Friday May 8 (if Monday is a holiday) |
| Tuesday Aug 18 | Monday Aug 17 | Friday Aug 14 (if Monday is Aug 15 — Independence Day shifts the chain too) |
| Tuesday holiday → expiry shifted to Monday | Friday previous week | Thursday previous week if Friday is also holiday |

#### 4.6.3 Trading calendar service

Add `services/trading_calendar.py` (new file, simple):

```python
class NSETradingCalendar:
    """NSE trading-day calendar with holiday awareness."""

    HOLIDAYS_FILE = 'config/nse_holidays_<year>.json'  # update annually

    def __init__(self, year: int = None):
        self.holidays = self._load_holidays(year or datetime.now().year)

    def is_trading_day(self, d: date) -> bool:
        if d.weekday() >= 5:  # Sat=5, Sun=6
            return False
        return d not in self.holidays

    def previous_trading_day(self, d: date) -> date:
        prev = d - timedelta(days=1)
        while not self.is_trading_day(prev):
            prev -= timedelta(days=1)
        return prev

    def next_trading_day(self, d: date) -> date:
        nxt = d + timedelta(days=1)
        while not self.is_trading_day(nxt):
            nxt += timedelta(days=1)
        return nxt

    def trading_days_between(self, start: date, end: date) -> int:
        n, d = 0, start + timedelta(days=1)
        while d <= end:
            if self.is_trading_day(d):
                n += 1
            d += timedelta(days=1)
        return n

    def _load_holidays(self, year: int) -> set[date]:
        # JSON file with NSE holiday list per year; updated annually
        with open(self.HOLIDAYS_FILE.replace('<year>', str(year))) as f:
            return {date.fromisoformat(d) for d in json.load(f)['holidays']}
```

`config/nse_holidays_2026.json` (operator updates annually from NSE's published calendar):

```json
{
  "year": 2026,
  "holidays": [
    "2026-01-26",
    "2026-02-19",
    "2026-03-04",
    "2026-03-31",
    "2026-04-10",
    "2026-04-14",
    "2026-05-01",
    "2026-08-15",
    "2026-08-27",
    "2026-10-02",
    "2026-10-21",
    "2026-11-04",
    "2026-12-25"
  ],
  "muhurat_session": {"date": "2026-10-21", "time": "18:00-19:00"}
}
```

#### 4.6.4 Engine integration

The MST engine consumes the calendar in two places:

1. **At entry / rollover:** `next_expiry_with_min_dte(from_date, min_dte=6)` returns the next weekly expiry from Kite's instruments list that is ≥ 6 calendar days away from `from_date`.
2. **At T-1 scheduling:** when a position is opened, compute `t_minus_1_dt = calendar.previous_trading_day(position.expiry_dt)` and store it. The daily 15:25 cron checks `if today.date() == any_open_position.t_minus_1_dt` → fire close + rollover.

This replaces the static "every Wednesday at 15:25" cron — the cron now runs every weekday at 15:25, and only acts if today is T-1 for some open position.

---

## 5. Data layer

### 5.1 Live tick source — REUSE `services/nas_ticker.py` singleton

Per [services/nas_ticker.py:26-28](../../services/nas_ticker.py#L26-L28): "*only one KiteTicker can exist per process... NAS owns the singleton; other strategies that need ticks should subscribe...*"

The MST engine **does not create its own KiteTicker**. It hooks into NasTicker:

1. NasTicker already subscribes to NIFTY 50 (instrument_token = 256265) and builds 5-min candles via `NiftyCandleAggregator`.
2. MST engine registers a callback on the 5-min candle close (alongside NAS's existing scan callback).
3. MST callback aggregates incoming 5-min candles into 30-min buckets (09:15-09:45, 09:45-10:15, …, 15:15-15:30 IST). On 30-min bucket close, fire `MSTEngine.on_new_bar()`.

**Why this matters:** keeps the WebSocket subscription single, avoids duplicate state, ensures NAS and MST see identical NIFTY data.

### 5.2 Historical seed

On startup, load the last ~200 NIFTY 30-min bars from `backtest_data/market_data.db` (table `market_data_unified`, symbol=`NIFTY50`, timeframe=`30minute` if present, else resample from `5minute`). 200 bars covers the longest indicator period (50) plus warmup for Stoch.

If service starts mid-day (after first bar of the day), backfill from Kite via `kite.historical_data(256265, ..., '30minute')` to cover any gap.

### 5.3 New SQLite DB — `backtest_data/mst_trading.db`

Mirrors the KC6 pattern in `services/kc6_db.py`. Five tables:

```sql
-- 30-min bars + computed indicators
CREATE TABLE mst_bars (
    bar_dt TEXT PRIMARY KEY,             -- ISO 8601 IST
    open REAL, high REAL, low REAL, close REAL,
    atr21 REAL,
    st_upper REAL, st_lower REAL,
    direction INTEGER,                    -- +1, -1, 0 if not seeded
    stoch_k REAL, stoch_d REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Signal events (separate from positions)
CREATE TABLE mst_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,             -- 'flip_armed' | 'flip_activated' |
                                          -- 'flip_discarded' | 'cst_trigger' |
                                          -- 'condor_built' | 'rolled' |
                                          -- 't_minus_1_close' | 'mst_flip_close' |
                                          -- 'kill_switch'
    direction INTEGER,
    bar_dt TEXT NOT NULL,
    price REAL,
    flip_high REAL,
    flip_low REAL,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Positions (one row per leg)
CREATE TABLE mst_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_label TEXT NOT NULL,             -- 'YYYY-MM-DD' = expiry date (Tuesday, or shifted by holidays)
    leg_role TEXT NOT NULL,               -- 'bull_long', 'bull_short', 'bear_short', 'bear_long'
                                          -- (or 'put_long', 'put_short', 'putw_short', 'putw_long' for short MST)
    side TEXT NOT NULL,                   -- 'BUY' | 'SELL'
    instrument_token INTEGER,
    tradingsymbol TEXT,                   -- e.g. NIFTY26MAY22500CE
    strike INTEGER,
    option_type TEXT,                     -- 'CE' | 'PE'
    qty INTEGER,                          -- 75 for 1 lot
    entry_price REAL,
    entry_time TEXT,
    exit_price REAL,
    exit_time TEXT,
    exit_reason TEXT,
    status TEXT,                          -- 'OPEN' | 'CLOSED'
    pnl_inr REAL,
    order_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Order audit
CREATE TABLE mst_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_dt TEXT,
    leg_id INTEGER,                       -- FK to mst_positions.id
    order_id TEXT,                        -- Zerodha order_id
    side TEXT,
    qty INTEGER,
    price REAL,
    order_type TEXT,                      -- LIMIT | MARKET
    status TEXT,                          -- PLACED | FILLED | REJECTED | CANCELLED
    error_msg TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Daily P&L curve
CREATE TABLE mst_equity (
    date TEXT PRIMARY KEY,
    realized_pnl REAL,
    unrealized_pnl REAL,
    total_pnl REAL,
    open_legs INTEGER,
    state TEXT,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_mst_events_bar ON mst_events(bar_dt);
CREATE INDEX idx_mst_events_type ON mst_events(event_type);
CREATE INDEX idx_mst_positions_status ON mst_positions(status);
CREATE INDEX idx_mst_positions_week ON mst_positions(week_label);
```

---

## 6. Signal & execution engine — `services/mst_engine.py`

Pattern reference: `services/kc6_scanner.py` + `services/kc6_executor.py`. Keep MST in a single class with explicit state machine method.

```python
class MSTEngine:
    """
    NIFTY 30-min Master SuperTrend + Stochastic CST signal & execution engine.

    States:  NO_POSITION → ARMED → DEBIT_OPEN → CONDOR_OPEN
             (with ROLL_PENDING fork on credit-too-low CST)

    Drivers:
      - on_5min_candle_close(candle):  called by NasTicker callback;
                                        aggregate to 30-min, run signals on 30-min close
      - on_30min_close(bar):           main signal evaluator + state transitions
      - on_t_minus_1_eod():            fires at 15:25 IST whenever today is the
                                        T-1 day of an open position; force-close
                                        and rollover (re-anchored to current ATM)
      - on_mst_flip():                 mid-state flip handler; close + reset
      - kill_switch():                 emergency close + halt
    """
    ATR_PERIOD = 21
    MULTIPLIER = 5.0
    STOCH_K = 14
    STOCH_D = 3
    STOCH_SMOOTH = 3
    STOCH_OB = 80
    STOCH_OS = 20
    MIN_CREDIT_PER_LOT = 1000   # rupees; below → reset path
    SPREAD_WIDTH = 200           # standard structure points
    RESET_WIDTH = 100            # reset (Reading D) structure points
    LOTS = 1                     # Phase 1: 1 lot per leg per level
    MIN_DTE_AT_ENTRY = 6         # weekly expiry rule
    PYRAMID_MAX_LEVEL = 2        # max pyramid level (research/36 cap)
    PYRAMID_TRIGGER_D_BARS = 2   # consecutive closes beyond CST bar (trigger D)
    PYRAMID_TRIGGER_B_OB = 80    # %K threshold for B (long bias)
    PYRAMID_TRIGGER_B_OS = 20    # %K threshold for B (short bias)
    PYRAMID_TRIGGER_B_EXIT = 70  # %K must drop below this before re-entering OB (long)

    def on_30min_close(self, bar):
        # 1. Append bar to rolling buffer
        # 2. Recompute SuperTrend, ATR, Stoch on the buffer
        # 3. Persist bar to mst_bars
        # 4. State machine transitions:
        #    - NO_POSITION & MST flip detected → emit 'flip_armed', go ARMED
        #    - ARMED & break of flip-bar high (long) / low (short) → activate
        #         → place debit spread (§4.2) → DEBIT_OPEN
        #    - ARMED & MST re-flips → 'flip_discarded' → re-arm
        #    - DEBIT_OPEN & first CST → check bear call credit at std strikes
        #         credit ≥ ₹1,000/lot → place credit spread → CONDOR_OPEN
        #         credit < ₹1,000/lot → 'rolled' event → close debit, open
        #                                fresh next-week reset structure (§4.3)
        #    - CONDOR_OPEN & subsequent CST → log only
        # 5. Persist any events / positions
        # 6. Dispatch alerts via NotificationService
```

### 6.1 Order placement helpers

Reuse the existing Kite order-placement pattern from `services/orb_live_engine.py` (look for `_place_order()`-style helpers). For MST:
- `place_leg(leg_role, strike, side, qty)` — places LIMIT-at-mid, falls back to MARKET after 30s
- `close_leg(position_id)` — places MARKET to square off; updates `mst_positions.status = CLOSED`
- `close_all_legs(reason)` — bulk close for T-1 EOD / MST flip / kill switch

Strike → tradingsymbol resolution: use existing `services/options_data_manager.py` patterns (see how NAS / strangle resolve weekly NIFTY strikes).

### 6.2 Scheduling

Add to `app.py` near KC6/ORB scheduled jobs:

```python
# 30-min bar close evaluator runs INSIDE NasTicker callback;
# no separate cron needed for signal generation.

# T-1 EOD check — runs every weekday at 15:25 IST. Inside the handler,
# checks if today.date() matches any open position's t_minus_1_dt; if so,
# closes those positions and triggers rollover. This is holiday-aware
# because t_minus_1_dt is computed via the trading calendar at position open.
scheduler.add_job(
    mst_t_minus_1_check, 'cron',
    day_of_week='mon-fri', hour=15, minute=25,
    id='mst_t_minus_1_check',
)

# Daily equity snapshot at 15:35 IST (after market close, post any T-1 close)
scheduler.add_job(
    mst_equity_snapshot, 'cron',
    day_of_week='mon-fri', hour=15, minute=35,
    id='mst_equity_snapshot',
)

# Position reconciliation on startup + every 30 min during market hours
scheduler.add_job(
    mst_reconcile_positions, 'cron',
    day_of_week='mon-fri', hour='9-15', minute='5,35',
    id='mst_reconcile',
)
```

---

## 7. Backend API — Flask routes in `app.py`

Match ORB conventions ([services/orb_live_engine.py](../../services/orb_live_engine.py) + the `/api/orb/*` routes).

| Route | Method | Returns |
|---|---|---|
| `/api/mst/state` | GET | Full state object (see §7.1) — feeds the dashboard |
| `/api/mst/bars?limit=200` | GET | Recent 30-min bars with indicators (chart data) |
| `/api/mst/events?limit=50&type=*` | GET | Recent signal/position events |
| `/api/mst/positions?status=*` | GET | Open + recently closed legs |
| `/api/mst/equity-curve?days=30` | GET | Daily P&L for equity curve chart |
| `/api/mst/scan` | POST | Force re-evaluate latest bar (debug aid) |
| `/api/mst/kill-switch` | POST | Close all legs + halt new entries |
| `/api/mst/toggle-mode` | POST | (placeholder; Phase 1 is live-only, no paper toggle) |

### 7.1 State object shape (mirrors `ORBState`)

```typescript
interface MSTState {
  // High-level state
  state_machine: 'NO_POSITION' | 'ARMED' | 'DEBIT_OPEN' | 'CONDOR_OPEN' | 'ROLL_PENDING';
  mst_direction: 1 | -1 | 0;          // 0 = idle
  armed_since: string | null;          // ISO IST
  position_since: string | null;
  current_week_expiry: string | null;  // ISO YYYY-MM-DD

  // Live indicators
  last_bar_dt: string;
  last_close: number;
  atr21: number;
  st_value: number;
  stoch_k: number;
  stoch_d: number;
  flip_armed_high: number | null;
  flip_armed_low: number | null;

  // Position summary
  open_legs: MSTLeg[];                 // 0, 2, or 4 legs
  closed_today: MSTLeg[];

  // P&L
  today_pnl: number;
  unrealized_pnl: number;
  realized_pnl_today: number;

  // Config
  config: {
    atr_period: 21,
    multiplier: 5.0,
    stoch: { k: 14, d: 3, smooth: 3, ob: 80, os: 20 },
    spread_width: 200,
    reset_width: 100,
    min_credit_per_lot: 1000,
    lots: 1,
    min_dte_at_entry: 6,
    t_minus_1_close_time: '15:25',
  };

  // Mode
  live_trading: true;                  // Phase 1: always live
  kill_switch_active: boolean;
}
```

---

## 8. Frontend — `/app/mst` (mirror ORB structure exactly)

**Files to create:**
```
frontend/src/pages/Mst.tsx
frontend/src/pages/Mst.module.css
frontend/src/api/types.ts             # add MSTState, MSTLeg, MSTEvent interfaces
frontend/src/components/Sidebar/Sidebar.tsx   # add nav entry
frontend/src/App.tsx                  # add route
```

### 8.1 Page layout (top → bottom; matches Orb.tsx section order)

| Section | Mirror of ORB equivalent |
|---|---|
| **Header row** — "MST · NIFTY 30-min" + subtitle (state, live trading, current weekly expiry) | Orb.tsx line 422-430 |
| **Error banner** (if any) | Orb.tsx line 432 |
| **Metrics row** — 4 `MetricCard`s: (a) Day P&L, (b) State + direction, (c) Stoch reading, (d) Trades today | Orb.tsx line 435-466 |
| **30-min chart panel** — OHLC with SuperTrend overlay, flip markers, CST markers; Stoch sub-panel below with %K, %D, 80/20 reference lines | Orb.tsx BookPnLChart pattern at line 469 |
| **Section: Position** — `DataTable` with all 4 condor legs (or 2 debit legs) showing strike, qty, entry, LTP, P&L, status; closed-today legs dimmed in place. | Orb.tsx Positions section line 472-489 (with the same closed-row dim pattern) |
| **Section: Current state** — visual state-machine indicator + key levels (flip-armed high/low, MST direction, days to expiry, T-1 close countdown if today is T-1) | Orb.tsx CandidatesSection pattern line 492 |
| **Section: Current indicators** — config grid showing all rules from §6 constants | Orb.tsx CurrentIndicators line 495-501 + 832-884 |
| **Section: Today's events** — `DataTable` of MST events (flip_armed, flip_activated, cst_trigger, condor_built, rolled, t_minus_1_close, mst_flip_close) with time, type, direction, price, notes | Orb.tsx Today's signals section line 575-586 |
| **Section: What's next** — schedule (next 30-min bar evaluation, T-1 close time if today/tomorrow is T-1, weekly expiry date, next trading day) | Orb.tsx WhatsNext section line 566-572 |
| **Section: Strategy rules** — collapsed `<details>` block, format-matched to ORB's rules section, with: Setup, MST signal, MST entry filter, CST trigger, Multi-CST policy, Spread structure (standard + reset), DTE rule, Order placement, Exit rules (T-1, MST flip, kill switch) | Orb.tsx Strategy rules section line 591-758 |
| **Section: Backtest baseline** — collapsed `<details>` block with research/35 numbers (252 cells swept, NIFTY p21,m5.0 winner, MFE/MAE 3.62 with break-of-extreme, multi-CST policy data) | Orb.tsx Backtest baseline section line 761-827 |

### 8.2 Sidebar entry

In [frontend/src/components/Sidebar/Sidebar.tsx:97-171](../../frontend/src/components/Sidebar/Sidebar.tsx#L97-L171):

```tsx
// Add MST entry between NWV and EOD (visually grouped with index strategies)
<NavItem
  to="/app/mst"
  icon={<IconLayers />}                // reuse existing IconLayers like NAS/NWV/strangle
  label="MST"
  active={active === 'mst'}
  collapsed={collapsed}
/>
```

Update the `Props.active` union: `'orb' | 'nas' | 'nwv' | 'strangle' | 'mst' | 'eod-breakout' | …`.

### 8.3 Routing

In `frontend/src/App.tsx`:
```tsx
<Route path="/app/mst" element={<Mst />} />
```

### 8.4 Build & deploy

`cd frontend && npm run build` produces `frontend/dist/` which Flask serves under `/app/*`. **Frontend-only changes do NOT require backend restart** — see `CLAUDE.md` "NO BACKEND RESTART DURING MARKET HOURS".

Backend changes (new SQLite DB, new APIs, new scheduler jobs, MSTEngine, NasTicker callback hook) **DO require a restart** and must be deployed **after 15:30 IST**.

---

## 9. Alerts — `services/notifications.py` (reuse, don't recreate)

Per [services/notifications.py:19-90](../../services/notifications.py#L19-L90), the existing `NotificationService` already supports email + in-app, async dispatch. Phase 1 channels per user: **email only** (in-app DB persistence is automatic via `InAppProvider`).

### 9.1 Wiring

```python
from services.notifications import NotificationService
from config import MST_DEFAULTS

mst_notifier = NotificationService(MST_DEFAULTS)  # reads email_enabled from config

# On state transition:
mst_notifier.send_alert(
    alert_type='trade_entry',          # or 'trade_exit', 'cst_trigger', 'system_alert'
    title='MST · LONG ACTIVATED',
    message='Bull call spread placed: NIFTY 22500/22700 CE, weekly Tue (12-May-2026)',
    data={'direction': 'LONG', 'spot': 22524, 'flip_high': 22518, 'expiry': '2026-05-15'},
    priority='high',
)
```

### 9.2 Alert types

| Event | Alert type | Priority | Subject example |
|---|---|---|---|
| `flip_armed` | `system_alert` | low | MST · LONG ARMED — waiting for break above 22518 |
| `flip_activated` | `trade_entry` | high | MST · LONG ACTIVATED — bull call placed 22500/22700 |
| `flip_discarded` | `system_alert` | low | MST · flip discarded — re-armed in opposite direction |
| `cst_trigger` (first, condor built) | `trade_entry` | high | MST · CONDOR BUILT — bear call added 22900/23100 |
| `cst_trigger` (subsequent) | `system_alert` | low | MST · CST in active week — informational |
| `rolled` (credit too low → reset to next week) | `trade_entry` | high | MST · ROLLED to next week — fresh condor placed |
| `mst_flip_close` | `trade_exit` | high | MST · flipped opposite — all legs closed |
| `t_minus_1_close` | `trade_exit` | normal | MST · T-1 EOD square-off — final P&L: +₹X |
| `kill_switch` | `system_alert` | critical | MST · KILL SWITCH — all legs closed, halted |

---

## 10. Testing approach

### 10.1 Replay test (offline)

`tests/test_mst_engine_replay.py`:
- Load NIFTY 30-min bars 2024-03-01 → 2026-03-25 from `market_data.db`
- Feed each bar to `MSTEngine.on_30min_close()` sequentially
- Mock the order-placement layer (don't hit Kite)
- Compare emitted events to a golden file derived from `research/35_.../scripts/run_mst_sweep_breakout.py` for cell `NIFTY50_30min_p21_m5.0`
- Required: every `flip_armed`, `flip_activated`, `flip_discarded`, `cst_trigger`, `condor_built`, `rolled` event matches within ±1 bar
- P&L sanity check: simulate fills at mid-price, verify total cycle P&L matches a recomputation from the underlying

### 10.2 Live paper-shadow run (1-2 weeks before money)

Even though Phase 1 is live, run a 1-2 week shadow first where the engine emits all events and would-be-orders to logs but does NOT call `kite.place_order()`. Validate:
- No state-machine bugs (no stuck states, no double-entries)
- Alerts arrive on time and with correct content
- T-1 close fires reliably (Mondays for normal weeks, shifted Fridays for Monday-holiday weeks)
- Holiday calendar correctly shifts expiry & T-1 (test against at least one mid-year holiday like Aug 15)
- Position reconciliation handles service restarts

### 10.3 Live with 1 lot

After shadow passes, flip the order-placement guard to live. Operator monitors:
- Order fills (LIMIT-at-mid hit rate; how often we fall back to MARKET)
- Slippage vs expected mid
- Reconciliation drift (DB vs Kite holdings)
- Daily P&L accuracy

### 10.4 Kill switch drill

Once before going live: trigger `/api/mst/kill-switch` while a 1-lot test condor is open in a low-risk environment. Verify all 4 legs close at market within 5 seconds, state transitions to halted, no further orders placed.

---

## 11. Open questions resolved + new ones

### Resolved (from user 2026-05-04)

| Question | Resolution |
|---|---|
| Alert channel | Email only (use existing `NotificationService`) |
| NIFTY ticker source | Reuse `NasTicker` singleton — no new WebSocket |
| Phase 2 timing | N/A — Phase 1 IS live with 1 lot |
| Sidebar grouping | Workspace group, between NWV and EOD |
| BANKNIFTY in Phase 1? | No — NIFTY only |
| KC6 conflict | None — confirmed, KC6 doesn't trade indices |
| Weekly expiry day | **Tuesday** (per user 2026-05-05) — query Kite at runtime, don't hardcode |
| T-1 EOD square-off | **Monday 15:25 IST** for normal Tue expiry; shifts to Fri 15:25 if Mon is holiday. Computed from trading calendar (§4.6) |
| Trading calendar / holidays | Required service — `services/trading_calendar.py` + `config/nse_holidays_<year>.json` |
| Rollover at T-1 if MST still active | Yes — open new debit at current spot ATM, next weekly expiry (≥6 DTE), reset to L1 |
| ≥6 DTE rule | Universal — applies to initial entry, post-MST-flip re-arm, weekly rollover, level-2 pyramid entry. No exceptions. |
| Multi-CST policy | One condor per weekly expiry cycle, with pyramiding (per research/36) |
| Credit threshold for reset | ₹1,000/lot total |
| **CST false-alarm problem** | **Confirmed real — 67% of first CSTs see trend continue (research/36)** |
| **Pyramid trigger** | **D AND B (research/36): two closes beyond CST bar + Stoch %K back to OB/OS** |
| **Pyramid cap** | **Max level 2 (single re-double); beyond that, log-only** |

### New — needed before Phase 1 implementation starts

1. **Reset structure choice (Reading A / C / D)** — default in this doc is Reading D (100/100/100 spot-centered). Confirm or override.
2. **NIFTY lot size verification** — currently 75 contracts per lot per `FNO_LOT_SIZES`. Verify against current Kite contract spec at startup; alert if changed.
3. **Holiday calendar source** — does the operator maintain `config/nse_holidays_<year>.json` manually each year (NSE publishes it in December for the following year), or do we want to also query Kite/NSE programmatically at startup? Manual JSON is simpler; Kite has no direct holiday API.
4. **Email subject/body format** — happy with the templates in §9.2, or want them tweaked?
5. **Order-rejection handling** — if Kite rejects a leg (margin shortfall, illiquid strike), should the engine: (a) skip that leg and proceed with what filled, (b) close any filled legs and abort, or (c) retry with adjacent strike? Default suggestion: (b) — atomic-or-nothing.
6. **Margin pre-check** — should we run a `kite.basket_margins()` call before placing the 4 legs to ensure sufficient margin, and alert if not? Recommended: yes.
7. **Shadow-run duration** — research/36 extended-period sample (6.3 yrs, 1,495 CSTs) tightens the confidence intervals enough that 1 week of shadow operation may suffice. Default suggestion: 1 week.

These are clarification questions — none of them are showstoppers. Implementation can start with the defaults proposed.

---

## 12. References

- Research artifacts: `research/35_nifty_bnf_master_child_supertrend/RESULTS.md` and CSV outputs
- Pyramid trigger validation: `research/36_mst_cst_continuation_pyramid/results/RESULTS.md`
- SuperTrend & Stoch reference impl: `research/35_nifty_bnf_master_child_supertrend/scripts/supertrend.py`
- Mirror page: `frontend/src/pages/Orb.tsx` and `Orb.module.css`
- NIFTY tick source: `services/nas_ticker.py` (singleton + 5-min aggregator)
- Notifications: `services/notifications.py`
- Order-placement reference: `services/orb_live_engine.py`, `services/kc6_executor.py`
- Options strike resolution: `services/options_data_manager.py`
- Sidebar: `frontend/src/components/Sidebar/Sidebar.tsx`
- App architecture: `docs/Design/ARCHITECTURE.md`, `docs/Design/LIVE-TRADING-ARCHITECTURE.md`
- Project rules to honor:
  - "ALL NEW PAGES GO IN THE REACT APP AT `/app/*` — NOT JINJA"
  - "NO BACKEND RESTART DURING MARKET HOURS" (09:15–15:30 IST)
  - "LIVE-STATUS MD CONVENTION" (apply if implementation runs > 5 min)
