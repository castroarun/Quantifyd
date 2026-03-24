# NAS — Nifty ATR Strangle (Intraday Options Selling)

**Status:** Implementation Ready
**Date:** 2026-03-20
**Deployed:** Railway Cloud (with main app)
**Instrument:** NIFTY 50 Index Options (NSE F&O)
**Timeframe:** 5-minute candles (intraday)

---

## 1. Strategy Overview

Intraday short strangle on NIFTY using ATR contraction as the entry trigger. When 5-min ATR is below its own moving average (market is ranging/quiet), sell OTM CE + PE. Adjust legs dynamically when one side's premium doubles (losing) or halves (winning). Close everything by EOD.

**Edge:** Selling options in low-volatility intraday regimes exploits theta decay + vol mean-reversion. ATR contraction identifies quiet periods where strangles collect premium without directional risk.

---

## 2. Entry Rules

### 2.1 ATR Squeeze Detection (5-min)

| Parameter | Default | Notes |
|-----------|---------|-------|
| ATR Period | 14 | 14-bar ATR on 5-min candles = ~70 min lookback |
| ATR MA Period | 50 | 50-bar SMA of ATR = ~250 min (full-day average) |
| Entry condition | ATR < ATR_MA | Current ATR below its moving average = squeeze |
| Min squeeze bars | 3 | ATR must be below MA for at least 3 consecutive bars |

**Signal:** When `ATR(14) < SMA(ATR, 50)` for 3+ consecutive 5-min bars → SELL STRANGLE.

### 2.2 Strike Selection

| Parameter | Default | Notes |
|-----------|---------|-------|
| Strike distance | 1.5x ATR (daily) | Use daily ATR for strike width, not 5-min ATR |
| Strike interval | 50 | Nifty options strike gap |
| Call strike | `round_up(Spot + daily_ATR × 1.5, 50)` | OTM call |
| Put strike | `round_down(Spot - daily_ATR × 1.5, 50)` | OTM put |
| Expiry | Current week (nearest Thursday) | Weekly options for max theta |
| Min DTE | 1 | Don't sell on expiry day (Thursday) |

### 2.3 Position Sizing

| Parameter | Default | Notes |
|-----------|---------|-------|
| Lots per leg | 2 | 2 lots × 75 qty = 150 qty per leg (Nifty lot = 75) |
| Max concurrent strangles | 1 | Only 1 strangle at a time |
| Capital required | ~3L margin | SPAN margin for short strangle |

### 2.4 Time Window

| Rule | Value | Notes |
|------|-------|-------|
| Entry window | 9:30 AM — 2:30 PM | No entries in first 15 min or last hour |
| No entry if | < 60 min to close | Not enough time for theta to work |
| No entry on | Expiry day (Thursday) | Gamma risk too high |
| No entry on | Budget day, RBI policy, Election results | Event filter |

---

## 3. Exit Rules

### 3.1 Primary Exits

| Exit Rule | Trigger | Action |
|-----------|---------|--------|
| **EOD Squareoff** | 3:15 PM | Close ALL legs — mandatory, no overnight |
| **Combined SL** | Total strangle loss > 2x initial premium | Close both legs |
| **Profit Target** | Total premium decays to 30% of entry | Close both legs (70% captured) |
| **Time Exit** | 2:45 PM if still open | Close — don't hold into last 45 min |

### 3.2 Adjustment Rules (The Core Edge)

These adjustments are what differentiate NAS from naive strangle selling:

| Condition | What Happened | Action |
|-----------|---------------|--------|
| **One leg premium DOUBLES (2x)** | Market moved against that leg | **Adjustment A:** Close losing leg, re-enter at new ATR-based strike |
| **One leg premium HALVES (0.5x)** | Market moved away from that leg | **Adjustment B:** Book profit on winning leg, re-enter closer to current spot |
| **Both legs profitable** | Market stayed flat | Hold — ideal scenario, let theta decay |
| **Max adjustments reached** | Already adjusted 2x | No more adjustments — hold or exit on combined SL |

#### Adjustment A — Losing Leg (Premium Doubled)

```
1. Close the losing leg (buy back at ~2x entry premium → loss)
2. Wait 1 bar (5 min)
3. Re-sell at new strike = current_spot ± 1.5x ATR (recalculated)
4. New SL = 2x new premium
5. Log adjustment in DB
```

#### Adjustment B — Winning Leg (Premium Halved)

```
1. Close the winning leg (buy back at ~0.5x entry premium → profit)
2. Re-sell closer to current spot at new ATR-based strike
3. Collect fresh premium on the re-positioned leg
4. Max 2 adjustments per leg per day
```

### 3.3 Exit Priority

```
1. Emergency kill switch (manual)
2. Combined SL (2x total premium lost)
3. EOD squareoff (3:15 PM)
4. Profit target (70% premium captured)
5. Time exit (2:45 PM)
6. Adjustment (single leg 2x/0.5x trigger)
```

---

## 4. Risk Management

### 4.1 Position-Level

| Rule | Value |
|------|-------|
| Max loss per strangle | 2x total collected premium |
| Max loss per day | Rs 15,000 (configurable) |
| Max adjustments per leg | 2 |
| Max total adjustments per day | 4 |
| No re-entry after combined SL | Done for the day |

### 4.2 System-Level

| Rule | Value |
|------|-------|
| Kill switch | Manual button on dashboard |
| Paper mode default | ON — must explicitly toggle to live |
| Kite auth check | Before every order |
| Market hours check | No orders outside 9:15–3:30 |
| Daily P&L circuit breaker | Stop if daily loss > Rs 20,000 |

### 4.3 Crash / Gap Protection

| Scenario | Protection |
|----------|-----------|
| Flash crash (Nifty drops 2%+ in 5 min) | ATR spikes above MA → no new entries; existing SL protects |
| Gap open against position | Won't happen — strictly intraday, no overnight |
| VIX spike > 20 | Block new entries (optional filter) |

---

## 5. Red Flags & Known Risks

### 5.1 Critical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Gamma blowup on expiry day** | HIGH | No entries on Thursday (expiry day) |
| **Black swan intraday move** | HIGH | Combined SL at 2x premium caps loss; kill switch |
| **Slippage on illiquid strikes** | MEDIUM | Use only strikes within 3% of spot; check OI > 1L |
| **ATR squeeze false signals** | MEDIUM | Min 3 bars of squeeze; confirm with BB width |
| **Adjustment over-trading** | MEDIUM | Max 2 adjustments per leg; stop after 4 total |
| **Kite API failure during exit** | HIGH | Retry logic with 3 attempts; alert on failure |

### 5.2 When NOT to Trade

- **Thursday** (weekly expiry) — gamma risk
- **First 15 minutes** — opening volatility
- **Last 45 minutes** — avoid closing auction issues
- **Budget day, RBI policy, Election results** — binary events
- **If VIX > 20** — market too volatile for strangle selling
- **If daily ATR expanding** — trending day, not a squeeze day

---

## 6. Suggested Improvements

### 6.1 High Priority

| Improvement | Impact | Effort |
|-------------|--------|--------|
| **VIX filter** | Block entries when India VIX > 18-20 | Low — read VIX from Kite |
| **OI-based strike validation** | Ensure sold strikes have OI > 1 lakh | Low — Kite instruments API |
| **Intraday trend filter** | Skip entries if Nifty already moved > 0.5% from open | Low — simple check |
| **Premium floor** | Don't sell if combined premium < Rs 30/lot | Low — BS pricing check |

### 6.2 Medium Priority

| Improvement | Impact | Effort |
|-------------|--------|--------|
| **Dynamic lot sizing** | Scale lots based on VIX (more lots in low VIX) | Medium |
| **Greeks-based adjustment** | Adjust on delta breach instead of premium multiple | Medium — need real-time greeks |
| **Multiple timeframe ATR** | Confirm 5-min squeeze with 15-min ATR contraction | Medium |
| **Bollinger Band squeeze confirmation** | Add BB width < BB width MA as second filter | Low |

### 6.3 Future Enhancements

| Enhancement | Notes |
|-------------|-------|
| **Iron Condor mode** | Add protective wings at 3x ATR for defined risk |
| **Auto strike selection from option chain** | Use live option chain instead of BS pricing |
| **Real-time IV from option chain** | Replace ATR-derived IV with actual market IV |
| **WebSocket-based execution** | React to premium changes in real-time vs polling |
| **Multi-index support** | Extend to BankNifty, FinNifty with same logic |

---

## 7. Options Data Recording

### 7.1 What to Record

The system must persist options data for analysis and paper trade tracking:

| Table | Purpose |
|-------|---------|
| `nas_state` | Current system state (ATR, squeeze status, VIX) |
| `nas_positions` | Active and closed option positions |
| `nas_trades` | Completed trades with full P&L |
| `nas_orders` | Order audit log (paper + live) |
| `nas_signals` | Every signal detected (entry, adjustment, exit) |
| `nas_daily_state` | EOD snapshot for equity curve |
| `nas_option_snapshots` | Periodic option chain snapshots (strike, premium, OI, IV) |

### 7.2 Option Chain Snapshots

Every 5 minutes during market hours, capture and store:
- Top 10 strikes around ATM (5 CE + 5 PE)
- LTP, bid, ask, OI, volume, IV for each
- Stored in `nas_option_snapshots` table
- Enables post-hoc analysis of premium behavior

---

## 8. Backtest Results (from nondirectional_simulator.py)

The original backtester (`services/nondirectional_simulator.py`) tested this concept on daily bars for both NIFTY and BANKNIFTY. Key findings:

| Config | Win Rate | Profit Factor | Notes |
|--------|----------|---------------|-------|
| BB Squeeze + ATR Contract, 2.5x ATR strikes, 5-bar hold | 68% | 1.8 | Best combination |
| ATR Contract only, 1.5x ATR strikes, 3-bar hold | 72% | 1.5 | Tighter strikes, more adjustments needed |
| BB Squeeze only, 2.0x ATR strikes, 7-bar hold | 65% | 1.6 | Slower but steadier |

**Note:** These are daily-bar backtests. The 5-min intraday version should perform better due to:
1. More frequent entry opportunities
2. No overnight gap risk
3. Faster theta decay on weekly options
4. Tighter adjustments possible

---

## 9. Infrastructure

### 9.1 Files

| File | Purpose |
|------|---------|
| `services/nas_db.py` | SQLite persistence (positions, trades, orders, signals, snapshots) |
| `services/nas_scanner.py` | 5-min ATR squeeze detection, strike computation, exit checks |
| `services/nas_executor.py` | Order execution with guardrails, adjustments, paper/live modes |
| `templates/nas_dashboard.html` | Dashboard with positions, trades, signals, equity curve |
| `config.py` | `NAS_DEFAULTS` configuration dict |
| `app.py` | Routes `/nas`, `/api/nas/*`, scheduled jobs |

### 9.2 Scheduled Jobs (Mon-Wed, Fri only — skip Thursday)

| Time | Job | What |
|------|-----|------|
| 9:20 AM | Auth check | Verify Kite session active |
| 9:30 AM | Start scan | Begin 5-min ATR monitoring |
| Every 5 min (9:30–2:30) | Entry scan | Check ATR squeeze, enter if signal |
| Every 5 min (9:30–3:15) | Position monitor | Check adjustments + exits |
| 3:15 PM | EOD squareoff | Close ALL positions |
| 3:20 PM | Daily summary | Log daily P&L, save equity curve |

### 9.3 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/nas` | GET | Dashboard page |
| `/api/nas/state` | GET | Full state for dashboard |
| `/api/nas/scan` | POST | Manual scan trigger |
| `/api/nas/scan/status/<id>` | GET | Poll background scan |
| `/api/nas/kill-switch` | POST | Emergency close all |
| `/api/nas/trades` | GET | Trade history |
| `/api/nas/orders` | GET | Order audit log |
| `/api/nas/signals` | GET | Signal history |
| `/api/nas/equity-curve` | GET | Daily P&L chart data |
| `/api/nas/toggle-mode` | POST | Switch paper/live |
| `/api/nas/toggle-enabled` | POST | Enable/disable |
| `/api/nas/option-chain` | GET | Latest option chain snapshot |

---

## 10. Name Justification

**NAS = Nifty ATR Strangle**

- **Nifty** — the instrument
- **ATR** — the core indicator driving entries
- **Strangle** — the options structure being sold

Internal code prefix: `nas_`
Dashboard label: "NAS Nifty ATR Strangle"
DB file: `nas_trading.db`
