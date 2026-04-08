# Nifty Monthly Straddle — Strategy Design Document

**Author:** Arun | **Date:** 8 April 2026 | **Status:** To Be Backtested

---

## Overview

Sell ATM straddle on NIFTY monthly expiry, initiated on the day after current month's expiry. Three distinct adjustment variants to manage breaches.

## Entry Rules

- **Timing:** Day after monthly expiry (e.g., if Jan expiry is 25th, enter on Jan 26)
- **Time:** 11:00 AM IST (let morning volatility settle)
- **Strike:** ATM (nearest to spot)
- **Instrument:** NIFTY monthly options (next month expiry)
- **Position:** Short 1 ATM CE + Short 1 ATM PE (straddle)
- **DTE at entry:** ~28-30 days

## Breach Definition

Straddle "breach" = spot moves beyond the breakeven point on either side.
- Upper breakeven = Strike + Total premium collected
- Lower breakeven = Strike - Total premium collected

---

## Three Adjustment Variants

### Variant 1: Double Straddle (Widen the Net)

**Logic:** At breach, don't close — add a second straddle further from spot to widen the overall breakeven range.

1. Initial straddle at ATM (e.g., 23000 CE + PE)
2. Spot breaches upper breakeven (say 23500)
3. Deploy Straddle 2 at new ATM (23500 CE + PE)
4. Now spot is between Straddle 1 (23000) and Straddle 2 (23500)
5. **Final exit:** Breach of the COMBINED breakeven (wider than either individual straddle)
6. Max straddles: 2 (no third adjustment)

**Risk profile:** Higher margin, wider breakeven, but doubled notional if both straddles go wrong in same direction.

### Variant 2: Close & Re-deploy (Reset ATM)

**Logic:** At breach, close the losing straddle entirely and deploy fresh ATM straddle. Repeat.

1. Initial straddle at ATM (23000)
2. Spot breaches upper breakeven
3. Close both legs of Straddle 1 (book loss on CE, profit on PE)
4. Deploy new ATM straddle at current spot (23500)
5. If breached again → repeat (close + redeploy)
6. **Max re-deployments:** TBD (2-3 suggested, after which accept loss)

**Risk profile:** Each reset crystallizes a loss but recenters. Risk of serial whipsaws in trending market.

### Variant 3: Roll the Losing Leg (Premium Match)

**Logic:** At breach, close ONLY the losing leg, roll it to OTM matching the winning leg's premium.

1. Initial straddle at ATM (23000 CE + PE, each at ₹300)
2. Spot rallies → CE is losing (say ₹500), PE is winning (say ₹100)
3. Close the CE (₹500 loss vs ₹300 collected = ₹200 net loss on leg)
4. Sell new OTM CE where premium ≈ ₹100 (matching PE premium)
5. Now position: Short PE @23000 + Short OTM CE (say 23800)
6. **Exit:** Breach of new combined breakeven

**Variant 3b (Double Down):** Same as above but when rolling, double the quantity on both sides. Higher premium collection but 2x risk.

---

## Assessment & Key Considerations

### Strengths
- Monthly expiry = high premium (vs weekly), more time for theta decay
- 11 AM entry avoids morning gap volatility
- Day-after-expiry entry maximizes DTE
- All 3 variants have defined adjustment triggers (not discretionary)

### Risks & Concerns

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Trending months (Budget, elections, global events)** | HIGH | Variant 2 resets help, but serial losses compound. Need max-loss cap per month. |
| **Gap-up/down next day** | HIGH | 11 AM entry helps but a 3% gap can blow past breakeven at open. Consider SL-M backup orders. |
| **VIX crush after entry** | MEDIUM | Actually helps short straddle — premiums fall, you profit. |
| **VIX spike (event-driven)** | HIGH | Premiums explode, unrealized MTM loss spikes. Need margin buffer. |
| **Margin requirements** | MEDIUM | Naked straddle on NIFTY requires ~₹1.5-2L per lot. Variant 1 (double straddle) needs 2x margin. |
| **Variant 3b (double down)** | HIGH | Doubling on a losing trade is dangerous. One big move = catastrophic loss. Suggest max 1 double. |
| **Physical settlement risk** | LOW | Monthly NIFTY is cash-settled (index options). No physical settlement concern. |

### Expected Returns (Rough Estimates)

- ATM NIFTY monthly straddle premium: ~₹800-1200 (4-5% of spot)
- If 70% of months expire within breakeven: ~₹700-900 avg monthly profit per lot
- If 30% of months breach and adjustment costs ~₹400-600: net ~₹500-700/month/lot
- Annual estimate: ₹6000-8400/lot = ~25-35% on margin deployed
- **Caveat:** One 2-sigma move can wipe 3-4 months of profits. Max monthly loss cap is critical.

### Recommended Parameters for Backtesting

| Parameter | Value | Notes |
|-----------|-------|-------|
| Entry day | Expiry+1 | Day after monthly expiry |
| Entry time | 11:00 AM | Post-morning settlement |
| Strike | ATM | Nearest 50-strike to spot |
| Expiry | Next month | ~28-30 DTE |
| Breach trigger | Spot crosses breakeven | BE = Strike ± total premium |
| Max adjustments | 2 (V1), 3 (V2), 1 (V3) | Prevent infinite adjustments |
| Time exit | 6 DTE (1 week before expiry) | Close ALL positions — avoid gamma risk |
| Max monthly loss | 2× premium collected | Hard stop |
| VIX filter | Entry only if VIX > 12 | Avoid selling cheap straddles |
| Backtest period | Jan 2020 - Mar 2026 | Includes COVID crash, election cycles |

---

## Backtesting Approach

### Option 1: Python Backtest (Preferred)

Build using our existing options data infrastructure:
- `services/options_data_manager.py` captures NIFTY option chains every minute
- Need: historical option chain data (prices at specific strikes/expiries)
- Can simulate entry, track premium decay, detect breaches, apply adjustments
- **Gap:** We may not have enough historical option price data (only capturing since recently)

### Option 2: Opstra Options Simulator (Browser Automation)

Opstra (opstra.definedge.com) has:
- Options strategy builder with payoff diagrams
- Historical P&L simulation
- Can backtest straddles with adjustments

**Can Claude automate Opstra via browser?**
- **Yes, technically possible** using browser automation tools (Puppeteer, Playwright, or Selenium)
- Claude can write the automation script, but executing it requires a browser runtime
- Approach: Write a Python Selenium/Playwright script that:
  1. Logs into Opstra
  2. Sets up the straddle parameters
  3. Simulates each month's entry/exit
  4. Captures P&L results
  5. Exports to CSV
- **Limitation:** Opstra's UI changes could break the script. Also rate limiting on repeated queries.
- **Better approach:** Use Opstra's underlying API if available (inspect network tab for XHR calls)

### Option 3: Hybrid

Use Opstra manually for a few months to validate, then build Python backtest for full 6-year period.

---

## Action Items

- [ ] Capture NIFTY monthly option chain data (if not already available) for backtest period
- [ ] Build `NiftyMonthlyStraddleBacktest` engine with 3 variant configs
- [ ] Backtest all 3 variants over Jan 2020 - Mar 2026
- [ ] Compare: CAGR, MaxDD, Sharpe, monthly win rate, max consecutive losses
- [ ] Investigate Opstra API (network tab inspection) for automated backtesting
- [ ] If no API, build Playwright browser automation for Opstra simulator
- [ ] Paper trade the winning variant for 2 months before going live

---

*Document generated: 8 April 2026*
