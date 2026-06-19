# NAS 6 ATM Systems — Chain Replay + Mechanic Optimization (research/68)

**VERDICT: SIGNAL (strong, internally consistent, holds train/test) — NOT yet validated.**
41 days, single Apr–Jun-2026 regime, 1-min fills, no slippage (optimistic tail), multiple
configs tested → this MAPS the response surface and ranks levers; it is not a deployable
optimum. Treat as paper-forward guidance, not a green light to rewire a live book.

Data: `backtest_data/options_data.db`, NIFTY front-weekly per-minute chain, 41 days
2026-04-20→06-19 (snapshots ~09:20→15:30). Cost ₹160/strangle, lots=2 normalized.

---

## Phase A — Baseline (6 systems as-live, premium 1.3× SL, ATM, 15:15)

Combined lots=2: **+₹16,389**, Sharpe 0.62, MaxDD −₹53,404. Live-lots (916=1): +₹27,924, Sharpe 1.9.
Per-system: the **squeeze family is the edge** (Sharpe 3.5–7, tiny DD); the **916 family bleeds**
far-DTE (Sharpe −1.6 to −1.8) — 916-ATM2's cascade churns 280 legs.

## Key structural finding — DTE / weekday gradient (this drives everything)

| Weekday (DTE) | Squeeze (3 sys) | 916 (3 sys) | Combined |
|---|---|---|---|
| Mon (1) | +13,793 | +54,566 | **+68,358** |
| Tue (0) | +20,073 | +2,129 | +22,203 |
| Wed (6) | +12,924 | −61,120 | −48,197 |
| Thu (5) | +3,475 | −7,741 | −4,266 |
| Fri (4) | −10,805 | −10,903 | −21,709 |

- **Squeeze is positive every weekday except Friday** (its ATR-compression filter self-selects
  calm days → robust across DTE). **916 is a Monday(+Tue) edge that bleeds Wed–Fri.**
- The far-DTE bleed is **entirely the 916 family.** Friday is mean-negative but high-variance
  (3 of 8 Fridays green, incl. 06-19 +₹10,459).

## Phase B/C — Gating + the two prior levers (move-stop, OTM)

Candidate books (lots=2, 41d):

| Book | Net | MaxDD | Sharpe | Calmar |
|---|---|---|---|---|
| A. LiveCurrent (prem, Fri/Mon/Tue) — *what arms Monday* | +68,852 | −17,662 | 3.62 | 3.9 |
| **B. SmartGate (squeeze all-days + 916 Mon/Tue)** | **+96,153** | −9,162 | 6.17 | **10.49** |
| C. SqueezeOnly (drop 916) | +39,459 | −8,948 | 4.45 | 4.41 |
| D. MoveUnified (±0.4% move-stop) | +45,552 | −5,767 | 6.84 | 7.9 |

**SmartGate** (squeeze trades all days incl. Wed/Thu; 916 restricted to Mon/Tue) = the biggest
gating win: +40% net, half the DD, Calmar 3.9→10.5, **no code — matrix only.**

## Phase D — Mechanic optimization (all angles, SmartGate, interleaved train/test)

**Robust wins (improve on BOTH train and test):**
1. **Naked-leg SuperTrend multiplier 2.0 → 3.0** (period 7, 5-min kept; period barely matters,
   5-min > 3-min). ATM-systems naked contribution ~17k → ~35k, DD slightly better. A wider ST
   band stops whipsawing the profitable survivor out early.
2. **Naked-method alternative:** a simple **breakeven-lock** ≈ ST×3.0 (~33k) and is far simpler;
   **%-trail-from-low is bad** (whipsaw). So: widen ST *or* use breakeven.
3. **ATM2 cascade → ±0.4% underlying move-stop:** net ~doubles (24.8k→45.6k), Calmar 3.5→7.9,
   robust train+test. (Confirms the user's intuition.)

**Debunked / already-optimal (do NOT change):**
- **SL threshold:** 30% is near-optimal; 40% a tie; **wider (75–100%) inflates net via fat
  tails but wrecks DD** (Calmar collapses). Keep 30–40%.
- **Profit-target:** HURTS (35/50/65% all reduce net) — on a book already force-exited at 15:15,
  a PT just truncates winners while the SL still caps losers. Don't add.
- **Exit time:** monotonic — **later is better; 15:15 (current) is best**, early exits sharply
  negative. Keep 15:15.
- **Whole-position stop:** worse risk-adjusted than per-leg (Calmar 10.5 → 0.9–2.6, DD −55k to
  −105k). Keep per-leg SL.
- **Structure/width:** **ATM best**, OTM monotonically worse here. Keep ATM.

### Best stack (SmartGate + ST×3.0 + ATM2 move-stop)

| Book | Net | MaxDD | Sharpe | Calmar | DayWin | Train net | Test net |
|---|---|---|---|---|---|---|---|
| SmartGate current mechanic | 96,153 | −9,162 | 6.17 | 10.49 | 59% | 21,338 | 74,815 |
| **SmartGate BEST stack** | **130,095** | **−6,808** | **8.43** | **19.11** | **68%** | 49,411 | 80,684 |

+35% net, −26% DD, Calmar 10.5→19.1 — and it improves **both** train (21k→49k) and test
(75k→81k), so it is not an in-sample artifact. Gains come from 916-ATM (+17.6k, ST×3.0) and the
ATM2 systems (+10k each, move-stop).

## Honest caveats

- **41 days, one regime** (calm-ish Apr–Jun 2026 short-vol tailwind). ~8 obs/weekday. **SIGNAL,
  not validation.** Friday's negative mean rests on 8 noisy samples.
- **No slippage; 1-min SL/ST resolution; LTP fills** → optimistic, especially for stop fills and
  the move-stop. Real fills will be worse.
- **Squeeze entry is reconstructed** (Wilder ATR) — won't match live entry times trade-for-trade
  (live 06-19 squeeze entered 10:50; replay differs). 916 entry is exact.
- **Multiple configs compared** → the "best stack" Calmar 19 is selection-flattered; deflate it.
  The train/test agreement is reassuring but the test set shares the same regime.
- Lots=2 normalized; live 916 = 1 lot.

## Next levers

1. **Paper-forward** the SmartGate gate + ST×3.0 + ATM2-move stack alongside the live current
   mechanic (G5) before any live change.
2. Keep extending the recorder; re-run on a **trend/high-vol regime** when available (the short-vol
   tailwind is the biggest unknown).
3. Test per-DTE SL tiers explicitly (low priority — SL was a weak lever).
4. Model **slippage** on the move-stop / SL fills and re-rank.
