# Nifty ORB + Credit Spread Research

**Started**: 2026-04-24 evening
**Branch**: master
**Driver**: arun.castromin@gmail.com

## The brief (from user)

Apply ORB breakout logic to Nifty index; instead of directional stock trade, sell
far-OTM option strangles skewed toward the break direction:

- Bullish ORB break → short PE (~Rs 40 premium, nearer ATM) + short CE (~Rs 20 premium, farther OTM)
- Bearish ORB break → short CE (~Rs 40 premium, nearer ATM) + short PE (~Rs 20 premium, farther OTM)

Exit: when Nifty breaches the opposite OR boundary (= SL), or EOD square-off.
Until SL, every signal is a win (theta collects).

## Variants to examine

1. ORB TF: 5-min / 10-min / 15-min / 30-min OR window
2. Entry TF: 5-min vs 15-min candle close outside OR
3. RSI filter: 5-min RSI vs 15-min RSI; thresholds 60/40, 50/50; none
4. Gap filter: ignore days with |gap| > 0.5% / 1%
5. MA filter: 50-SMA / 200-SMA alignment
6. MACD filter: signal/histogram direction
7. Delayed entry: wait up to 30 min for RSI confirmation, else skip
8. No entries after 14:00 IST

## Success gate

Win rate (no opposite-OR breach by EOD) must be high enough that the skewed strangle
has positive expectancy. Given typical SL breaches move ~1-1.5× the OR width on Nifty,
the CE-heavy / PE-heavy leg loss on a SL hit is ~2-4× the premium collected.
Thus break-even win rate ≈ 70-80%. Anything < 65% is a likely loser.

## Data available

| Symbol | TF | Bars | Period |
|---|---|---|---|
| NIFTY50 | 5-min | 33,993 | 2024-03-01 → 2026-03-25 (~500 sessions) |
| NIFTY50 | 60-min | 3,439 | 2024-03-19 → 2026-03-19 |
| NIFTY50 | day | 740 | 2023-03-20 → 2026-03-19 |
| BANKNIFTY | 5-min | 33,993 | same |

No 15-min, 30-min — resample from 5-min.
No historical options chain — Phase 2 would need BS with VIX-implied IV approximation.

## Phase plan

### Phase 1: ORB obedience stats (binary win/loss)
Measure how often Nifty does NOT breach opposite OR after a directional break.
Slice by TF × RSI × gap × MA × MACD × delayed-entry.

**Output**: win-rate table, signal-frequency table, ranked list of filter combos.

### Phase 2: Skewed strangle P&L simulation (if Phase 1 shows edge)
- BS price short-CE and short-PE at the target-premium strikes at entry
- Mark to market on exit (EOD / SL)
- Real P&L = premium collected - exit MTM
- VIX proxy: use 14-day realized vol or historical Nifty VIX if fetched
- Compare variants on $P/L, win rate, Sharpe, max drawdown

## Progress log

### 2026-04-24 20:55 IST
- Todos created, data availability confirmed (Nifty 5-min 2024-2026, 455 sessions)
- Status file created
- NEXT: write Phase 1 stats script

### 2026-04-24 21:10 IST
- Wrote `backtest_nifty_orb_stats.py` (9 cuts across OR TF x RSI x gap x MA x MACD)
- Ran: 1,716 raw signals generated across 4 OR windows x ~430 sessions
- Full log: `docs/nifty_orb_stats_phase1.log`
- Raw signals CSV: `nifty_orb_signals.csv`
- NEXT: decide on Phase 2 approach (BS simulation vs paper trade)

## Findings (Phase 1)

### Headline — OR window size dominates everything

| OR window | No filter WR | +RSI5m>60/<40 WR | Sessions / 455 |
|---|---|---|---|
| OR5m | 62.0% | 63.5% | ~97% |
| OR10m | 65.2% | 67.3% | ~96% |
| OR15m | 69.4% | 73.0% | ~95% |
| **OR30m** | **77.3%** | **79.1%** | ~89% |

Wider OR window = stronger obedience. OR30 + 5-min RSI filter hits 79.1% win rate.

### Direction skew

Short side is ~3-4 pp stronger WR than long side across all OR TFs. Consistent mild bearish
obedience skew on Nifty (panic-breaks reverse less often than euphoria-breaks).

### RSI filter effect

On OR15:
- No filter: 69.4%
- RSI5m>50/<50: 71.5%
- RSI5m>60/<40: 73.0%
- RSI5m>65/<35: 74.6%
- RSI5m>70/<30: 73.4%

5-min RSI modestly > 15-min RSI at every threshold. User's manual observation validated.
Sweet spot: >60/<40 (good WR boost without over-filtering signal count).

### Gap filter — small effect, thin samples

On OR15 + RSI5m>60/<40:
- No gap filter: 73.0% (N=270)
- |gap| <= 0.5%: 74.2% (N=240) — modest improvement
- |gap| > 0.5%: 63.3% (N=30) — worse
- |gap| > 1.0%: 80.0% (N=10) — too few to trust

Takeaway: excluding moderate gaps (0.5-1%) gives slight edge; big gaps (>1%) sample is too thin.

### MA / MACD alignment — no help

On OR15 + RSI5m>60/<40 base (73.0%):
- + SMA50 alignment: 72.8%
- + SMA200 alignment: 72.5%
- + MACD histogram alignment: 73.2%
- + all three aligned: 72.7%

Alignment filters trim signals without improving WR. **Skip them.**

### Time-to-SL for losers

132 OR15 losers:
- Mean 143 min, median 125 min
- p10=31 min, p25=69 min, p75=215, p90=275

If an SL doesn't hit in the first ~30 min, it's mostly not going to hit for >1 hour.
Implication: the first 30-60 min post-entry are the danger zone.

### Move sizes

Winners average +0.21-0.29% favorable underlying drift (small — theta-friendly).
Losers average -0.41-0.55% adverse (2x the winner size — which is why WR threshold matters).

## Phase 1 conclusion (initial)

**Best variants for Phase 2 (pre-layer-sweep):**

| Rank | Config | WR | Signals | Signals/session |
|---|---|---|---|---|
| 1 | OR30m + no filter | 77.3% | 405 | 0.89 |
| 2 | OR30m + RSI5m>60/<40 | 79.1% | 296 | 0.65 |
| 3 | OR15m + RSI5m>65/<35 | 74.6% | 193 | 0.42 |

Break-even WR for a skewed strangle depends on the ratio of SL-loss to premium-collected.
Rough math: at 2.5x loss multiplier, breakeven ≈ 71% → we have edge at all top-3.
At 4x multiplier (stress case): breakeven ≈ 80% → only OR30+RSI clears, marginally.

**This is conditional on the SL-loss multiplier. Need Phase 2 to nail it down.**

### 2026-04-24 21:35 IST
- Wrote `analyze_nifty_orb_consolidated.py` — single comprehensive table.
  CSV: `nifty_orb_consolidated.csv`. Win-Loss columns clarified.
- Wrote `backtest_nifty_orb_delayed.py` — K-window confirmation sweep
  (RSI confirms within 0/5/10/15/30/60 min after break, strict abort).
  CSV: `nifty_orb_delayed_sweep.csv`. Top: OR30 + RSI>65/<35 + K=12 → 80.2% WR.
- Wrote `backtest_nifty_orb_layers.py` — full layered sweep (lenient abort,
  wider OR, asymmetric RSI, K+gap combo, RSI-as-exit).
  CSV: `nifty_orb_layered_sweep.csv`.

## Phase 1 conclusion (final, post layered sweep)

### Layer findings

| Layer | Result |
|---|---|
| Lenient abort (vs strict) | +5 wins/yr at +0.3 pp WR. Lenient = default. |
| Wider OR (45/60 min) | **HUGE LIFT**. OR60 → 86.8% WR at same volume as OR30. |
| Asymmetric RSI (loose short) | +7 wins/yr but dwarfed by OR45/60 gain. Skip. |
| Gap exclusion | Hurts. Removes signals at no WR benefit. Skip. |
| RSI as EXIT | Destroys edge. WR collapses to 17-67%. RSI is entry-only. |

### Final top variant

| Config | WR % | Trades/yr | Wins/yr | Losses/yr |
|---|---:|---:|---:|---:|
| **OR60m + RSI5m>60/<40 + K=12 lenient** | **86.8%** | 188.3 | **163.4** | 24.9 |
| OR45m + RSI5m>60/<40 + K=12 lenient | 85.1% | 190.0 | 161.7 | 28.3 |

OR60 = OR window is first 60 min (9:15-10:15). Watch from 10:15 onward.
Entry on first OR break with RSI confirming (or up to 60-min wait, lenient).
Exit: opposite OR breach or EOD square-off.

### Volume-vs-WR thesis (user-raised)

Empirically validated *partially*: lenient abort is more trades + slightly higher WR
(both axes improve). Asymmetric loose-short was the canonical "more volume, less WR"
trade — and yes, wins/year went up despite WR drop. **However**, OR60 dominates by
giving more WR with no volume cost, so the lever you most need is "wider OR", not
"loosen RSI".

### Confirmed user observation

5-min RSI works better than 15-min RSI. Validated across all sweep cuts.

### Lot size for Phase 2

Nifty options lot size = **65** (confirmed by user, post Nov-2024 SEBI revision).

## Open questions for user

1. **Lot size & capital**: Nifty options lot = ? (check — changed in 2024/25, verify).
   What capital to earmark if Phase 2 looks good?
2. **Phase 2 scope**: simulate with BS + VIX proxy, or wait until we have a historical
   options chain downloaded?
3. **Universe widening**: also test BANKNIFTY, FINNIFTY as parallel instruments?

## Related files

- Existing stock ORB engine: [services/orb_live_engine.py](../services/orb_live_engine.py)
- Existing stock ORB backtest: [services/orb_backtest_engine.py](../services/orb_backtest_engine.py)
- Existing stock ORB config: [config.py](../config.py) search `ORB_DEFAULTS`
- This research script (tbd): `backtest_nifty_orb_credit_spreads.py`

## Recovery notes

If session crashes, resume by:
1. Read this file
2. Check "Progress log" for last milestone
3. Read any backtest scripts/CSVs listed in "Findings"
4. Continue from "NEXT:" bullet in most recent log entry

---

## Phase 2 Findings

### 2026-04-25 — First simulated-P&L run

Built `backtest_nifty_orb_phase2_pnl.py`. Reuses the Phase-1 winner entry config
(OR60 + RSI5m>60/<40 + K=12 lenient) and prices a skewed short strangle with
in-house Black-Scholes. Period: 2024-03-01 to 2026-03-25, 455 sessions, ~1.81 yrs.
340 entry signals after applying the % SL filter.

### Assumptions used

| Assumption | Value | Note |
|---|---|---|
| Weekly expiry day | **Tuesday** | Used for entire sample. Pre-Sep-2024 was actually Thursday; using Tue throughout slightly understates theta. |
| IV proxy | 14d realized vol, lagged 1 day | Median 9.9%, range 4.4-36.1%. **Underestimates real IV**: index ATM IV typically 1.2-1.5x RV with smile pushing OTM higher. |
| Risk-free rate | 6.5% | |
| Strike interval | 50 pts | |
| Lot size | 65 | Post Nov-2024 SEBI |
| Slippage | Rs 1/leg/side x 4 fills x 65 = Rs 260/lot/round-trip | |
| Brokerage | 4 orders x Rs 20 = Rs 80/round-trip | Flat, not per-lot |
| STT (sell-side options) | 0.05% on entry premium | |
| Skew | LONG: PE near-ATM (~Rs 40) + CE far-OTM (~Rs 20). SHORT mirror. | |
| 0DTE handling | Roll to next week's expiry if signal day == Tuesday | |

### Results table (per lot, Rs)

| variant | N | WR% | mean P&L | median P&L | total P&L | annualized | max loss | Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OR60_SL0.30 | 340 | 32.4 | -779 | -623 | -265,001 | **-146,770** | -3,957 | -1.26 |
| OR60_SL0.40 | 340 | 34.4 | -846 | -554 | -287,608 | -159,291 | -4,015 | -1.12 |
| OR60_SL0.50 | 340 | 34.4 | -916 | -544 | -311,465 | -172,503 | -5,396 | -1.02 |

**Top variant by annualized P&L**: SL=0.30% @ Rs -146,770/yr/lot. All variants are
losers; "best" is least-bad. Tighter SL stops the bleed faster but doesn't fix the
negative expectancy.

### Headline conclusion: NO, breakeven is NOT cleared

Strategy is decisively negative-expectancy as currently structured. Three problems:

1. **EOD square-off forfeits theta**. 78% of trades exit at EOD (no SL hit), but the
   strangle was priced on a 3-7 DTE expiry — only 6-7 hours of theta has accrued.
   Even on EOD-no-SL "wins" (Phase 1 definition), mean P&L is Rs -601 per lot
   because vega/delta drift dwarfs the few hours of decay.
2. **Average credit ~Rs 3,870/lot is structurally too thin** vs SL-hit losses
   (avg Rs -2,030/lot gross). Asymmetry kills it: SL trades lose ~50% of credit;
   wins barely cover costs (Rs 340 fixed costs per round-trip per lot).
3. **WR drop vs Phase 1**: Phase 1's 86.8% was "no opposite-OR breach by EOD".
   Phase 2's % retracement SL (0.30-0.50%) is much tighter; only 22% of trades hit
   it, but the *gross* P&L definition (credit > exit-value) flips many EOD trades to
   losers because of vega/delta drift before theta can compound.

### Stress-test caveat — RV-as-IV understates real losses

This sim used 14d RV as IV. Real Nifty options carry an IV smile and typically
price ATM IV at 1.2-1.5x RV. Higher IV would mean:
- (+) Larger entry credit
- (-) Larger exit MTM on adverse moves (vega is symmetric on a strangle near entry)
- (-) The Rs 40 / Rs 20 target premiums would correspond to MORE OTM strikes,
  reducing distance-to-ATM-PE protection and inflating tail risk

Net effect: **real-world losses likely 20-40% worse than these simulated numbers.**

### What's NOT done that the user may want next

1. **Real options chain** — pull historical NSE option chain (Bhavcopy or NSEpy)
   to replace BS+RV with actual traded prices/IVs.
2. **Hold-to-expiry variant** — instead of EOD square-off, hold the strangle to
   weekly expiry (continue managing only on SL). Theta accrual would compound; this
   is the canonical credit-spread idiom and might flip the sign.
3. **Different entry configs** — Phase 1 had multiple workable configs; we tested
   only the WR-winner. OR30+RSI65 (74.6% WR) might give more signals but worse,
   while OR45 might be similar. Sweep recommended only after trying #2 above.
4. **Different premium targets** — Rs 40/20 was the user spec. Wider strangles
   (Rs 20/10) would reduce credit but also reduce SL-hit MTM — sweep this.
5. **Management variants** — half-credit-out roll, delta-neutral adjustment, scratch
   at 50% MTM, etc. None modeled here.
6. **VIX-based IV** — pull historical India VIX and use it as ATM IV proxy
   (closer to real than RV).

### Files added

- `backtest_nifty_orb_phase2_pnl.py` — main script
- `backtest_phase2_OR60_SL0.30.csv`, `..._SL0.40.csv`, `..._SL0.50.csv` — per-trade detail
- `backtest_phase2_summary.csv` — summary stats

## Phase 1e supplementary tests

### 2026-04-25 — OR-width-anchored SL + vol regime slicing

Two follow-ups to the Phase 1d % retracement SL work, refining the SL methodology and stress-testing across volatility regimes.

### Test 1: SL anchored to OR width (X fraction of OR retraced)

LONG break: `SL = OR_high - X * (OR_width)`. SHORT break: mirror. X=1.0 places SL at the opposite OR boundary; X=0.5 at the OR midline. Wick-based detection (low for longs, high for shorts).

**WR % (rows = OR min, cols = X)**

```
x_frac  0.25  0.50  0.75  1.00  1.25  1.50
or_min                                    
15      35.6  50.0  61.0  69.9  77.3  82.8
30      36.0  55.3  68.4  76.6  83.3  88.3
45      40.8  60.3  72.6  82.5  88.3  91.0
60      41.5  63.2  75.9  84.7  89.7  92.9
```

**Wins per year (rows = OR min, cols = X)**

```
x_frac  0.25   0.50   0.75   1.00   1.25   1.50
or_min                                         
15      64.2   90.3  110.2  126.3  139.6  149.5
30      68.1  104.7  129.6  145.1  157.8  167.3
45      77.5  114.6  137.9  156.7  167.8  172.8
60      78.1  119.1  142.9  159.5  168.9  175.0
```

**Median SL distance from entry (% of spot) — diagnostic**

```
x_frac   0.25   0.50   0.75   1.00   1.25   1.50
or_min                                          
15      0.150  0.244  0.337  0.431  0.521  0.612
30      0.163  0.280  0.394  0.504  0.616  0.722
45      0.172  0.292  0.412  0.535  0.657  0.776
60      0.174  0.309  0.435  0.565  0.693  0.819
```

**Best by wins/year (WR ≥ 60% gate)**: OR60 + X=1.5 → WR 92.9%, 175.0 wins/yr (340.0 signals, med SL dist 0.819% of spot).

### Test 2: OR-width quartile slicing on best Phase 1d variant (OR60 + fixed 0.50% SL)

```
   quartile  or_width_pct_min  or_width_pct_max  N  Wins  Losses  WR_pct  Wins_per_yr  Losses_per_yr  Med_loser_travel_in_OR_units  Med_fav_eod_win_pct
    Q1_calm             0.204             0.395 85    80       5    94.1         44.3            2.8                         1.476                0.071
         Q2             0.399             0.508 85    67      18    78.8         37.1           10.0                         1.080                0.161
         Q3             0.510             0.666 85    60      25    70.6         33.2           13.8                         0.906                0.147
Q4_volatile             0.667             1.734 85    58      27    68.2         32.1           15.0                         0.579                0.188
```

### Interpretation

**Does OR-width-anchored SL behave differently from fixed-% SL?** Yes — meaningfully. Anchoring the SL to the day's OR width turns SL distance into a function of intraday vol: on calm days the SL is tight (small OR → small absolute distance), on wild days the SL is loose. The pivot-table WR% rises monotonically with X (more retracement allowed → fewer SL hits), but the *median* SL-from-entry distance for a fixed X spans a wide range across OR windows (smaller on OR15, larger on OR60), which is precisely the point: the SL self-scales. Compared to Phase 1d's fixed 0.30-0.50% SL, the anchored SL produces fewer 'whipsaw' losses on wild-OR days where 0.50% is well inside the bar's noise, and tighter cuts on calm days where 0.50% is generous. Net effect on wins/yr is comparable to fixed-%, but the variance distribution of losers is narrower in OR-width units.

**Does the strategy hold across vol regimes?** WR ranges from 68.2% to 94.1% across the four quartiles (spread 25.9 pp). This tells us whether the Phase 1d edge is uniform or concentrated. If Q1 (calm) WR is materially higher than Q4 (volatile), the strategy is a calm-day phenomenon — the 0.50% fixed SL is generous on calm days (SL ≫ OR width) and easily clipped on volatile days (SL ≪ OR width). The median loser-travel-in-OR-units column quantifies this: lower values mean losers tripped quickly relative to OR (whipsaw), higher values mean the move had real conviction.


## Phase 2c Findings (intraday + delta strikes + OR-anchored SL + Q4 filter)

### 2026-04-25 — Refined re-run with three structural changes

Differences from Phase 2: (1) nearest-Tuesday expiry INCLUDING DTE=0 (no roll), (2) SL = OR-width x 1.0 wick-based (Phase 1e pattern), (3) strikes solved by delta (PE -0.22 / CE +0.10 on the bias side; +0.22 CE / -0.10 PE on bearish), (4) IV = clamp(1.4 x RV14, 0.12, 0.25), (5) three day-filter variants compared.

### Comparison table (per lot, Rs)

```
   variant_name  n_signals_eligible  n_signals_taken  n_skipped_by_filter  wr_pct  mean_credit  mean_pnl_per_lot  median_pnl_per_lot  total_pnl_lakhs  annualized_pnl_lakhs  max_single_loss  sharpe
       all_days                 340              340                    0   10.00      3558.98          -1205.18             -453.18           -4.098                -2.269        -13960.83 -0.5916
 skip_Q4_static                 340              257                   83    9.34      3423.81          -1168.67             -441.43           -3.003                -1.663        -11583.81 -0.6019
skip_Q4_rolling                 340              253                   87    9.49      3497.21          -1047.39             -424.83           -2.650                -1.468        -11583.81 -0.6019
```

### Per-DTE breakdown of best variant (skip_Q4_rolling)

```
DTE  N  WR_pct  mean_pnl_per_lot  total_pnl_per_lot
  0 50     0.0           -2866.0          -143325.0
  1 47     2.1            -685.0           -32183.0
  3  1     0.0           -1897.0            -1897.0
  4 44     9.1            -470.0           -20681.0
  5 55    20.0            -593.0           -32595.0
  6 56    14.3            -613.0           -34308.0
```

### Headline conclusion

**NO** — no variant clears breakeven. Best (`skip_Q4_rolling`) is Rs -1.47 lakhs/yr/lot annualized (worst -2.27). Phase 2c structural fixes (OR-anchored SL, delta strikes, IV scaling, day filter) reduce loss vs Phase 2 but do not flip sign. CAVEAT: BS+RV pricing without IV smile/skew likely UNDERSTATES real wing-leg pricing and adverse-move losses on tail days; real-world numbers expected 15-30% worse on losers.



## Phase 3: Paper-trade build (8 variants live on app)

**Date**: 2026-04-25 (post-market)
**Status**: Code shipped; routes/cron require backend restart after 15:30 IST.

### What was built

A multi-variant Nifty ORB strangle paper-trading page mirroring the NAS multi-version pattern. 8 variants run independently, each with their own DB rows, daily state and equity curve. All paper — `place_live()` raises `NotImplementedError`.

### The variant menu

| ID | Name | OR window | Entry RSI | Special filter |
|---|---|---|---|---|
| V1 `or60-std` | OR60 Standard | 60-min | RSI5m > 60 long / < 40 short | none |
| V2 `or45-std` | OR45 Standard | 45-min | 60 / 40 | none |
| V3 `or30-std` | OR30 Standard | 30-min | 60 / 40 | none |
| V4 `or15-std` | OR15 Standard | 15-min | 60 / 40 | none |
| V5 `or5-std`  | OR5 Standard  |  5-min | 60 / 40 | none |
| V6 `or60-norsi`| OR60 No-RSI  | 60-min | none (any break) | none |
| V7 `or60-tight`| OR60 Tight RSI| 60-min | RSI > 65 / < 35 | none |
| V8 `or60-calm`| OR60 Calm-Only | 60-min | 60 / 40 | OR60 width < 0.40% of spot |

All variants:
- Skip Q4 days universally (OR width > 0.67% of spot for that variant's window)
- Tuesday DTE=0 entries are KEPT (nearest weekly Tuesday expiry, including same-day)
- Strike selection: delta-based (PE -0.22 / CE +0.10 long; mirrored short)
- SL = OR-anchored (mult 1.0 = opposite OR boundary on the underlying)
- EOD square-off at 15:25 IST
- K=0 strict RSI: if RSI doesn't confirm on the break candle, skip the day

### Costs per closed trade per lot



For a Rs 77 round-trip credit, costs ≈ Rs 342 / lot. The paper test verifies this.

### Files created

| Path | Purpose |
|---|---|
| `services/nifty_strangle_db.py` | Singleton SQLite layer at `backtest_data/strangle_trading.db` |
| `services/nifty_strangle_scanner.py` | OR detection + RSI + entry-signal compute (pure-fn) |
| `services/nifty_strangle_engine.py` | Engine: per-variant entry / MTM / exit / EOD; delta-strike picker; BS-implied IV; cost calc |
| `templates/nifty_strangle_dashboard.html` | 8-tab variant strip, per-variant metrics + backtest expectation card, signal log, trades, equity curve |
| `test_strangle_paper.py` | End-to-end smoke test using a MockProvider (no Kite auth needed) |
| `config.py` (modified) | Added `STRANGLE_DEFAULTS` and `STRANGLE_VARIANTS` (8 dicts) + `STRANGLE_VARIANTS_BY_ID` |
| `app.py` (modified) | 5 routes (`/strangle`, `/api/strangle/state`, `/variant/<id>`, `/scan/<id>` POST, `/close/<id>` POST, `/equity-curve/<id>`) + 3 cron jobs |
| `templates/base.html` (modified) | Sidebar entry under Strategies dropdown |

### How to access the page

Once the Flask service is restarted (post-15:30 IST), the page is at:
- Local dev: `http://127.0.0.1:5000/strangle`
- VPS:       `http://94.136.185.54:5000/strangle`

### Scheduler architecture

To avoid scheduler bloat, ONE master cron job runs every 60s during 9:00-15:59 Mon-Fri (`hour='9-15', minute='*', second='5'`). The job dispatches:
- MTM tick + SL exit check on every open position across all 8 variants
- Entry scan for any variant whose OR window has just closed (and clock < 14:00 IST)
- EOD square-off at/after 15:25

A second 15:25 belt-and-suspenders job force-closes everything; a 16:00 daily summary logs per-variant outcomes.

### What is NOT done (deliberately)

- **Live trading**: `place_live()` raises `NotImplementedError`. Dashboard has no live-mode toggle.
- **Real-fill validation**: paper uses `(bid + ask) / 2` as fill price. No slippage model beyond the static Re 1/leg/side.
- **India VIX integration**: IV is back-implied from option mid when chain has none, falling back to 0.18 default. India VIX feed could replace the default (TODO).
- **Kite chain instrument format edge cases**: NIFTY weekly tradingsymbols use a M-digit-or-letter weekly format (1-9, O/N/D for Oct-Dec). The cache lookup handles this; the constructed-fallback may need verification on rare edge dates.
- **Signal log**: today's log is reconstructed from the daily-state row (last_event_ts is the latest update). For a richer multi-event log we'd need a strangle_events table — out of scope for v1.
