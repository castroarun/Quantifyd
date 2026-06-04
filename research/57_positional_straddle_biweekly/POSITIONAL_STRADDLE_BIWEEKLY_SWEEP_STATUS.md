# Positional Bi-Weekly Short Straddle + EOD Overnight Wing Protection — NIFTY

STATUS: RUNNING

## 1. The Ask (restated precisely)
**What you asked:** "carry positional straddles bi-weekly (always enter / manage in the next-next
week contract). By EOD buy some far-OTM market protection, removed next morning. The main part is
the straddle — how and where best to manage it. Work it out with the month of options data."

**What we're testing:** A **short ATM straddle** held POSITIONALLY (multi-day) in the **2nd-nearest
weekly expiry** (bi-weekly, ~8-12 DTE at entry). Overnight gap risk is capped by **buying far-OTM
CE+PE wings at EOD (~15:20) and removing them next morning (~09:20)** — defined-risk overnight,
naked intraday. Goal: find the best **management** (stop / profit-target / roll-DTE / entry) that
maximizes net-of-cost P&L while bounding the tail. Data = recorded NIFTY chain, 30 days (real premiums).

## 2. The Base (mechanics — locked for G0)
- **Underlying:** NIFTY. **Contract:** 2nd-nearest weekly expiry at entry (Tuesday weeklies; ~8-12 DTE).
- **Core:** SELL ATM straddle (ATM CE + ATM PE), 1 lot (65), MARKET. ATM = round(spot/50)*50.
- **Carry:** hold multi-day; mark MTM daily at 09:20 (post un-wing) and 15:20 (pre-wing).
- **Overnight wings:** at 15:20 BUY far-OTM CE (ATM+W) + PE (ATM-W), W≈far-OTM (G0: 500pts ≈ 2%);
  at 09:20 next day SELL them. Wings carried ONLY overnight (gap protection), cost = entry-exit.
- **Roll:** when the held expiry's DTE <= roll_dte, close the straddle (buy back) + (for cycle view)
  re-enter the new 2nd-nearest. **Costs:** Rs 80/leg.
- **Management (THE research question — to optimize, not fixed):** underlying-move stop %, premium
  stop, profit-target % of credit, max-hold / roll-DTE, entry day/time.

## 3. Plan (staged gates; kill cheap)
| Stage | Test | Gate |
|---|---|---|
| G0 | baseline naive carry (no stop) + measure: gross/net per-trade, daily theta decay, overnight-wing cost/benefit, worst overnight gap | does short bi-weekly straddle decay net-positive gross? is the wing worth its cost? |
| G1 | MANAGEMENT sweep — underlying-move stop (%), profit-target (% credit), roll-DTE — one axis at a time | which management lifts net + bounds tail (monotonic > peak) |
| G2 | WING optimization — distance W (300/500/700pt), keep-all-day vs overnight-only, cost vs gap-saves | best protection per rupee |
| G3 | ENTRY — which day/DTE to enter; hold-to-expiry vs roll-early | best entry/exit cadence |

## 4. Metric + guards
Rank by **net P&L per trade** (net of Rs80/leg) + **worst trade / overnight gap** (tail). 30 days =
SIGNAL not validation; few independent bi-weekly cycles + overlapping daily entries are correlated
(report per-trade stats + the overlap caveat). Net-of-cost throughout. Guard: look-ahead (mark at
discrete 09:20/15:20 only), cost neglect (wings + 80/leg counted), single-regime (Apr-Jun 2026 only).

## 5. Status log
| Time | Event |
|---|---|
| 2026-06-04 | research/57 folder + STATUS; building G0 engine on recorded chain |
| 2026-06-04 | G0 DONE: naked straddle +7430/trade (theta REAL); wings net-costly but cut gap tail. Launching G1 management sweep. |
| 2026-06-04 | G1 DONE: tight move-stop 1.5%% caps tail (-12k->-5k); wide stops backfire; PT40 raises mean to +9.1k. Next G2 = combo+wings. |

## 6. Crash recovery (run WITHOUT me)
- Scripts: `research/57_positional_straddle_biweekly/scripts/*.py` — standalone, read-only on
  `backtest_data/options_data.db`. Run: `cd /home/arun/quantifyd && ./venv/bin/python3 research/57_.../scripts/<x>.py`.
- Each stage writes `results/RESULTS_<stage>.md` + PNG + a per-trade CSV. Re-run is idempotent.
- This STATUS doc is the single source of truth for what's been tested + the cumulative verdict
  (kept current after every run). Final = `results/RESULTS.md` with the bold verdict label.
- Safe to run any time — no service impact, no Kite calls (reads recorded premiums only).

## 7. Files
| File | Purpose | Commit? |
|---|---|---|
| `scripts/biweekly_engine.py` | shared carry engine (chain reader + straddle MTM + wings) | yes |
| `scripts/g0_baseline.py` ... | per-stage runners | yes |
| `results/RESULTS_*.md`, `*.png`, `*_trades.csv` | per-stage findings | yes (small) |
| `results/RESULTS.md` | final cumulative verdict | yes |

## 8. Findings (cumulative — updated every run)
**G0 (26 trades, real chain):** naked short bi-weekly straddle decays NET-POSITIVE — mean +Rs7,430/trade, median +Rs8,988, 65%% win, worst -12,396. EOD ±500 wings are NET-COSTLY (-Rs36,804 total -> mean drops to +6,014) BUT raise win%% 65->73 by catching gap nights (+13-16k late-May down-gaps); they DON'T help sustained-move losers. => core theta edge REAL; G1 tests move-stop + profit-target to cut the trend losers.

**G1 management (26 trades):** TIGHT move-stop 1.5%% cuts worst -12.4k->-5.0k, std -24%%, for ~Rs1.4k mean give-up. WIDE stops 2-3%% BACKFIRE (exit late -> worst -23k). Profit-target 40%%-of-credit RAISES mean +7.9k->+9.1k (median +15k) by banking quick theta before directional weeks. NEXT: combine PT40 + move1.5 + re-add optimized overnight wings (G2).

| 2026-06-04 | G2 DONE: COMBO (move1.5+PT40) = WINNER +7792/trade, worst -4999, 73% win (keeps full mean AND caps tail). WINGS REDUNDANT once stop is in. Recipe locked = SIGNAL. Next G3 entry-timing + forward paper logger. See results/RESULTS.md. |
| 2026-06-04 | G2b GAP-RISK (user Q): overnight gaps max -1.25%% (below 1.5%% stop) in sample. GAP-AWARE stop (check 09:20 open too) is BETTER: +8072/trade, 81%% win, worst -4917. Wings STILL dont add P&L gap-modeled BUT sample has NO large-gap event -> wings = cheap tail insurance vs unobserved gap. Recipe refined: stop must be CONTINUOUS/intraday (fire at open after gap), not daily-EOD. |
| 2026-06-04 | G3a INTRADAY-FREQ (user Q, after fixing a day-boundary bug): CONTINUOUS intraday stop BACKFIRES - whipsaws out of reverting trades, mean +7133->+3000, win 69->50, NO better worst. OPTIMAL = check stop at OPEN(09:20)+CLOSE(15:20) twice-daily = +8072/trade, 81% win, worst -4917. Corrects earlier continuous-is-better. Caveat: no intraday crash in sample (continuous would catch one). |