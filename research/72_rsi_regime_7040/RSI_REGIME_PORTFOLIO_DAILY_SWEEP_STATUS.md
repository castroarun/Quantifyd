# RSI Momentum-Regime Slot Portfolio — Does Diversification Beat the Nifty? (Nifty50 & Nifty200, N=5/10/15/20)

STATUS: DONE

## 2. The Ask
- **What was asked:** Build a slot-based portfolio backtest of the RSI momentum-regime system and test whether diversification makes it beat the Nifty index.
- **What we're testing:** Long-only slot portfolio. Per name: enter at close when daily RSI(14)>=ENTRY if a slot is free; exit at close when RSI<EXIT. N equal-capital slots; if more candidates than free slots, take highest-RSI. Does any (universe,N,ENTRY,EXIT) beat NIFTYBEES buy&hold by net CAGR>=1.5x AND with lower MaxDD?

## 3. The Base
- Signal: daily Wilder RSI(14), close-basis. Entry RSI>=ENTRY, exit RSI<EXIT. Long-only, flat=cash.
- Portfolio: N slots equal capital. Daily loop over union calendar: mark-to-market (shares*close), process exits (free slot, exit cost), process entries (rank RSI desc, buy at close, allocate NAV/N, entry cost).
- Cost: 15 bps on each entry and exit notional. Idle cash 0% (6% would only help — noted).
- Universe: current Nifty50 / Nifty200 membership, restricted to names with data reaching back to <=end-2015 and >=1500 day-rows. SURVIVORSHIP-BIASED (current membership on the past); stated loudly.
- Period: full available window (max span of qualified names), bench clipped to identical window per cell.
- Success: net CAGR>=1.5x NIFTYBEES CAGR AND net MaxDD<NIFTYBEES MaxDD; rank by net Calmar.

## 4. Plan
- Universes {nifty50, nifty200} x N{5,10,15,20} x (ENTRY,EXIT){(70,40),(60,40),(70,50),(65,45),(60,30)} = 40 cells.
- rsi_len=14 fixed. Preload prices once per universe.

## 5. Status
- See chat + results/phaseC_portfolio.csv (incremental, append+flush per cell).

## 6. Crash Recovery
- `cd /home/arun/quantifyd`; check `wc -l research/72_rsi_regime_7040/results/phaseC_portfolio.csv` (41 lines incl header when done).
- Resume: `venv/bin/python research/72_rsi_regime_7040/scripts/run_phaseC_portfolio.py` — it skips labels already in the CSV.
- Log: `/tmp/phaseC.log`.

## 7. Files
| File | Purpose | Commit? |
|---|---|---|
| scripts/portfolio_engine.py | slot portfolio engine | yes |
| scripts/run_phaseC_portfolio.py | sweep runner | yes |
| results/phaseC_portfolio.csv | per-cell results | yes |
| results/RESULTS_phaseC_portfolio.md | final verdict | yes |

## 8. Findings
- TBD (written on completion).
