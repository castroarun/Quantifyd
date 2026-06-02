# NAS Systems — True Replay Backtest on Recorded NIFTY Weekly Option Chain

STATUS: RUNNING

## 1. The Ask
"You have options data for the past 28 days of Nifty weekly contracts. Backtest our
NAS systems on this data and give me the report." → **Replay each system's RULES**
against the recorded `option_chain` (real per-minute premiums), generating the trades
the logic *would* make, and report simulated P&L. (Not phase-1 actual trades, not
phase-2 re-pricing of actual fills — a genuine strategy replay.)

## 2. Base — data
- Underlying: per-minute spot from `option_chain.underlying_spot` (market_data.db has
  no NIFTY 5-min for this window). Reconstruct 5-min candles for ATR/ST.
- Premiums: real recorded `option_chain` ltp per strike per minute. NIFTY only, 28
  days (2026-04-20→06-02), weeklies.
- Lots fixed at 2 (130 qty) for ALL systems → normalized, comparable (avoids the live
  10→2 distortion).

## 3. Systems (parameterized engine)
| System | Entry | Strike | Management |
|---|---|---|---|
| Squeeze OTM / 916 OTM | ATR squeeze / 09:16 | OTM ~Rs20 | cross-leg roll on >=2x imbalance |
| Squeeze ATM / 916 ATM | squeeze / 09:16 | ATM | per-leg SL 1.3x; naked leg ST(7,2) |
| ATM2 / 916 ATM2 | squeeze / 09:16 | ATM | any SL closes both, re-enter (max 5) |
| ATM4 / 916 ATM4 | squeeze / 09:16 | ATM | 1st SL roll-to-match; 2nd SL naked ST(7,2) |
| Exit (all) | time 14:45 / EOD 15:15 | | priced from chain |

## 4. Caveats
- 28 days, one regime → SIGNAL/audit, not validation.
- 1-min snapshots → SL/ST precision ~1 min (intra-minute spikes missed).
- Squeeze-entry reconstruction is approximate (live uses its own NIFTY feed); 9:16
  entry is exact. **Replay validated against actual recorded trades** for faithfulness.
- Prices at recorded LTP (no bid/ask slippage) → mildly optimistic.

## 5. Status
| Time | Event |
|---|---|
| 2026-06-02 ~15:55 IST | folder + STATUS; building engine |


## 6. VERDICT (DONE)
True replay, all 8 systems, lots=2, 28 days, priced from recorded chain. **Combined net Rs-54,123 (NEGATIVE).** Only Squeeze ATM marginally + (Rs1.7k); 916 ATM2 cascade worst (-23.9k); 9:16 family worse than squeeze-entry; day-win 32-44%. Agrees with research/50 (actuals ~flat): NO EDGE in this regime. Fixed a roll-churn bug (916 OTM 1252->157 legs) before finalizing. 28 days + squeeze-entry/1-min approximations = SIGNAL not validation. STATUS: DONE.