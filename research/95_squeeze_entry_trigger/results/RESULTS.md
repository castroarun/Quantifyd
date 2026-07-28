# RESULTS — Squeeze-family entry-trigger bake-off

**Verdict: SIGNAL / actionable — the ATR SQUEEZE trigger is SUB-OPTIMAL. A plain EARLY time entry (09:16/09:30) beats it; late (10:00+) and price-move (+/-100) entries LOSE.**

Avg P&L/trade (68d, short ATM straddle held 15:15, 2 lots, net):
- time 09:30 +633 (total +43049) BEST
- 09:16 +576 (+39204)
- ATR squeeze (current) +407 (+21577, fires 53/68 days) -- MEDIOCRE
- price +/-50 +373 (+23514)
- time 10:00 -137 ; 10:30 -443 ; price +/-100 -460 ; 11:00 -999 ; 12:00 -1010

## Read
1. Enter EARLY -- the straddle decays most in the morning; every delay costs. 09:16/09:30 win, >=10:00 loses (11-12:00 ~ -1000/tr).
2. The squeeze WAIT delays entry and gives up the morning decay -> loses to just entering at 09:16/09:30 (~half the total), and skips 15 no-squeeze days.
3. Price entries dont help: +/-50 mediocre, +/-100 loses (enters after a big directional push).
4. Robust in the recent-window split (early entries stay top).

## Recommendation
Drop the squeeze wait on the squeeze family (nas_atm/atm2/atm4, paper) -> enter 09:16/09:30. The squeeze trigger does not earn its keep. Sign-off needed before any change.

Caveats: isolates ENTRY (holds naked to 15:15, no mgmt); -8k portfolio stop caps the -24k/-32k tail days; 68d, optimistic fills (LTP, no slippage).
