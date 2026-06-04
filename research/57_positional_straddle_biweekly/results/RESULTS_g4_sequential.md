# research/57 G4 — REALISTIC sequential book (one straddle, re-enter at CMP, 30-day chain)

Hold one ATM bi-weekly straddle; EOD 1.5%+PT40 stop + 5-min-polled crash stop; re-enter at CMP. **True running equity + book max-DD. 30d SIGNAL.**

| config | closes | final P&L | book max-DD |
|---|---|---|---|
| crash3% reenter-CMP | 6 | +74238 | -3683 |
| crash3% reenter-next0920 | 6 | +74238 | -3683 |
| crash2.5% reenter-CMP | 6 | +74238 | -3683 |
| NO crash stop reenter-CMP | 6 | +74238 | -3683 |

## Read
- This is the ACTUAL deployed book (one straddle, sequential), not overlapping per-trade stats.
- book max-DD = the real running drawdown to size against (NOT the per-trade worst).
- crash stop level + re-entry timing compared. 30d SIGNAL, single regime.