# RESULTS — Convex & Defined-Risk Structures (NIFTY, G1+G2)

**VERDICT: NO EDGE vs NIFTY B&H.** No option structure — sold or bought, timed or adjusted, weekly or monthly — beat NIFTY buy-&-hold on return over 2015–2026 (B&H +₹38L / 190% on ₹20L). One SIGNAL-grade satellite survived.

## G1 — naked structures (hold-to-expiry)

All long-convexity families (put/call backspread, Batman, broken-wing fly, iron fly) LOSE naked — pure theta bleed. Credit families (front ratio, short strangle) earn but with account-killing tails (−₹18L to −₹40L on ₹20L). Front-ratio +₹155–172L total is a mirage — the −₹35–40L drawdown wipes the account. Nothing naked beats B&H risk-adjusted.

## G2 Thread A — signal-timed convexity

Timing REMOVES the bleed but does not create B&H-beating alpha:

- CALL_BACKSPREAD + ATR-contraction: +₹7.25L, 42% win, −₹0.90L DD — the one genuine positive convex overlay (a SIGNAL/satellite, far below B&H's +₹38L).
- PUT_BACKSPREAD / BATMAN stay net-negative even timed (moves too rare to pay for convexity bought every cycle).
- LONG_STRANGLE at VIX<14 = breakeven → index convexity is ~fairly priced.
- Best gate = ATR-contraction (buy convexity when vol compresses); CPR<0.10% second.

## G2 Thread B — credit + tail adjustments

EOD adjustments CANNOT rescue the credit tail (it is a fast crash):

- FRONT_PUT_RATIO: convert-to-fly makes it WORSE (+₹172L to −₹10.7L, DD −₹35L to −₹67L); roll-out worsens DD (−₹41L); only stop-3× helps modestly (+₹72L, DD ~unchanged).
- SHORT_STRANGLE: only stop-3× helps (return down for DD down), as previously found.
- Lesson: short-vol books must be SIZED for the tail, never "adjusted" out of it.

## G3 — directional signal expressed through levered/convex option structures

Gate a NIFTY trend signal (200DMA / 50>200), express the long via long calls / call-debit / call-backspread, sweep size, vs B&H (+₹38L, 16.5% CAGR). Result: still NO robust beat.

- The only configs that "beat" B&H (OTM call / call-debit, 3 lots, NO gate) beat it by a hair (+₹40L, 17.4%) — pure leveraged beta in a bull decade; their −₹10.7L drawdown is DEEPER than B&H's own 2020 hit → worse risk-adjusted. Not an edge.
- The trend gate HURT returns (200dma < always everywhere) — monthly gating sits out the rebound months. The momentum-book's index-EMA gate does NOT transfer to a monthly index-option gate.
- Leverage bleeds: CAGR falls as lots rise (3→6→10), DD balloons — theta scales with size.
- Best risk-adjusted: CALL_BACKSPREAD 3-lot (50>200) — ~15% CAGR, −₹0.66L DD, Calmar 0.45 — a capital-efficient low-DD equity substitute, NOT alpha (small capital at risk).

## Takeaway / next

Return comes from DIRECTIONAL exposure to the underlying, not option geometry — confirmed by the efficient pricing here vs the momentum-30 book (net ~31.8% CAGR, beats B&H). Survivors as satellites: (1) ATR-contraction-timed call backspread (cheap uncorrelated upside overlay); (2) the irreducibility of the credit tail. SENSEX/BANKEX (2024–26 only) too short to add. BankNifty G3 replication only worth it if a core edge had survived — none did.
