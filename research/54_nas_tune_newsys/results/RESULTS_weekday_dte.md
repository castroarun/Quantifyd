# research/54 Stage 2 — Weekday × DTE P&L map (real NIFTY chain, ±0.4% move stop)

Derived from `stage1_iv_perday.csv` (short ATM straddle, enter 09:20, ±0.4% underlying-move
stop, exit 14:45, net ₹80/leg). **23-29 days => SIGNAL.** Direct input to the live/paper
day-selection decision (user directive 2026-06-03: live Mon/Tue/Fri, paper all other days).

## Weekday → DTE → P&L

| Weekday | DTE | n | total ₹ | ₹/day | Read |
|---|---|---|---|---|---|
| Mon | 1 | 7 | +15,988 | **+2,284** | the edge |
| Tue | 0 | 6 | +2,368 | +395 | modestly positive |
| Fri | 4 | 5 | −352 | −70 | ~flat / breakeven |
| Wed | 6 | 6 | −4,386 | −731 | bleeds |
| Thu | 5 | 5 | −8,717 | −1,743 | worst bleeder |

(Tue/0-DTE P&L from the full stage-1 set; those days have null ATM-IV so they drop out of the
IV-tercile subset but are valid for the straddle replay.)

## Read
- The edge is **Monday (1-DTE)**; Tuesday (0-DTE) modestly positive; **Friday (4-DTE) ~flat**.
- The real bleeders are **Wed (6-DTE) and Thu (5-DTE)** — both 5-6 DTE, consistent with finding #1.
- **The user's live schedule (Mon/Tue/Fri) is data-consistent**: trades the edge + positive +
  breakeven days, excludes the two worst (Wed/Thu) which now run paper. Earlier "Friday bleeds"
  call was WRONG — Friday is breakeven, not a bleeder.

## Caveat
23-29 days = SIGNAL, single regime (Apr-Jun 2026). Each weekday cell has 5-7 obs. The DTE
gradient (1-DTE best, 5-6 DTE worst) matches research/51/52 on independent data → trustworthy
direction; absolute ₹/day will move as the recorder accumulates.
