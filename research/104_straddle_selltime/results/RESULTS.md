# research/104 — ATM Straddle Sell-Timing (75 days, REAL 1-min chain) — RESULTS

**Verdict: the 08-05/06 hunch ("premiums don't decay, sell later") does NOT generalize. Over 75 days
the driver is the DTE / day-of-week, NOT the intraday entry time. Keep the 09:16 entry. The
"premium-peaked" signal is REFUTED. The real risk is SENSEX-Wednesday (DTE1) tail; the sweet spot is
NIFTY-Monday (DTE1).**

> Real data (option_chain 1-min LTP), 75 days SENSEX+NIFTY (2026-04-20→08-06). Trade = sell running-
> ATM straddle at T, square (buy back same strike) at 15:25 (MIS same-day, matches NAS), **no stop** —
> so these are the raw hold-to-EOD distributions the stop then modifies. Net of ₹160 + 0.5pt/leg.
> **Caveat: n=15 per DTE0/DTE1 cell (only 15 expiry days in the window) → modest confidence on those;
> DTE2+ n≈45. One regime (Apr–Aug 2026).**

## Headline table (mean net ₹/lot · win% · p05 tail)

| | SENSEX DTE0 (Thu) | SENSEX DTE1 (Wed) | NIFTY DTE0 (Tue) | NIFTY DTE1 (Mon) |
|---|---|---|---|---|
| **09:20 (early)** | **+2,575 · 93% · −139** | −1,340 · 53% · **−17,761** | −143 · 73% · −15,650 | +494 · 62% · −3,634 |
| 11:45 (midday) | +549 · 40% | −1,666 · 53% · −17,356 | −2,299 · 33% · −14,310 | +441 · 62% · −856 |
| 13:15 (late) | +935 · 73% | −1,389 · 67% · −16,096 | +717 · 60% · −6,059 | **+737 · 75% · −904** |
| SIGNAL (peaked) | +1,314 · 67% | −1,708 · 47% · −18,957 | −1,151 · 47% · −16,280 | +284 · 67% · −2,360 |

## Findings

1. **09:16 is NOT the worst sell-time — on SENSEX expiry day it's the BEST.** SENSEX DTE0 09:20
   entry, held to expiry = **+2,575/lot, 93% win, p05 only −139** (the cleanest cell in the study).
   Selling at open and holding captures the full expiry-day decay; the midday entries are *worse*
   (+291–623). **08-06 was an unrepresentative bad-morning draw** — held, it still ended a winner;
   the systems lost because the **stop fired during the rare morning spike**, not because of the
   entry time (this is the research/103 tension, live).

2. **The driver is the DTE / day-of-week, not the clock:**
   - **SENSEX Wednesday (DTE1) is the dangerous day** — mean-negative at *every* entry, catastrophic
     left tail (**p05 ≈ −₹17,000/lot**), median ≈ 0. "Usually a tiny win, occasionally a disaster."
     08-05's −₹11.5k live loss was exactly this. **No entry time fixes it — the portfolio stop is
     what earns its keep here.**
   - **NIFTY Monday (DTE1) is the sweet spot** — positive mean, clean tails, best late (13:15:
     **+737/lot, 75% win, p05 −904**).
   - **SENSEX Thursday (DTE0)** — good if *held* (early entry 93% win), but stop-sensitive.
   - **NIFTY Tuesday (DTE0)** — tail-heavy early; later entry (13:15+) is safer.

3. **The "sell when premium rolled over" SIGNAL is refuted** — it catches a falling knife (enters a
   pullback that keeps going), worse than the best clock time with fat tails (NIFTY DTE0 −1,151,
   p05 −16,280). Do not build it.

## What it means for the live book (Monday)

- **Do NOT change the entry time.** The data does not support "sell later" — on the expiry days
  early-and-hold is fine/best. My 2-day working hypothesis was wrong; keep 09:16.
- **The real lever is DTE/day risk, mediated by the stop:**
  - **SENSEX Wednesday (DTE1)** is the fat-tail day → this is where the venue portfolio stop is
    essential and where sizing-down / a tighter DTE0-style stop is most justified (ties to
    research/103: stops are the DTE lever, not the entry clock).
  - **NIFTY Monday (DTE1)** is the best day — lean in, it barely needs the stop.
- **The stop is the crux, not the clock.** With no stop, SENSEX DTE1 mean is −₹1,340/lot (tail
  −17k); the stop's whole job is to truncate that tail. On SENSEX DTE0 the stop *costs* you (held
  wins 93%). So the open question is **stop calibration by DTE** — not entry timing. That's the
  thread worth pulling next (research/103 + a real-fill version once we record BFO option intraday).

**Gate:** G2 on real data. Corrects the research/104 premise. Does not clear a live change on its own
(n=15/DTE cell, one regime) — but it firmly says: **don't move the entry time; focus stop/sizing on
SENSEX-Wednesday.** Seven sins: look-ahead none (causal); cost netted; overfitting guarded (whole
window, no param picked, signal tested & refuted); small-sample + single-regime stated loudly.
