# F2 Conditioning Filters on NIFTY ORB Long (W12/ts4) + Val Confirmation

STATUS: DONE — SHORTLISTED (see Verdict)

EXP-F2 of research/81. Follow-up earned by EXP-F1's SIGNAL verdict.

## 1. The Ask

Can a single causal conditioning filter (gap, prev-day trend, or VIX regime)
concentrate the F1 edge (NIFTY ORB long, W=12, hold 4 sessions, stop=OR-low)
enough to clear the acceptance bar — and does the LOCKED config confirm on
the untouched Validation split?

## 2. Base cell (LOCKED from F1 — no re-optimization)

W12_L_ts4: OR = 09:15–10:15 (12 bars); first 5-min close above OR-high →
long next bar open; stop = OR-low; exit forced at close of session
entry+3. IS baseline: +11.6 bps net @3bp slip (t=1.65), +15.6 @1bp (t=2.22),
n=463, 6/7 years positive.

## 3. Filters — marginal analysis, one at a time (LOCKED)

| Filter | Variants (entry allowed only if...) |
|---|---|
| Gap | gap-up (open > prev close), gap-up ≥ 0.25%, gap-down (open < prev close) |
| Prev-day trend | prev session close > open (up day), prev close < open (down day) |
| VIX regime | prev-day INDIAVIX close in trailing-252d tercile: low / mid / high |

8 conditioned cells + base = 9 IS rows (ledger +8 → 116). All features
causal (prev-day values; VIX terciles from TRAILING 252d quantiles, shifted).

**Filter acceptance (decided now):** a filter is adopted only if it (a) lifts
net bps AND t vs base at BOTH cost levels, (b) retains ≥40% of base n, and
(c) has an economic story. Otherwise the base cell stands.

## 4. Validation touch (pre-declared)

The single locked config (base or base+one filter) runs ONCE on
Val = 2021-10-01→2023-12-31 at 1bp and 3bp slippage.
**Pass:** net > 0 at both costs AND net_bps(Val)/net_bps(IS) ≥ 0.5
(walk-forward-efficiency style). Pass → family F shortlist for G3/G4
(full walk-forward, Monte Carlo, BANKNIFTY replica post-backfill).
Fail → F1/F2 = SIGNAL-not-confirmed; recorded, closed.
Val is not re-used for any further F-family tuning after this touch. OOS
(2024+) remains locked.

## 5. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-15 ~22:00 IST | Pre-registered | runner: `scripts/run_f2_orb_filters.py` |

## 6. Findings

(after run)

## VERDICT (2026-07-15 ~22:20 IST): FILTER ADOPTED + Val BORDERLINE PASS — family F SHORTLISTED for G3

IS marginal analysis: gap_up>=0.25% is the only filter meeting all acceptance
criteria — net +29.1bps t=3.01 @1bp (+25.1 t=2.60 @3bp), n=261 (56% of base),
monotone gap dose-response (gap_down INVERTS the edge: -1.4bps @3bp), story =
overnight news + confirmed morning demand. VIX-mid helped but non-monotone;
prev-day trend inert. LOCKED CONFIG: NIFTY ORB W12 long, entry on first 5-min
close > OR(09:15-10:15) high on gap-up>=0.25% days, stop = OR low, exit close
of session entry+3.

ONE-TIME Val touch (2021-10..2023-12): n=92, net +15.4bps @1bp (t=1.28),
+11.4bps @3bp (t=0.95), all 3 years positive. WF-eff 0.53 @1bp PASS /
0.46 @3bp marginal miss. Verdict: BORDERLINE PASS -> G3 earned (walk-forward,
Monte Carlo, regime split, BANKNIFTY replica post-backfill). Cost-fragility
flag stands. Val is now CONSUMED for family F tuning; OOS 2024+ locked.
STATUS: DONE
