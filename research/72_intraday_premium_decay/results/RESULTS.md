# research/72 — Intraday ATM Premium Decay-then-Rise — RESULTS

**VERDICT: CONFIRMED (pass 1).** The pattern Arun observed is real and repeatable across 53 days.
It is overwhelmingly a **near-expiry theta collapse**, with a **repeatable IV spike at 15:15**
that briefly props the premium back up. Direction-neutral (vega, not delta). DTE is the master
variable. Robustness caveat: 53 days, single calm short-vol regime; `underlying_spot` had garbage
rows (filtered to 20k–28k); 15:15/15:30 minutes have fewer clean days (n≈15–23).

## 1. The decay — a near-expiry theta story, steepest ~14:50–15:10

ATM straddle premium indexed to its 14:00 value (=100), re-picked ATM each snapshot (so it is
direction-neutral — a fall is pure time/vol, not the underlying drifting):

| Minute | ALL days | **DTE 0-1 (expiry)** | DTE 2+ (far) |
|---|---|---|---|
| 14:00 | 100 | 100 (₹133) | 100 (₹288) |
| 14:30 | 91 | 87 | 99 |
| 15:00 | 80 | 72 | 99 |
| 15:10 | 80 | 70 | 98 |
| **trough** | **72.8 @15:16** | **60.7 @15:14** | 95.5 @15:30 |
| 15:30 | 78 | 70 | 96 |

- On **expiry days the ATM straddle loses ~40%** of its 14:00 value by ~15:15. On **far-DTE days it
  barely moves (~4%)**. So the "sudden 3 PM decay" only happens near expiry — it's theta accelerating
  as time-to-expiry → 0.
- Not a cliff at exactly 3 PM — a smooth acceleration, fastest **~14:50–15:10**.

## 2. The "rise at 15:15" — a repeatable IV spike (vega, not direction)

The ATM straddle is direction-neutral, yet it bounces at 15:15 — that can only be **IV rising**.
ATM IV (%) confirms a sharp, precise spike at 15:15, biggest on expiry days:

| Minute | ALL IV% | DTE 0-1 IV% |
|---|---|---|
| 15:05 | 16.58 | 18.84 |
| 15:10 | 16.60 | 18.67 |
| **15:15** | **17.08 ↑** | **20.16 ↑↑ (+1.5)** |
| 15:20 | 16.68 | 19.08 |
| 15:25 | 16.37 | 18.51 |

- IV pops **+1 to +1.5 vol points at 15:15 sharp**, then fades within ~10 min.
- **Symmetric** — CE and PE indices move together (15:15: CE 79.6 / PE 77.4), and the underlying's
  median move *grows* into the close (10→47 pts). So it is a **volatility burst, not a directional
  move** — consistent with the **final-15-min square-off / 15:00–15:30 VWAP-close positioning**
  (MIS auto-square-offs + closing-price jockeying spike realized vol → IV ticks up).

## 3. Master variable = DTE; concurrency = IV + rising activity, NOT direction

- **DTE decides everything:** expiry-day = violent collapse; far-DTE = flat. Any "premium decay"
  intuition must be conditioned on DTE.
- The 15:15 rise concurs with **IV up + realized-movement up**, symmetric across CE/PE.
- Day-type / CPR concurrency (range vs trend) = **pass 2** (not yet tested).

## 4. Read-through for the NAS short-straddle systems

- The **expiry-day 14:00→15:15 theta collapse IS the short-vol edge** — holding a short ATM straddle
  through the final hour on expiry captures ~40% premium decay.
- Premium **troughs ~15:15–15:20 then ticks back up into 15:30** (expiry 60.7→70). The systems'
  **15:15 EOD square-off sits right at the trough** — near-optimal for a buyback; holding to 15:30
  would give a little back. The 15:15 IV pop is a brief adverse tick but theta dominates.
- **Pass 2 candidates:** optimal exit-minute sweep (15:10 vs 15:15 vs 15:20 vs 15:25) for a short
  straddle; CPR-width / range-vs-trend concurrency; is the 15:15 IV pop tradeable (buy vol 15:10,
  sell 15:16)?

---

# PASS 2 — exit-minute economics, day-type concurrency, is the pop tradeable

## (a) Exit-minute sweep — short ATM straddle entered 09:16, buyback at minute M

P&L = entry − exit premium (₹/lot 65), DTE 0-1 (expiry), n≈16–19 days:

| Exit | mean P&L | median | p5 (worst) |
|---|---|---|---|
| **15:00** | **+1085** | 1963 | **−4,630** |
| 15:05 | +121 | 1639 | −9,078 |
| 15:10 | +175 | 1940 | −8,695 |
| **15:15** | **−275** | 1601 | **−10,451** |
| 15:20 | +237 | 1852 | −9,428 |
| 15:30 | +342 | 1888 | −9,609 |

- **Theta is essentially done by 15:00** (median P&L already ~₹1,900). Holding past 15:00 adds **no
  median return** but the **tail risk roughly doubles** (p5 −4.6k → −9 to −10.5k) — that damage
  materialises right after 15:00.
- **15:15 is the single WORST exit** (mean −275, worst p5) — the IV pop makes the buyback momentarily
  expensive *and* you eat late-day gamma. Far-DTE (2+) shows none of this (15:15 fine, +971 / p5 −2.4k).
- **Actionable (small n, validate forward): move the expiry-day EOD square-off earlier (~15:00).** Same
  typical P&L, ~half the tail. The systems currently exit 15:15 = the worst minute on expiry.

## (b) Day-type / CPR concurrency — it's a TREND-vs-RANGE story

Re-picked ATM premium (14:00=100) trough + the 15:15 IV pop, by day-type:

| Day type | n | premium trough | 15:15 IV pop |
|---|---|---|---|
| **Trend day** (|C−O|/range > .55) | 14 | **63 @15:15 (deep)** | +0.44 |
| **Range day** (< .35) | 8 | **84 @15:30 (shallow)** | +0.09 (~none) |
| CPR-wide | 15 | 70 | +0.56 |
| CPR-narrow | 14 | 77 | +0.35 |

- **The decay depth is driven by whether the day trends.** Trend days gut the straddle (→63%); range
  days barely decay (→84%). So the "3 PM cliff" is really a *trend-day + expiry* phenomenon.
- **The 15:15 IV pop concurs with movement** — biggest on wide-CPR (+0.56) and trend days (+0.44),
  ~absent on quiet range days (+0.09). Consistent with a square-off / closing-vol burst that only
  fires when there's directional activity to unwind. (CPR width is a weak, small-n signal here; the
  realised trend/range split is the clean one.)

## (c) Is the 15:15 pop tradeable? — NO

| Trade | mean ₹/lot | win% |
|---|---|---|
| Long straddle buy 15:10 / sell 15:15 | −48 (ALL), −12 (exp), −97 (far) | 25–45% |
| Tight 1-min: buy 15:14 / sell 15:15 | +75 (exp) | 64% (exp) |

- As a **5-minute long, theta beats the IV pop** — it loses across the board. The pop only survives as
  a **1-minute blip** (buy 15:14 / sell 15:15 ≈ +₹75 on expiry, 64% win) — too small/fast to trade
  after slippage. **Leave it.**

## Pass-2 verdict + next
The observation is fully explained: **near-expiry theta, deepened on trend days, with a movement-driven
15:15 IV pop.** The one concrete lever for the book is **(a): an earlier expiry-day exit (~15:00)** —
worth a proper exit-time sweep on the live short-straddle engine (14:45/14:50/15:00/15:10/15:15,
risk-adjusted), which is the natural G-gate follow-up.

## Files
| File | Purpose |
|---|---|
| `scripts/pass1_atm_path.py` | Pulls per-day per-minute ATM straddle premium + IV (17M-row query, ~3 min) |
| `scripts/pass1_agg.py` | Normalises + aggregates the decay/rise report (reads the CSV) |
| `results/atm_minute_paths.csv` | Per-day per-minute ATM prem/iv/spot (committable) |
| `results/allday_minute_agg.csv` | Averaged minute path (committable) |
