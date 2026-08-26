# research/128 — SENSEX naked-survivor trailing stop: build the NIFTY-equivalent and calibrate it

## Verdict

> **SIGNAL — DEPLOY AS A CORRECTNESS FIX, NOT AS AN ALPHA CLAIM.**
>
> The deployed SENSEX survivor "trail" is a defect: on **62%** of episodes the SuperTrend value
> it writes into `sl_price` lands **below** the live premium and self-triggers on the same poll,
> ~38 minutes after the leg goes naked. Over 86 replayed episodes it books **Rs2,480/lot** —
> **worse** than doing nothing but breakeven protection (Rs2,536/lot). The Rs6,981 it appears to
> have gained across the 12 real live episodes is a small-sample accident, not an edge.
>
> A trailing stop is nevertheless **required**, and the evidence supports it: holding the survivor
> to 15:15 with no stop at all is the worst arm on the board (Rs2,136/lot, worst episode
> **−Rs15,716/lot**), because **23.3% of decaying naked survivors round-trip all the way back to
> their entry premium** and 38.4% give back at least half the decay.
>
> The recommended design — **the NIFTY mechanism ported verbatim**: a ratcheting ST(7,3) *ceiling*
> that is never written into `sl_price`, seeded from the day's own premium history, confirmed over
> ~60 s, clamped at breakeven — books **Rs2,795/lot**, beats the incumbent by **+Rs315/lot
> (t = 2.31)** and the random-exit null by +Rs394/lot, and has the **best tail of any arm tested**
> (worst episode −Rs613/lot). Against breakeven-only it is **+Rs259/lot, t = 1.35 — not
> statistically distinguishable**, and out-of-sample the two are a wash (Rs2,146 vs Rs2,083).
>
> So: **the fix is measured; the parameter choice is a judgement call on thin evidence.** ST(7,3)
> is chosen because it sits on a broad, flat plateau *and* because it is exactly what the working
> NIFTY counterpart uses — which is what Arun asked for. It is not a measured optimum.

---

## 1. Data & episodes

| Item | Value |
|---|---|
| Source | `backtest_data/options_data.db :: option_chain`, SENSEX, 1-minute snapshots |
| Candidate days | 89 (2026-04-20 → 2026-08-26) |
| Rejected | `2026-04-20` thin (78 min) · `2026-05-01`, `2026-05-28`, `2026-06-26` frozen-chain holiday guard (<50 distinct spot prints) · `2026-08-26` partial session (last snapshot 12:08) |
| Usable days | **84** |
| Days that produced a survivor | 50 |
| **Episodes** | **86** (ATM 50, ATM4 36) |
| Lot | 20 (`option_chain.lot_size` is wrong — research/119) |

Episodes were synthesised by replaying the **deployed** rules: 09:16 ATM straddle, per-leg 30% SL,
`leg_sl_disabled_dtes=(0,)`, EOD 15:15; ATM's survivor goes naked on the **first** SL, ATM4 rolls the
stopped leg to a premium-matched OTM strike (`_find_roll_strike` mirrored on the recorded chain,
survivor SL tightened to `price_x × 1.3`) and its survivor goes naked on the **second** SL.

| DTE (trading) | Episodes | | Weekday | Episodes |
|---|---|---|---|---|
| 1 | 31 | | Mon | 16 |
| 2 | 28 | | Tue | 30 |
| 3 | 18 | | Wed | 29 |
| 4 | 9 | | Fri | 11 |
| **0** | **0** — no per-leg stop on expiry day, so DTE0 has no survivor route at all | | Thu | 0 |

---

## 2. Harness fidelity — reconciled against the 12 real live episodes first

The replayed **INCUMBENT** arm reproduces the actual live exit **to the minute on 9 of the 12** real
`SENSEX_ST_TRAIL` SL_HIT legs; the 3 divergences fire *later* in the replay (1-minute polls vs the
live 10-second polls produce slightly tamer 5-minute candle extremes). That error direction is
conservative: it flatters the incumbent, so the finding "incumbent ≤ breakeven-only" is a floor.

Net booked over the 12 real episodes at their traded lot sizes: **live +Rs64,727 · breakeven-only
+Rs60,079** — i.e. the accident gained ~Rs4,650 net (Rs6,981 gross, matching the pre-registered
figure in the brief). See §7 for the full episode-by-episode table.

---

## 3. THE decision statistic — how often does a decaying survivor round-trip?

Measured on all 86 episodes, from the naked moment to 15:15. `recovery_frac` = (highest premium
after the low − low) ÷ (entry − low), i.e. the fraction of the decay handed back.

| Statistic | Value |
|---|---|
| Median depth of decay | to **44%** of entry premium |
| Median time naked → the low | **152 min** |
| **Full round-trip back to entry premium** | **20 / 86 = 23.3%** |
| Gave back ≥ 75% of the decay | 27.9% |
| Gave back ≥ 50% | **38.4%** |
| Gave back ≥ 25% | 47.7% |
| Clean one-way decay (≤ 10% given back) | 32.6% |
| `recovery_frac` percentiles | p10 0.02 · p25 0.08 · **p50 0.23** · p75 0.93 · p90 2.41 · p95 2.76 |
| Median time from the low back to entry (when it round-trips) | **72 min** |
| Worst hold-to-EOD outcome | **−Rs15,716/lot** |

**Per DTE and per weekday** — the risk is concentrated exactly where SENSEX's known fat tail lives:

| Cut | n | Full round-trip | Gave back ≥50% | Mean `recovery_frac` |
|---|---|---|---|---|
| DTE1 | 31 | **29.0%** | **51.6%** | 1.155 |
| DTE2 | 28 | 25.0% | 35.7% | 0.826 |
| DTE3 | 18 | 11.1% | 22.2% | 0.495 |
| DTE4 | 9 | 22.2% | 33.3% | 0.719 |
| Mon | 16 | 12.5% | 18.8% | 0.496 |
| Tue | 30 | 23.3% | 33.3% | 0.777 |
| **Wed (DTE1)** | 29 | **31.0%** | **55.2%** | **1.229** |
| Fri | 11 | 18.2% | 36.4% | 0.677 |

**Reading:** the round-trip risk is real, persistent and heavily Wednesday/DTE1-weighted — the same
slot research/104 flagged as SENSEX's fat-tail danger. But the median episode gives back only 23% of
its decay and a third decay cleanly one-way. **That combination argues for a patient trail, not an
aggressive take-profit**: you want to stay in the two-thirds that keep decaying and be pulled out of
the quarter that reverses. The take-profit arms confirm it (§5): the best of them (TP at 50% of
entry) adds only +Rs148/lot vs breakeven at t = 0.88 — the accident's apparent profit-take edge is
noise.

---

## 4. Mechanism diagnostics — exactly what is broken and what fixes it

**(a) The self-trigger.** `services/sensex_naked_trail.py` calls the repo's TradingView band-locked
`calc_supertrend`, whose `direction` initialises to 1 and returns the **lower** band while it stays
there. Replayed over the 86 episodes:

| | |
|---|---|
| Episodes where an ST value is produced at all | 84 / 86 |
| Time from naked to the first ST value | **median 38 min** (min 36, max 40 — the 9-bar warm-up) |
| ...and the value lands **BELOW** the live premium | **52 / 84 = 62%** → fires on the same poll |

That is the live signature exactly: 9 of the 12 real episodes exited 37–40 minutes after their
sibling stopped out, at market. **The incumbent is not a trail; it is an arbitrary ~40-minute
delayed market exit** with a 38% chance of behaving itself.

**(b) The ceiling design is structurally correct.** `compute_short_trailing_stop` (the NIFTY
function) takes `min(upper_band, previous_stop)` — it can only ratchet **down** and by construction
sits above the last completed bar's close:

| Design | Armed on | Stop is ABOVE the premium when it arms | Minutes with no trail |
|---|---|---|---|
| Ceiling, **seeded** from the day's own history | 86 / 86 | **100%** | median **0**, max 37 |
| Ceiling, **cold** start (today's SENSEX behaviour) | 83 / 86 | 100% | median **39**, max 41 |

Seeding is what the NIFTY path already does (`_seed_naked_candles`); the SENSEX port omitted it.
It is **return-neutral** (grid mean 2,713 seeded vs 2,721 cold) but it removes a ~40-minute window
in which the survivor carries no trail at all — the same window in which the 23% round-trip risk
bites.

**(c) The breakeven clamp is free.** Running the ceiling with and without the `min(ST, entry)` clamp
gives **byte-identical results on all 86 episodes**: once armed, the ratcheted ceiling is already
below entry in every single episode. The clamp costs nothing and guarantees the survivor can never
be turned into a loser by a wide ATR. Keep it — as the warm-up fallback and as a floor.

---

## 5. Arm comparison — net of the MEASURED cost model

Costs: research/122's outcome-aware model reduced to one leg — Zerodha F&O option rate card
(brokerage Rs20/order over 2 orders, STT 0.1% on the sell premium, exchange txn 0.03503%, IPFT, SEBI,
stamp 0.003% on the buy, GST 18%) plus the measured slippage from 443 real live leg-sides: entry 0,
**triggered/stop exit +6.548 pt**, time/EOD exit +0.178 pt. The retired flat Rs250/lot is **not** used.

All figures **Rs per lot, net, n = 86**. SENSEX lot = 20; the live book runs **2 lots**, so multiply
by 2 for book rupees.

| Arm | Mean net | Median | Worst episode | Win % | Fire % | Mean hold | vs BE_ONLY (t) | vs INCUMBENT (t) | vs random-exit |
|---|---|---|---|---|---|---|---|---|---|
| `HOLD_EOD` (no stop) | 2,136 | 3,010 | **−15,716** | 82.6 | 0 | 241 min | −400 (−1.40) | −345 (−0.81) | −266 |
| `INCUMBENT` (today) | 2,480 | 2,369 | −866 | 93.0 | **90.7** | **69 min** | −56 (−0.24) | — | +79 |
| `BE_ONLY` | 2,536 | 2,710 | −1,002 | 69.8 | 30.2 | 194 min | — | +56 (0.24) | +135 |
| `TP_50%` (take profit at half) | 2,684 | 2,946 | −1,002 | 77.9 | 80.2 | 113 min | +148 (0.88) | +204 (1.13) | +283 |
| `GIVEBACK_40%` (from the low) | 2,772 | 3,010 | −488 | 88.4 | 51.2 | 160 min | +236 (1.43) | +292 (1.65) | +371 |
| **`CEIL_p7_m3.0_N1_SEED`** ← **recommended** | **2,795** | 2,628 | **−613** | **93.0** | 72.1 | 132 min | **+259 (1.35)** | **+315 (2.31)** | **+394** |
| `CEIL_p10_m2.0_N3_COLD` (grid peak — *declined*) | 3,009 | 3,087 | −1,732 | 88.4 | 62.8 | 159 min | +473 (2.80) | +529 (2.95) | +608 |
| random-exit null (400 draws/episode) | 2,401 | — | — | — | — | — | −135 | −79 | — |

Two things matter more than the ranking:

1. **`HOLD_EOD` is decisively the worst arm and carries a −Rs15,716/lot tail.** A stop is not
   optional. This is the evidence for Arun's requirement, not just a preference.
2. **`INCUMBENT` is worse than doing nothing at all beyond breakeven protection.** Its very high fire
   rate (90.7%) and very short hold (69 min) show what it actually is.

---

## 6. Robustness

### 6.1 Plateau map — mean net Rs/lot by ST period × multiplier (averaged over confirm-count and seeding)

| period ↓ / mult → | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | row avg |
|---|---|---|---|---|---|---|
| **5** | 2,647 | 2,659 | 2,663 | 2,730 | 2,697 | 2,679 |
| **7** | 2,760 | 2,672 | **2,698** | 2,709 | 2,734 | 2,715 |
| **10** | 2,754 | 2,699 | 2,698 | 2,697 | 2,762 | 2,722 |
| **14** | 2,732 | 2,752 | 2,726 | 2,798 | 2,757 | 2,753 |
| **col avg** | 2,723 | 2,695 | 2,696 | 2,734 | 2,737 | |

*(BE_ONLY 2,536 · INCUMBENT 2,480 · HOLD_EOD 2,136 for reference.)*

Every one of the 20 period×multiplier combinations beats all three nulls, and the spread across the
whole map is 151 Rs/lot — **a genuine plateau, not a peak**. The recommended `p7 / m3.0` cell sits
on it (2,698 vs a 2,715 grid mean). There is a mild tilt toward longer periods and wider
multipliers, well inside noise.

### 6.2 Confirmation-count sensitivity (a clean dose-response, then a cliff)

| N (1-minute polls above the ceiling) | Mean net | vs BE | Cells beating BE | Worst episode |
|---|---|---|---|---|
| 1 | 2,725 | +189 | 35/40 | **−866** |
| 2 | 2,786 | +250 | **40/40** | −2,414 |
| 3 | **2,798** | **+262** | **40/40** | −4,046 |
| 5 | 2,560 | +24 | 19/40 | −4,046 |

More confirmation helps up to ~3 minutes and then collapses — waiting 5 polls means riding the
reversal. At the recommended `p7/m3.0` specifically, N = 1 is best on *both* return (2,795 vs 2,716
at N=2) and tail (−613 vs −2,414), which is why the recommendation lands at the short end.

### 6.3 Out-of-sample (chronological 60/40 split at 2026-07-07; 54 IS / 32 OOS episodes)

| Arm | IS mean | OOS mean |
|---|---|---|
| IS-best cell (`CEIL_p14_m2.0_N3_SEED`) | 3,479 | 2,168 |
| Recommended `CEIL_p7_m3.0_N1_SEED` | 3,180 | **2,146** |
| `BE_ONLY` | 2,805 | **2,083** |
| `INCUMBENT` | 2,727 | 2,064 |
| `HOLD_EOD` | 2,478 | 1,558 |
| median of ALL 160 ST cells | — | 2,079 |

**Honest reading: out-of-sample the ST ceiling is a wash against breakeven-only** (+Rs63/lot;
50% of cells beat BE OOS). What survives OOS is the *ordering against the incumbent and against
holding*, and the tail improvement. The P&L uplift over BE_ONLY does not survive OOS and must not
be claimed.

### 6.4 Multiple-testing haircut

188 arms were run, 160 of them ST-ceiling cells. 134/160 (84%) beat `BE_ONLY`; 152/160 beat
`INCUMBENT`. The best single cell reaches t = 2.80 vs BE; **Bonferroni over 188 arms needs
|t| ≥ ~3.55 (α = 5%, df = 85), so no single cell survives a family-wise correction.** The
equal-weight family-average arm gives +Rs181/lot vs BE at t = 1.55 — also not significant. (A sign
test over the 160 cells returns p ≈ 5e-19, but the cells are *highly correlated*, so that p-value is
not a valid significance test; it only evidences that the plateau is broad and consistent.)
The recommendation is therefore made on **plateau breadth + tail + mechanism**, not on a t-stat.

### 6.5 Cost sensitivity

| Stop-slippage | Recommended | BE_ONLY | INCUMBENT | HOLD_EOD |
|---|---|---|---|---|
| ×1.0 (measured, 6.548 pt/side) | 2,795 | 2,536 | 2,480 | 2,136 |
| ×1.5 | 2,748 | 2,516 | 2,421 | 2,136 |
| ×2.0 | 2,700 | 2,496 | 2,361 | 2,136 |
| ×3.0 | 2,606 | 2,457 | 2,243 | 2,136 |

The ordering is stable at every cost level; the recommended arm's lead over BE_ONLY survives until
stop slippage reaches ~37.5 pt/side, ~5.7× the measured value.

### 6.6 Where the recommendation earns its keep

| Cut | n | Recommended | BE_ONLY | INCUMBENT | HOLD_EOD |
|---|---|---|---|---|---|
| DTE1 | 31 | **3,011** | 2,323 | 2,527 | 1,435 |
| DTE2 | 28 | 2,310 | 2,059 | 1,980 | 1,928 |
| DTE3 | 18 | 2,977 | 3,184 | 2,904 | 3,221 |
| DTE4 | 9 | 3,197 | 3,456 | 3,030 | 3,022 |
| **Wed** | 29 | **2,903** | 2,167 | 2,450 | **1,219** |
| Tue | 30 | 2,461 | 2,227 | 2,090 | 2,105 |
| Mon | 16 | 2,993 | 3,324 | 2,952 | 3,365 |
| Fri | 11 | 3,133 | 3,204 | 2,938 | 2,848 |
| ATM | 50 | 2,458 | 2,367 | 2,104 | 2,210 |
| ATM4 | 36 | 3,262 | 2,770 | 3,002 | 2,033 |

The entire benefit is on **DTE1 / Wednesday** — the round-trip days. On calm far-DTE days the trail
costs a little (it exits early on days that would have decayed on). That is the correct shape for a
risk control and it matches the round-trip distribution in §3.

---

## 7. Reconciliation — what the recommended design would have done on the 12 real live episodes

`REC` = `CEIL_p7_m3.0_N1_SEED`. Rs/lot net. Replay uses each leg's real recorded 1-minute chain path.

| System | Day | Leg | Lots | Entry | Naked | LIVE exit | LIVE | BE-only | **REC** | Hold-to-EOD |
|---|---|---|---|---|---|---|---|---|---|---|
| ATM | 08-11 | 78400CE | 3 | 376.80 | 11:10 | 11:50 @248.45 | +2,418 | +2,184 | 13:06 @251.75 **+2,352** | +2,184 |
| ATM | 08-12 | 78100CE | 3 | 299.15 | 10:03 | 10:40 @169.40 | +2,449 | +2,583 | 13:19 @90.25 **+4,033** | +2,583 |
| ATM | 08-14 | 77900PE | 3 | 420.00 | 14:02 | 14:40 @283.30 | +2,584 | +2,932 | EOD @272.25 **+2,932** | +2,932 |
| ATM | 08-17 | 77800CE | 3 | 357.10 | 10:42 | 11:20 @258.60 | +1,822 | **−159** | 11:51 @290.00 **+1,194** | **−2,221** |
| ATM | 08-18 | 77600CE | 3 | 264.05 | 10:55 | 11:35 @199.30 | +1,150 | +1,036 | 11:51 @221.15 **+713** | +1,036 |
| ATM | 08-19 | 77100CE | 2 | 247.75 | 10:31 | 11:10 @154.25 | +1,726 | +1,938 | 11:50 @181.60 **+1,179** | +1,938 |
| ATM | 08-24 | 77900CE | 2 | 310.20 | 10:59 | 14:30 @146.20 | +3,134 | +3,027 | 14:31 @151.40 **+3,030** | +3,027 |
| ATM | 08-25 | 77300PE | 2 | 319.85 | 09:19 | 09:55 @279.65 | **+657** | +4,343 | 13:02 @246.65 **+1,317** | +4,343 |
| ATM | 08-26 † | 78000CE | 2 | 212.50 | 10:20 | 11:00 @132.60 | +1,455 | +2,071 | 12:22 @108.20 **+2,071** | +2,071 |
| ATM4 | 08-12 | 78100CE | 3 | 298.90 | 10:58 | 11:35 @102.85 | +3,776 | +2,578 | 13:19 @90.25 **+4,028** | +2,578 |
| ATM4 | 08-24 | 77900CE | 2 | 310.20 | 11:05 | 11:45 @159.55 | +2,867 | +3,027 | 14:31 @151.40 **+3,030** | +3,027 |
| ATM4 | 08-25 | 77600CE | 2 | 223.05 | 10:22 | 11:00 @154.60 | +1,226 | **−1,097** | 11:56 @205.65 **+204** | **−2,322** |

† 08-26 is a partial recording (chain ends 12:22), so its EOD figures are truncated.

**Book total at the lots actually traded, net of measured costs:**

| | Total |
|---|---|
| What the broken trail actually booked | **+Rs64,727** |
| Breakeven-only counterfactual | +Rs60,079 |
| **Recommended design** | **+Rs67,417** |

The recommended design beats **both** on the live sample: +Rs2,690 vs the accident and +Rs7,338 vs
breakeven-only. It captures the two days where holding would have surrendered everything (08-17,
08-25 ATM4 — the two round-trips) while still riding 08-12 down to 90.25, which the accident's
40-minute exit and the breakeven arm both missed. On 08-25 ATM (the accident's worst day, where the
early exit cost Rs7.1k against holding) it recovers Rs1,317/lot of the Rs3,686/lot gap.

---

## 8. RECOMMENDATION — exact design to implement

**Design: the NIFTY mechanism, ported verbatim. A ceiling, never an `sl_price`.**

| Parameter | Value | Basis |
|---|---|---|
| Stop object | **Ceiling held in memory**; `sl_price` keeps the breakeven value only | measured — the write-into-`sl_price` is the defect (62% self-trigger) |
| Trail function | `compute_short_trailing_stop` (ratcheting upper band), **not** `calc_supertrend` | measured — ratchet sits above price in 100% of episodes vs 38% |
| ST period | **7** | judgement on a flat plateau; identical to NIFTY |
| ST multiplier | **3.0** | judgement on a flat plateau; identical to NIFTY |
| Candle | 5-minute premium candles (unchanged) | matches the live builder and the NIFTY path |
| Confirmation | premium must hold above the ceiling for **≈60 s = 6 consecutive 10-second polls** | measured plateau (30 s–3 min good, ≥5 min measurably worse); the replay resolves 1-minute polls, not 10-second ones |
| Warm-up | **seed** the 5-min candles from the leg's own 09:16→now premium history | measured — return-neutral, removes a median 39-minute trail-less window |
| Warm-up fallback | if seeding fails/returns <8 bars, `sl_price = entry` (today's `SENSEX_BE_PROTECT`) | unchanged, and it is what makes the fix safe to ship |
| Breakeven clamp | keep `stop = min(ceiling, entry)` | measured — never binding after arming, so free insurance |

**Expected effect, stated honestly:** +Rs315/lot vs the incumbent (t = 2.31, the one comparison that
holds up) and a much better tail (worst episode −Rs613 vs −Rs866 incumbent, −Rs1,002 breakeven-only,
−Rs15,716 no-stop). At 2 lots and ~1 survivor episode per usable day, that is order Rs600/episode.
**Do not expect it to beat breakeven-only out-of-sample — it does not.** Its job is to be a correct
trail that caps the 23% round-trip tail without the 40-minute random exit.

### Proposed patch sketch — NOT APPLIED. Deploy after 15:40 IST with sign-off.

**`services/sensex_naked_trail.py`** — return a *ceiling* plus a confirm decision, never an `sl_price`:

```python
ATR_PERIOD, MULT, CONFIRM_POLLS = 7, 3.0, 6      # ~60 s at the 10 s SL-monitor cadence
_state = {}   # pos_id -> {candles, cur, bkt, breach}

def seed(pos_id, ohlc_5min):
    """Prime the candle history from the leg's own 09:16->now 5-min premium candles
    (mirror of nas_ticker._seed_naked_candles). Called once when the leg goes naked."""
    _state.setdefault(pos_id, {...})['candles'] = list(ohlc_5min)[-200:]

def trail_ceiling(pos_id, ltp, entry, now=None):
    """Return (ceiling, exit_now). ceiling is ALWAYS above the premium by construction, or
    None during warm-up. exit_now is True only after CONFIRM_POLLS consecutive polls above it."""
    ...aggregate the 5-min bar exactly as today...
    st = NasAtm4Executor.compute_short_trailing_stop(completed_bars, ATR_PERIOD, MULT)[0]
    if st is None:
        s['breach'] = 0
        return None, False                      # -> caller uses breakeven
    ceiling = min(st, entry)                    # clamp: measured non-binding, kept as a floor
    if ltp > ceiling:
        s['breach'] += 1
        return ceiling, s['breach'] >= CONFIRM_POLLS
    s['breach'] = 0
    return ceiling, False
```

**`app.py` (~8144-8163)** — stop writing the ST into `sl_price`; fire the exit explicitly:

```python
_be = round(_p['entry_price'], 1)
_ceil, _fire = _sx_trail_ceiling(_p['id'], ltp_map.get(_p['tradingsymbol']), _p['entry_price'])
# sl_price ALWAYS stays at breakeven: the generic check_and_handle_sl remains the hard floor.
if abs((_p.get('sl_price') or 0) - _be) > 0.05:
    executor.db.update_position(_p['id'], sl_price=_be,
                                notes='SENSEX_ST_CEIL(7,3)' if _ceil else 'SENSEX_BE_PROTECT (warmup)')
_p['sl_price'] = _be
if _fire:
    logger.warning("[%s] TRAIL EXIT %s: live %.1f > ceiling %.1f for %d polls",
                   name, _p['tradingsymbol'], _live, _ceil, CONFIRM_POLLS)
    executor._close_leg(_p, _live, 'ST_TRAIL_EXIT')
    continue
```

**Emergency one-line stopgap** if the full patch cannot ship immediately — never write a stop at or
below the market:

```python
if _st is not None and _st < _be and _st > _live:      # <-- the missing guard
    _sl, _note = round(_st, 1), 'SENSEX_ST_TRAIL(7,3)'
else:
    _sl, _note = _be, 'SENSEX_BE_PROTECT (warmup/fallback)'
```

This alone removes the self-trigger (the 62% case). It does **not** add seeding or confirmation, so
it lands between `BE_ONLY` and the recommended arm.

---

## 9. Honest caveats

1. **Sample.** 86 episodes over 84 usable days spanning one four-month window (2026-04-28 →
   2026-08-25). One regime. Every conclusion is provisional on that.
2. **Resolution gap.** The live trail builds its 5-minute candles from **10-second** polls; the
   replay uses **1-minute** chain snapshots. Candle highs/lows are therefore slightly tamer, ATR
   slightly narrower and confirm counts coarser. Fidelity is 9/12 exact on the real episodes with
   the 3 misses all firing *later* in replay — conservative for the incumbent, so the incumbent's
   ranking is a ceiling on its true performance. **The recommended confirm count cannot be resolved
   in 10-second units by this study**; the 60 s recommendation is an interpolation onto the measured
   1-minute plateau.
3. **Leg-in-isolation.** Episodes are scored on the survivor leg alone. The ATM re-entry cycle and
   the ATM4 roll P&L are arm-independent and excluded. The **venue book stop (−Rs1,300/lot) and take
   profit (+Rs4,000/lot)** are an outer layer that can truncate any arm on a bad day; they are not
   modelled and would compress all arms toward each other on the extreme days.
4. **Not statistically distinguishable from breakeven-only.** +Rs259/lot at t = 1.35 in-sample,
   +Rs63/lot out-of-sample. No cell survives a Bonferroni haircut over 188 arms. The claim that
   holds is the comparison against the **incumbent** (t = 2.31) and against **no stop**.
5. **Parameters are a judgement call.** ST(7,3) is chosen for plateau membership and NIFTY parity,
   not because it is the measured optimum (the grid peak was p10/m2.0/N3, deliberately declined as
   peak-picking).
6. **DTE0 untouched.** Expiry-day SENSEX carries no per-leg stop (`leg_sl_disabled_dtes=(0,)`,
   research/114), so no survivor episode exists there and this study says nothing about it.
7. **2026-08-26 truncation.** The most recent live episode's chain recording ends at 12:22, so its
   EOD counterfactuals in §7 are truncated (they favour no arm in particular).
8. **Seven sins.** Look-ahead: every decision uses only ≤ t prices, exits fill at the poll price.
   Survivorship: none — every usable chain day is replayed, including the 16 days where no leg
   stopped, the 17 DTE0 days and 1 day whose survivor path was too short (50+1+16+17=84). Overfitting: family-wise haircut reported, peak declined, plateau
   preferred. Cost: measured model, gross↔net and ×3 sensitivity shown. Regime: single 4-month
   window — the main weakness. Correlation: single-leg study, book effects flagged in (3).
   Capacity: unchanged from the live book (2 lots).

---

## 10. Next levers

1. **Deploy the fix after 15:40 IST with sign-off**, then re-measure on the next ~20 live survivor
   episodes against this study's predicted +Rs315/lot vs incumbent. Register the re-check in the Ops
   & Review Centre.
2. **The bigger lever is not the trail — it is Wednesday/DTE1 sizing.** Every arm's spread is driven
   by the 29 Wednesday episodes (31% round-trip rate, HOLD mean Rs1,219 vs Rs3,365 on Monday).
   Sizing down the SENSEX Wed book is worth more than any trail parameter, and research/104 and
   research/113 already point the same way.
3. **Re-run this grid once ~150 episodes exist** (roughly February 2027 at the current rate). At
   n = 86 nothing survives family-wise; at n = 150 the +Rs250-450/lot plateau would.
4. **Port the same audit to the NIFTY path.** The NIFTY trail is structurally correct, but its
   confirm count (3 ticks, sub-second) has never been calibrated against this kind of evidence —
   the N-sensitivity curve here suggests a *time*-based confirm would be better than a tick count.

---

### Reproducibility stamp

| | |
|---|---|
| Data snapshot | `backtest_data/options_data.db` on the VPS as of 2026-08-26 12:00 IST (SENSEX 2026-04-20 → 2026-08-26) |
| Live episode source | `backtest_data/sensex_atm_trading.db`, `sensex_atm4_trading.db` (`nas_atm_positions`, `exit_reason='SL_HIT' AND notes LIKE '%ST_TRAIL%'`) |
| Scripts | `scripts/build_episodes.py` → `scripts/sweep_arms.py` → `scripts/analyze.py` → `scripts/diagnostics.py` → `scripts/reconcile_live.py` |
| Cost assumption | research/122 measured model, one leg: brokerage Rs20×2/`NLOTS_REF`=10, STT 0.1% sell, txn 0.03503%, IPFT 5e-6, SEBI 1e-6, stamp 0.003% buy, GST 18%, slip entry 0 / triggered +6.548 pt / EOD +0.178 pt |
| Lot | 20 · live book size 2 lots |
| Recommended arm label | `CEIL_p7_m3.0_N1_SEED` |
| Arms run | 188 (160 ST-ceiling cells + 6 no-clamp + 13 giveback + 6 take-profit + 3 nulls) |
