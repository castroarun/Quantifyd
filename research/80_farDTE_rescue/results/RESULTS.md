# research/80 — Rescuing the far-DTE days: five ideas tested, four dead, one standing

**Status: IN PROGRESS (G1–G4 done; overnight/positional still to run)**

## The question
research/79 showed NAS-OPT makes **+1,578/day on 0/1-DTE** and **loses 441/day on DTE>=4**. Can the
far-DTE days be rescued — by a better stop, a different band, a premium SL, a filter, a skew, or a
different system entirely?

---

## G1 — WHY far-DTE loses (decomposition on the real chain, 58 days)

**The keep rate is the whole story.** How much of the premium you sell do you still have at 14:45?

| DTE | credit sold | kept if calm | **keep rate** | given back if band hit |
|---|---|---|---|---|
| 0 | 44.6 | 37.2 | **83%** | −3.3 |
| 1 | 138.2 | 25.8 | **19%** | +0.2 |
| 4 | 178.2 | 11.5 | **6%** | −2.7 |
| 5 | 190.1 | 12.5 | **7%** | −13.1 |
| 6 | 273.0 | 27.7 | **10%** | −12.8 |

On expiry day you sell 45 points and keep 37. On Wednesday you sell 273 and keep 28 — carrying 6×
the premium risk to harvest crumbs, because **theta lives in the final DAYS of an option, not the
final hours**. And a far-DTE option has real delta, so a 0.4% move costs about as much as a calm day
pays. The asymmetry that makes DTE0 work (+37.2 vs −3.3) is simply absent.

**Ceiling:** far-DTE pays +1,753 calm / −1,539 hit, hit rate 67%. Even a PERFECT COSTLESS stop —
one that exits every losing day at exactly zero — caps out at **+578/day**. No exit rule at the same
band can beat that.

## G2a — 22 exit rules swept on the real premium paths → **DEAD**

Band widths 0.2–1.0%, no-stop, combined-premium SL (relative + absolute), per-leg SL, profit targets.

| rule | mean/day |
|---|---|
| move 0.4% (live) | −441 |
| **best of all 22 (prem ×1.3)** | **+9** |
| NO STOP | −501 |

**The best of 22 rules earns +9/day over 33 days** — i.e. zero — and it is the *maximum of 22 tries*,
which is precisely how a fake winner is manufactured. Confirms G1: you cannot tune a stop into an
edge that is not in the payoff.

## G2b — is any of it predictable? (2,693 days of NIFTY 5-min) → **DEAD as a rescue**

- **The weekday effect was NOISE.** P(move >= 0.4%): Mon 76.2 / Tue 77.1 / Wed 74.4 / Thu 78.3 /
  Fri 78.5%. Identical. The "Wed/Thu move more" signal from the 11-day chain sample does not exist.
  So far-DTE's problem is 100% the theta structure, not wilder days.
- **Volatility IS predictable, monotonically:** VIX <12 → 50.3% hit; >=22 → 97.6% (corr +0.585).
  Opening range corr +0.588, |gap| +0.439, CPR width +0.328.
- **But it cannot rescue selling.** The breakeven hit rate is 53.3%. Only VIX<12 (50.3%) clears it —
  and on low-VIX days the premium collected shrinks too, so the payoff is not invariant to the
  filter. Not an edge.

## G3 — after the band breaks, does price continue? (2,133 break days)

- **Direction is a COIN FLIP: 53.8% close beyond the break** — and it is 53–57% in every single
  bucket. **Buying direction after a break is dead.**
- **But "does it come back" is highly predictable, and monotone:**

| break time | comes back to entry | full whipsaw |
|---|---|---|
| before 10:00 | 61% | 31% |
| 10:00–11:00 | 40% | 14% |
| 11:00–12:00 | 29% | 7% |
| **after 12:00** | **18%** | **3%** |

  Same monotone structure by VIX (15%/3% at VIX<12 vs 50%/23% at VIX>=17), by opening range, and by
  CPR width.

## G4 — DIRECTIONAL SHORT after the break (sell the far side) → **DEAD**

The idea: a short does not need continuation, only that price does not come back — and G3 says a
late break rarely does. It also side-steps the theta curse, because a directional short is paid by
DELTA (the option going OTM as price walks away), not by time decay. Swept break% × OTM distance ×
break-time × VIX over 11 years on the calibrated engine.

**Every cell is negative after 1% slippage.** Best: −40/trade.

| slippage | mean/trade |
|---|---|
| 0% | **+101** |
| 1% | **−40** |
| 2% | −181 |

**The entire edge lives inside the transaction cost.** Only 5/12 years positive.

**And the trap:** the best cell has a **+343 MEDIAN and a 60% win rate — with a NEGATIVE MEAN.**
Many small wins, rare −44,797 disasters. Reporting win-rate or median would have sold this as a
discovery. VIX conditioning is non-monotone (+50 / −348 / −145 / +219) = noise, in sharp contrast to
G3's clean monotonicity — which is exactly how you tell the two apart.

**Why G3's true insight was not sufficient:** the option you sell after a 0.4% break is CHEAP
(50–200pt OTM, a few hours of life). The premium collected does not cover the 3–18% of days that do
come back, plus costs. Being right 60% of the time does not help when the other 40% costs 3× more.

## The engine (built for G4 and everything after it)

Black-Scholes on NIFTY 5-min + India VIX as the IV base, with an IV multiplier calibrated PER DTE
against the real chain, then validated on it:

| DTE | IV mult vs VIX | strangle credit error |
|---|---|---|
| 0 | 1.51 | 14% |
| 1 | 1.23 | 4% |
| 4–6 | 0.92–1.01 | 5–6% |

The IV term structure (short-dated ~1.5× VIX, far-dated ~1.0×) **emerged from the fit rather than
being imposed** — a good sign it is not nonsense. VERDICT: usable, with the caveat that k(DTE) is
fitted on 58 days and assumes a stable term structure across 11 years.

---

## Scoreboard

| idea | verdict |
|---|---|
| better / different move-stop | **DEAD** (+9/day, best of 22) |
| premium SL (%, absolute, per-leg) | **DEAD** (same sweep) |
| "Wed/Thu move more" → trade it | **DEAD** (weekday effect = 11-day noise) |
| filter calm days (VIX / CPR / OR / gap) | **DEAD** (the premium shrinks with the filter) |
| directional selling after a break | **DEAD** (edge < slippage; 5/12 years) |
| **overnight / positional holding** | **STILL STANDING** — the only idea that attacks G1's actual cause |

## Next
**Overnight / positional far-DTE selling.** It is the ONLY idea that stops fighting the G1 finding:
if theta lives in the final days, then HOLD for days instead of renting the risk for five hours.

To test: enter Wed/Thu (DTE 4–6), hold N days into expiry; strangle vs iron condor (capped tail);
OTM width; VIX / CPR filters; and **overnight gap risk — the thing that can kill it, and precisely
what an intraday system never has to face.**
