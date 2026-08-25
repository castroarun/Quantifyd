# research/125 — Portfolio Profit Trail vs Bought Wings vs Strike Diversification — RESULTS

**Verdict: NO EDGE for the portfolio profit trail (Arm A) — a fourth independent
reproduction of "static wins". NO EDGE for bought OTM wings at deployable size (Arm B):
they are measurable and they do cap the tail, but the premium costs 28–100 % of the
book's entire P&L, and the cost is *theta* — the very thing the book earns. The one
positive is Arm C, the arm nobody asked for first: SIGNAL — spreading the same
notional across strikes cuts the tail for no premium and no firing cost, and on SENSEX
it improves the mean as well.**

Stage: **G2 (mechanics, real data, net of the measured cost model).**
Sample: 62 full-book sessions (2026-05-20 → 2026-08-24) for Arms A/B, 83–84 sessions
(2026-04-27 → 2026-08-24) for the sleeve-only and Arm-C samples. Real 1-minute option
chain + the real per-minute live MTM of the 9:16 suite.

---

## 0. Stage-0 reconciliation — the day that triggered the commission

2026-08-25, rebuilt independently from the DBs before any sweep:

| minute | portfolio | TimeB | COMB | 916-ATM | 916-ATM2 | 916-ATM4 |
|---|---|---|---|---|---|---|
| 14:00 | **+13,741** | +3,640 | +4,914 | +1,306 | +4,888 | −1,008 |
| **14:03 (peak)** | **+13,865** | +3,640 | +5,064 | +1,306 | +4,862 | −1,008 |
| 14:30 | +9,730 | +3,640 | +3,094 | +1,306 | +2,698 | −1,008 |
| 14:33 | +2,970 | +3,640 | −260 | +1,306 | −708 | −1,008 |
| 15:03 | **−5,160** | +3,402 | −3,725 | +1,306 | — | — |

- **The structural claim is confirmed exactly.** The 9:16 suite peaked at **+₹5,187**
  against its **+₹12,000** arm threshold — the existing venue trail **could never have
  armed**, on this day or on any day like it.
- Portfolio peak **+₹13,865 at 14:03**, give-back to close **₹19,026**. (The commission
  quoted +₹14,983; the ₹1,118 gap is book scope — this figure counts **live-money books
  only**, per the STATUS-MD scope. The peak *minute* and the mechanism reconcile exactly.)
- TimeB was the only book that kept its gain, because it had **already exited at 11:00**.
  That is the shape of the whole answer: what protected the profit was a *time exit*,
  not a trail.

---

## 1. Arm A — the portfolio profit trail. **NO EDGE, and not marginally.**

Baseline, as deployed, no defence (n = 62): **total ₹626,965 · mean ₹10,112/day ·
median ₹9,668 · win 75.8 % · worst day −₹29,950 · p10 −₹11,516.**
Give-back (peak MTM − final MTM): **median ₹4,137**, p75 ₹10,701, p90 ₹19,401, max ₹46,101.
Peak MTM: median ₹17,158, p90 ₹34,294; 57 of 62 days peaked ≥ ₹5k, 46 ≥ ₹10k.

### 1.1 Nothing beats doing nothing

| variant | total | mean | median | win% | worst | **Δ vs no-trail** | fires | needless | rescues |
|---|---|---|---|---|---|---|---|---|---|
| **NULL_NAKED (as deployed)** | **626,966** | 10,112 | 9,669 | 75.8 | −29,950 | **0** | 0 | 0 | 0 |
| NULL_SUITETRAIL_2000_350 *(the live overlay)* | 605,593 | 9,768 | 10,007 | 75.8 | −29,950 | **−21,373** | 10 | 7 | 3 |
| TRAIL_A5000_G2500_ONLYLOSERS *(best of 132)* | 542,539 | 8,751 | 9,280 | 72.6 | −29,950 | **−84,427** | 55 | 12 | 4 |
| NULL_FIXEDTP_30000 | 541,245 | 8,730 | 10,007 | 75.8 | −29,950 | −85,721 | 12 | 8 | 4 |
| TRAIL_A12000_G4000_ONLYLOSERS | 526,486 | 8,492 | 7,991 | 74.2 | −29,294 | −100,480 | 34 | 9 | 2 |
| TRAILPCT_A20000_P50 | 517,567 | 8,348 | 7,165 | 71.0 | −29,950 | −109,398 | 9 | 8 | 1 |
| TRAIL_A20000_G2500 *(best flat cell)* | 451,873 | 7,288 | 8,771 | 77.4 | −29,950 | −175,093 | 20 | 16 | 4 |
| TRAIL_PERVENUE_A5000_G2500 | −239,675 | −3,866 | — | — | −33,512 | −866,640 | 58 | 49 | 8 |

**0 of 132 trail cells beat the null.** The best one costs **₹84,427 — 13.5 % of the
book's entire P&L** — and does not improve the worst day by a single rupee.

### 1.2 The plateau map runs monotonically to the boundary, and the boundary is "no trail"

Δ total (₹ vs no-trail), flat trail, ARM × GIVEBACK:

| ARM \ GB | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 4,000 | 5,000 | 7,500 |
|---|---|---|---|---|---|---|---|---|
| 3,000 | −781,246 | −744,165 | −688,654 | −679,084 | — | — | — | — |
| 5,000 | −722,580 | −663,540 | −625,550 | −618,855 | −614,996 | −603,972 | — | — |
| 8,000 | −625,794 | −614,414 | −597,050 | −582,646 | −554,706 | −546,082 | −501,117 | −486,805 |
| 10,000 | −575,489 | −568,510 | −540,032 | −507,474 | −476,280 | −448,885 | −348,691 | −365,772 |
| 12,000 | −535,120 | −512,911 | −511,146 | −465,799 | −453,253 | −408,042 | −296,684 | −329,325 |
| 15,000 | −332,260 | −285,739 | −289,045 | −278,708 | −270,188 | −253,712 | −234,497 | −263,772 |
| 20,000 | −218,174 | −178,158 | −182,391 | −175,093 | −176,929 | −195,956 | −175,400 | −205,924 |

Every row improves as the arm rises; every row improves as the give-back widens. **There
is no interior optimum — the gradient runs to "never fire", which is the live rule.**
This is precisely the r/116 signature, reproduced on a different construction (portfolio
rather than sleeve) and a different sample.

### 1.3 Why — the arithmetic that decides it

- **The give-back is small and the right tail is large.** Median give-back ₹4,137, but the
  median *peak* is ₹17,158 and the book's mean day is ₹10,112. A rule that harvests a
  ₹4k median give-back must forgo the days that finish above their mid-session dip.
- **Firing is expensive and certain.** A forced mid-session exit pays **+6.548 pt/leg-side**
  against +0.178 for a time exit (measured, 443 real live leg-sides). The live suite trail
  fired 10 times, of which **7 were needless**, at a cost of **₹35,661** on those 7 — about
  **₹5,094 per needless fire**.
- **The tail is untouched.** 59 of 132 cells "improve" the worst day — by **+₹656**. The
  worst day is a day that never reached profit, so a *profit* trail never arms on it.
  A profit trail is structurally incapable of fixing the left tail.

### 1.4 The placebo — the trail is *skilful*, and skill is not the problem

Random-minute exit after the book first clears ARM (200 draws per level):

| ARM | placebo p05 | placebo median | placebo p95 | best real trail | NULL_NAKED |
|---|---|---|---|---|---|
| 5,000 | 34,147 | 132,755 | 226,838 | 542,539 | **626,965** |
| 8,000 | 85,226 | 172,946 | 261,461 | 517,022 | **626,965** |
| 10,000 | 136,632 | 219,412 | 314,747 | 508,510 | **626,965** |
| 12,000 | 206,816 | 285,512 | 349,407 | 526,486 | **626,965** |
| 20,000 | 363,572 | 417,468 | 486,691 | 451,873 | **626,965** |

The peak-tracking machinery beats random exiting decisively — it is *not* a coin flip.
**It is skilful early exiting, and skilful early exiting still destroys value here.** That
is a stronger negative than "the rule is noise": the rule works as designed, and the design
is wrong for this book.

### 1.5 The fixed-TP null reproduces r/90 exactly

TP 30k → −85,721 · 20k → −239,779 · 15k → −362,209 · 10k → −541,391 · 7.5k → −683,847 ·
5k → −730,276. Monotone-bad as the target tightens, independently reproducing r/90's
"a daily take-profit is value-destructive; it caps the fat right tail that carries the edge".
**A profit trail is a soft take-profit, and it inherits the same defect.**

### 1.6 Byproduct: the currently deployed suite trail is itself mildly negative

`NULL_SUITETRAIL_2000_350` costs **−₹21,373** over 62 days (10 fires, 7 needless, 3 rescues).
**Do not act on this here** — n = 10 firings, one regime, and the overlay was justified by
r/90 on a different sample. It is logged as a dated re-check, not a recommendation.

---

## 2. Arm B — buy OTM wings. **Measurable (the trap was checked). Uneconomic.**

### 2.1 The staleness audit — PASSED, and this is the part to read first

A held-wing intraday backtest on this project was previously invalidated by stale far-OTM
quotes. That failure does **not** reproduce in this strike band:

| venue | dist | sleeve-days | minutes | zero-bid % | zero-ask % | median spread %mid | max identical-print run | mean run | days wing never traded |
|---|---|---|---|---|---|---|---|---|---|
| NIFTY | 100 | 98 | 27,139 | 0.0 | 0.0 | 0.3 | 4 | 1.0 | **0** |
| NIFTY | 200 | 98 | 27,139 | 0.0 | 0.0 | 0.3 | 7 | 1.1 | **0** |
| NIFTY | 300 | 98 | 27,102 | 0.0 | 0.0 | 0.4 | 7 | 1.1 | **0** |
| NIFTY | 500 | 98 | 25,535 | 0.0 | 0.0 | 0.9 | 8 | 1.2 | **0** |
| SENSEX | 400 | 51 | 9,908 | 0.0 | 0.0 | 0.3 | 6 | 1.0 | **0** |
| SENSEX | 1200 | 51 | 9,825 | 0.0 | 0.0 | 0.5 | 4 | 1.0 | **0** |
| SENSEX | 2000 | 51 | **2,076** | 0.0 | 0.0 | 1.5 | 5 | 1.2 | **0** |

Every wing minute carries a **two-sided quote**; the median bid-ask is **0.3–1.5 % of mid**;
identical-print runs average **1.0–1.2 minutes**, i.e. the quote moves essentially every
minute; and **every wing strike traded on every sleeve-day** (r/89 liquidity rule passes
with zero exclusions). The one caveat: **SENSEX 2000-wide is present on only 2,076 of ~9,900
minutes** — that strike is thin in coverage and its numbers are the least trustworthy row.

Wings are priced **bought at the ASK, sold back at the BID** throughout. So the previous
invalidation was a property of *that* construction (far-OTM, LTP-marked, held overnight),
not of this data at these distances. **Arm B is measurable.**

### 2.2 And having established we can trust the numbers — the numbers are bad

Per-sleeve, wings bought at entry and held to the sleeve's own exit:

| sleeve | dist | n | naked total | wing cost | hedged total | **of which spread** | **of which decay** | naked worst | hedged worst |
|---|---|---|---|---|---|---|---|---|---|
| TB_NIFTY | 100 | 32 | 184,837 | −168,103 | 16,733 | −4,341 | **−157,794** | −26,668 | −10,951 |
| TB_NIFTY | 200 | 32 | 184,837 | −101,018 | 83,818 | −2,860 | −93,443 | −26,668 | −16,864 |
| TB_NIFTY | 500 | 32 | 184,837 | −26,487 | 158,349 | −1,871 | −21,190 | −26,668 | **−26,272** |
| COMB20 | 100 | 66 | 188,892 | −206,716 | **−17,824** | −3,974 | −193,401 | −13,076 | −4,625 |
| COMB20 | 500 | 66 | 188,892 | −78,857 | 110,034 | −1,673 | −70,453 | −13,076 | −12,917 |
| TB_SENSEX | 400 | 34 | 245,040 | −145,392 | 99,647 | −2,768 | −137,703 | −29,233 | −23,088 |
| TB_SENSEX | 800 | 34 | 245,040 | −75,774 | 169,265 | −1,527 | −70,288 | −29,233 | **−29,073** |
| **SXWED** | 400 | 17 | **−25,136** | **+12,733** | **−12,402** | −1,011 | **+15,984** | −15,741 | **−3,153** |

**The decisive line is the decomposition.** For TB_NIFTY at 100-wide the wing costs
₹168,103, of which **₹4,341 is bid-ask and ₹157,794 is decay**. This is not an execution
problem that better fills could fix — **the wing simply hands back theta, and theta is the
entire edge of a short-premium book.** Buying protection here is structurally equivalent to
turning the strategy down.

Portfolio-level totals: NIFTY 100-wide costs **₹374,819 against ₹373,729 of naked P&L —
100 % of the book** — to improve the worst day by ₹15,716. NIFTY 500-wide costs ₹105,345
(28 %) to improve the worst day by **₹396**. On SENSEX, every wing **≥ 600 points leaves
the worst day essentially unchanged** (−29,233 → −29,126 / −29,074 / −28,962): the SENSEX
tail is a slow grind inside the wing, not a gap through it, so the insurance never pays.

### 2.3 The two shapes that are *not* silly

- **"Lock the profit with wings" (AFTERUP)** — buy wings only once the sleeve is up ≥ 40 %
  of its credit. It arms on only **11 of 98 sleeve-days**, costs little (NIFTY 100-wide:
  −₹4,998 total, mean −₹454), and turned the worst of those 11 days from −₹6,275 to
  **+₹955**. This is the cheapest wing shape by a wide margin. **n = 11 — a hint, not a
  finding**, and it protects days that were already winning.
- **Hedging a losing sleeve.** SXWED (the full-day SENSEX-Wednesday cell) is net **−₹25,136**
  over 17 days; 400-wide wings **pay +₹12,733** and cut its worst day from −₹15,741 to
  −₹3,153. Correct reading: **wings rescue a sleeve that should not be trading.** The
  cheaper fix is the sleeve, not the hedge. (Config note records this cell was deployed as
  a user override against the study verdict.)

---

## 3. Arm C — strike / entry diversification. **SIGNAL — the cheapest defence of the three.**

### 3.1 The concentration is real, and it is provable from config, not from one day

`NAS_916_ATM_DEFAULTS`, `NAS_916_ATM2_DEFAULTS`, `NAS_916_ATM4_DEFAULTS` all inherit
`NAS_ATM_DEFAULTS` and set `entry_start_time: '09:16'`. They sell the **same ATM straddle,
same venue, same expiry, at the same minute**. They differ **only in exit machinery**
(ATM2 = ₹2,500/lot rupee stop, one-and-done; ATM4 = `max_rolls: 1`).

Measured on the sample:

- all live NIFTY books on **one strike on 46 %** of multi-book days (36 of 78);
- **the three suite systems alone share a strike on 78 %** of days (45 of 58).

So "three systems at 2 lots" is, on most days, **one position at 6 lots with three exit
rules**.

### 3.2 But the "correlation ≈ 1 when it matters" hypothesis is REFUTED

Worst-decile days (n = 9), pairwise correlation of daily net P&L:

| pair | corr, all days | **corr, worst decile** |
|---|---|---|
| 916_ATM ↔ 916_ATM2 | +0.50 | **−0.58** |
| 916_ATM ↔ 916_ATM4 | +0.81 | **−0.19** |
| 916_ATM2 ↔ 916_ATM4 | +0.55 | **−0.32** |
| COMB20 ↔ TB_NIFTY | −0.07 | n/a (1 common day) |

**Every book lost on only 2 of the 9 worst-decile days.** The differing *exit rules* do
de-correlate the clones precisely when it matters — which is the strongest available
defence of the current design. Exit clustering is likewise not typical: across 30 days with
≥ 2 mid-session suite exits, the median span between first and last is **9,220 s (2.6 h)**,
and only **7 %** of days saw all exits inside 120 s (2026-08-25's own suite exits spanned
10,678 s: 11:55, 13:01, 14:53). The 90-second cluster that prompted this question was a
cross-book coincidence on one day, **not the standing behaviour of the book.**

### 3.3 The equal-notional test — and here the tail really does move

Three clones × 2 lots of the COMB-shape construction, 84 days per venue, net of the same
cost model:

| portfolio | NIFTY total | NIFTY worst | NIFTY p10 | SENSEX total | SENSEX worst | SENSEX p10 |
|---|---|---|---|---|---|---|
| **CLONE_SAME** *(today's shape)* | **428,547** | **−33,753** | −16,776 | 541,728 | **−29,166** | −10,156 |
| DIV_STRIKE_1 (−1/0/+1) | 347,845 | −30,016 | −16,001 | 519,099 | −27,478 | −10,357 |
| **DIV_STRIKE_2 (−2/0/+2)** | 335,518 | **−23,582** | **−11,209** | **605,476** | **−20,392** | **−7,789** |
| DIV_ENTRY (09:16/09:31/09:46) | 314,332 | −30,858 | −16,657 | 468,995 | −23,102 | −9,927 |
| DIV_BOTH | 273,218 | −30,339 | −10,307 | 479,182 | −20,001 | −8,347 |

Paired per-day deltas vs CLONE_SAME:

| | mean Δ/day | median Δ | t | worst day |
|---|---|---|---|---|
| NIFTY DIV_STRIKE_2 | −1,107 | −366 | **−1.26 (NS)** | −33,753 → **−23,582 (+10,171)** |
| SENSEX DIV_STRIKE_2 | **+759** | +258 | **+1.79** | −29,166 → **−20,392 (+8,774)** |
| NIFTY DIV_ENTRY | −1,360 | −1,126 | −2.16 | +2,895 |
| SENSEX DIV_ENTRY_WIDE | −1,712 | −1,244 | −2.08 | +8,756 |

**Entry-time staggering is the worse idea** (significantly negative mean on both venues,
small tail gain) — consistent with r/95's "early time entry wins; don't wait". **Strike
spreading is the good one**: on NIFTY it buys ₹10,171 of worst-day for a mean cost that is
statistically indistinguishable from zero; on SENSEX it buys ₹8,774 of worst-day **and
pays ₹63,748 for the privilege**.

### 3.4 Placebo discipline on Arm C — where it holds and where it does not

200 random 3-leg portfolios drawn from the same cell menu:

| | CLONE_SAME | placebo p05 | placebo median | placebo p95 |
|---|---|---|---|---|
| NIFTY worst day | −33,753 | −41,711 | −29,317 | −22,064 |
| SENSEX worst day | **−29,166** | **−22,871** | −16,752 | −12,801 |
| NIFTY total | **428,547** | 128,050 | 228,319 | **356,827** |
| SENSEX total | 541,728 | 348,424 | 477,858 | **597,551** |

Two honest readings:

1. **The concentration penalty is real on SENSEX.** CLONE_SAME's worst day (−29,166) is
   *worse than the 5th percentile* of random 3-leg portfolios (−22,871). Selling the same
   strike three times measurably fattens the SENSEX tail relative to almost any spread of
   legs. That is the concentration finding, quantified and independent of 2026-08-25.
2. **The NIFTY tail gain is inside the noise band.** DIV_STRIKE_2's −23,582 sits between
   the placebo median (−29,317) and p95 (−22,064) — a gain of the magnitude that
   *leg-choice noise alone* produces at n = 84. And CLONE_SAME's NIFTY *total* sits above
   the placebo p95, i.e. the ATM-at-09:16 cell genuinely is the best-performing cell, which
   is exactly why moving off it costs return.

---

## 4. The three defences on one axis — cost per rupee of tail removed

| defence | ₹ cost over the sample | worst-day improvement | **₹ paid per ₹1 of tail cut** |
|---|---|---|---|
| Portfolio profit trail (best of 132) | 84,427 | **₹0** | **∞ — buys nothing** |
| Existing suite trail (live today) | 21,373 | ₹0 | ∞ |
| Wings, NIFTY 100-wide at entry | 374,819 | 15,716 | 23.8 |
| Wings, NIFTY 200-wide at entry | 261,108 | 9,804 | 26.6 |
| Wings, SENSEX ≥600-wide at entry | 96,226+ | ~100 | ~1,000 |
| **Strike spread ±2, NIFTY** | 93,029 | **10,171** | **9.1** |
| **Strike spread ±2, SENSEX** | **−63,748 (it PAYS)** | **8,774** | **free** |

**Strike diversification is ~2.6× cheaper than wings for the same tail benefit on NIFTY,
and on SENSEX it is better than free.** The profit trail is the only one of the three that
buys no tail protection at all.

---

## 5. Success criteria (STATUS-MD §7) — scored

| criterion | Arm A trail | Arm B wings | Arm C strike spread |
|---|---|---|---|
| (a) raises mean/median, or cuts tail without cutting mean | **FAIL** (0/132 cells; tail untouched) | FAIL — caps tail but at 28–100 % of P&L | **PASS on SENSEX**, marginal on NIFTY |
| (b) plateau, not a peak | **FAIL** — monotone to the boundary "no trail" | PASS (monotone in distance) but monotone *toward not buying* | PASS — ±1 and ±2 both cut the tail |
| (c) survives a family-wise haircut | n/a — **nothing beat the null**, so no haircut is needed (r/116 precedent) | n/a | **FAIL as stated** — t=+1.79 over 12 portfolio×venue cells does not survive; needs OOS |
| (d) beats all three nulls | **FAIL** on all three | FAIL | beats naked on SENSEX only |
| (e) fire count + cost when needless | 10 fires / 7 needless / ₹35,661 (live overlay) | n/a — paid every day | **no firing at all** |
| (f) wing staleness + liquidity audit | n/a | **PASS** — 0 % one-sided, meanrun 1.0–1.2, 0 excluded days | n/a |

---

## 6. The seven deadly sins — how each is controlled

| sin | control |
|---|---|
| **Look-ahead** | The trail tests the peak **carried in from prior bars**, then updates it with the current bar; a bar can never fire a trail it just set. Sleeve stops are tested on the printed bar only. Strikes are chosen from `underlying_spot` at the entry minute. Wings are bought at the **ask** of the buy minute and sold at the **bid** of the exit minute — never a same-bar mid. |
| **Survivorship** | Every recorded session is replayed; the only exclusions are the three frozen-chain holidays (<50 distinct spot prints), partial sessions (last snapshot < 15:15), and 2026-08-25 (market still open at build time). All skips are logged in `results/stage1.log`. |
| **Overfitting / multiple testing** | 139 Arm-A cells, 28 Arm-B cells, 12 Arm-C portfolio×venue cells. **No haircut is needed for Arm A because nothing beat the null** (0/132). Arm C's positive SENSEX cell is explicitly declared **not** to survive a family-wise haircut and is labelled OOS-pending. A random-minute placebo (Arm A) and a 200-draw random-leg placebo (Arm C) are reported. |
| **Cost neglect** | The **measured outcome-aware** model throughout: forced exit +6.548 pt/leg-side vs +0.178 for a time exit (443 real live leg-sides), plus the exact Zerodha rate card. The retired flat ₹250/lot constant is not used. Wings pay ask/bid plus the rate card, and the cost is **decomposed into spread vs decay**. |
| **Regime dependence** | **The binding weakness.** One regime, ~4 months, 62 full-book days, no VIX shock and no gap-down cluster. Reported per venue and per sleeve; the Arm-A conclusion is directionally identical in every slice and reproduces three prior studies on different constructions. |
| **Correlation / single-factor** | Directly measured, and it is a headline finding (§3.1–3.2): the books share a strike 46–78 % of the time, yet worst-decile correlations are **negative** because the exit rules differ. Arm C prices the concentration explicitly at equal notional. |
| **Capacity / liquidity** | Front-expiry strikes within ±500 (NIFTY) / ±2000 (SENSEX); the wing audit confirms two-sided quotes and non-zero traded volume on every used strike. **SENSEX 2000-wide is flagged as thin** (2,076 of ~9,900 minutes). Book size 2–8 lots; no capacity constraint, but §7 notes the one impact effect that the lot-rescale cannot capture. |

---

## 7. Honest caveats a reader must carry

- **One regime, 62 full-book days.** Four months. A trail's and a wing's whole case rests on
  tail days; this sample can say "defence does not pay for itself in a normal regime", it
  cannot say "wings would not have helped in March 2020". Wings in particular are insurance
  against an event this window does not contain.
- **The suite MTM was rescaled** from its 5 / 1 / 10 / 2 / 3-lot eras to the currently
  deployed **2 lots/system**. P&L is linear in lots so the rescale is exact — but it cannot
  rescale the *market impact* of a bigger clip, so the 10-lot era's real slippage is
  understated at the rescaled size.
- **Hybrid sourcing, deliberately.** The 9:16 suite is **live truth** (never modelled — its
  cascade/ST-trail/rupee-stop machinery is not faithfully replayable); the CSL sleeves are
  **replayed** from the frozen config. Live sources: 62 days for the suite, 9 REAL CSL
  book-days. Replay: 83 days, 149 sleeve-days. `csl_paper_state.json` keeps only a rolling
  ~8-day window, which is why replay is the workhorse.
- **Sleeve stops are modelled as immediate on the breaching minute**, not with the live
  2-poll dwell. This is consistent with r/116/122/124 for comparability, and it applies
  identically to every variant, so it cancels in the comparison.
- **1-minute granularity understates intrabar breaches.** The bias runs **in the trail's
  favour** (a finer series would fire it more often); it loses anyway.
- **Arm C tests a COMB-shape proxy, not the suite itself.** It answers "does spreading the
  same notional across strikes help a 3-clone book of this shape?" — not "what happens if
  you re-strike ATM2/ATM4", whose exit machinery differs. Treat the magnitudes as
  indicative and the sign as the finding.
- **Wings are held to the sleeve's own exit and never rolled.**
- **Margin relief is NOT modelled.** A defined-risk structure needs materially less margin
  than a naked short, which is a genuine economic argument for wings that this study does
  not price. If the binding constraint is capital rather than risk, Arm B deserves a second
  look on that axis alone.
- **Give-back is measured on gross MTM**; costs are identical across variants and cancel.
- Live-money scope only (per STATUS-MD §3); paper and parked books are excluded consistently.

---

## 8. Recommendation — for Arun's sign-off. **No live change is proposed by this study.**

1. **Do not deploy a portfolio profit trail, in any shape.** 0 of 132 configurations beat
   doing nothing; the best costs 13.5 % of the book's P&L and protects the left tail by
   ₹0. This is the fourth independent reproduction (r/114, r/116, r/121/122, now r/125) of
   the same result: **tightening defence on this book manufactures losses.**
2. **Do not buy wings at deployable size as a P&L measure.** They work exactly as advertised
   — they cap the disaster day — but at 28–100 % of the book's P&L, and the bill is theta,
   not execution. *If* the motivation is margin rather than P&L, that is a different study.
3. **The real answer to "the ₹19k give-back" is TimeB's answer: a time exit.** The only book
   that kept its gain on 2026-08-25 kept it by being flat at 11:00. Windowed books already
   do this; the full-day books (COMB, the 9:16 suite) are the ones that hand profit back.
   That is a **window** question, and r/122's atlas is the instrument for it — not a trail.
4. **The one thing worth taking further is Arm C — strike spreading.** It costs no premium
   and no firing cost, it cuts the NIFTY worst day by ₹10,171 for a statistically
   insignificant mean cost, and on SENSEX it improves the tail *and* the mean. **It is not
   ready to deploy**: it does not survive a family-wise haircut, and it is tested on a
   COMB-shape proxy. **Proposed next step: a G3 out-of-sample re-run and a paper twin**,
   not a live change.
5. **Dated re-check:** the existing suite trail (arm ₹2,000/lot, give-back ₹350/lot) is
   −₹21,373 on this sample with 7 of 10 firings needless. n is far too small to act on.
   Register a re-assessment for **2026-11** alongside the r/122 window re-run.

---

## 9. Next levers

- **Window, not trail** — take the full-day books (COMB20, the 9:16 suite) to the r/122
  atlas and ask whether a *scheduled* exit beats holding to 15:15/15:20. That is where the
  give-back actually lives, and it costs a time exit (+0.178 pt), not a forced one (+6.548).
- **Arm C G3** — walk-forward the ±2-step strike spread, per venue and per DTE, on a proper
  hold-out; if it survives, a paper twin before any live re-strike.
- **Wings on the margin axis** — price the defined-risk structure against margin released
  and lots gained, which is the only frame in which it can win.
- **Re-open Arm A only if** the recorder accumulates a genuinely stressed regime, or if the
  peak-then-round-trip event rate rises materially above what these 62 days show.

---

## 10. Files

| file | purpose | committable |
|---|---|---|
| `scripts/stage0_live_recon.py` | rebuilds the REAL live portfolio curve; reconciles 2026-08-25 | yes |
| `scripts/stage1_build_book.py` | one chain pass → sleeve minute paths + wing price paths | yes |
| `scripts/stage2_trail_sweep.py` | Arm A: 139 variants + nulls + random-minute placebo | yes |
| `scripts/stage3_concentration.py` | strike overlap, cross-book correlation, exit clustering | yes |
| `scripts/stage4_diversify.py` | Arm C data: alt-strike / alt-entry replay | yes |
| `scripts/stage5_wings.py` | Arm B: staleness+liquidity audit, then economics | yes |
| `scripts/stage6_analyse_c.py` | Arm C portfolios + random-leg placebo | yes |
| `results/stage0_recon.txt` | the reconciliation + per-day peak/give-back table | yes |
| `results/trail_grid.csv` | 139 Arm-A variants | yes |
| `results/wing_audit.txt`, `results/wing_grid.csv` | Arm B audit + economics | yes |
| `results/concentration.txt`, `results/diversify_summary.txt` | Arm C | yes |
| `results/sleeve_days.csv` | 149 replayed sleeve-days | yes |
| `results/book_minute.csv.gz`, `results/wing_minute.csv.gz`, `results/trail_daily.csv`, `results/diversify_cells.csv`, `results/stage0_live_portfolio.csv` | heavy intermediates | **NO — gitignored** |

**Regeneration** (VPS, `/home/arun/quantifyd`, READ-ONLY on all DBs, ~6 min total):

```bash
nice -n 15 python3 research/125_portfolio_profit_trail/scripts/stage1_build_book.py    # ~3.5 min
nice -n 15 python3 research/125_portfolio_profit_trail/scripts/stage4_diversify.py      # ~2 min
for s in stage0_live_recon stage2_trail_sweep stage3_concentration stage5_wings stage6_analyse_c; do
  nice -n 15 python3 research/125_portfolio_profit_trail/scripts/$s.py
done
```

**Reproducibility stamp:** `options_data.db` snapshot 2026-08-25 (88 recorded days from
2026-04-20); `nas_916_atm{,2,4}_trading.db` MTM through 2026-08-25; live rules from
`backtest_data/csl_paper_config.json` frozen 2026-08-13T14:17. Cost model: SLIP_ENTRY 0.0,
SLIP_TIME 0.178, SLIP_STOP 6.548 pt/leg-side + exact Zerodha rate card. Suite rescaled to
2 lots/system. Placebo seed 20260825.
