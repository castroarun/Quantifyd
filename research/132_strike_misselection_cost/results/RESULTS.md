# research/132 — Strike Mis-selection: What the Spot-vs-Forward Gap Cost

## **VERDICT: NO NET P&L COST — BUT A REAL, SYSTEMATIC RISK-CONTROL DEFECT**

Three findings, in order of how much they matter:

1. **It cost nothing. If anything it paid.** Replaying every CSL entry at the
   forward-snapped strike makes the book **worse by Rs 24,992** over 37 replayable entries
   (mean −Rs 1,000 per mis-struck trade, **t = −1.87, n = 25 — indistinguishable from zero**).
   The defect was not expensive. Say that plainly.
2. **But the direction was not random — it was systematically SHORT the index**, and roughly
   **half the P&L these books recorded was that unchosen bet**. On the 37 mis-struck CSL
   entries the accidental delta contributed **+Rs 41,395**, positive on 29 of 37, and its
   absolute size was a **median 49% of the booked P&L on the same trade**. A "delta-neutral"
   book was running a directional position of the same order as its entire intended edge. It
   happened to be short into a falling tape.
3. **NIFTY is NOT immune — that prior is refuted.** NIFTY mis-struck on **36.2%** of all
   recorded minutes and on **72.2%** of the CSL entries actually taken. Its basis is smaller in
   points than SENSEX's, but its strike grid is half as wide, so relative to the grid the two
   are the same order of magnitude (median |offset| / step: NIFTY 0.28, SENSEX 0.45).

The fix (`019ae8f`) was correct and remains worth having — not because it recovers rupees, but
because it removes an unchosen directional exposure worth ~50% of the book's P&L. **This is a
risk-control fix that should be judged on risk, not on return.**

---

## 0. Reconciliation gate — passed, with a stated error bar

Before interpreting anything: the 1-minute chain replay at the **actual** strike, compared to
what the daemon booked from live 5-second polls.

| | TOUCH | DWELL2 (headline) |
|---|---|---|
| n | 51 | 51 |
| credit \|err\| median | 1.15 pt (p90 6.08) | 1.15 pt (p90 6.08) |
| **exit-reason match** | **49/51 = 96%** | **49/51 = 96%** |
| gross Rs error, median | +97 | +78 |
| gross Rs error, p10..p90 | −357 .. +574 | −675 .. +539 |
| median \|relative\| error | 13.9% | 18.7% |
| sign agreement | 94% | 94% |

**Match rate: 96% on exit reason, 94% on sign, median 14–19% on magnitude.** Good enough to
quote rupees with a stated error bar; not good enough to quote them to the rupee. Every P&L
figure below carries roughly a ±15% modelling band from this source alone, on top of its
sampling noise.

---

## 1. The gap is a genuine forward basis, not a bad feed

The decisive test is expiry convergence. At DTE0 the synthetic forward **must** collapse onto
the cash index. It does — on both venues, in every hour of the DTE0 session:

| DTE0 hour | NIFTY median offset | SENSEX median offset |
|---|---|---|
| 09 | +0.7 | +3.1 |
| 11 | +1.5 | +0.2 |
| 13 | +1.6 | −0.2 |
| 15 | +2.4 | −3.0 |

The recorded spot is sound. The gap is carry-minus-dividends, and it behaves exactly as a
forward basis should: **monotone in time-to-expiry, zero at expiry, negative in dividend
season.** Two independent sources agree on the level — on 2026-08-26 the CSL record implies a
SENSEX forward of 78,004.5 and the NAS executor's own entry implies 78,016.5, against a cash
print of 77,894.

An internal consistency check is built in: the forward is read at three adjacent strikes and
the readings agree to a **median 0.60 pt (NIFTY) / 1.65 pt (SENSEX)**. Put-call parity holds;
this is not LTP noise.

---

## 2. Mis-strike frequency

### Across all recorded minutes — 61,142 minutes, 85 days per venue, 2026-04-20 → 2026-08-26

| | NIFTY (step 50) | SENSEX (step 100) |
|---|---|---|
| median offset | +9.2 | +37.4 |
| median \|offset\| | 13.8 | 44.7 |
| **median \|offset\| / step** | **0.28** | **0.45** |
| p05 .. p95 | −19.3 .. +58.6 | −40.5 .. +163.1 |
| **MIS-STRIKE RATE** | **36.2%** | **50.3%** |
| 2 or more steps off | 2.6% | 7.4% |

By trading-DTE — the basis, and therefore the defect, scales with tenor:

| DTE | NIFTY median | NIFTY mis% | SENSEX median | SENSEX mis% |
|---|---|---|---|---|
| 0 | +1.4 | 11.9% | +0.4 | 18.5% |
| 1 | +10.8 | 32.3% | +25.9 | 46.7% |
| 2 | +16.6 | 45.6% | +40.7 | 45.0% |
| 3 | +10.5 | 37.1% | +66.0 | 67.0% |
| 4 | +22.6 | 55.8% | +89.6 | 78.7% |

**The 09:16 full-day books are the most exposed** — they enter at the widest DTE the venue
offers, where the mis-strike rate runs 56% (NIFTY DTE4) to 79% (SENSEX DTE4).

### On the trades actually taken

| Population | Venue | n | mis-struck | rate | steps off |
|---|---|---|---|---|---|
| **CSL (the defective path)** | NIFTY | 36 | 26 | **72.2%** | −1:1 0:10 +1:21 +2:3 +3:1 |
| **CSL (the defective path)** | SENSEX | 15 | 12 | **80.0%** | 0:3 +1:8 +2:4 |
| NAS pre-snap (< 2026-06-01) | NIFTY | 119 | 51 | 42.9% | −2:1 −1:10 0:68 +1:40 |
| **NAS post-snap (≥ 2026-06-01)** | NIFTY | 374 | 37 | **9.9%** | −1:9 0:337 +1:28 |
| NAS post-snap | SENSEX | 73 | 13 | 17.8% | −1:9 0:60 +1:4 |

CSL's rate (72–80%) far exceeds the all-minutes rate (36–50%) for two compounding reasons: its
books cluster at 09:16 on high-DTE days, and its recorded window (14–27 Aug 2026) is the widest
basis month in the sample.

**Per CSL book:**

| Book | Venue | n | mis | rate | steps off |
|---|---|---|---|---|---|
| CSL30F_NIFTY | NIFTY | 7 | 6 | 85.7% | 0:1 +1:5 +2:1 |
| CSL30F_SENSEX | SENSEX | 8 | 6 | 75.0% | 0:2 +1:4 +2:2 |
| CSL30F_SENSEX_WED | SENSEX | 1 | 1 | 100.0% | +1:1 |
| CSL_TIMEB_SENSEX | SENSEX | 6 | 5 | 83.3% | 0:1 +1:3 +2:2 |
| CSL_TIMEB_NIFTY | NIFTY | 5 | 2 | 40.0% | 0:3 +1:2 |
| CSL_TIMEB2_NIFTY | NIFTY | 2 | 1 | 50.0% | 0:1 +1:1 |
| CSL_TIMEB_NIFTY_MON | NIFTY | 1 | 0 | 0.0% | 0:1 |
| NAS_COMB20 | NIFTY | 7 | 4 | 57.1% | 0:3 +1:4 |
| NAS_C20_TRAIL | NIFTY | 7 | 6 | 85.7% | 0:1 +1:5 +2:1 |
| NAS_C20_SHIFT | NIFTY | 7 | 7 | 100.0% | −1:1 +1:4 +2:1 +3:1 |

### The NAS control validates the method — and the fix

`nas_atm_executor.py` has carried the forward snap since `57eb8c2` (2026-06-01).

| Venue | Era | n | mis-strike rate |
|---|---|---|---|
| NIFTY | pre < 2026-06-01 | 119 | **42.9%** |
| NIFTY | post ≥ 2026-06-01 | 374 | **9.9%** |
| SENSEX | post ≥ 2026-06-01 | 73 | 17.8% |

**A 4.3× reduction across the ship date.** The snap fired and moved the strike on 30.7% of
post-snap NIFTY entries and 56.2% of SENSEX entries. The residual 10–18% is the expected
combination of quote-race edge cases (the snap bails out if the forward strike has no usable
quote) and the timing difference between the executor's live tick and our 1-minute chain — it
is not evidence the fix is broken. SENSEX's higher residual tracks its larger basis.

---

## 3. The unintended delta — the finding that matters

Net delta of a short straddle at `K` when the forward is `F`, from Black-76 with σ inverted
from the observed combined premium. **Negative = the book was accidentally SHORT the index.**

| Population | n | median \|Δ\| | p90 \|Δ\| | median \|Rs/100pt\| | p90 | max | signed sum |
|---|---|---|---|---|---|---|---|
| **CSL NIFTY** | 35 | **0.185** | 0.232 | **Rs 2,750** | 5,298 | 8,854 | **−67,768** |
| **CSL SENSEX** | 15 | 0.116 | 0.194 | Rs 1,007 | 2,720 | 4,287 | **−15,491** |
| NAS NIFTY post-snap | 374 | 0.041 | 0.096 | Rs 508 | 2,432 | 15,941 | −93,318 |
| NAS NIFTY pre-snap | 119 | 0.061 | 0.159 | Rs 570 | 1,523 | 7,476 | −51,822 |
| NAS SENSEX post-snap | 73 | 0.033 | 0.066 | Rs 142 | 346 | 568 | +12 |

**CSL NIFTY carried 4.5× the delta of the fixed NAS book on the same index** (0.185 vs 0.041).
Its worst single entry — CSL_TIMEB_NIFTY, 2026-08-21, K = 24250 against a forward of 24295.4 —
was worth **Rs 8,854 per 100 index points**, on a book whose whole intended edge is theta.

Per CSL book, rupees per 100 index points (signed; − = accidentally short):

| Book | Venue | n | mis | median | p90 \|·\| | max \|·\| |
|---|---|---|---|---|---|---|
| CSL_TIMEB2_NIFTY | NIFTY | 2 | 1 | **−3,473** | 3,922 | 3,922 |
| NAS_C20_SHIFT | NIFTY | 6 | 6 | −2,577 | 3,019 | 5,594 |
| NAS_C20_TRAIL | NIFTY | 7 | 6 | −2,444 | 2,945 | 3,019 |
| NAS_COMB20 | NIFTY | 7 | 4 | −2,436 | 3,019 | 6,011 |
| CSL30F_NIFTY | NIFTY | 7 | 6 | −2,425 | 2,947 | 3,019 |
| CSL_TIMEB_SENSEX | SENSEX | 6 | 5 | −1,856 | 2,720 | 4,287 |
| CSL_TIMEB_NIFTY | NIFTY | 5 | 2 | −1,145 | 8,854 | 8,854 |
| CSL30F_SENSEX_WED | SENSEX | 1 | 1 | −1,026 | 1,026 | 1,026 |
| CSL30F_SENSEX | SENSEX | 8 | 6 | −593 | 1,007 | 1,015 |
| CSL_TIMEB_NIFTY_MON | NIFTY | 1 | 0 | −280 | 280 | 280 |

**Every book's median is negative.** This is the single most important structural point in the
study: **the mis-strike was directionally BIASED, not a coin flip.** Because the basis is
positive on 57 of 85 NIFTY days and 65 of 85 SENSEX days, `round(spot/step)` systematically
lands *below* the forward, which makes the sold call the richer leg and leaves the book
**persistently short the index**. A defect that flipped a fair coin would wash out over time.
This one does not: it is a standing short-delta tilt that grows with DTE.

### What it was actually worth in rupees

Attribution: `net_delta × realised index move over the holding period × qty`. First-order —
gamma makes the true figure **worse** than this on large moves, so it is a lower bound.

| Venue | n | sum | mean | median | p10 | p90 |
|---|---|---|---|---|---|---|
| NIFTY, all | 35 | +25,200 | +720 | +313 | −780 | +2,939 |
| NIFTY, mis-struck | 25 | +27,735 | +1,109 | +726 | −780 | +3,103 |
| SENSEX, all | 15 | +12,134 | +809 | +673 | −1,014 | +2,962 |
| SENSEX, mis-struck | 12 | +13,660 | +1,138 | +1,004 | −996 | +2,962 |

**All mis-struck CSL entries: n = 37, total +Rs 41,395**, positive on 29 and negative on 8.
Per trade, |delta P&L| median Rs 996, p90 Rs 2,962, max Rs 6,299.

> **|delta P&L| was a median 49% of the booked P&L on the same trade.**

The index fell over the window (NIFTY median −23.3 pts per holding period, SENSEX −44.4). The
accidental short tilt was therefore on the right side of the tape. **That is luck, not edge**,
and it is the whole reason the P&L answer below comes out benign.

Worst single accidental bets:

| Book | Day | steps | Δ | move | delta P&L | booked P&L |
|---|---|---|---|---|---|---|
| NAS_C20_SHIFT | 2026-08-25 | −1 | +0.430 | +112.6 | +6,299 | −1,535 |
| CSL30F_SENSEX | 2026-08-24 | +2 | −0.168 | −414.1 | +4,171 | −1,678 |
| CSL_TIMEB2_NIFTY | 2026-08-24 | +1 | −0.302 | −87.1 | +3,416 | −654 |
| NAS_C20_TRAIL | 2026-08-24 | +1 | −0.188 | −126.9 | +3,103 | −3,442 |
| CSL30F_SENSEX_WED | 2026-08-26 | +1 | −0.171 | −288.7 | +2,962 | +2,115 |
| NAS_COMB20 | 2026-08-24 | +1 | −0.187 | −120.6 | +2,939 | −4,899 |

---

## 4. The counterfactual P&L — what the fix would have changed

Every CSL entry replayed twice on the same 1-minute chain, under that book's own rule, with the
measured outcome-aware cost model. The management arms (`NAS_C20_TRAIL`, `NAS_C20_SHIFT`) are
excluded: their post-stop re-entry path cannot be honestly re-struck.

| | TOUCH | **DWELL2 (headline)** |
|---|---|---|
| all entries, n | 37 | 37 |
| total difference (fwd − actual) | **−Rs 22,506** | **−Rs 24,992** |
| mis-struck only, n | 25 | 25 |
| total difference | −Rs 22,506 | −Rs 24,992 |
| mean per mis-struck trade | −Rs 900 | −Rs 1,000 |
| median | −Rs 505 | −Rs 505 |
| sd | 2,523 | 2,673 |
| range | −8,133 .. +3,985 | −8,667 .. +3,985 |
| p10 .. p90 | −4,560 .. +1,530 | −4,560 .. +1,530 |
| forward arm better on | 10 of 25 (40%) | 8 of 25 (32%) |
| **t on the mean difference** | **−1.78** | **−1.87** |

**t = −1.87 on n = 25 — indistinguishable from zero.** The dispersion (sd Rs 2,673 against a
mean of −Rs 1,000) dwarfs the central estimate. The honest reading is: *over this window the
mis-strike happened to make about Rs 25,000, and we cannot distinguish that from noise.*

Per book, DWELL2:

| Book | Venue | n | mis | actual net | fwd net | difference | per trade |
|---|---|---|---|---|---|---|---|
| CSL30F_SENSEX | SENSEX | 8 | 6 | +12,620 | −1,579 | **−14,199** | −1,775 |
| CSL_TIMEB_SENSEX | SENSEX | 6 | 5 | +13,113 | +7,163 | −5,950 | −992 |
| CSL_TIMEB2_NIFTY | NIFTY | 2 | 1 | +468 | −6,140 | −6,608 | −3,304 |
| CSL30F_SENSEX_WED | SENSEX | 1 | 1 | +2,425 | −587 | −3,012 | −3,012 |
| CSL30F_NIFTY | NIFTY | 7 | 6 | −5,533 | −6,087 | −554 | −79 |
| CSL_TIMEB_NIFTY_MON | NIFTY | 1 | 0 | +1,752 | +1,752 | 0 | 0 |
| CSL_TIMEB_NIFTY | NIFTY | 5 | 2 | +9,404 | +9,950 | +546 | +109 |
| NAS_COMB20 | NIFTY | 7 | 4 | −6,818 | −2,027 | **+4,791** | +684 |
| **TOTAL (non-mgmt)** | | **37** | **25** | **+27,431** | **+2,445** | **−24,986** | |
| _NAS_C20_SHIFT (mgmt, excluded)_ | NIFTY | 7 | 7 | +16,025 | −5,716 | −21,741 | −3,106 |
| _NAS_C20_TRAIL (mgmt, excluded)_ | NIFTY | 7 | 6 | −4,583 | −5,716 | −1,133 | −162 |

**The counterfactual is a re-simulation, not a re-pricing.** The forward-snapped straddle is a
different instrument: it collects a different credit, so its stop sits at a different level.
Exit reason flipped on 2 of 37 entries — the alternative straddle stopped where the real one
held, or the reverse. The book-level totals disagree in sign per book (SENSEX loses, NAS_COMB20
gains), which is exactly what one draw of 37 path-dependent coin flips looks like.

---

## 5. Offset stability — swings day-to-day, steady within a day

| | NIFTY | SENSEX |
|---|---|---|
| per-day median offset spans | −37 .. +75 | −95 .. +173 |
| p05 .. p95 of per-day medians | −17 .. +54 | −30 .. +138 |
| median **within-day** IQR | 9.1 pts | 25.1 pts |
| between-day spread ÷ within-day IQR | **7.9×** | **6.7×** |
| days whose median offset alone exceeds half a step | 22 / 85 = **26%** | 40 / 85 = **47%** |
| sign | 57 pos / 28 neg | 65 pos / 20 neg |

Monthly median-of-medians:

| Month | NIFTY | SENSEX |
|---|---|---|
| 2026-04 | +8.7 | +35.5 |
| 2026-05 | +9.3 | +40.6 |
| 2026-06 | +9.2 | +52.5 |
| 2026-07 | **−1.1** | **−8.7** |
| 2026-08 | **+31.4** | **+82.4** |

**It is a slowly-moving level, not intraday noise.** Within a session it barely budges (IQR 9 /
25 pts); across days and months it swings by 7×–8× that, and it **changes sign** — the July
trough is dividend season pulling the forward below spot, and August's spike is that unwinding.

Two consequences:

- **The fix is sufficient.** The forward snap is self-calibrating: it reads whatever level the
  chain is using at that minute and does not care whether the basis is +5 or +170.
- **The historical mis-strike rate is not a forecast.** On 47% of SENSEX days the median offset
  alone exceeded half a step, meaning the strike was structurally wrong *all day*. In a
  low-basis month the same code would have looked almost fine. Anyone reasoning about "how
  often this bites" from a single month will be wrong in either direction.

---

## 6. Does NIFTY have the same gap? — **YES, materially. The prior is refuted.**

The prior came from one observation (27-Aug COMB: K = 24300, CE 115.28 / PE 90.41 → forward
K+24.87, which rounds correctly). That was a **near miss**: 24.87 points out of a 25-point
half-step. One more point of basis and it flips.

| | NIFTY | SENSEX | Verdict |
|---|---|---|---|
| median \|offset\| in points | 13.8 | 44.7 | SENSEX 3.2× bigger — the source of the "NIFTY is fine" intuition |
| step | 50 | 100 | but the grid is half as wide |
| **median \|offset\| / step** | **0.28** | **0.45** | **same order of magnitude** |
| mis-strike, all minutes | **36.2%** | 50.3% | NIFTY is 72% of SENSEX's rate |
| mis-strike, CSL entries taken | **72.2%** | 80.0% | essentially the same |
| median unintended \|Rs/100pt\|, CSL | **Rs 2,750** | Rs 1,007 | **NIFTY is 2.7× WORSE in rupees** |

NIFTY's *rupee* exposure is the larger of the two, because the CSL NIFTY books run 130–520 qty
against SENSEX's 60–160 and a NIFTY lot is 65 against SENSEX's 20. **NIFTY was the bigger
problem, not the smaller one.** The fix needed to ship on both, and it did.

---

## 7. How the seven deadly sins are controlled

| Sin | Control |
|---|---|
| Look-ahead | Every forward is read from the same minute's chain; no future bar informs the strike choice or the stop. |
| Survivorship | The population is the complete recorded set — 617 audited entries, no selection. Guards reject days, never trades. |
| Overfitting / multiple testing | **No grid, no search.** One counterfactual per entry, fixed a priori by the already-shipped fix. Nothing was tuned. |
| Cost neglect | Measured outcome-aware model from r/122 (stop exits +6.548 pt/leg-side vs time +0.178), applied per path so a changed exit reason changes the cost. Retired flat Rs250/lot not used. |
| Regime dependence | The single biggest caveat, and it is **stated**: the CSL window is one month, and it is the widest-basis month in the sample. §5 gives the month-by-month drift so the reader can discount accordingly. |
| Correlation / single factor | The books are near-clones on the same two indices — this is why per-book totals are not independent evidence and only the pooled t is quoted. |
| Capacity / shortability | Not applicable — no change to size, universe or instrument is proposed. |

---

## 8. Honest limitations

1. **n = 25 mis-struck replayable entries over 10 trading days.** The P&L conclusion is
   low-power by construction; the CSL state file simply does not go back further. The
   frequency and delta findings rest on 61,142 minutes and 617 entries and are solid.
2. **One draw.** The counterfactual straddle has its own path. Exit reason flipped on 2 of 37.
   With a different month the sign of the Rs 25,000 could plausibly reverse.
3. **±15% modelling band** on every rupee figure, from the 1-minute-vs-5-second reconciliation.
4. **Delta attribution is first-order.** Gamma on a short straddle makes the true contribution
   *worse* than the linear estimate on large moves, so +Rs 41,395 is a lower bound on how much
   of the P&L was directional.
5. **2026-08-27 is missing** from every stage — the recorder's partial-session guard rejects
   today. That is the day the fix shipped, so it is the right day to lose.
6. **Strangle legs excluded.** ATM2/ATM4 pairs whose CE and PE sit on different strikes are not
   straddles and were dropped (that is why nas_916_atm4 audits only 17 of 76 pairs). The
   mis-selection question for a deliberately-offset strangle is a different question.

---

## 9. What this changes

**Nothing operationally — the fix already shipped.** But three things are now on the record:

1. The forward snap should be treated as a **risk control**, not a P&L improvement. Judged on
   return over this window it looks like a Rs 25,000 mistake; judged on risk it removed a
   standing short-delta tilt worth ~50% of the book's P&L.
2. **Any future straddle/strangle entry path must read the forward, not the index.** The NAS
   before/after (42.9% → 9.9%) is the proof it works. This should be a checklist item for any
   new options book, not a per-book rediscovery.
3. **The residual 10–18% post-snap mis-strike rate is worth one look.** It is consistent with
   the snap's fail-safe (keep the spot strike when the forward strike has no usable quote) plus
   chain-vs-live timing, but it has not been separately confirmed. Low priority — the exposure
   at that residual is a median Rs 508/100pt on NIFTY and Rs 142 on SENSEX, an order of
   magnitude below the CSL numbers.

---

## Files

| File | Rows | Purpose |
|---|---|---|
| `results/offset_atlas.csv` | 61,142 | per-minute spot / forward / offset / mis-strike, both venues |
| `results/entry_audit.csv` | 617 | every recorded entry: actual K, forward K, net delta, Rs/100pt |
| `results/replay.csv` | 102 | actual-K reconciliation + forward-K counterfactual, TOUCH and DWELL2 |
| `results/delta_attrib.csv` | 50 | rupees attributable to the unintended delta |
| `results/aggregate.txt` | — | full Stage D console output |
| `results/delta_attrib.txt` | — | full Stage E console output |
| `results/stage.log` | — | per-day progress log |

Period: chain 2026-04-20 → 2026-08-26 (85 days/venue after guards); CSL records 2026-08-14 →
2026-08-27; NAS DBs 2026-03-25 → 2026-08-27. All DBs read-only. Fix under study: `019ae8f`.
