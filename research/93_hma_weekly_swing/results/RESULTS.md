# RESULTS — HMA 30/44 + MACD(21,39,9) + RSI(9/3/21W) Weekly Retracement-Reversal Swing (research/93)

**Verdict: SIGNAL (not investable as tested).** The triple-aligned weekly entry (Nitin
Hulaji, Market Aur Main Ep.5) has a genuine per-trade selection edge — **+3.2%/trade net
over a year-matched random-entry control (Welch t = 7.15)**, robust across all 27 parameter
cells, both decade-halves, super-winner exclusion, and costs. But as a book it FAILS G4:
a 20-slot, 5%-of-NAV portfolio returns **6.7% CAGR (Sharpe 0.47, MaxDD −48.9%)** vs
NIFTYBEES buy-and-hold **12.75% (0.73, −58.0%)** on the same window, and the
idle-cash-in-index variant is worse (8.93% CAGR, MaxDD −63.8%). The signal picks
better-than-random entries in its names; the names/timing/sizing still lose to the index.

## The tested system (locked before launch)

Weekly bars (W-FRI from daily), long-only. Entry at next week's open when at completed bar i:
(1) close reclaimed HMA30 within 4 wks and still ≤ max(HMA30, HMA44); (2) MACD(21,39,9)
histogram rising after a ≥8-bar below-zero run; (3) RSI(9) 3-SMA crossed above its 21-WMA
within 4 wks. SL = last confirmed 2/2 fractal low (0.1% buffer); target = 26-wk swing high;
52-wk time stop; stop-first same-bar; 25 bps round-trip. Universe: 629 NSE names surviving
depth/liquidity screens (from 1,628 daily symbols, 2000→2026-07). The video's daily
"+3% day → sell 10%" overlay was NOT tested (moot — the book already trails the index).

## Key numbers

| | n | gross/tr | net/tr (25bps) | win% | PF | t | hold |
|---|---|---|---|---|---|---|---|
| Setup (base 8/4/26) | 4,537 | +4.87% | +4.62% | 37.8% | 1.66 | 11.25 | 12.6 wk |
| Random control (year-matched, same exits) | 13,352 | +1.91% | +1.45% | — | — | — | — |
| **Setup − control** | | | **+3.17%** | | | **7.15** | |

- **Sub-periods:** ≤2018 diff +2.22% (t 3.50) · ≥2019 diff +4.11% (t 6.65).
- **Sweep:** all 27 cells (neg_bars 5/8/12 × recent_win 2/4/8 × target_lb 13/26/52) beat
  control by +2.6→+4.3%/tr — flat grid, no lone peak; the MACD run-length barely matters.
- **Super-winner guard:** ex-top-3 names 4.41% vs 4.62% — no dent. Cost-sens 0→50 bps: 4.87→4.37%.
- **Skew:** median trade **−6.1%**; 61% of exits are stop-outs. All profit is in the tail.
- **2020:** +49.7%/tr on 250 trades (diff +30.7%); ex-2020 the edge ≈ +1.7%/tr. 7 of 26
  years have negative diff (2008, 2015, 2018, 2019, 2022, 2024, 2026-YTD).

### G4 portfolio (why it's not a strategy)

| Book (2005→2026-07) | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| 20-slot 5%-NAV, cash idle | 6.70% | 0.47 | −48.9% | 0.14 |
| same, idle cash in NIFTYBEES | 8.93% | 0.51 | −63.8% | 0.14 |
| NIFTYBEES B&H | **12.75%** | **0.73** | −58.0% | **0.22** |

Failure mechanics: entries cluster violently after crashes (up to 52 in one week vs 20
slots) so 65% of candidates — concentrated in the best vintages — are turned away; average
exposure only ~65%; equal 5% sizing undersizes the tail winners that carry the whole edge.
The book realized far less than the population +4.6%/tr.

## Honest caveats

- **Survivorship**: universe = today's symbol list. The random-entry control shares the
  bias, but deep-retracement entries interact with survivorship more than random entries do
  (survivors by definition recovered). The true forward edge is likely smaller than +3.2%.
- Costs modeled flat 25 bps round-trip; no impact model. Fine at retail size, weekly bars.
- Slot priority was alphabetical (worst-case-honest, no cherry-pick); a ranking rule could
  do better but would be a new, tunable degree of freedom (multiple-testing risk).
- 27 cells tried → the best single cell's stats are inflated; we report the base cell and
  the grid floor, not the peak.
- Data: VPS `market_data.db` snapshot 2026-07-26; scripts in `research/93_hma_weekly_swing/scripts/`.

## Phase 2 (2026-07-27, Arun's ask): optimization for investability

**Outcome: materially improved — from "clearly loses to index" to "modestly beats index" —
but still NOT investable by the pre-set bar (MaxDD ≤ ~35%).** Verdict stays **SIGNAL**,
now with a much better exit rule on record.

**Stage 1 — exit variants (per-trade, same entries/stops):**

| Exit | n | net/tr | PF | t | hold |
|---|---|---|---|---|---|
| Target at prior swing high (as taught) | 4,537 | +4.62% | 1.66 | 11.2 | 12.6 wk |
| Trail: close < HMA44 | 4,813 | +3.48% | 1.90 | 9.5 | 6.1 wk |
| **Trail: close < prior 10-wk low (Donchian)** | 4,560 | **+11.11%** | **2.72** | **13.9** | 17.4 wk |

The r/71 lesson repeats: **never a profit target** — riding past the old high 2.4×'s
expectancy. (Also: Arun's ASIANPAINT near-miss prompted a 1%-buffer target fill test —
it LOSES −0.23%/tr net; the haircut on the 33% of clean touches outweighs rescued
near-misses. Exact-touch at the full high is the best *target* rule; the trail beats both.)

**Stage 2 — 14 portfolio replays (2005→2026 vs NIFTYBEES 12.75% / 0.73 / −58.0% / 0.22):**

Best cell: **Donchian trail, 40 slots × 2.5%, no gate → 15.04% CAGR, Sharpe 0.87,
MaxDD −51.2%, Calmar 0.29** (76% exposure, 2,123 trades). Full table:
`/tmp/hma93_g7b.log` + `results/g7_best_nav.csv`.

Key structural findings:
- **The regime gate (NIFTYBEES > 40-wk SMA) HURTS every cell** (best gated Donchian: 9.2%
  CAGR). Opposite of r/71/75 — because this is a retracement-REVERSAL system: its best
  entries (2009/2020/2023 vintages) fire BELOW the SMA. The alpha lives exactly where a
  gate blocks. Retracement systems are anti-gate; breakout/momentum systems are pro-gate.
- R:R slot-ranking never helps (contention adverse-selection isn't fixable by that rank).
- 40 slots > 20 slots for the trail book (absorbs clustering; DD −58.8→−51.2, Sharpe up).

**Why it still isn't investable:** −51% MaxDD (bar was ≤35%); excess is lumpy — positive
14/21 years but −28pp (2018), −16pp (2019), −24pp (2025); best-of-14-cells carries a
multiple-testing haircut; survivorship flatters the stock book more than the index. And at
Calmar 0.29 it is far below the live/candidate books (midcap RS120 ~1.7, r/75 momentum
~1.26) — capital would be better deployed there.

## Next levers (only if revisited)

1. **Regime gate** (NIFTY > 200DMA) — the one overlay with a track record here (research/71/75);
   would have avoided 2008/2011/2015/2018-type bleed years.
2. Slot-contention fix: rank clustered candidates (e.g. by retracement depth or liquidity)
   instead of alphabetical; or scale slots dynamically post-crash.
3. Trailing exit instead of fixed swing-high target (research/71: trailing ≫ target) —
   though that departs from the taught system.
4. The +3%-day/sell-10% overlay — untested, requires daily-path sim; pointless until the
   core book beats the index.

*Reproduce: run `scripts/run_g1_probe.py` → `run_g3_robustness.py` → `run_g2_param_sweep.py`
→ `run_g4_portfolio.py` (env `CASH_IN_BENCH=1` for the variant) on the VPS.*
