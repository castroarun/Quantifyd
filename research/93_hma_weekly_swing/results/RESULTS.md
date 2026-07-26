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
