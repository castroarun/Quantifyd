# Stock Short Strangles Hedged with INDEX Wings (NIFTY/BNF) — research/128

STATUS: RUNNING — G1 launched 2026-08-26 night

## 1. Headline

Can the r/127 C1 book replace its per-stock wings with **cheap index wings**
(NIFTY for most stocks, BANKNIFTY for banks) sized to the stock position's
notional — keeping the crash protection while pocketing the wing-cost saving?

## 2. The Ask

**What you asked (Arun, 2026-08-26):** "Instead of buying the stock wings, buy
NIFTY wings equivalent to the stock's lots — NIFTY wings are inexpensive
compared to stocks. BNF wings for banking stocks. Consider monthly/biweekly/
weekly NIFTY wings. Roll out a study."

**What we're testing:** on the SAME 628 liquid C1 entries (45→21 DTE, ±2.5%
stock strangle, TP50, no stop), replace the ±7% stock wings with long index
wings (same monthly expiry, notional-matched units = stock qty × stock spot ÷
index spot), at index-OTM {3%, 5%, 7%}, BNF for the 10 bank names; plus a
NAKED control (no wings at all) to price what any wing is worth. Success
metric: net %S0/trade with t-stat, per-trade AND tail (p05/p01) — and an
honest margin note (index wings do NOT reduce stock-leg SPAN at the broker).

**G0 hypothesis:** wings in r/127 are crash insurance. The crash component of
a stock's tail is beta×(index crash) + idiosyncratic gap. Index wings insure
the first term at a fraction of the premium (lower IV, no single-stock event
pricing); the second term goes UNINSURED. If most of the historical tail was
systemic (2020-03, 2024-06, 2026 vol events), index wings keep most of the
protection at much lower cost. If the tail is idiosyncratic (earnings gaps —
HDFCBANK Jul-26 −10% on flat NIFTY), this trade is strictly worse. The NAKED
control separates wing value from wing cost. Counterparty: index option
sellers (cheapest vol in the market). Decay risk: none specific.

**Known costs/risks stated up front:**
- **Margin regression:** naked stock strangle needs SPAN+exposure with NO
  hedge benefit (Kite nets only same-underlying) — likely 1.5-2.5× the condor
  margin. Even a P&L win can lose on return-on-margin. Measured at G4 via
  basket API.
- **Idiosyncratic gaps uninsured** — per-trade p01 expected to worsen; the
  question is by how much and whether the saved premium pays for it.
- Weekly/biweekly wing tenors (rolling) deferred to G2 — roll costs and
  weekly-expiry availability (NIFTY weeklies only) need their own grid.

**Falsification:** if no index-wing variant beats C1 (stock wings) on net/trade
at comparable tail, OR the tail degradation exceeds what sizing can absorb
(p01 worse than naked-within-10%), verdict NO EDGE vs C1 and the idea is
CONCLUDED; wings stay per-stock.

## 3. The Base

- Entries/exits: identical to r/127 C1 (45→21 DTE, ±2.5% shorts nearest traded,
  ATM vol ≥100; TP 50% of net credit; time exit 21 DTE; no stop). Same universe,
  same liquidity gate on the STOCK legs; index wings must also have traded.
- Wings: long CE+PE on NIFTY (BANKNIFTY for AXISBANK/BANKBARODA/FEDERALBNK/
  HDFCBANK/ICICIBANK/IDFCFIRSTB/INDUSINDBK/KOTAKBANK/PNB/SBIN), same monthly
  expiry as the stock, strikes nearest index_spot×(1±w), units notional-matched
  (fractional units in G1 — lot-rounding at portfolio stage, stated caveat).
- Costs: 0.5%/side stock legs, 0.25%/side index legs + same taxes model.
- Data: nse_options_bhav (stocks + NIFTY + BANKNIFTY, real EOD), daily marks.

## 4. Plan (G1 grid)

| Config | Wings |
|---|---|
| NAKED | none (control — prices the insurance) |
| NW3 / NW5 / NW7 | NIFTY wings at ±3/5/7% of NIFTY spot (BNF for banks) |
| C1 reference | ±7% stock wings (from r/127 phase_b2, not re-run) |

5 configs × 81 symbols ≈ 15 min. G2 (if G1 passes): beta-adjusted units,
weekly/biweekly rolled wings, book-level pooled overlay (ONE index wing pair
for all 10 slots — likely the practically superior construction), margin
measurement.

## 5. Status log

| When | Event |
|---|---|
| 2026-08-26 night | STATUS written (sections 1–4), G1 runner launched on VPS |

## 6. Crash recovery

Runner: `research/128_stock_shorts_index_wings/scripts/run_g1.py` on VPS,
resume-safe per (config,symbol), output `results/g1_trades.csv`, log
`results/g1.log`. Analyzer: `scripts/analyze_g1.py` (compares vs r/127
phase_b2 C1 rows). Relaunch: same nohup pattern as r/127 phase runners.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| this STATUS | crash-recovery source | yes |
| scripts/run_g1.py, analyze_g1.py | G1 runner + analyzer | yes |
| results/g1_trades.csv | per-trade output | if small |
| results/RESULTS.md | verdict | yes |

## 8. Findings

(pending)

## VERDICT (2026-08-26): CONCLUDED — index wings refuted (idiosyncratic tails). See results/RESULTS.md.
STATUS: DONE
