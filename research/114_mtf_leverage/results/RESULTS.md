# RESULTS — MTF leverage on the live momentum book (research/114)

**VERDICT: 2.5x is SURVIVABLE and materially raises return — 67.3% CAGR vs 32.9% — but it is the
same edge magnified, not new alpha (Calmar FALLS 1.50 -> 1.30), and it costs a -52% drawdown.
3.0x is NOT recommended: it survives the backtest on a knife-edge, not with margin.**

Engine unchanged from research/104 (`run_lev62`), imported not copied. Rules: rsblend, top-8,
top-22 buffer, Donchian-15, weekly NIFTYBEES-100SMA gate. 2006-2026 incl. 2008 and 2020,
daily-marked, net 0.3% round-trip, leverage applied only while the gate is risk-on.

## The frontier at the REAL Zerodha MTF rate (14.6%/yr)

| Leverage | CAGR | Max DD | Sharpe | Calmar | Margin calls |
|---|---|---|---|---|---|
| 1.0x (own cash) | 32.9% | -22.0% | 1.78 | **1.50** | 0 |
| 1.3x | 40.4% | -28.6% | 1.68 | 1.41 | 0 |
| 1.6x | 47.5% | -35.0% | 1.61 | 1.36 | 0 |
| 2.0x | 56.5% | -43.1% | 1.54 | 1.31 | 0 |
| **2.5x** | **67.3%** | **-52.0%** | 1.48 | 1.30 | **0** |
| 3.0x | 77.6% | -59.7% | 1.45 | 1.30 | 0 |

NIFTYBEES buy-and-hold over the same period: 11.6% CAGR, -59.7% DD, Calmar 0.19.

## What the rate actually costs

research/104 financed at 10.5%; the runbook puts real Zerodha MTF at ~14.6% (0.04%/day).

| Leverage | CAGR @10.5% | CAGR @14.6% | Give-up |
|---|---|---|---|
| 1.3x | 40.7% | 40.4% | -0.3pp |
| 2.0x | 58.2% | 56.5% | -1.7pp |
| 2.5x | 70.3% | 67.3% | **-3.0pp** |
| 3.0x | 82.0% | 77.6% | -4.4pp |

The rate matters less than it looks because leverage is only carried while the gate is risk-on
(~75% of months) and only on the borrowed slice. It is NOT free: 3pp/yr at 2.5x.

## Why zero margin calls is real — and where it stops being real

**CORRECTION (Arun caught this).** The first version of this study justified the 3.0x risk by
asserting "a momentum basket moves 1.1-1.4x the index". That was an assumption, and it was wrong —
the same report shows the book drawing down -22% against the index's -59.7%, which should have
prompted measuring it rather than guessing. Measured over the 3,051 days the book was actually
HOLDING (2006-2026):

| Measure | Value |
|---|---|
| Beta while holding, all days | **0.91** |
| Beta while holding, index days <= -2% | **0.56** |
| Worst single day of the held basket | **-14.52%** (2008-01-21) |

So the basket is *defensive* against the index, not amplifying — on bad index days it falls barely
half as much. The conclusion below survives anyway, but for a completely different reason.

**The risk is not market beta. It is idiosyncratic concentration.**

| Date | Basket | Index | Ratio | Names held |
|---|---|---|---|---|
| 2008-01-21 | -14.52% | -9.44% | 1.5x | 1 |
| **2024-06-04** | **-12.74%** | -2.97% | **4.3x** | **8** |
| 2023-09-12 | -7.81% | -0.09% | 90x | 8 |

Eight momentum names gap together on news — the 2024 election result being the clearest case, where
the book fell 4.3x the index. An index-beta argument would never have found this.

**What that does to each leverage, fully deployed, on the worst day with all 8 held (2024-06-04):**

| Leverage | equity/gross after the day | Verdict |
|---|---|---|
| 2.0x | 42.7% | survives easily |
| **2.5x** | **31.2%** | **survives, 6.2pp above the 25% line** |
| 3.0x | **23.6%** | **MARGIN CALL** |

**This is why the engine's "0 margin calls at 3.0x" must not be taken at face value.** It is true
of the path the book actually took, because on those specific days it was not fully deployed —
2008-01-21 had just ONE name held. That is timing luck, not a designed buffer. Run 3.0x fully
deployed into 2024-06-04 and it is called.

The gate still does the heavy lifting for the big crashes — on 5 of the 6 worst INDEX days the book
was already in cash (28 days before the 2020 low, 73 before 2008). But the gate is a weekly,
index-level signal: it cannot protect against a single-day idiosyncratic hit to eight concentrated
momentum names, which is precisely what kills a 3.0x book.

## The honest read

1. **It is leverage, not edge.** Calmar falls 1.50 -> 1.30. You are buying return with risk at a
   slightly worse exchange rate, which is exactly what leverage should do.
2. **-52% is the real question**, not 67%. The book halves at some point in a 20-year cycle. Every
   backtested recovery assumes you did not stop it at the bottom.
3. **The gate is the ONLY thing making this safe.** Disable or widen it, and 2.5x becomes ruinous.
   It must never be treated as a tunable once leverage is on.
4. **Untested, and it matters:** per-stock MTF factors. 2.5x is Arun's stated conservative floor,
   not something derived — Kite does not expose the factors. If any held name carries a lower
   factor, achievable leverage is lower, never higher. Also unmodelled: intraday square-off (the
   engine only checks at daily close), pledge haircuts, and the broker's discretionary right to
   liquidate early.
5. **Sequencing:** the book is currently unhedged and below the size where the NIFTY put hedge can
   be sized (research/105). Leverage first, protection later is the wrong order.

## Recommendation

If leverage is wanted, **1.6x-2.0x is the defensible band** (47.5-56.5% CAGR, -35 to -43% DD,
Calmar 1.31-1.36) — it keeps a 33% one-day buffer to a margin call rather than 20%. **2.5x is
defensible only with the gate untouched and a written commitment to sit through -52%.** 3.0x adds
10pp of CAGR for a 0.2pp safety margin on the one exposed crash day in 20 years; that is not a
trade, it is a coin flip with the book.
