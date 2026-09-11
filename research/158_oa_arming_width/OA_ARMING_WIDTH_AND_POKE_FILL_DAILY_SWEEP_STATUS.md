# Open Alpha entries: the scanner arms the INVERSE of the backtested condition — STATUS: RUNNING

One line: the live entry scanner buys names that have **not** broken out; the backtest buys names
that **just** did. At the live spec the live condition measures **−1.4% CAGR at −81.8% drawdown**
against **40.8% at −33.9%** for the backtested one. Entry crons paused 11-Sep 11:18 IST.

## 1. The Ask

**What Arun asked (11-Sep-2026, ~11:10 IST):**

> "look at even most [names] and place the orders because few orders might not get filled. If
> something gets filled, then we'll leave out the rest. Is that not the plan?"

and, after the first findings:

> "we shud rather enter on the same day as the breakout close towards the market close - say
> 3:10 PM, for our study u can take day close as the entry price if price data before that is
> not available."

**What we are actually testing.** The width question opened a much larger one. In order:

- **Q1 (width, ANSWERED then SHELVED).** Should the scanner arm more buy-stops than free slots
  and cancel the remainder? Measured: yes on fill mechanics — but the question is void, because
  what it would arm more of is the wrong signal.
- **Q2 (the mechanic, ANSWERED).** What does the live book's resting-stop entry measure at,
  versus the published close-above-pivot entry? This is the finding.
- **Q3 (the tradeable entry, RUNNING).** The published entry cannot be traded: it decides at the
  open using that same day's closing price. Three candidate replacements, one of them Arun's:
  buy at the breakout day's close (~15:10), buy the next day at the broken pivot, or the
  published same-day open as the look-ahead control.

## 2. The Base — the defect, stated precisely

`services/oa_entry.py`, `signal()`:

```python
pivot = close.loc[:last].cummax().loc[last]      # highest close THROUGH today
prev_close = close.loc[last]                     # today's close
cand = cand[(cand['close'] < cand['pivot'])      # today's close BELOW that high
            & (cand['close'] >= BASE_DEPTH * cand['pivot'])
            & (cand['rs'] >= RS_MIN)]
```

`research/142/scripts/bluesky_replay.py`:

```python
athcp = df['close'].shift(1).cummax()            # highest close through YESTERDAY
trig  = setup & (close > athcp) & athcp.notna()  # today's close ABOVE it
```

`close < pivot` and `close > athcp` are mutually exclusive on any given day: the first condition
excludes a new closing high **by definition**, because `pivot` includes today. So the scanner can
never select a name on the day it breaks out. It selects names approaching their high and rests a
stop at it.

The register of record states the intended rule and agrees with the backtest, not the code:
*"A close above the prior all-time-high close in an RS>=70, Rs5cr/day-liquid name -> buy-stop at
the pivot next day."*

**The live behaviour is exactly modelled by the replay's `--poke-trigger` arm.** Verified by index
mapping, not assumed — with the live scan running after day *d*'s close and the replay indexing
day *i*:

| | live scanner (scan on day *d*) | replay `--poke-trigger` (day *i*) |
|---|---|---|
| pivot | `max(close[0..d])` | `athcp[i] = max(close[0..i−1])` |
| condition | `close[d] < max(close[0..d])` → *d* set no new high | `close[i−1] < athcp[i]` → *i−1* set no new high |
| fill | stop at pivot touched on *d+1* | `high[i] >= athcp[i]`, at `max(athcp[i], open[i])` |

With *d = i−1* the two are identical. The measurement is therefore of the live book, not an
analogy — and it is **generous** to it, because the replay picks the highest-RS name among those
that touched, while the live book commits the night before and takes whatever fills.

## 3. Plan

| Step | State |
|---|---|
| Q1 fill-rate by gap bucket | **DONE** |
| Q2 trigger x trail, 2x2, 30 seeds, pre-tax | **DONE** |
| Pause the two entry crons | **DONE** 11:18 IST |
| Fork the engine, add `open_next` | **DONE** — `scripts/oa_entry_mechanics.py` |
| Q3 five-arm mechanic bake-off | RUNNING |
| Correct `signal()` against the Q3 winner | NOT STARTED |
| Two-year trade ledger published on the app | NOT STARTED (Arun: "do this at last") |

## 4. Status

| Date/time (IST) | Event | Notes |
|---|---|---|
| 11-Sep 10:26 | Armed 3 slots live; book to 16/16 committed | MOREPENLAB, AEROFLEX, CYIENTDLM; the act that raised the question |
| 11-Sep ~11:05 | Q1 fill-rate DONE | 1.26M setups; the four resting orders expect **0.76 fills** |
| 11-Sep ~11:10 | Arun chose width 2x + cancel-on-fill | later shelved by Q2 |
| 11-Sep ~11:12 | **Trail claim retracted** | I reported trail-15 as an unproven deviation. Wrong: trail-15 is the after-tax paired winner, adopted 03-Sep (+1.59pp on 24/30 seeds, worst seed 30.34 vs 27.33, DD −29.1 vs −30.0). Arun caught it. `RESULTS.md` line 286 was stale and has been corrected. |
| 11-Sep ~11:14 | **Q2 DONE — the inverted condition found** | see §5 |
| 11-Sep 11:18 | Entry crons 18:50 and 09:25 PAUSED | backup `/tmp/mpf/ct.bak.20260911-111828`; 129 -> 131 lines; **exits untouched** |
| 11-Sep ~11:20 | Q3 bake-off launched | log `/tmp/mpf/modes.log` |

## 5. Findings

### Q2 — the live entry mechanic has negative measured expectancy

Live spec otherwise held constant: 16 slots @ 6.25%, −8% close stop, no market gate, 25 bps/side,
2006-01-01 → 2026-08-31, 30-seed ensemble, pre-tax.

| trail | trigger | signals | CAGR median | range | DD median | worst seed DD | win |
|---|---|---|---|---|---|---|---|
| 20 | close above pivot | 18,208 | 38.7% | 34.7 .. 44.4 | −30.8% | −41.3% | 46% |
| 20 | **touch of pivot (LIVE)** | 43,010 | **4.0%** | −4.0 .. 11.3 | **−68.6%** | −83.2% | 35% |
| 15 (live trail) | close above pivot | 18,208 | **40.8%** | 34.1 .. 48.2 | −33.9% | −47.4% | 45% |
| 15 (live trail) | **touch of pivot (LIVE)** | 43,010 | **−1.4%** | −7.3 .. 5.8 | **−81.8%** | −90.9% | — |

**2.4x the signals, 11 points of win rate gone, and the edge with it.** A stop resting at the high
fills on every intraday poke through it, and most pokes close back below. Requiring the breakout
to **hold to the close** was carrying the strategy. The faster trail makes the poke version worse,
not better, because it churns the failures harder.

### Q1 — a resting stop at the pivot usually is not touched (kept for the record)

1,257,728 setups, 2015-01-01 onward, scanner universe:

| Gap below pivot | Setups | 1d | 2d | 3d | 5d | 10d |
|---|---|---|---|---|---|---|
| 0-1% | 93,811 | 70.4% | 78.6% | 82.5% | 86.7% | 90.9% |
| 1-2% | 80,826 | 42.4% | 56.1% | 63.9% | 72.7% | 82.0% |
| 2-4% | 152,865 | 20.2% | 31.8% | 39.8% | 50.6% | 65.0% |
| 4-7% | 213,256 | 6.5% | 13.4% | 19.0% | 27.7% | 41.9% |
| 7-11% | 251,443 | 2.0% | 5.0% | 8.0% | 13.4% | 24.3% |
| 11-20% | 465,527 | 0.5% | 1.2% | 2.3% | 4.6% | 10.4% |

Why it no longer drives a decision: under the corrected condition a candidate has already closed
**above** its pivot, so there is no gap left to rank by and no order waiting to be touched. The
same reasoning dissolves Arun's nearest-to-pivot-versus-RS question — under the corrected
condition relative strength is the only available axis, which is what the study uses.

### Q3 — pending

## 6. Crash Recovery

**Running now:** one background shell, sequential, `/tmp/mpf/modes.log`.

```bash
ssh arun@94.136.185.54 'pgrep -af oa_entry_mechanics'      # alive?
ssh arun@94.136.185.54 'grep -E "entry=|ENSEMBLE" /tmp/mpf/modes.log'
```

**To re-run from scratch:** `/tmp/mpf/modes_run.sh` is self-contained (5 arms, ~2 min each).

**State of the live book while this is open.** Nothing is half-applied.

- The two entry crons are commented out. **To restore them, uncomment lines 112 and 114 of the
  crontab**, or `crontab /tmp/mpf/ct.bak.20260911-111828`. Do NOT restore them before `signal()`
  is corrected.
- Exits, marks, reconcile and the OHLC bakes all still run. The book keeps its −8% stop and its
  15-SMA trail.
- The four buy-stops placed at 10:29 are DAY orders: they expire at today's close whatever
  happens, and with the crons paused nothing re-arms them.
- 12 of the 16 held positions came from the 04-Sep seed, which used the **correct** breakout
  condition ("top-16 by RS of the 21 triggered candidates"). Only scanner entries since 08-Sep
  used the inverted one.

**Do NOT touch:** `backtest_data/oa_real_state.json`, `backtest_data/market_data.db`,
`research/142_bananapatterns_replication/scripts/bluesky_replay.py` (published engine; r/158 works
on its own copy).

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `OA_ARMING_WIDTH_AND_POKE_FILL_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/fill_rate.py` | Q1 touch-rate measurement | yes |
| `scripts/oa_entry_mechanics.py` | forked engine, adds `--entry-mode` | yes |
| `results/grid.log` | Q2 trigger x trail, 4 arms | yes |
| `results/modes.log` | Q3 mechanic bake-off, 5 arms | yes |
| `results/RESULTS.md` | verdict | yes |

## 8. Findings (final)

Pending Q3.
