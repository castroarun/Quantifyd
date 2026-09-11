# Entry-mechanic audit across all three books — STATUS: RUNNING (last arm set in flight)

One line: **TN is sound, IPO Base is honest in the live engine but half its published number,
and Open Alpha has no entry that both works and can be placed.** Open Alpha buying is paused.
The last open question is whether Arun's fundamental screen rescues it.

## 1. The Ask

Three requests, in the order they arrived (11-Sep-2026):

> "look at even most [names] and place the orders because few orders might not get filled. If
> something gets filled, then we'll leave out the rest. Is that not the plan?"

> "we shud rather enter on the same day as the breakout close towards the market close - say
> 3:10 PM, for our study u can take day close as the entry price..."

> "see likwise valudate our TN adn IPO systems too, jsut confirm is the trade entries adn
> exit are correct in line with ouer learning heres"

> "Let's go check the fundamental analysis... for the last three years... profit growth more
> than fifteen... sales growth more than fifteen... debt to equity point two zero or less...
> ROE and ROCE above fifteen... no negative figures"

**What is actually being tested.** Does each book buy and sell the way the study that
justified it measured? And where it does not, what can the book actually earn?

## 2. The Base — the defect class

A backtest that decides using a day's CLOSE but pays a price available EARLIER that day.
It is not a rounding error: it silently removes every trade that failed intraday, because a
trade only enters the record when the close confirmed it.

Tested per book by reading the engine, then measuring the honest variant on the same spec,
seeds, window and code, changing only the entry.

## 3. Findings — SETTLED

### TN (True North) — SOUND, no action

Engine holds only closes (`self.cf`), no opens or highs, so the mismatch is structurally
impossible. Gate computed on the dropna'd series then reindexed — the exact fix r/142 needed
after phantom holiday rows disabled its gate for months. Universe point-in-time. Every live
dial matches: 8 slots, 22 buffer (2.75×n), 200 universe, NIFTYBEES 100-SMA weekly gate to
cash, donch-15, 0.3% round trip, 6.5% cash. **Published numbers stand.**
NOT re-verified: survivorship, the RS formula, line-by-line implementation.

### IPO (IPO Base) — live engine CORRECT, study overstated

`ipo_replay.py`: `trig = ... & (ctx.C > piv)` on day i, `fill = max(pv, O[i, c])` on day i.
`services/ipo_paper.py` is right and fills the NEXT morning.

| Arm, 2006→2026-09, 30 seeds, 25bps, AFTER TAX | CAGR | DD | Calmar |
|---|---|---|---|
| Study headline, reproduced | 31.48% | −20.88% | 1.52 |
| Study's own published figure | 31.03% | −20.88% | 1.50 |
| Study's close-fill control | 17.49% | −32.07% | 0.53 |
| **LIVE engine: next-day stop at the broken pivot** | **15.00%** | **−37.55%** | **0.40** |
| NIFTYBEES held (pre-tax) | 11.5% | −59.7% | 0.19 |

98.5% of same-day signals survive as reachable next-day fills, so the loss is the entry
price, not missed trades. **Survives at ~half strength.** Study page corrected (`21dd8fa6`).

### OA (Open Alpha) — no placeable entry has an edge

**Proven on the source site's own 54 published trades, not on simulation:**

| Test | Result |
|---|---|
| Entry-day close finished above the pivot | 49 of 50 (98%) |
| Entry day opened above the price booked | 9 of 50 |
| Failed resting-order fills in the prior 120 days | **348 = 7.0 per published trade** |

HCLTECH: the clean 10-Jan-2025 entry at 1972.20 follows 20 earlier touches of the same level
that closed back below. A live order was filled 18-Dec-2024 and lost.

**Every placeable entry, 2006→2026-08, 30 seeds, 16 slots @6.25%, −8% stop, 25bps:**

| Entry | CAGR | After tax | DD |
|---|---|---|---|
| Published (NOT placeable) | 40.8% | — | −33.9% |
| Stop above the breakout candle, trail-20 | 9.9% | — | −56.0% |
| Buy at the breakout close (~15:10) | 8.5% | 5.5% | −57.3% |
| Stop above the breakout candle, trail-15 | 6.5% | 3.7% | −71.1% |
| Next-day stop at the pivot | 2.7% | −0.5% | −72.6% |
| **Touch of the pivot (what the code does)** | **−1.4%** | — | **−81.8%** |
| Same-day abort + slot recycling | −21.0% | −22.2% | −99.5% |
| **NIFTYBEES held** | **11.5%** | — | **−59.7%** |

Decomposition of the published number: ~8pp is booking the pivot on gap-up days, ~42pp is
counting only breakouts that held. **Zero costs AND the inflated fill together still reach
only 11.9%**, so this is not a friction problem. RS selection is no better than random
(−1.6% vs −1.4%), so nothing above understates the book.

**Second, separate defect:** `services/oa_entry.py` selects `close < pivot` — names that
have NOT broken out. The design, the register and the backtest all say `close > pivot`. The
two can never pick the same name on the same day. The 04-Sep seed used the correct
condition; only scanner entries from 08-Sep did not.

## 4. Findings — IN FLIGHT

### OA fundamental overlay (Arun's screen)

Screener annual history for 609 of 638 breakout candidates, median 12 fiscal years, 601 with
the 4 years a three-year growth rate needs. Point-in-time with a 4-month filing lag.

| Mask | Pass rate |
|---|---|
| All five criteria | **9.7%** (59 of 609) |
| Without debt/equity | 20.2% |
| Without ROCE | **9.7%** |
| Without growth | 20.7% |
| Growth + no-negatives only | 34.3% |

Per criterion (of 601): growth 210 pass / 391 fail · D/E 282 / 319 · ROE 254 / 347 ·
ROCE 315 / 246 (40 lenders n/a) · no-negatives 558 / 43.

**ROCE is inert** — dropping it leaves 9.7% unchanged, so it never rejects a name ROE has
not already rejected.

Window baseline (Aug-2024→Sep-2026, no filter, stop-above-candle trail-20):
**−7.2% CAGR [−26.6..17.4], DD −40.8%.** A two-year window cannot settle anything; the
question is filter vs no-filter in the SAME window.

22 arms: 2 entries × (no filter + 5 masks × 2 missing-policies). Then 3 abort arms.

## 5. Status log

| Time (IST) | Event |
|---|---|
| 10:26 | Armed 3 OA slots live; book to 16/16 committed |
| ~11:05 | Fill-rate by gap measured (1.26M setups) |
| ~11:12 | **Trail claim retracted** — trail-15 IS the after-tax paired winner; RESULTS.md line 286 was stale and is corrected. Arun caught it |
| ~11:14 | Inverted scanner condition found |
| 11:18 | Entry crons PAUSED (129→131 lines); exits untouched |
| ~11:40 | Trigger × trail 2×2 done: touch entry −1.4% at trail-15 |
| ~12:10 | Published entry proven not placeable on the site's own 54 trades |
| ~12:30 | Arun cancelled the 4 resting orders by hand |
| ~12:45 | IPO honest number: 31.0% → 15.0% |
| ~13:10 | strategies.ts corrected + built (`677b696a`) |
| ~13:30 | TODO.md updated (`d6eb397b`) |
| ~13:50 | Trade ledger published on /app/bluesky-paper (`1b0612aa`) |
| ~14:05 | IPO study page corrected (`21dd8fa6`) |
| ~14:30 | Screener fetch done (609/638); masks built; 22 overlay arms started |

## 6. Crash Recovery

**Running:** `/tmp/mpf/chain2.sh` → 22 arms to `/tmp/mpf/screener_chain.log`, then
`/tmp/mpf/abort2_run.sh` → 3 arms to `/tmp/mpf/abort2.log`.

```bash
ssh arun@94.136.185.54 'pgrep -af oa_entry_mechanics'
ssh arun@94.136.185.54 'grep -E "^####|ENSEMBLE" /tmp/mpf/screener_chain.log'
```

Re-runnable from scratch: `/tmp/mpf/fundrun.sh` and `/tmp/mpf/abort2_run.sh` are
self-contained. The Screener cache is under `results/screener_cache/` and resumable.

**LIVE STATE — nothing is half-applied.**

- OA entry crons COMMENTED OUT (lines 112, 114). Restore with
  `crontab /tmp/mpf/ct.bak.20260911-111828` — but NOT before the entry is corrected.
- OA exits, marks, reconcile, OHLC bakes all still run. Stops and the trail still protect
  the open positions.
- Arun cancelled the four resting buy-stops by hand; they were DAY orders in any case.
- 12 of the 16 positions came from the 04-Sep seed on the CORRECT condition.
- TN and IPO untouched and running normally.

**Do NOT touch:** `backtest_data/oa_real_state.json`, `backtest_data/market_data.db`,
`research/142_bananapatterns_replication/scripts/bluesky_replay.py` (published engine;
r/158 works on its own copy `scripts/oa_entry_mechanics.py`).

## 7. Files

| File | Purpose |
|---|---|
| `scripts/oa_entry_mechanics.py` | forked r/142 engine: `--entry-mode`, `--eod-abort`, `--abort-keeps-slot`, `--fund-mask`, `--base-start` |
| `scripts/verify_published_trades.py` | the audit of the site's own 54 trades |
| `scripts/screener_fetch.py` | annual history, stdlib HTML parsing, resumable |
| `scripts/fund_mask2.py` | the five eligibility masks, point-in-time |
| `scripts/ipo_honest.py` | IPO next-day entry measurement |
| `scripts/gen_ledger.py` | the two-year ledger behind /app/bluesky-paper |
| `scripts/fill_rate.py` | touch-rate by gap (1.26M setups) |
| `results/*.log`, `results/*.csv`, `results/screener_cache/` | every run and its inputs |

## 8. Verdict

Pending the overlay. Everything else is settled and recorded in `TODO.md`,
`frontend/src/data/strategies.ts` and the two study pages.
