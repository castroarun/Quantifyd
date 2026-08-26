# Nifty Total Market (750) — Top-15 by the Live Book's Momentum Rule

STATUS: DONE

## 1. The Ask

**What Arun asked:** "if we are to source stocks from Nifty 750 for our momentum pf today, can u
list down the current top 15 of them?" — followed by "preferably, look for data on nse for the
price action rather than depending on our own database".

**What we are actually producing:** today's top 15 of the Nifty Total Market (750) ranked by the
SAME score the live Momentum-30 book already uses, so the answer is comparable to what the book
holds now. This is a SCREEN, not a backtest — it says what the existing rule points at over a wider
universe. It does NOT say the wider universe is better; that is a separate study.

## 2. Why NSE data and not our DB

`backtest_data/market_data.db` holds daily bars for 1,666 symbols but only **409 are current** to
2026-08-19. Ranking a "750" off that would silently deliver a Nifty-400-ish list under the wrong
label. So:

- **Constituents:** official `ind_niftytotalmarket_list.csv` from niftyindices.com (752 EQ rows).
- **Prices:** Kite historical API — the NSE feed itself — pulled fresh, ~430 calendar days each.

## 3. The Score (unchanged from the live book)

`services/momentum_paper.py::_rs_basket`:

    rsblend = 0.5 x (6m return / NIFTYBEES 6m return) + 0.5 x (12m return / NIFTYBEES 12m return)

126 and 252 trading-day lookbacks, benchmark NIFTYBEES (same as the live macro gate). A score of
1.00 means the stock exactly matched the index. No new formula is introduced.

## 4. Plan

1. Parse the official constituent list
2. Map symbols to Kite instrument tokens
3. Pull daily closes + volume for each (0.35s spacing = 3 req/sec; ~750 calls ~ 5-7 min)
4. Build the panel, compute rsblend + 6m/12m returns + 6-month median rupee ADV
5. Write full ranking CSV, print the top 15 with a liquidity flag

## 5. Status

| Time | Event |
|---|---|

## 6. Crash Recovery

- Progress: `tail -f research/113_nifty750_momentum_rank/results/run.log`
- Alive? `pgrep -af rank750`
- Bars are cached incrementally in `results/bars_cache.csv`; re-running SKIPS symbols already
  cached, so a re-run resumes rather than re-downloading.
- Resume: `cd /home/arun/quantifyd && nohup ./venv/bin/python3 research/113_nifty750_momentum_rank/scripts/rank750.py > research/113_nifty750_momentum_rank/results/run.log 2>&1 &`
- This script is READ-ONLY: it places no orders and writes nothing to the live book.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/rank750.py` | Fetch + rank runner | yes |
| `results/bars_cache.csv` | Raw daily bars (large) | NO |
| `results/ranking.csv` | Full 750 ranking | yes |
| `results/run.log` | Progress log | yes |

## 8. Findings

See `results/RESULTS.md`. Top 15 produced from 715 scorable constituents. **Screen only** — today's constituent list is survivorship-selected, so trailing-return ranking selects on the outcome. All current holdings rank #16-61, i.e. a 750-sourced book would replace the whole portfolio. Not backtested.
| 2026-08-26 10:42:54 | constituents parsed: 748 EQ symbols from the official NSE list |
| 2026-08-26 10:42:54 | matched to Kite instruments: 746 (unmatched: 2) |
| 2026-08-26 10:43:13 | fetched 50 / 747 (at ANUP, 50 scanned) |
| 2026-08-26 10:43:33 | fetched 100 / 747 (at BATAINDIA, 100 scanned) |
| 2026-08-26 10:43:45 |   fetch failed BOSCHLTD: Too many requests |
| 2026-08-26 10:43:56 | fetched 150 / 747 (at CGPOWER, 151 scanned) |
| 2026-08-26 10:44:15 | fetched 200 / 747 (at DYNAMATECH, 201 scanned) |
| 2026-08-26 10:44:35 | fetched 250 / 747 (at GLENMARK, 251 scanned) |
| 2026-08-26 10:44:54 | fetched 300 / 747 (at HOMEFIRST, 301 scanned) |
| 2026-08-26 10:45:14 | fetched 350 / 747 (at ITI, 351 scanned) |
| 2026-08-26 10:45:34 | fetched 400 / 747 (at KPITTECH, 401 scanned) |
| 2026-08-26 10:45:53 | fetched 450 / 747 (at MGL, 451 scanned) |
| 2026-08-26 10:46:15 |   fetch failed ONESOURCE: Too many requests |
| 2026-08-26 10:46:16 | fetched 500 / 747 (at ORKLAINDIA, 502 scanned) |
| 2026-08-26 10:46:36 | fetched 550 / 747 (at RAILTEL, 552 scanned) |
| 2026-08-26 10:46:56 | fetched 600 / 747 (at SFL, 602 scanned) |
| 2026-08-26 10:47:16 | fetched 650 / 747 (at TARC, 652 scanned) |
| 2026-08-26 10:47:36 | fetched 700 / 747 (at ULTRACEMCO, 702 scanned) |
| 2026-08-26 10:47:54 | fetch complete: 745 newly fetched, 745 total, 2 failed |
| 2026-08-26 10:47:54 | panel built: 745 symbols x 293 days, last 2026-08-26 |
| 2026-08-26 10:47:54 | ranking written: 713 symbols scored. Benchmark 6m -3.8% / 12m -1.6% |
| 2026-08-26 10:48:19 | constituents parsed: 748 EQ symbols from the official NSE list |
| 2026-08-26 10:48:20 | matched to Kite instruments: 746 (unmatched: 2) |
| 2026-08-26 10:48:20 | resuming: 745 symbols already cached |
| 2026-08-26 10:48:21 | fetch complete: 2 newly fetched, 747 total, 0 failed |
| 2026-08-26 10:48:21 | panel built: 747 symbols x 293 days, last 2026-08-26 |
| 2026-08-26 10:48:21 | ranking written: 715 symbols scored. Benchmark 6m -3.8% / 12m -1.6% |
