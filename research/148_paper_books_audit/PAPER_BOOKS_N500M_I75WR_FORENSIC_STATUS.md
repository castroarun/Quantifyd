# N500M + I75WR Green Paper Books — Honest Audit (selection-on-noise risk, n-size truth)

STATUS: DONE — **VERDICTS: N500M INCONCLUSIVE / consistent with selection-on-noise (t=1.40 at the 10bps floor, 53% promotion-shrinkage) — paper only, review 2027-03-31 or n≥100. I75WR TOO EARLY (n=8) — review 2026-12-31 or n≥40. Both reviews REGISTERED in Ops.** See `results/RESULTS.md`.

## 1. The Ask (verbatim)

> "How ab I75WR, N500M from our paper books, showing greens. pls study them." — Arun

**What we're actually testing:** are the green P&Ls of these two young intraday paper books
EVIDENCE OF EDGE, or the expected output of promotion-on-noise plus a small sample? Central
prior: the intraday OHLCV family was concluded **NO EDGE at the ~10bps cost floor**
(research/109-110), and N500M's per-stock rules are each the *argmax over a variant grid by
backtest Sharpe, then top-N across stocks* (`services/n500m_configs.py` `_load_csv`/
`load_all_configs`) — double selection, the textbook multiple-testing construction.

## 2. The Base — what the books are (read-only; no live changes)

- **N500M**: per-stock promoted volbo/CCRB intraday rules (research/30/31 sweeps), paper
  since 2026-05-08. State: `backtest_data/n500m_trading.db` → `n500m_positions`. 35 CLOSED
  trades through 2026-09-03 (34 volbo, 1 ccrb), +₹18,151 gross-of-cost, WR 57.1%.
- **I75WR**: 3 intraday configs (A: TP0.5/SL1.5 · B: TP2.0/SL1.5 · C: multi-bar short
  bounce TP1.5/SL1.0), paper since 2026-08-17. State: `backtest_data/intraday_75wr.db` →
  `i75_positions`. **Only Config C has fired: 8 trades, ALL on AARTIIND short, +₹1,989.**
  Configs A/B: zero closed trades.

## 3. Pre-registered evaluation criteria (BINDING, before computing anything)

- **N500M verdict bar:** the book shows a defensible edge ONLY if net-of-realistic-cost
  per-trade expectancy is positive with **t ≥ 2** on the live sample. Costs: paper fills
  book NO transaction costs (verify) — stress at 5/10/15 bps per round trip (r/109 floor).
  Additional selection test: compare each promoted rule's EXPECTED per-trade mean (the
  backtest number it was promoted on, carried in `expected_*` fields) vs its LIVE realized
  mean — heavy shrinkage toward zero is the signature of selection-on-noise.
  **Limitation declared up front:** a full alternative-cell live replay (the ideal null)
  requires re-running the r/30/31 intraday grid over May-Sep 2026 intraday data — NOT run
  here (heavy, and moot unless the book first clears t≥2); the shrinkage test + selection
  arithmetic stand in.
- **I75WR:** n=8 from one sub-system on one symbol = NO verdict possible, and none will be
  manufactured. Deliverable = pre-registered pass criteria + a dated review:
  **judge at n ≥ 40 closed trades or 2026-12-31, whichever first; PASS = net expectancy > 0
  with t ≥ 2 AND every config that traded ≥ 10 times individually non-negative net.**
- Blend tests vs TN+OA: only if a book clears its bar (per the workstream brief).
- Both reviews get registered in the Ops & Review registry (binding 2026-08-16).

## 4. Plan

1. Pull both books' trade histories (read-only), compute: per-trade % series, mean/std/
   t-stat, 95% CI (mean + Wilson WR), cost stress rows, per-symbol concentration, per-month
   stability, exit-reason mix.
2. N500M shrinkage table: expected_mean_pct / expected_sharpe (promotion-time) vs live
   realized, per rule with ≥3 live trades.
3. Verdicts + review registrations (ops_center REVIEWS + LABS_AND_JOBS_REFERENCE mirror).

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-04 01:2x | Trade histories pulled (35 + 8 closed), configs promotion mechanism read | STATUS written before analysis |
| 2026-09-04 01:3x | Audit run; N500M t=1.78 gross / 1.40 @10bps, CI includes 0 at all tiers; shrinkage expected +1.33 → live +0.62%/tr; WR CI [41,72]; 32/35 exits EOD; Aug flat + cadence slowing. I75WR: only Config C fired (8 trades, AARTIIND only); A/B silent — ops flag. | results/RESULTS.md |
| 2026-09-04 01:4x | Reviews REGISTERED (ops_center REVIEWS + LABS_AND_JOBS_REFERENCE): N500M 2027-03-31/n≥100 pass=t≥2 net-of-10bps; I75WR 2026-12-31/n≥40. Committed + pushed. | DONE |

## 6. Crash recovery

- Read-only audit; script `scripts/audit_books.py` on VPS; rerun any time:
  `cd /home/arun/quantifyd && venv/bin/python research/148_paper_books_audit/scripts/audit_books.py`
- Outputs `results/n500m_audit.csv`, `results/i75wr_trades.csv`, printed stats → RESULTS.md.
- Touches NOTHING live (the paper books keep running).

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| this STATUS md | live status | yes |
| `scripts/audit_books.py` | audit computations | yes |
| `results/*.csv`, `results/RESULTS.md` | outputs + verdicts | yes |

## 8. Findings

(see RESULTS.md)
