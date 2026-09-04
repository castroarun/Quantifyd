# Research 148 — N500M + I75WR Paper-Book Audit — RESULTS

**VERDICTS: N500M — INCONCLUSIVE, currently CONSISTENT WITH SELECTION-ON-NOISE (pre-registered
bar not met: t = 1.40 at the 10bps cost floor; keep on paper, review at n≥100 / 2027-03-31).
I75WR — TOO EARLY BY DESIGN (n = 8, one sub-system, one symbol; no verdict manufactured;
review registered for n≥40 / 2026-12-31).** Neither book qualifies for the TN+OA blend test
(pre-registered: survivors only), and neither supports a real-money conversation today. Both
reviews are registered in the Ops & Review Center with their pass bars.

## N500M (per-stock promoted volbo/CCRB intraday, paper since 2026-05-08)

**The green headline:** 35 closed trades, +₹18,151, WR 57.1%. **The honest read:**

| Cost (RT) | Mean %/trade | t-stat | 95% CI | P&L |
|---|---|---|---|---|
| 0 bps (as booked — paper books NO costs) | +0.473% | 1.78 | [−0.05, +0.99] | ₹18,151 |
| 5 bps | +0.423% | 1.59 | [−0.10, +0.94] | ₹15,812 |
| **10 bps (r/109 floor)** | **+0.373%** | **1.40** | **[−0.15, +0.89]** | ₹13,474 |
| 15 bps | +0.323% | 1.21 | [−0.20, +0.84] | ₹11,135 |

- **The pre-registered bar (net expectancy > 0 with t ≥ 2) is NOT met at any cost tier.**
  Every CI includes zero. The WR's Wilson CI is [41%, 72%] — "58% WR" carries almost no
  information at n=35. Tradeability gate: avg win +1.55% / avg loss −0.97%, max losing
  streak 3 (fine), but 32/35 exits are EOD — the SL/target/trailing machinery barely
  engages; this is mostly an "enter intraday momentum, ride to close" book.
- **The selection-on-noise signature is present and strong.** Each rule was promoted as the
  argmax-by-Sharpe over a variant grid, then top-N across stocks (`n500m_configs.py`).
  Promotion-time expectation vs live, n-weighted: **expected +1.33%/trade → live +0.62%
  gross (53% shrinkage)**; expected WR ~83% (avg of promoted cells) → live 57%. Per-rule
  dispersion looks like regression to the mean: LAURUSLABS/RBLBANK/COCHINSHIP ran at-or-above
  expectation while ADANIGREEN (+2.95% expected → +0.07% live), HDFCAMC (1.46 → 0.04),
  GODFRYPHLP (1.88 → −0.44) and EICHERMOT collapsed.
- Concentration & stability: top symbol = 42% of net P&L; per-month +2.6k/+7.9k/+5.7k/
  +0.03k/+1.9k — August was flat on 3 trades (the book's cadence is also slowing: 9-11
  trades/month May-Jul → 3/month Aug-Sep; worth an ops look at why).
- **What was NOT run (declared in STATUS):** the ideal null — replaying alternative
  (non-promoted) grid cells over the same live window — needs a full r/30/31 intraday
  re-run on May-Sep 2026 data. Deliberately skipped: it is heavy, and moot until the book
  first clears t≥2 on its own trades. The shrinkage table is the cheap stand-in and it
  points the same way as the r/109-110 family verdict (NO EDGE at the cost floor).

**Verdict: keep on paper, zero promotion pressure. Review 2027-03-31 or n≥100** (registered):
pass = net-of-10bps expectancy > 0 with t ≥ 2. If it passes THEN, the alternative-cell null
replay becomes worth its cost before any sizing discussion.

## I75WR (3 intraday configs, paper since 2026-08-17)

- 8 closed trades, ALL from **Config C (multi-bar short bounce) on AARTIIND**: +₹1,989,
  WR 50%, mean +0.083%/trade. **Configs A and B have produced ZERO closed trades in 2.5
  weeks** — flagged as an ops observation (are their scanners firing at all?), not a defect
  claim.
- **No verdict.** n=8 from one symbol/one sub-system supports nothing, positive or negative,
  and this audit refuses to torture it (per the brief).
- **Pre-registered judgement (registered in Ops):** review at **n≥40 closed trades or
  2026-12-31**, whichever first. PASS = net expectancy > 0 with t ≥ 2 AND every config with
  ≥10 trades individually non-negative net. Also resolve the A/B silence before the review
  date, else the review judges only C.

## Registrations & scope

- Ops & Review Center: both reviews added to `research/111_sensex_manual_mgmt/scripts/
  ops_center.py` REVIEWS (renders at /app/straddles#ops-center with due-badges) and mirrored
  in `docs/LABS_AND_JOBS_REFERENCE.md`.
- Blend tests vs TN+OA: NOT run — pre-registered as survivors-only; neither survived.
- Nothing live was touched; both paper books keep running unchanged.

## What was NOT tested, and why

- Alternative-cell null replay over the live window (above — deferred behind the t≥2 gate).
- Fill-quality vs real ticks (paper fills come from live quotes at signal time; a
  slippage-vs-tick audit becomes relevant only alongside a passing review).
- I75WR anything beyond inventory (n=8).

## Reproducibility

`research/148_paper_books_audit/scripts/audit_books.py` (read-only, rerun anytime);
`results/n500m_audit.csv` (per-rule shrinkage), `results/i75wr_trades.csv`. DBs:
`n500m_trading.db`, `intraday_75wr.db` as of 2026-09-04.
