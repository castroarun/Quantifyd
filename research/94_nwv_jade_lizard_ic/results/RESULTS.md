# research/94 — NWV Weekly View → Jade Lizard / Iron Condor Bake-off — RESULTS

**Date:** 2026-07-27 · **Verdict: NO EDGE for the directional-view JL/IC mapping as
specced; weak SIGNAL (t≈2.2, not investable as-is) for far-OTM premium selling on
NEUTRAL-view weeks.** Real NIFTY option EOD 2020-02 → 2026-07 (no modeled arm),
318 replayed Mondays, net of 0.5 pt/leg/side, 10 lots (65×10), r/89 liquidity rule
(every leg must show traded contracts on entry day).

## What was tested

Replay of the live NWV Phase-0 engine; per view bucket (BULL n≈36-40, BEAR n≈33-36,
NEU n≈101-112, `ignore` skipped n=128), Monday-EOD entry on next-week expiry:
incumbent debit spread, fixed-offset ICs (neutral / locked −50 / bull-skew), fixed
jade lizard (naked put −250 + call spread +250/+450), and the user's 2026-07-27
pivot-anchored family: short put @S1 (±550 disaster wing or naked), short call @R2
or @R1, +200 call wing. Exits: hold-Fri, PT50/stop−1×, PT50/stop−2× (EOD fills =
pessimistic under gaps).

## Key numbers (net, 10 lots)

| Cell | n | avg/wk | PF | worst wk | t | years + |
|---|---|---|---|---|---|---|
| **BULL-view: user's JL (S1/R2, 550 wing)** | 35 | **−₹1,044** | 0.92 | −₹119,762 | −0.18 | 3/7 |
| BULL-view: true JL (naked S1 put) | 36 | +₹3,560 | 1.27 | −₹157,365 | 0.52 | 4/7 |
| BULL-view: debit spread (incumbent) | 40 | +₹665 | 1.04 | −₹61,392 | ~0.2 | 4/7 |
| BEAR-view: fixed JL (bullish struct!) | 36 | +₹14,493 | 1.76 | −₹190,840 | 1.21 | 5/7 |
| BEAR-view: ic_bull (June pick) | 36 | +₹2,705 | 1.28 | −₹48,880 | 0.65 | 5/7 |
| **NEU-view: true JL hold** | 104 | **+₹14,338** | 1.99 | −₹395,427 | **2.22** | 5/7 |
| NEU-view: fixed JL hold | 112 | +₹13,917 | 1.88 | −₹452,302 | 2.16 | 5/7 |
| NEU-view: user's JL PT50/stop1× | 101 | +₹7,121 | 1.66 | −₹150,572 | 1.85 | 5/7 |
| NEU-view: ic_bull PT50/stop1× | 112 | +₹1,861 | 1.22 | −₹63,830 | 0.86 | 5/7 |
| ALL-weeks control: true JL hold | 300 | +₹6,408 | 1.31 | −₹427,602 | — | 5/7 |
| ALL-weeks control: user's JL hold | 293 | **+₹145** | 1.01 | −₹230,457 | — | 3/7 |

## Findings

1. **The user's exact construction (put spread @S1/−550 + call spread @R2/+200) is
   ≈ breakeven over history** — always-on +₹145/wk net (PF 1.01); its profitable
   years are 2020-21 only; 2023-26 all negative. The 550-pt disaster wing +
   far-R2 call anchor eat the premium edge the naked variant shows.
2. **On BULLISH-view weeks (like 2026-07-27's live trade) NO credit structure has
   an edge** — the user's JL is −₹1.0k/wk (t −0.18) there. Ironically the JL's
   best bucket is NEUTRAL weeks. Bullish weeks are already priced (drift ≈ credit).
3. **The bear-view inversion (June study) re-confirms on 2× the sample:** bearish
   debit spread −₹4.3k/wk PF 0.70; the *bull-aligned* structures win on bear-view
   weeks (fixed JL +₹14.5k) while the bear-mirror JL loses (−₹2.8k). But t≈1.2 —
   suggestive, not proven.
4. **The only cell family with any signal: far-OTM premium selling on NEUTRAL-view
   weeks** (true JL: +₹14.3k/wk, 82% win, PF 1.99, t 2.22, positive 5/7 years,
   negative 2024 −₹66k and 2026 −₹127k). After ~90-cell multiple-testing this is a
   weak SIGNAL, not a strategy. It needs a naked short put (~₹10-11L margin for 10
   lots) and carries −₹395k worst-week tail (EOD-measured; intraday worse).
   Consistent with r/89: index short-vol edge decayed post-2022 — here 2024/2026
   are exactly the negative years.
5. **Iron condors are flat-to-marginal everywhere** (best: ic_bull NEU +₹1.9k/wk,
   t 0.86). Defined risk, small tails, no meaningful expectancy at 200 wings.
6. **View value = the `ignore` filter, mostly.** View-selected weeks keep ~all of
   the always-on profit in 58% of the exposure (ignore weeks net −₹110k for JL),
   but BULL/BEAR/NEU conditioning of the structure itself adds little that
   survives significance.
7. Stops barely matter at EOD granularity (PT50/stop cells ≈ hold in most buckets);
   r/90's intraday premium-stop lesson does not transfer to this EOD harness —
   untested intraday.

## Seven-sins control status

Look-ahead: view built from prior-week + Monday-morning data only; entry at Monday
EOD (slight optimism: strikes anchored at 09:45 spot, priced at close — same for
all structures). Survivorship: index options, N/A. Multiple testing: ~90 cells
disclosed; best t 2.2 → treated as weak. Costs: 0.5 pt/leg/side + real-volume
filter. Regime: per-year tables shown; profits front/back-loaded by cell. Capacity:
NIFTY weeklies, fine at 10 lots. Shortability: N/A.

## Recommendation

- **Do not hard-wire the current construction into an auto-trader with live money
  expectations.** History says it is a ≈zero-EV structure net of costs, and the
  bullish-view instance of it is its weakest deployment.
- If automation proceeds (user directive), build it as a **G5 PAPER book**
  (nwv_trade paper executor) so forward data accumulates against these priors;
  candidate mappings, in order of historical support: (a) NEU-view → true JL
  (naked put; margin-hungry, tail −₹400k class), (b) NEU-view → user's JL
  (defined-risk, +₹7.1k/wk prior but front-loaded), (c) all-non-ignore-weeks →
  user's JL (breakeven prior, purest test of the user's template).
- The June-study 30-min R1/S1 structural stop remains untested on credit
  structures intraday — worth wiring into the paper executor as the tail-control.

Files: `results/run.log`, `results/jl_ic_ranking.csv`, `results/jl_ic_by_year.csv`.

---

## Phase 2 — WHEN/HOW to adjust (price-action triggers) — 2026-07-27 PM

Arun asked for a backtested answer on adjustments. On the locked template
(user JL, all 169 non-ignore weeks, same harness): triggers = daily close
breaching the short strike vs breaching the weekly pivot (S1/R2); actions =
exit-all, exit-threatened-side, roll-away (defensive), roll-chase (roll the
other side toward spot — the mentor-W30 manual habit), vs hold.
Script `scripts/run_adjustment_sweep.py`; CSV `results/adjustment_ranking.csv`.
(NB: the run.log `hold` row double-counts — true hold n=169, t≈1.16, totals ÷2.)

| Policy | avg/wk | PF | worst | t |
|---|---|---|---|---|
| hold to Friday | +₹3,964 | 1.29 | −₹230k | ~1.2 |
| **pivot breach → exit threatened side** | **+₹5,848** | **1.62** | **−₹144k** | **2.48** |
| pivot breach → exit all | +₹5,236 | 1.53 | −₹144k | 2.19 |
| strike breach → exit side | +₹5,503 | 1.51 | −₹144k | 2.08 |
| roll_away | +₹4,783 | 1.42 | −₹207k | 1.64 |
| roll_chase | +₹4,677 | 1.39 | −₹195k | 1.56 |

**Answer: never roll — exit.** Both roll styles re-widen the tail and lag the
simple exits. Best cell = exit the threatened side (spread + wing) on a daily
close beyond weekly S1/R2 (fires earlier than the strike). Fixes 2021 (hold
−₹100k → +₹173k) and cuts worst-week −₹230k → −₹144k. Fourth independent
confirmation of the project-wide pattern (r/92 hold>adjust, June morph net-neg,
mentor W30 manual rolls −₹6k). Caveats: EOD granularity (intraday
breach-and-recover untested; 30-min triggers = forward-test question), one
adjustment/week, best-of-8-arms t 2.48 = moderate.

## Build (G5 paper book) — DEPLOYED 2026-07-27

Arun picked mapping (b): **his JL template, all non-ignore weeks, paper-only.**
`services/nwv_trade.py` (clones nsrw_paper pattern): Mon 09:50 entry from the
live Phase-0 view + weekly-state pivots, next-week expiry, 10 lots, sells@bid /
buys@ask, PT50 / stop −1× (per-minute), **15:25 pivot exit-side rule** (phase-2
winner; combo with PT/stop is untested — the paper book is its forward test),
Fri 15:15 flat, DTE≤1 backstop, kill `/api/nwv-trade/kill-switch`, state
`/api/nwv-trade/state`, JSON state `backtest_data/nwv_trade_paper.json`.
Registered in app.py (after nsrw; `.bak_nwvtrade`); goes live at the next
09:00 pre-open restart (no market-hours restart). Week of 2026-07-27 cycle
SEEDED from Arun's actual manual fills (credit 44.44 pts) so the book mirrors
the live position from day one. Dry-run reproduced his exact strikes/expiry.
Pending: card on /app/nwv (build on VPS), git commit.
