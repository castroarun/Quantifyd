# NIFTY Strangle Rules — G2 Robustness: Pessimistic Fills, Post-Stop Actions, Indicator Exits, Iron Condor

**STATUS: DONE** · Ran 2026-07-24 15:19 IST, 22s, 58,028 rows · Verdict: **G2 PASS → STRATEGY-CANDIDATE** (`results/RESULTS.md` G2 section)

## 1. The Ask

**What Arun asked (2026-07-24):** (a) "After the 2× stop fires — what's the action?
Roll the leg out?" (b) "Why only premium exits — no VIX/ATR/ADX?" (c) "Proceed" on
the G2 that answers both with data plus the G1 caveats.

**What we're testing (4 questions):**
1. **Fill realism (make-or-break):** do the G1 stop-family results survive
   pessimistic gap-aware stop fills (gap-open → fill at open; intraday touch →
   fill at stop level) instead of optimistic close fills?
2. **Post-stop action:** flat-both (A0, the G1 rule) vs close-stopped-leg-only
   (ride the healthy leg) vs roll-stopped-leg-AWAY-once — head-to-head.
3. **Indicator exits vs premium stop:** ATR(14) percentile 80/90, ADX(14) 20/25,
   VIX day-jump ≥10%, VIX ≥1.25× entry — each replacing the premium stop,
   exiting causally at NEXT day's open (indicators are EOD signals; premium stop
   is a resting order — this asymmetry is real and modeled).
4. **Iron condor:** wings (M ±500, W ±200 — r/60 prior) — does defined-risk keep
   enough mean while structurally capping the weekly gap tail (and cutting margin)?

Plus: per-year stability tables, 2020-crash isolation, and a written
reconciliation vs research/89's post-2022 no-edge finding.

**Falsification (decided now):** if pessimistic fills push the monthly stop-family
(2.0–2.5×) post-2022 net below ~0, the mechanical-strangle idea is DEAD as a
strategy and research/90 concludes as "guardrail overlay only." If indicator exits
beat premium stops on tail-vs-mean, the premium-stop recommendation is withdrawn.

## 2. Economic hypothesis

Unchanged from G1/DESIGN.md (VRP harvesting, decayed post-2022; objective =
risk-shape not alpha). New sub-hypotheses: (a) premium already embeds
direction+vol → indicator exits should act later and score worse net of their
1-day lag; (b) post-stop re-entry (roll) re-adds tail exactly when realized vol
is elevated → should underperform flat (prior: r/54 one-and-done).

## 3. The Base — locked mechanics (delta vs G1 only)

- Data adds option daily OPEN/HIGH (same table, same liquidity filter).
- **Pessimistic stop fills:** leg open ≥ stop level → fill BOTH legs at that
  day's open (gap-through); else leg high ≥ level → stopped leg at stop level,
  other at close. PT/time/expiry remain close-based (PT is not a resting order
  at a fixed price on the combined, so close eval is fair).
- Post-stop arms at fixed p (M 2.5%, W 1.2%): `leg` = remaining leg rides with
  its own stop (same multiple on its own credit) to time-exit/expiry; `roll` =
  stopped leg re-sold same expiry at same %OTM from current parity-spot at that
  day's close (liquidity-checked), own stop; SECOND stop event → flat everything.
- Indicator exits at fixed p: signal computed on Kite NIFTY/VIX daily closes
  (ATR14 Wilder pctile vs trailing 252d; ADX14 Wilder; VIX jump/level); exit both
  legs at next trading day's OPEN.
- Condor: wings bought at entry close (nearest strike to short∓wing offset,
  price ≥ 0.5); wings closed at same day close whenever shorts exit; grid:
  stop {none, 2.0} × PT {50%, none}.
- Core G1 grid (192 configs) re-run under pessimistic fills for a direct
  fill-model delta. Giveback axis retained in core only for comparability.

## 4. Plan — grid

| Family | Configs | Cycles |
|---|---|---|
| Core re-run, pessimistic fills (all G1 axes) | 192 | 477 |
| Post-stop actions (stop 1.5/2/2.5 × PT 50/none × leg/roll × M/W) | 24 | 468 (p-fixed) |
| Indicator exits (6 mechanisms × PT 50/none × M/W) | 24 | 468 |
| Iron condor (M/W × stop none/2.0 × PT 50/none) | 8 | 468 |
| **Total** | **248** | ~58k rows |

## 5. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-07-24 15:0x | STATUS-MD written; runner authored | pre-launch |
| 2026-07-24 15:19 | Launched (setsid); DONE in 22s | 58,028 rows, 27 skips |
| 2026-07-24 15:2x | Per-year pulled; RESULTS.md G2 section written | PASS |

## 8. Findings

See `results/RESULTS.md` G2 section. Summary: pessimistic fills SURVIVED at stop
2.0–2.5× monthly (t 2.2–2.6, worst −161/−301) but killed 1.5× monthly (post-22
negative). Post-stop: monthly=flat-both; **weekly=roll-away-once is the study's
best family (t 4.73, 7/8 years positive, 2020 flat)**. Indicator exits all lose
to premium stop except VIX≥1.25×-entry on monthly (higher mean, 2.7× fatter tail
— mean-maximizer, rejected for the loss-minimization objective). Monthly condor
UNTESTABLE at EOD (stale wing marks — worst exceeds structural cap, biased
against condor); weekly condor consistent but thin. NSR v0.9 spec in RESULTS §5.

## 6. Crash Recovery

- Runner: `/home/arun/quantifyd/research/90_nifty_strangle_rules/scripts/run_g2_robustness.py`
- Launch: `ssh arun@94.136.185.54 'cd /home/arun/quantifyd && setsid nohup venv/bin/python research/90_nifty_strangle_rules/scripts/run_g2_robustness.py > research/90_nifty_strangle_rules/results/run_g2.log 2>&1 < /dev/null &'`
- Check: `tail -40 .../results/run_g2.log`; `wc -l .../results/g2_cycles.csv`; `pgrep -af run_g2`
- Idempotent full re-run (Kite caches reused). Read-only on market_data.db.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/run_g2_robustness.py` | G2 runner | yes |
| `results/g2_cycles.csv` | per cycle-config rows (~58k) | yes |
| `results/g2_ranking.csv` | per-config aggregates | yes |
| `results/g2_yearly.csv` | per-config per-year means | yes |
| `results/run_g2.log` | log | no |
| `results/RESULTS.md` | updated with G2 verdict | yes |
