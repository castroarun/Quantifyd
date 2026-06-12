# Gradual De-Risk Bake-Off — Staggered Chunk-Out vs Keep-Top8 vs 1-Shot Cash

**STATUS: DONE** · research/41 · Phase 29 · daily-marked engine · VPS canonical

**VERDICT (RESULTS_phase29.md): staggering HURTS — slower exit = deeper DD (−22.8→−23.9
as chunk shrinks), more tax; all stagger Calmar 1.42-1.53 < 1-shot 1.54 < keep-top8 1.66.
Gentleness comes from WHICH names you keep (keep-top8), not from spreading exit over time.
FINAL gradual answer = keep-top8 + month-end (Cal 1.66, DD −20.2, never full-dump).

## 1. The Ask

**What you asked (paraphrased):** you dislike the all-cash gate dumping the whole book
to debt in one move; you want a GRADUAL de-risk that's gentle for a 1-2 Cr client.

**What we're testing:** put every "gradual" candidate on ONE daily-marked engine and
rank them net-of-tax: (a) locked 1-shot all-cash, (b) keep-top8 (partial: keep 8
strongest, cash 7), (c) STAGGERED chunk-out — each weekly risk-off bar move another
`chunk` of the book to cash (chunk=0.25 → ~4 weeks to flat); stop the moment the market
is back above its 100-SMA; full redeploy at the next monthly rebalance (slow re-entry).
chunk=1.0 == the current all-at-once SMOOTHEST (parity sanity).

## 2. Base (unchanged core)

PIT mid band, RS-120, q0.5, ATH≤10% entry, N=15, top-22 buffer, per-stock-100SMA +12%
trail at month-ends, weekly 100-SMA gate, 0.4% RT, 6.5% cash, 2014-2026, daily-marked.
The ONLY thing varied = the risk-off ACTION (1-shot / partial-keep8 / time-staggered).

## 3. Plan — 7 configs × {gross, post-tax@20%}

| # | Config | Risk-off action |
|---|---|---|
| 0 | BASE allcash (p22 anchor) | dump 100% in 1 week |
| 1 | stagger c=1.0 (=allcash sanity) | should match #0 → validates engine |
| 2 | keep-top8 (p22) | keep 8, cash 7 (partial, 1 step) |
| 3-6 | stagger c=0.50/0.33/0.25/0.20 | scale to flat over 2/3/4/5 weeks |

**Sanity:** #0 ≈ 34.2%/−22.2%/1.54; #1 must ≈ #0. **Win bar:** a staggered config that
beats keep-top8's Calmar 1.66 / post-tax 28.3 OR matches it with a smoother exit path.

## 4. Status

| Time (IST) | Event | Notes |
|---|---|---|
| 2026-06-10 ~13:4x | STATUS + runner staged, launching | sections 1-4 pre-launch |

## 5. Crash Recovery

- Script: `research/41_.../scripts/29_staggered_derisk.py` (VPS), venv python.
- Run: `cd /home/arun/quantifyd && nohup venv/bin/python3 research/41_midsmall400_mq_concentrated/scripts/29_staggered_derisk.py > /tmp/phase29.log 2>&1 &`
- Progress: `tail -f /tmp/phase29.log`. Alive: `pgrep -af 29_staggered`.
- Output: `results/phase29_staggered.csv` (+ `_peryear.csv`), incremental. Don't touch market_data.db or scripts 01-28.

## 6-8. Files / Findings / Verdict

Pending run. Results → `results/RESULTS_phase29.md`.
