# Daily 09:16 ₹-Premium Strangle — Intraday Replay Sweep (1-min chain, Apr–Jul 2026)

**STATUS: DONE** · Ran 2026-07-24 20:29, 72s, 318 cycles · Verdict: **NO as an everyday system; the edge is entirely Mon/Tue (DTE ≤ 2)** — independent confirmation of research/51's "edge lives on 0/1-DTE"

## 1. The Ask

**What Arun asked:** "Also test this: every day morning 9:16 we sell this ₹20
premium (also test 10/15/25/30) CE and PE, apply the same exit principles —
see how it goes."

**What we're testing (interpretation, stated):** INTRADAY daily system — enter
at first snapshot ≥ 09:16 every recorded day, NEAREST expiry (incl. 0-DTE on
expiry days — reported separately), sell OTM CE+PE nearest ₹T mid premium,
T ∈ {10, 15, 20, 25, 30}; sell at BID; GTT stop 2.0×/leg (LTP trigger, ASK
fill); PT 50% combined; ONE roll-away (same T rule); survivors ride;
**time exit same day ≥ 15:15** (no overnight). Costs 0.25% premium + 0.1 pts.
If Arun meant multi-day holds (stacked overlapping book), that is a different
test — flag in report.

**Success read:** expectancy by T and by DTE bucket (0 / 1-2 / 3-7); is any
cell tradeable, and does expiry-day help or hurt? 65 days = directional read,
not proof.

## 2. Plan

~65 days × 5 targets = ~325 cycles. Output `daily916_cycles.csv` + aggregates
by T and DTE in the log.

## 3. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-24 20:3x | Runner written, launching | |

## 4. Crash Recovery

Runner `scripts/run_daily916.py`; launch via setsid nohup (see prior STATUS
files); log `results/run_daily916.log`; read-only on options_data.db.

## 5. Findings

64 trading days, 318 cycles (5 targets). Net pts/day, intraday (09:16 → PT/stop/15:15):

| Target | All days (t) | Mon+Tue (DTE ≤ 2) | Wed–Fri (DTE 3+) | Worst day |
|---|---|---|---|---|
| ₹10 | +1.43 (1.24) | +3.5 | **+0.04** | −21 |
| ₹15 | +2.41 (1.44) | +5.6 | +0.37 | −29 |
| ₹20 | +2.47 (1.13) | +7.0 | **−0.44** | −49 |
| ₹25 | +2.39 (0.90) | +6.8 | −0.44 | −55 |
| ₹30 | +3.20 (1.09) | +8.2 | −0.01 | −56 |

- **Wed/Thu/Fri intraday premium selling is dead** — every target ≈ 0 or negative
  after costs. Far-DTE intraday theta is too small vs spread + gamma.
- **All the money is Monday (DTE 1) + Tuesday (expiry day)**: T30 ≈ +8.2 pts/day
  ≈ ₹5.3k/day at 10 lots. This independently reproduces research/51's NAS
  finding ("edge lives on 0/1-DTE") with a different structure — strong
  cross-validation, but n = 25 days here.
- Caution: a Mon/Tue-only intraday variant would DOUBLE UP with NSR-W's Monday
  entry on the same underlying — correlated short-gamma. Recommendation: do not
  stack; NSR-W remains the primary. Revisit as a separate small paper variant
  only after the NSR-W paper book has a track record.
