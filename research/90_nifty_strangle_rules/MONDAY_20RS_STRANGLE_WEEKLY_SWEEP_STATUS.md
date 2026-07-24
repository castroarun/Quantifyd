# Arun's Weekly Spec — Monday Entry, Next-Week Expiry, ₹20-Premium Strangle + Roll-Away

**STATUS: DONE** · Ran 2026-07-24 15:36 IST, 25s, 12,188 rows · Verdict: **SPEC VALIDATED — best t-stat in research/90** (see §6)

## 1. The Ask

**What Arun asked (2026-07-24):** "Only the weekly sleeve. Enter on a Monday
(next-week expiry contracts). I generally target ₹20 premium each for call and
put. Roll-away yes, your stop is fine. Please assess this."

**What we're testing:** Arun's exact spec vs the validated NSR weekly sleeve,
isolating each difference: (a) entry day Monday vs day-after-expiry; (b) strike
selection by PREMIUM TARGET (₹15/20/25/30 per leg) vs %OTM; (c) DTE ~7–12
(next-week expiry — note: spans TWO weekends pre-2025 Thu-expiry era) vs ~4–7.
Stops 1.5×/2.0× pessimistic fills, roll-away once (rolled leg re-targeted to the
same premium rule), second stop → flat, PT {none, 50%}, time exit DTE≤1.
Also reports what ₹20-premium actually maps to in %OTM per year.

**Note:** Arun's W30 Monday entry (23350PE @19.6 / 24800CE @16.9, 8 DTE) was
already this spec — this study effectively replays his entry rule with
mechanical management, 2019→2026.

**Falsification:** if the Monday/next-week variant is materially worse than the
validated day-after-expiry sleeve (lower t AND fatter tail), recommend shifting
his entry day rather than automating his current habit.

## 2. The Base (delta vs G2 only)

- Cycles: every first-trading-day-of-week (Mon or first day after holiday),
  2019→2026. Target expiry: nearest weekly expiry with cal DTE in [6, 12].
- Strikes: OTM side (PE < spot < CE), liquid (contracts ≥ 50), close premium
  nearest to target T ∈ {15, 20, 25, 30}; both legs must quote ≥ 1.5 pts.
- Roll-away: on stop, re-sell same side at premium nearest T at that day's close.
- Everything else per G2 weekly sleeve (pessimistic fills, costs, time exit).
- Comparator arm: day-after-expiry entry with premium-target 20 (isolates the
  entry-day effect from the strike-rule effect).

## 3. Plan

Grid: entry {monday, after-expiry} × T {15,20,25,30} × stop {1.5, 2.0} ×
PT {none, 0.5}, roll always on = 32 configs × ~390 cycles ≈ 12.5k rows.

## 4. Status

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-07-24 15:4x | STATUS-MD + runner written | pre-launch |

## 5. Crash Recovery

- Runner: `research/90_nifty_strangle_rules/scripts/run_g3_arun_weekly.py` (VPS)
- Launch: `cd /home/arun/quantifyd && setsid nohup venv/bin/python research/90_nifty_strangle_rules/scripts/run_g3_arun_weekly.py > research/90_nifty_strangle_rules/results/run_g3.log 2>&1 < /dev/null &`
- Check: `tail -40 .../results/run_g3.log`; outputs `g3_cycles.csv`, `g3_ranking.csv`, `g3_yearly.csv`
- Idempotent; read-only on DB.

## 6. Findings

**Arun's spec validated, with two data-driven upgrades.** Monday entry beats
day-after-expiry across the whole grid (t 4.3–5.5 vs 3.4–4.3) — premium targeting
at 9.6 avg DTE lands FURTHER OTM for the same ₹ (T20 → 3.24% OTM vs 2.01%), more
room to be right. Premium-target axis monotonic (T15→T30 mean 6.1→11.1, t stable).

- **His T20 + PT50 + stop2.0 + roll:** net 7.86 pts/wk (t 4.79, win 73%, p5 −28.6,
  avg credit 39.8). Per-year: positive 2019–2025, 2026 flat (−0.3).
- **T30 + PT50 + stop2.0 + roll: best cell in all of research/90** — 11.06 pts/wk,
  **t 5.47**, win 71%, p5 −39.9, **positive ALL 8 years incl. 2026 (+9.7)**.
- **Stop 2.0× beats 1.5× on Monday/9.6-DTE entries** (1.5× whipsaws on the longer
  hold); PT50 improves t everywhere. G2's 1.5× finding was for ~5-DTE entries —
  stop tightness is DTE-dependent.
- **Residual tail:** worst −445 pts (T20; entry 2026-02-02 crash week, TIME exit —
  ground through without a clean leg-stop breach; ₹−2.9L at 10 lots). Gap/grind
  weeks survive all stop rules; the untested mitigation is an event-skip rule
  (election results/budget/Fed weeks). Proposed for the paper-book phase.

**Locked spec — NSR-W v1.0 (Arun variant):** Monday entry, next-week expiry
(~9–10 DTE), sell CE+PE nearest ₹20–30 premium each (liquid strikes), GTT stop
2.0× per leg, PT 50% of credit, ONE roll-away re-targeted to same premium,
second stop → flat, time exit DTE≤1, fixed 10 lots, no size-ups, margin ≤70%.
