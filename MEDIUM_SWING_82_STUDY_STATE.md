# MEDIUM_SWING_82_STUDY_STATE — research/82 Medium-Swing (5-15d, long+short) crash-recovery master

> User mandate (2026-07-17): research/81 hard-capped holds at 3-4 days; this
> study sweeps the UNTESTED band — **5-15 session holds, long AND short via
> futures-proxy** — on the same engine/data/discipline. Doctrine:
> `research/QUANT_RESEARCH_PLAYBOOK.md`. Canonical copy on VPS
> `/home/arun/quantifyd/MEDIUM_SWING_82_STUDY_STATE.md`.

**Last updated:** 2026-07-17 ~12:10 IST

## 1. Current phase & sub-task

M-battery (M1-M4, 36 cells) pre-registered; IS screen launching.

## 2. Study definition

- Engine: research/81 engine with `ENGINE_MAX_TIME_STOP=15`; futures-proxy
  costs 3bp; daily bars for G1 (5-min precision only if a family survives).
- Universe: F&O (~81 names), CA-guarded; daily table 2005+.
- Splits (as research/81 daily families): IS 2005-01..2017-12, Val 2018-01..
  2022-06, OOS 2022-07+ (untouched; one look, user-authorized, at the end).
- Prior knowledge (honest): long side of this band already has a live
  occupant (KC6, 15d max hold, PF 1.70); shorts at ≤4d are comprehensively
  dead (research/81); breakouts work at multi-week trailing (research/71).
  The open question is the 5-15d fixed-hold band, especially SHORTS.

## 3. Experiments log

| ID | What | Cells | Verdict |
|---|---|---|---|
| M1 | Donchian {20,55}-day break × {L,S} × ts {5,10,15}, stop 2.5×ATR | 12 | pre-registered |
| M2 | Deep-z reversion z {2.0,2.5} × {L above SMA200, S below} × ts {5,10,15}, target SMA20, stop 2.5×ATR | 12 | pre-registered |
| M3 | Short-specific: (a) bear-breakdown (close<55d-low & <SMA200), (b) bear pullback-fade (SMA20 down-cross while SMA20<SMA200) × ts {5,10,15} | 6 | pre-registered |
| M4 | Prev-WEEK range break {H→L, L→S} × ts {5,10,15}, stop 0.33×week-range | 6 | pre-registered |

**Ledger: 36 cells. G1 gate: pooled net t≥3, ≥55% syms positive, coherent in
ts (5-15d should be monotone or plateau, not a spike).**

## 4. In progress / resume

Runner `research/82_medium_swing/scripts/run_m_battery.py` (resumable via
done-set CSV); log `/tmp/m_battery.log`; re-run = same command.
`ENGINE_MAX_TIME_STOP=15` must be in env.

## 5. OOS-touch ledger

None consumed.

## 6. Next 3 actions

1. Run M-battery IS screen → verdicts per gate.
2. Survivors → Val + robustness per playbook; else NO EDGE close-out.
3. Update this file + INDEX + TODO at every milestone.

## M-battery IS verdicts (2026-07-17 ~12:35 IST)

- **SHORTS: ALL 24 cells negative** (most gross-negative, t -3..-10) — combined
  with research/81, standalone short swing is DEAD across 1-15d holds. CLOSED.
- **M1_N20_L_ts15 G1 PASS**: +51.3bps t=4.00, 57% syms, monotone in ts.
- **M4_PWH_L_ts15 G1 PASS**: +28.4bps t=3.69, 64% syms, monotone in ts.
- M2 long dip-buy: +32bps t=2.17 (below gate; KC6 occupies this zone live).
- Next: per-year stability + Val (2018-2022) on the two survivors — the
  research/81 decay lesson makes this THE gate that matters. OOS untouched.

## STUDY CONCLUDED 2026-07-17 ~13:00 IST

Shorts: NO EDGE (final, 1-15d spectrum closed). Longs: real cyclical
breakout edge at 10-15d holds converges on research/71's already-live family
(breakout paper book) — no new build; OOS UNCONSUMED. Full verdict:
research/82_medium_swing/results/RESULTS.md
