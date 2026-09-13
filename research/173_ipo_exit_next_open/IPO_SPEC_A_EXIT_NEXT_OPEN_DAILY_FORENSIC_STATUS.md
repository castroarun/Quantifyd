# IPO Base Spec A — exit at the NEXT OPEN vs the SIGNAL CLOSE (fidelity check)

**STATUS: DONE — IMMATERIAL** — research/173, opened 13-Sep-2026 20:10 IST

## The Ask

**What Arun asked (via the launching agent, 13-Sep-2026):** "what does it cost IPO Base Spec A
to exit at the NEXT OPEN instead of at the SIGNAL CLOSE?" Live exits are being automated on
13-Sep-2026: the 18:45 run detects the exit on the day's close and places an after-market sell
that fills at the next open (MARKET AMO; if MARKET is refused, a LIMIT floored 2% under the
signal close). The backtest sells at the signal close. Is that one-open delay material?

**What we are testing:** on research/167's engine and panel, with Spec A held fixed, three exit
fills on the SAME exit decisions and the SAME 30 seeds:

| Arm | Exit fill for an exit that fires on day i's close |
|---|---|
| A | close[i], the study as published |
| B | open[i+1], no floor. Slot freed for day i+1's entries as in A; sale cash available on day i+1 before entries; last bar of the window sells at its close |
| C | as B, but only if open[j] >= 0.98 x the most recent close; otherwise not filled that day and re-tried at the next open with a floor 2% under the latest close |
| C2 (sensitivity, not in the bar) | as C, but a day LIMIT also fills AT the floor if the day's high reaches it |

Plus the overnight gap close[i] -> open[i+1] on the exit signals, by exit reason.

This is a fidelity check, not a new strategy. No services/ or frontend/ edits.

## The Base (Spec A, fixed)

- Entry: next-day buy-stop at the 25-bar close-pivot, filled max(pivot, open); signal-day BARS >= 25,
  listing age <= 6 months, base depth <= 30%, RS off, name-based fund exclusion (research/167).
- Gate: no new entries when NIFTYBEES < SMA-150 (shifted one day).
- Exits on the close: stop close <= fill x 0.90; target close >= fill x 1.25; trail close < SMA-50.
- Book: 8 slots @ 18.75% of equity, Rs 10L, 25 bps a side, 20% STCG / 12.5% LTCG with FY netting.
- Idle cash 5.0% (research/167 basis) and 5.2% (house standard, research/168).
- Windows: 2006-01-01 -> 2026-09-04 full, 2006-2015, 2016-2026-09-04 (research/167's).
- Seeds 1-30, random selection among more triggers than slots.

## Plan

- Step 0 — reproduction gate: our forked simulator in arm A must equal `ipo_replay.simulate_ipo`
  bit-for-bit on seeds 1-3, and arm A at 5.0% must reproduce 21.80% / -26.63% median within 0.15pp.
  If the uncommitted listing_dates.csv (rewritten 13-Sep 10:15) breaks it, re-run on the
  committed HEAD copy and say so.
- Step 1 — 4 arms x 3 windows x 2 yields x 30 seeds = 720 simulations.
- Step 2 — gap distribution, floor-hit count, extra days held, attribution by exit reason.

**Pre-registered bar (written before running):** MATERIAL if arm B or arm C, on the FULL period,
shows a median PAIRED loss vs A of more than 1.0pp of CAGR, or more than 0.10 of Calmar, AND A beats
that arm on that metric on at least 20 of 30 seeds. Otherwise IMMATERIAL. Judged at 5.0% idle
cash (published basis); 5.2% and the two halves are reported as confirmation.

## Status log

| Date/time | Event | Notes |
|---|---|---|
| 13-Sep-2026 20:10 IST | STATUS written, scripts staged | research/173 (172 was the highest taken) |
| 13-Sep-2026 20:25 IST | Step 0 PASSED | fork arm A bit-identical to simulate_ipo on seeds 1-3 (396/396, 398/398 trades, same final NAV); Spec A @5.0% = 21.80 [worst 20.83] / -26.63 [worst -32.88] / Cal 0.819, identical on the refreshed and the committed listing_dates.csv -> refreshed file used |
| 13-Sep-2026 20:26 IST | Step 1 launched | 24 blocks x 30 seeds, log /tmp/r173_exit.log |
| 13-Sep-2026 20:28 IST | Step 1 DONE (0.9 min) | 5.0% full period median CAGR: A 21.80, B 22.97, C 23.42, C2 23.16 -- next-open exits come out AHEAD; treated as suspect until the gap audit (Step 2) checks it |
| 13-Sep-2026 20:45 IST | Step 2 DONE, report written | report crashed once on a gap-column merge clash (fixed, re-run); gaps sane (worst -6.85%, no split artefacts); RESULTS.md written; verdict IMMATERIAL |

## Crash recovery

```
ssh arun@94.136.185.54
cd /home/arun/quantifyd
tail -30 /tmp/r173_exit.log
ps aux | grep exit_timing | grep -v grep
ls -l research/173_ipo_exit_next_open/results/
# resume (blocks already in seed_stats.csv are skipped):
setsid nohup venv/bin/python -u research/173_ipo_exit_next_open/scripts/exit_timing.py run > /tmp/r173_exit.log 2>&1 < /dev/null &
# aggregate only:
venv/bin/python research/173_ipo_exit_next_open/scripts/exit_timing.py report
```
Do not edit research/153, 167 or 168 files; this study only reads them.

## Files

| File | Purpose | Committable |
|---|---|---|
| scripts/exit_timing.py | fork of simulate_ipo with exit-fill arms, runner + report | yes |
| results/seed_stats.csv | one row per (yield, window, arm, seed) | yes |
| results/summary.csv, paired.csv | medians and paired deltas | yes |
| results/gaps_by_reason.csv, floor_stats.csv | gap distribution, floor hits | yes |
| results/RESULTS.md | verdict | yes |

## Findings

**IMMATERIAL: the next-open exit does not cost, it adds.** Full period, 5.0% cash, 30 paired seeds, after tax:
A close 21.80 / -26.63 / 0.819; B next open 22.97 / -26.10 / 0.880 (+1.18pp CAGR, A wins 0/30, +0.048 Calmar);
C next open with 2% floor 23.42 / -26.16 / 0.895 (+1.59pp, 0/30, +0.100 Calmar). Both halves and 5.2% cash agree.
Overnight gap on exit signals: mean +0.46%, median +0.42%, p10 -1.07%, p90 +2.00%, 4.3% below -2%.
Floor hits: 5.3% of exits (stops 10.6%), 1.1 extra days on average, max 2. Cost side: B worst-seed DD -34.19 vs -32.88.
Do not build a 15:05 close-proxy exit. Full write-up: results/RESULTS.md
