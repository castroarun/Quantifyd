# IPO Base — minimum-bars floor 60 → 25, the value Spec A was validated at

**STATUS: DONE** — deployed 13-Sep-2026 (Sunday, non-trading day), verified.

## The ask

**What Arun asked:** "Capital Desk target from 40/40/20 to 37.5/37.5/25 is your call - lets do
this. but b4 that, pls chk this... what if we apply this to other universes". Earlier, on
12-Sep: "implement this in the current IPO system, make the required changes".

**What this change is:** research/169 answered the universe question (no transplant beats its
random control) and found that the live book was not running the spec it was funded on. The
25% weight is supported only for the validated spec, so the floor is corrected in the same
change as the weight.

## The base — what changed and what did not

| Rule | Before | After |
|---|---|---|
| Minimum bars since listing, at the signal date | 60 | **25** |
| Everything else in research/167 Spec A | unchanged | unchanged |
| Capital Desk targets | TN 40 / OA 40 / IPO 20 | **TN 37.5 / OA 37.5 / IPO 25** |

**Why 60 was wrong.** research/153's panel loader filters `... group by symbol) where n >= 60`.
That counts a symbol's rows over the whole database today. The rule that decides a signal is
`ctx.BARS >= min_bars` on the signal day, and research/167 ran it at 25. Verified by reading both
engines before the change.

**Evidence** (research/167's engine and panel, 30 seeds, after tax, idle cash 5.0%;
`research/169_ipo_rules_universe_transplant/results/s1b_minbars_r167engine.json`):

| Minimum bars | signals | CAGR | max drawdown | Calmar | edge over random (wins / 30) |
|---|---|---|---|---|---|
| 25 | 1,548 | 21.80% | −26.63% | 0.819 | +4.78pp (30) — +2.25pp on research/169's clean panel |
| 40 | 1,308 | 17.87% | −32.40% | 0.552 | −1.38pp (3) |
| 60 | 988 | 11.57% | −38.97% | 0.297 | +2.07pp (30) |

Inside TN 37.5 / OA 37.5 / IPO 25, the 60-bar book costs 2.46 points of blend CAGR on 30 of 30
paths (research/169 Q6).

## Plan

1. Engine: `MIN_BARS = 25`, the misreading explained in place, `SPEC_VERSION = 'r167-A-mb25'`.
2. `--migrate` to log the spec change on state (no stop changes: exits are untouched).
3. Full nightly cycle on the last completed bar, to refresh candidates and the page.
4. Capital Desk targets through `POST /api/sleeves/allocation/targets`, which writes its own
   changelog entry.
5. Register, IPO page, report and study entries corrected; report rebuilt; front end rebuilt.

## Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 13-Sep-2026 09:28 | research/169 findings verified | both engines read; capacity figures read from stage9_adoption.csv |
| 13-Sep-2026 ~09:45 | Capital Desk targets set to 37.5 / 37.5 / 25 | via the Capital Desk route |
| 13-Sep-2026 ~09:50 | engine deployed, migration logged, cycle run | gate ON, nothing armed for 14-Sep |
| 13-Sep-2026 ~10:00 | pages, register and report rebuilt | front end built on the VPS |

## What it means on 14-Sep-2026

The scan widens from 20 names in the age band to 52. CORDELIA (72 days listed, ₹27.5 cr a day)
closed above its pivot on 11-Sep. **Nothing is armed**, because NIFTYBEES closed 2.34% below its
150-day average, so the gate blocks new entries under either floor. The open position, KISSHT,
is past 50 bars and unaffected.

## Crash recovery — to revert without Claude

```
ssh arun@94.136.185.54
cd /home/arun/quantifyd
sed -i 's/^MIN_BARS = 25$/MIN_BARS = 60/' services/ipo_paper.py
venv/bin/python -c "import services.ipo_paper as m; print(m.MIN_BARS)"
# targets back to 40/40/20:
curl -s -X POST http://127.0.0.1:5000/api/sleeves/allocation/targets \
  -H 'Content-Type: application/json' \
  -d '{"targets":{"truenorth":0.4,"openalpha":0.4,"ipo":0.2},"note":"reverted"}'
```
No restart is needed: this book runs from cron (18:45 cycle, 09:35–15:35 reconcile, per-minute
mark). Do not hand-edit `backtest_data/ipo_paper_state.json`.

## Files

| File | Purpose | Committable |
|---|---|---|
| `services/ipo_paper.py` | the engine | yes |
| `backtest_data/allocation_targets.json` | Capital Desk targets and changelog | yes |
| `frontend/src/data/strategies.ts`, `frontend/src/pages/IpoPaper.tsx` | register and book page | yes |
| `frontend/src/data/mpf_report.ts`, `frontend/src/pages/MpfReport.tsx`, `research/_utilities/mpf_report_build.py` | the portfolio report | yes |
| `frontend/src/data/backtests.ts` | research/167 study entry corrected | yes |

## Open

- **Capacity of the earliest entries** as the sleeve grows. At ₹10L research/167's 90th-percentile
  position is 9.05% of a name's 20-day traded value — not the 1.56% first quoted, which was the
  median. Review 15-Oct-2026.
- **The edge is narrow.** It holds at 25 bars and fails at 40, and on a clean panel it ties random
  young names since 2016. The soak review should watch realised fills and selection, not just
  return.
