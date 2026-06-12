# Per-Stock Trailing-MA Exit (event-driven) vs the Nifty Gate — does it help?

**STATUS: DONE** · research/41 · Phase 28 · daily-marked engine · VPS canonical

**VERDICT (RESULTS_phase28.md): NO NET EDGE.** Standalone per-stock (no gate) KILLED —
DD −28.7 to −33% (≥ Phase 11's −30%) + tax-ruinous (L100 post-tax 34.2→18.1%). +gate
version a wash (best L200+gate Calmar 1.59 vs 1.54 but post-tax 27.6 vs 28.4). Dominated
by Phase 27 allcash+weekly (Cal 1.72 / net 29.0). Don't replace OR augment the gate.

## 1. The Ask

**What you asked:** *"Instead of / in addition to the Nifty gate — trail each stock by
its own 100/150/200 MA; when it closes below, exit (find an optimal exit point), then
look for other passing candidates."*

**What we're testing:** an EVENT-DRIVEN per-stock trailing-MA book. Each week, any
holding closing below its own L-day MA is sold and the freed slot is IMMEDIATELY
backfilled from the next RS candidate that passes the full screen (incl. above its own
L-MA). Sweep L ∈ {100,150,200}, run standalone (no market gate) AND with the Nifty
100-SMA gate. v1 exit = simple close-below-MA; "optimal exit point" refinements are G2.

**Prior art (don't re-test):** Phase 11 showed per-stock-SMA100 applied *at month-ends*,
standalone = −30.2% DD / Calmar 1.10 (vs gate −15.1% / 2.32). The NEW axis here is the
EVENT-DRIVEN weekly cadence + immediate backfill + longer MAs — untested.

## 2. The Base (unchanged core)

PIT mid band, RS-120 vs NIFTYBEES, q0.5, ATH≤10% AND above-own-L-MA entry, N=15,
top-22 buffer, monthly RS rotation, per-stock-L-MA + 12% trail at month-ends, 0.4% RT,
6.5% cash, 2014-2026, daily-marked. NEW: weekly per-stock-L-MA exit + immediate backfill.

## 3. Plan — 7 configs × {gross, post-tax@20%}

| # | Config | Market gate | Per-stock L |
|---|---|---|---|
| 0 | REF allcash+gate (locked, from p22) | yes | 100 (month-end only) |
| 1-3 | perStock L100/150/200, NO gate | no | event-driven weekly |
| 4-6 | perStock L100/150/200, +gate | yes | event-driven weekly |

**Sanity:** REF must reproduce Phase 27 BASE (34.2%/−22.2%). **Win bar:** a perStock
config beats the locked book on Calmar/post-tax, OR the no-gate variants beat Phase 11's
−30%/1.10 enough to be interesting.

## 4. Status

| Time (IST) | Event | Notes |
|---|---|---|
| 2026-06-10 ~13:1x | STATUS + runner staged, launching | sections 1-4 pre-launch |

## 5. Crash Recovery

- Script: `research/41_.../scripts/28_perstock_trailing.py` (VPS), venv python.
- Run: `cd /home/arun/quantifyd && nohup venv/bin/python3 research/41_midsmall400_mq_concentrated/scripts/28_perstock_trailing.py > /tmp/phase28.log 2>&1 &`
- Progress: `tail -f /tmp/phase28.log` (one line/config). Alive: `pgrep -af 28_perstock`.
- Output: `results/phase28_perstock.csv` (+ `_peryear.csv`), incremental. Don't touch market_data.db or scripts 01-27.

## 6-8. Files / Findings / Verdict

Pending run. Results → `results/RESULTS_phase28.md`.
