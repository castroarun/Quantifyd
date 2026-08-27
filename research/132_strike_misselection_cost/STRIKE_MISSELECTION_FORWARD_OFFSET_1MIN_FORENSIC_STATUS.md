# Strike Mis-selection — What the Spot-vs-Forward Gap Cost, and Whether NIFTY Shares It

**STATUS: DONE** — verdict **NO NET P&L COST, REAL RISK DEFECT**. Full findings in `results/RESULTS.md`.
(Sections 1-3 were locked before compute, per the mandatory convention.)

Study: `research/132_strike_misselection_cost` · Venues NIFTY + SENSEX · Chain window 2026-04-20 → 2026-08-27
Host: Contabo VPS `94.136.185.54`, `/home/arun/quantifyd`, all DBs opened **read-only**.

---

## 1. The Ask

**What you asked**

> The CSL daemon picked entry strikes as `round(index_ltp / step) * step`. On SENSEX the options
> price off a level that runs consistently ABOVE the `BSE:SENSEX` feed — measured 2026-08-27 at
> ~66 points, with put-call parity holding to within Rs1.35 across five strikes. A 66-point offset
> on a 100-point grid pushes the rounding into the wrong strike whenever the index sits low in a
> strike interval. The "ATM straddle" has routinely been struck 1-2 strikes off, making it a
> DIRECTIONAL position rather than a neutral one. A fix already shipped today (`019ae8f`), ported
> from `services/nas_atm_executor.py` ~641 (research/119b). Your job is NOT to design the fix — it
> is to price what the defect has already cost, and to check whether the same defect exists
> anywhere else.
>
> **Across all recorded history, what has strike mis-selection cost, and does it also affect NIFTY?**

**What we are actually testing** — four separable questions, in ascending order of how much they
matter:

| # | Question | Population | Statistic |
|---|---|---|---|
| Q1 | How big is the spot-to-forward basis, per venue? | every recorded minute, 90 days x 2 venues | `offset = F - spot`, distribution not mean |
| Q2 | How often does that basis flip the rounding? | same | `P[ round(F/step) != round(spot/step) ]` |
| Q3 | What did it cost in rupees on the trades actually taken? | 52 CSL records + 9 NAS-suite DBs | booked net vs forward-snapped counterfactual net |
| Q4 | What **unchosen directional risk** was carried? | every mis-struck entry | net short-straddle delta, expressed Rs per 100 index points |

Q4 is the one that matters most. A mis-struck straddle is a coin-flip on direction, so its *mean*
P&L cost can be ~zero while it is still a live risk-control defect. **Establishing that
distinction is the primary deliverable**, and a verdict of "it cost little" is an acceptable and
useful answer.

Two secondary questions fall out of Q1:

- **Q1b — is NIFTY affected?** NIFTY step is 50 and the observed gap looked small
  (27-Aug COMB: K=24300, CE 115.28 / PE 90.41 → forward = K+24.87, which still rounds to 24300).
  We must say plainly: materially affected, or effectively immune.
- **Q1c — is the gap stable or time-varying?** By time-of-day and by DTE. The forward snap is
  self-calibrating either way, so this does not change the fix — but the *magnitude* governs how
  often the rounding flips, and a swinging basis means the historical mis-strike rate is not a
  good forecast of the future one.

---

## 2. The Base — what is being measured

### 2.1 The forward, and why it is the right ATM

Put-call parity on European index options: `C(K) - P(K) = F - K`, so the synthetic forward is
`F = K + (C(K) - P(K))` and it is readable off any liquid strike. The delta-neutral straddle
strike is the one nearest **F**, not the one nearest the cash index. The two differ by the basis
(carry minus dividends) plus any feed lag or staleness in the cash print.

- **Reference strike** for reading F: the strike nearest the recorded `underlying_spot`, using
  the front (nearest non-expired) expiry only. Both legs must have a non-null LTP.
- **Robustness cross-check**: F is re-read at the two neighbouring strikes (+/-1 step). If PCP
  holds, the three readings agree; the spread between them is reported as the measurement noise
  floor, and any minute whose spread exceeds 0.25 x step is discarded as an unreliable print.

### 2.2 Mis-strike definition

For a given minute: `K_spot = round(spot/step)*step`, `K_fwd = round(F/step)*step`.
**Mis-strike** iff `K_spot != K_fwd`. Magnitude in steps = `|K_fwd - K_spot| / step`.

### 2.3 Unintended delta

For a **short** straddle at `K` when the forward is `F`:

- Back out sigma from the observed combined premium at the reference strike by Black-76 inversion
  (forward measure, r discounted out — the chain prices are already forward-referenced, so
  Black-76 with `F` is the correct model, not spot-BS).
- `d1 = (ln(F/K) + sigma^2 T/2) / (sigma sqrt(T))`, `D_CE = N(d1)`, `D_PE = N(d1) - 1`.
- Net position delta of a **short** straddle = `-(D_CE + D_PE) = 1 - 2*N(d1)`.
- Reported as **rupees per 100 index points** = `net_delta x qty x 100`, signed
  (positive = the book was long the index without anyone choosing to be).
- `T` in years from trading-DTE, floored at 1/(252*6.25) so a 09:16 DTE0 entry does not divide
  by zero. Sigma inversion by bisection on [1%, 400%]; failures are dropped and counted.

### 2.4 Books, exit rules, sizes

CSL daemon (`research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py`) — each record replays
under **its own** rule as recorded in `cfg`:

| Shape | Rule |
|---|---|
| `SL<n>` | combined stop at `(1+n/100)*credit`, 2-consecutive-poll **dwell**, exit next poll |
| `SLnone` | 50% disaster backstop (`BACKSTOP = 0.50`) — never truly stopless |
| `SLrsN` | `credit + N/lot` points — DTE-agnostic rupee stop (research/96 shape) |
| time | flat exit at the window end |
| `mgmt: trail` / `shift` | post-CSL management arms — **excluded from the counterfactual** (path-dependent re-entry cannot be honestly replayed at a different strike); reported for mis-strike/delta only |

Dwell on 1-minute bars is modelled **bounded both ways**: (a) *touch* — exit the first bar the
threshold is breached (pessimistic), (b) *dwell-2* — breached on two consecutive bars, exit the
third (matched to the live mechanic). Both are reported; dwell-2 is the headline.

Sizes: **SENSEX lot 20, NIFTY lot 65** — `option_chain.lot_size` is WRONG (reads 10) and is
ignored. `qty` and `lots` are taken from the record / DB row.

### 2.5 Costs — the MEASURED outcome-aware model

`cost_per_lot()` lifted verbatim from `research/122_window_risk_atlas/scripts/stage_a_alldays.py`:
entry slippage 0, time/EOD exit +0.178 pt per leg-side, forced/stop exit +6.548 pt per leg-side,
plus the exact Zerodha F&O option rate card (Rs20/order x 4 spread over `NLOTS_REF = 10`, STT
0.1% on sell notional, txn 0.03503%, IPFT, SEBI, stamp 0.003% on buy, GST 18%). The retired flat
Rs250/lot is **not** used. Because the stop slippage is 37x the time slippage, a counterfactual
that changes *whether* the stop fires changes the cost too — this is handled per-path, not as a
constant.

### 2.6 Guards

- **Frozen-chain holidays rejected**: `< 50 distinct spot prints/day` (catches 2026-05-01,
  2026-05-28, 2026-06-26).
- **Partial sessions rejected**: last snapshot before 15:15.
- **Thin days rejected**: fewer than 200 usable minutes.
- **DTE-era labelling mandatory** — trading-DTE from the chain's own front expiry, reported on
  every cut. Weekday reported alongside, since venue expiry weekday differs (NIFTY Tue, SENSEX Thu).
- **Read-only**: every connection opened `file:...?mode=ro`. No live or paper system, config,
  service or engine is touched.

### 2.7 The honest caveat, stated up front

**A counterfactual strike would have had its own path.** The forward-snapped straddle is a
*different* instrument: it collects a different credit, so its stop sits at a different level, and
it can stop out on a day the real one held (and vice versa). The replay is a genuine
re-simulation of the alternative path — not a re-pricing of the same path — but it is still one
draw, on 10 trading days of CSL history. Dispersion is reported, not just a mean, and the P&L
number is explicitly framed as low-power. The **frequency** and **delta** results (Q1/Q2/Q4) rest
on 90 days x ~370 minutes and are the statistically solid part of this study.

---

## 3. Plan

| Stage | What | Output |
|---|---|---|
| **A** | Offset atlas — every venue x day x minute: spot, F, offset, K_spot, K_fwd, mis-strike flag, PCP noise spread, trading-DTE, weekday | `results/offset_atlas.csv` |
| **B** | Entry audit — every CSL record + every NAS-suite DB entry: actual K, F at entry minute, K_fwd, mis-strike, steps off, net delta, Rs/100pt | `results/entry_audit.csv` |
| **C** | Counterfactual replay — actual-K replay (for reconciliation) and forward-K replay, each book's own rule, touch + dwell-2, MEASURED costs | `results/replay.csv` |
| **D** | Aggregation → per-book cost table, mis-strike rate per venue, unintended-delta table, NIFTY-vs-SENSEX, offset stability by ToD/DTE | `results/RESULTS.md` |

**Reconciliation gate (blocking).** Stage C's actual-K replay is compared against the booked `pnl`
on the 19 CSL records carrying `ce0`/`pe0`. The match rate is reported **before** any
interpretation. If it is poor, the counterfactual is reported as directional-only, not as rupees.

**No grid, no search.** This is a measurement, so the family-wise machinery does not apply. There
is exactly one counterfactual per entry — the forward-snapped strike — chosen a priori by the
already-shipped fix, not selected from a sweep.

### Expected population

| Source | Rows | Window |
|---|---|---|
| `option_chain` | 90 days x 2 venues (~370 min/day) | 2026-04-20 to 2026-08-27 |
| `csl_paper_state.json` `records[]` | 52 entries, 10 books | 2026-08-14 to 2026-08-27 |
| `nas_*_trading.db` / `sensex_*_trading.db` `nas_atm_positions` | 1,759 leg rows across 9 DBs | earlier, per-DB |

The NAS DBs are also the **control**: the forward snap has been in `nas_atm_executor.py` since
`57eb8c2` (2026-06-01, "ATM straddles enter at the forward price for balanced legs"). NAS entries
after that date should show a mis-strike rate near zero. If they do, the method is validated; if
they do not, the fix is not doing what it claims and that is a finding in its own right.

---

## 4. Status — live event log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-08-27 13:25 | Folder created, STATUS-MD written | sections 1-3 locked before compute |
| 2026-08-27 13:33 | Stage A launched | 90 candidate days x 2 venues off the 12.7 GB chain DB |
| 2026-08-27 13:49 | **Stage A DONE** — 61,142 minute-rows, 85 days kept per venue (5 skipped by the holiday/partial guards) | `results/offset_atlas.csv` |
| 2026-08-27 13:52 | **Spot-feed validation PASSED** | DTE0 median offset NIFTY +1.4 / SENSEX +0.4, flat across every DTE0 hour. The forward collapses onto the cash index at expiry, so the recorded spot is sound and the gap is a genuine forward basis, not a bad feed. |
| 2026-08-27 13:53 | **Headline finding: NIFTY is NOT immune** | mis-strike on 36.2% of NIFTY minutes vs 50.3% of SENSEX minutes. NIFTY's basis is smaller in points but its grid is half as wide. |
| 2026-08-27 13:55 | Stage B launched (entry audit: 52 CSL records + 9 NAS DBs) | |
| 2026-08-27 14:01 | Stage C DONE — 51 CSL records replayed x 2 arms x 2 dwell models | `results/replay.csv` |
| 2026-08-27 14:05 | **Stage B DONE — 617 entries audited** (51 CSL of 52, 566 NAS straddle pairs) | `results/entry_audit.csv` |
| 2026-08-27 14:08 | **RECONCILIATION GATE PASSED** | exit-reason match 49/51 = **96%**, sign agreement 94%, credit err med 1.15 pt, gross err med 14-19% relative. Rupees quotable with a +/-15% band. |
| 2026-08-27 14:10 | **NAS control validates the fix** | mis-strike 42.9% pre-snap -> **9.9%** post-snap on NIFTY (57eb8c2, 2026-06-01). A 4.3x reduction across the ship date. |
| 2026-08-27 14:14 | **Counterfactual: the defect cost NOTHING** | forward-snapping is WORSE by Rs 24,992 over 37 entries; mean -Rs 1,000/mis-struck trade, **t = -1.87 (n=25) = indistinguishable from zero**. |
| 2026-08-27 14:18 | Stage E added — attribute real rupees to the accidental delta | not in the original plan; added because "Rs per 100 pts" is an exposure, not an outcome |
| 2026-08-27 14:22 | **THE FINDING**: the mis-strike was directionally BIASED, not a coin flip | every book's median delta is NEGATIVE (accidentally SHORT the index), because the basis is positive on 57/85 NIFTY and 65/85 SENSEX days. **|delta P&L| was a median 49% of the booked P&L on the same trade.** |
| 2026-08-27 14:26 | **RESULTS.md written, STATUS -> DONE** | verdict: NO NET P&L COST - REAL RISK DEFECT |

---

## 5. Crash Recovery — resume without Claude

All work is on the VPS at `/home/arun/quantifyd/research/132_strike_misselection_cost`.

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd/research/132_strike_misselection_cost
```

**What finished?**

```bash
ls -la results/                       # which CSVs exist
wc -l results/*.csv                   # how far each stage got
tail -40 results/stage.log            # the running log, one line per day/venue
```

**Is anything still alive?**

```bash
pgrep -af 'research/132'              # empty => nothing running
```

**Re-run a stage** (each is idempotent — it truncates and rewrites its own CSV, and reads nothing
but read-only DBs):

```bash
cd /home/arun/quantifyd
nice -n 15 venv/bin/python3 research/132_strike_misselection_cost/scripts/stage_a_offset_atlas.py
nice -n 15 venv/bin/python3 research/132_strike_misselection_cost/scripts/stage_b_entry_audit.py
nice -n 15 venv/bin/python3 research/132_strike_misselection_cost/scripts/stage_c_replay.py
nice -n 15 venv/bin/python3 research/132_strike_misselection_cost/scripts/stage_d_aggregate.py
```

Stage A is the long one (12.7 GB chain DB, 90 days x 2 venues; expect a few minutes).
Stages B/C/D are seconds. Stage D depends on A+B+C; the others are independent.

**Do NOT touch:** `backtest_data/options_data.db`, `backtest_data/csl_paper_state.json`,
`backtest_data/*_trading.db` — all read-only inputs to a LIVE book. Nothing in this study writes
outside its own `results/` folder. **Safe to inspect:** everything under this study folder.

---

## 6. Files

| File | Purpose | Committable? |
|---|---|---|
| `STRIKE_MISSELECTION_FORWARD_OFFSET_1MIN_FORENSIC_STATUS.md` | This file | yes |
| `scripts/stage_a_offset_atlas.py` | Per-minute spot-vs-forward atlas | yes |
| `scripts/stage_b_entry_audit.py` | Entry-by-entry mis-strike + delta audit | yes |
| `scripts/stage_c_replay.py` | Actual-K reconciliation + forward-K counterfactual | yes |
| `scripts/stage_d_aggregate.py` | Tables for RESULTS.md | yes |
| `results/offset_atlas.csv` | 61,142 minute rows (5.4 MB) | NO — gitignored; the day summary is committed instead |
| `results/entry_audit.csv` | one row per recorded entry | yes |
| `results/replay.csv` | booked vs actual-replay vs forward-replay | yes |
| `results/stage.log` | progress log | yes |
| `results/offset_by_day.csv` | 170 day-rows: per-day offset quantiles + mis-strike rate | yes |
| `scripts/stage_e_delta_attrib.py` | Rupees attributable to the unintended delta | yes |
| `scripts/qa_atlas.py` | Atlas QA incl. the expiry-convergence spot-feed test | yes |
| `results/aggregate.txt` / `results/delta_attrib.txt` | Full console output of stages D and E | yes |
| `results/RESULTS.md` | Final findings + verdict | yes |

---

## 7. Findings

**Full write-up: `results/RESULTS.md`.** Headlines:

### VERDICT: NO NET P&L COST — BUT A REAL, SYSTEMATIC RISK-CONTROL DEFECT

1. **It cost nothing.** Forward-snapping every CSL entry makes the book **worse by Rs 24,992**
   over 37 replayable entries (mean −Rs 1,000/mis-struck trade, **t = −1.87, n = 25**).
2. **But the direction was not random — it was systematically SHORT the index.** Every book's
   median net delta is negative, because the basis is positive on 57/85 NIFTY and 65/85 SENSEX
   days, so `round(spot/step)` lands *below* the forward. **|delta P&L| was a median 49% of the
   booked P&L on the same trade** — half the recorded P&L was an unchosen direction bet that
   happened to be short into a falling tape (+Rs 41,395 over 37 mis-struck entries).
3. **NIFTY is NOT immune — that prior is refuted.** 36.2% of all recorded NIFTY minutes and
   **72.2% of CSL NIFTY entries** were mis-struck, and in *rupees* NIFTY was the worse of the
   two (median Rs 2,750 per 100 index points vs SENSEX's Rs 1,007).

### Q1 — the gap is a genuine forward basis (validated by expiry convergence)

DTE0 median offset: NIFTY **+1.4**, SENSEX **+0.4**, flat across every DTE0 hour. The forward
collapses onto the cash index at expiry, so the recorded spot is sound and this is carry, not a
feed defect. PCP holds across three adjacent strikes to a median 0.60 / 1.65 pt.

| | NIFTY (step 50) | SENSEX (step 100) |
|---|---|---|
| median \|offset\| | 13.8 | 44.7 |
| median \|offset\| / step | **0.28** | **0.45** |
| **MIS-STRIKE RATE (all minutes)** | **36.2%** | **50.3%** |
| MIS-STRIKE RATE (CSL entries taken) | **72.2%** | **80.0%** |
| median unintended \|Rs per 100 pts\| | **Rs 2,750** | Rs 1,007 |

Basis by trading-DTE — monotone, ~0 at expiry:

| DTE | NIFTY med | NIFTY mis% | SENSEX med | SENSEX mis% |
|---|---|---|---|---|
| 0 | +1.4 | 11.9% | +0.4 | 18.5% |
| 1 | +10.8 | 32.3% | +25.9 | 46.7% |
| 2 | +16.6 | 45.6% | +40.7 | 45.0% |
| 3 | +10.5 | 37.1% | +66.0 | 67.0% |
| 4 | +22.6 | 55.8% | +89.6 | 78.7% |

### Q5 — stability: swings day-to-day, steady within a day

Between-day spread is **7.9x (NIFTY) / 6.7x (SENSEX)** the within-day IQR, and the basis
**changes sign** (July dividend season: NIFTY −1.1, SENSEX −8.7; August: +31.4 / +82.4). On
47% of SENSEX days the median offset alone exceeded half a step — the strike was structurally
wrong all day. The forward snap is self-calibrating so this does not change the fix, but it
does mean the historical mis-strike rate is **not** a forecast.

### The NAS control

| Venue | Era | n | mis-strike |
|---|---|---|---|
| NIFTY | pre < 2026-06-01 | 119 | 42.9% |
| NIFTY | post >= 2026-06-01 | 374 | **9.9%** |
| SENSEX | post >= 2026-06-01 | 73 | 17.8% |

A 4.3x reduction across the `57eb8c2` ship date — the method is validated and the fix works.
