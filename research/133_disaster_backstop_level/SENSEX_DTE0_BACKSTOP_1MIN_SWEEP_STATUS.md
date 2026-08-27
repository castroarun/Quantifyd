# SENSEX DTE0 Disaster Backstop — what level should the stop-less TimeB straddle carry?

**STATUS: DONE** (2026-08-27) — verdict: **KEEP `BACKSTOP = 0.50`**, no live change

Study: `research/133_disaster_backstop_level`
Book under study: **CSL_TIMEB_SENSEX, DTE0 cell** — `entry 13:00, exit 15:20, sl "none"`, **6 lots
= qty 120** (scaled 8→6 on 2026-08-27, effective the next 09:12). SENSEX lot = **20**, so
1 premium point = ₹20/lot = **₹120 at 6 lots**.

---

## 1. The Ask

**What Arun asked (2026-08-27):**
> "we cannot hv 50% disaster combined SL... we need to work on this as well, we shud not only
> depend on our 80+ days of options data in database, but take price action's support also which
> we have for a few years."

**What we are actually testing:**

> For a stop-less short ATM straddle on SENSEX expiry day (DTE0), held 13:00 → 15:20, **what
> combined-premium backstop level — expressed as a multiple of the entry credit — is correct,
> judged on a multi-year tail estimate rather than on the ~85 recorded options days alone?**

The methodological instruction is the point of the study. A *disaster* stop by construction
protects against events that have not happened in the four months of recorded chain. The
85-day options window can measure **fidelity** (what a level does to real P&L when it fires);
it cannot measure **frequency of the disaster**. The multi-year index price-action sample must
carry the tail, bridged into premium space.

`BACKSTOP = 0.50` lives at `research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py:62`. Any
book cell whose config carries `"sl": "none"` exits when combined premium ≥ **1.50 × credit**,
confirmed by a 2-consecutive-poll dwell (5 s polls). In production this applies to
**CSL_TIMEB_SENSEX DTE0** and nothing else.

**Worked example, today 2026-08-27:** credit 231.63 → backstop at combined **347.44** → loss cap
≈ **−₹13,900 at 6 lots** (−₹2,317/lot before slippage). Is that number right, or an artefact of
a round 50%?

**No live change is made by this study.** The recommendation goes to Arun for sign-off. Next
DTE0 is Thursday **2026-09-03**.

---

## 2. The Base — what is being tested

**Construction (the live rule, replayed):**
- SENSEX ATM straddle, front expiry (= the same day, DTE0), strike = `round(spot/100) * 100`
- Sell at **13:00**, cover at **15:20** or on the backstop, whichever first
- Backstop fires when `combined_premium ≥ credit × (1 + L)`
- **Dwell model:** the live executor requires 2 consecutive polls above the level and exits at
  the *next* poll. On 1-minute data this is modelled as: breach detected at minute *m*, exit at
  minute *m+1*'s combined premium. The same-minute variant is reported as a sensitivity.
- **Costs:** the measured outcome-aware model from r/122 (`cost_per_lot()` in
  `research/122_window_risk_atlas/scripts/stage_a_alldays.py`) — entry slippage 0, time/EOD exit
  **+0.178 pt** per leg-side, forced/stop exit **+6.548 pt** per leg-side, plus the exact Zerodha
  F&O option rate card. A backstop exit therefore costs **2 × 6.548 = 13.10 pts** of slippage
  = ₹262/lot = **₹1,571 at 6 lots** on top of the level itself.

**Levels tested (L, as a fraction of credit):** 0.25, 0.30, 0.35, 0.40, **0.50 (deployed)**,
0.60, 0.75, 1.00 — plus fixed rupee-per-lot equivalents ₹1,000 / 1,500 / 2,000 / 2,500 / 3,000 /
4,000 / 5,000 per lot, so a level that is credit-invariant can be compared against one that is not.

**Sizing:** every rupee figure is quoted **per lot** and **at 6 lots (qty 120)**.

**Success criterion.** A backstop is insurance. It is judged on four numbers together, not one:
1. **fire rate** — on the options sample *and* on the bridged long sample. ≤ ~2% of sessions = a
   disaster stop; ≥ ~15% = a trading stop wearing the wrong name (and r/114/116/121/122/124/131
   have all shown that a trading stop on this book destroys the edge).
2. **save** — P&L with the level vs holding to 15:20, on the days it fires.
3. **cost** — what it gives up on the days it fires and the session then recovers.
4. **tail cap** — the p99/max loss it converts, at 6 lots, versus unstopped.

A level is only recommended if it sits on a **plateau** (its neighbours behave the same), not as
a lone winner.

---

## 3. The two samples — explicitly separated

### Stage A — recorded options chain (fidelity, small n)
`backtest_data/options_data.db :: option_chain`, **1-minute**, SENSEX, 2026-04-20 → 2026-08-26.
DTE0 = the day on which the front expiry equals the day itself (**derived from the chain, never
from the weekday** — SENSEX expiry moved Fri → Tue → Thu inside our history).
Guards carried from r/120/121/122: reject days with **< 50 distinct spot prints** (frozen-chain
holidays 2026-05-01, 2026-05-28, 2026-06-26), reject partial sessions (last snapshot < 15:15),
reject thin days (< 200 minutes).
**Expected n ≈ 17–18 DTE0 sessions.** That is the whole point: n=17 cannot price a disaster.

Also replayed for **reconciliation only**: the r/114 construction — 09:16 entry, hold to 15:15,
and the 30% per-leg stop — so this study's Stage A can be checked against r/114's published
**HOLD +2,630/lot at 92% win, n=12** before anything is interpreted.

### Stage B — multi-year index price action (the tail; the reason this study exists)
`backtest_data/market_data.db :: market_data_unified`, READ-ONLY, always pinned on symbol AND
timeframe:
- **SENSEX `minute`**, 2021-01-01 → now (~1,354 trading days)
- **NIFTY50 `5minute`**, 2015-01-01 → now (cross-check, and the only series in our data that
  contains **COVID March 2020** — the single most informative disaster observation available)

**The r/121 licence:** for the *maximum excursion inside a fixed window*, 5-minute bars equal
1-minute bars exactly (the high/low of the covering bars is the same set). Excursions are valid;
**paths are not**, so no 5-minute series is used for dwell / gap-through timing.

**DTE labelling is built from an expiry calendar, not a weekday.** For each era we take the
era's expiry weekday and, if that date is not a trading day in `market_data_unified`, walk back
to the previous trading day — so holiday-shifted expiries are labelled correctly.

| Venue | Era | Expiry weekday |
|---|---|---|
| SENSEX | 2024-01-01 → 2024-12-31 | Fri |
| SENSEX | 2025-01-01 → 2025-08-31 | Tue |
| SENSEX | 2025-09-01 → | Thu |
| NIFTY | 2019-02-11 → 2025-08-31 | Thu |
| NIFTY | 2025-09-01 → | Tue |

Before those dates no weekly expiry existed, so those days are **NOLABEL** and are used only in
the "all days" deep-tail scope, never as DTE0.

Per day we measure, inside **13:00 → 15:20**: the maximum absolute excursion from the 13:00
level, in **points and bp**, and the terminal move.

### The bridge — three of them, because one is not trustworthy

**B1 — linear slope (the r/122 method, reused not reinvented).**
`b = median over Stage-A NOSTOP DTE0 rows with excursion ≥ 20 bp of (MAE_frac ÷ excursion_bp)`.
A long-sample day breaches level `L` when `exc_bp ≥ L / b`.

**B2 — intrinsic floor (model-free, and it is the honest one for DTE0).**
On expiry day the combined premium is **never below intrinsic**: `combined ≥ |S − K|`. With
K ≈ S₀, an excursion of `E` points guarantees `combined ≥ E`. So level `L` is **certain** to fire
whenever `E ≥ (1 + L) × credit`. This needs no fitted slope and cannot be over-fit. It gives a
**lower bound on the fire rate** and an accurate loss estimate deep in the tail (loss ≥ (E − C)
points). It is the price-action support Arun asked for, in its cleanest form.

**B3 — observed worst** from Stage A itself.

**The bridge's limit, stated up front (r/122's finding):** bridged tails are **FLOORS**. r/122
found an observed worst that already exceeded its own bridged p99, because a violent move carries
an IV pop that an excursion-to-premium slope cannot see. Wherever B1, B2 and B3 disagree, the
**more conservative** figure is the one reported and the one the recommendation rests on.

---

## 4. Plan

| Stage | Script | Output |
|---|---|---|
| A | `scripts/stage_a_chain_backstop.py` | `results/stage_a_days.csv`, `results/stage_a_levels.csv`, `results/r114_reconciliation.txt` |
| B | `scripts/stage_b_longsample.py` | `results/stage_b_days.csv` |
| C | `scripts/analyse_backstop.py` | `results/analysis.txt`, `results/fire_rates.csv`, `results/save_cost.csv`, `results/tail.csv`, `results/RESULTS.md` |

**Grid.** 8 credit-relative levels × 7 rupee levels = **15 arms**, × 2 dwell variants = 30
Stage-A replays per day × ~17 days. Stage B: 2 series × ~1,354 + ~2,800 days × 15 arms via the
three bridges. Compute is trivial (minutes); the work is in the interpretation, not the cycles.

**Family-wise haircut.** 15 arms are screened. Any per-arm t is haircut by the Šidák factor for
15 tests, and no arm is promoted on a t that does not survive it. With n≈17 firing on a handful
of days, we expect **no arm to be statistically separable on Stage A** — that expectation is
itself part of the finding, and is why the long sample carries the verdict.

**OOS split** where n permits: Stage A split at the 2026-06-30 midpoint.

**Standing prior, stated honestly.** r/114, r/116, r/121, r/122, r/124 and r/131 have each found
that *tightening* a stop on this book destroys the edge, and r/131 found the venue book stop
(−₹3,000/lot on DTE0) already resolves 82% of expiry sessions before 15:15. A defensible answer
is therefore "50% is already close to right; the real protection is the book stop plus size."
If the numbers say that, the study says it plainly. If the long sample shows a level that
materially caps the tail for near-zero cost, that is the finding.

---

## 5. Status — live event log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-08-27 | Study opened, STATUS-MD written BEFORE compute | folder `research/133_disaster_backstop_level` |
| 2026-08-27 | Stage A run — recorded chain | 90 SENSEX recorded days -> **17 clean DTE0** sessions; 3 frozen holidays + 1 partial rejected as pre-registered |
| 2026-08-27 | r/114 reconciliation | HOLD 09:16->15:15 on r/114's exact 12 days = **+2,831/lot, 92% win** — ties r/131 to the rupee, ties r/114's win rate |
| 2026-08-27 | Stage B run — long sample | SENSEX 1-min **1,356** days 2021->2026-08-26; NIFTY50 5-min **2,754** days 2015->2026-07-17 |
| 2026-08-27 | Expiry-calendar bug found and fixed | single-symbol data holes (2026-05-14, 07-09 missing from SENSEX 1-min) were walking the expiry back a day and mislabelling neighbours; calendar rebuilt on the **union** of four series. Validation: 15/17 chain-derived 2026 DTE0 dates reproduced exactly, incl. both holiday-shifted Wednesdays |
| 2026-08-27 | Bridge built + validated | F(R) = max(0.331R, R-1); the linear and intrinsic routes cross at R=1.49; bridge conservative on **15 of 17** recorded days |
| 2026-08-27 | Stage C — fire / save / cost / gap-through / tail | all four scopes; long-sample save-vs-cost is the deciding table |
| 2026-08-27 | RESULTS.md written, STATUS closed | verdict **CONCLUDED — keep 0.50**; recommendation to Arun for sign-off before Thu 2026-09-03 |

---

## 6. Crash Recovery — how to resume without Claude

All work is on the VPS at `/home/arun/quantifyd/research/133_disaster_backstop_level`.
Everything is **READ-ONLY on the databases** and touches **no live or paper system**.

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd/research/133_disaster_backstop_level

# what finished?
ls -la results/
wc -l results/*.csv
tail -40 results/*.log

# re-run from scratch (each stage is independent and idempotent; total < 15 min)
cd /home/arun/quantifyd
nice -n 15 venv/bin/python3 research/133_disaster_backstop_level/scripts/stage_a_chain_backstop.py
nice -n 15 venv/bin/python3 research/133_disaster_backstop_level/scripts/stage_b_longsample.py
nice -n 15 venv/bin/python3 research/133_disaster_backstop_level/scripts/analyse_backstop.py
```

Stage C depends on A and B; A and B are independent of each other. Nothing here writes to any
DB, config, state file or service. **Do not** edit `csl_paper_exec.py` — this study makes no
live change.

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `SENSEX_DTE0_BACKSTOP_1MIN_SWEEP_STATUS.md` | this file | yes |
| `scripts/stage_a_chain_backstop.py` | recorded-chain replay | yes |
| `scripts/stage_b_longsample.py` | multi-year excursion clock | yes |
| `scripts/analyse_backstop.py` | bridge + fire/save/cost + verdict | yes |
| `results/stage_a_days.csv` | per DTE0 session, per arm | yes (small) |
| `results/stage_b_days.csv` | per day × window excursion | yes if < 5 MB |
| `results/fire_rates.csv`, `save_cost.csv`, `tail.csv` | the three headline tables | yes |
| `results/RESULTS.md` | final verdict | yes |

---

## 8. Findings

Full write-up in `results/RESULTS.md`. Headlines:

1. **Arun's premise is confirmed.** The recorded options window is a **calm slice**: its median
   DTE0 afternoon excursion is 78% of the multi-year DTE0 sample's, its p95 is 87%, and its
   **worst day is only 45%** of the long sample's worst. It contains **no disaster at all**.
   A disaster level set on it would be set on a sample from which disasters are absent.
2. **But the level it points to is the one already deployed.** The expectation cost of the
   backstop shrinks monotonically as the level widens and **crosses zero at L ~ 0.45-0.60 in
   all four independent scopes** (SENSEX DTE0 n=131, SENSEX all n=1,356, NIFTY DTE0 n=375,
   NIFTY all n=2,754). 0.50 is the corner: below it you pay, above it you buy nothing.
3. **0.50 is not a disaster stop.** It fires on 17.6% of the recorded sample and **~35-43% of
   DTE0 afternoons over the long run**. No tested level is a true disaster stop — even L=1.00
   fires on 15-28%. On this book a 1x-credit move is an ordinary event.
4. **What it does buy:** at 6 lots it turns the worst SENSEX DTE0 afternoon from **-Rs75,674
   into -Rs29,829** while *adding* Rs1,732 of expectation (27 rescues vs 26 regrets).
5. **The rupee cap is wrong in the commission's worked example.** Overshoot + the 2-poll dwell
   + the measured +6.548 pt forced-exit slippage make the real cap at today's credit
   **-Rs17,200 at 6 lots (median fire), -Rs21,400 (p90)** — not -Rs13,900.
6. **38% of L=0.50 DTE0 breaches gap through the level** inside one minute; the worst historical
   crossing cleared it by **1.82 credits**. The backstop is a budget, not a guarantee.
7. **r/118 reconciles.** The short 2024-2026 DTE0 window shows a worst of -12,612/lot only
   because it is short; deepen the sample (SENSEX 2021->, or NIFTY 2015-> with COVID) and the
   **afternoon window alone reaches -20,781 to -21,644/lot** — r/118's -21,500/lot full-day
   figure. **Plan against ~ -Rs126,000 at 6 lots unstopped.**
8. **Seventh reproduction** (after r/114, r/116, r/121, r/122, r/124, r/131) that tightening
   this book is destructive — and the first to locate where the destruction stops.

**No live change made or recommended.** Next DTE0 is Thursday 2026-09-03.
