# Redeploying the IPO Sleeve's Idle Cash into Open Alpha / True North — Daily Sweep

**STATUS: DONE** — all phases complete. **Verdict: CONCLUDED — the idle cash stays in cash.**
Full findings: `results/RESULTS.md`
Study: `research/155_ipo_cash_redeployment` · Host: **VPS 94.136.185.54** (canonical)
Started: 05-Sep-2026 · Author: quant-researcher agent
Extends: **research/153** (IPO-Base MID sleeve) and **research/154** (six-sleeve blends)

---

## 1. Headline

The r/153 IPO-Base MID sleeve is **32.7% invested on average**. Two thirds of its capital
sits in cash at 5% p.a. because the Indian IPO pipeline swings from 8 listings (2014) to 182
(2025). In **2013 and 2014 the sleeve took zero trades** and returned exactly the idle-cash
yield, while Open Alpha returned +14.0% and +77.7%.

**Question:** does redeploying that idle cash into Open Alpha (OA) and/or True North (TN) —
with *fully modelled* pull-back friction — beat leaving it in cash, on the same paired paths,
after tax?

---

## 2. The Ask

### What Arun asked (verbatim)

> "since ipos listing within 6 months is our candaites, but at the same time, ipo listing
> within a day or 2 cannot be our candidate (pls confirm), we would have an idee before hand
> if there are any listings or potential candidates that might meet our critreia at all, in
> which case we can look at deploying the idle cash in case of no canidates for the forward x
> period into OA and/or TN and then when the supply comes through, a mechanism to pull back
> money from cash reservers/oa/tn... did we try this approach?"

And on the design:

> "Pull-back friction must be modelled, not waived - agree to this... proceed"

### Confirming Arun's premise (the "pls confirm")

**Confirmed, and it is structural.** The adopted r/153 spec requires `min_bars = 25` (25
trading bars of history) and a 25-trading-day base window (`L = 25`). A name listed one or
two days ago therefore **cannot** produce a signal for ~5 weeks. Combined with the ≤6-month
age ceiling, the set of names that can possibly trigger over the next 25 trading days is
**fully determined by listings that have already happened** — which is why forward visibility
here carries **no look-ahead bias**.

### Three facts already established in r/153 / r/154 (sanity-checked, not re-derived)

1. The eligibility lag above → forward visibility is causal.
2. IPO supply is **not** a market-regime signal: corr(listings, NIFTY same year) = −0.01,
   corr(listings, OA same year) = +0.18, corr(listings, NIFTY prior year) = +0.16,
   corr(listings, OA prior year) = +0.35. 2014 had 8 listings while NIFTY rose 31.6%; 2009
   had 15 while NIFTY rose 75.6%. **The idle cash is genuinely idle, not conditionally
   protective.**
3. r/153 tested **fixed weights only**. Its "cash-null" was a *control* (sleeve replaced
   entirely by cash), **not** a redeployment arm. Nothing dynamic has ever been simulated.

### What we are actually testing

For each of 30 paired paths, simulate the **IPO sleeve at position level** with an external
cash sink/source: idle cash above a reserve is parked in a chosen asset (OA / TN / 50-50 /
NIFTYBEES), pulled back when an IPO candidate triggers, paying **every** real friction —
transaction cost, tax on realised gain, T+1 settlement delay, and the choice of which lot is
liquidated. Then blend the resulting sleeve NAV into 40/40/20 TN+OA+IPO, monthly rebalanced,
and compare against the incumbent (idle → cash) blend **on the same paths**.

---

## 3. The Base — what is being tested

### 3.1 The sleeve (unchanged from r/153, `results/ipo_adopted_spec.json`)

| Dial | Value |
|---|---|
| Universe | NSE dailies with a vetted listing date (r/153 `listing_dates.csv`, 1,293 accepted), ETFs excluded, pre-listing junk rows masked |
| Recency | listed ≤ **6 months** ago **and** ≥ **25 trading bars** of history |
| Base | last **25 trading days**; pivot = highest **close**; depth (pivot→lowest low) ≤ **30%** |
| Not extended | `close[t−1] < pivot` |
| Liquidity | 20-day median traded value at t−1 ≥ **₹5 cr** |
| Trigger | `close[t] > pivot` |
| Fill | **buy-stop AT the pivot**, filled `max(pivot, open[t])` |
| Hard stop | close ≤ buy × 0.92 (−8%, on the close) |
| Trail | close < **SMA-20** (not on the entry day) |
| Take profit | **+25%** on the close |
| Book | ₹10,00,000, **8 slots @ 18.75%** of sleeve NAV, cash-constrained, no leverage |
| Market gate | **OFF** |
| Costs / tax / cash | 25 bps per side · 20% STCG / 12.5% LTCG, FY (1-Apr) netting with loss carry-forward · idle cash **5% p.a.** |

**Nothing in the signal or exit logic is changed by this study.** The only new machinery is
what happens to cash the sleeve is not using.

### 3.2 The parking mechanic (new)

Daily order of operations inside the sleeve:

1. accrue 5% p.a. on **settled cash only**;
2. receive any cash in settlement (T+1 pipe);
3. settle the financial-year tax bucket on 1 April;
4. **entries** — if settled cash is short, force-redeem parked units (see friction below);
5. **exits** at the close, proceeds to cash (instant, as in r/153, so the comparison stays
   paired — the T+1 pipe applies only to the *new* parking mechanism);
6. **re-park** cash above the reserve, on the chosen cadence.

Sleeve NAV = settled cash + cash in settlement + parked value + marked open IPO positions.

### 3.3 Pull-back friction — modelled, not waived

| Friction | How it is modelled |
|---|---|
| **Transaction cost** | 25 bps per side on **both** the redemption and the re-parking of the parked leg (ladder 25 / 40 / 60) |
| **Tax on realised gain** | Two treatments, both reported (see below) |
| **Settlement delay** | **T+1 primary**: cash from a sale on day *t* arrives on *t+1*, so the IPO entry that triggered the sale is **missed**. **T+0 arm** run separately to isolate how much of any edge is a settlement artefact |
| **Which lot is sold** | **pro-rata** (average cost across all lots) · **LIFO** (most-recent parked first — least unrealised gain) · **FIFO** (oldest first — larger gain, but LTCG rate) |
| **The smarter mechanic** | A **liquidity reserve** of *k* slot-sizes held in settled cash, plus a slower **re-park cadence** (daily / weekly / monthly), so IPO entries are funded from the sleeve's own natural exits and uninvested cash first, force-trimming only when short. Arms `k ∈ {0,1,2}` × cadence |

**Tax treatment — why two, and which is primary.** `oa_navs30.csv` and `tn_navs12.csv` are
**already after-tax** NAV series (r/154 `build_sleeves.py` passes `stcg=0.20, ltcg=0.125`).
Taxing the NAV-lot gain again therefore **double-counts**. Both are run:

- **`tax=full`** — Arun's literal instruction: tax every pull-back's NAV-lot gain at
  20% STCG / 12.5% LTCG with FY netting. This **double-counts** and is therefore a strict
  **upper bound on friction**. **The adoption decision is made on this arm** (conservative).
- **`tax=txn`** — transaction cost only; the parked leg's gains were already taxed inside its
  own NAV, so a forced sale costs only the trade plus a timing effect. This is the
  economically correct **lower bound**.

The truth lies between them, and the gap is itself a reported number.

### 3.4 The arms

| Arm | Parked asset / rule |
|---|---|
| **A — Incumbent (control)** | no parking; idle cash at 5% p.a. This is r/153's adopted sleeve |
| **B** | idle → **Open Alpha** (the same OA seed as the blend's OA leg) |
| **C** | idle → **True North** (the same TN offset as the blend's TN leg) |
| **D** | idle → **50/50 OA + TN**, daily rebalanced |
| **N — null** | idle → **NIFTYBEES** (plain index beta). If redeployment only works because "more equity beta is good in a 20-year bull sample", this shows it |
| **E — forward-visibility gate** | park **only** while the eligible candidate pool is empty for the next **N ∈ {25, 50, 100}** trading days. This is Arun's actual proposal; A–D are the simpler bounds it must beat |

**Arm E's pool definition (causal).** At day *t*, using only information available at *t*:
name *c* is a potential candidate in (*t*, *t*+N] if (i) it clears the ₹5 cr liquidity floor
**as of t**, and (ii) it is inside the young window (`bars ≥ 25` and `age ≤ 6 months`) on some
day in (*t*, *t*+N]. Both bars-since-listing and calendar age are deterministic given a
listing date already in the past, so this uses no future prices. The two mild assumptions —
that the exchange trading calendar is known forward (it is published), and that a name liquid
today stays liquid — are stated as caveats, not hidden.

### 3.5 Controls and nulls

| Control | Purpose |
|---|---|
| **Cash-null** = arm A | the incumbent; the thing to beat |
| **NIFTYBEES null** = arm N | is this just equity beta? |
| **Static-tilt null** | a grid of *static* weight vectors TN/OA/IPO_A. If any static vector dominates the winning dynamic arm on **both** CAGR and MaxDD, redeployment adds nothing that simply holding more OA does not |
| **Cost ladder** | 25 / 40 / 60 bps per side — the ranking must not flip |
| **Tax ladder** | `full` (upper bound) vs `txn` (lower bound) |
| **Settlement** | T+1 vs T+0 |

### 3.6 Windows

- **Full:** 2006-01-01 → 2026-09-04.
- **Drought sub-windows:** 2008-01 → 2009-12 and 2012-01 → 2014-12; **2013–2014 isolated**
  (the sleeve took zero trades).
- **Boom sub-window:** 2021-01 → 2026-09.

> **Drawdown convention (binding, r/154).** Every sub-window drawdown is measured from the
> **running peak of the FULL curve**, never from the window's first bar. Measuring within a
> slice reported −2.4% for 2008 where the truth was −16.5%.

### 3.7 Paths (30 paired)

Path *p* (p = 0…29) = **OA seed s(p+1)** from `oa_navs30.csv` × **IPO simulation seed p+1** ×
**TN offset (p mod 12)** from `tn_navs12.csv`. Every arm is run on the **same 30 paths**, and
every A-vs-B comparison is the **distribution of paired differences**, never unpaired medians.

*Deviation from the brief, stated:* the brief suggested 10 OA seeds × 3 TN offsets. r/154
established that TN offsets 0/4/8 alone **miss both tails** of TN's 14.9%–25.0% CAGR range.
The construction above keeps exactly **30 paired paths** (so the pre-registered "≥26 of 30"
bar is unchanged) while covering all 12 TN offsets and 30 distinct OA seeds.

---

## 4. The Plan — grid, cell count, and the **pre-registered** adoption bar

### 4.1 Phases and cells

| Phase | What | Cells |
|---|---|---|
| **R** | Replication check: reproduce r/153's 40/40/20 blend (28.27% / −12.79% / 2.21) from the cached inputs. **If this fails, STOP and report.** Also reproduce the arm-A sleeve exactly against `ipo_equity_seeds.csv` | 1 |
| **1** | Arms A / B / C / D / N at default mechanics (T+1, reserve k=0, cadence daily, pro-rata, `tax=full`, 25 bps) | 5 |
| **2** | Mechanics sweep on the winning asset: settlement {T+0, T+1} × reserve {0,1,2} × cadence {daily, weekly, monthly} × sell {pro-rata, LIFO, FIFO} | 54 |
| **3** | Arm E forward-visibility gate: N {25, 50, 100} × asset {OA, TN, 50/50} at the best mechanic | 9 |
| **4** | Ladders and nulls: cost {25,40,60} × tax {full, txn} on the winner, plus the static-tilt null grid | ~12 |
| **Total** | **disclosed for multiple-testing** | **≈ 81** |

Each cell is a **30-path ensemble** → ≈ 2,430 position-level sleeve simulations.
The best cell's numbers are discounted for 81 tests; the **median cell**, not just the best,
must also clear.

### 4.2 Pre-registered adoption bar (fixed before the first run — not to be relaxed)

A redeployment variant is **adopted** over the incumbent (idle → cash) only if, versus the
**incumbent 40/40/20 TN+OA+IPO blend on the same 30 paired paths, after tax**:

1. **+0.10 Calmar OR −2 pp MaxDD at ≥ equal CAGR**, **and**
2. it **wins on ≥ 26 of the 30 paired paths** (paired, not an unpaired median), **and**
3. it does **not** worsen MaxDD by more than **1.5 pp** in **either** drought sub-window
   (2008-01→2009-12 and 2012-01→2014-12), measured from the **full curve's running peak**,
   **and**
4. it **survives the cost ladder** (25 / 40 / 60 bps per side) **without the ranking
   flipping**, **and**
5. it is **not dominated** by any static weight vector on both CAGR and MaxDD
   (the static-tilt null), **and**
6. the sleeve's **correlation to OA and TN stays below ~0.4** — the property that got IPO
   admitted in the first place (0.16 / 0.18 daily). If redeployment pushes this up
   materially, the sleeve is losing the reason it exists.

**If it clears CAGR but fails Calmar, the verdict is SIGNAL, not adopted, and it will be said
plainly.** r/152 is the precedent for holding a bar the outcome wanted relaxed.

### 4.3 Headline number Arun asked for first

**Total friction drag in pp of CAGR** — the cost of the mechanism, reported before the
benefit: `CAGR(arm, frictionless) − CAGR(arm, all frictions on)`.

### 4.4 Diagnostics reported per year

blend return · blend intra-year DD (full-curve peak convention) · **% of the IPO sleeve's
capital actually redeployed** · **number of pull-back events and their cost** · number of
entries missed to T+1 settlement.

---

## 5. Status (live log)

**Phase: COMPLETE.** 114 cells × 30 paired paths = 3,420 position-level sleeve simulations.
Host VPS. All rows in `results/paths.csv`; verdict in `results/RESULTS.md`.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 05-Sep-2026 19:50 | Study folder created, STATUS §1–4 written | Adoption bar pre-registered above |
| 05-Sep-2026 19:58 | **Phase R PASSED** | Arm A reproduces r/153's `ipo_equity_seeds.csv` **bit-for-bit** (max abs NAV difference **0.0** across all 30 seeds × 5,128 days). `replicate.py` reproduces the published blend exactly: 27.14 / −16.42 / 1.65 at w=0, 27.72 / −14.44 / 1.92 at w=10%, **28.27 / −12.79 / 2.21 at w=20%** |
| 05-Sep-2026 20:02 | **Phase 1 done** (5 cells + 3 frictionless twins) | Naive continuous parking is a decisive failure — see below |
| 05-Sep-2026 20:04 | Phase 2 launched (54 mechanic cells on asset = OA) | ~7 s/cell |
| 05-Sep-2026 20:09 | Phase 2 done | **No mechanic rescues it.** Best realistic (T+1) cell `r0_monthly_lifo`: blend 29.16 / −17.24 / **Calmar 1.69** vs incumbent 2.18 — Δ Calmar **−0.49, 0/30 paired wins**. Best T+0 cell (settlement waived) `r1_weekly_prorata`: 30.47 / −16.95 / 1.81 — **+1.6pp CAGR, 30/30 wins on CAGR, but −0.37 Calmar and 0/30 on Calmar**. Redeployment buys return and pays for it in drawdown; the drawdown cost dominates |
| 05-Sep-2026 20:11 | **Phase 3 done — the gated arm (Arun's actual proposal) is SAFE but IMMATERIAL.** The candidate pool is empty on **19.0% of days**, and *identically so for N = 25 / 50 / 100* — the droughts are multi-month, so the look-ahead horizon does not matter. Zero entries missed (`n_missed = 0` on every path), only ~30 pull-backs in 20 years. Blend **29.01 / −13.65 / 2.150** vs incumbent **28.92 / −13.59 / 2.181**: median paired **ΔCAGR +0.096pp, 30/30 paths**, but **ΔCalmar +0.007, only 21/30** — a tenth of a percentage point, nowhere near the pre-registered +0.10 Calmar bar. Correlations stay inside the ceiling (0.28 OA / 0.26 TN). Reserve = 2 slots was throttling it → Phase 3b re-runs the gate with no reserve |
| 05-Sep-2026 20:25 | **Phase 3b / 4 / 5 done, report + factsheet written, STUDY COMPLETE** | Gate without the reserve moves 13.4% of sleeve NAV (18.5% daily) and still only buys **+0.105pp CAGR / +0.006 Calmar**. Cost ladder kills it: **+0.105pp at 25 bps → +0.005pp at 40 → −0.128pp at 60**. Static-tilt null kills it: **TN 35 / OA 35 / IPO 30 static returns 29.39% at −13.64%** vs the gated machinery's 29.02% at −13.66%. Best gated parking asset is the **NIFTYBEES null**, i.e. plain beta. All four headline arms **REJECT** on the pre-registered bar |

### Live findings

**The paired incumbent baseline (30 paths, monthly-rebalanced, after tax):**

| Book | CAGR | MaxDD | Calmar | DD 2008–09 | DD 2012–14 |
|---|---|---|---|---|---|
| TN+OA 50-50 (the deployed pair) | 27.85 | −17.18 | 1.67 | −16.63 | −8.15 |
| **40/40/20 + IPO idle→cash (INCUMBENT, to beat)** | **28.92** | **−13.59** | **2.18** | −12.55 | −4.36 |

**Confirming the r/154 drawdown-convention correction on this study's own numbers:** the
40/40/20 blend's 2008 drawdown is **−1.66%** measured inside the 2008 slice and **−12.23%**
measured from the full curve's running peak. Everything in this study uses the latter.

**Phase 1 — naive continuous parking (T+1 settlement, no cash reserve, daily re-park,
pro-rata lots, `tax=full`, 25 bps). Medians over the same 30 paired paths:**

| Arm | Parked asset | Sleeve CAGR | Sleeve DD | Sleeve trades | % invested in IPOs | % parked | Blend CAGR | Blend DD | Blend Calmar | Δ Calmar vs incumbent | wins/30 | corr(daily) to OA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A incumbent** | cash | 31.38 | −20.88 | 674 | 32.7% | 0% | **28.92** | **−13.59** | **2.18** | — | — | 0.21 |
| B | Open Alpha | 22.45 | −37.22 | **183** | **8.6%** | 88.0% | 26.86 | −19.55 | 1.38 | **−0.77** | **0/30** | **0.90** |
| C | True North | 15.01 | −32.92 | 196 | 9.2% | 86.2% | 25.35 | −18.67 | 1.41 | −0.75 | 0/30 | 0.40 |
| D | 50/50 OA+TN | 19.11 | −32.73 | 187 | 8.7% | 87.8% | 25.98 | −19.18 | 1.40 | −0.75 | 0/30 | 0.76 |
| N (null) | NIFTYBEES | 7.80 | −62.23 | 191 | 9.2% | 87.3% | 23.93 | −25.29 | 0.95 | −1.21 | 0/30 | 0.32 |

**Why it fails so completely, and it is not the tax.** With no cash reserve, *every* IPO entry
needs a pull-back: 1,206 pull-back events per path — and under **T+1 every one of them missed
its trade** (`n_missed = n_pull` exactly). The sleeve's trade count collapses from **674 to
183** and its invested share from 32.7% to 8.6%. The sleeve stops being an IPO sleeve.

**The frictionless twin proves the bar is not just friction.** Arm B with zero transaction
cost, T+0 settlement and no tax on the parked leg: sleeve 48.17% CAGR / −27.40% DD, blend
**31.60% / −16.55% / Calmar 1.91**. Blend CAGR +2.68pp, but Calmar still **below** the
incumbent's 2.18 and drawdown 3.0pp worse — and daily correlation to OA rises 0.21 → 0.72.
**Total friction drag on arm B = 4.74 pp of blend CAGR** (31.60 → 26.86); at sleeve level a
staggering 25.7 pp (48.17 → 22.45).

**Criterion 6 (correlation) is already breached by every continuous-parking arm**, frictionless
or not. Parking IPO's idle cash in OA turns the sleeve into 88%-OA — it stops being the
uncorrelated thing that earned it a place in the book.

---

## 6. Crash recovery — how to resume without Claude

All compute is on the **VPS 94.136.185.54**, user `arun`, repo `/home/arun/quantifyd`,
interpreter `venv/bin/python`. SSH key auth works from Arun's laptop (`ssh arun@94.136.185.54`).

```bash
# 1. what finished?
ssh arun@94.136.185.54 'cd /home/arun/quantifyd/research/155_ipo_cash_redeployment && \
    ls -la results/ && tail -40 results/*.log'

# 2. is anything still running?
ssh arun@94.136.185.54 'pgrep -af "155_ipo_cash|redeploy" ; uptime'

# 3. resume — every runner is resume-safe (it skips cells already present in its CSV)
ssh arun@94.136.185.54 'cd /home/arun/quantifyd && flock /tmp/qf_sweep.lock \
    setsid nohup venv/bin/python -u research/155_ipo_cash_redeployment/scripts/run_sweep.py \
    > /tmp/r155_sweep.log 2>&1 < /dev/null & sleep 2; pgrep -af run_sweep.py'

# 4. aggregate only (if the sweep finished but the report died)
ssh arun@94.136.185.54 'cd /home/arun/quantifyd && \
    venv/bin/python research/155_ipo_cash_redeployment/scripts/report.py'
```

**Do NOT touch:** anything under `services/`, any crontab, the systemd unit, or
`backtest_data/market_data.db`. This study is read-only against the database and writes only
inside `research/155_ipo_cash_redeployment/results/`.

**Safe to inspect:** every file under `research/155_ipo_cash_redeployment/`, and the cached
inputs `research/154_multi_system_blends/results/{oa_navs30,tn_navs12}.csv`,
`research/153_ipo_base/results/{ipo_equity_seeds,listing_dates}.csv`.

**No backend restart is required by this study at any point.**

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `IPO_CASH_REDEPLOYMENT_DAILY_SWEEP_STATUS.md` | this file — sole crash-recovery source | yes |
| `scripts/ipo_park.py` | position-level IPO sleeve with an external cash sink/source | yes |
| `scripts/run_sweep.py` | phase runner, resume-safe, one CSV row per completed cell | yes |
| `scripts/report.py` | blend arithmetic, YoY house table, paired stats, charts | yes |
| `results/p*_*.csv` | one row per cell | yes (small) |
| `results/nav_*.csv` | sleeve NAVs per arm, 30 seeds | yes if small |
| `results/*.log` | run logs | yes (small) |
| `results/RESULTS.md` | final findings + verdict | yes |
| `results/ipo_cash_redeployment_research155.png` | factsheet | yes |

---

## 8. Findings

**VERDICT: CONCLUDED — the idle cash stays in cash.** Continuous redeployment is **NO EDGE**;
the forward-visibility gate is a real but **immaterial SIGNAL** that fails the pre-registered
bar and dies on the cost ladder. See `results/RESULTS.md` for the full package.

**The one-line reason:** the IPO sleeve is 20% of the blend and sits 67.3% in cash, so the idle
cash is 13.5% of the portfolio — but the candidate pool is empty on only **19.0% of days**, so
Arun's gate can only ever touch **2.7% of the portfolio**. A 2.7% tilt cannot move a Calmar.
Redeploying the *whole* 13.5% (continuous parking) does move it: **+1.54pp CAGR for −3.85pp of
drawdown, Calmar −0.375 on 30/30 paths**, and the sleeve's correlation to Open Alpha goes from
**0.21 to 0.90** — it stops being the uncorrelated thing that earned it a place in the book.

**Friction drag (the headline Arun asked for first):** **0.28 pp of blend CAGR on the gated arm
(73% of its gross benefit)** and **5.26 pp on the continuous arm** — enough to flip +3.30pp into
−1.95pp. But friction is not what kills the idea: frictionless, continuous parking still loses
28 of 30 paths on Calmar.

**Kept for the record:** T+1 settlement is catastrophic for continuous parking (1,206 pull-backs,
**all 1,206 missing their trade**) and essentially free for the gated design (**0 missed entries
in 20 years**), because the gate guarantees N days ahead that nothing can trigger. If the IPO
sleeve is ever run at a much larger weight, that is the only structure worth revisiting —
registered as a dated review for **31-Mar-2027**.
