# NIFTY 45-DTE Short Straddle — Replicating "The Long & The Short Ep. 48" + Monitoring-Timeframe Bake-off

> **2026-08-31 — STRESS MARGIN ANSWERED (moneyness x DTE axis).** Measured against the broker instead of reconstructed: a straddle m%% offside == a strike m%% away today. **3 lots BREACH the Rs 11.96L reserve at an 8%% adverse move (110-115%% of capital) and sit at 95%% by 5%%** — and that is at India VIX 10.6, near the floor, so the true breach is below 8%%. Safe size at this reserve is **2 lots**. Go-live blocker as currently sized; nothing changed. See RESULTS.md section 7b. The VOL axis is still unmeasured (5 recorder days, 0.65 VIX pts) — review stays dated 2026-11-30.

**STATUS: DONE — verdict STRATEGY-CANDIDATE (G3 passed, G4 conditional pass). Full write-up in `results/RESULTS.md`.**
Owner: Claude · Host: Contabo VPS 94.136.185.54 · Started 2026-08-20

---

## 1. The Ask

**What Arun asked:** "Does 45 DTE Options Selling Really Work in India? — a straddle strategy as
explained in a YouTube video. Let's test it and come up with a simple report. Attached is their
report. We can first test this for the same period and confirm if the report is correct, we can
assume 10 lots of straddles and report."
Follow-ups: *"they did the testing on 1 hour timeframe, 1 hour candle close price... we can do that
and then 30 mins and more and see which works best"* and *"they are using this VIX percentile filter
criteria (vix today against the past 252 days/sessions)"*.

**What we are actually testing — three questions, in order:**

1. **Replication.** On REAL NSE NIFTY option prices, Jan-2019 → Jun-2026, does Sandeep Rao's
   published table reproduce? Target numbers to match: 83 trades, 69.9% win rate,
   exits 1 target / 4 stop / 78 time, avg premium 758.9 pts, total +5,951.6 pts,
   +71.7 pts/trade, avg win +196.1 / avg loss −216.8, best +805.3, worst −1,062.6, MaxDD −1,062.6.
2. **Monitoring frequency.** Their exits are checked on **1-hour candle closes** (7 windows/day).
   Does checking more often (30m / 15m / 5m) or less often (daily close) change the result — i.e.
   is the edge robust to how tightly you watch it, or is it an artefact of a coarse check?
3. **VIX percentile filter.** India VIX at entry ranked against the **previous 252 sessions**;
   trade only when rank > 25 / 50 / 75. Their claim: win rate rises to 85.7% at >75 with 21 trades.
   VIX plays no role after entry.

**Success criterion:** primary = **net points per trade** with its t-stat; secondary = win rate,
MaxDD in points, and the exit mix. A result only counts if it survives realistic cost and the
data caveats in section 3 are stated.

**Falsification (decided now):** if net points/trade is not distinguishable from zero after cost
(|t| < 2), or if the headline P&L is concentrated in <= 3 trades, or if it flips sign on the
monitoring-frequency axis, this is **NOT a strategy** regardless of how good the win rate looks.

---

## 2. Economic hypothesis (G0)

Short ATM straddles harvest the **volatility risk premium** — index implied vol has historically
sat above subsequent realized vol (research/89 measured mean INDIAVIX/RV = **1.28** on NIFTY).
The counterparty is hedgers and lottery-ticket buyers who overpay for convexity. At 45 DTE you
collect a large absolute credit (vega/theta rich, gamma still low), and you leave at 21 DTE
**before** gamma turns vicious — that is the whole design idea.

**Decay risk / prior:** research/89 already found the unconditional NIFTY monthly straddle to be
**net −20 bps** over 2015-26 — the VRP was real but did not cover the 2020/2022 tails, and the
post-2022 retail options-selling boom compressed it. So the prior on this being a live edge is
**low**, and the burden of proof is on the replication.

---

## 3. Data — what we have, and the one thing we do NOT have

| Need | Source | Coverage | Verdict |
|---|---|---|---|
| Real NIFTY option EOD prices | `market_data.db` -> `nse_options_bhav` (NSE bhavcopy) | **2011-01-03 -> 2026-07-21**, 5.13M NIFTY rows, real OHLC + settle + contracts + OI | OK — ground truth |
| NIFTY spot daily | `market_data_unified` NIFTY50 day | 2011 -> 2026-08-20 | OK |
| NIFTY spot 5-min | `market_data_unified` NIFTY50 5minute | 2015-02-02 -> 2026-07-17, 206,990 bars | OK (build 60m/30m/15m from this) |
| India VIX daily | `market_data_unified` INDIAVIX day | 2015-01-01 -> 2026-08-20 | OK — 252-session rank available from 2016 |
| Intraday option prices **2026-04-20 onward** | `options_data.db` -> `option_chain` | **28.3M REAL 1-minute NIFTY quotes**, 2026-04-20 -> 2026-08-20. BUT the recorder picks each contract up only **~27 days before expiry**, so it cannot host a 45-DTE ENTRY — it covers DTE 27->0 | OK for the monitoring question |
| Intraday option prices **before 2026-04-20** | — | **DOES NOT EXIST and CANNOT BE OBTAINED.** `option_ohlc` is empty; `nas_option_snapshots` are empty. **VERIFIED, not assumed:** Kite `historical_data()` returns `invalid token` for expired contracts — tested on NIFTY 24000/24050/24100 CE of the *July-2026* expiry (one month old) at both `60minute` and `day` | **HARD CONSTRAINT** |

**Consequence for question 2 (the timeframe bake-off).** We cannot replay real 1-hour option
prices back to 2019 — that data does not exist in this repo or at the broker. So the hourly/30-min
test is run on a **reconstructed intraday premium path**:

> intraday mark(t) = Black-Scholes straddle value using the **real NIFTY 5-min spot** at t,
> the real calendar time-to-expiry at t, and an IV **backed out of the previous session's real
> option closes** (causal — no same-day look-ahead). At each daily close the path is **snapped
> back to the real observed market premium**, so error cannot accumulate across days.

This is stated as a **modelled path, not real ticks**, and is bracketed two ways:
- **IV bracket:** re-run with the *same-day* close IV (anticipatory — overstates how fast the mark
  reacts to a vol spike). The truth on stop-timing lies between the two.
- **Touch bracket:** the real bhav daily **high/low of each leg** gives an absolute worst-case
  (`CE.high + PE.high`) and best-case (`CE.low + PE.low`) bound on what *any* intraday monitoring
  scheme could have triggered. If the 50%/200% levels are never touched inside that bracket,
  monitoring frequency provably cannot matter on that day.

Everything in Phase A (the replication) uses **real traded option prices only**.

**Seven deadly sins — how each is controlled**

| Sin | Control |
|---|---|
| Look-ahead | Entry uses that day's close only; intraday path uses *previous* day's IV; VIX rank uses trailing 252 sessions ending the prior session |
| Survivorship | N/A — single instrument (NIFTY index options), every monthly expiry taken, none skipped |
| Overfitting | Rules are fixed by the video, not fitted. Only 3 pre-declared axes (monitoring TF, VIX threshold, cost). Report the full grid, not the best cell |
| Cost neglect | Gross + net side by side; slippage sweep 0.25 / 0.5 / 1.0% per side; break-even cost reported |
| Regime dependence | Per-year table mandatory (2019 / COVID-2020 / 2021 / 2022 / 2023 / 2024 / 2025 / 2026-H1) |
| Correlation | One trade at a time, non-overlapping (45 -> 21 DTE ~ 24 days, monthly cadence) -> observations independent; t-stat is honest |
| Capacity / liquidity | Require real traded `contracts > 0` on both legs at entry (binding repo rule from research/89); report entry-day ATM volume + OI |

---

## 4. The base — exact mechanics

- **Instrument:** NIFTY **monthly** expiry options. Identified as *the last expiry of the
  calendar month that was already listed on its own entry day (expiry − 45d)*. The naive
  "last expiry of the month" is wrong from 2025 on — after NSE moved the weekly expiry day,
  some months have a weekly expiring **after** the monthly (Apr-2025 monthly = 24th, weekly
  on the 30th). The 45-day listing test separates them, since weeklies are listed ~5 weeks
  out and monthlies ~3 months. Verified: 98 monthlies, all last-Thursday through Aug-2025
  then last-Tuesday from Sep-2025, with correct holiday shifts.
- **Entry:** expiry − **45 calendar days**; if not a trading day, roll **back** to the prior
  session (sensitivity: roll forward). Sell 1x ATM CE + 1x ATM PE.
  ATM = strike nearest the NIFTY spot close that day. Entry price = bhav **close** (~15:30;
  the video uses 15:15 — noted as an approximation; sensitivity run on `settle_price`).
- **Size and capital:** **10 lots**. NIFTY lot = **65** (confirmed against Kite's LIVE instrument
  master 2026-08-20 and the archived dumps — an earlier pass wrongly used 75, inflating every
  rupee figure by 15%). 10 lots = **650 qty**, so **1 point = Rs 650**.
  **Capital basis: Rs 3 lakh margin per lot x 10 = Rs 30L, Rs 36L blocked with a 20% buffer**
  (Arun's broker figure). P&L is reported in points (lot-agnostic) AND as CAGR / MaxDD / Calmar
  on the Rs 36L base, benchmarked against NIFTY 50 over the identical window.
  NOTE: `option_chain.lot_size` says 75 — that is a recorder bug, worth a separate look.
- **Exits (first to trigger):**
  1. **Target** — combined premium <= **50%** of entry credit
  2. **Stop** — combined premium >= **200%** of entry credit
  3. **Time** — expiry − **21 calendar days** (roll back to prior session if a holiday)
- **Exit check frequency (the bake-off axis):** daily close, 60-min, 30-min, 15-min, 5-min.
- **Costs:** slippage % of premium per side (0.25 / 0.50 / 1.00) + STT 0.1% of sell premium +
  exchange txn 0.05% of both sides + Rs 20/order brokerage. Gross and net both reported.
- **VIX filter (Phase C):** INDIAVIX close at entry, percentile rank vs previous 252 sessions;
  thresholds none / >25 / >50 / >75.

### Grid

| Axis | Values | n |
|---|---|---|
| Monitoring timeframe | daily, 60m, 30m, 15m, 5m | 5 |
| VIX rank threshold | none, >25, >50, >75 | 4 |
| Slippage | 0.25%, 0.50%, 1.00% | 3 |
| IV-path variant (intraday only) | prev-close IV (causal), same-day IV (bracket) | 2 |

Phase A = 1 cell (daily, no filter, 0.25%) — the replication.
Phase B = 5 TF x 2 IV-variants. Phase C = 4 VIX thresholds x 5 TF. Cost sweep applied post-hoc
to stored trade legs (no re-run needed). ~50 evaluated cells, all cheap (~90 trades each).

---

## 5. Status

**Phase:** writing engine. Nothing launched.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-08-20 20:11 | VPS reachable, data audit started | |
| 2026-08-20 20:25 | Data audit complete | bhav 2011 -> 2026-07-21 OK; **no intraday option history before 2026-04-20** |
| 2026-08-20 20:35 | STATUS sections 1-4 locked | Plan above |
| 2026-08-20 20:50 | Phase A v1 ran | 88 trades, net 80.4 pts/tr — but expiry calendar was wrong (see below) |
| 2026-08-20 21:05 | Phase B/C v1 ran | monitoring frequency ~ irrelevant; VIX filter reproduces their trade counts |
| 2026-08-20 21:20 | **BUG FOUND — monthly-expiry identification** | "last expiry of the calendar month" picks a **weekly** after NSE shifted the weekly expiry day (Apr-2025 monthly = 24th, weekly on the 30th; Mar-2026 similar). First fix ("longest-listed contract") over-corrected to 48 trades. Final rule: **the last expiry of the month that was already listed on its own entry day (expiry − 45d)** — weeklies are listed ~5 weeks out, monthlies ~3 months, so the 45-day listing test separates them |
| 2026-08-20 21:30 | Expiry calendar verified | 98 monthlies, every one a last-Thursday through Aug-2025 then last-Tuesday from Sep-2025 (matches NSE's expiry-day change), with correct holiday shifts (2023-01-25 Wed, 2023-03-29 Wed) |
| 2026-08-20 21:35 | Phase A v2 (corrected) | **89 trades, win 70.8%, net +6,952 pts, +78.1/trade, t = 3.03, MaxDD −998** |
| 2026-08-20 22:10 | Arun: "use our intraday data" | Re-audited every option table on the VPS; **proved** (not assumed) that Kite refuses expired contracts; found `option_chain` = 28.3M real 1-min quotes but only from ~27 DTE |
| 2026-08-20 22:30 | **BUG FOUND — lot size** | Used 75; Kite live master says **65**. Every rupee figure was 15% too high. Points unaffected. Fixed in `engine45.py` and all docs |
| 2026-08-20 22:45 | Arun: capital basis = Rs 3L/lot margin, Rs 36L blocked | Re-cut every result as CAGR / MaxDD / Calmar on Rs 36L + a year-by-year returns sheet + NIFTY benchmark |
| 2026-08-20 23:00 | **Phase D — real 1-minute evidence** | 240 day-contracts. DTE>=21: travel 6.3% above / 4.3% below close, **0 of 60 sessions >=50%**. 3 real 45-DTE trades overlap: premium stayed **0.55x-1.08x** of credit, neither trigger approached. Bhav close faithful to within 17.8 pts on ~900 |
| 2026-08-23 | **Phase E — delta management** | Arun: "keep the straddle until x% underlying move, then exit, redeploy at new ATM". Swept 7 thresholds x 3 arms x 3 re-entry caps x 2 trigger conventions. **EVERY variant loses to holding.** Cycles cut on a move: **−28.6 pts, 38% win**; cycles left to 21 DTE: **+83.0 pts, 81% win**. Best managed cell (1.5% exit-only) keeps 36% of the return. Friction is only ~12 of the 67-pt shortfall; the rest is the mechanism. Intraday triggering (real 5-min spot) makes it worse. Up-move cuts cost ~3x down-move cuts (vega offset thrown away). **Hold @ 5 lots dominates the best managed arm @ 10 lots on return, DD and Calmar simultaneously** |
| 2026-08-20 23:10 | **VERDICT REVISED** | On the margin basis the G4 failure does not hold: **CAGR 11.47% vs NIFTY 11.60%, MaxDD −13.8% vs −38.4%, Calmar 0.83 vs 0.30** -> STRATEGY-CANDIDATE, open item = stress margin |

---

## 6. Crash recovery (how Arun resumes without Claude)

Everything runs on the VPS at `/home/arun/quantifyd/research/119_45dte_short_straddle/`.

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd/research/119_45dte_short_straddle
tail -50 results/run.log                 # where it got to
wc -l results/trades_*.csv               # per-trade output already written
ps aux | grep run_phase                  # is anything still alive?

# re-run from scratch (cheap — minutes, not hours):
python3 scripts/run_phase_a.py           # replication, real prices, daily close
python3 scripts/run_phase_b.py           # monitoring-timeframe bake-off
python3 scripts/run_phase_c.py           # VIX percentile filter
```

Read-only against `backtest_data/market_data.db` — **nothing writes to any DB**. Safe to
kill and restart at any point; each script rewrites its own CSV from scratch.

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `NIFTY_45DTE_SHORT_STRADDLE_MULTITF_BACKTEST_STATUS.md` | this file | yes |
| `scripts/engine45.py` | expiry calendar, ATM picker, BS pricer, trade simulator | yes |
| `scripts/run_phase_a.py` | replication vs the published table | yes |
| `scripts/run_phase_b.py` | monitoring-timeframe bake-off | yes |
| `scripts/run_phase_c.py` | VIX percentile filter | yes |
| `results/trades_*.csv` | per-trade ledger (small) | yes |
| `results/RESULTS.md` | final verdict | yes |

---

## 8. Findings

**Full write-up: `results/RESULTS.md`. Verdict: STRATEGY-CANDIDATE (G3 passed, G4 conditional).**

1. **The published table replicates.** 89 trades vs his 83; win 70.8 vs 69.9%; avg win/loss
   +200.2/−217.8 vs +196.1/−216.8; exit mix 1/3/85 vs 1/4/78. Net **+78.0 pts/trade, t = 3.12**.
2. **On real broker margin it is competitive with the index and far safer.** 10 lots on Rs 36L:
   **CAGR 11.47% vs NIFTY 11.60%, MaxDD −13.8% vs −38.4%, Calmar 0.83 vs 0.30.** Six positive
   years, one flat, one −3.2%; 2020 was POSITIVE (+14.6%).
3. **Monitoring frequency settled on REAL 1-minute quotes.** In the DTE>=21 band the ATM straddle
   travels a mean 6.3% above / 4.3% below its close and **0 of 60 real sessions travelled >=50%**;
   across the 3 real 45-DTE trades the recorder overlaps, the premium stayed within
   **0.55x–1.08x** of credit and neither the 0.50 target nor the 2.00 stop was ever approached.
   Daily->hourly changes ONE trade (worst trade −29%, MaxDD −23%); 60m=30m=15m=5m to the decimal.
   The DTE 0–2 row (travel up to 7,669% of close) is the gamma the 21-DTE exit exists to dodge.
4. **VIX filter works through the wrong channel.** Counts match his exactly (21 vs 21 at >75) but
   the 85.7% win rate does NOT reproduce (71.4%). Premium collected rises 786->1,053; hit rate
   does not. Best cell on capital is **>25** (Calmar 1.05); **>75 is worst of both** (6.95% CAGR).
5. **Two of my own errors, corrected:** notional sizing (wrong basis for a margin-financed short)
   and lot size 75 vs the true **65**. The first version's "~7.8%/yr, below an index fund"
   conclusion was wrong on both counts.
6. **The open item is stress margin, not the edge.** Rs 3L/lot is today's margin at VIX 10.83;
   VIX peaked **83.61 on 2020-03-24** and SPAN scales with vol, so Rs 36L would likely have been
   breached exactly when the book was losing. Until a margin-call-aware re-run exists,
   **11.47% is an upper bound**. Gap risk and short-vol correlation with THE STACK / NAS / the
   straddle paper books remain live concerns.

7. **Delta management makes it worse — every variant (Phase E).** Holding the straddle to an x%
   underlying move then cutting (and optionally re-centring at the new ATM) was tested across
   7 thresholds, 3 arms, 3 re-entry caps and both close/intraday triggers. Not one cell beats
   the 78.1 pts/campaign baseline. The mechanism is explicit: **a cycle cut by the move rule
   realises −28.6 pts at a 38% win rate; a cycle left to run to 21 DTE earns +83.0 pts at 81%.**
   Cost explains only ~12 of the 67-pt shortfall — the rest is forfeited decay. Re-deployment
   adds nothing (2nd cycle −10.2, 3rd −6.2). Cutting on UP moves costs ~3x cutting on down moves
   (rallies come with falling IV, so the position repairs itself if left alone).
   **If the goal is a smaller drawdown, size down, do not manage: hold @ 5 lots (6.73% CAGR /
   9.0% DD / Calmar 0.75) strictly dominates 1.5% exit-only @ 10 lots (5.16% / 9.6% / 0.54).**

**Recommendation:** believe the table; hourly checks; **do not delta-manage — to cut risk, cut
lots**; VIX>25 for best risk-adjusted or no filter for best CAGR (never >75); **run the
stress-margin test before sizing live**; paper-first with a margin-call rule. This study does
not recommend arming it live today.
