# NIFTY Long-Dated Short Premium — 60/90/180/365-DTE straddles, strangles and condors vs the live 45-DTE book

**STATUS: DONE — verdict NO EDGE beyond 45 DTE, CONCLUDED. See `results/RESULTS.md`.**
Research number 174. Owner: quant-researcher. Opened 2026-09-15.

> **Prime directive for this study.** The 45-DTE NIFTY short straddle (research/119) is LIVE with
> real money since 2026-09-11 (3 lots, 195 qty, 27-Oct expiry, 23450 strike, credit 744.82 pts,
> exit due 06-Oct). **This study does not touch it.** Everything here is a candidate to run
> ALONGSIDE or INSTEAD, and nothing changes in the live book without its own STATUS doc,
> its own evidence and an after-15:40 deploy.

---

## 1. The Ask

**What Arun asked (near-verbatim, 2026-09-15):**

> "now that we are live with 45 DTE, can we test for more like 2 months away
> straddles/strangles/condors with different DTE entries and DTE exits and different stop
> losses, same for other liquid ones 3 months, 6 months away, 1 year away etc?"

and, on the stop-loss axis:

> "sls can be combined premium, single side, underlying price move, vix, relative vix or
> combinations or more that i cudnt think about"

**What we are actually testing.** Four separate questions, in this order, each gating the next:

1. **Feasibility.** At each entry tenor (~45 / 60 / 90 / 180 / 365 calendar DTE), does a NIFTY
   ATM (or near-ATM) option pair actually **trade** — real contracts, real open interest — on
   the day we would sell it and on every day we would need to mark or exit it? A contract that
   exists in the bhavcopy with a settlement price and zero volume is **not tradeable** and must
   not be filled in a backtest (binding rule from research/89).
2. **Tenor.** Among the tenors that pass (1), does any beat the live 45-DTE book on
   **net points per unit of margin per unit of time**, after realistic cost?
3. **Structure.** Within a surviving tenor, does a strangle (swept OTM offset) or a
   condor / winged strangle (swept wing distance) beat the naked straddle on the same
   margin-adjusted basis? research/127's body at 2.5% out with 7% wings is carried as a prior.
4. **Stops.** Does ANY stop family beat NO-STOP? NO-STOP is the incumbent and the baseline in
   every cell, because it has won seven times already (r/119 E and G, r/127 B3, r/128, r/129,
   r/130, r/135).

**The falsification plan, pre-registered.** If the tenors longer than ~100 DTE have no tradeable
ATM volume, we kill them on the data and say so — that is a complete result, not a failure.
If they are tradeable but return less per rupee of margin-year than the 45-DTE book, we kill them
on economics. If no stop family beats no-stop at t >= 2 on >= 60 trades, the stop question is
answered NO for the eighth time and we stop asking it.

---

## 2. The Base — what is being tested

### 2.1 Ground truth and its limits (state these before any number)

| Item | Reality |
|---|---|
| Price source | `nse_options_bhav` in `backtest_data/market_data.db` — real NSE daily bhavcopy (open/high/low/close/settle/contracts/open_interest per strike per expiry per session). NIFTY rows 2011-01-03 to 2026-09-11, 5.19M rows, 525 expiries. |
| Resolution | **DAILY CLOSE ONLY.** Expired-contract intraday option data is unobtainable from Kite ("invalid token"); the 1-minute `option_chain` recorder starts 2026-04-20 and only covers a contract from ~27 DTE. A long-dated study is necessarily a daily-close study. Any intraday stop is therefore **modelled**, not observed, and is labelled so. |
| Liquidity rule | **BINDING.** A strike is fillable only if BOTH legs have `contracts > 0` and `close > 0` on that session. Long-dated strikes carry stale settlement prints; filling them manufactures edge. Additional volume/OI floors are an explicit axis, not an assumption. |
| Lot | NIFTY lot = **65** (not 75 — `option_chain.lot_size` is wrong). 1 point = Rs 65 per lot. Live book runs 3 lots = 195 qty = Rs 195 per point. |
| Expiry derivation | The monthly weekday moved Thursday to Tuesday; "last expiry of the month" picks a WEEKLY post-Sep-2025. Monthlies are derived from data (the r/119 `monthly_expiries` rule: last expiry of the month already listed on its own entry day). **Legacy far-dated Thursday contracts (2026-12-31, 2027-06-24, ...) survive alongside real monthlies and are later in the month** — they are exactly what this study is about, so they are selected explicitly by target DTE, never by a "last expiry" rule. |
| Split adjustment | Irrelevant for index options. Becomes relevant only if this extends to stocks. |
| VIX | `INDIAVIX` daily close, 2015-01-01 to 2026-09-11. Percentile rank is causal (vs the previous 252 sessions), per r/119. |

### 2.2 Phase-0 finding already in hand (2026-09-15) — the download has a long-dated hole

A first pass over `nse_options_bhav` shows NIFTY expiries beyond ~75 DTE are **missing for most
of the history, and it is a downloader artifact, not the market**:

| Trade months | Max DTE present | Expiries per day | Long-dated (>100 DTE) |
|---|---|---|---|
| 2015-01 to 2015-12 | 1,820 | 15 | yes (11-12) |
| **2016-01 to 2024-02** | **73-75** | 3 then 13 | **none** |
| 2024-03 to 2026-03 | 1,826 | 22 | yes (11-12) |
| **2026-04 to 2026-06** | **75** | 10 | **none** |
| 2026-07 to 2026-08 | 1,798 | 22 | yes (11) |
| **2026-09** | **75** | 9 | **none** |

The cause is `MAX_DTE = 75` in `research/89_short_monthly_straddle/scripts/download_nse_bhav_stocks.py`
(and `ATM_BAND = 0.25` on strikes). Months populated by that script are truncated at 75 DTE;
months populated by the earlier production downloader are complete. So **the long-dated history
is backfillable** and Phase 0 must attempt the backfill before any tenor is killed on "no data".

### 2.3 Structures

| Code | Structure | Definition |
|---|---|---|
| `STR` | short straddle | sell 1 ATM CE + 1 ATM PE (the live 45-DTE shape) |
| `STG_x` | short strangle | sell CE near spot x(1+x), PE near spot x(1-x); x swept |
| `IC_x_w` | iron condor | `STG_x` plus long CE near spot x(1+x+w) and long PE near spot x(1-x-w); w swept |
| `WS_w` | winged straddle | `STR` plus long wings at +/-w (research/127 shape with an ATM body) |

Every leg of every structure must independently satisfy the liquidity rule on the entry day and
on the exit day, or the cell is not counted.

### 2.4 Entry, exit, costs

- **Entry:** decided on the daily close of the session on/before `expiry - DTE_entry` calendar days
  (`prev_session` roll, as r/119). Filled at that session's close, plus slippage.
- **Time exit:** daily close of the session on/before `expiry - DTE_exit`. `DTE_exit` is swept
  per tenor — a 365-DTE entry has no business exiting at 21 DTE by default.
- **Target:** buy back at `p x credit` (p swept, 50% is the live rule), checked on the daily close.
- **Costs:** the r/119 `costs_points` model — slippage on both sides + STT 0.10% of sell premium
  + exchange txn 0.05% both sides + Rs 20 per order over 2 x legs orders + GST 18% on brokerage+txn.
  **Slippage is the axis that matters here.** 25 bps of premium is the 45-DTE assumption; a
  365-DTE strike that trades 58 contracts a day is far wider. Slippage is swept
  **0.25% / 0.75% / 1.5% / 3.0% of premium** and the break-even slippage is reported per cell.
- **Margin:** SPAN+exposure is the binding constraint for a year-long short. Reported as
  **net points per Rs lakh of margin per year**, using measured Kite `basket_order_margins` for
  the live-today shape (method: `research/119_45dte_short_straddle/scripts/margin_stress_live.py`;
  note the wings-first leg-ordering trap, and that margin must be compared as margin + MTM loss).

### 2.5 Stop families (NO-STOP is the baseline in every cell)

| Family | Rule | Prior art |
|---|---|---|
| `NONE` | hold to target or time exit | **the incumbent; has won 7 times** |
| `PREM_c` | exit both legs when combined premium >= c x credit | **already REFUTED at 45 DTE** (r/119 G: -130.9 pts, t -2.34). Re-tested at new tenors as a control only |
| `MOVE_m` | exit when abs(spot/entry_spot - 1) >= m | **already REFUTED at 45 DTE** (r/119 E: 0 of 63 cells beat holding). Control only |
| `SIDE_c` | **NEW** — buy back only the threatened leg at c x its own entry price; the other leg runs to the time exit | untested |
| `SIDE_DELTA` | **NEW** — buy back the leg whose strike the spot has crossed by >= m; other leg runs | untested |
| `VIXL_v` | **NEW** — exit when India VIX closes above absolute level v | untested |
| `VIXR_r` | **NEW** — exit when VIX percentile rank (causal, 252d) >= r | untested |
| `VIXD_d` | **NEW** — exit when VIX >= entry-day VIX x d (relative to the VIX we sold into) | untested |
| combinations | the best single survivor x the best second survivor, only if both clear alone | — |

**Economic reason to expect the NEW families might differ from the killed ones.** The refuted
stops all cut the WHOLE position on a mark-to-market trigger, which realises the loss and
forfeits the theta that pays for it (r/119 mechanism: a cut cycle books -28.6 pts at 38% win
vs +83.0 at 81% for one left to run). A single-side stop keeps the surviving leg's theta and
only removes the gamma that is actually hurting; a VIX stop triggers on the *price of risk*
rather than on our own mark, so it can fire before the loss is realised in premium terms.
Those are genuinely different mechanisms. They may still fail; the point is that they have not
been asked.

### 2.6 Universe

NIFTY first. BANKNIFTY second (3.47M bhav rows, same period) **only if NIFTY produces a
survivor**. Liquid F&O stocks last, and noting that **research/127 already owns the 45-to-21 DTE
stock strangle** (body 2.5% out, 7% wings, no stop, TP 50%, +0.264% of spot per trade, t 5.06).

### 2.7 Success metric — pre-registered BEFORE the sweep

Primary ranking metric: **net points per Rs lakh of SPAN+exposure margin per 365 days held**,
after costs at the 0.75%-slippage setting, over the longest clean window each tenor supports.

Adoption bar for a new tenor/structure to be proposed alongside the live 45-DTE book:

- net-positive at **t >= 2.5** on **>= 40 independent (non-overlapping-expiry) trades**, AND
- **>= 1.25x** the live 45-DTE book's margin-time return on the SAME window, AND
- survives slippage at 1.5% of premium, AND
- maximum peak-to-trough drawdown in points <= 1.5x the 45-DTE book's on the same window, AND
- correlation of trade-level P&L to the 45-DTE book < 0.6 if proposed as an addition.

Adoption bar for a stop family to be proposed: beats `NONE` **paired on the same trades**,
median paired delta > 0, wins on >= 60% of trades, t >= 2.0 on the paired difference, and
survives the cost sweep. Anything less is reported as "no better than holding", again.

---

## 3. Plan — phases and cell counts

| Phase | Question | Output | Size |
|---|---|---|---|
| **P0a** | Does the long-dated bhav hole backfill? | rebuilt `nse_options_bhav` rows for DTE>75, 2016-01 to 2026-09 | ~2,100 session files |
| **P0b** | At each DTE bucket, does the ATM pair TRADE? | `results/liquidity_by_dte.csv` — per (entry day, expiry): ATM strike, CE/PE contracts, OI, and the same at the intended exit day | a tenor passes only if >= 60% of its candidate entry days have a fillable ATM pair at or above the volume floor |
| **P0c** | How wide is long-dated really? | measured spread proxies: intraday high-low of the option, close-vs-settle gap, and live Kite quotes for the currently-listed long-dated strikes | a slippage assumption defensible per tenor |
| **P1** | Tenor bake-off, straddle only, NO-STOP, target sweep | `results/p1_tenor.csv` | ~5 tenors x ~6 exit-DTE x 4 targets x 4 slippage = **480 cells** |
| **P2** | Structure bake-off on surviving tenors | `results/p2_structure.csv` | ~3 tenors x (1 STR + 4 STG + 6 IC + 3 WS) x 3 exits = **126 cells** |
| **P3** | Stop bake-off, paired against NO-STOP | `results/p3_stops.csv` | ~3 tenors x 8 families x ~4 params = **96 cells**, each paired |
| **P4** | Entry filter (VIX rank) as an axis, not an assumption | `results/p4_filter.csv` | ~4 filter settings x survivors |
| **P5** | Margin, capacity, correlation to the live book, report package | `results/RESULTS.md`, factsheet, `/app/backtest/<slug>` | — |

Cells are disclosed for the multiple-testing discount. Every CSV is written one row per completed
cell and is resume-safe (a re-run skips rows already present).

## 4. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-15 | Study opened, folder + STATUS created | sections 1-4 written before any run |
| 2026-09-15 | P0 probe 1 complete | NIFTY bhav has a long-dated hole 2016-01 to 2024-02 plus 2026-04/05/06 plus 2026-09; cause = `MAX_DTE=75` in the r/89 downloader; backfillable |
| 2026-09-15 14:20 | P0b liquidity probe v2 done | v1 conflated dead weeklies with real monthlies (NSE lists weeklies 5-9 weeks out that never print a contract). v2 asks per session which listed contract a trader could actually sell. 45/60 DTE fillable ~99% of sessions every year; 90 DTE 88-96% where data exists; 120-365 DTE 37-100% in 2015/2024-2026 only |
| 2026-09-15 14:25 | P0a backfill smoke-test passed | 4 sample sessions re-downloaded, 5,219 long-dated rows added. Confirms the hole is a download artifact |
| 2026-09-15 14:26 | P0a full backfill LAUNCHED | 2,102 truncated sessions, staged to `results/bhav_longdated_stage.db` so it never takes a write lock on the DB the live executors read during market hours. `--merge` after 15:40 IST |
| 2026-09-15 14:30 | MEASURED margin from Kite | NRML standalone per lot, spot 23,190: 42d Rs 2.21L, 69d Rs 2.29L, 105d Rs 2.55L, 196d Rs 3.20L, 287d Rs 5.69L, 469d Rs 6.28L, 1015d Rs 7.76L. Margin roughly TRIPLES to the 1-year tenor. Long-dated strikes sit on a **1,500-point grid**, so at spot 23,190 the nearest listed strike is 22,500 - 3.0% away. A 1-year ATM straddle is not placeable |
| 2026-09-15 14:35 | Engine validated vs r/119 | rebuild of the live 45-DTE arm: n=92, avg credit 782.4, **t = 3.12** against published n=89, 786.3, t = 3.12 |
| 2026-09-15 14:40 | P3 stop bake-off (45 DTE) | **Eighth kill.** Every family loses PAIRED against no-stop; damage is monotone in fire rate. Single-side is NOT better than whole-position at matched fire rates (-28.8 vs -28.9), refuting the "keep the surviving leg's theta" hypothesis. Stops do cut the worst trade (-1,049 to -464) - insurance with a premium |
| 2026-09-15 15:40 | Backfill DONE, merged after the market-hours cutoff | 2,102 sessions, 7,567,118 staged rows, 0 errors; table 33,217,523 -> 36,054,242 (+2,836,719). Coverage verified: all years 2015-2026 now hold >100 DTE contracts |
| 2026-09-15 16:30 | Full sweep on repaired data DONE | P1 624 cells / 36,506 trades, P1b decay windows, P2 168 cells, P3 ~340 paired cells, P5 finalists. Verdict written |
| 2026-09-15 14:50 | P2 structure bake-off (45 DTE) | Condors and winged straddles KILLED on the index (IC2.5w3 avg net -0.73). Consistent with r/128; not a contradiction of r/127, whose wings work on idiosyncratic STOCK tails. A **5% strangle** beats the ATM straddle on t (3.62 vs 2.96), worst trade (-615 vs -1,047) and return on margin (31.0 vs 24.5 %/yr) at the same tenor - flagged, not recommended, pending window split and paired test |

## 5. Crash recovery

Everything runs on the VPS (`94.136.185.54`, `/home/arun/quantifyd`, `venv/bin/python`).
Read-only against `backtest_data/market_data.db` except the P0a backfill, which does
`INSERT OR IGNORE` into `nse_options_bhav` only.

- **Check what finished:** `ls -la research/174_long_dated_short_premium/results/` and
  `wc -l` each CSV. Each CSV has a header plus one row per completed cell.
- **Check a background run is alive:** `ps -ef | grep 174_` and `tail -40 /tmp/r174_*.log`.
- **Resume anything:** re-run the same script; it skips completed cells by label.
- **Safe to inspect:** every file under `research/174_long_dated_short_premium/`.
- **Do NOT touch:** `services/straddle45_live.py` or any live book state.

## 6. Files

| File | Purpose | Committable |
|---|---|---|
| `NIFTY_LONGDATED_SHORT_PREMIUM_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/probe_longdated_liquidity.py` | P0b ATM-liquidity probe | yes |
| `scripts/backfill_longdated_bhav.py` | P0a bhavcopy backfill, no DTE cap | yes |
| `results/liquidity_by_dte.csv` | per-entry-day ATM liquidity by tenor | yes |
| `results/RESULTS.md` | the honest verdict | yes |

## 7. Findings

Full verdict in `results/RESULTS.md`. Headline:

**NO EDGE beyond 45 DTE — CONCLUDED.** Nothing deployed, nothing live touched.

1. **Tenor is a monotone decay from 45 outward, not a peak.** Fixed rule (enter at T, exit
   21 DTE), 2015-2026: 45d +56.6 pts/trade t 2.96 -> 60d +46.3 t 1.36 -> 75d +30.5 t 0.78 ->
   90d -5.5 -> 105d -32.2 -> 180d -70.4 -> 210d -164.3. Win rate falls 73.6 / 66.7 / 59.0 /
   48.0. Of 624 cells, exactly two clear t = 2 and both are the live book.
2. **1-year and longer are untradeable.** ATM call at 365 DTE trades 54 contracts a day; at
   730 DTE, four. Killed on data, not on P&L. Past ~105 DTE the strike grid is 1,000-1,500
   points wide, so the nearest strike is 1.0-1.2% from spot - a long-dated "ATM straddle"
   cannot be placed.
3. **The edge is holding one strike, not harvesting theta.** Chopping the 45->21 hold into
   five re-centred 5-day pieces turns +68.5 gross / +56.6 net into +45.2 gross / **-12.8 net**.
   Re-centring costs a third of the gross plus four extra round trips. Every individual DTE
   window across the whole contract life is indistinguishable from zero. This one fact explains
   r/119 phases E and G and all seven stop families here.
4. **Eighth refutation of stops.** Every family loses paired at every tenor; damage is monotone
   in fire rate. Single-side is no better than whole-position at matched fire rates, refuting
   the theta-preservation hypothesis. Stops do cut the worst trade (-1,049 -> -464) - insurance
   with a premium, not an edge.
5. **Condors and winged straddles are dead on the index** at all six tenors tested, worsening
   with tenor. Not a contradiction of r/127 (stock tails are idiosyncratic); consistent with r/128.
6. **Exit DTE is not resolvable** at n=140, SD ~225 pts: exits at 9 / 14 / 21 / 27 DTE have
   overlapping +-2SE intervals. 21 DTE is defensible and unrefuted, NOT demonstrated.
7. **The VIX-rank entry filter is validated and monotone** - 27.8 / 37.3 / 41.5 %/yr on margin
   at off / >25 / >50, both halves positive throughout.
8. **A 5% strangle at the same tenor is lower-return, lower-variance, lower-margin** - paired,
   it loses 26 pts/trade and wins on 39% of trades. An earlier interim called it a winner on the
   unpaired table; that was retracted. Open as a capital-allocation question only.
9. **Repo-wide data repair, permanent.** `nse_options_bhav` was missing every NIFTY expiry
   beyond ~75 DTE for 2016-01 -> 2024-02, 2026-04/05/06 and 2026-09 (cause: `MAX_DTE=75` in the
   r/89 downloader, which resumed by trade-date so the uncapped production downloader skipped
   those sessions forever). 2,102 sessions re-downloaded, 0 errors, **+2,836,719 rows merged**.
   All years 2015-2026 now complete.
