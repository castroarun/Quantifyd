# 52W — Buy the 52-week-high close, sell the 52-week-low close, on Nifty 100

**STATUS: DONE — 13-Sep-2026.** Verdict: **NO EDGE as written; SIGNAL when optimised;
not a STRATEGY. Nothing adopted, no live book touched.** Full write-up in
`results/RESULTS.md`; published at `/app/backtest/52wk-channel-n100-research172`.

Book name for every report line in this study: **52W**.

---

## 1. The Ask

**What Arun asked (verbatim):**

> "buy a stock when a day closes above its 52-week high and exit when a day closes its
> 52-week low. apply this to simple nifty 50 and nifty next 50 stocks, if i were to hold
> a portfolio of them... Test it, optimize it comprehensively and report back."

**What we are actually testing.**

A long-only, always-eligible Donchian *channel* system on the Nifty 100 (Nifty 50 +
Nifty Next 50): enter on the first daily CLOSE above the highest close of the prior 252
trading days, exit on the first daily CLOSE below the lowest close of the prior 252
trading days, held as an equal-weight multi-slot portfolio. Then sweep the entry
lookback, the exit rule, the slot count, the universe, an index regime gate and an entry
buffer, and decide whether any plateau of that surface is a STRATEGY, a SIGNAL, or NO
EDGE — after costs, after tax, against a random-entry null and against the buy-and-hold
drift of the same universe.

**Economic hypothesis (G0).** Breakout/channel momentum: a new 52-week high is a
liquidity event where the marginal seller (anchored holders, index re-balancers) has been
exhausted and late-comer flow pushes price further; the counterparty is the anchoring
investor selling into strength. Decay risk is high — this is the oldest published trend
rule there is (Donchian 1960s, Turtle 1983), so any surviving edge should be modest and
should show up as a *plateau*, never a spike.

**Prior art in this repo that constrains expectations.** r/135 (Turtle optimisation)
concluded the classic 20/10 channel is the survivable core and that every Turtle add-on
subtracts; r/83 concluded channel shorts are dead on Indian equities; r/161 found
SuperTrend(14,4) beats OA's 15-SMA/−8% exit by +11.9pp on identical entries; r/159 found
the same slow trail wins; r/71 and r/75 found a NIFTY-above-its-long-MA gate is decisive
for breakout books; r/145 rejected broadening a momentum book's universe because it
re-imports beta the existing book already harvests. All of these are cited in the grid.

### Intake decisions — THIS SESSION IS NON-INTERACTIVE, all taken as defaults

| Item | Decision taken (assumption) |
|---|---|
| Published claim to reproduce | None. No replication gate applies. |
| Entry trigger | Daily **close** strictly above the channel high of the **prior** 252 bars (window EXCLUDES today). |
| Fill mechanic | **Next day's OPEN.** Honest and placeable. The same-close fill is reported once as an explicit look-ahead upper bound and labelled as such. |
| Exit trigger | Daily **close** below the channel low of the prior 252 bars; filled at the **next open**. Symmetric with entry — neither leg gets a look-ahead advantage. |
| Universe | Nifty 100 = current official NIFTY 50 + NIFTY NEXT 50 lists (`backtest_data/nifty50_official.csv`, `niftynext50_official.csv`, fetched Aug-2026). **These are CURRENT constituents → survivorship bias is present and is a named sin.** Controls: (a) a point-in-time liquidity-rank top-100 proxy built with the research/169 method, (b) Nifty 500, (c) random-entry null, (d) equal-weight buy-and-hold drift control on the identical name set. |
| Slots / sizing / capital | **20 slots × 5% of NAV, ₹1,00,00,000 (₹1 crore)**. One open position per symbol, no pyramiding in Spec A. |
| Slot contention | More signals than free slots → rank candidates by **252-day relative strength (close/close252 − 1) descending**; tie-break alphabetical by symbol (deterministic, recorded). A 30-seed random-draw contention arm is run as a robustness cross-check. |
| Instrument | NSE cash, CNC, long only. |
| Standalone or complement | Judged **both**: standalone against NIFTYBEES, and as a 4th sleeve against the live TN + OA(Base Age) + IPO book. |
| Costs | **15 bps per side** = 0.10% charges + 0.05% slippage. Sensitivity ladder **0 / 15 / 30 / 45 bps** per side. |
| Tax | 20% STCG / 12.5% LTCG above 365 days, Indian FY loss-netting settled 1 April, **₹1.25 lakh annual LTCG exemption**. |
| Idle cash | **5.2% post-tax p.a.**, credited daily on the cash balance (project standard, memory: idle-cash-standard-5p2-arbitrage). |
| Window | **2006-01-02 → 2026-09-11** primary. Sub-windows W1 2006-2015 / W2 2016-2026 (matches r/168's wa/wb so blends align). Data before 2005 covers only 2-47 symbols and is unusable. |
| Deployment intent | **Research only.** Nothing is deployed, nothing is paper-traded, no live engine is touched. |

---

## 2. The Base — data reality first

`backtest_data/market_data.db`, table `market_data_unified`, `timeframe='day'`, on the
**VPS** (canonical host).

| Fact | Value |
|---|---|
| Symbols with daily bars | 2,902 in 2026; 538 in 2005; **2 in 2000-2002, 43-47 in 2003-2004** |
| Nifty 100 names present in DB | **100 / 100** (no gaps) |
| Nifty 100 data starts | 1 in 2000, 29 in 2003, 3 in 2004, 26 in 2005, 3 in 2006 … 10 in 2025 |
| Last bar | 2026-09-11 for every name |
| Benchmark series | NIFTYBEES 2005-01-03 → 2026-09-11 (5,378 bars) — **primary**; NIFTY50 index only from 2011-01-03; NIFTYNEXT50 index only from 2015-01-01. **NIFTY 50 TR does not exist in this DB** — NIFTYBEES price is used as the closest tradeable proxy and is labelled as such. |

**Known defects handled explicitly (each is a toggle so its effect is measurable):**

1. **Splits are not retroactively adjusted.** A 52-week-high screen is precisely the
   screen this corrupts. Handled with the research/169 back-adjustment: a single-day
   close ratio ≤ 0.60 or ≥ 1.80 is treated as a corporate action and the whole prior
   series is multiplied back. The count of detected events, and how many landed on names
   with ≥ ₹5 crore of traded value, is reported. An `adjust=False` arm quantifies what
   the defect would have been worth.
2. **Phantom holiday rows** (O=H=L=C, volume 0, on non-trading days) NaN-poison rolling
   windows. Detected by the sparse-day + >90%-zero-volume signature and dropped.
3. **NaN-robust indicators.** Every rolling window is computed on the symbol's own
   `dropna()`'d series and scattered back onto the master calendar — never on a union
   frame.
4. **Funds in the universe.** `backtest_data/etf_exclusions.json` (name-based, not a
   ticker regex) plus the r/142 ETF regex. Irrelevant for the Nifty 100 arm but binding
   for the Nifty 500 and PIT-proxy control arms.
5. **Liquidity floor** 20-day median traded value ≥ ₹5 crore, causal (shifted one day).

**Spec A, locked bar-by-bar:**

```
eligible[t,s] = data present, >= L+1 bars, TVmed20[t-1,s] >= Rs 5 cr, not a fund,
                s in universe[t]
piv[t,s]      = max(close[t-252 .. t-1])                 (excludes today)
entry signal  = eligible & close[t] > piv[t,s]           -> BUY at open[t+1]
lo[t,s]       = min(close[t-252 .. t-1])
exit  signal  = close[t] < lo[t,s]                       -> SELL at open[t+1]
```

No stop, no target, no time stop, no gate in Spec A. **The literal exit is enormously
wide** — a stock must fall all the way from a 52-week high to a 52-week low before it is
sold. Per-position maximum adverse excursion is measured and reported plainly.

**Success criterion, pre-registered before any cell runs:**

- Primary ranking metric: **after-tax Calmar** (after-tax CAGR ÷ |max drawdown|) on the
  full window at 15 bps per side.
- A cell is only readable as a candidate if it ALSO clears, simultaneously:
  (a) after-tax CAGR above NIFTYBEES over the same window,
  (b) positive net-of-cost per-trade expectancy,
  (c) beats the random-entry null on the same universe and the same number of trades,
  (d) beats the equal-weight buy-and-hold drift control of the same universe,
  (e) is a **plateau**, not a peak — its immediate parameter neighbours agree within
      ~15% of its Calmar,
  (f) both sub-windows W1 and W2 positive and within 4pp of CAGR of each other's sign.
- **Adoption bar as a 4th sleeve (pre-registered):** ≥ +0.10 blend Calmar or ≥ −2pp blend
  drawdown at no worse blend CAGR, after tax, versus the incumbent TN+OA+IPO book, with
  correlation < 0.40 to each existing leg, and beating a cash sleeve at the same weight.

**Falsification plan, decided now:** if the literal Spec A fails to beat NIFTYBEES
after tax, AND the optimised plateau's advantage over the random null is smaller than
its advantage over the drift control (i.e. the "edge" is just owning the same stocks),
the family is written up as NO EDGE / SIGNAL and nothing is proposed for adoption.

---

## 3. Plan — the grid

Engine: a vectorised numpy panel (`p172.py`, modelled on research/169 `xpanel.py`) plus a
day-loop portfolio simulator (`bt172.py`, modelled on research/161 `bt_core.py` with FY
tax netting, next-open fills and idle-cash carry). Prices loaded once; every cell reuses
the same panel.

**Phase 1 — literal Spec A (G1/G2).** 1 headline cell + controls:
same-close-fill upper bound, random-entry null, drift control (equal-weight monthly-
rebalanced buy-and-hold of the same eligible universe), cash-only null, and the
split-adjust-off / phantom-keep arms.

**Phase 2 — the main surface (entry × exit), base config = Nifty 100, 20 slots, no gate,
0% buffer, 15 bps.**

| Axis | Values | n |
|---|---|---|
| Entry lookback | 63 / 126 / 189 / 252 / 378 / 504 | 6 |
| Entry reference | prior-close-max, prior-high-max | 2 |
| Exit rule | channel-close-min and channel-low-min at 21/42/63/126/189/252 (12), ST(7,3), ST(10,3), **ST(14,4)** [r/161, r/159], ATR(14) trail ×2.5/×4/×6, SMA15 close, SMA15+(−8% hard stop) [OA's old exit], EMA50 close | 20 |

→ **6 × 2 × 20 = 240 cells.**

**Phase 3 — one-axis-at-a-time from the Phase-2 plateau centre** (top 3 entry×exit
combinations carried forward):

| Axis | Values | cells |
|---|---|---|
| Slots (equal split 100%/slots) | 10 / 15 / 20 / 30 / unlimited | 3 × 5 = 15 |
| Universe | Nifty 50 / Next 50 / Nifty 100 / Nifty 500 / **PIT top-100 liquidity proxy** | 3 × 5 = 15 |
| Index gate | none / NIFTYBEES > 200-SMA blocks new entries [r/71, r/75] / NIFTYBEES > 100-SMA | 3 × 3 = 9 |
| Entry buffer | 0% / +1% / +3% / within-5%-of-all-time-high / at-a-new-all-time-high | 3 × 5 = 15 |
| Cost ladder | 0 / 15 / 30 / 45 bps per side | 3 × 4 = 12 |
| Contention rule | RS-rank (default) / 30 random seeds | 3 × 31 = 93 |

**Phase 4 — robustness.** 12 start-date offsets (monthly phase) on the shortlist;
per-year table; W1/W2 split; parameter-surface monotonicity; top-10-trade deletion;
winner caps at +50%/+100%; per-position max adverse excursion; losing-streak and
win-rate tradeability columns.

**Phase 5 — portfolio fit.** Daily and monthly correlation vs TN, OA(Base Age) and IPO
using the r/168 NAV archives (`tn_navs_cash052.npz`, `ba_navs_cash052.npz`,
`ipo_navs_cash052.npz`), then a 4-sleeve weight sweep against the incumbent 3-sleeve
blend, plus the cash-null at the same weight.

**Total ≈ 240 + 159 + ~200 robustness ≈ 600 cells.** Multiple-testing discount is stated
explicitly in RESULTS.md.

---

## 4. Status (live log)

**Phase: DONE.** All six phases complete. ~600 cells + 150 null draws + 24 ensemble paths.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-13 ~14:20 | Research number 172 confirmed free on laptop and VPS | last used = 171 |
| 2026-09-13 ~14:30 | STATUS doc written; sections 1-4 locked before any compute | metric + adoption bar pre-registered |
| 2026-09-13 ~19:35 | Panel built: 5,890 x 1,490, 2003-01-01..2026-09-11, 103 s | 202 split events back-adjusted (33 on >= Rs 5 cr names); 2 phantom dates dropped (2014-04-24, 2014-10-15) |
| 2026-09-13 ~19:40 | **BUG CAUGHT before any result was reported**: positions marked at zero on a missing bar, printing a fake -99% drawdown | fixed with a forward-filled MARK price (`Panel.CM`), signals keep the raw NaN close. Same defect research/161 `bt_core` had to fix |
| 2026-09-13 ~19:50 | Phase specA + 30 random-null draws + drift/cash/benchmark controls | Spec A 14.25% / -46.7% / 0.305 |
| 2026-09-13 ~19:55 | Phase grid: 252 entry x exit cells in 1.1 min | ST(14,4) wins the exit axis; the literal 52-week-low exit ranks 18th of 21 |
| 2026-09-13 ~20:00 | Phase axes: 220 one-axis cells in 0.9 min | slots / universe / gate / buffer / cost / 30 contention seeds |
| 2026-09-13 ~20:05 | Part B: 60 matched nulls at the optimum AND at the literal spec, risk-matched cash null, 24 start-date offsets, per-year curves | **Spec A loses to its own null** (14.25 vs median 16.47) |
| 2026-09-13 ~20:10 | Part C: four adversarial nulls x 30 draws, shortlist x 4 cost levels, blend vs TN/OA/IPO | **the momentum-matched null N3/N4 beats the optimum on CAGR outside its whole range** |
| 2026-09-13 ~20:12 | Report artifacts: factsheet PNG, curves PNG, YoY house table | published |
| 2026-09-13 ~20:20 | RESULTS.md written, study published, INDEX/TODO/ops updated | DONE |

### Findings as they landed

1. The literal spec is beaten by its own random-entry null on CAGR **and** Calmar.
2. The exit, not the entry, is the entire optimisation. SuperTrend(14,4) again (3rd time).
3. The entry lookback is a flat plateau from 126 to 504 days. 252 is not special.
4. The NIFTYBEES 200-SMA gate does NOT help this family (contradicts r/71 / r/75) because
   the trend exit has already removed everything the gate would remove.
5. RS-rank slot contention is inside the noise of an arbitrary random draw.
6. **The momentum-matched null is the kill.** Random names from the top half by 252-day
   relative strength beat the system by 1.8-2.5pp of CAGR on every one of 30 draws.
7. **Survivorship premium on the official Nifty-100 CSV: ~4.2pp of CAGR, 0.24 of Calmar.**
8. Correlation to Open Alpha 0.69 daily. It is very nearly the same book.
9. Blend: best cell +0.033 Calmar at -0.67pp CAGR on 22/30 paths against a +0.10 bar, and a
   plain cash sleeve wins 30/30 at every weight.

---

## 5. Crash Recovery — resume without Claude

Everything runs on the **VPS** at `/home/arun/quantifyd`, python = `venv/bin/python`.

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd
tail -50 /tmp/r172_<phase>.log            # what the runner last printed
ls -la research/172_52wk_channel_n100/results/
wc -l research/172_52wk_channel_n100/results/*.csv
pgrep -af "run172.py"                      # is it still alive?
```

Every runner writes **one CSV row per completed cell and skips cells already present**,
so re-launching the identical command resumes. Phases:

```bash
cd /home/arun/quantifyd
setsid nohup venv/bin/python -u research/172_52wk_channel_n100/scripts/run172.py specA  > /tmp/r172_specA.log  2>&1 </dev/null &
setsid nohup venv/bin/python -u research/172_52wk_channel_n100/scripts/run172.py grid   > /tmp/r172_grid.log   2>&1 </dev/null &
setsid nohup venv/bin/python -u research/172_52wk_channel_n100/scripts/run172.py axes   > /tmp/r172_axes.log   2>&1 </dev/null &
setsid nohup venv/bin/python -u research/172_52wk_channel_n100/scripts/run172.py robust > /tmp/r172_robust.log 2>&1 </dev/null &
setsid nohup venv/bin/python -u research/172_52wk_channel_n100/scripts/run172.py blend  > /tmp/r172_blend.log  2>&1 </dev/null &
setsid nohup venv/bin/python -u research/172_52wk_channel_n100/scripts/run172.py report > /tmp/r172_report.log 2>&1 </dev/null &
```

Run phases **in order** — `axes` needs `grid`'s CSV, `robust` needs `axes`, `report`
needs all of them. Only one phase at a time (the panel costs ~1.5 GB of the VPS's 7 GB).

**Safe to inspect / delete and regenerate:** anything under
`research/172_52wk_channel_n100/results/`.
**Do NOT touch:** `backtest_data/market_data.db` (read-only here, but it is the live
data store), anything under `services/`, and any live engine. This study writes nothing
outside its own results folder, `frontend/public/`, `frontend/src/data/backtests.ts`,
`research/INDEX.md`, `TODO.md` and the ops registry.

---

## 6. Files

| File | Purpose | Committable? |
|---|---|---|
| `52WK_CHANNEL_N100_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/p172.py` | panel: load, split-adjust, phantom-drop, NaN-robust channels, PIT liquidity rank | yes |
| `scripts/bt172.py` | portfolio simulator: next-open fills, RS contention, FY tax netting, idle cash | yes |
| `scripts/run172.py` | all phases (specA / grid / axes / robust / blend / report) | yes |
| `scripts/report172.py` | tearsheet PNG, YoY house table, backtests.ts payload | yes |
| `results/specA.json` | literal-spec headline + controls | yes |
| `results/grid.csv` | 240-cell entry×exit surface, one row per cell | yes |
| `results/axes.csv` | slots / universe / gate / buffer / cost / contention | yes |
| `results/robust.csv` | offsets, sub-windows, per-year, outlier deletion | yes |
| `results/blend.csv`, `results/correlations.json` | 4-sleeve blend vs TN/OA/IPO | yes |
| `results/navs.npz` | shortlist NAV paths | yes (small) |
| `results/RESULTS.md` | final verdict | yes |
| `frontend/public/52wk-channel-n100-research172.png` | tearsheet | yes |

---

## 7. Findings

**Full write-up: `results/RESULTS.md`.** Headline, after tax, 15 bps/side, Rs 1 cr, 20 slots:

| Book | CAGR | MaxDD | Calmar |
|---|---|---|---|
| 52W Spec A (the literal rule) | 14.25% | -46.73% | 0.305 |
| 52W OPT (same entry, SuperTrend(14,4) exit) | 14.95% | -26.01% | 0.575 |
| 52W OPT on a survivorship-free PIT top-100 | 10.77% | -32.35% | 0.333 |
| Random-entry null, 52-week-low exit (30 draws) | median 16.47% | -46.41% | 0.352 |
| **Momentum-matched null (top half by 252d RS), 30 draws** | **median 16.79%** | | **0.584** |
| Equal-weight B&H of the same Nifty 100 (pre-tax) | 19.22% | -58.13% | 0.331 |
| NIFTYBEES | 11.37% | -59.71% | 0.190 |

Verdict: **NO EDGE as written; SIGNAL when optimised; not a STRATEGY.** Nothing adopted.

---

## 8. The seven deadly sins — how each is controlled

| Sin | Control |
|---|---|
| Look-ahead | Every channel window excludes today; every rolling stat is `.shift(1)`; entries and exits both fill at the NEXT open. The same-close fill is run once and labelled a look-ahead upper bound. |
| Survivorship | Named and unfixable for the official Nifty 100 lists (current membership, no PIT source in this repo). Controlled by: a PIT liquidity-rank top-100 proxy, a Nifty 500 arm, and a buy-and-hold drift control on the identical name set so the survivorship premium is subtracted rather than claimed. |
| Overfitting / multiple testing | ~600 cells disclosed; ranking metric and adoption bar pre-registered above; plateau required, peak rejected; W1/W2 split; parameter monotonicity reported. |
| Cost neglect | 15 bps/side base, 0/15/30/45 ladder, after-tax with FY netting, idle cash at 5.2%. |
| Regime dependence | Per-year table, W1/W2 split, 2008 and 2020 crash windows and 2018 / 2022H1 grind windows reported separately, each measured from the **full curve's running peak** (r/154 convention fix). |
| Correlation / single factor | Daily and monthly correlation to TN, OA and IPO; blend value measured against the incumbent 3-sleeve book and against a cash sleeve at the same weight. |
| Capacity / shortability | Long-only cash equity on the Nifty 100 — the most liquid names on the exchange. Position size at ₹1 crore / 20 slots = ₹5 lakh is reported against the held names' median traded value. |
