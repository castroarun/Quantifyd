# IPO Spec A Rules Transplanted to Nifty-50 / 100 / 200 / 500 / Midcap / Smallcap / All-Stock Universes — Is the Edge the Young Stock or the Mechanics?

**STATUS: DONE** (opened 13-Sep-2026 08:10 IST, closed 09:40 IST) — verdict **NO EDGE** for every transplant; two corrections to IPO Base itself. Read `results/RESULTS.md`.

Research number 169 (checked free on the VPS 13-Sep-2026 07:55 IST; 160-168 taken).

---

## 1. The Ask

**What you asked (Arun, verbatim):** "Why is the IPO system IPO specific, what if we apply this
to other universes like nifty 50, 100, 200, midcap, 500, small cap, all stocks?"

Context: Arun has decided to move the Capital Desk targets from 40/40/20 to TN 37.5 / OA·BaseAge
37.5 / IPO 25, but asked for this answer first, because the verdict may change what the third
sleeve should be.

**What we are actually testing.** For a stock listed less than six months ago, "the highest close of
the last 25 bars" is close to its all-time high since listing, and the base is its FIRST base. For a
seasoned stock the identical rule is just a 25-day Donchian close breakout — a different signal. So
the transplant answers two questions, and this study separates them:

1. **Where does IPO Spec A's edge live** — in the young-stock condition, or in the breakout +
   SMA-50-trail mechanics? Test: run Spec A with the age band removed (and, separately, with ONLY
   seasoned names > 6 months) on each size universe, each against its own date-matched random-entry
   null. Then sweep the age band on the all-stock universe (<=6m, <=12m, <=24m, any vetted, none,
   >6m, >24m).
2. **Is a transplanted version just an existing book in disguise?** Prior families cited, not
   rediscovered: r/71 (breakout exit bake-off), r/82 (10-15d breakout = breakout-paper book),
   r/142 (Blue-Sky ATH breakout), r/152 (multi-year breakout = OA in disguise), r/161 (OA·Base Age,
   live). Test: correlation + paired blend against TN, OA·BaseAge and IPO-A (r/168's book).

## 2. The Base — the rules (Spec A, research/167 stage 9; live in services/ipo_paper.py, NOT edited)

```
universe   NSE equities; funds excluded by LONG NAME (etf_exclusions.json) + r/142 ticker regex
age        listed <= 6 months (vetted listing date, r/153 listing_dates.csv) AND >= min_bars
           <- THE IPO-SPECIFIC PART; this is the variable
liquidity  20-day median traded value at t-1 >= Rs 5 cr
base       last 25 bars; pivot = highest CLOSE, shifted 1; depth pivot -> base low <= 30%;
           close[t-1] < pivot
RS         off
gate       no NEW entries while NIFTYBEES < its 150-day SMA (prior close)
trigger    close[t] > pivot
fill       NEXT day buy-stop AT the pivot, filled max(pivot, open), ONLY if high[t+1] >= pivot
exits      stop close <= fill x 0.90 -> target close >= fill x 1.25 -> trail close < SMA-50
book       8 slots @ 18.75%, Rs 10L, 25 bps/side, after tax (20% STCG / 12.5% LTCG, FY netting)
idle cash  5.2% post-tax (binding standard, r/163); reproduction gate at 5.0%
```

**Size universes — CAUSAL PROXY, labelled as such everywhere.** No point-in-time index
constituent history exists in the repo (only TODAY's official CSVs, and research/41's own
traded-value reconstruction). Using today's constituents back to 2006 is look-ahead survivorship.
So: on the first trading day of each month, every non-fund symbol with >= 60 bars is ranked by its
trailing 126-bar median traded value known at t-1 (the research/41 method), held for the month:

| universe label | proxy definition |
|---|---|
| Nifty-50-like `top50` | TV rank 1-50 |
| Nifty-100-like `top100` | rank 1-100 |
| Nifty-200-like `top200` | rank 1-200 |
| Nifty-500-like `top500` | rank 1-500 |
| Midcap-like `mid101_250` | rank 101-250 |
| Smallcap-like `small251_500` | rank 251-500 |
| beyond-500 `rank501plus` | rank 501+ (still >= Rs 5 cr) |
| All stocks `all` | every name clearing the Rs 5 cr floor |

Proxy validation (second, shorter check, NOT the headline): overlap of the TV ranks with
`fundamentals.db features_pit_monthly.mcap_pit` ranks from Aug-2018, and with today's official lists.

**Age arms.** `le6/le12/le24` = vetted listing date AND age <= 6/12/24 months (clean). `vet_any` =
vetted listings at any age. `none` = no age condition. `gt6/gt24` = definitely seasoned: vetted age
> m, or no vetted date and data since before Jun-2005, or first DB bar more than m months back (a
lower bound on true age, so a name is only ever called seasoned when it provably is).

**Data defenses (from the agent's defect list, measured 13-Sep-2026).**
- Phantom holiday rows: 24-Apr-2014 (121 symbols, 100% zero volume) and 15-Oct-2014 (129, 100%)
  are dropped from the date index. The r/167 engine keeps them.
- NaN-robust rolling: every pivot / SMA / TV median computed on each symbol's own rows.
- Split adjustment: 191 one-day closes <= -35% and 158 >= +60% since 2005 in non-fund names. The
  panel back-adjusts at every one-day close ratio <= 0.60 or >= 1.80 (NSE circuit filters make real
  moves of that size in >= Rs 5 cr names vanishingly rare). Events on liquid names are counted.
- The r/167 panel (no adjustment, union-index rolling, phantom rows kept) is reproduced first,
  then each defense is toggled on in turn so its effect on Spec A is disclosed.

## 3. Pre-registration (written BEFORE the first cell)

**Ranking metric:** W2 (2006-01-01 -> 2026-09-04) median paired real-minus-null CAGR, after tax.

**"The edge exists on universe X"** — ALL of:
- W2 median paired (real - null) CAGR >= +1.0pp, AND real beats null on >= 25 of 30 paired seeds;
- WA (2006-2015) real beats null on >= 25/30 seeds; WB (2016-2026) real beats null on >= 25/30.

The null: same days, same number of entries per day, names drawn at random from THAT universe's
(and that age arm's) eligible set whose day high reached their own shifted 25-bar pivot, same fill
convention, same gate, same exits, 30 paired seeds (r/167 stage 7/8 construction, rng 20260912).

**"The age band is the edge"** if `all/le6` passes AND `all/gt6` fails AND `gt6` fails on a majority
of the size universes. **"The mechanics are the edge"** if `gt6` passes on a majority of the size
universes.

**"Worth a sleeve"** (r/168's bar) — ALL of:
- passes the edge test above;
- EITHER replacing IPO-A at 25% in TN 37.5 / OA·BaseAge 37.5 / IPO-A 25 (monthly), OR added as a
  fourth sleeve at 10/15/20/25% (others pro-rata), improves the blend by +0.10 Calmar, or +2pp CAGR
  at no worse drawdown, on >= 20 of 30 paired paths versus that three-sleeve book;
- beats the RISK-MATCHED cash null at that weight (cash weight solved on a 1% grid to the same
  median blend drawdown) on >= 25 of 30 paths;
- monthly correlation to OA·BaseAge < 0.60 (otherwise it is OA in disguise, whatever the blend says).

**Re-fit rule:** only for universes that pass the edge test; trail {20,30,50,75,100} x base length
{25,50}; no other dial. At most 4 survivors carried (top by W2 edge), disclosed.

**Capacity yardstick** (reported, not a gate): book size at which the p90 position reaches 5% of the
name's 20-day median traded value; IPO-A on the same yardstick.

## 4. Plan — cells

| stage | what | cells |
|---|---|---|
| S0 | reproduce r/167 Spec A with r/167's own engine (5.0% and 5.2%), per-seed vs r/168 `A_25bps_y52`, and the Spec A null (+4.78pp, 30/30) | 3 checks — STOP if off |
| S1 | new panel equivalence on the IPO universe: r/167-like panel, then + NaN-robust, + phantom drop, + split adjust, + min_bars 60 | 5 real cells |
| S2 | transplant: 8 universes x {age none, seasoned > 6m} + Spec A on the new panel, real + null, 3 windows | 17 x 2 |
| S3 | age axis on `all`: le12, le24, gt24, vet_any (le6 / none / gt6 from S2) | 4 x 2 |
| S4 | re-fit on survivors: trail x L | <= 4 x 9 x 2 |
| S5 | cost ladder 40/60 bps on carried specs; capacity; tradeability | <= 8 |
| S6 | portfolio fit: weekly+monthly correlation, replace-IPO and fourth-sleeve blends, risk-matched cash | per carried spec |
| S7 | proxy validation vs mcap_pit / official lists | 1 |

Each cell = 30 seeds x 3 windows (W2, WA, WB). Every drawdown from the running peak of the full
curve. If nothing survives the null, S4 is skipped and S6 runs on the two best transplants as
INFORMATION ONLY, labelled.

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-13 07:55 | research/169 claimed | 160-168 taken on the VPS |
| 2026-09-13 08:15 | STATUS written, pre-registration locked | before any cell ran |
| 2026-09-13 08:12 | S0 reproduction PASSED | r/167 engine: 21.80 / -26.63 (delta 0.000pp); 30 seed paths vs r/168 `A_25bps_y52` max abs diff 0.0; null 16.98, edge +4.78pp, 30/30 — all exact |
| 2026-09-13 08:14 | S1 equivalence done (3.1 min) | new panel built like r/167's reproduces 21.80 / -26.63 exactly (1,545 signals). + NaN-robust windows 22.24 / -24.84 (1,707 signals: r/167's panel lost ~10% of signals to poisoned windows); + phantom drop 22.23; + split adjust 22.23 (no IPO-universe trade touched; 273 events full-universe, 38 on >= Rs5cr names, all real corporate actions e.g. ADANIENT 2015, ARVIND 2018, INFIBEAM 2018) |
| 2026-09-13 08:14 | **SUSPICIOUS: min_bars 60 -> 12.24% / -36.42%** vs 22.23% at min_bars 25 | the validated Spec A (r/167 INCUMBENT) is min_bars 25; services/ipo_paper.py runs MIN_BARS = 60 on a comment that misreads the harness (its `n >= 60` counts rows over the WHOLE DB today, not at the signal date). Being confirmed with r/167's own engine + null (S1b) |
| 2026-09-13 08:28 | **S1b CONFIRMED with r/167's own engine** (`results/s1b_minbars_r167engine.json`) | Spec A at min_bars 25: 21.80% / -26.63 / 0.819, null +4.78 (30/30). min_bars 40: 17.87 / -32.40, null edge -1.38 (3/30). **min_bars 60 (the live book): 11.57% / -38.97% / Calmar 0.297**, WA 9.25 / WB 13.39. The live IPO book is not running the spec that earned 21.8% |
| 2026-09-13 08:33 | grid stage killed silently after panel load (no traceback) | out-of-memory: it was built while the S1b r/167 context was also in memory. Relaunched alone. Rule for the rest of the study: ONE panel process at a time |
| 2026-09-13 08:53 | S2 transplant + S3 age axis + S7 proxy DONE (grid, 11.7 min) | **0 of 22 cells pass the pre-registered edge test.** Every seasoned universe (top50..top500, mid, small, 501+, all; age none or > 6m / > 24m) is a 2.3-9.0% CAGR book at -34 to -54% DD with a W2 edge between -3.24 and +1.53pp. Age axis on `all`: le6 22.39 (edge +2.25, WB 14/30 - fails on WB), le12 16.28 (edge -1.93), le24 14.68 (-1.29), vet_any 7.55 (+1.53, 22/30), none 6.97 (-0.04), gt6 5.86 (-1.50), gt24 4.63 (+0.73). Live-book mb60 12.39 (+0.79, fails). Proxy vs mcap_pit (Aug-2018+): top500 ~79-80% overlap, top200 ~75-79%, top50 ~60%, mid/small bands ~43-47%; Spearman ~0.70 |
| 2026-09-13 08:55 | queued, sequential (one panel at a time): S1c null attribution, S4 refit (skips: no survivor), S5 costs, S2b mcap_pit universes 2018+, S6 blend (information only) | S2b added because the mid/small TV-rank bands overlap mcap bands only ~45% |
| 2026-09-13 09:03 | **S1c null attribution DONE** (`results/s1c_null_attribution.csv`) | Spec A real vs null, 3 windows, 5.0% cash. r/167-like panel: real 21.80, null 16.87, edge **+5.14 (W2 30/30, WA 30/30, WB 30/30)** - r/167 only ever ran W2. NaN-robust panel: real 22.24 (+0.44), null **20.03 (+3.16)**, edge **+2.25 (W2 30/30, WA +4.62 30/30, WB -0.18 13/30)**; phantom drop and split adjust change nothing. On the correct panel Spec A's selection edge exists in 2006-2015 and VANISHES in 2016-2026 (real 28.26 vs random young liquid names 28.16). The real-arm return is robust; the "beats random" claim is half a panel artifact. Mechanism being measured (S1d) |
| 2026-09-13 09:03 | S4 refit SKIPPED as pre-registered | 0 survivors |
| 2026-09-13 08:20 | DESIGN NOTE (before any transplant cell) | transplant / age cells use min_bars = 25 (the validated spec). The live-book variant (60) is carried as its own labelled cell `all__le6__mb60_livebook` with its own null. Irrelevant to seasoned arms (bars >> 60) |

## 6. Crash recovery

```
ssh arun@94.136.185.54
cd /home/arun/quantifyd
ps aux | grep run169 | grep -v grep          # is a stage still running?
tail -30 /tmp/r169_<stage>.log               # stage progress
ls -la research/169_ipo_rules_universe_transplant/results/
# every cell-table stage is resume-safe (skips labels already in its CSV):
setsid nohup venv/bin/python -u research/169_ipo_rules_universe_transplant/scripts/run169.py <stage> \
    > /tmp/r169_<stage>.log 2>&1 < /dev/null &
# stages in order: repro equiv transplant age refit costs blend proxy
```
Do not touch `services/ipo_paper.py` or any r/167 / r/168 file. `results/navs/*.npz` are heavy and
gitignored; CSV/JSON/MD are safe to read at any time.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/xpanel.py` | full-universe panel (proxy ranks, adjust, NaN-robust) + signal / null / cell runner | yes |
| `scripts/run169.py` | stages S0-S7 | yes |
| `results/s0_repro.json` | reproduction gate | yes |
| `results/s1_equiv.csv` | panel equivalence and defect toggles | yes |
| `results/s2_transplant.csv`, `s3_age.csv`, `s4_refit.csv`, `s5_costs.csv` | cell tables | yes |
| `results/s6_blend.json`, `s6_*.csv` | portfolio fit | yes |
| `results/s7_proxy.json` | proxy validation | yes |
| `results/navs/*.npz` | 30-seed W2 NAV paths per cell | NO (gitignored) |
| `results/RESULTS.md` | final verdict | yes |

## 8. Findings

**IPO — final (13-Sep-2026).** Full evidence in `results/RESULTS.md`.

1. **Transplants: NO EDGE.** 0 of 16 size-universe cells (8 universes x {no age limit, seasoned > 6m})
   pass the pre-registered control test. 2.3-9.0% after tax at -34% to -54% DD; best W2 edge +1.52pp
   (Midcap-like, 20/30, WB 15/30). The PIT market-cap re-run (2018-09 -> 2026-09) agrees. Every
   transplant lowers TN 37.5 / OA 37.5 / IPO-A 25 on 30/30 paths (replacing IPO-A: -3.5 to -3.8pp CAGR)
   and loses to risk-matched cash on 30/30.
2. **Why IPO-specific:** the return lives in a stock's first months after listing. Age <= 6m 22.39%,
   <= 12m 16.28%, <= 24m 14.68%, vetted any age 7.55%, none 6.97%, seasoned > 6m 5.86%, > 24m 4.63%.
   At <= 12 and <= 24 months the random control beats the breakout. Within <= 6m, the entries 25-60
   sessions after listing (before an SMA-50 exists: bracket trades) carry about half the return.
3. **research/167's null edge was half a panel artifact:** +5.14 -> +2.25pp on per-symbol windows;
   WA +4.62 (30/30), WB -0.18 (13/30). Real-arm curves unaffected (r/168 blend stands).
4. **research/167's capacity line quoted the median (1.56%) as the p90 (9.05%)** of 20-day traded
   value at Rs 10L.
5. **The live book runs MIN_BARS 60; Spec A was validated at 25.** research/167's engine: 21.80% at 25,
   11.57% at 60. Blend: the 60-bar book costs -2.46pp CAGR on 30/30 paths vs the validated spec, and
   beats risk-matched cash by only +0.61pp. Capital Desk: 25% to IPO is supported only for the
   validated spec. Nothing live changed; review 2026-09-19 registered.

Pre-registered labels: "mechanics are the edge" — 0/8 universes, NO. "Age band is the edge" — not met
in its exact form, because Spec A itself fails WB on the clean panel; the honest wording is "the
young-listing cohort is where the return is; the breakout selection added edge only in 2006-2015".
S4 refit skipped (no survivor), as pre-registered.
