# Third-Sleeve Archetypes — Broadened Search Beyond Equity Mean-Reversion (all viable system classes)

STATUS: DONE — **VERDICT: STRATEGY (candidate) — GOLD sleeve (GOLDBEES B&H) at 10% clears the pre-registered blend bar; all other archetypes rejected.** See `results/RESULTS.md`.

## 1. The Ask (verbatim)

> "apart from kc6, did u explore all possible systems?? not just stick to the studies we hv
> done" — Arun, following r/146.

**What we're actually testing:** every viable third-sleeve ARCHETYPE implementable in our
EOD infra (cash equities + ETFs + index futures; options explicitly out of scope), triaged
G0-cheap, G1 for survivors, blend-tested against the TN+OA pair. The full inventory —
including what was considered and DISCARDED, with reasons — is in section 4 (Arun explicitly
wants the breadth documented).

## 2. The Base — baseline, target profile, data reality

- **Baseline:** TN+OA 50-50 after-tax (r/144-146). Full-window: 27.2% / −16.4% / Calmar 1.65.
- **Target profile (r/146 structural finding, now an explicit G0 screen):** the pair has NO
  crash tail left to hedge (blend DD −2.4% inside the 2008 window, −1.5% in the 2020 crash);
  its real drawdowns are GRINDING phases (2018 −11.0%, 2022H1 −8.8%). A candidate must
  plausibly EARN IN GRINDS (2018 / 2022H1 window P&L reported explicitly) while staying
  ~flat in crashes (2pp tolerance).
- **Data reality (VPS market_data.db, checked 2026-09-04):** NIFTYBEES 2005+; GOLDBEES,
  MON100 (Nasdaq-100 ETF), LIQUIDBEES, all 9 sector indices, JUNIORBEES: **2015-01+**;
  LTGILTBEES (10y+ gilt ETF) 2016-07+; SILVERBEES only 2022+ (too short). **Therefore the
  candidate/blend common window is 2015+ (gilt 2016-07+); the 2008 crash is NOT testable for
  candidates** — stated on every table. Blends are evaluated on the candidate's common
  window with the baseline RECOMPUTED on that same window (fair same-window bar).

## 3. Pre-registered metric (BINDING — r/146 bar re-anchored to common window)

Adoptable only if ALL hold, on the candidate's common window:

1. Best w3 ∈ {10,15,20,25,33}% (TN=OA split the rest, monthly, after-tax legs) beats the
   SAME-WINDOW TN+OA baseline by **+0.10 Calmar (blend CAGR give-up ≤ 2pp)** OR **−2pp blend
   DD at ≥ equal CAGR**.
2. Candidate daily corr < 0.4 vs BOTH legs.
3. Robust across the 10 OA seeds (median; worst seed not catastrophic) and TN offsets {0,4,8}.
4. **Grind windows (2018, 2022H1): blend DD must IMPROVE vs baseline; crash window (2020,
   2015-16 partial): not worsened by >2pp.**
5. Beats the plain-cash null at the same weight (de-levering is not alpha).

**Tax treatment (pre-stated):** TN/OA legs after-tax as before. Candidate sleeves are
net-of-cost (10bps/side on turnover; futures legs 5bps/side + carry modeling) but
GROSS-of-tax, EXCEPT that buy-and-hold ETF sleeves genuinely defer tax (fair), while churny
variants (trend-filtered, rotation, futures L/S) get a favorable bias from this shortcut —
**so a churny candidate that FAILS is safely dead, and any churny candidate that PASSES must
be re-run with the r/144 FY-netted tax model before adoption.** Killing is bias-safe;
adopting is not — the asymmetry is deliberate.

## 4. The archetype inventory (G0 triage — FULL breadth, discards documented)

### Screened (G1 built and run)

| # | Archetype | Construction(s) | Prior confronted |
|---|---|---|---|
| A1 | **Gold sleeve** | GOLDBEES B&H; 10-month-SMA trend-filtered (cash when below) | r/63: gold leg of GTAA validated |
| A2 | **Nasdaq sleeve** | MON100 B&H; 10m-SMA filtered | r/63/64: Nasdaq leg validated clean |
| A3 | **Gilt duration sleeve** | LTGILTBEES B&H (2016-07+) | new — the only true non-equity carry in the DB |
| A4 | **Gold+Nasdaq 50-50** | monthly rebal (deliberately EXCLUDES the Nifty leg — the pair is already 100% Indian equity) | r/63 lesson: diversification > selection |
| A5 | **GTAA equal-weight** | NIFTYBEES/GOLDBEES/MON100 monthly (r/63 winner verbatim) + all-legs-10m-SMA-filtered variant | r/63 Calmar 1.73; r/64 corrupt-series list respected (only verified-clean series used) |
| A6 | **Gold+Nasdaq+Gilt equal-weight** | monthly, the max-diversification non-equity basket | — |
| B1 | **Index trend LONG/SHORT via futures** | NIFTY momentum-sign (majority of 3/6/12m >0 → long; <0 → short), monthly; short leg = −index + cash yield on collateral − 5bps/side switches; NIFTYBEES series (2005+) | r/83 killed EQUITY shorts (all horizons) — this is the INDEX-level different question; skeptical, kill on evidence |
| B2 | **Index SHORT-ONLY overlay** | short when momentum-sign negative, else cash — pure grind/crisis alpha claim | same |
| C1 | **Sector rotation** | top-2 of 9 Nifty sector indices by 6m momentum, monthly, 2015+ — SCREEN-LEVEL (sector indices are not directly tradeable; sector ETFs are thin/short-history), promoted only if profile screams | r/64 factor rotation; corr-to-TN is the expected killer |

### Considered and DISCARDED at G0 (the breadth Arun asked about)

| Archetype | One-line reason |
|---|---|
| Market-neutral stock pairs / factor long-short (D) | Stock shorting is not implementable in our EOD infra (no SLB depth at size; r/46 precedent: reversal short book unexecutable in cash); futures-only universe (86 names) + margin/rollover ops put the cost floor above every spread edge we have measured; screen-level DISCARD. |
| 52-week-low turnaround / long-horizon value (E) | Profile cannot be crash-flat by construction (loads into falling knives through crash slides — the exact r/146 failure mode); r/87-88 killed structure/GCO screens once drift + date-matched controls were applied; r/84 dip-buy prior. DISCARD without spending G1. |
| Corporate-bond / credit carry | No corporate-bond ETF history in the DB; gilt ETF (A3) is the implementable duration proxy; LIQUIDBEES is already the cash leg. |
| Currency (USDINR) / MCX commodities | No data in market_data.db and outside the EOD cash infra. DISCARD (data/infra). |
| VIX-based sleeve | India has no tradeable VIX instrument in cash/ETF form; options out of scope. DISCARD. |
| Quality / Low-vol / value factor-index ETFs | r/64: Kite Quality/LowVol/Commodities INDEX series are CORRUPT; the ETFs' own histories are short. DISCARD (data integrity). |
| International beyond Nasdaq (MAFANG etc.) | 2021+ history only. DISCARD (window). |
| Equity intraday sleeves | r/109-110 closed the line: no OHLCV intraday edge clears the ~10bps cost floor. |
| RSI-regime / HMA-weekly / medium-swing / GCO | Prior kills: r/72 (converges on momentum, no new alpha), r/93 (SIGNAL not investable), r/82, r/88. |
| Equity mean reversion (KC6 et al.) | r/146 — just killed: re-imports the crash tail. |
| Options-based sleeves (short-vol, collars, hedges) | OUT OF SCOPE by the brief — separate book with its own portfolio lab (r/111/134 territory). |

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-04 00:2x | Data triage done (window reality: candidates 2015+, gilt 2016-07+, 2008 untestable); STATUS written with pre-registered bar + full inventory BEFORE any run | — |
| 2026-09-04 00:3x | `scripts/arch_sleeves.py` building (candidate NAVs → G1 table → blends on common window, reusing r/146 cached OA-seed and TN-offset NAVs) | — |

## 6. Crash recovery

- VPS `/home/arun/quantifyd/research/147_third_sleeve_archetypes/`.
- `ps aux | grep arch_sleeves`; log `/tmp/tn147.log`.
- Incremental CSVs: `results/g1_archetypes.csv`, `results/blend_arch.csv`,
  `results/windows_arch.csv`; candidate NAVs `results/nav_*.csv`.
- Blend legs are CACHED inputs: `research/146_complementary_third_sleeve/results/oa_navs.csv`
  (10 OA seeds, after-tax) and `.../tn_nav_off{0,4,8}.csv` — do not delete; regenerate via
  r/146 `blend3.py` if lost.
- Resume: `cd /home/arun/quantifyd && setsid nohup venv/bin/python -u
  research/147_third_sleeve_archetypes/scripts/arch_sleeves.py > /tmp/tn147.log 2>&1 &`
- Nothing deployed is touched.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| this STATUS md | live status | yes |
| `scripts/arch_sleeves.py` | candidate NAV builders + G1 + blend + windows | yes |
| `results/*.csv` | results (small) | yes |
| `results/RESULTS.md` | final verdict | yes |

## 8. Findings

Full write-up in `results/RESULTS.md`. Headlines:

1. **GOLD (GOLDBEES B&H) at w3=10%: baseline 29.6/−14.9/Calmar 2.02 → 28.4/−12.0/2.37**
   (+0.35 Calmar at −1.3pp CAGR, beats cash-null 2.13, corr −0.04/+0.08, improves ALL four
   stress windows, robust at TN offsets 4/8: +0.24/+0.28). 15% also passes (2.60); ≥20%
   breaches the pre-registered CAGR give-up. Concrete spec + review clause in RESULTS.
2. **Gold is the ONLY leg this book wants from the r/63 GTAA basket**: the Nifty leg
   duplicates pair beta (worsens the 2020 crash window), the Nasdaq leg fails the 2022
   grind (−25.2% in-window). Diversify with what the book doesn't already own.
3. **Index trend long/short via futures KILLED** — the monthly momentum-sign goes short
   after crashes and eats India's V-recoveries (2020 window −35.6%); short-only overlay has
   no positive expectancy. r/83's equity-short kill extends to index level, different
   mechanism. **Sector rotation KILLED** (corr 0.37-0.39 AND crash-converges, 2020 −31.6%).
4. Big caveat stated loud: 2015-2026 is a gold-favorable window and 2008 is untestable
   (ETF data starts 2015); the diversification case is structural, the return contribution
   is regime — hence minimum passing weight + review clause.
5. **Bug disclosure:** v1 mix() built calendar-month-end NAVs whose intersection with daily
   legs zeroed ~half the months (broken GN5050/GNG3/GTAA blend rows); found via impossible
   same-window baselines, fixed to a daily-grid construction, ALL numbers regenerated before
   any conclusion was drawn.
6. LTGILTBEES carries ±16-19% single-day thin-ETF dislocation prints (Mar-2020) — gilt/GNG3
   numbers noisy; flagged.

Closing log: 2026-09-04 00:4x G1+blends run → mix bug found+fixed → 00:5x full rerun (40s)
→ 01:0x RESULTS.md written, verdict STRATEGY (candidate, gold 10%); committed + pushed +
published to app.
