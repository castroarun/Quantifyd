# MPF UNIFIED REPORT — HANDOVER FROM THE ENTRY-AUDIT SESSION (11-Sep-2026)

**You are being asked to compile one unified Momentum Portfolio report page.** This file is
everything that session produced: what was found, what it means, where every artefact lives,
what is still running, and what is still owed. Read it once before touching anything — two
of the conclusions overturn figures that are still quoted elsewhere in the app.

**Author:** the session that audited how each book BUYS.
**Sibling session:** a second session ran in parallel on research/160 (Quality Summit) and
research/161 (Open Alpha v2). Its work is referenced throughout and **must not be treated as
this session's**. See §8 — you will be editing a page both sessions have written to.

---

## 1. THE ONE FINDING EVERYTHING ELSE HANGS ON

A backtest may not read a bar's CLOSE to decide a trade and then pay a price from EARLIER in
that same bar.

```python
trig = setup & (close > pivot)        # decided from the CLOSE
fill = max(pivot, open)               # priced from the OPEN, same bar
```

Both lines look defensible. Together they delete every trade that started well and failed by
the bell, because a trade only enters the record once the close confirmed it. It passes every
generic look-ahead check — no future BAR is read, and the fill price genuinely traded.

**Measured cost, same spec, same seeds, same window, changing only the entry:**

| Book | Published | Honest | 
|---|---|---|
| Open Alpha (research/142) | 40.8% CAGR | **−1.7%** |
| IPO Base (research/153) | 31.0% CAGR | **15.0%** |
| True North | — | unaffected, engine holds only closes |

**Proven on the source site's own published trades, not on our simulation.** Of its 54
published Open Alpha trades, 49 of the 50 checkable ones closed above the level they were
bought at — by construction. And in the 120 days before each published entry, an order
resting at that same level would already have been filled and lost **7.0 times on average**.
HCLTECH: the clean 10-Jan-2025 entry at 1972.20 follows twenty earlier days when that level
was touched and the close fell back.

> **A list of trades taken can never show the trades that are missing.** Say this when
> anyone reports having verified the trade list by hand. They were not wrong; they were
> looking at the only thing a trade list can show.

This is now the **eighth deadly sin** in `research/QUANT_RESEARCH_PLAYBOOK.md` §4, with a
full operational section at §5A, and mirrored into
`.claude/agents/quant-researcher.md` §5A and the global playbook at `~/.claude/`.

---

## 2. THE SECOND-ORDER FINDING — PARAMETERS ARE WRONG TOO

Open Alpha's 680-cell parameter sweep (research/142) was scored entirely against the
look-ahead entry. So were IPO Base's (research/153, `ipo_sweep.py` calls `build_trigger`
with `fill_close=False`).

**The surfaces do not merely shift, they INVERT.** Across 100 cells:

| Trail (SMA bars) | 10 | 15 | 20 | 30 | 50 |
|---|---|---|---|---|---|
| Buy at breakout close (placeable) | 4.35% | 11.04% | 14.64% | 15.81% | 20.00% |
| Stop above breakout candle (placeable) | −2.29% | 6.31% | 9.19% | 15.18% | 16.73% |
| Next-day stop at pivot (placeable) | −4.79% | 4.43% | 9.14% | 15.41% | 19.42% |
| **Same-bar open on close trigger (LOOK-AHEAD)** | **43.79%** | **40.24%** | **37.45%** | **32.66%** | **30.62%** |

Every placeable entry improves as the trail lengthens. The unplaceable one degrades. A
parameter fitted on the second surface is wrong for the first. The live book runs a 15-day
trail and returns about 9%; at 75 days three independent entry mechanics converge on 16–20%.

**IPO Base has the identical problem and it has NOT been re-optimised.** That was the next
piece of work and it is unstarted. See §7.

---

## 3. THE NUMBERS — every live/candidate book, AFTER TAX, full common period

Common window **2006-04-03 → 2026-09-03, 20.4 years**, set by the shortest series (True
North). After 20% STCG / 12.5% LTCG with Indian FY loss netting, 25 bps a side, 5% credited
on idle cash.

| System | Test period | CAGR | Max DD | Calmar | Growth of 100 | Avg invested | Avg in cash |
|---|---|---|---|---|---|---|---|
| **Open Alpha · Base Age** (r/161) | 2005-01 → 2026-09 | **20.3%** | −32.5% | 0.62 | 4334 | 67% | 33% |
| **True North** (incumbent) | 2006-04 → 2026-09 | 19.5% | **−23.7%** | **0.82** | 3787 | 43% | **57%** |
| **IPO Base · First Base** | 2006-01 → 2026-09 | 15.1% | −35.9% | 0.42 | 1767 | 33% | 67% |
| NIFTYBEES (index) | 2005-01 → 2026-09 | 10.6% | −59.7% | 0.18 | 780 | 100% | 0% |

**On the shorter 2016–2026 window the picture changes materially** — True North 23.0% /
−23.7% / 0.97, Base Age 23.0% / −32.4% / 0.71, IPO 16.0% / −35.9% / 0.45, index 12.1%.
**And on the sibling session's 2018–2026 window, Base Age shows 26.2% / −26.3% / 0.99.**
Same system, three windows, three pictures. **Window choice is the single biggest lever on
how this book looks — state it on every table.**

**Weekly correlations (2016–2026, 557 weeks):**

| | TN | OA v2 | OA v3 | IPO | NIFTY |
|---|---|---|---|---|---|
| True North | 1.00 | 0.40 | 0.39 | 0.20 | 0.26 |
| OA · Base Age | 0.40 | 1.00 | 0.55 | 0.26 | 0.49 |
| OA · ATH+VIX | 0.39 | 0.55 | 1.00 | 0.23 | 0.40 |
| IPO Base | 0.20 | 0.26 | 0.23 | 1.00 | 0.18 |

**IPO Base is the genuine diversifier** at 0.18–0.26 against everything including the index.
The two Open Alpha versions at 0.55 are the closest pair — run one, not both.

**CASH YIELD IS NOT A SMALL TERM and the books differ wildly.** At 5%: True North holds cash
57% of the time so ~2.8 points of its 19.5% is the sweep; IPO Base 67% so ~3.4 of 15.1%;
Base Age 33% so ~1.7 of 20.3%. NIFTYBEES is fully invested and gets none. **Also
inconsistent:** True North's curve comes from research/144's own file which assumes **6.5%**,
not 5% — worth ~0.9 points a year to it. Not enough to reorder the table, but not
like-for-like.

---

## 4. VERDICT PER BOOK

| Book | Verdict | Action |
|---|---|---|
| **True North** | **CLEAN.** Audited for the same defect and found sound — engine holds only closes, gate is NaN-robust, universe is point-in-time, every live dial matches its study. Published numbers stand. | none |
| **Open Alpha** | **Live entry is broken twice over.** (a) `services/oa_entry.py` selects `close < pivot` — names that have NOT broken out — the INVERSE of the designed rule. (b) Even corrected, the published number is unreachable. **Buying is PAUSED.** | adopt Base Age (r/161); correct or retire `oa_entry.py` |
| **IPO Base** | **Live engine CORRECT** (buys next morning); the STUDY is wrong. 31.0% → 15.0%. Still beats the index on return and Calmar. Study page corrected. | re-optimise on the honest entry — NOT STARTED |
| **Open Alpha v3** (this session's repair) | 19.2% (2016–26). Superseded by Base Age on every axis. **Keep only as the correction's evidence.** | do not present as a candidate |

---

## 5. TWO MORE DEFECTS FOUND ALONG THE WAY

**5.1 The universe contained 221 funds.** The ETF filter matched ticker spellings written
before the 2023–25 gold/silver wave, so EGOLD, GOLD1, GROWWGOLD, HDFCGOLD, TATAGOLD,
ESILVER, SILVERBETA, MON100 (Nasdaq 100), MAFANG, ICICIB22 and ~180 sector/index funds were
reaching an **equity** momentum book. Gold then ran through 2024–26 and produced a spurious
+32% arm in a fundamental-overlay test.
**Fixed:** `backtest_data/etf_exclusions.json`, built by
`research/158_oa_arming_width/scripts/build_etf_list.py` from the instrument's **long name**
in the Kite dump — every fund says ETF, no operating company does. 125 → 346 exclusions,
zero real companies caught (SKYGOLD, GOLDIAM, DECNGOLD, SHANTIGOLD, SILVERTUC all kept).
**research/142's published figures still carry this contamination in their recent years.**

**5.2 Renamed symbols go stale silently.** `LOTUSDEV` is absent from the instrument dump —
the tradeable symbol is `LOTUSDEV-BE`. The nightly refresh asks for the dead name, gets
nothing, treats it as "no new bars". Its data is frozen 126 days. Ten of eleven stale young
names are missing from the dump; six freeze on the same day (2026-05-08), a batch series
migration. **IPO Base is the most exposed** — it trades exactly the young, thin names NSE
moves to trade-for-trade. **NOT FIXED.**

---

## 6. WHERE EVERYTHING LIVES

**App pages (all live, hard-refresh to pick up the bundle):**

| Page | URL |
|---|---|
| **The roster — start here** | `/app/backtest/mpf-honest-entries-roster-2026-09` |
| Open Alpha original study (headline now known look-ahead) | `/app/backtest/bluesky-ath-breakout-research142` |
| Open Alpha v2 — Base Age (the candidate) | `/app/backtest/ath-base-age-breakout-research161` |
| IPO Base — corrected 11-Sep | `/app/backtest/ipo-base-breakout-research153` |
| Open Alpha book page, with a 2-year trade ledger (collapsible, last card) | `/app/bluesky-paper` |
| Strategies register — OA row corrected | `/app/strategies` |

**Research folders:**

| Path | What |
|---|---|
| `research/158_oa_arming_width/` | the entry audit across all three books; STATUS-MD, all logs |
| `research/158_oa_arming_width/scripts/verify_published_trades.py` | **the audit against the source's own 54 trades — reusable template** |
| `research/158_oa_arming_width/scripts/build_etf_list.py` | the fund-exclusion builder |
| `research/158_oa_arming_width/scripts/oa_entry_mechanics.py` | forked engine: `--entry-mode`, `--fund-mask`, `--base-start`, gate actions |
| `research/159_oa_honest_reoptimization/` | the re-optimisation; stageA/B/C CSVs, curves, correlations |
| `research/159_.../results/full_period_after_tax.csv` | **the 20.4-year curves behind §3** |
| `research/159_.../results/all_systems_after_tax.csv` | the 2016–2026 curves + correlations |
| `research/159_.../results/stageC_gates.csv` | the 29-gate bake-off |
| `research/160_quality_growth_near_ath/` | **sibling session** — Quality Summit |
| `research/161_ath_base_age_breakout/` | **sibling session** — Open Alpha v2 |
| `frontend/public/mpf-honest-roster-2026-09.png` | log growth + drawdown chart |

**Doctrine updated (committed):** `research/QUANT_RESEARCH_PLAYBOOK.md` §4 (8th sin) and
§5A; `.claude/agents/quant-researcher.md` §5A and known-data-defects; `~/.claude/
QUANT_RESEARCH_PLAYBOOK.md` (global, **awaiting Arun's claude-state sync**).

**`TODO.md`** carries five dated 11-Sep entries: the ETF universe defect, the rename/refresh
defect, Open Alpha paused, IPO halved, True North clean.

**Commits:** 46 on 11-Sep. Key: `18066bfc` (finding), `f68efc9c` (three-book audit),
`677b696a` (register), `1b0612aa` (trade ledger), `21dd8fa6` (IPO page), `07cc8709`
(playbook), `56ae5ebf` (push unblocked).

---

## 7. OPEN ITEMS — what is NOT done

**COMPLETED after this file was first written:** the after-tax re-run finished (70
cells, `research/159_.../results/after_tax_tables.csv`) and **the roster page is now 100%
post-tax** — zero pre-tax figures remain on it, and its three evidence tables are generated
from that CSV rather than retyped. Committed `c92a4749`. Every conclusion survived the tax
layer unchanged in direction: the breakout rule still beats random name selection by +9.9
points after tax (was +11.1 pre-tax), and the VIX 70th-percentile gate still adds +0.137 of
Calmar (was +0.208).

**Owed, in priority order:**

1. **IPO Base re-optimisation on the honest entry — NOT STARTED.** Its 680-cell sweep was
   scored on the look-ahead entry, exactly like Open Alpha's. Given the trail inverted for
   OA, IPO's SMA-20 trail and +25% target are suspect. Staged plan: (a) exit economics —
   trail 10–50 × target {+25%, +50%, none} × stop {6,8,10,none}; (b) base geometry — L,
   depth, age band; then null control and gate bake-off, after tax throughout.
2. **`services/oa_entry.py` still has the inverted condition AND the old ETF filter.** Entry
   crons are commented out (backup `/tmp/mpf/ct.bak.20260911-111828`). **Do not restore them
   before both are fixed.**
3. **The rename/refresh defect (§5.2).** `scripts/refresh_daily_universe.py` should resolve
   each symbol against the instrument dump and follow or raise a rename rather than
   returning quietly.
4. **research/142's published figures carry the ETF contamination** — re-run owed before
   they are cited again.
5. **True North's curve is a single path** while every other book is a 30-seed median. A
   like-for-like re-run under the same ensemble is owed before comparing to the decimal.
6. **Research-number collision:** two folders numbered 159 (`159_oa_honest_reoptimization`
   this session, `159_rounding_base_breakout` sibling). Mine should renumber, but it is
   referenced from the roster page, its STATUS-MD and several scripts.
7. **Blend/allocation work across TN + Base Age + IPO — NOT STARTED.** This is the only
   structure that plausibly clears Arun's 25% target: no single book does, and the year table
   shows why a blend might (IPO returned +43.7% and +68.5% in 2016 and 2020 when Open Alpha
   was flat; True North carried 2022 and 2025).

---

## 8. INSTRUCTIONS FOR THE UNIFIED REPORT — READ THIS BEFORE EDITING

**8.1 The page already exists and TWO sessions have written to it.**
`frontend/src/data/backtests.ts`, slug `mpf-honest-entries-roster-2026-09`. This session
built it; the sibling session has since added Quality Summit tables on a 2018–2026 window.
**Check `git log -1` on that file and re-read it before editing** — do not assume your copy
is current.

**8.2 Arun's explicit, repeated instructions about this page.** He raised each more than
once, so treat them as binding:

- **POST-TAX ONLY.** Mixed pre/post-tax tables made the page unreadable. He asked three
  times. Any pre-tax number must go or be converted.
- **Every table states WHICH SYSTEM, WHICH WINDOW, WHICH BASIS.** He could not tell whether
  a 24.45% figure belonged to the same book as a 23.0% figure. It did not.
- **Name systems, don't version them.** "Open Alpha · Base Age" and "Open Alpha · ATH + VIX"
  rather than v2/v3 — they are different signals, not revisions.
- **Show the test period, average invested and average in cash ON the table**, and state the
  cash-yield assumption as a caveat.
- **Use the entire common testing period**, not a clipped window.
- **Answer his questions in Q&A form** (`.claude/CLAUDE.md`, top section, binding).

**8.3 The hardest editorial problem you inherit: THREE WINDOWS.**
Full period 2006–2026 (summary), 2016–2026 (year table + correlations, because OA v3's VIX
gate needs 2015+ data), 2018–2026 (Quality Summit, because point-in-time fundamentals need
four filed fiscal years and Screener starts FY2015). Arun has already complained about
exactly this confusion. **Decide one headline window and make everything else visibly
secondary.** My recommendation: headline on the longest common period that includes the
systems actually being chosen between, with shorter-history systems in a clearly-labelled
second table — because 2008 and 2020 are where the drawdowns that matter happened, and a
window starting 2018 throws both away.

**8.4 Drop Open Alpha v3 from the headline.** Arun asked why it was still there. Keep it as
one labelled row or a footnote — it is the evidence for the 40.8% → −1.7% correction, and
nothing more.

**8.5 What the unified report must not lose.** The correction is the most important thing on
the page. Two live books were justified by numbers that cannot be earned, and the page has
to say so plainly, with the trade-list evidence, before it shows anyone a CAGR.

---

## 9. LIVE STATE AT HANDOVER

- **Open Alpha: BUYING PAUSED.** Both entry crons commented out. Exits, stops, trail,
  marks and reconcile all still run. 12 held positions came from the 04-Sep seed on the
  correct condition. Arun cancelled the four resting buy-stops by hand on 11-Sep.
- **True North: untouched, running normally.**
- **IPO Base: untouched, running normally.** Seeded 08-Sep-2026, 1 position.
- Nothing else was changed in live trading. No orders placed after 10:29 IST on 11-Sep.
