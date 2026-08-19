# TimeB/COMB Naming + Live-Plan-vs-Study Verdict — HANDOFF for the research/111 session

STATUS: **AWAITING VERDICT** (packaged 2026-08-19 eve by the ops session; the receiving
session holds the deeper research/111 context and should return the filled VERDICT FORM
in section 6. Until it returns, the live config runs AS DEPLOYED — Option B, unchanged.)

> Laptop: `docs/TIMEB_COMB_NAMING_AND_PLAN_VERDICT_HANDOFF.md` · VPS: same path under
> `/home/arun/quantifyd/`. Companion context: `docs/CSL_TIMEB_SENSEX_LIVE_DEPLOY_STATUS.md`,
> `/app/scaleup`, `research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py` (BOOKS),
> `backtest_data/csl_paper_config.json` (FROZEN per-DTE cells).

---

## 1. Why this handoff exists

Two things surfaced on 19-Aug that the ops session cannot settle alone:

1. **Naming is construction-false.** "TimeB" and "COMB" are used as book names, but they
   are really **constructions**: COMB = full-day straddle with a combined-premium SL;
   TimeB = the same thing in a sub-day window. On expiry days the frozen "TimeB" windows
   are full-day — i.e., **TimeB collapses into COMB on DTE0 (and NIFTY DTE3)**. Today a
   book named `CSL_TIMEB_NIFTY_THU` was created that is, by construction, a **COMB**.
   Arun: "1st of all lets name systems rightly."
2. **The live plan should be exactly what the study prescribes.** The ops session built
   the current allocation from the lab JSON (`csl_best_configs.json`) + margin math, but
   the research/111 session owns the study's real conclusions (risk-adjusted selection
   logic, SL choices, sizing doctrine). Several deviations/uncertainties are listed in
   section 5 — each needs a study-grounded verdict.

## 2. Current book inventory (as deployed tonight, exact)

| Book key | Construction (true) | Days & window (frozen cfg) | Lots/qty | Mode |
|---|---|---|---|---|
| `NAS_COMB20` | COMB · NIFTY full-day | Mon/Tue/Thu/Fri 09:16→15:20, SL D0 25 / D1 30 / D2 30 / D3 20 (ex-Wed) | 2 / 130 | LIVE |
| `CSL_TIMEB_NIFTY` | TimeB · NIFTY windows | Mon 13:00–14:00 SL20 · Tue 09:30–11:00 SL25 · Fri 10:00–12:00 SL20 (**Thu removed 19-Aug**) | 8 / 520 (**10 from 24-Aug**) | LIVE |
| `CSL_TIMEB_NIFTY_THU` | **COMB** · NIFTY Thu-only (misnamed) | Thu 09:25→15:20 SL20 | 3 / 195 | LIVE (first run 20-Aug) |
| `CSL_TIMEB_SENSEX` | TimeB Wed + **COMB Thu** (one book, two constructions) | Wed 10:30–12:00 SL20 · Thu 09:20→15:20 **SL40** (Mon/Tue/Fri removed 19-Aug) | 8 / 160 (**10 from 24-Aug**) | LIVE |
| `CSL30F_NIFTY` | COMB · fixed SL30 control | all days 09:16→15:20 SL30 | 2 / 130 | paper |
| `CSL30F_SENSEX` | COMB · fixed SL30 control | all days 09:16→15:20 SL30 | 3 / 60 | paper |
| `NAS_C20_TRAIL` / `NAS_C20_SHIFT` | COMB + mgmt variants | all days, SL20 | 2 / 130 | paper (mgmt books have no live path) |

Margin context (measured 19-Aug): NIFTY ≈ ₹1.65L/lot, SENSEX ≈ ₹2.04L/lot straddle MIS;
capital ₹44.7L (cash-equiv ₹20.8L incl. LIQUIDCASE); Thursday peak as deployed ≈ ₹40.9L.

## 3. Data the ops session used (verify against the study's own artifacts)

Per-DTE cells from `static/app/straddles/csl_best_configs.json` (2,450-cell sweep;
assumed bases: **NIFTY cells at 10 lots, SENSEX at 5 lots** — inferred, VERIFY):

| Venue·DTE (day) | Chosen window (frozen) | mean/win | Best FULL-DAY cell | mean/win |
|---|---|---|---|---|
| NIFTY·D0 (Tue) | 09:30–11:00 SL25 | 13,264 / 93% | 09:20 SL25 | 18,423 / 71% |
| NIFTY·D1 (Mon) | 13:00–14:00 SL20 | 3,790 / 93% | 09:16 SL30 | 13,250 / 80% |
| NIFTY·D2 (Fri) | 10:00–12:00 SL20 | 5,524 / 81% | 09:20 SL40 | 9,021 / 62% |
| NIFTY·D3 (Thu) | 09:16–15:20 SL20 (= full-day) | 16,956 / 91% | same cell | — |
| NIFTY·D4 (Wed) | 10:30–12:00 SL20 | 2,926 / 75% | 09:20 SL30 | **−75 / 56%** |
| SENSEX·D0 (Thu) | 09:20–15:20 **frozen SL40** | 14,322 / 94% | 09:16 **SL none** | 15,042 / **100%** (n=11) |
| SENSEX·D1 (Wed) | 10:30–12:00 SL20 | 1,612 / 75% | 09:16 SL20 | **−571 / 64%** |
| SENSEX·D2 (Tue) | 09:25–11:00 SL20 | 3,564 / 93% | 09:16 SL20 | 2,995 / 78% |
| SENSEX·D3 (Mon) | 13:00–14:00 SL20 | 1,080 / 80% | 09:16 SL20 | 4,854 / 78% |
| SENSEX·D4 (Fri) | 10:30–12:00 SL20 | 1,614 / 80% | 09:16 SL25 | 3,077 / 73% |

19-Aug decisions already deployed from this data (Arun signed off): TB-SENSEX live days
restricted to Wed+Thu; NIFTY Thursday runs COMB 2L + the new 3L Thu book (Option B —
"reduced NIFTY TimeB to match current capital"); ladder page `/app/scaleup` targets
₹89k/wk at 1.0×.

## 4. Proposed naming taxonomy (for ratification, NOT yet applied)

Name = `<VENUE>_<CONSTRUCTION>[_<SCOPE>]`, where construction ∈ {COMB, TIMEB} is decided
by the *frozen window*, not by history:

| Current key | Proposed name | Rationale |
|---|---|---|
| `NAS_COMB20` | `NIFTY_COMB` | is what it is (per-DTE SLs, ex-Wed) |
| `CSL_TIMEB_NIFTY` | `NIFTY_TIMEB` | windows only, now that Thu is out |
| `CSL_TIMEB_NIFTY_THU` | `NIFTY_COMB_THU` | full-day construction — a COMB, not a TimeB |
| `CSL_TIMEB_SENSEX` | either split into `SENSEX_TIMEB_WED` + `SENSEX_COMB_THU`, or rename `SENSEX_WEDTHU` | one book currently holds two constructions |
| `CSL30F_NIFTY` / `CSL30F_SENSEX` | `NIFTY_COMB30_CTL` / `SENSEX_COMB30_CTL` | they are fixed-SL COMB control arms |

**Implementation caution (ops session's input):** book keys thread through the frozen
config JSON, `csl_paper_state.json` records ("already recorded today" guard + cum),
the publish JSONs, the desktop alert watcher, the React pages, and the register.
A rename needs either (a) a one-shot key-migration script across state+config with the
daemon stopped (weekend job), or (b) keys stay, display labels change (zero risk).
Recommend (b) now, (a) only if the receiving session wants canonical keys.

## 5. Questions needing the study session's VERDICT

| # | Question | Ops session's current position | Confidence |
|---|---|---|---|
| Q1 | Ratify or amend the taxonomy in §4 (incl. split-vs-rename of the SENSEX book)? | proposal as written, labels-first | medium |
| Q2 | **SENSEX Thu SL: frozen cfg says SL40, lab best full-day says SL *none* (15,042/100%, n=11) and the 13-Aug stop-by-DTE study said "Thu = HOLD, any stop sabotages the decay".** Which is the study's true prescription? | suspect SL none (HOLD) is right and SL40 is a frozen-config artifact — but did NOT change it | **low — this is the big one** |
| Q3 | NIFTY Thursday now = 5 lots of the same full-day SL20 cell split across 2 books (COMB 2L + THU 3L, entries 09:16/09:25). Would the study rather size it as one 5L book (or different total)? | keep split (margin-gate sequencing reason documented) but unify naming | medium |
| Q4 | Wed/Fri/Mon cuts of TB-SENSEX and the ex-Wed rule for NIFTY COMB — consistent with the study's per-day verdicts? (Wed full-day cells are negative BOTH venues; windows small-positive) | yes per the cells; cuts deployed | high |
| Q5 | `CSL30F_SENSEX` promotion path: its Thursday pure-hold twin scores ~15,042/100% — is more Thursday SENSEX size (via promoting it or upsizing TB-SX Thu) the study's preferred next scale step vs. the generic 1.25× ladder on `/app/scaleup`? | undecided; needs study's sizing doctrine + more paper days | low |
| Q6 | The 1.0× weekly target ₹89k (lab in-sample sum, bases per §3) — does the study endorse this as the Row-1 bar, or prescribe a haircut (live-vs-lab gap: e.g. TB-N Tue live +276/lot vs lab 1,326/lot)? | posted lab basis with an explicit in-sample caveat | medium |
| Q7 | Cell bases: confirm NIFTY cells = 10 lots, SENSEX = 5 lots (all §3 math rests on this) | inferred from ₹2,753/lot Thu study match | medium |

## 6. VERDICT FORM (the other session fills and returns; ops session implements)

```
Q1 naming:        RATIFIED / AMENDED: ____________________  (keys-migrate | labels-only)
Q2 SENSEX Thu SL: KEEP SL40 / CHANGE TO none(HOLD) / OTHER: ____
Q3 NIFTY Thu:     KEEP 2L+3L split / MERGE AS ____L single book / RESIZE TO: ____
Q4 day cuts:      CONFIRMED / AMEND: ____________________
Q5 next scale:    LADDER 1.25x AS-IS / PREFER SENSEX-THU CONCENTRATION: ____ / WAIT n=__ days
Q6 Row-1 target:  KEEP 89k LAB BASIS / SET TO: ₹____k (haircut rule: ________)
Q7 bases:         CONFIRMED 10L/5L / CORRECT TO: ____
Constraints honoured: no restart before 15:40 IST; daemon config edits land before 09:12;
any SL/size change is a strategy change needing Arun's sign-off in that session or this.
```

## 7. Standing state while awaiting verdict

- Tomorrow (Thu 20-Aug) trades **Option B exactly as deployed** (§2 table).
- Monday 24-Aug: TB books step 8→10 (registered in Ops) — proceeds unless the verdict
  says otherwise.
- No renames, no SL changes, no sizing changes until the form returns.
