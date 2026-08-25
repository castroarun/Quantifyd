# Portfolio Defence Bake-off — trail the peak, buy wings, or spread the strikes?

> **RENUMBERED 125 → 126 on 2026-08-25.** `research/125` collided with a parallel session's
> `125_expiry_afternoon_straddle`. Folder, INDEX row and all paths now read **126**.

STATUS: **DONE (v2)** (commissioned 2026-08-25 by Arun, mid-session, after a live give-back;
scope extended twice the same day — bought OTM wings, then strike/entry diversification)

**VERDICT: NO EDGE — portfolio profit trail (Arm A), entry-time wings (Arm B).
SIGNAL (not investable) — profit-triggered portfolio wings (Arm B2), symmetric strike
spread (Arm C).** Plus a **GOVERNANCE DEFECT** found while fixing a scope error: an 8-lot
real-money book invisible to every automated view. Full write-up in `results/RESULTS.md`.

## 2. The Ask

**What Arun said (2026-08-25 ~14:30 IST):** "from almost 19k profits once, we are now in
losses, we must look to trail the profits of our entire portfolio, pls study"
**and, same conversation:** "or even look to buy OTM options for security".

**What we are actually testing.** Today's live book (9:16 suite + COMB + TimeB) peaked at
**+₹14,983 at 14:03** and was **+₹7,442 by 14:33 — ₹7,540 handed back in 30 minutes**; the
suite alone went +₹5,187 → −₹410. Nothing intervened, and nothing *could* have:

- the NIFTY venue trail watches **only the three 9:16 systems (6 lots)** and arms at
  **+₹2,000/lot = +₹12,000**. The suite peaked at **+₹5,187** — it has never been close to
  arming on a day like this.
- the day's profit lived in books the overlay does not watch: **TimeB +₹3,402** (banked
  11:00) and **COMB +₹4,550 at its peak, −₹566 now**.
- so the *portfolio* peak is unmonitored by construction. This is a structural gap, not a
  malfunction.

> **Arm A (trail).** Does a PORTFOLIO-level profit trail — armed on the combined live book
> across venues and sleeves — beat both (a) no trail and (b) today's per-venue suite trail,
> net of the REAL cost of firing it? If yes, at what arm level and give-back?
>
> **Arm B (bought OTM wings).** Does converting the short straddles into defined-risk
> structures (iron fly / iron condor) by BUYING OTM options pay for itself — as an
> ALTERNATIVE to the trail, or as a COMPLEMENT to it — against the same nulls?

## 3. The Base — what is being tested

- **Scope:** the live book only. NIFTY 9:16 suite (ATM/ATM2/ATM4, 2L each) + NAS_COMB20
  (2L, 5L on DTE3) + CSL_TIMEB_NIFTY (8L) + CSL_TIMEB_SENSEX (8L) + CSL30F_SENSEX_WED (3L).
  Rules are read from the FROZEN live config, not from memory. Paper and parked books are
  excluded from the P&L but MUST be excluded consistently.

### 3A. Arm A — the portfolio profit trail

- **The instrument under test:** arm at combined live P&L ≥ ARM; thereafter track the
  running peak; exit everything still open when P&L falls to peak − GIVEBACK. Sweep ARM and
  GIVEBACK as both absolute ₹ and as a fraction of peak.
- **Variants that must be tested alongside, not assumed away:**
  1. flat trail (arm + fixed give-back) — today's suite shape, scaled to the portfolio
  2. ratchet / percentage-of-peak give-back
  3. time-conditioned (a trail that only arms after N minutes, or tightens after 14:00 —
     today's give-back was entirely post-14:00)
  4. per-venue portfolio trail vs one global trail across NIFTY+SENSEX
  5. **close-everything vs close-only-the-losing-sleeve** — a portfolio trail that shuts a
     book which is individually fine has a real cost; price it.

### 3B. Arm B — BOUGHT OTM WINGS (added 2026-08-25, Arun scope addition)

Convert each short straddle into a defined-risk structure by buying an OTM CE and an OTM PE
of the same expiry (iron fly / iron condor). Axes:

| Axis | Values |
|---|---|
| wing distance | NIFTY 100 / 150 / 200 / 250 / 300 / 400 / 500 pts; SENSEX scaled ×4 by index level (400/600/800/1000/1200/1600/2000) |
| when bought | **AT ENTRY** (defined-risk all day) vs **AFTER THE BOOK IS UP** (buy wings once portfolio P&L ≥ ARM — the direct analogue of the trail, "lock the profit with wings") |
| scope | per-book wings (every sleeve hedged) vs portfolio-level (hedge only the largest sleeve — TimeB 8L — and leave the 2L sleeves naked) |
| combination | wings alone (alternative to trail) vs wings + trail (complement) |

- **Wings are held to the sleeve's own exit** and sold back at that exit (or abandoned if
  worthless). A wing is never rolled.
- **Structural point that makes this genuinely different from the trail:** wings are a
  CERTAIN premium cost paid EVERY day for an UNCERTAIN benefit; the trail is an occasional
  but expensive firing cost. Both are insurance with different payment schedules. The
  comparison must be made on that basis **and on the tail** — wings cap the disaster day,
  the trail does not (it only limits give-back on days that were already winning).

### 3C. Arm C — STRIKE / ENTRY DIVERSIFICATION (added 2026-08-25, second scope addition)

Triggered by the 2026-08-25 observation that every NIFTY book sold the **same strike**.
Confirmed at source, not just from one day: `NAS_916_ATM_DEFAULTS`, `NAS_916_ATM2_DEFAULTS`
and `NAS_916_ATM4_DEFAULTS` all inherit `NAS_ATM_DEFAULTS` with `entry_start_time: '09:16'`
— **same straddle, same strike, same minute**, differing only in exit machinery (ATM2 =
₹2,500/lot rupee stop one-and-done; ATM4 = `max_rolls: 1`). So the "suite" is one position
at 6 lots with three exit rules.

This is the **cheapest candidate in the bake-off**: it pays no premium and no firing cost.
Tested at **equal notional** — 3 clones × 2 lots of the COMB-shape construction:

| portfolio | legs |
|---|---|
| CLONE_SAME | (09:16, +0) × 3 — today's shape |
| DIV_STRIKE_1 / DIV_STRIKE_2 | offsets −1/0/+1 and −2/0/+2 |
| DIV_ENTRY / DIV_ENTRY_WIDE | 09:16 / 09:31 / 09:46 and 09:16 / 09:46 / 10:01 |
| DIV_BOTH | (−1, 09:16) (0, 09:31) (+1, 09:46) |

Measured alongside: strike-overlap frequency, cross-book correlation **overall vs on the
worst decile**, and exit-time clustering — to test (not assume) the "correlation ≈ 1 when
it matters" hypothesis and the "we are our own counterparty on the exit" hypothesis.

- **Null alternatives both arms must beat:** no trail / no wings at all (naked, as deployed);
  the existing suite-only trail; and a plain daily profit target (fixed TP), which is the
  cheaper thing that also caps give-back.

## 4. The prior it must confront — three studies say defence like this LOSES

This is not a blank slate and the study must engage the existing evidence rather than
rediscover it:

- **r/116** tested exactly this family (breakeven-clamp, multiplicative ratchet,
  peak-giveback, time-scaled) on the SENSEX backstop and concluded **STATIC IS OPTIMAL —
  every ratchet gave back more than it saved.**
- **r/114 / r/121 / r/122 / r/124** independently found that tightening defence
  **manufactures losses** and makes the worst day worse (four reproductions).
- **r/90 (portfolio bracket)** is the one positive for a portfolio-level overlay: a WIDE
  daily STOP (−8k) ~4× the book's P&L. But the same study found **a daily take-profit is
  value-destructive** (TP 4k = −34,020 vs +17,530 baseline) because it caps the fat right
  tail. **A profit trail is a soft take-profit** — that is the prior Arm A must beat.
- **r/60** validated POSITIONAL (overnight, multi-day) NIFTY iron flies with ±500 wings
  (~+₹8L / 7yr at 10 lots). **The INTRADAY held-wing result on this project was INVALID**
  (stale far-OTM quotes produced impossible P&L). Arm B is therefore under suspicion by
  default and must prove its quotes were live (§5B).

The honest hypothesis is **narrow**: defence may pay at the PORTFOLIO level (where peaks are
~3× larger and genuinely unmonitored) even though it fails at the sleeve level. If the data
says otherwise, say so — a fourth "static wins" is a perfectly good answer.

## 5. The cost that decides it — use the MEASURED model (2026-08-25)

### 5A. Arm A cost — the trail only ever fires mid-session at market

| leg-side | measured slippage |
|---|---|
| entry (sell) | −0.228 pt (favourable) |
| exit, time/EOD | +0.178 pt |
| **exit, stop/forced** | **+6.548 pt** (median +4.80, p95 +17.85) |

Measured on 443 real live leg-sides (Kite `average_price` vs `option_chain` LTP, same
minute). **Every trail firing pays ~6.5 pt per leg-side on everything it closes**, plus the
exact Zerodha rate card (₹20/order brokerage, STT 0.1% on sell premium, txn 0.03503%, IPFT,
SEBI, stamp 0.003% on buy, GST 18% on brokerage+txn+ipft+sebi). At 14 lots across books that
is a large, certain cost paid to avoid an uncertain give-back — the crux of Arm A. Do NOT use
the retired flat ₹250/lot constant. Implementation: `cost_per_lot()` shape from
`research/122_window_risk_atlas/scripts/stage_a_alldays.py`.

### 5B. Arm B cost — and the DATA TRAP that has already invalidated this exact test once

**BINDING guards (a held-wing backtest on this project produced impossible P&L before):**

1. **Bought wings are priced at the ASK + slippage. Sold back at the BID.** Never at LTP.
   A wing marked from a stale LTP shows a fictitious gain.
2. **r/89 binding rule — real traded volume/OI filter.** A wing strike with no traded volume
   that day is NOT tradable: move to the nearest liquid strike or drop the day. Report how
   often that happened.
3. **Staleness diagnostics are mandatory and must be REPORTED, not assumed:** consecutive
   identical prints at the wing strike, bid==0 or ask==0 fraction, spread as % of mid, and
   the fraction of minutes where the wing quote did not move while the underlying moved.
4. If the intraday wing numbers look good, **be suspicious and prove the quotes were live.**
   If Arm B dies on data quality rather than on economics, **say exactly that** — "cannot be
   measured with the data we hold" is a valid and more useful verdict than an untrustworthy
   number.
5. Wing purchase pays the same rate card plus brokerage ₹20/order on 2 extra legs per sleeve;
   STT on the sell-back of the long option is modelled explicitly.


---

## 3D. Arm B2 — PROFIT-TRIGGERED PORTFOLIO WINGS (added 2026-08-25, third scope addition)

Arun, precisely: *"buying wings only after we achieve a profit level at the PORTFOLIO in
order to lock it, not wings from the beginning."* Arm B's §2.3 AFTERUP probe was the
closest thing but triggered **per-sleeve** at one level and one distance on n=11 — it did
not answer the question. Promoted here to a full arm.

**Mechanism under test.** When the book is UP it is up *because premium has decayed*, so at
the moment protection is wanted the wings are at their cheapest of the day and are funded
out of profit already earned. Unlike the trail, a wing **caps the tail without surrendering
the remaining theta** and without paying the measured +6.548 pt/leg-side forced-exit
slippage. **Counter-hypothesis to test honestly:** wings bought when the book is up are far
from the money (the market has not moved), so they are cheap but rarely pay.

| axis | values |
|---|---|
| trigger | portfolio P&L ≥ ₹5k/8k/10k/12k/15k/20k · ≥20/30/40% of total credit · time-conditioned (arm only after 13:00 / 14:00) |
| distance | NIFTY 100–500, SENSEX 400–2000 |
| coverage | every open book · only the largest exposure · only the venue moving against us |
| unwind | hold to EOD · sell back if the book recovers above the trigger |
| nulls | naked · the trail · entry-time wings · **and the incumbent champion, TimeB's CLOCK EXIT** |

## 3E. Arm C v2 — the engine that failed reconciliation and was rebuilt

The first Arm-C engine replayed `config.py`'s documented 9:16 rules (per-leg 30% SL,
trail-to-cost, re-enter ×5). It produced **−₹437,588** against the live book's **+₹164,988**
(4.04 cycles/day vs 1.04 live). Cause, from the live trade table: real `916_ATM` exit
reasons are **58 eod_squareoff, 10 ST_EXIT, ZERO SL_HIT** — the documented per-leg stop is
**dormant live**. Numbers discarded; Arm C re-based on three constructions that reconcile:
**HOLD** (09:16→15:15 no stop, what the suite actually does 84% of days), **COMB** (r/116
per-DTE combined SL) and **RUPEE2500** (the ATM2 ₹2,500/lot stop, isolated). Offsets
0/±1/±2/±3/±4; configurations symmetric / all-up / all-down / laddered / random control;
nulls = ALL_ATM, random-leg placebo, **and plain downsizing at equal worst-day**.

## 5D. Scope error found and fixed — TIMEB2's 8 real lots

`CSL_TIMEB2_NIFTY` sold **520 qty (8 lots)** on 2026-08-25 (tag `TIMEB2_NIFTY`, booked
−₹2,990) — larger than COMB — and was missing from v1 because the harness derived the book
universe from the daemon's `BOOKS` dict. The real book is a standalone one-shot with the
**same name** as a 2-lot PAPER book, publishing only its own JSON. It was created and first
run 2026-08-25 (commit `1b78873`), so the historical samples are **not** understated; what
it corrupted was the worked example and the method. Full defect list in RESULTS §1.


## 6. Data

- **Live truth (small n, high fidelity):** `nas_mtm_snapshots` in each live NAS DB (per-minute
  day_pnl per system; 70 days 2026-05-20 → 2026-08-25) + `csl_paper_state.json`
  records/series for COMB and TimeB (**rolling window — only 8 days retained**, 2026-08-14 →
  2026-08-25) + `static/app/csl_paper_live.json` for still-open books (a book is DROPPED from
  the live json once it closes — fall back to the day record).
- **Replay (larger n):** `options_data.db :: option_chain`, 1-min, **88 recorded days from
  2026-04-20**, both venues, with `bid`/`ask`/`volume`/`oi` columns populated
  (NIFTY bid>0 ~100% of rows, SENSEX ~68–78%). Rebuild each book's intraday curve from its
  deployed rules, sum to a portfolio curve, then sweep both defence arms over it. Reject
  frozen-chain holidays (2026-05-01, 05-28, 06-26; guard <50 distinct spot prints) and
  partial sessions.
- **Tail context:** SENSEX 1-min 2021→, NIFTY 5-min 2015→ under the r/121 max-excursion
  licence — how often does a day that peaks at +X give it all back?
- DTE-era labelling mandatory (NIFTY Thu→Tue 2025-09; SENSEX Fri→Tue→Thu).
- Facts: NIFTY lot 65, SENSEX lot 20 (`option_chain.lot_size` is WRONG).

## 7. Success criterion

A defence — trail (Arm A) or wings (Arm B) — is recommended only if, net of its measured
cost, it:

(a) raises the mean or median day, **or** materially cuts the left tail without cutting the
    mean;
(b) does so on a **plateau** of neighbouring cells (ARM × GIVEBACK for A; wing distance ×
    trigger for B) rather than at one peak;
(c) survives a family-wise haircut over the swept grid (Westfall–Young style, per r/120/121);
(d) beats **all three nulls**: naked/no-defence, the existing suite-only trail, and fixed TP;
(e) states how many days it would have fired / been paid for, and what it cost on the days it
    was **needless** (the book recovered on its own);
(f) **for Arm B additionally:** passes the §5B staleness + liquidity audit. Wing economics are
    only reportable on days where both wings had live, traded quotes. **Any wing result that
    depends on stale quotes is reported as UNMEASURABLE, not as a number.**

(g) **for Arm C additionally:** it must beat a **random-leg placebo** — 200 random 3-leg
    portfolios drawn from the same cell menu — on both the tail and the total. A tail gain
    inside the placebo band is reported as noise, not as a finding.

Report the give-back distribution with and without each defence, and the worst-day delta.
All three arms are finally compared on **₹ paid per ₹1 of worst-day tail removed**, which is
the only axis on which insurance with different payment schedules is comparable.

## 8. Process

Standard: read-only DBs, niced, `scripts/` + `results/`, live event log in this file,
`results/RESULTS.md` with a bold verdict (NO EDGE / SIGNAL / STRATEGY / CONCLUDED), row added
to `research/INDEX.md`, commit only this folder + INDEX. **No live rule changes** — any
recommendation goes to Arun for sign-off with its own after-15:40 deploy.

## 9. Status log

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-25 14:35 IST | Commissioned; sections 1–8 written before any compute | live give-back measured: peak +14,983 @14:03 → +7,442 @14:33 |
| 2026-08-25 ~16:10 IST | Context absorbed: playbook, r/116, r/90-bracket, r/122, r/121 | key priors: r/116 static optimal; r/90 "a daily TP is value-destructive" — a trail is a soft TP |
| 2026-08-25 ~16:20 IST | **SCOPE EXTENDED by Arun: "buy OTM options for security"** | §3B / §5B / §7(f) added BEFORE compute; wings arm carries the stale-quote trap guards |
| 2026-08-25 ~16:25 IST | Data recon done | option_chain 88 days, bid/ask/vol/oi present; live NAS mtm 70 days; csl_paper_state records only 8 days (rolling) → replay is the workhorse |
| 2026-08-25 ~16:30 IST | Live book rules read from FROZEN config (`csl_paper_config.json`, frozen 2026-08-13) | TB-N: DTE0 09:30–11:00 SL25, DTE2 10:00–12:00 SL20 @8L. TB-SX: DTE0 13:00–15:20 no-SL @8L, DTE1 10:30–12:00 SL20 @8L. COMB20: 09:16–15:20 SL 25/30/30/20 (DTE3 @5L). SX-WED 09:16–15:20 SL30 @3L (DTE1 only). NOTE: brief said TB-SX Thu 5L; config says 8L — **config is truth** |
| 2026-08-25 ~14:52 IST | **Stage 0 DONE — reconciliation PASSES.** Suite-only peak **+₹5,187 @14:00** vs its **+₹12,000** arm → the live overlay **could never have armed**. Portfolio peak **+₹13,865 @14:03** → −₹5,160, give-back **₹19,026** | commission quoted +14,983; the ₹1,118 gap is book scope (this counts LIVE-money books only). Mechanism + peak minute reconcile exactly |
| 2026-08-25 ~14:52 IST | Stage 1 DONE — 83 days/venue, 149 sleeve-days, wing price paths built in the same chain pass | live-first: suite = REAL MTM (never modelled); CSL sleeves = replay from frozen config |
| 2026-08-25 ~14:56 IST | **DATA BUG CAUGHT AND FIXED before any conclusion** | the live suite ran **5 / 1 / 10 / 2 / 3 lots** across the window. An absolute-₹ trail grid is meaningless across a size change → all suite MTM **rescaled to the deployed 2 lots/system** (r/90 precedent: "replays the CURRENT config"). Every headline number post-dates this fix |
| 2026-08-25 ~15:00 IST | **Arm A DONE — NO EDGE. 0 of 132 trail cells beat the null.** Best = −₹84,427 (13.5% of book P&L) and improves the worst day by **₹0** | plateau map is **monotone to the boundary "no trail"** — the r/116 signature reproduced on a portfolio construction. Fixed-TP ladder independently reproduces r/90 |
| 2026-08-25 ~15:04 IST | Placebo: real trail ≫ random-minute exit (arm 10k: 508,510 vs placebo p95 314,747) but ≪ naked 626,965 | **the machinery is skilful; skilful early exiting still destroys value.** A stronger negative than "it's noise" |
| 2026-08-25 ~15:05 IST | **SECOND SCOPE ADDITION from Arun: strike/entry diversification.** §3C / §7(g) written before computing it | the free defence — no premium, no firing cost |
| 2026-08-25 ~15:06 IST | **Arm B DONE — staleness audit PASSES**, so the numbers are trustworthy; and they are bad | 0% one-sided quotes, median spread 0.3–1.5% of mid, identical-print runs 1.0–1.2 min, **0 sleeve-days excluded** by the r/89 volume rule. Economics: wings cost **28–100% of book P&L**; decomposition shows the bill is **decay (₹157,794) not spread (₹4,341)** — the wing hands back the theta the book earns |
| 2026-08-25 ~15:08 IST | **Arm C DONE — SIGNAL.** Concentration proved from **config**, not from one day | but the "correlation ≈ 1 when it matters" hypothesis is **REFUTED**: worst-decile pairwise corr is **−0.58 / −0.19 / −0.32**; every book lost on only 2 of 9 worst-decile days. Exit clustering also atypical (median span 9,220 s; only 7% of days inside 120 s) |
| 2026-08-25 ~15:08 IST | Arm C result: ±2-step strike spread — NIFTY worst −33,753 → **−23,582** for a mean cost that is **not significant** (t −1.26); SENSEX worst −29,166 → **−20,392 AND +₹63,748 total** (t +1.79) | placebo honesty: SENSEX CLONE_SAME's tail is **worse than the p05 of random 3-leg portfolios** (concentration penalty is real); the NIFTY tail gain sits **inside** the placebo band |
| 2026-08-25 ~15:15 IST | **RESULTS.md written; STATUS → DONE.** Recommendation: deploy nothing; take Arm C to G3 OOS + paper twin; the real answer to the give-back is a **window/time exit** (r/122), not a trail | no live change proposed — sign-off item for Arun |
| 2026-08-25 ~15:35 IST | **SCOPE ERROR caught by Arun: TIMEB2's 8 REAL lots excluded.** Universe rebuilt from broker evidence, not the daemon's BOOKS dict | name collision (real 8L one-shot vs paper 2L daemon book); `journal_kite_reconciliation` is EMPTY; journal `mode` flags force-paper squeeze variants as LIVE; no broker orderbook is persisted (Kite = today-only). TimeB2 created 2026-08-25 (commit 1b78873) → historical samples unaffected |
| 2026-08-25 ~15:40 IST | Worked example corrected: **peak +₹18,817 @14:01 → −₹7,240, give-back ₹26,058** | reconciles to the coordinator's booked +19,201/−8,402 — the gap is gross MTM vs booked net (TimeB2 marks −2,080, books −2,990 after the +6.548 pt exit slippage) |
| 2026-08-25 ~15:45 IST | **Arm C v1 DISCARDED on reconciliation failure** (−437,588 replay vs +164,988 live) | live exit reasons: 58 eod_squareoff / 10 ST_EXIT / **0 SL_HIT** — config's per-leg 30% SL is dormant. Re-based on HOLD / COMB / RUPEE2500 |
| 2026-08-25 ~15:55 IST | **Arm C v2 DONE — SIGNAL, tail only.** Symmetric ±4: NIFTY worst −50,146→−31,747, SENSEX −48,122→−31,428, **monotone in k on both venues** | **credit RISES** (722→844) → NOT downsizing; clears the random-leg placebo (99.8th / 99.6th pctile on worst-day). But **FAILS the family-wise haircut** (max&#124;t&#124; 2.09 vs 2.97) and the **mean advantage reverses OOS** (+989 → −1,064/day). Only SYMMETRIC works; all-up/all-down add directional delta |
| 2026-08-25 ~15:55 IST | Arm C interaction: **the stated prior is REFUTED** — the ₹2,500/lot stop fires MORE off-ATM (28.9% → 47–53%), not less | an off-ATM straddle carries intrinsic + delta, so it moves more in rupees. Both stops are minimised AT the money; for stopped constructions ATM is optimal |
| 2026-08-25 ~16:05 IST | **Arm B2 DONE — SIGNAL, not established.** 504 cells; **71 beat naked**; plateau clean (every trigger ≥₹12k positive at every distance, ≤₹10k negative) | mechanism VERIFIED on the days that matter: 07-08 peaked +21,425 → naked −72,351 → hedged **+39,468**; 06-12 peaked +21,161 → −54,614 → **+40,940**; and it never armed on the bad days that were never up |
| 2026-08-25 ~16:10 IST | Arm B2 robustness: **super-winner guard fails the best cells** (07-08 = 163% of ABS_15000's total; ex-top1 −98,391) | only **ABS_20000** survives: ex-top1 +42,098, t +0.32, same sign in both OOS halves. No cell reaches t=2. Rests on 2–3 events in 84 days |
| 2026-08-25 ~16:15 IST | **Renumbered 125 → 126** (collision with the parallel `125_expiry_afternoon_straddle`); RESULTS v2 written; STATUS → DONE | recommendation: fix the governance gap first; paper twins for Arm B2 (₹20k/100–300/ALL/EOD) and Arm C (NIFTY symmetric ±2..±4); no live change |
