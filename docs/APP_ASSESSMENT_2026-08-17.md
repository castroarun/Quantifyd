# Quantifyd — App Assessment (independent review, 2026-08-17)

Scope read on the VPS: **37 page files · 36 SPA routes · 27 sidebar entries · 391 backend routes ·
90 scheduled jobs · 128 research folders · 19,254 lines of page TSX · app.py 11,930 lines.**
Every number below is reproducible from the tree (audit scripts were in the session scratchpad;
re-derivable with greps described inline).

**Headline:** the engine is ahead of the shell. Execution safety and research rigour are strong;
what lags is legibility — naming, one vocabulary, one page scaffold, and registries that stay true
without someone remembering. The top three problems are drift, not bugs.

## Scorecard

| Dimension | Grade | Evidence |
|---|---|---|
| Execution & safety | Strong | kill switches, freeze flag, guardians, 15:40 rule + deferred restart |
| Research rigour | Strong | 26 published studies, 24 chart assets all present, no duplicate slugs |
| Link integrity | Good | 1 of 27 nav links dead · 1 of 102 API literals unrouted |
| Registry truth | **Drifting** | 3 of 5 live systems absent from the ledger · ~24 in-scope jobs unregistered |
| Uniformity | Uneven | 5 byte-identical CSS modules · 484 inline styles in one page |
| Clarity | Uneven | 18 of 27 nav labels differ from the page's own title |
| Front-end health | Uneven | one 1.25 MB chunk, no route splitting, 23 pages polling 1–60 s |
| Accessibility | Thin | 12 aria attrs total · `:focus-visible` in 1 of 30 stylesheets |

## P0 — broken now

1. **Settings nav item goes nowhere.** No React route, no Flask route; SPA catch-all returns 200 and
   renders NotFound. Build it (mode master switch, alert routing, Kite auth status) or remove it.
2. **`/app/reports` does not exist.** `nas_analyzer.py:5`, `options_outlier_scan.py:5` and
   `docs/LABS_AND_JOBS_REFERENCE.md:46-47` all point there; the real page is `/app/report`, nav label
   "Performance", page title "NAS performance report" — three names for one thing. Report.tsx does
   consume both JSONs, so only the naming is wrong.
3. **`/api/v2-ironfly/`** (bare prefix, trailing slash) is fetched from Straddles.tsx and matches no
   route — the only genuine unrouted call of 102.

## P1 — registries that stopped being true

> **Re-scoped 2026-08-17 per the owner's rulings:** the Journal is **live-systems-only by design**, and the
> ops centre needs to cover **live systems + research/re-assessment jobs only** — paper and parked books are
> deliberately out of scope. Findings 4 and 5 are rewritten to that rule: 5 shrank, 4 got sharper.

4. **Journal is live-only — and by that rule it is wrong in both directions.**
   Of the 5 live systems, **3 cannot reach the ledger**: TB-CSL NIFTY and NAS_COMB20 (real money, state in
   `csl_paper_state.json`) and Momentum ₹3L (own DB). Meanwhile the ledger **does** ingest 3 non-live
   systems — ORB Index 349 trades (−₹2,02,510), ORB Cash 46 (−₹21,888), KC6 6 (−₹5,528) — i.e.
   **≈ −₹2.3 lakh of paper/parked P&L is mixed into a real-money ledger**. NAS/SENSEX (691 trades, +₹12.8L)
   is correctly covered.
   **Fix:** filter the journal to live strategies (totals become true immediately, no new code), then add a
   CSL source + a momentum source, and retire/tag the orb, strangle and kc6 sources. Read-only projection.
5. **Ops gap, scoped to live + research: ≈24 jobs — including the chain the live book depends on.**
   52 of 90 apscheduler jobs are out of scope (orb, i75, kc6, maruthi, mq, n500m, nwv, pair, mst, eod, bnf,
   collar, trident, scanner). What remains and is **missing from both `ops_center.py` and the labs doc**:
   - **The morning token chain** — `auto_login.sh` 08:50, `token_heal.sh` 09:06, `preopen_restart.sh` 09:00,
     `killflag_premarket_check.py` 09:05. A stale-token cascade is a *known* failure class (dark ticker →
     the 09:16 one-shot can't be replayed) and none of its four guards are registered anywhere.
   - Per-minute live-page writers: `sensex_live_writer.py`, `publish_nifty_5m.py`
   - Live momentum book: `gen_momentum_scan.py` 16:20
   - Holdings: `holdings_snapshot/meta/events` (apscheduler) + `gen_holdings_ohlc.py`,
     `update_holdings_today_candle.py`
   - Guards: `db_integrity_watchdog`, `instruments_dump`, `premarket_brief_build/_fallback`
   - Research/re-assessment recorders: `sl_reanchor_shadow.py`, `dl_sensex_1min.py`, r/56 dual-ST, r/80
     condor, r/82 sensex-expiry, r/90 `regen_travel.sh`, mentor `capture_daily.py`
   - In the labs doc but **not** in ops_center (the two registries disagree): `snapshot_nas_eod`,
     `dump_nas_mtm`, `options_study_agg`, `backup_to_github_release`
   **Fix:** ops page diffs `scheduler.get_jobs()` + `crontab -l` against the curated table with paper/parked
   families filtered out by design; in-scope jobs with no entry render as **UNREGISTERED**. Register the
   token chain first. Pure read-only introspection.
6. **A book can be off for months invisibly.** I75WR: 7 jobs, 3 configs, dashboard, *empty DB*
   (every config `mode=off`). Pairs: same. MST: 10 positions on 07 May, nothing since, and it
   re-restores 6 legs on the dead 2026-05-19 expiry at every boot. Nothing surfaced any of it.
   **Fix: a liveness rule per system — mode + last trade + last signal + days idle, computed.**
7. **Seven legacy Jinja dashboards are live and unlinked**: `/agent /kc6 /collar /maruthi /bnf
   /tactical /trident`. `/maruthi` is the sharp edge (9 known correctness bugs, must not be
   re-enabled, dashboard still offers controls). Project CLAUDE.md lists only `/collar` as debt.

## P2 — structure & consistency

8. **Three route-registration idioms**: 336 `@app.route`, 31 blueprint routes (4 services),
   27 `add_url_rule` inside service modules. The third style made a systematic audit mis-flag 24
   endpoints as broken. 247 of 355 `/api` routes are never referenced by the SPA (mostly legitimate
   — Jinja pages, ops tooling — but unenumerated). Fix: blueprint per subsystem going forward +
   a `/api/_routes` introspection endpoint.
9. **Five paper-book pages are the same page five times** — HaPaper / FnomsPaper / OholPaper /
   OrbPaper / BreakoutPaper have byte-identical stylesheets (md5 `434740f8…`) and 194–256-line
   components. Collapse into one `<PaperBook>` driven by a config record.
10. **Nas.tsx 3,328 lines** (969 CSS, 276 inline styles, 12 intervals + SSE) and **Straddles.tsx
    1,925 lines with no stylesheet** (484 inline styles — against the project's CSS-modules rule) =
    27% of all page code. Extract by section on the next real change; don't rewrite.
11. **Nine words for three states**: LIVE (53 uses/12 pages), PAPER (40/10), REAL (26/3), Off (14/9),
    ARMED (7/4), Disabled (6/4), Parked (4/4), "Live trading", "Paper trading". Fix: three words and
    a shared `<ModeChip>` + `<ModeControl>` (only 5 pages have a mode toggle, 9 a kill switch, with
    no pattern to which).
12. **Money formatted four ways in two currencies**: formatPnl 16 pages, toLocaleString 18,
    Math.round 17, toFixed(0) 11; "₹" 15 pages vs "Rs" 12; `tabular-nums` in 20 of 30 stylesheets.
13. **Research references outgrew the index**: 128 folders, 80 with RESULTS.md, 26 published,
    **47 absent from `research/INDEX.md`** (including `111_sensex_manual_mgmt`, source of truth for
    the live stack). Numbering collision: `109_breakout_gate_freq` + `109_intraday_stocks`.
    Fix: generate INDEX.md from the tree; renumber one 109.
14. **Pages show today and hide their history.** N500M: page shows 1 trade; DB holds 31 closed over
    25 sessions, 58% wins, **+₹13,852**, and `n500m_equity` has 0 rows. Fix: standard footer —
    cumulative net, sessions, win rate, sparkline derived from the trade table.
15. **Acronym-only nav.** 18 of 27 labels differ from the page title (N500M = "Nifty 500 Intraday
    Momentum", NWV = "Nifty Weekly View", MST unexpanded anywhere). Keep short labels, add full name
    as tooltip + page lede, read both from the register.
16. **One 1.25 MB bundle** (389 kB gzip), `React.lazy` unused; polling 1 s–60 s with no shared policy
    (OptionsData 1 s; Strategies polls 11 endpoints every 5 s). Fix: lazy routes + `usePolling(url,
    tier)` that pauses on hidden tab and outside market hours.

## P3

17. **Four working pages have no way in**: `/strangle` (ORB Index, real day P&L), `/nas-panic`,
    `/journal/insights`, `/holdings/history`. ORB Index belongs in Paper Books; NAS Panic belongs as
    an action on the NAS page.
18. **17 of 36 pages lack the title/subtitle scaffold**; NotFound (39 lines, inline styles) offers no
    way back. Fix with a shared `<PageHeader>` (name, purpose, mode chip, size, Rules/Study/Journal).
19. **Accessibility not started**: 12 aria attributes across 19k lines, 24 clickable non-button
    elements (Straddles 8), `:focus-visible` in 1 of 30 stylesheets (the one written today).

## Self-audit — where 2026-08-17's work is part of the problem

- **The register is hand-maintained**, duplicating status/size that already live in configs and DBs —
  the same failure mode as the ops centre. Derive mode/size/activity; keep rules, evidence and
  change log by hand.
- **Day P&L reads "—" for 19 of 23 systems** — honest, but an empty column invites the very question
  ("is this running?") that triggered this review. Fill from trade tables or show last-activity.
- **Single-letter hotkeys** will collide with the first custom text input or `/`-to-search;
  a `g`-prefix or ⌘K palette is the durable form.
- **27 nav rows now carry a badge each** — more visual noise on a list already too long; the real fix
  is fewer top-level groups plus a palette.

## Ranked opportunities

| Move | Impact | Effort |
|---|---|---|
| Liveness rule per system | High | S |
| Journal sources for every book | High | M |
| Ops centre reads the live scheduler | High | S |
| `<PageHeader>` / `<ModeChip>` / `<ModeControl>` | High | M |
| Derive the register instead of declaring it | High | M |
| One `<PaperBook>` page from config | Med | M |
| Route-level `React.lazy` + polling tiers | Med | S |
| Per-book history footer (cumulative + sparkline) | Med | S |
| Generate `research/INDEX.md` | Med | S |
| Decide the seven legacy dashboards | Med | S |
| ⌘K palette over the hotkey map | Nice | M |
| Blueprint-per-subsystem migration | Nice | L |

## Suggested order

1. Half a day of dead-link hygiene (P0s + ORB Index into Paper Books + 404 link home).
2. **Register the morning token chain** (4 cron entries) — highest-consequence gap, ten-minute edit.
   Then make the ops page diff `scheduler.get_jobs()` + `crontab -l` with paper/parked filtered out.
3. The liveness rule — the finding that would have caught I75WR / Pairs / MST months ago.
4. **Journal = exactly the live book**: filter to live strategies today, then add CSL + momentum sources
   and retire the paper/parked ones.
5. Shared page furniture, applied to the five paper books first (they collapse anyway).
6. Only then the register's second pass (derive mode/size/activity).

## GUARDRAIL for all of the above (owner instruction, 2026-08-17)

**No change to live or paper trading logic.** Engines, scanners, executors, stops, trails, sizing and
entry/exit rules are off-limits for this cleanup. Every item in this plan is one of:

- a **read-only projection** (journal sources, liveness queries, ops introspection),
- a **display / routing** change (page furniture, nav, redirects, code splitting),
- a **shared component** that renders existing endpoints without changing their semantics, or
- **documentation**.

If a fix would require touching an executor or engine, it stops and becomes a separate, explicitly
approved strategy change with its own STATUS doc and an after-15:40 deploy.

## Not checked

Runtime behaviour in a browser, mobile/narrow viewports, the internals of the seven Jinja
dashboards, backend correctness or test coverage, auth/session and secret hygiene, DB schema and
migration safety, and whether the numbers each page displays are *right* — this review is the shell,
not the arithmetic. Strategy quality and capital allocation deliberately excluded (that belongs in
the weekly re-assessment).

Rendered version: published artifact "Quantifyd App Assessment" (2026-08-17).
