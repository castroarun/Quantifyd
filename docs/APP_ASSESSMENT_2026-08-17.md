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
| Registry truth | **Drifting** | ops centre 22 of 90 jobs · journal 4 of 23 systems |
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

4. **Journal sees 4 of 23 systems** (sources: nas, orb, kc6, strangle). Uncovered: momentum-3l,
   ha-paper, fnoms-paper, breakout-paper, ohol-paper, nwv, n500m, i75wr, pairs, mst, maruthi. The
   month P&L therefore looks complete and isn't. Two sources would fix most: one for JSON-state
   paper books, one for the SQLite books (`n500m_positions`, `i75_positions`, `pair_trades`).
5. **Ops & Review Centre covers 22 of 90 scheduler jobs** (6 groups / 22 rows / 18 reviews). Absent
   families: orb (9), i75 (7), kc6 (6), maruthi (6), mq (4), n500m (4), eod (3), holdings (3),
   strangle (3), nwv (2), bnf (2), premarket (2), trident (2), collar, db-integrity, instruments,
   mst, pair, scanner ≈ 59 jobs. **Fix: read `scheduler.get_jobs()` and show unregistered jobs as
   UNREGISTERED rows** instead of hand-maintaining existence.
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
2. The liveness rule — the finding that would have caught I75WR / Pairs / MST months ago.
3. Ops centre diffs the live scheduler; curate the 59 missing jobs over a week.
4. Journal sources — momentum ₹3L first, then JSON paper books, then N500M / I75WR / Pairs.
5. Shared page furniture, applied to the five paper books first (they collapse anyway).
6. Only then the register's second pass (derive mode/size/activity).

## Not checked

Runtime behaviour in a browser, mobile/narrow viewports, the internals of the seven Jinja
dashboards, backend correctness or test coverage, auth/session and secret hygiene, DB schema and
migration safety, and whether the numbers each page displays are *right* — this review is the shell,
not the arithmetic. Strategy quality and capital allocation deliberately excluded (that belongs in
the weekly re-assessment).

Rendered version: published artifact "Quantifyd App Assessment" (2026-08-17).
