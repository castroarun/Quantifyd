# research/90 — Results: NIFTY Rules-Based Short Strangle (2019→2026)

---

# G3 (2026-07-24): Arun's Monday ₹20-Premium Spec — VALIDATED, best t in program

Full detail: `MONDAY_20RS_STRANGLE_WEEKLY_SWEEP_STATUS.md` §6. One-liner: Monday
entry + next-week expiry + premium-targeted strikes beats the G2 day-after-expiry
sleeve (t 4.3–5.5); **T30+PT50+stop2.0+roll = 11.06 pts/wk, t 5.47, positive all
8 years**; his T20 variant 7.86 pts/wk t 4.79 (2026 flat). Stop tightness is
DTE-dependent: 2.0× for ~9-DTE Monday entries (1.5× was right only for ~5-DTE).
Residual gap-week tail ~−445 pts survives all stops → event-skip rule proposed.
**NSR-W v1.0 locked** (spec in STATUS §6) → G5 paper book.

---

# G2 (2026-07-24, same day): Pessimistic Fills, Post-Stop Actions, Indicator Exits, Condor

**Verdict: G2 PASS — upgraded to STRATEGY-CANDIDATE.** 58,028 cycle-configs, 22s.
Full log: `results/run_g2.log`; per-year: `results/g2_yearly.csv`.

## 1. Fill realism (the make-or-break): SURVIVED at 2.0–2.5×, KILLED 1.5× monthly

Gap-aware fills (gap-open → both legs at open; intraday touch → stop level):

| M p2.5% config | net | t | worst | p5 | post-22 |
|---|---|---|---|---|---|
| stop 2.5× + PT50 | **47.8** | **2.61** | −301 | −159 | **+33.8** |
| stop 2.0× + PT50 | 33.2 | 2.17 | **−161** | −119 | +15.9 |
| stop 1.5× (any PT) | 17–19 | 1.4 | −184 | −98 | **−7 to −9 (NEGATIVE)** |

Realistic fills actually IMPROVED the tail vs G1 (gap exits book the open, which is
earlier than G1's close eval) but shaved the means ~15-20%. **1.5× is too tight
monthly — whipsawed to death under real fills.** Falsification threshold not hit.

## 2. Post-stop action (Arun's question #1): answer differs by horizon

- **MONTHLY: flat both legs.** Rolling away lifts mean (43–46 vs 33) but re-fattens
  the tail (worst −161 → −670) — exactly what the stop exists to prevent.
- **WEEKLY: ROLL AWAY once — the best family in the study.** stop 1.5× + roll-away:
  net 14.0, **t 4.73**, p5 −39 (vs −180 unstopped), worst halved (−633 vs −1,001),
  post-22 +17.9. Per-year: positive 7/8 years, **2020 ≈ flat (−0.6)**, every
  post-22 year positive incl. 2026. Whole family consistent (PT variant t 4.71;
  2.0/2.5× rolls t 2.3–2.8) — not a lucky cell. Mechanism: tight stop cuts the
  threatened leg early + re-selling further OTM re-arms theta without doubling risk.
  Second stop in a cycle → flat everything (as simulated).

## 3. Indicator exits (Arun's question #2): premium stop confirmed, one honest exception

- ATR80/90, ADX20/25, VIX-day-jump: ALL worse (t 1.1–1.6, tails −450 to −660) —
  they act a day late by construction, and the data shows it.
- **Exception:** VIX ≥ 1.25× entry-VIX on monthly: mean 63.4 (t 2.17, post-22 +71)
  — HIGHER mean than any premium stop, but tail 2.7× fatter (worst −817, p5 −413).
  It's a mean-maximizer, not a loss-minimizer. Given the stated objective
  (consistency, bounded loss), the premium stop stands. Recorded for a possible
  G3 combo test (stop OR vix-exit, whichever first).

## 4. Iron condor: monthly UNTESTABLE at EOD (not "dead")

M condor shows worst −1,762 — structurally impossible (wing cap ≈ −320), proving
**stale far-OTM wing marks** (the r/89 sin, here biasing AGAINST the condor: wings
don't appreciate in crashes in stale data). Do not conclude; retest with intraday
chain data. Weekly condor is internally consistent (worst −172 ≈ cap) but thin
(mean 2.6 pts) — real, capped, marginal.

## 5. The emerging system (NSR v0.9 — for the G5 paper book)

- **Monthly sleeve:** entry day-after-monthly-expiry, ~2.5% OTM, per-leg stop
  **2.5×** (GTT at entry), PT 50% of credit, time-exit DTE≤2, **flat after any
  stop — no rolls**, prefer entries at VIX≥16.
- **Weekly sleeve:** entry day-after-expiry, ~1.2% OTM, per-leg stop **1.5×**,
  **one roll-away allowed** (same %OTM from current spot), second stop → flat,
  time-exit DTE≤1.
- Per-year caveat (honest): monthly sleeve was negative in 2024 (−13) and the
  five 2026 cycles (−48); weekly roll sleeve was positive both years. The weekly
  sleeve carries the recent regime.

## 6. r/89 reconciliation (why this isn't a contradiction)

r/89 tested ATM straddles (pure vol bet) held/managed — post-22 EV ≈ 0. This
study sells 1.2–2.5% OTM strangles (range + skew bet) with mechanical stops: the
surviving premium sits in the OTM wings, not ATM vol, and the stop family's post-22
means (+10–34 monthly, +18 weekly-roll) are modest — consistent with "mostly
decayed, residual VRP in the wings, harvestable only with strict loss control."
Same data table, same liquidity filter, compatible conclusions.

## Next gates

- G3 residual: multiple-testing haircut is partially addressed (family
  monotonicity + per-year); remaining: replay 2026-W30 through the rules on the
  chain recorder, and re-test the CPR/VIX entry gates with regime controls.
- **G5: build the paper book** (NSR monthly + weekly-roll sleeves, 10 lots) next
  to the straddle V1/V2 books; weekly mentor review compares human vs robot.

---


**Verdict: SIGNAL — G1 PASS, proceed to G2.** A mechanical monthly short strangle
with a per-leg premium stop shows net-of-cost expectancy with t ≈ 2.0–2.4, a 6×
smaller catastrophic tail than hold-to-expiry, and monotonic behavior across the
stop family (not a single lucky cell). NOT yet a strategy: close-based fills,
multiple-testing risk, and a conflict with research/89's post-2022 no-edge finding
must survive G2 before any paper book.

Run: 2026-07-24, VPS. 477 cycles (387 weekly + 90 monthly), 44,928 cycle-configs,
2.75M option rows, liquidity-filtered (contracts ≥ 50). Net of ~0.5% premium
friction; ×2-cost sensitivity survives (net_2x within ~10% of net).

## Headline findings

### 1. MONTHLY arm: the premium stop is the whole story (p=2.5% OTM shown)

| Config (no PT, no giveback) | net/cycle | t | post-22 | worst | p5 | win% |
|---|---|---|---|---|---|---|
| No stop (hold to DTE-2/expiry) | 51.3 | 1.38 | 82.9 | **−1,878** | −554 | 73% |
| Stop 2.5× | 57.7 | **2.41** | 45.3 | −665 | −279 | 54% |
| Stop 2.0× | 41.8 | 2.08 | 19.4 | **−298** | −175 | 40% |
| Stop 1.5× | 33.3 | 1.98 | 20.4 | **−298** | −133 | 44% |

- **Stops keep most of the mean and DOUBLE the t-stat** (variance collapses).
  Worst cycle improves 6× (−1,878 → −298 pts; at 10 lots: −₹12.2L → −₹1.9L).
  This is exactly the stated objective (consistency, bounded loss) — the
  "insurance premium" costs ~40–60 pts/cycle in the calm post-22 regime.
- **Giveback rule HURTS monthly everywhere** (e.g., no-stop 51→21; stop2.5
  58→38): it exits winners too early. Kill it for monthly.
- Profit-take 50–60% is roughly neutral on mean, shortens holds 27→13-19 days
  (frees margin — relevant at Arun's 97% utilization).

### 2. WEEKLY arm: real mean, UNFIXABLE tail at EOD granularity

- Hold-to-expiry/PT: net ~14–15 pts/cycle, **t 2.4–2.5** (n=379), post-22 ~21.
- **But worst = −1,001 pts in EVERY config including stop 1.5×** — overnight
  gaps blow through any close-evaluated stop on a 5-day trade. And stops crush
  the small weekly premium (mean 15 → 1–4). Conclusion: weeklies are only
  viable with intraday stops (NAS-style ±move-stop) or defined-risk wings
  (iron condor — r/60 prior). G2 must test the wings arm.

### 3. Gates (recorded per cycle, evaluated post-hoc — unbiased)

- **VIX**: monthly BETTER at VIX≥16 (46 vs 10 pts) — sell vol when vol is paid.
  Weekly better at VIX<16 (9 vs −4). Opposite signs by horizon; plausible
  (weekly = gap risk dominates; monthly = premium richness dominates) but
  n small — G2 confirms before gating.
- **Weekly CPR width (r/67)**: NARROW-CPR weeks were the BEST for short premium
  (monthly terc1 55 vs terc2 −3; weekly monotonic 7.9/1.8/−0.1) — **opposite
  sign to the r/67 trend-week hypothesis**. Likely regime confound (narrow CPR
  clusters in calm markets). Do NOT gate on CPR yet; re-test in G2 with
  regime controls.

### 4. Honest caveats (why this is SIGNAL, not STRATEGY)

1. **Close-based fills**: stops book the day's close — real gap-throughs fill
   worse. G2 must re-run with pessimistic fills (stop level vs next-day open,
   high-based touch).
2. **Conflict with r/89** (index short-vol ≈0 post-2022): different structure
   (strangle vs straddle) and mechanics, but the conflict must be reconciled,
   not waved away. The post-22 stop-family means (+19–45) are the number to
   attack.
3. **Multiple testing**: 192 configs; defense = family monotonicity (stop axis
   behaves smoothly), not peak-picking. G2 adds per-year tables for the chosen
   family and a 2020-crash stress isolation.
4. 2026 data ends 07-21 — this week's whipsaw Friday not included.
5. No intraday PANIC flatten at EOD granularity (G2 on chain recorder).

## G2 plan (next)

1. Pessimistic-fill re-run (gap-aware stop fills) — the make-or-break test.
2. Iron-condor arm (±wings) esp. weekly; margin ~60-70% lower — directly fixes
   the 97%-utilization problem.
3. Per-year stability tables for monthly stop-2.0/2.5 family; 2020 isolation.
4. Intraday validation on options_data.db chain recorder (Apr–Jul 2026),
   including replaying W30 itself against the rules.
5. Reconciliation memo vs research/89 methodology.

## Immediate practical takeaway for the manual book (usable Monday)

Even before G2: the data supports the W30 mentor rules — enter wide, place a
**2× per-leg premium-stop GTT at entry**, take profit ~50-60%, **no giveback
rule**, **no rolls toward spot**, and prefer monthly over weekly for naked
strangles (weekly tail is gap-dominated). This is decision-support, not a
directive.
