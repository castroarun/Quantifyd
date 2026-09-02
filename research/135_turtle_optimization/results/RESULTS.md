# research/135 — Turtle Optimization on Indian Equities: RESULTS

**VERDICT: CONCLUDED — NO DEPLOY.**

**(1) The attached Turtle spec, taken literally, is the WORST book we have
tested on Indian equities: 1.7% CAGR, −67.9% MaxDD, Calmar 0.02 (2005-2026,
net). It loses to NIFTYBEES buy-and-hold by ~11 points of CAGR while taking
more drawdown.**

**(2) The optimization is REAL in-sample and it comes from DELETING rules, not
adding them. Dropping Rule 3 (the 2N stop), Rule 4 (pyramiding) and Rule 2
(N-sizing) — keeping only Rule 1/5 at the original 20/10 channel — lifts
Calmar 0.02 → 0.53 and CAGR 1.7% → 15.9% over the full period.**

**(3) It still does not earn a book. Out-of-sample 2024-2026 (held out until
one single consumption) is NEGATIVE for every Turtle arm — TT_OPT −8.3% CAGR
against a benchmark of +5.3%. Era means decay monotonically: +27.4% (2005-17)
→ +14.8% (2018-23) → −5.4% (2024-26). This is r/83's "decay shadow" confirmed
on genuinely unseen data.**

**(4) CORRECTED 2026-08-30 (Stage G): the momentum book beats every Turtle
variant in EVERY era, not just the recent ones — 31.78% CAGR vs 15.97% at the
same drawdown (−31.7% vs −31.8%), 299× vs 21×. The first version of this
document put momentum at 12.58%, which Arun challenged against research/75's
published 31.9%. That arm was wrong; see §5b. The correction makes the verdict
stronger, not weaker.**

---

## 1. The attached rules, scored one by one

Each rule tested on the same 78 F&O names, 2005-2017 IS, net of futures-proxy
costs, gated on NIFTYBEES>200DMA, equal-notional, 12-position cap.

| Attached rule | Verdict | Evidence |
|---|---|---|
| **Rule 1 — 20/55-day breakout entry** | **KEEP (20)** | 20/10 is the best cell of 11 channels tested. The Turtles' own System-1 parameter is genuinely the right one. |
| **Rule 2 — N-based (1%) position sizing** | **REJECT** | Equal-notional Calmar 1.06 vs N-sized 0.83 (0.5% risk), 0.84 (1%), 0.70 (2%). **Fourth** independent failure of vol-keyed sizing in this engagement. |
| **Rule 3 — 2N hard stop** | **REJECT — this is the single most damaging rule** | Mean Calmar across all 11 channels: no-stop **0.65** > 3N 0.51 > 2N 0.46 > 1.5N 0.45. Monotone, and 10 of 11 channels individually prefer no stop. Removing the stop improves CAGR (20.5% vs 15.8%) **and** drawdown (−32.1% vs −34.7%). |
| **Rule 4 — pyramiding to 4 units every ½N** | **REJECT** | With the 2N stop it is catastrophic: mean Calmar 0.47 → 0.40 → 0.24 → 0.08 as units go 1→4. Adding units ratchets the shared stop up under the last unit, so an ordinary pullback stops out the *enlarged* position. |
| **Rule 5 — 10/20-day opposite-channel exit** | **KEEP (10)** | The trailing channel is the exit that works; the whole edge is in letting it run. Tight (10) beats loose (20/40) at every entry length. |

### The Rule 3 × Rule 4 interaction is the core defect

The attached system's stop and its pyramid are individually questionable and
jointly ruinous, because they are *coupled*: every add moves all stops up to
2N below the newest unit. The position gets biggest exactly when its stop is
tightest. That is why TT_ATTACHED spends 4,226 days — over 11 years — below
its prior peak.

## 2. Where the optimization landed

`TT_OPT` = **20-day breakout entry, 10-day trailing exit, NO hard stop, no
pyramiding, equal-notional 10% per position, 12 positions max, entries gated
on NIFTYBEES > 200DMA.** Long only.

| Book (2005-2026, net) | CAGR | Sharpe | MaxDD | Calmar | Days in DD |
|---|---|---|---|---|---|
| TT_ATTACHED (spec as written) | 1.67% | 0.19 | −67.9% | 0.02 | 4,226 |
| TT_R83 (r/83 incumbent) | 10.52% | 0.63 | −44.7% | 0.24 | 2,282 |
| **TT_OPT (r/135 finalist)** | **15.85%** | **0.98** | **−30.1%** | **0.53** | **974** |
| TT_OPT_PYR (speculative) | 19.05% | 0.92 | −41.6% | 0.46 | 1,226 |
| ~~MOM_RECON~~ **SUPERSEDED — see §5b** | ~~12.58%~~ | ~~0.93~~ | ~~−30.5%~~ | ~~0.41~~ | ~~1,169~~ |
| NIFTYBEES buy & hold | 12.47% | 0.70 | −59.7% | 0.21 | 1,087 |

## 3. The pre-registered plateau test did its job

Stage B found pyramiding lifted IS Calmar 0.84 → **1.06** on the 20/10 channel
— the single best number in the whole study. The pre-declared rule was that a
winner must have top-quartile *neighbours*, not just a top score. It failed:

| Channel (no stop) | 1 unit | 2 units | 4 units |
|---|---|---|---|
| **20/10** | 0.84 | 0.94 | **1.06** |
| 40/10 | 0.71 | 0.66 | 0.50 |
| 20/20 | 0.54 | 0.52 | 0.49 |
| 55/10 | 0.70 | 0.64 | 0.52 |
| 55/20 | 0.66 | 0.59 | 0.54 |
| 80/20 | 0.71 | 0.61 | 0.55 |

Pyramiding helped in **1 of 6** channels and degraded the other 5. Flagged
OVERFIT before the holdout was touched — and the holdout agreed: TT_OPT_PYR
went IS Calmar 1.06 → VAL **0.08** (worst Turtle arm in validation). Had we
selected on the best IS number we would have shipped the worst validation book.

## 4. Out-of-sample — the result that decides it

OOS 2024-01-01 .. 2026-08-28, consumed exactly once.

| Arm | OOS CAGR | OOS MaxDD | OOS Calmar |
|---|---|---|---|
| TT_ATTACHED | −8.88% | −26.6% | −0.34 |
| TT_R83 | −20.21% | −46.5% | −0.43 |
| TT_OPT | −8.32% | −25.5% | −0.33 |
| TT_OPT_PYR | −2.08% | −21.3% | −0.10 |
| ~~MOM_RECON~~ **SUPERSEDED — see §5b** | ~~−2.20%~~ | ~~−13.8%~~ | ~~−0.16~~ |
| **BENCH** | **+5.25%** | −15.2% | +0.34 |

Per-year era means:

| Book | 2005-2017 | 2018-2023 | 2024-2026 |
|---|---|---|---|
| TT_OPT | +27.4% | +14.8% | **−5.4%** |
| TT_R83 | +21.5% | +8.9% | −8.3% |
| ~~MOM_RECON~~ **SUPERSEDED — see §5b** | ~~+16.9%~~ | ~~+16.2%~~ | ~~+3.1%~~ |
| BENCH | +18.5% | +14.4% | +5.1% |

The Turtle family decays monotonically across eras. The momentum rules do not —
and by far more than these superseded MOM_RECON figures suggest (§5b).

## 5b. CORRECTION — the momentum arm (Stage G)

Stage F's momentum arm was a hand-rolled reconstruction and it was wrong three
ways. Arun caught it: the published research/75 book does 31.9% CAGR, and the
chart showed it at benchmark.

| Defect | Effect |
|---|---|
| **Wrong universe** — ranked momentum inside the 78 F&O large caps; the book's real universe is a point-in-time top-250 by traded value | the big one: cross-sectional momentum lives in the mid-cap tail |
| **Wrong rules** — conflated research/75's monthly-rebalance backtest with the LIVE paper book's extra machinery, bolting a daily 15-day Donchian stop onto it | added churn the real book does not have |
| **Idle-cash bug** — names stopped out mid-month, and everything sold at a risk-off gate, could not be re-bought until the next month-end | book sat in cash through recoveries |

Stage G drives **research/75's own runner** instead of re-implementing it, and
reproduces its published result to within 0.1pt — 31.78% CAGR, −31.66% DD,
Calmar 1.00, 298.6× against the published 31.9% / −31.6% / 1.01 / 292×. That
confirms the engine is being driven correctly.

### The two effects, separated (2006–2026, net)

| Book | Universe | CAGR | Sharpe | MaxDD | Calmar | Mult |
|---|---|---|---|---|---|---|
| **Momentum (r/75 A1)** | **top-250** | **31.78%** | 1.45 | −31.7% | **1.00** | 299× |
| Momentum (r/75 A1) | F&O-78 | 20.19% | 1.17 | −31.3% | 0.65 | 45× |
| Turtle-OPT | F&O-78 | 15.97% | 0.97 | −31.8% | 0.50 | 21× |
| Turtle-OPT | top-250 | 11.61% | 0.65 | −47.9% | 0.24 | 10× |
| NIFTYBEES B&H | — | 11.55% | 0.66 | −59.7% | 0.19 | 10× |

- **Universe effect: +11.6 points of CAGR** for momentum (20.19% → 31.78%) at
  the same drawdown. My Stage F reconstruction lost a further ~7.6 points below
  even the F&O-78 figure, which is the size of the reconstruction bug itself.
- **The Turtle gets no such gift — it gets WORSE on the broad universe**
  (Calmar 0.50 → 0.24, MaxDD −31.8% → −47.9%). Breakout-and-trail on thinner
  mid-caps buys whipsaws, not trends. This is a real finding in its own right:
  the two systems want opposite universes.

### Era means, corrected

| Book | 2006–2017 | 2018–2023 | 2024–2026 |
|---|---|---|---|
| Momentum · top-250 | +43.1% | +33.4% | **+21.0%** |
| Momentum · F&O-78 | +28.1% | +18.2% | +13.5% |
| Turtle-OPT · F&O-78 | +27.7% | +14.2% | **−3.3%** |
| Turtle-OPT · top-250 | +24.6% | +6.4% | −2.8% |
| NIFTYBEES B&H | +17.2% | +14.4% | +5.1% |

Momentum does **not** decay — it earns +21%/yr in the held-out window while
every Turtle variant loses money. The original conclusion understated the gap.

## 5. Momentum gate and put overlay (Stage E)

Both borrowed from `services/momentum_paper.py` and applied to the Turtle book.

- **Gate: no stable winner.** IS prefers NIFTYBEES>200DMA (Calmar 1.06) over
  100-SMA (0.72) over no gate (0.89). VAL reverses it: 100-SMA (0.205) > none
  (0.158) > 200DMA (0.084). The gate that looks best depends entirely on the
  era you fit it in — do not read the IS ranking as a finding.
- **Put overlay: does not rescue the book.** At the momentum spec (ATM, 2×
  equity notional, ~14 DTE, rolled) no hedge arm beats simply gating to cash
  in-sample (best hedge 0.93 vs plain gate 1.06). In VAL the hedge arms lead
  marginally (0.256 vs 0.205), but the margin is inside the noise of the gate
  choice itself.
- **One stable sub-finding: 5%-OTM beats ATM in every single pairing**, in
  both eras, at both spread assumptions — cheaper convexity buys more
  protection per rupee than at-the-money.
- This is **consistent with the live momentum book's own conclusion**, where
  `hedge_enabled=False` and the note records that "the cash-exit gate beat the
  hedge over the full cycle."

⚠️ Stage E was run on the U4-pyramid base before that base was disqualified,
so its *absolute* levels are inflated. The *rankings within* Stage E are still
informative; the levels should not be quoted.

## 6. Honest caveats

- **Survivorship.** The universe is today's F&O list back-projected to 2005.
  IS returns are inflated by construction; this is one more reason the IS→OOS
  gap should be read as decay plus bias, not decay alone.
- **OOS is short** — 2.7 years, 661 sessions, and 2026 is partial (to Aug 28).
  A negative OOS on that sample is suggestive, not proof. It is the *direction
  of the era means* that carries the weight.
- **Costs** are futures-proxy (3bp/side slippage + charges). Every headline is
  reported net; gross is in the CSVs.
- **Option modelling** uses INDIAVIX from 2015 and a VRP-scaled realised-vol
  proxy before it (calibrated k = 1.331 on the 2,884-day overlap). Pre-2015
  hedge numbers carry that assumption.
- **A bug was found and fixed mid-study**: the first put-overlay run expensed
  the full premium *and* marked the option's decay from that same level,
  double-counting ~1.8% of NAV per roll and producing absurd −99.9% results.
  Caught by a hand-check (11.8%/yr expected drag vs 84%/yr reported). The
  invalid output is retained as `_INVALID_stage_E_premium_doublecount.csv`.
- **Multiple testing:** ~100 book configurations were scored. Controls were the
  pre-declared plateau requirement, a single OOS consumption, and per-year
  reporting. TT_OPT sits in a broad plateau (the whole no-stop / tight-exit
  region is strong), which is why it is reported at all.

## 7. Recommendation

**Do not build a Turtle book.** Specifically:

1. The attached system as written should not be traded on Indian equities in
   any size — it is the worst-performing construction in this engagement.
2. The optimized version is a legitimate improvement and a clean demonstration
   that *removing* the stop and the pyramid is where the value is — but its
   OOS is negative and its era decay is monotone. It does not clear G5.
3. **The momentum book already owns this space and does it far better** —
   31.78% vs 15.97% CAGR at identical drawdown, 299× vs 21×, and it stays
   positive in the held-out era (+21.0%/yr) where the Turtle loses (−3.3%/yr).
   Nothing here justifies competing with it.
4. **Bankable, reusable lessons** — these transfer beyond Turtle:
   - hard stops on multi-week equity trend books *cost* return AND add
     drawdown (now consistent with r/71's exit bake-off);
   - pyramiding coupled to a ratcheting stop is actively destructive;
   - vol-keyed sizing lost to equal-notional for the **fourth** time — treat
     equal-notional as settled house default and stop re-testing it;
   - 5%-OTM index puts dominate ATM for portfolio hedging.

## 8. Reproducibility

VPS `94.136.185.54`, `/home/arun/quantifyd`, `venv/bin/python`, market_data.db
snapshot 2026-08-30. Universe 78 F&O names (CA-guarded, ≥300 bars).

| Script | Purpose |
|---|---|
| `scripts/turtle_core.py` | unit-level Turtle simulator (pyramiding) + book NAV |
| `scripts/run_turtle_opt.py` | Stages A (channel×stop), B (pyramid), C (book) |
| `scripts/run_stage_e.py` | Stage E — momentum gate + put overlay |
| `scripts/run_stage_f.py` | Stage F — head-to-head incl. momentum reconstruction |
| `scripts/run_full_report.py` | continuous 2005-2026 curves + per-year table |
| `scripts/run_stage_g.py` | **Stage G — momentum-arm correction**: drives r/75's own runner, universe bake-off |

Ledger: Stage A 44 cells · B 40 · C 9 · E 54 · F 18 · full-period 6 · G 5.
OOS consumed once (Stage F); Stage G re-scores already-decided arms on a
universe axis, adding no new parameter search.
