# VIX Shocks and What Follows — Does a Volatility Jump or Crush Change What We Should Do?

STATUS: **QUEUED** — launches after 15:20 IST 2026-08-20 (read-only, but held until the
live book is flat so nothing competes for the box during market hours)

## 2. The Ask

**What Arun asked (2026-08-20):** "if VIX drops by over 5% or hikes by 5% (5% is a sample
only, not to be biased), what are the following price action moves — any patterns we
should work into our systems?"

**What we are actually testing.** We run **short-volatility** books, so a VIX shock is not
an abstract market-structure question — it is directly about our exposure:

> After a VIX shock of magnitude X, what happens to (a) the underlying's subsequent move,
> (b) realised volatility, and critically (c) **the P&L of the exact straddle we sell**?
> And does conditioning our entry, size, stop or exit on that shock beat ignoring it?

**Explicitly not biased to 5%:** the threshold is a swept parameter, not an assumption.
We test a ladder and look for **monotonicity** — a real effect should strengthen smoothly
with shock size, not appear only at one hand-picked cut-off.

## 3. The Base

- **VIX source:** `backtest_data/market_data.db :: market_data_unified` (INDIA VIX; the
  agent must confirm the exact symbol string, timeframes and date range in G0 — token
  264969 per our notes). Daily for the long history; intraday if available.
- **Underlying + our construction:** NIFTY (and SENSEX where data allows). For the
  straddle-P&L outcome use `options_data.db :: option_chain` (1-min, ~85 days,
  2026-04-20 onward) replaying the live construction: 09:16 ATM straddle to 15:15/15:20.
- **Two shock definitions, tested separately** (they are different animals):
  1. **Overnight/gap shock** — VIX close-to-open or close-to-close change. Known BEFORE
     our 09:16 entry, so it can gate entry. This is the actionable one.
  2. **Intraday shock** — VIX change within the session. Arrives while we are already
     short, so it can only drive adjustment or exit.
- **Threshold ladder:** ±2, 3, 4, 5, 7, 10% (and absolute-point equivalents, since a 5%
  move from VIX 11 is not the same event as from VIX 22 — report both normalisations).
- **Horizons:** same-day, next-day, next-3-day, next-5-day.

## 4. Plan — gates

| Gate | Question | Pass criterion |
|---|---|---|
| G0 | Is the VIX series usable? | symbol, coverage, gaps, and whether intraday VIX exists; document the answer |
| G1 | Does a shock predict anything? | conditional mean/median of forward underlying return, forward realised vol, and forward straddle P&L vs the **unconditional baseline** on the same days; effect must be monotonic in threshold |
| G2 | Is it actionable for our books? | does gating entry / sizing / exiting on the shock beat not doing so, net of costs, over the recorded-chain window? |
| G3 | Robust? | sign-stable across years (VIX daily history is long — use it), per-DTE, per-venue; random and date-matched controls; not driven by a handful of crisis days |

**Directional prior to be tested, not assumed:** a VIX *spike* usually means premium has
already expanded — selling into it can be the best or the worst thing depending on whether
realised vol follows. A VIX *crush* means we collect less premium for the same risk. Both
directions must be reported; the interesting result may well be "the crush days are the
ones to avoid", which is the opposite of intuition.

**Sins to guard explicitly:** regime dependence (VIX shocks cluster — a few 2020/2024-style
episodes can carry the whole result; report with and without the top decile of days),
multiple testing (threshold × horizon × direction is a big grid — pre-register and
haircut), look-ahead (gap shock must use only pre-09:16 information), and survivorship
(n/a for indices, state so).

## 5. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-20 ~13:5x IST | Question raised, brief written, queued | launches after 15:20 |

## 6. Crash Recovery

Read-only on both DBs; no live state. Scripts in `scripts/`, outputs in `results/`.
`market_data.db` is 30 GB — query with explicit symbol/timeframe filters, never a bare
LIKE scan (that is what made the first recon query time out).

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `VIX_SHOCK_PRICE_ACTION_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/*.py` | shock construction, conditional stats, rule test | yes |
| `results/*.csv` | conditional tables, per-rule detail | yes |
| `results/RESULTS.md` | verdict + what (if anything) to wire into the systems | yes |

## 8. Findings

(pending)
