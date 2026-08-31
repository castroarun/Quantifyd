# Universal Swing Rule — One Rule, Every Name, 2–10 Day Holds

STATUS: **CONCLUDED — NO INVESTABLE EDGE**
Opened 2026-08-31 13:45 IST · research/136

---

## 1. The Ask

**What you asked:** "Consider dropping per-stock fitting entirely for one universal
rule across all names — yes i prefer this." Then: *"all i want is a tradeable system
(intraday to positional, roughly 2–10 days), not worried ab WR, just want a better
system where we can put money that will give decent returns."*

**What we are testing:** Does a **single parameter set, applied identically to every
name in a wide Indian equity universe**, produce a net-of-cost edge on 2–10 day
holds — large enough to carry real money?

Explicitly NOT: per-stock fitting (that is what N500M does, and its 30 bespoke rules
produced t=1.32 gross on 32 trades — see `N500M_EDGE_ASSESSMENT_2026-08-28.md`).
Explicitly NOT: intraday (research/109 + /110 closed that line — 58 constructions,
none cleared the ~10 bps cost floor).

**Falsification criterion, set before running:** if the signal's forward return does
not beat a **date-matched random entry** from the same eligible universe by a margin
with t ≳ 3 gross, the idea dies at G1. Beating zero is not the test — beating the
drift of a rising market is.

## 2. The Base — G0 hypothesis

**Mechanism:** short-horizon momentum. A stock making a fresh multi-week high has
resolved an information event that the market prices in slowly.
**Who loses:** holders who sell winners early (disposition effect), and mean-reversion
traders fading strength. **Decay risk:** high — this is the most-published anomaly in
existence, and Indian retail flow has grown enormously since 2015.

## 3. G1 probe design

| | |
|---|---|
| Data | `market_data_unified`, `day` timeframe, 3.6M rows, 2015-01-01 → 2026-08-29 |
| Universe | every symbol with ≥250 prior daily bars **and** trailing-20d median turnover ≥ ₹5 crore, computed causally on the signal date |
| Signal | close = highest close of the trailing N days (Donchian breakout), N ∈ {20, 55} |
| Entry | **next day's open** — never the signal bar's close |
| Horizons | forward return at 2, 5, 10 trading days |
| Costs | swept: 0 / 20 / 30 / 50 bps round trip (delivery brokerage + STT + slippage) |

**Controls — all three are mandatory and none is optional:**

1. **Date-matched random entry** — same day, same eligible universe, same horizon.
   Answers "is the signal better than picking a liquid stock at random that day?"
2. **Unconditional drift** — the eligible universe's mean forward return.
   Answers "is this just a rising market?"
3. **Per-year breakdown** — answers "is it one regime?"

Without control (1) a rising market makes any long signal look profitable. This is
binding here after research/87 + /88, where a raw t=10 dissolved into drift and
survivorship once controls were added.

**Known limitation, stated up front:** the symbol list was assembled from today's
Nifty-500 and is therefore survivorship-tainted. Delisted names are absent. The
date-matched control absorbs much of this, because the control draws from the same
tainted pool — the *difference* is far more trustworthy than the absolute level.
Absolute returns from this probe are upper bounds, not forecasts.

## 4. Plan

| Stage | Test | Kill condition |
|---|---|---|
| **G1** (this) | signal vs random vs drift, 2/5/10d, gross + net | excess t < 3, or not monotonic in horizon |
| G2 | tradeable rule: trailing stop (research/71 says never a target), position cap, regime gate | no net edge at 30 bps |
| G3 | walk-forward, per-year, parameter surface must be monotonic not peaked | any single year carries it |
| G4 | portfolio: equal-notional sizing (research/83 — N-sizing lost 3 times), MaxDD, capacity, correlation to existing books | Calmar below the momentum book already live |
| G5 | paper book on the VPS | — |

## 5. Status log

| Time (IST) | Event |
|---|---|
| 2026-08-31 13:45 | Folder + this doc written. Probe not yet launched. |
