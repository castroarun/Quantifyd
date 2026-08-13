# SENSEX DTE Short-Straddle — RESULTS

**Verdict: SIGNAL (robust, all-years-positive) — but the edge is an EXPIRY-EVE (DTE-1)
phenomenon, NOT a DTE-3 one. "Filling the off-days" by holding longer dilutes it.**

Data: real BSE UDiFF bhavcopy, `bse_options_bhav`, SENSEX weeklies 2024-01-01 → 2026-08-04
(638 trade days, ~134 weeklies). Entry = ATM straddle at option OPEN on DTE-N; exit = DTE-1
CLOSE. Net 0.3% slippage/leg + ₹160/RT. OI≥100 ATM. India VIX (refreshed to 08-05) 13-28 gate
tested. 10-lot book (SENSEX lot 20 → QTY 200).

## The surprise: the edge peaks at DTE-1 for BOTH indices

Same window (2024-01 → mid-2026), same method, naked:

| DTE | SENSEX M/SD | SENSEX win | SENSEX worst | NIFTY M/SD | NIFTY win | NIFTY worst |
|----:|:-----------:|:----------:|:------------:|:----------:|:---------:|:-----------:|
| **1** | **1.45** | 96% | −33k | **1.27** | 96% | −18k |
| 2 | 0.42 | 81% | −274k | 0.94 | 85% | −158k |
| 3 | 0.45 | 78% | −282k | 0.77 | 82% | −201k |
| 4 | 0.31 | 74% | −465k | 0.75 | 81% | −128k |
| 5 | 0.25 | 68% | −465k | 0.68 | 80% | −244k |

- **DTE-1 (the expiry-eve, same-day open→close) is dramatically the best for both** — 96% win,
  M/SD 1.27–1.45, and a *tiny* drawdown (−18k NIFTY / −33k SENSEX on 10 lots).
- **NIFTY degrades gracefully with DTE; SENSEX collapses.** By DTE-3 NIFTY still holds M/SD 0.77,
  but SENSEX is down to 0.45 with an 8× bigger tail (−282k vs −33k). Every extra held day on
  SENSEX adds risk far faster than return — SENSEX is the more gap-prone book.

## SENSEX DTE-3 (the requested variant) — solid but second-best

Naked: n=134, **+₹47.95 L**, mean +35,785, win 78%, M/SD 0.45, worst −281,744, maxDD −292,220.
Fly (3% next-wk): +₹40.37 L, M/SD 0.42, worst −258,892 — wings cost ~16% of P&L to trim the tail
~8%, same trade-off NIFTY showed.

Per-year DTE-3 (all positive, growing mean):
| Year | n | Total | Mean | Win% | M/SD |
|---|---|---|---|---|---|
| 2024 | 51 | +₹13.85 L | +27,157 | 78% | 0.64 |
| 2025 | 53 | +₹19.62 L | +37,025 | 79% | 0.42 |
| 2026* | 30 | +₹14.48 L | +48,263 | 77% | 0.44 |

VIX 13-28 gate barely moves DTE-3 (M/SD 0.45→0.43) — 2024-26 is a calm regime, so the gate just
cuts sample. Unlike NIFTY's 7-yr study where the gate dodged the 2020 crash.

## Implication for the combined book

The user's premise was "fill the days NIFTY isn't trading." The data says **don't** — the edge is
expiry-eve, and stretching to DTE-3 to keep capital busy trades away most of the Sharpe and adds
fat tails (worst on SENSEX). Better structures:

- **Best risk-adjusted (recommended):** two crisp **DTE-1 same-day** trades —
  **NIFTY on Monday** (Tue expiry) + **SENSEX on Wednesday** (Thu expiry). 96% win each, tiny
  drawdowns, minimal overnight/crash exposure (1 session each). Capital idle Tue/Thu/Fri — and that
  is *correct*, because there's no edge those days.
- **Max utilization (only if you insist):** DTE-3 both — NIFTY Fri→Mon + SENSEX Mon→Wed. More ₹
  deployed, but SENSEX at M/SD 0.45 with a −282k tail is a much worse trade than its own DTE-1.

## Caveats (honest)

- **2024-26 is a calm, low-VIX window with NO 2020-style crash.** The naked tails (esp. SENSEX
  DTE-3/4/5) are understated for a crisis; the fly's insurance value is bigger out-of-sample than
  this sample shows. DTE-1's 1-session exposure is the natural crash mitigant.
- **DTE-1 is a same-day open→close** on 1-DTE ATM options — execution/slippage sensitive; the
  0.3%/leg assumption may be optimistic on the thinnest expiry-eve strikes. Still, 96% win over
  ~130 weeks is a real, repeatable theta pattern.
- **Not the 7-year NIFTY comparison** — that study was VIX-gated across regimes and found DTE-3
  best; on a like-for-like calm window DTE-1 wins for both. The earlier "DTE-3 optimal for NIFTY"
  was regime-dependent, not universal.
- 2023 SENSEX weeklies (BSE relaunch) need the legacy pre-UDiFF format — not included; would add
  ~½ year if wanted.

## Files
- `scripts/bse_bhav_downloader.py` — BSE UDiFF → `bse_options_bhav` (638 days, 370,746 rows)
- `scripts/sensex_study.py` — SENSEX DTE sweep
- `scripts/nifty_sameperiod.py` — NIFTY 2024-26 head-to-head
- `scripts/refresh_vix.py` — INDIAVIX refresh (now through 2026-08-05)
