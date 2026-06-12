# Calm-Day Entry for NIFTH Iron-Fly Selling — Compression-Gate Hunt

**STATUS: P3+P4 DONE — recommendation ready; live wiring pending your go.**
short neutral/iron fly survives its 2% stop), then push toward the net-₹ sweet spot and wire it live.

## 1. The Ask
**You asked:** backtest neutral/iron-fly entry timing on NIFTY price action in our DB; find the calm-days
pattern; assess *all* indicators/combos comprehensively (CPR multi-TF, day/week breaks, Bollinger, HTF,
MAs, Ichimoku, bands, trend-then-halt, inside candles D/W, inverses, RSI, Stoch, ADX, ATR…); go deepest.
**What we're testing:** which causal entry-time conditions predict the **next-H-day window stays within
the 2% move-stop** (= the fly wins), and the best *combination/composite* of them — net of the win-rate
≠ P&L caveat.

## 2. The Base (locked)
- **Instrument proxy:** short ATM straddle + 2% wings; the dominant loss driver is the **2% underlying
  move-stop**. Outcome `calm_H` = stop NOT hit within next H trading days (entry_spot = prior close).
- **No look-ahead:** every feature uses data ≤ prior close (shifted 1). Entry next morning.
- **Universe/period:** NIFTH daily + India VIX from Kite, 2015-01-01 → 2026-06-12 (2,828 entry days).
- **Success criterion:** calm-rate lift over the 59.4% base (H5), with **3-era + walk-forward stability**
  and **tradeable coverage** (a gate that skips everything is useless). calm-rate = win-rate proxy.
- **Caveat (binding):** calm-rate ≠ net ₹. Low-vol = calmer but thinner premium → true optimum is a
  tradeoff needing option premiums (AlgoTest / forward recorder), handled in P3.

## 3. Plan
- **P1 (DONE):** univariate screen, ~24 features. → vol/range COMPRESSION wins; trend/MA/Ichimoku/ADX/
  inside-candles ≈ noise.
- **P2 (RUNNING):** deepest combination work — correlation/redundancy, threshold→coverage curves,
  conditional lift, best 2-/3-way AND gates (walk-forward), a single composite "compression score",
  multivariate logistic (train/test AUC + coefs), EV proxy.
- **P3:** premium-aware net-edge framing (EV sensitivity to credit/stop; flag what needs AlgoTest).
- **P4:** recommend wiring the winning compression gate into live V2 entry (replace weak inside-week leg),
  forward-validate via shadow log. (Proposal only — live change is your call.)

## 4. Status (live log)
| Time | Event |
|---|---|
| P1 | univariate done; committed research/64 (commit) |
| P2 | running combination/composite/multivariate screen |

## 5. Crash Recovery (resume without me)
- All self-contained in `research/64_calm_day_ironfly_entry/`. Re-run on VPS:
  `cd /home/arun/quantifyd && venv/bin/python research/64_calm_day_ironfly_entry/scripts/calm_p2.py`
  (pulls fresh NIFTH+VIX daily from Kite; deterministic; ~1-2 min). P1 = `scripts/calm_study.py`.
- Results land in `research/64_calm_day_ironfly_entry/results/`; findings in `RESULTS.md`.
- Needs: Kite token at `backtest_data/access_token.json`; venv has pandas/numpy (sklearn optional).
- Nothing live is touched — pure research, read-only.

## 6. Files
| File | Purpose | Commit? |
|---|---|---|
| `scripts/calm_study.py` | P1 univariate screen | yes |
| `scripts/calm_p2.py` | P2 combinations/composite/multivariate | yes |
| `results/*.csv` | ranked outputs | yes (small) |
| `RESULTS.md` | findings (all phases) | yes |
| `CALM_DAY_IRONFLY_ENTRY_DAILY_SWEEP_STATUS.md` | this crash-recovery doc | yes |

## 7. Findings so far
- **P1:** base calm H5 = 59.4%; best single-feature quintile ~80% (VIX/ATR). Survivors = VIX, ATR%,
  realized-vol, Donchian/BB/CPR/5d-range width, gap. Dead = ADX, Ichimoku, MA dist/compress, RSI, weekly
  CPR, inside-day/week (inside-week barely beats base — challenges the live filter). Friday slightly calmer.
- **P2:** (filling in this run — see RESULTS.md)
