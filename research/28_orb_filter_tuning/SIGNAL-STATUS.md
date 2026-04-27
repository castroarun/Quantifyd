# ORB Filter Tuning — STATUS

## Goal + scope

Today (2026-04-27) live ORB execution surfaced two filter-related misses:
- **BEL** clean SHORT breakdown blocked all morning by `rsi_short_threshold=40` +
  `signal_age_max_mins=15`. RSI eventually dropped to 38.5 at 10:43 (would have
  qualified) but age window had already expired the original breakout candle.
- **RELIANCE** entered 8 min late because first eligible bar's 15-min RSI was
  50.64 (needs ≥60). Re-check on next bar passed at 60.04. Cost ~Rs 8/share entry slippage.

Sweep validates whether the four ORB filter knobs in `config.py:542-560` are
optimally set. RSI 60/40 was a baseline assumption from research/20 (not swept);
`signal_age=15` and `signal_drift=0.5%` were empirical fixes added 2026-04-24
after the VEDL/TRENT stale-entry incident (also not swept).

Universe: 15 ORB live stocks. Period: 2024-03-18 to 2026-03-12 (full 5-min data window).

## Plan

**Phase 1 — 21 cells:**
- `rsi_short_threshold` ∈ {40, 42, 45, 48} × `signal_age_max_mins` ∈ {15, 20, 30, 45} = 16 cells
- `rsi_long_threshold` ∈ {52, 55, 58, 60} (signal_age=15 fixed) = 4 cells
- Baseline = 60/40/15/0.005 = 1 cell

**Phase 2 — 4 cells:** `signal_drift_max_pct` ∈ {0.005, 0.0085, 0.012, 0.015} with Phase 1 winner baked in.

Pass criteria vs baseline (rsi_l=60, rsi_s=40, age=15, drift=0.005):
- PF ≥ baseline
- MaxDD ≤ baseline + 2%
- Trades > baseline

## Status

| Phase | Status | Output |
|---|---|---|
| Phase 1 sweep | ✅ done | `results/summary.csv` rows 2-22 |
| Phase 2 sweep | ✅ done | `results/summary.csv` rows 23-26 |
| Verdict | ✅ done | this doc |
| Live config change | pending user approval | `config.py:559` |

## Phase 1 results — top 5 by composite (PF × Sharpe + trade volume bonus)

| Rank | Cell | Trades | PF | Sharpe | DD% | Calmar | NetPnL | Δ vs baseline |
|---|---|---|---|---|---|---|---|---|
| Baseline | rsi_l=60, rsi_s=40, age=15 | 2918 | 1.597 | 3.94 | 34.57 | 4.75 | 12950 | — |
| **🥇 #1** | **rsi_s=40, age=30** | 3018 | **1.617** | **4.13** | **32.07** | **5.33** | 13634 | +100 trades · +0.020 PF · **−2.50% DD** |
| #2 | rsi_s=40, age=45 | 3067 | 1.616 | 4.13 | 32.13 | 5.35 | 13746 | +149 trades · +0.019 PF · −2.44% DD |
| #3 | rsi_s=42, age=45 | 3200 | 1.594 | 4.03 | 32.70 | 5.30 | 13818 | +282 trades · −0.003 PF (slight degradation) |
| #4 | rsi_s=42, age=30 | 3158 | 1.590 | 4.00 | 32.66 | 5.25 | 13634 | +240 trades · −0.007 PF |
| #5 | rsi_s=42, age=20 | 3104 | 1.589 | 3.98 | 32.59 | 5.22 | 13489 | +186 trades · −0.008 PF |

**Cells that FAILED to beat baseline (PF dropped below 1.597):** all rsi_s=42/45/48 cells, all rsi_l=52/55/58 cells (12 of 21).

## Phase 2 results — drift sweep (with age=30 baked in)

| Cell | Drift | Trades | PF | Sharpe | DD% | Calmar | NetPnL |
|---|---|---|---|---|---|---|---|
| **Baseline (age=30, drift=0.005)** | **0.005** | **3018** | **1.617** | **4.13** | **32.07** | **5.33** | **13634** |
| drift=0.85% | 0.0085 | 3070 | 1.590 | 4.04 | 32.76 | 5.16 | 13426 |
| drift=1.2% | 0.012 | 3081 | 1.584 | 4.02 | 32.85 | 5.13 | 13338 |
| drift=1.5% | 0.015 | 3084 | 1.574 | 3.96 | 32.85 | 5.09 | 13192 |

**Drift verdict:** Loosening drift HURTS — every relaxation degrades PF, Sharpe, Calmar, NetPnL while only adding 50-66 trades. The 0.5% drift cap is correctly tight and should NOT be changed. Today's VEDL miss is not a tuning opportunity — anti-chase is doing its job.

## Verdict — RECOMMEND single config change

```python
# config.py:559
'signal_age_max_mins': 30,    # was 15 — Phase 1 sweep 2026-04-27 winner
```

All other knobs stay:
- `rsi_long_threshold`: 60 (Phase 1: loosening to 52/55/58 all degrade)
- `rsi_short_threshold`: 40 (Phase 1: loosening to 42/45/48 all degrade)
- `signal_drift_max_pct`: 0.005 (Phase 2: loosening degrades)

### Expected effect (vs current live config)

- **+3.4% more trades** (+100 trades over 2 years on this universe)
- **+1.3% PF** (1.597 → 1.617)
- **+4.8% Sharpe** (3.94 → 4.13)
- **−7.2% MaxDD** (34.57% → 32.07%) — drawdown actually IMPROVES
- **+12.2% Calmar** (4.75 → 5.33)
- **+5.3% NetPnL** (Rs 12,950 → Rs 13,634 cumulative on per-stock 1-lot equivalents)

### Caveats

- **Today's BEL miss is not fully fixable here.** BEL's RSI didn't drop to ≤40 until 10:43 (~70 min after breakout). Even age=45 wouldn't have captured it. BEL was structurally unsuited to the system today. The `age=30` change wouldn't have caught BEL today, but it improves the broader 2-year sample.
- **Today's RELIANCE 8-min lag is unaffected.** That was an RSI threshold issue (50.64 < 60 on first bar), not signal_age. Phase 1 confirms RSI 60 is correct.
- **Today's VEDL miss is unaffected.** That was signal_drift (price >0.5% past breakout), and Phase 2 confirms 0.5% drift is correct.
- **age=30 vs age=45:** age=45 has marginally higher NetPnL (+0.8%) but marginally higher DD (+0.06%). age=30 is the cleaner improvement profile.

## How to apply (post-market only — requires backend restart)

1. Edit `config.py:559`: change `'signal_age_max_mins': 15` → `'signal_age_max_mins': 30`
2. Add a comment referencing this sweep: `# 2026-04-27 sweep: 30 beats 15 on PF + DD + Calmar`
3. Commit + push
4. After 15:30 IST: `ssh arun@94.136.185.54 'cd /home/arun/quantifyd && git pull && sudo systemctl restart quantifyd'`
5. Update [/app/orb](http://94.136.185.54:5000/app/orb) Strategy rules row "Re-evaluation & staleness guards" — change "15" to "30" in the displayed text

## Files

- Sweep script: `scripts/run_filter_sweep.py` (Phase 1)
- Phase 2 script: `scripts/run_phase2.py`
- Results: `results/summary.csv` (25 cells: 21 Phase 1 + 4 Phase 2)
- Logs: `logs/sweep_run.log`, `logs/phase2_run.log`
