"""Update research/INDEX.md + TODO.md for research/64. Run on VPS, idempotent."""
from pathlib import Path
ROOT = Path("/home/arun/quantifyd")

idx = (ROOT / "research" / "INDEX.md").read_text()
entry = """

## 64 - factor_index_rotation (2026-06-14)
**Nifty factor-index rotation/diversification** (follow-on to research/63). Does
"diversify > select" transfer from asset classes to the Nifty single factors
(Momentum/Quality/Value/LowVol/Alpha)? **NO** - factors are mostly the same bet
(mean cross-corr 0.65; 0.79-0.91 vs Nifty). Equal-weight factors tops Calmar 0.76
(best pure-factor = Mom+LowVol, 17.4%/-22.9% DD); rotation 0.67. **The win is COMBINING:
Momentum factor + Gold + Nasdaq, INVERSE-VOL, monthly = Calmar 1.77, CAGR 22.1%, DD -12.5%,
cost-insensitive** - marginally beats research/63 Nifty book 1.75 by upgrading the equity
sleeve + taming Nasdaq vol. AllFactors+Gold+Nasdaq DILUTES 1.18 -> concentrate equity into
one factor. **VERDICT: STRATEGY (candidate)** - incremental upgrade to research/63; factor
selection/diversification alone = SIGNAL. Published /app/backtest/factor-index-rotation.
See research/64_factor_index_rotation/results/RESULTS.md.
"""
if "64 - factor_index_rotation" not in idx and "64 — factor_index_rotation" not in idx:
    (ROOT / "research" / "INDEX.md").write_text(idx.rstrip() + "\n" + entry)
    print("INDEX updated")
else:
    print("INDEX already has 64")

todo_p = ROOT / "TODO.md"
todo = todo_p.read_text()
qhdr = "## ⏸ QUEUED — research/64 Nifty FACTOR-INDEX rotation/diversification (user request 2026-06-14)"
done = ("## ✅ DONE 2026-06-14 — research/64 Nifty factor-index rotation (was QUEUED)\n"
        "Verdict STRATEGY (candidate). Factors too correlated (0.65) to diversify "
        "(EW Calmar 0.76); WINNER = Momentum+Gold+Nasdaq inverse-vol monthly = Calmar 1.77 / "
        "CAGR 22.1% / DD -12.5%, marginal upgrade to research/63. Published "
        "/app/backtest/factor-index-rotation. Next: Momentum ETF NAV vs index + market trend "
        "overlay; SILVERBEES add-on (2022+, short window) tested per user request. See RESULTS.md.")
if qhdr in todo and "DONE 2026-06-14 — research/64" not in todo:
    todo = todo.replace(qhdr, done, 1)
    todo_p.write_text(todo)
    print("TODO updated")
else:
    print("TODO already updated or queued header not found")
