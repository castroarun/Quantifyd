"""Demo: generate a client tearsheet for the midcap-momentum book (risk-off=cash)
vs the Nifty 50, using the reusable research/_utilities/tearsheet.py generator.
"""
import sys
from pathlib import Path

import run_overlay as ov
rs2 = ov.rs2
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_utilities"))
from tearsheet import generate_tearsheet  # noqa: E402

close, tv = rs2.load()
strat = ov.run(close, tv, "cash", strikes=ov.STRK)["nav"]
bench = close[ov.BENCH].reindex(strat.index).ffill()

out = generate_tearsheet(
    strat, bench,
    name="Midcap Momentum — RS Concentrated (Regime-Gated)",
    meta=dict(bench="Nifty 50", tag="BACKTEST 2014–2026",
              disclosures="Backtest, net of 0.4% round-trip cost; idle cash @6.5%. "
                          "Universe = point-in-time mid-cap liquidity band. Survivorship "
                          "partially controlled. Past performance is not indicative of "
                          "future results. For discussion only."),
    out_dir=Path(__file__).resolve().parents[1] / "results")
print("wrote", out)
