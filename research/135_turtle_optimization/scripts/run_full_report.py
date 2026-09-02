"""research/135 - continuous 2005-2026 run of the comparison arms, for the
report's equity/drawdown visuals and the per-year stability table.

Selection happened on IS only (see the STATUS doc); this is presentation of
the already-decided arms over one unbroken NAV, plus the per-year detail that
shows WHEN each book made its money.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R81 = HERE.parents[1] / "81_swing_edge_discovery"
for p in (str(HERE), str(R81), str(R81.parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader, metrics                      # noqa: E402
import run_turtle_opt as R                              # noqa: E402
import run_stage_f as F                                 # noqa: E402

RESULTS = HERE.parent / "results"


def main():
    R.SPLITS["FULL"] = ("2005-01-01", "2026-08-29")
    R.setup()
    nb = loader.load_bars("NIFTYBEES", "day", start="2003-01-01",
                          end="2026-08-29")["close"]
    cal = R._CAL["FULL"]

    def dual(stop, units, step):
        return (R.get_positions(20, 10, stop, units, step)
                + R.get_positions(55, 20, stop, units, step))

    arms = {
        "TT_ATTACHED": dict(positions=dual(2.0, 4, 0.5), sizing="N",
                            risk_pct=0.01, stop_mult=2.0),
        "TT_R83": dict(positions=dual(2.0, 1, 0.5), sizing="EQ", stop_mult=2.0),
        "TT_OPT": dict(positions=R.get_positions(20, 10, None, 1, 0.5),
                       sizing="EQ", stop_mult=None),
        "TT_OPT_PYR": dict(positions=R.get_positions(20, 10, None, 4, 0.5),
                           sizing="EQ", stop_mult=None),
    }
    curves = {}
    for arm, kw in arms.items():
        R.run_cell("FULL", arm, "FULL", cap=12, gate_on=True, n_in=20, n_out=10,
                   max_units=1, add_step=0.5, **kw)
        curves[arm] = pd.read_csv(RESULTS / f"nav_FULL_{arm}_FULL.csv",
                                  index_col=0, parse_dates=True).iloc[:, 0]
    curves["MOM_RECON"] = F.mom_recon(cal, R._CLOSES, nb, costs_on=True)
    b = nb.reindex(cal).ffill().dropna()
    curves["BENCH"] = b / b.iloc[0]

    out = pd.DataFrame(curves)
    out.to_csv(RESULTS / "full_curves_2005_2026.csv")

    print("\n=== FULL PERIOD 2005-2026 (continuous NAV, net of costs) ===")
    rows = []
    for arm, eq in curves.items():
        cs = metrics.curve_stats(eq.dropna())
        rows.append({"arm": arm, "CAGR%": round(cs["cagr"] * 100, 2),
                     "Sharpe": round(cs["sharpe"], 2),
                     "Sortino": round(cs["sortino"], 2),
                     "MaxDD%": round(cs["max_dd"] * 100, 2),
                     "Calmar": round(cs["calmar"], 2),
                     "DDdays": cs["dd_duration_days"]})
    tbl = pd.DataFrame(rows).set_index("arm")
    print(tbl.to_string())
    tbl.to_csv(RESULTS / "full_summary.csv")

    print("\n=== PER-YEAR RETURN % (net) ===")
    per = {}
    for arm, eq in curves.items():
        r = eq.dropna().pct_change()
        per[arm] = (metrics.per_year_table(r)["return"] * 100).round(1)
    pt = pd.DataFrame(per)
    print(pt.to_string())
    pt.to_csv(RESULTS / "full_per_year.csv")

    print("\n=== era means (per-year avg) ===")
    for name, yrs in (("2005-2017", range(2005, 2018)),
                      ("2018-2023", range(2018, 2024)),
                      ("2024-2026", range(2024, 2027))):
        sel = pt.loc[[y for y in yrs if y in pt.index]]
        print(f"  {name}: " + "  ".join(
            f"{a}={sel[a].mean():6.1f}%" for a in pt.columns))

    print("\nFULL REPORT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
