"""research/135 Stage F - head-to-head, identical universe / period / costs.

Arms:
  TT_ATTACHED  the attached spec taken literally: S1(20/10) + S2(55/20),
               2N stop, pyramid to 4 units every 0.5N, N-sized at 1% risk
  TT_R83       research/83's incumbent: S1+S2, 2N stop, no pyramid, equal-notional
  TT_OPT       research/135 finalist: 20/10, NO hard stop, no pyramid, EQ, gated
  TT_OPT_PYR   speculative: TT_OPT + 4-unit pyramid (fails the plateau test -
               carried to OOS precisely because IS liked it and its neighbours did not)
  MOM_RECON    the momentum book's RULES rebuilt on this universe: top-8 by
               6m+12m relative strength, monthly rebalance, weekly 100-SMA cash
               gate, daily 15-day Donchian stop
  BENCH        NIFTYBEES buy & hold

Same 78 F&O names, same FUTURES_PROXY costs, same splits. The momentum arm is
a RECONSTRUCTION of rules on this universe, not the live book's own numbers -
the live book trades a Nifty-200 universe, so this isolates the rules.
"""
from __future__ import annotations

import csv
import os
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
from turtle_core import COST                            # noqa: E402
import run_turtle_opt as R                              # noqa: E402

RESULTS = HERE.parent / "results"
FIELDS = ["arm", "split", "cagr", "cagr_gross", "sharpe", "sortino", "max_dd",
          "calmar", "dd_days", "years"]


# ------------------------------------------------------------ momentum arm

def mom_recon(cal, closes, nb, top_n=8, keep_rank=22, gate_sma=100,
              donch=15, costs_on=True):
    """Momentum book rules on this universe. Monthly re-rank, weekly gate,
    daily Donchian stop."""
    syms = sorted(closes.keys())
    px = pd.DataFrame({s: closes[s] for s in syms}).reindex(cal).ffill()
    dlow = pd.DataFrame({s: closes[s].rolling(donch).min() for s in syms}) \
        .reindex(cal).ffill()
    nbr = nb.reindex(cal).ffill()
    gate = (nbr.shift(1) > nbr.rolling(gate_sma).mean().shift(1))

    r6 = px / px.shift(126) - 1
    r12 = px / px.shift(252) - 1
    b6 = nbr / nbr.shift(126) - 1
    b12 = nbr / nbr.shift(252) - 1
    score = (r6.sub(b6, axis=0) + r12.sub(b12, axis=0)) / 2.0

    cash, hold = 1.0, {}
    eq_series = pd.Series(np.nan, index=cal)
    months = pd.Series(cal, index=cal).dt.to_period("M")
    last_of_month = set(pd.Series(cal, index=cal).groupby(months).max().values)
    weeks = pd.Series(cal, index=cal).dt.to_period("W")
    last_of_week = set(pd.Series(cal, index=cal).groupby(weeks).max().values)

    def sell(s, d, p):
        nonlocal cash
        q = hold.pop(s)
        fp = COST.fill_price(p, False) if costs_on else p
        cash += q * fp - (COST.side_cost(fp, q, False) if costs_on else 0.0)

    for d in cal:
        row = px.loc[d]
        # 1. daily Donchian stop
        for s in list(hold):
            p = row.get(s, np.nan)
            dl = dlow.loc[d].get(s, np.nan)
            if not np.isnan(p) and not np.isnan(dl) and p <= dl:
                sell(s, d, p)
        # 2. weekly cash gate
        if d in last_of_week and not bool(gate.get(d, False)):
            for s in list(hold):
                p = row.get(s, np.nan)
                if not np.isnan(p):
                    sell(s, d, p)
        # 3. monthly rebalance
        if d in last_of_month and bool(gate.get(d, False)):
            sc = score.loc[d].dropna()
            if len(sc) >= top_n:
                ranked = sc.sort_values(ascending=False)
                target = list(ranked.index[:top_n])
                keep = set(ranked.index[:keep_rank])
                for s in list(hold):
                    if s not in keep:
                        p = row.get(s, np.nan)
                        if not np.isnan(p):
                            sell(s, d, p)
                nav = cash + sum(q * row.get(s, 0.0) for s, q in hold.items())
                slot = nav / top_n
                for s in target:
                    if s in hold or len(hold) >= top_n:
                        continue
                    p = row.get(s, np.nan)
                    if np.isnan(p) or p <= 0:
                        continue
                    fp = COST.fill_price(p, True) if costs_on else p
                    q = min(slot, cash) / fp
                    if q * fp < nav * 0.005:
                        continue
                    cash -= q * fp + (COST.side_cost(fp, q, True) if costs_on else 0.0)
                    hold[s] = q
        eq_series[d] = cash + sum(q * row.get(s, 0.0) for s, q in hold.items())
    return eq_series.ffill()


# ------------------------------------------------------------------- main

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.setup()
    nb = loader.load_bars("NIFTYBEES", "day", start="2003-01-01",
                          end="2026-08-29")["close"]

    path = RESULTS / "stage_F_bakeoff.csv"
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["arm"], r["split"]) for r in csv.DictReader(f)}
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def emit(arm, split, cs, gs):
        row = {"arm": arm, "split": split,
               "cagr": round(cs["cagr"] * 100, 2),
               "cagr_gross": round(gs["cagr"] * 100, 2) if gs else "",
               "sharpe": round(cs["sharpe"], 3),
               "sortino": round(cs["sortino"], 3),
               "max_dd": round(cs["max_dd"] * 100, 2),
               "calmar": round(cs["calmar"], 3),
               "dd_days": cs["dd_duration_days"],
               "years": round(cs["years"], 2)}
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"[F {split:4s}] {arm:14s} CAGR {row['cagr']:7.2f}%  "
              f"DD {row['max_dd']:7.2f}%  Cal {row['calmar']:6.2f}  "
              f"Sh {row['sharpe']:5.2f}", flush=True)
        return row

    def dual(stop, units, step):
        return (R.get_positions(20, 10, stop, units, step)
                + R.get_positions(55, 20, stop, units, step))

    for split in ("IS", "VAL", "OOS"):
        cal = R._CAL[split]
        arms = {
            "TT_ATTACHED": dict(positions=dual(2.0, 4, 0.5), sizing="N",
                                risk_pct=0.01, stop_mult=2.0),
            "TT_R83": dict(positions=dual(2.0, 1, 0.5), sizing="EQ",
                           stop_mult=2.0),
            "TT_OPT": dict(positions=R.get_positions(20, 10, None, 1, 0.5),
                           sizing="EQ", stop_mult=None),
            "TT_OPT_PYR": dict(positions=R.get_positions(20, 10, None, 4, 0.5),
                               sizing="EQ", stop_mult=None),
        }
        for arm, kw in arms.items():
            if (arm, split) in done:
                continue
            row = R.run_cell("F", arm, split, cap=12, gate_on=True,
                             n_in=20, n_out=10, max_units=1, add_step=0.5, **kw)
            cs = {"cagr": row["cagr"] / 100, "sharpe": row["sharpe"],
                  "sortino": row["sortino"], "max_dd": row["max_dd"] / 100,
                  "calmar": row["calmar"], "dd_duration_days": row["dd_days"],
                  "years": 0}
            eq = pd.read_csv(RESULTS / f"nav_F_{arm}_{split}.csv", index_col=0,
                             parse_dates=True).iloc[:, 0]
            emit(arm, split, metrics.curve_stats(eq),
                 {"cagr": row["cagr_gross"] / 100})

        if ("MOM_RECON", split) not in done:
            eqm = mom_recon(cal, R._CLOSES, nb, costs_on=True)
            eqg = mom_recon(cal, R._CLOSES, nb, costs_on=False)
            eqm.to_csv(RESULTS / f"nav_F_MOM_RECON_{split}.csv")
            emit("MOM_RECON", split, metrics.curve_stats(eqm),
                 metrics.curve_stats(eqg))

        if ("BENCH", split) not in done:
            b = nb.reindex(cal).ffill().dropna()
            emit("BENCH", split, metrics.curve_stats(b / b.iloc[0]), None)

    print("\nSTAGE F COMPLETE", flush=True)


if __name__ == "__main__":
    main()
