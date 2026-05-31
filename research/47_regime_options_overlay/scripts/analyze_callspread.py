"""Does the defined-risk DEBIT CALL SPREAD during the bearish (risk-off) regime
actually help? Isolate its marginal P&L vs plain cash for EVERY risk-off month,
across all years, and chart equity curves vs NIFTY50.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_overlay as ov
rs2 = ov.rs2
RES = ov.RES


def main():
    print("loading...", flush=True)
    close, tv = rs2.load()
    log_cs = []
    cash = ov.run(close, tv, "cash", strikes=ov.STRK)
    cs = ov.run(close, tv, "callspread", strikes=ov.STRK, log=log_cs)
    nifty = close[ov.BENCH].dropna()

    # ---- per risk-off month: marginal call-spread return over cash ----
    lg = pd.DataFrame(log_cs, columns=["date", "mode", "nf", "marg", "base"])
    lg["year"] = pd.to_datetime(lg["date"]).dt.year
    lg["up"] = lg["nf"] > 0
    lg["win"] = lg["marg"] > 0

    print("\n=== Does the debit call-spread help during bearish regime? "
          "(marginal vs cash, per risk-off month) ===")
    print(f"{'year':>5} {'roff_mo':>7} {'mkt_up':>6} {'cs_wins':>7} "
          f"{'cs_marg_sum%':>12} {'CASH_yr%':>9} {'CALLSPR_yr%':>11} {'diff_pp':>8}")
    cash_py, cs_py = cash["py"], cs["py"]
    tot_marg = 0.0
    for y in range(2014, 2027):
        sub = lg[lg["year"] == y]
        if len(sub) == 0 and y not in cash_py.index:
            continue
        roff = len(sub)
        ups = int(sub["up"].sum())
        wins = int(sub["win"].sum())
        marg = sub["marg"].sum() * 100          # scale-free sum of monthly % impacts
        tot_marg += marg
        cy = cash_py.get(y, np.nan)
        sy = cs_py.get(y, np.nan)
        print(f"{y:>5} {roff:>7} {ups:>6} {wins:>7} {marg:>+12.1f} "
              f"{cy:>9.1f} {sy:>11.1f} {sy-cy:>+8.1f}")
    n = len(lg)
    print(f"\n  TOTAL risk-off months: {n}   mkt-up: {lg['up'].sum()} "
          f"({lg['up'].mean()*100:.0f}%)   call-spread wins: {lg['win'].sum()} "
          f"({lg['win'].mean()*100:.0f}%)")
    print(f"  Call-spread marginal sum (scale-free): {tot_marg:+.1f}%   "
          f"avg/risk-off-month: {lg['marg'].mean()*100:+.2f}%")
    print(f"  Full-period: CASH CAGR {cash['cagr']:.1f}% (DD {cash['dd']:.1f}%, "
          f"Cal {cash['cal']:.2f})  vs  CALLSPREAD CAGR {cs['cagr']:.1f}% "
          f"(DD {cs['dd']:.1f}%, Cal {cs['cal']:.2f})")

    # ---- Chart 1: full-period equity curves vs NIFTY50 (log scale) ----
    cN, sN = cash["nav"], cs["nav"]
    n9 = nifty.reindex(cN.index, method="ffill")
    base_date = cN.index[0]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(cN.index, cN / cN.iloc[0], label=f"Risk-off=CASH (CAGR {cash['cagr']:.0f}%)", lw=1.8)
    ax.plot(sN.index, sN / sN.iloc[0], label=f"Risk-off=CALL-SPREAD (CAGR {cs['cagr']:.0f}%)", lw=1.8)
    ax.plot(n9.index, n9 / n9.iloc[0], label="NIFTY 50 (NIFTYBEES)", lw=1.5, color="gray", ls="--")
    ax.set_yscale("log")
    ax.set_title("Midcap-momentum strategy: CASH vs CALL-SPREAD risk-off action vs Nifty 50\n"
                 "(2014-2026, log scale, modelled IV)")
    ax.set_ylabel("Growth of 1 (log)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "curve_cash_vs_callspread_vs_nifty.png", dpi=110)

    # ---- Chart 2: per-year call-spread vs cash difference (the answer) ----
    yrs = [y for y in range(2014, 2027) if y in cash_py.index]
    diff = [cs_py.get(y, np.nan) - cash_py.get(y, np.nan) for y in yrs]
    roffmo = [len(lg[lg["year"] == y]) for y in yrs]
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    colors = ["#2e7d32" if d >= 0 else "#c62828" for d in diff]
    ax2.bar([str(y) for y in yrs], diff, color=colors)
    for i, (d, rm) in enumerate(zip(diff, roffmo)):
        ax2.annotate(f"{rm}mo", (i, d), ha="center",
                     va="bottom" if d >= 0 else "top", fontsize=8, color="#555")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_title("Call-spread re-entry MINUS cash, per year (pp)  —  green=helped, red=hurt\n"
                  "label = # risk-off months that year")
    ax2.set_ylabel("Annual return difference (pp)")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout(); fig2.savefig(RES / "callspread_minus_cash_peryear.png", dpi=110)
    print(f"\nsaved: {RES / 'curve_cash_vs_callspread_vs_nifty.png'}")
    print(f"saved: {RES / 'callspread_minus_cash_peryear.png'}")


if __name__ == "__main__":
    main()
