"""research/135 Stage E - port the MOMENTUM book's risk machinery onto Turtle.

Two things are borrowed from services/momentum_paper.py:

  E1  the GATE. The momentum book gates on NIFTYBEES vs its 100-day SMA
      (checked weekly for stocks, DAILY for the hedge decision - r/108b).
      Turtle here has been using a 200-DMA gate. Bake them off, plus no-gate.

  E2  the PUT HEDGE (r/105: bi-weekly ~14 DTE, ATM, notional = 2.0x equity).
      The momentum design's key idea: at a risk-off gate do NOT go to cash -
      hold the stocks and buy NIFTY puts instead. Test that on the Turtle book.

Option modelling (stated plainly, because it drives the answer):
  - underlying = NIFTYBEES close (NIFTY tracker; the only series back to 2005)
  - IV = INDIAVIX/100 where available (2015-01 onward, REAL implied vol, so the
    variance risk premium that makes index puts expensive is included)
  - pre-2015 there is no VIX, so IV = 20d realised vol x k, where k is the
    median VIX/realised ratio measured on the 2015+ overlap. This is an
    ASSUMPTION and it flatters nothing: k > 1 makes the puts expensive.
  - Black-Scholes European put, r = 6.5%, rolled when DTE < 8 back to ~14 DTE,
    always re-struck ATM. Roll pays a spread on the premium both ways.
  - reported at spread 1.5% and at 3.0% (cost sensitivity, per the playbook).
"""
from __future__ import annotations

import csv
import os
import sys
from math import log, sqrt, exp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
R81 = HERE.parents[1] / "81_swing_edge_discovery"
for p in (str(HERE), str(R81), str(R81.parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader, metrics                      # noqa: E402
import run_turtle_opt as R                              # noqa: E402

RESULTS = HERE.parent / "results"
RF = 0.065


def bs_put(S, K, T, sigma, r=RF):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _load_vix() -> pd.Series:
    """INDIAVIX rows carry a ' 00:00:00' suffix the shared loader can't parse,
    so read it directly with a mixed-format date parse."""
    import sqlite3
    db = Path("/home/arun/quantifyd/backtest_data/market_data.db")
    con = sqlite3.connect(str(db))
    try:
        df = pd.read_sql_query(
            "SELECT date, close FROM market_data_unified "
            "WHERE symbol='INDIAVIX' AND timeframe='day' ORDER BY date", con)
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.normalize()
    s = df.set_index("date")["close"].astype(float)
    return s[~s.index.duplicated(keep="first")].sort_index()


def iv_series(nb: pd.Series) -> pd.Series:
    """INDIAVIX/100 spliced onto a VRP-scaled realised-vol proxy pre-2015."""
    iv = (_load_vix() / 100.0).reindex(nb.index).astype(float)
    rv = nb.pct_change().rolling(20).std() * np.sqrt(252)
    both = pd.concat([iv, rv], axis=1).dropna()
    k = float((both.iloc[:, 0] / both.iloc[:, 1]).median())
    print(f"  VRP calibration: median VIX/realised = {k:.3f} "
          f"(n={len(both)} overlap days)")
    proxy = (rv * k).clip(0.06, 0.90)
    out = iv.fillna(proxy).ffill().bfill()
    return out.clip(0.06, 0.90), k


def put_overlay(nb: pd.Series, iv: pd.Series, riskoff: pd.Series,
                cal: pd.DatetimeIndex, ratio: float = 2.0,
                dte_target: int = 14, dte_min: int = 8,
                spread: float = 0.015, moneyness: float = 0.0) -> pd.Series:
    """Daily P&L of the hedge sleeve as a FRACTION of NAV.

    Holds puts only while `riskoff` is true. Notional = ratio x NAV, so the
    contract count is re-struck at each roll off the prevailing spot.
    `moneyness` -0.05 = strike 5% below spot (cheaper, OTM).

    ACCOUNTING: the option is an ASSET, marked to market daily. The cost of
    the hedge is therefore its DECAY (plus the bid/ask spread paid on entry
    and exit) - the premium is NOT expensed on top of that, which would
    double-count it.
    """
    pay = pd.Series(0.0, index=cal)
    K = None
    expiry_i = None
    n = 0.0
    prev_val = 0.0
    cal_list = list(cal)
    for i, d in enumerate(cal_list):
        S = float(nb.get(d, np.nan))
        sig = float(iv.get(d, np.nan))
        if np.isnan(S) or np.isnan(sig):
            continue

        # 1. mark the existing holding to today
        if K is not None:
            T = max((expiry_i - i) / 252.0, 0.0)
            val = bs_put(S, K, T, sig)
            pay[d] += n * (val - prev_val)
            prev_val = val

        on = bool(riskoff.get(d, False))
        expiring = K is not None and (expiry_i - i) < dte_min

        # 2. close if the gate turned risk-on, or the contract is near expiry
        if K is not None and (not on or expiring):
            pay[d] -= n * prev_val * spread          # exit spread
            K, n, prev_val, expiry_i = None, 0.0, 0.0, None

        # 3. open a fresh one if we should be hedged and are not
        if on and K is None:
            K = S * (1.0 + moneyness)
            expiry_i = i + dte_target
            val = bs_put(S, K, dte_target / 252.0, sig)
            if val <= 0:
                K, expiry_i = None, None
                continue
            n = ratio / S                            # notional = ratio x NAV
            pay[d] -= n * val * spread               # entry spread
            prev_val = val
    return pay


FIELDS = R.FIELDS + ["gate_kind", "hedge", "spread"]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.setup()
    nb = loader.load_bars("NIFTYBEES", "day", start="2003-01-01",
                          end="2026-08-29")["close"]
    iv, k = iv_series(nb)

    g200 = (nb.shift(1) > nb.rolling(200).mean().shift(1))
    g100 = (nb.shift(1) > nb.rolling(100).mean().shift(1))
    gates = {"none": None, "sma200": g200, "sma100": g100}

    base = dict(n_in=20, n_out=10, stop_mult=None, max_units=4, add_step=0.5,
                sizing="EQ", cap=12, unit_frac=0.10)

    path = RESULTS / "stage_E_gate_hedge.csv"
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["label"], r["split"]) for r in csv.DictReader(f)}
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def emit(row):
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

    for split in ("IS", "VAL"):
        cal = R._CAL[split]
        # ---- E1: gate bake-off, no hedge
        for gk, g in gates.items():
            label = f"gate_{gk}"
            if (label, split) in done:
                continue
            row = R.run_cell("E", label, split, **base,
                             gate_on=(g is not None), gate_override=g)
            row.update(gate_kind=gk, hedge="none", spread="")
            emit(row)
            print(f"[E1 {split}] {label:14s} CAGR {row['cagr']:6.2f}%  "
                  f"DD {row['max_dd']:7.2f}%  Cal {row['calmar']:5.2f}  "
                  f"Sh {row['sharpe']:5.2f}", flush=True)

        # ---- E2: no cash gate, hold through + buy puts when risk-off
        for gk in ("sma100", "sma200"):
            riskoff = ~gates[gk].reindex(cal).fillna(False).astype(bool)
            for mny in (0.0, -0.05):
                for ratio in (0.5, 1.0, 2.0):
                    for sp in (0.015, 0.030):
                        if sp == 0.030 and ratio != 1.0:
                            continue          # spread sensitivity on one ratio
                        label = f"hedge_{gk}_r{ratio}_m{mny}_sp{sp}"
                        if (label, split) in done:
                            continue
                        pay = put_overlay(nb, iv, riskoff, cal, ratio=ratio,
                                          spread=sp, moneyness=mny)
                        row = R.run_cell("E", label, split, **base,
                                         gate_on=False, put_payoff=pay)
                        row.update(gate_kind=gk,
                                   hedge=f"put_x{ratio}_m{mny}", spread=sp)
                        emit(row)
                        print(f"[E2 {split}] {label:30s} CAGR {row['cagr']:6.2f}%  "
                              f"DD {row['max_dd']:7.2f}%  Cal {row['calmar']:5.2f}  "
                              f"Sh {row['sharpe']:5.2f}", flush=True)

        # ---- E3: gate AND hedge (cash out AND own puts while out)
        for gk in ("sma100", "sma200"):
            riskoff = ~gates[gk].reindex(cal).fillna(False).astype(bool)
            for mny in (0.0, -0.05):
                for ratio in (0.5, 1.0):
                    label = f"gate+hedge_{gk}_r{ratio}_m{mny}"
                    if (label, split) in done:
                        continue
                    pay = put_overlay(nb, iv, riskoff, cal, ratio=ratio,
                                      spread=0.015, moneyness=mny)
                    row = R.run_cell("E", label, split, **base, gate_on=True,
                                     gate_override=gates[gk], put_payoff=pay)
                    row.update(gate_kind=gk, hedge=f"put_x{ratio}_m{mny}",
                               spread=0.015)
                    emit(row)
                    print(f"[E3 {split}] {label:30s} CAGR {row['cagr']:6.2f}%  "
                          f"DD {row['max_dd']:7.2f}%  Cal {row['calmar']:5.2f}  "
                          f"Sh {row['sharpe']:5.2f}", flush=True)

    print("\nSTAGE E COMPLETE", flush=True)


if __name__ == "__main__":
    main()
