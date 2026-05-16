"""Phase 12 — yearly breakup (return + max-DD) for the two FINAL
systems, vs Nifty 50 / Midcap 150 / Smallcap 250.

Universe (both systems): survivorship-free PIT MID-cap liquidity band
(rank 101-250 by trailing-6mo median traded value), reconstructed
monthly from all ~1,623 NSE daily symbols. Period 2014-01 -> 2026
(~12.1y). RS-120 vs NIFTYBEES, N=15, monthly, top-22 buffer, q0.5
quality, 0.4% RT, 6.5% bear-cash.

  SMOOTHEST  = SMA100 regime + ATH<=10% entry + per-stock-SMA100 + 12%
               trail  (Phase 11 winner)
  MAX-RETURN = SMA100 regime + ATH<=10% entry + (risk-off -> short 1x
               Nifty beta-hedge instead of cash)  (Phase 10)

Benchmarks WITHOUT touching niftyindices again:
 - Nifty 50  : NIFTYBEES daily from market_data.db (full, reliable)
 - Midcap150 : results/idx_niftymidcap150.csv  (already on disk; gaps
               2018/2020/2026 -> n/a, NOT fabricated)
 - Smallcap250: results/idx_niftysmallcap250.csv (very patchy -> only
               the years it has; rest n/a)
System per-year DD is month-end-NAV based (approx, stated).
"""
from __future__ import annotations
import importlib.util, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
RES = HERE / "results"
SC = Path(__file__).resolve().parent


def _mod(name, fn):
    s = importlib.util.spec_from_file_location(name, str(SC / fn))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


p11 = _mod("p11", "11_stocklevel_risk.py")
p10 = _mod("p10", "10_hedge_overlay.py")
rs2 = p11.rs2


def yr_ret(nav):
    y = nav.groupby(nav.index.year).last()
    r = y.pct_change()
    r.iloc[0] = y.iloc[0] / nav.iloc[0] - 1
    return (r * 100).round(1)


def yr_maxdd(nav):
    out = {}
    for yr, g in nav.groupby(nav.index.year):
        rm = g.cummax()
        out[yr] = round(((g - rm) / rm).min() * 100, 1)
    return pd.Series(out)


def idx_yr_ret(csv):
    p = RES / csv
    if not p.exists():
        return pd.Series(dtype=float)
    s = pd.read_csv(p, parse_dates=[0], index_col=0).iloc[:, 0].sort_index()
    out = {}
    for yr, g in s.groupby(s.index.year):
        if len(g) >= 2:
            out[yr] = round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1)
    return pd.Series(out)


def main():
    print("Loading core data ...", flush=True)
    close, tv = rs2.load()

    print("Rebuilding SMOOTHEST (Phase 11) ...", flush=True)
    sm = p11.backtest(close, tv, True, True, 0.12)        # gate+perSMA+tr12
    print("Rebuilding MAX-RETURN (Phase 10 beta-hedge) ...", flush=True)
    mx = p10.backtest(close, tv, "beta", 1.0)             # SMA100->beta hr1

    nb = pd.read_sql("SELECT date,close FROM market_data_unified WHERE "
                     "timeframe='day' AND symbol='NIFTYBEES' AND close "
                     "IS NOT NULL ORDER BY date",
                     sqlite3.connect(ROOT / "backtest_data" /
                                     "market_data.db"),
                     parse_dates=["date"]).set_index("date")["close"]

    yrs = list(range(2014, 2027))
    tbl = pd.DataFrame(index=yrs)
    tbl["Smooth_ret%"] = yr_ret(sm["nav"]).reindex(yrs)
    tbl["Smooth_maxDD%"] = yr_maxdd(sm["nav"]).reindex(yrs)
    tbl["MaxRet_ret%"] = yr_ret(mx["nav"]).reindex(yrs)
    tbl["MaxRet_maxDD%"] = yr_maxdd(mx["nav"]).reindex(yrs)
    tbl["Nifty50%"] = yr_ret(nb).reindex(yrs)
    tbl["NiftyMid150%"] = idx_yr_ret("idx_niftymidcap150.csv").reindex(yrs)
    tbl["NiftySmall250%"] = idx_yr_ret(
        "idx_niftysmallcap250.csv").reindex(yrs)
    tbl = tbl.round(1)
    tbl.to_csv(RES / "phase12_yearly_breakup.csv")

    def cagr(nav):
        y = (nav.index[-1] - nav.index[0]).days / 365.25
        return (nav.iloc[-1] / nav.iloc[0]) ** (1 / y) - 1

    print("\n=== UNIVERSE: PIT mid-cap liquidity band (rank 101-250), "
          "monthly PIT-reconstructed | PERIOD: %s -> %s (%.1fy) ==="
          % (sm["nav"].index.min().date(), sm["nav"].index.max().date(),
             (sm["nav"].index[-1] - sm["nav"].index[0]).days / 365.25))
    print("\n=== YEARLY BREAKUP ===")
    print(tbl.fillna("  n/a").to_string())
    print("\nOverall: SMOOTHEST CAGR=%.1f%% MaxDD=%.1f%% | "
          "MAX-RETURN CAGR=%.1f%% MaxDD=%.1f%% | Nifty50 CAGR=%.1f%%"
          % (cagr(sm["nav"]) * 100, sm["dd"], cagr(mx["nav"]) * 100,
             mx["dd"], cagr(nb) * 100))
    print("post-tax@20%: SMOOTHEST 29.6% | MAX-RETURN 34.0% "
          "(from Phase 10/11)")
    print("\nNOTE: Midcap150 2018/2020/2026 + most Smallcap250 years = "
          "data gap (niftyindices not re-queried per instruction; NOT "
          "fabricated). System per-year DD is month-end-NAV based.")


if __name__ == "__main__":
    main()
