"""EXP-G4: portfolio construction for gap-up + ORB long (locked configs).

Combined book = index sleeve (NIFTY W12) + stock sleeve (9 names, W6).
Sizing pre-declared from G3 Monte Carlo: index 1.0% risk/trade, stocks 0.5%.
Concurrency cap 6 positions; per-position notional cap 25% of equity.
Daily mark-to-market NAV 2015-02 .. 2023-12 (IS+Val; OOS 2024+ locked).
Gates (brief section 6): Sharpe >= 1, Calmar >= 1, MaxDD <= 20%,
DD duration <= 9 months, PF >= 1.3, expectancy > 2x cost.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
for p in (str(STUDY), str(ROOT), str(STUDY / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader, metrics                     # noqa: E402
from run_g3_orb_validation import trades_for, CFG1, CFG3, STOCKS  # noqa: E402

RESULTS = STUDY / "experiments" / "G4_portfolio" / "results"
START, END = "2015-02-01", "2023-12-31"
RISK_IDX, RISK_STK = 0.010, 0.005
MAX_CONCURRENT = 6
MAX_NOTIONAL_PCT = 0.25
START_EQUITY = 1.0


def collect_trades() -> pd.DataFrame:
    nifty = loader.load_bars("NIFTY50", "5minute", start=START, end=END)
    tr_i = trades_for(nifty, 12, 0.0025, START, END, CFG1, "NIFTY50")
    tr_i["risk_pct"] = RISK_IDX
    parts = [tr_i]
    for s in STOCKS:
        df = loader.load_bars(s, "5minute", start=START, end=END)
        t = trades_for(df, 6, 0.0025, START, END, CFG3, s)
        t["risk_pct"] = RISK_STK
        parts.append(t)
    tr = pd.concat(parts, ignore_index=True)
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    tr["exit_time"] = pd.to_datetime(tr["exit_time"])
    return tr.sort_values("entry_time").reset_index(drop=True)


def daily_closes() -> dict[str, pd.Series]:
    out = {}
    for s in ["NIFTY50"] + STOCKS:
        d = loader.to_daily(loader.load_bars(s, "5minute", start=START, end=END))
        out[s] = d["close"]
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trades = collect_trades()
    closes = daily_closes()
    cal = sorted(set().union(*[set(c.index) for c in closes.values()]))
    cal = pd.DatetimeIndex([d for d in cal
                            if pd.Timestamp(START) <= d <= pd.Timestamp(END)])
    day_pos = {d: i for i, d in enumerate(cal)}

    equity = pd.Series(np.nan, index=cal)
    eq = START_EQUITY
    open_pos = []          # dicts: symbol, qty, dir, entry_px, exit_time, exit_px, marks
    ti = 0
    n_taken = n_skipped = 0
    taken_rows = []

    for d in cal:
        # 1) take today's signals (entry during session d) subject to caps
        while ti < len(trades) and trades.loc[ti, "entry_time"].normalize() <= d:
            tr = trades.loc[ti]
            ti += 1
            if tr["entry_time"].normalize() != d:
                continue
            if len(open_pos) >= MAX_CONCURRENT:
                n_skipped += 1
                continue
            risk_per_unit = abs(tr["entry_px"] - tr["stop"])
            if risk_per_unit <= 0:
                continue
            qty = (eq * tr["risk_pct"]) / risk_per_unit
            qty = min(qty, (eq * MAX_NOTIONAL_PCT) / tr["entry_px"])
            # engine's net_ret = slipped price return - charges; recover the
            # charge component so the NAV deducts it at exit
            slipped_ret = tr["direction"] * (tr["exit_px"] / tr["entry_px"] - 1)
            charges_frac = slipped_ret - tr["net_ret"]
            open_pos.append({"symbol": tr["symbol"], "qty": qty,
                             "dir": tr["direction"], "entry_px": tr["entry_px"],
                             "exit_day": tr["exit_time"].normalize(),
                             "exit_px": tr["exit_px"], "mark": tr["entry_px"],
                             "charges": charges_frac * tr["entry_px"] * qty})
            n_taken += 1
            taken_rows.append({"entry_time": tr["entry_time"],
                               "symbol": tr["symbol"], "net_ret": tr["net_ret"],
                               "notional_pct": qty * tr["entry_px"] / eq})
        # 2) mark-to-market all open positions at today's close (or exit px)
        pnl = 0.0
        still = []
        for p in open_pos:
            c = closes[p["symbol"]].get(d, np.nan)
            if d >= p["exit_day"]:
                px = p["exit_px"]
                pnl += p["qty"] * p["dir"] * (px - p["mark"]) - p["charges"]
            else:
                if not np.isnan(c):
                    pnl += p["qty"] * p["dir"] * (c - p["mark"])
                    p["mark"] = c
                still.append(p)
        open_pos = still
        eq += pnl
        equity[d] = eq

    equity = equity.ffill()
    cs = metrics.curve_stats(equity)
    print(f"taken={n_taken} skipped_at_cap={n_skipped}")
    print("\n=== G4 combined book (2015-02..2023-12, IS+Val) ===")
    for k, v in cs.items():
        print(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")
    yr = metrics.per_year_table(equity.pct_change())
    print("\nper-year returns:")
    print((yr * 100).round(2).to_string())

    # benchmark: NIFTY buy-and-hold
    bench = closes["NIFTY50"].reindex(cal).ffill()
    bs = metrics.curve_stats(bench / bench.iloc[0])
    print(f"\nNIFTY B&H same period: CAGR {bs['cagr']:.2%}  Sharpe {bs['sharpe']:.2f}  "
          f"MaxDD {bs['max_dd']:.2%}  Calmar {bs['calmar']:.2f}")
    corr = equity.pct_change().corr(bench.pct_change())
    print(f"daily corr vs NIFTY: {corr:.3f}")

    equity.to_csv(RESULTS / "g4_nav.csv")
    pd.DataFrame(taken_rows).to_csv(RESULTS / "g4_taken_trades.csv", index=False)
    print(f"\ndone in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
