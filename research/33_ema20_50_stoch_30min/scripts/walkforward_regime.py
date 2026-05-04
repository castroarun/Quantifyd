"""Walk-forward with NIFTY-regime gate applied at every trade.

Findings from regime_signals.py — these regime indicators cleanly
separated winning vs losing trades in the AGGREGATE 1042-trade set:

  Regime gate              | Trades when ON  | Sharpe  | Total ret
  ----------------------------------------------------------------
  NIFTY 30m ATR/close > 0.30%  |   113  |  +3.34   |  +66.8%
  NIFTY 30m ATR/close > 0.20%  |   674  |  +1.10   |  +82.3%
  NIFTY daily close < EMA200   |    86  |  +2.57   |  +35.9%
  NIFTY daily close < EMA50    |   205  |  +1.17   |  +32.2%
  NIFTY 30m ADX 15-20         |   134  |  +1.98   |  +39.8%
  NIFTY daily vol20 > 1.2%    |   131  |  +1.76   |  +34.8%

These are IN-SAMPLE attributions on a fixed trade set. Walk-forward
with the regime gate as a real-time on/off switch is the proper test.

Best rule from prior walkforward.py was R4 (trailing Sharpe >= 1.5),
but that produced negative OOS Sharpe (-0.24). Best from regime_signals
was R9 (trailing total return >= 5%) with OOS Sharpe +1.53.

Here we test combinations: best stock-selection rule (R9) × regime gates.
"""

from __future__ import annotations

import csv
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data30 import to_30min  # noqa: E402
from indicators_stoch import ema, atr, adx  # noqa: E402
from final_run import (  # noqa: E402
    list_universe, prepare_30min, gen_setups_locked,
    UNIVERSE_START, UNIVERSE_END,
    SIDE, EXIT_ID, COST_PCT,
)
from param_sweep import sim  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "results"
LOG_DIR = SCRIPT_DIR.parent / "logs"
DB_PATH = SCRIPT_DIR.parent.parent.parent / "backtest_data" / "market_data.db"

LOOKBACK_MONTHS = 6
TRADE_MONTHS = 3
MIN_TRADES_LOOKBACK = 5


def per_stock_score(rets, holds):
    n = len(rets)
    if n == 0: return {"trades":0,"sharpe":0,"total_ret":0,"expectancy":0,"pf":0}
    arr = np.array(rets)
    wins = arr > 0; losses = arr <= 0
    pf = (arr[wins].sum() / -arr[losses].sum()) if losses.any() and arr[losses].sum() < 0 else 99.0
    mean_r = float(arr.mean()); std_r = float(arr.std(ddof=1)) if n > 1 else 0
    avg_hold = float(np.mean(holds))
    tpy = 3125 / max(avg_hold, 1)
    sharpe = (mean_r/std_r * math.sqrt(max(tpy,1))) if std_r > 0 else 0
    return {"trades":n,"sharpe":float(sharpe),"total_ret":float(arr.sum()),
            "expectancy":float(mean_r),"pf":float(min(pf,99))}


def select_basket(rule, scores, prev):
    q = scores[scores["trades"] >= MIN_TRADES_LOOKBACK].copy()
    if q.empty: return list(prev)
    if rule == "R4_sh1.5":  return q[q["sharpe"] >= 1.5]["symbol"].tolist()
    if rule == "R9_totret5": return q[q["total_ret"] >= 5]["symbol"].tolist()
    if rule == "ALL":        return q["symbol"].tolist()
    raise ValueError(rule)


def load_nifty():
    con = sqlite3.connect(str(DB_PATH))
    sql5 = """SELECT date, open, high, low, close, volume FROM market_data_unified
              WHERE symbol='NIFTY50' AND timeframe='5minute'
              AND date >= ? AND date <= ? ORDER BY date"""
    df5 = pd.read_sql(sql5, con, params=("2023-01-01", UNIVERSE_END + " 23:59:59"))
    df5["date"] = pd.to_datetime(df5["date"])
    n30 = to_30min(df5.set_index("date"))
    sqld = """SELECT date, open, high, low, close, volume FROM market_data_unified
              WHERE symbol='NIFTY50' AND timeframe='day'
              AND date >= ? AND date <= ? ORDER BY date"""
    nd = pd.read_sql(sqld, con, params=("2022-01-01", UNIVERSE_END))
    con.close()
    nd["date"] = pd.to_datetime(nd["date"]).dt.normalize()
    nd = nd.set_index("date")
    n30["adx14"] = adx(n30["high"], n30["low"], n30["close"], 14)
    n30["atr14"] = atr(n30["high"], n30["low"], n30["close"], 14)
    n30["volpct"] = n30["atr14"] / n30["close"] * 100
    nd["ema50d"] = ema(nd["close"], 50)
    nd["ema200d"] = ema(nd["close"], 200)
    nd["above_ema50d"] = nd["close"] > nd["ema50d"]
    nd["above_ema200d"] = nd["close"] > nd["ema200d"]
    nd["vol20"] = nd["close"].pct_change().rolling(20).std() * 100
    return n30, nd


# Regime gates to test
def make_gate(name, n30, nd):
    """Return a function: ts -> bool (trade allowed?)."""
    n30s = n30.copy()
    nd_sorted = nd.sort_index()

    def lookup_30(ts, col):
        idx = n30s.index.searchsorted(pd.Timestamp(ts), side="right") - 1
        if idx < 0: return np.nan
        return n30s.iloc[idx][col]

    def lookup_d(ts, col):
        target = pd.Timestamp(ts).normalize()
        sub = nd_sorted[nd_sorted.index < target]
        if sub.empty: return np.nan
        return sub.iloc[-1][col]

    if name == "G0_no_gate":
        return lambda ts: True
    if name == "G1_vol30m_above_0.20":
        return lambda ts: not pd.isna(lookup_30(ts, "volpct")) and lookup_30(ts, "volpct") > 0.20
    if name == "G2_vol30m_above_0.25":
        return lambda ts: not pd.isna(lookup_30(ts, "volpct")) and lookup_30(ts, "volpct") > 0.25
    if name == "G3_below_ema50d":
        return lambda ts: lookup_d(ts, "above_ema50d") == False
    if name == "G4_below_ema200d":
        return lambda ts: lookup_d(ts, "above_ema200d") == False
    if name == "G5_vol30m_above_0.20_OR_below_ema50d":
        return lambda ts: (not pd.isna(lookup_30(ts, "volpct")) and lookup_30(ts, "volpct") > 0.20) or \
                          (lookup_d(ts, "above_ema50d") == False)
    if name == "G6_dailyvol20_above_0.8":
        return lambda ts: not pd.isna(lookup_d(ts, "vol20")) and lookup_d(ts, "vol20") > 0.8
    raise ValueError(name)


GATES = ["G0_no_gate", "G1_vol30m_above_0.20", "G2_vol30m_above_0.25",
         "G3_below_ema50d", "G4_below_ema200d",
         "G5_vol30m_above_0.20_OR_below_ema50d", "G6_dailyvol20_above_0.8"]


def main():
    log_path = LOG_DIR / "walkforward_regime.log"
    log = log_path.open("w")
    def say(msg: str):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    say("=== WALK-FORWARD with regime gate ===")
    say("Selection rule (top from previous): R4 (trailing Sharpe >= 1.5) and R9 (trailing total ret >= 5)")
    say(f"Gates tested: {GATES}")
    say("")

    # Pre-gen all trades
    universe = list_universe()
    say(f"Phase 1: prep + generate all trades on {len(universe)} stocks...")
    t0 = time.time()
    trades_dfs = {}
    for i, sym in enumerate(universe, 1):
        df = prepare_30min(sym)
        if df.empty: continue
        ss = gen_setups_locked(df)
        rows = []
        for s in ss:
            ex_idx, ex_p, ex_r, held, ret = sim(
                s["df_ref"], SIDE, s["fill_idx"], s["fill_price"],
                s["anchor_low"], s["anchor_high"], EXIT_ID,
            )
            rows.append({
                "entry_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[s["fill_idx"]]),
                "exit_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[ex_idx]),
                "candles_held": held,
                "net_return_pct": ret - COST_PCT,
            })
        if rows:
            tdf = pd.DataFrame(rows); tdf["entry_dt"] = pd.to_datetime(tdf["entry_dt"])
            trades_dfs[sym] = tdf
        if i % 15 == 0: say(f"  {i}/{len(universe)}  ({time.time()-t0:.0f}s)")
    say(f"  Done: {time.time()-t0:.0f}s, {len(trades_dfs)} stocks have trades")
    say("")

    # Load NIFTY
    say("Phase 2: load NIFTY regime indicators...")
    n30, nd = load_nifty()
    say(f"  NIFTY 30m {len(n30)} rows, daily {len(nd)} rows")
    say("")

    # Set up rebalance schedule
    start = pd.Timestamp(UNIVERSE_START); end = pd.Timestamp(UNIVERSE_END)
    rebalance_points = []
    t = start + pd.DateOffset(months=LOOKBACK_MONTHS)
    while t < end:
        rebalance_points.append(t); t = t + pd.DateOffset(months=TRADE_MONTHS)
    oos_y = (end - rebalance_points[0]).days / 365.25

    say(f"OOS period: {rebalance_points[0].date()} -> {end.date()} ({oos_y:.2f} years)")
    say("")

    # Run grid: rules × gates
    rules = ["R4_sh1.5", "R9_totret5", "ALL"]
    out_csv = OUT_DIR / "walkforward_regime_summary.csv"
    fields = ["rule", "gate", "trades", "trades_per_yr", "win_rate", "profit_factor",
              "expectancy_pct", "annual_ret_pct", "sharpe_ann", "max_dd_pct", "calmar"]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    say("=== Walk-forward grid: rules × gates ===")
    rule_gate_results = {}
    for rule in rules:
        for gate in GATES:
            gate_fn = make_gate(gate, n30, nd)
            all_t = []
            prev = set()
            for rp in rebalance_points:
                lb_start = rp - pd.DateOffset(months=LOOKBACK_MONTHS)
                te = min(rp + pd.DateOffset(months=TRADE_MONTHS), end)
                # Score (apply gate to lookback trades too — only trades that
                # would have passed gate count for selection)
                score_rows = []
                for sym, tdf in trades_dfs.items():
                    lb = tdf[(tdf["entry_dt"] >= lb_start) & (tdf["entry_dt"] < rp)]
                    # Apply gate to lookback (consistent with what we'd actually
                    # have done in real time)
                    lb_gated = lb[lb["entry_dt"].apply(gate_fn)]
                    rets = lb_gated["net_return_pct"].tolist()
                    holds = lb_gated["candles_held"].tolist()
                    s = per_stock_score(rets, holds)
                    score_rows.append({"symbol": sym, **s})
                scores = pd.DataFrame(score_rows)
                basket = select_basket(rule, scores, prev)
                # Trade window with gate
                for sym in basket:
                    tdf = trades_dfs.get(sym)
                    if tdf is None: continue
                    qt = tdf[(tdf["entry_dt"] >= rp) & (tdf["entry_dt"] < te)]
                    qt_gated = qt[qt["entry_dt"].apply(gate_fn)]
                    for _, row in qt_gated.iterrows():
                        all_t.append({"symbol": sym, **row.to_dict()})
                prev = set(basket)

            if not all_t:
                say(f"  rule={rule} gate={gate:<35s}: no trades")
                continue
            rets = [t["net_return_pct"] for t in all_t]
            holds = [t["candles_held"] for t in all_t]
            arr = np.array(rets); wins = arr > 0; losses = arr <= 0
            pf = (arr[wins].sum() / -arr[losses].sum()) if losses.any() and arr[losses].sum() < 0 else 99
            mean_r = float(arr.mean()); std_r = float(arr.std(ddof=1)) if len(arr) > 1 else 0
            avg_hold = float(np.mean(holds))
            tpy = 3125 / max(avg_hold, 1)
            sharpe = (mean_r/std_r * math.sqrt(max(tpy,1))) if std_r > 0 else 0
            eq = np.cumsum(arr); peak = np.maximum.accumulate(eq); dd = peak - eq
            max_dd = float(dd.max()) if len(dd) else 0
            annual = arr.sum() / oos_y
            calmar = annual / max_dd if max_dd > 0 else 0
            row = {"rule": rule, "gate": gate, "trades": len(arr),
                   "trades_per_yr": round(len(arr)/oos_y, 0),
                   "win_rate": round(float(wins.mean()*100), 2),
                   "profit_factor": round(min(pf, 99), 3),
                   "expectancy_pct": round(mean_r, 4),
                   "annual_ret_pct": round(annual, 2),
                   "sharpe_ann": round(sharpe, 2),
                   "max_dd_pct": round(max_dd, 2),
                   "calmar": round(calmar, 2)}
            rule_gate_results[(rule, gate)] = row
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            say(f"  rule={rule:<10s} gate={gate:<38s}: trades={row['trades']:>4} "
                f"WR={row['win_rate']:>5}%  PF={row['profit_factor']:>5}  "
                f"Sharpe={row['sharpe_ann']:>5}  DD={row['max_dd_pct']:>5}%  "
                f"AnnRet={row['annual_ret_pct']:>6}%")

    say("")
    say("=== TOP 10 BY OOS SHARPE ===")
    sumdf = pd.DataFrame([v for v in rule_gate_results.values()])
    sumdf = sumdf.sort_values("sharpe_ann", ascending=False)
    say(sumdf.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
