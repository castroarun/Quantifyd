"""Walk-forward with regime gate — VECTORIZED version (merge_asof).

Same as walkforward_regime.py but pre-attributes regime values to all
trades up front using merge_asof, so the gate check is a fast column
comparison instead of a per-trade lookup.
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


def load_nifty_with_regime():
    con = sqlite3.connect(str(DB_PATH))
    sql5 = """SELECT date, open, high, low, close, volume FROM market_data_unified
              WHERE symbol='NIFTY50' AND timeframe='5minute'
              AND date >= ? AND date <= ? ORDER BY date"""
    df5 = pd.read_sql(sql5, con, params=("2023-01-01", UNIVERSE_END + " 23:59:59"))
    df5["date"] = pd.to_datetime(df5["date"])
    n30 = to_30min(df5.set_index("date"))
    sqld = """SELECT date, close FROM market_data_unified
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
    nd["above_ema50d"] = (nd["close"] > nd["ema50d"]).astype(float)
    nd["above_ema200d"] = (nd["close"] > nd["ema200d"]).astype(float)
    nd["vol20"] = nd["close"].pct_change().rolling(20).std() * 100

    return n30, nd


def attribute_regime_vectorized(trades_df: pd.DataFrame,
                                  n30: pd.DataFrame,
                                  nd: pd.DataFrame) -> pd.DataFrame:
    """Use merge_asof to attribute regime values to each trade's entry_dt."""
    trades_df = trades_df.sort_values("entry_dt").reset_index(drop=True)

    # 30-min: most recent candle <= entry_dt
    n30s = n30.reset_index().rename(columns={"date": "n30_dt"}).sort_values("n30_dt")
    n30s = n30s[["n30_dt", "adx14", "atr14", "volpct"]]
    merged = pd.merge_asof(trades_df, n30s, left_on="entry_dt", right_on="n30_dt",
                            direction="backward")
    merged = merged.rename(columns={"adx14": "n30_adx14",
                                     "atr14": "n30_atr14",
                                     "volpct": "n30_volpct"})

    # Daily: most recent daily row strictly before entry_dt's calendar date
    nds = nd.reset_index().rename(columns={"date": "d_dt"}).sort_values("d_dt")
    nds = nds[["d_dt", "above_ema50d", "above_ema200d", "vol20"]]
    # Use the floor of entry_dt to a day, then look strictly before
    merged["entry_day"] = merged["entry_dt"].dt.normalize()
    merged = pd.merge_asof(merged.sort_values("entry_day"), nds,
                            left_on="entry_day", right_on="d_dt",
                            direction="backward",
                            allow_exact_matches=False)  # strictly before
    merged = merged.rename(columns={"above_ema50d": "d_above_ema50d",
                                     "above_ema200d": "d_above_ema200d",
                                     "vol20": "d_vol20"})
    merged = merged.drop(columns=["n30_dt", "d_dt", "entry_day"])
    return merged


def apply_gate(df: pd.DataFrame, gate: str) -> pd.Series:
    """Return boolean mask: True = trade allowed under this gate."""
    if gate == "G0_no_gate":
        return pd.Series(True, index=df.index)
    if gate == "G1_vol30m_above_0.20":
        return df["n30_volpct"] > 0.20
    if gate == "G2_vol30m_above_0.25":
        return df["n30_volpct"] > 0.25
    if gate == "G3_below_ema50d":
        return df["d_above_ema50d"] == 0
    if gate == "G4_below_ema200d":
        return df["d_above_ema200d"] == 0
    if gate == "G5_vol30m_above_0.20_OR_below_ema50d":
        return (df["n30_volpct"] > 0.20) | (df["d_above_ema50d"] == 0)
    if gate == "G6_dailyvol20_above_0.8":
        return df["d_vol20"] > 0.8
    raise ValueError(gate)


GATES = ["G0_no_gate", "G1_vol30m_above_0.20", "G2_vol30m_above_0.25",
         "G3_below_ema50d", "G4_below_ema200d",
         "G5_vol30m_above_0.20_OR_below_ema50d", "G6_dailyvol20_above_0.8"]


def main():
    log_path = LOG_DIR / "walkforward_regime_v2.log"
    log = log_path.open("w")
    def say(msg: str):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    say("=== WALK-FORWARD with regime gate (vectorized) ===")
    say(f"Gates: {GATES}")
    say("")

    universe = list_universe()
    say(f"Phase 1: prep + generate trades on {len(universe)} stocks...")
    t0 = time.time()
    all_rows = []
    for i, sym in enumerate(universe, 1):
        df = prepare_30min(sym)
        if df.empty: continue
        ss = gen_setups_locked(df)
        for s in ss:
            ex_idx, ex_p, ex_r, held, ret = sim(
                s["df_ref"], SIDE, s["fill_idx"], s["fill_price"],
                s["anchor_low"], s["anchor_high"], EXIT_ID,
            )
            all_rows.append({
                "symbol": sym,
                "entry_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[s["fill_idx"]]),
                "exit_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[ex_idx]),
                "candles_held": held,
                "net_return_pct": ret - COST_PCT,
            })
        if i % 15 == 0: say(f"  {i}/{len(universe)}  ({time.time()-t0:.0f}s)")
    say(f"  Done: {time.time()-t0:.0f}s, {len(all_rows)} trades total")
    say("")

    trades = pd.DataFrame(all_rows)
    trades["entry_dt"] = pd.to_datetime(trades["entry_dt"])

    say("Phase 2: load NIFTY + attribute regime...")
    t0 = time.time()
    n30, nd = load_nifty_with_regime()
    trades = attribute_regime_vectorized(trades, n30, nd)
    say(f"  Done in {time.time()-t0:.1f}s")
    say(f"  Sample trade: {trades.iloc[0].to_dict()}")
    say("")

    # Build per-stock trade groups for fast iteration
    trades_by_sym = {sym: g.sort_values("entry_dt").reset_index(drop=True)
                     for sym, g in trades.groupby("symbol")}

    # Rebalance schedule
    start = pd.Timestamp(UNIVERSE_START); end = pd.Timestamp(UNIVERSE_END)
    rebalance_points = []
    t = start + pd.DateOffset(months=LOOKBACK_MONTHS)
    while t < end:
        rebalance_points.append(t); t = t + pd.DateOffset(months=TRADE_MONTHS)
    oos_y = (end - rebalance_points[0]).days / 365.25
    say(f"OOS period: {rebalance_points[0].date()} -> {end.date()} ({oos_y:.2f}y)")
    say("")

    rules = ["R4_sh1.5", "R9_totret5", "ALL"]
    out_csv = OUT_DIR / "walkforward_regime_summary.csv"
    fields = ["rule", "gate", "trades", "trades_per_yr", "win_rate", "profit_factor",
              "expectancy_pct", "annual_ret_pct", "sharpe_ann", "max_dd_pct", "calmar"]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    say("=== Walk-forward grid: rules × gates ===")
    rows = []
    for rule in rules:
        for gate in GATES:
            t_cell = time.time()
            # Pre-compute gate mask for ALL trades up front
            full_mask = apply_gate(trades, gate)
            gated = trades[full_mask].copy()

            # Walk-forward
            all_t = []
            prev = set()
            for rp in rebalance_points:
                lb_start = rp - pd.DateOffset(months=LOOKBACK_MONTHS)
                te = min(rp + pd.DateOffset(months=TRADE_MONTHS), end)
                # Score on lookback (gated trades only)
                lb_trades = gated[(gated["entry_dt"] >= lb_start) & (gated["entry_dt"] < rp)]
                score_rows = []
                for sym, group in lb_trades.groupby("symbol"):
                    s = per_stock_score(group["net_return_pct"].tolist(),
                                         group["candles_held"].tolist())
                    score_rows.append({"symbol": sym, **s})
                # Add stocks with 0 lookback trades (still in universe)
                seen = {r["symbol"] for r in score_rows}
                for sym in trades_by_sym.keys():
                    if sym not in seen:
                        score_rows.append({"symbol": sym, "trades": 0,
                                           "sharpe": 0, "total_ret": 0,
                                           "expectancy": 0, "pf": 0})
                scores = pd.DataFrame(score_rows)
                basket = select_basket(rule, scores, prev)
                # Trade window: pull from gated trades for the basket
                for sym in basket:
                    sym_trades = gated[(gated["symbol"] == sym)
                                        & (gated["entry_dt"] >= rp)
                                        & (gated["entry_dt"] < te)]
                    for _, row in sym_trades.iterrows():
                        all_t.append({"symbol": sym, **row.to_dict()})
                prev = set(basket)

            if not all_t:
                say(f"  rule={rule:<10s} gate={gate:<38s}: no trades")
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
            rows.append(row)
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            say(f"  rule={rule:<10s} gate={gate:<38s}: trades={row['trades']:>4} "
                f"WR={row['win_rate']:>5}%  PF={row['profit_factor']:>5}  "
                f"Sharpe={row['sharpe_ann']:>5}  DD={row['max_dd_pct']:>5}%  "
                f"AnnRet={row['annual_ret_pct']:>6}%  ({time.time()-t_cell:.1f}s)")

    say("")
    say("=== TOP 10 BY OOS SHARPE ===")
    sumdf = pd.DataFrame(rows).sort_values("sharpe_ann", ascending=False)
    say(sumdf.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
