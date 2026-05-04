"""Phase iii + v: alternative selection signals AND regime study.

Phase iii — Walk-forward with alternative selection signals:
  R6  : trailing profit factor >= 1.5 AND >= 5 trades
  R7  : trailing Calmar >= 1.0 AND >= 5 trades
  R8  : trailing raw expectancy >= 0.30% per trade AND >= 5 trades
  R9  : trailing total return > 5% AND >= 5 trades
  R10 : combined — Sharpe >= 1.0 AND PF >= 1.3

Phase v — Regime study:
  Compute NIFTY50 regime indicators (30-min ADX, daily EMA50 state,
  realized vol, EMA20vs50 state) across the full period.
  Attribute each strategy trade to its regime AT ENTRY.
  Bin trades by regime, see if any indicator cleanly separates winning
  vs losing periods.
  If separation exists, run walk-forward with that regime as an on/off gate.

Outputs:
  results/regime_signals_summary.csv   — per-rule + per-regime metrics
  results/regime_attribution.csv       — per-trade with regime tags
  results/regime_curves.png            — equity curves per regime bin
"""

from __future__ import annotations

import csv
import json
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


# ---------------------------------------------------------------------------
# Per-stock score with multiple metrics
# ---------------------------------------------------------------------------

def per_stock_multi_score(rets: list[float], holds: list[int]) -> dict:
    n = len(rets)
    if n == 0:
        return {"trades": 0, "sharpe": 0, "pf": 0, "calmar": 0, "expectancy": 0, "total_ret": 0}
    arr = np.array(rets)
    wins = arr > 0; losses = arr <= 0
    pf = (arr[wins].sum() / -arr[losses].sum()) if losses.any() and arr[losses].sum() < 0 else 99.0
    mean_r = float(arr.mean()); std_r = float(arr.std(ddof=1)) if n > 1 else 0
    avg_hold = float(np.mean(holds))
    tpy = 3125 / max(avg_hold, 1)
    sharpe = (mean_r / std_r * math.sqrt(max(tpy, 1))) if std_r > 0 else 0
    eq = np.cumsum(arr); peak = np.maximum.accumulate(eq); dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0
    annual = arr.sum() * (3125 / max(avg_hold * n, 1))  # rough
    calmar = annual / max_dd if max_dd > 0 else 0
    return {"trades": n, "sharpe": float(sharpe), "pf": float(min(pf, 99.0)),
            "calmar": float(calmar), "expectancy": float(mean_r),
            "total_ret": float(arr.sum())}


def select_basket(rule: str, scores: pd.DataFrame, prev_basket: set[str]) -> list[str]:
    q = scores[scores["trades"] >= MIN_TRADES_LOOKBACK].copy()
    if q.empty: return list(prev_basket)
    if rule == "R6_pf1.5":
        return q[q["pf"] >= 1.5]["symbol"].tolist()
    if rule == "R7_calmar1.0":
        return q[q["calmar"] >= 1.0]["symbol"].tolist()
    if rule == "R8_expect0.3":
        return q[q["expectancy"] >= 0.30]["symbol"].tolist()
    if rule == "R9_totret5":
        return q[q["total_ret"] >= 5]["symbol"].tolist()
    if rule == "R10_sh1.0_pf1.3":
        return q[(q["sharpe"] >= 1.0) & (q["pf"] >= 1.3)]["symbol"].tolist()
    raise ValueError(rule)


# ---------------------------------------------------------------------------
# Regime indicators on NIFTY50
# ---------------------------------------------------------------------------

def load_nifty_regime() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (nifty_30min_df, nifty_daily_df) with regime indicators."""
    con = sqlite3.connect(str(DB_PATH))
    # 5-min for 30-min resample
    sql5 = """SELECT date, open, high, low, close, volume FROM market_data_unified
              WHERE symbol='NIFTY50' AND timeframe='5minute'
              AND date >= ? AND date <= ? ORDER BY date"""
    df5 = pd.read_sql(sql5, con, params=("2023-01-01", UNIVERSE_END + " 23:59:59"))
    df5["date"] = pd.to_datetime(df5["date"])
    n30 = to_30min(df5.set_index("date"))

    # Daily
    sqld = """SELECT date, open, high, low, close, volume FROM market_data_unified
              WHERE symbol='NIFTY50' AND timeframe='day'
              AND date >= ? AND date <= ? ORDER BY date"""
    nd = pd.read_sql(sqld, con, params=("2022-01-01", UNIVERSE_END))
    con.close()
    nd["date"] = pd.to_datetime(nd["date"]).dt.normalize()
    nd = nd.set_index("date")

    # 30-min indicators
    n30["adx14"] = adx(n30["high"], n30["low"], n30["close"], 14)
    n30["ema20"] = ema(n30["close"], 20)
    n30["ema50"] = ema(n30["close"], 50)
    n30["ema_bull"] = (n30["ema20"] > n30["ema50"]).astype(int)
    n30["atr14"] = atr(n30["high"], n30["low"], n30["close"], 14)
    n30["volpct"] = n30["atr14"] / n30["close"] * 100  # volatility as % of price

    # Daily indicators
    nd["ema50d"] = ema(nd["close"], 50)
    nd["ema200d"] = ema(nd["close"], 200)
    nd["above_ema50d"] = (nd["close"] > nd["ema50d"]).astype(int)
    nd["above_ema200d"] = (nd["close"] > nd["ema200d"]).astype(int)
    nd["ret20"] = nd["close"].pct_change(20) * 100  # trailing 20-day return
    nd["vol20"] = nd["close"].pct_change().rolling(20).std() * 100  # daily vol

    return n30.reset_index().rename(columns={"date": "dt"}), nd


def attribute_regime(trades: pd.DataFrame, n30: pd.DataFrame, nd: pd.DataFrame) -> pd.DataFrame:
    """For each trade, look up regime indicators at entry candle."""
    n30 = n30.set_index("dt")
    nd_sorted = nd.sort_index()

    out = trades.copy()
    out["entry_dt"] = pd.to_datetime(out["entry_dt"])

    # 30-min regime values at entry candle (most recent before entry_dt)
    def lookup_30(ts, col):
        # Most recent 30-min candle <= ts
        idx = n30.index.searchsorted(ts, side="right") - 1
        if idx < 0: return np.nan
        return n30.iloc[idx][col]

    def lookup_daily(ts, col):
        # Most recent daily row strictly before ts (avoid same-day look-ahead)
        target_day = pd.Timestamp(ts).normalize()
        sub = nd_sorted[nd_sorted.index < target_day]
        if sub.empty: return np.nan
        return sub.iloc[-1][col]

    for col in ["adx14", "ema_bull", "volpct"]:
        out[f"nifty_30m_{col}"] = out["entry_dt"].apply(lambda t: lookup_30(t, col))

    for col in ["above_ema50d", "above_ema200d", "ret20", "vol20"]:
        out[f"nifty_d_{col}"] = out["entry_dt"].apply(lambda t: lookup_daily(t, col))

    return out


# ---------------------------------------------------------------------------
# Bucketing utilities
# ---------------------------------------------------------------------------

def bucket_metrics(trades: pd.DataFrame, by_col: str, bins: list[float],
                   labels: list[str]) -> pd.DataFrame:
    t = trades.dropna(subset=[by_col]).copy()
    t["bucket"] = pd.cut(t[by_col], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for bucket, sub in t.groupby("bucket", observed=True):
        rets = sub["net_return_pct"].values
        if len(rets) == 0: continue
        wins = rets > 0
        pf = (rets[wins].sum() / -rets[~wins].sum()) if rets[~wins].sum() < 0 else 99
        mean_r = float(rets.mean()); std_r = float(rets.std(ddof=1)) if len(rets) > 1 else 0
        avg_hold = float(sub["candles_held"].mean())
        tpy = 3125 / max(avg_hold, 1)
        sharpe = (mean_r / std_r * math.sqrt(max(tpy, 1))) if std_r > 0 else 0
        rows.append({
            "bucket": str(bucket),
            "trades": len(rets),
            "win_rate": round(float(wins.mean()*100), 2),
            "expectancy": round(mean_r, 4),
            "profit_factor": round(min(pf, 99), 3),
            "sharpe_ann": round(sharpe, 2),
            "total_ret": round(float(rets.sum()), 2),
        })
    return pd.DataFrame(rows)


def main():
    log_path = LOG_DIR / "regime_signals.log"
    log = log_path.open("w")
    def say(msg: str):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    universe = list_universe()
    say("=== PHASE iii + v: Alternative signals + Regime study ===")
    say(f"Period: {UNIVERSE_START} -> {UNIVERSE_END}")
    say(f"Universe: {len(universe)} stocks")
    say("")

    # ---- Phase 1: Pre-generate all trades for 79 stocks ----
    say("Phase 1: prep + generate all trades on 79 stocks (one pass)...")
    t0 = time.time()
    all_trades_full = []
    trades_dfs = {}  # per-stock pre-generated trades for walkforward
    for i, sym in enumerate(universe, 1):
        df = prepare_30min(sym)
        if df.empty: continue
        ss = gen_setups_locked(df)
        sym_trades = []
        for s in ss:
            ex_idx, ex_p, ex_r, held, ret = sim(
                s["df_ref"], SIDE, s["fill_idx"], s["fill_price"],
                s["anchor_low"], s["anchor_high"], EXIT_ID,
            )
            net = ret - COST_PCT
            row = {
                "symbol": sym,
                "signal_dt": s["signal_dt"],
                "entry_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[s["fill_idx"]]),
                "exit_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[ex_idx]),
                "entry_price": s["fill_price"],
                "exit_price": ex_p,
                "exit_reason": ex_r,
                "candles_held": held,
                "net_return_pct": net,
            }
            all_trades_full.append(row)
            sym_trades.append(row)
        if sym_trades:
            trades_dfs[sym] = pd.DataFrame(sym_trades)
        if i % 15 == 0: say(f"  {i}/{len(universe)}  ({time.time()-t0:.0f}s)")
    say(f"Phase 1 done: {len(all_trades_full)} trades, {time.time()-t0:.0f}s")
    all_trades_df = pd.DataFrame(all_trades_full)
    all_trades_df["entry_dt"] = pd.to_datetime(all_trades_df["entry_dt"])
    say("")

    # ---- Phase 2: Load NIFTY regime indicators ----
    say("Phase 2: loading NIFTY50 regime indicators...")
    n30, nd = load_nifty_regime()
    say(f"  NIFTY 30-min rows: {len(n30)}, daily rows: {len(nd)}")

    # ---- Phase 3: Attribute regime to each trade ----
    say("Phase 3: attribute regime to each trade...")
    attributed = attribute_regime(all_trades_df, n30, nd)
    attributed.to_csv(OUT_DIR / "regime_attribution.csv", index=False)
    say(f"  Wrote regime_attribution.csv ({len(attributed)} trades)")

    # ---- Phase 4: Bin and analyze ----
    say("")
    say("=== Phase 4: regime bin analysis ===")

    say("\n--- NIFTY 30-min ADX(14) at entry ---")
    say(bucket_metrics(attributed, "nifty_30m_adx14", [0, 15, 20, 25, 30, 100],
                       ["<15","15-20","20-25","25-30",">30"]).to_string(index=False))

    say("\n--- NIFTY 30-min EMA20 vs EMA50 at entry (bull=1, bear=0) ---")
    say(bucket_metrics(attributed, "nifty_30m_ema_bull", [-0.5, 0.5, 1.5],
                       ["bear","bull"]).to_string(index=False))

    say("\n--- NIFTY 30-min vol% (ATR/close) at entry ---")
    say(bucket_metrics(attributed, "nifty_30m_volpct", [0, 0.10, 0.15, 0.20, 0.30, 1],
                       ["<0.10","0.10-0.15","0.15-0.20","0.20-0.30",">0.30"]).to_string(index=False))

    say("\n--- NIFTY daily close vs daily EMA50 (T-1) ---")
    say(bucket_metrics(attributed, "nifty_d_above_ema50d", [-0.5, 0.5, 1.5],
                       ["below","above"]).to_string(index=False))

    say("\n--- NIFTY daily close vs daily EMA200 (T-1) ---")
    say(bucket_metrics(attributed, "nifty_d_above_ema200d", [-0.5, 0.5, 1.5],
                       ["below","above"]).to_string(index=False))

    say("\n--- NIFTY daily trailing 20-day return % (T-1) ---")
    say(bucket_metrics(attributed, "nifty_d_ret20", [-100, -5, 0, 5, 100],
                       ["<-5","-5to0","0to5",">5"]).to_string(index=False))

    say("\n--- NIFTY daily trailing 20-day vol % (T-1) ---")
    say(bucket_metrics(attributed, "nifty_d_vol20", [0, 0.5, 0.8, 1.2, 100],
                       ["<0.5","0.5-0.8","0.8-1.2",">1.2"]).to_string(index=False))

    # ---- Phase 5: Walk-forward with NEW selection signals ----
    say("")
    say("=== Phase 5: walk-forward with alt selection signals (R6-R10) ===")
    start = pd.Timestamp(UNIVERSE_START); end = pd.Timestamp(UNIVERSE_END)
    rebalance_points = []
    t = start + pd.DateOffset(months=LOOKBACK_MONTHS)
    while t < end:
        rebalance_points.append(t); t = t + pd.DateOffset(months=TRADE_MONTHS)

    NEW_RULES = ["R6_pf1.5", "R7_calmar1.0", "R8_expect0.3", "R9_totret5", "R10_sh1.0_pf1.3"]
    rule_results = {}
    for rule in NEW_RULES:
        all_t = []
        prev = set()
        for rp in rebalance_points:
            lb_start = rp - pd.DateOffset(months=LOOKBACK_MONTHS)
            te = min(rp + pd.DateOffset(months=TRADE_MONTHS), end)
            score_rows = []
            for sym, tdf in trades_dfs.items():
                lb = tdf[(tdf["entry_dt"] >= lb_start) & (tdf["entry_dt"] < rp)]
                rets = lb["net_return_pct"].tolist()
                holds = lb["candles_held"].tolist()
                s = per_stock_multi_score(rets, holds)
                score_rows.append({"symbol": sym, **s})
            scores = pd.DataFrame(score_rows)
            basket = select_basket(rule, scores, prev)
            for sym in basket:
                tdf = trades_dfs.get(sym)
                if tdf is None: continue
                qt = tdf[(tdf["entry_dt"] >= rp) & (tdf["entry_dt"] < te)]
                for _, row in qt.iterrows():
                    all_t.append({"symbol": sym, "rebalance_dt": rp, **row.to_dict()})
            prev = set(basket)
        if not all_t:
            say(f"  {rule}: no trades")
            continue
        rets = [t["net_return_pct"] for t in all_t]
        holds = [t["candles_held"] for t in all_t]
        arr = np.array(rets)
        wins = arr > 0; losses = arr <= 0
        pf = (arr[wins].sum() / -arr[losses].sum()) if losses.any() and arr[losses].sum() < 0 else 99
        mean_r = float(arr.mean()); std_r = float(arr.std(ddof=1)) if len(arr) > 1 else 0
        avg_hold = float(np.mean(holds))
        tpy = 3125 / max(avg_hold, 1)
        sharpe = (mean_r/std_r * math.sqrt(max(tpy,1))) if std_r > 0 else 0
        eq = np.cumsum(arr); peak = np.maximum.accumulate(eq); dd = peak - eq
        max_dd = float(dd.max()) if len(dd) else 0
        oos_y = (end - rebalance_points[0]).days / 365.25
        annual = arr.sum() / oos_y
        calmar = annual / max_dd if max_dd > 0 else 0
        rule_results[rule] = {
            "trades": len(arr), "win_rate": round(float(wins.mean()*100), 2),
            "profit_factor": round(min(pf, 99), 3), "expectancy_pct": round(mean_r, 4),
            "annual_ret_pct": round(annual, 2), "sharpe_ann": round(sharpe, 2),
            "max_dd_pct": round(max_dd, 2), "calmar": round(calmar, 2),
        }
        say(f"  {rule}: trades={len(arr)} WR={rule_results[rule]['win_rate']}% PF={rule_results[rule]['profit_factor']} "
            f"Sharpe={rule_results[rule]['sharpe_ann']} DD={rule_results[rule]['max_dd_pct']}% AnnRet={rule_results[rule]['annual_ret_pct']}%")

    rdf = pd.DataFrame([{"rule": r, **m} for r, m in rule_results.items()])
    rdf = rdf.sort_values("sharpe_ann", ascending=False)
    rdf.to_csv(OUT_DIR / "regime_signals_summary.csv", index=False)
    say("")
    say("=== Alt-signal walk-forward ranking ===")
    say(rdf.to_string(index=False))


if __name__ == "__main__":
    main()
