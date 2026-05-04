"""Final run with the gap=0 fix on the full 79-stock universe.

Locked spec:
  side          = long
  entry filter  = E1 (close > EMA20 at Stoch cross)
  Stoch         = (14, 5, 3)
  Stoch OS      = 35 (touch threshold for "armed")
  GAP FILTER    = 0  (cross MUST fire on the SAME candle as the touch)
  Regime filter = ADX(14) >= 25 at trigger candle (F2)
  Exit          = X9_OR_X4 (Chandelier(22, 3*ATR) OR RSI(14)<50 OR EMA20<EMA50)
  Cost          = 0.10% per trade (round-trip)
  Period        = 2024-03-18 -> 2026-03-12 (24 months, 79 stocks)

Outputs:
  results/final_per_symbol.csv      — per-symbol metrics
  results/final_curated_trades.csv  — every trade in the curated basket
  results/final_summary.json        — headline portfolio metrics
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
from indicators_stoch import ema, stochastics, rsi, atr, supertrend, adx  # noqa: E402
from param_sweep import sim, COST_PCT  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "results"
LOG_DIR = SCRIPT_DIR.parent / "logs"
DB_PATH = SCRIPT_DIR.parent.parent.parent / "backtest_data" / "market_data.db"

UNIVERSE_START = "2024-03-18"
UNIVERSE_END = "2026-03-12"

# Locked params
SIDE = "long"
STOCH_K, STOCH_SK, STOCH_SD = 14, 5, 3
STOCH_OS = 35.0
ADX_MIN = 25.0
MAX_TOUCH_GAP = 0
EXIT_ID = "X9_OR_X4"
ENTRY_VALIDITY_CANDLES = 10
TICK = 0.05

CURATION_SHARPE_THRESHOLD = 1.5  # min per-symbol Sharpe to be in curated basket


def _round_tick(p: float) -> float:
    return round(round(p / TICK) * TICK, 2)


def list_universe() -> list[str]:
    con = sqlite3.connect(str(DB_PATH))
    sql = """
        SELECT symbol FROM market_data_unified
        WHERE timeframe='5minute'
        GROUP BY symbol
        HAVING MIN(date) <= ? AND MAX(date) >= ?
        ORDER BY symbol
    """
    df = pd.read_sql(sql, con, params=(UNIVERSE_START + " 23:59:59",
                                       UNIVERSE_END + " 00:00:00"))
    con.close()
    return [s for s in df["symbol"].tolist() if s not in ("NIFTY50", "BANKNIFTY")]


def load_5min(symbol: str) -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    sql = """
        SELECT date, open, high, low, close, volume
        FROM market_data_unified
        WHERE symbol = ? AND timeframe = '5minute'
          AND date >= ? AND date <= ?
        ORDER BY date
    """
    df = pd.read_sql(sql, con, params=(symbol, UNIVERSE_START, UNIVERSE_END + " 23:59:59"))
    con.close()
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def prepare_30min(symbol: str) -> pd.DataFrame:
    d5 = load_5min(symbol)
    if d5.empty: return pd.DataFrame()
    d30 = to_30min(d5)
    if d30.empty or len(d30) < 100: return pd.DataFrame()
    df = d30.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    df["adx14"] = adx(df["high"], df["low"], df["close"], 14)
    st7 = supertrend(df["high"], df["low"], df["close"], 7, 3.0)
    df["st7_3_dir"] = st7["dir"]
    st10 = supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    df["st10_3_dir"] = st10["dir"]
    df["hh22"] = df["high"].rolling(22, min_periods=22).max()
    df["ll22"] = df["low"].rolling(22, min_periods=22).min()
    df["hh15"] = df["high"].rolling(15, min_periods=15).max()
    df["ll15"] = df["low"].rolling(15, min_periods=15).min()
    return df.reset_index().rename(columns={"date": "dt"})


def gen_setups_locked(df: pd.DataFrame) -> list[dict]:
    """Locked spec setup gen: gap=0, ADX>=25, E1, Stoch(14,5,3) os=35, long."""
    st = stochastics(df["high"], df["low"], df["close"], STOCH_K, STOCH_SK, STOCH_SD)
    df = df.assign(k=st["k"], d=st["d"]).dropna(
        subset=["ema20", "ema50", "k", "d", "atr14", "adx14",
                "st7_3_dir", "st10_3_dir", "hh22", "ll22", "hh15", "ll15"]
    ).reset_index(drop=True)
    if df.empty: return []
    n = len(df)
    e20 = df["ema20"].values; e50 = df["ema50"].values
    k_arr = df["k"].values; d_arr = df["d"].values
    closes = df["close"].values; opens = df["open"].values
    highs = df["high"].values; lows = df["low"].values
    adx_arr = df["adx14"].values

    bias = e20 > e50
    period_starts = []
    in_p = False
    for i in range(n):
        if bias[i] and not in_p: period_starts.append(i); in_p = True
        elif not bias[i] and in_p: in_p = False

    setups: list[dict] = []
    for ema_cross in period_starts:
        end = ema_cross
        while end < n and bias[end]: end += 1
        last_touch = None
        armed = False
        for j in range(ema_cross, end):
            if min(k_arr[j], d_arr[j]) <= STOCH_OS:
                armed = True; last_touch = j
            cross = j > 0 and k_arr[j-1] <= d_arr[j-1] and k_arr[j] > d_arr[j]
            if not (armed and cross): continue
            # Gap filter (gap=0: touch must be on the SAME candle as cross)
            gap = j - last_touch
            if gap > MAX_TOUCH_GAP:
                armed = False; continue
            # E1 filter
            if not (closes[j] > e20[j]):
                armed = False; continue
            # ADX filter
            if adx_arr[j] < ADX_MIN:
                armed = False; continue

            anchor_high = float(highs[j]); anchor_low = float(lows[j])
            trigger = _round_tick(anchor_high + TICK)
            window_end = min(j + ENTRY_VALIDITY_CANDLES, n - 1)
            filled = False; fill_idx = None; fill_price = None
            for t in range(j+1, window_end+1):
                if e20[t] <= e50[t]: break
                if opens[t] >= trigger:
                    fill_price = float(opens[t]); fill_idx = t; filled = True; break
                if highs[t] >= trigger:
                    fill_price = float(trigger); fill_idx = t; filled = True; break
            armed = False
            if filled:
                setups.append({
                    "fill_idx": fill_idx, "fill_price": fill_price,
                    "anchor_low": anchor_low, "anchor_high": anchor_high,
                    "df_ref": df,
                    "signal_dt": df["dt"].iloc[j],
                    "signal_k": float(k_arr[j]), "signal_d": float(d_arr[j]),
                    "signal_adx": float(adx_arr[j]),
                })
    return setups


def portfolio_metrics(rets: list[float], holds: list[int], period_years: float) -> dict:
    if not rets: return {"trades": 0}
    arr = np.array(rets)
    wins = arr > 0
    losses = arr <= 0
    n = len(arr)
    pf = (arr[wins].sum() / -arr[losses].sum()) if losses.any() and arr[losses].sum() < 0 else float("inf")
    mean_r = float(arr.mean()); std_r = float(arr.std(ddof=1)) if n > 1 else 0
    avg_hold = float(np.mean(holds)) if holds else 0
    candles_per_year = 3125
    tpy_indicator = candles_per_year / max(avg_hold, 1)
    sharpe = (mean_r / std_r * math.sqrt(max(tpy_indicator, 1))) if std_r > 0 else 0
    down = arr[arr < 0]
    sortino = (mean_r / down.std(ddof=1) * math.sqrt(max(tpy_indicator, 1))) if down.size > 1 else 0
    eq = np.cumsum(arr); peak = np.maximum.accumulate(eq); dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0
    total = float(arr.sum())
    annual = total / period_years
    calmar = annual / max_dd if max_dd > 0 else float("inf")
    return {
        "trades": n,
        "trades_per_yr": round(n / period_years, 0),
        "win_rate": round(float(wins.mean()*100), 2),
        "avg_win_pct": round(float(arr[wins].mean()), 4) if wins.any() else 0,
        "avg_loss_pct": round(float(arr[losses].mean()), 4) if losses.any() else 0,
        "profit_factor": round(pf, 3) if math.isfinite(pf) else "inf",
        "expectancy_pct": round(mean_r, 4),
        "annual_ret_pct": round(annual, 2),
        "total_ret_pct": round(total, 2),
        "sharpe_ann": round(sharpe, 2),
        "sortino_ann": round(sortino, 2),
        "max_dd_pct": round(max_dd, 2),
        "calmar": round(calmar, 2) if math.isfinite(calmar) else "inf",
        "avg_hold": round(avg_hold, 1),
    }


def main():
    log_path = LOG_DIR / "final_run.log"
    log = log_path.open("w")
    def say(msg: str):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    universe = list_universe()
    period_years = (pd.Timestamp(UNIVERSE_END) - pd.Timestamp(UNIVERSE_START)).days / 365.25

    say("=== FINAL RUN — gap=0 + ADX>=25 + locked params on 79 stocks ===")
    say(f"Period: {UNIVERSE_START} -> {UNIVERSE_END} ({period_years:.2f} years)")
    say(f"Universe: {len(universe)} symbols")
    say(f"Cost: {COST_PCT}% per trade")
    say(f"Locked: side={SIDE}, Stoch({STOCH_K},{STOCH_SK},{STOCH_SD}) os={STOCH_OS}, "
        f"E1+ADX>={ADX_MIN}, gap<={MAX_TOUCH_GAP}, exit={EXIT_ID}")
    say("")

    say("Preparing 79 symbols...")
    t0 = time.time()
    prepped: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(universe, 1):
        df = prepare_30min(sym)
        if not df.empty: prepped[sym] = df
        if i % 10 == 0:
            say(f"  {i}/{len(universe)}  ({time.time()-t0:.0f}s)")
    say(f"Prepared {len(prepped)} symbols in {time.time()-t0:.0f}s")
    say("")

    # ---- Per-symbol simulation ----
    say("Simulating each stock with locked spec...")
    per_sym_metrics: dict[str, dict] = {}
    all_trades: list[dict] = []
    for sym, df in prepped.items():
        ss = gen_setups_locked(df)
        sym_rets, sym_holds = [], []
        sym_trades_full = []
        for s in ss:
            ex_idx, ex_p, ex_r, held, ret = sim(
                s["df_ref"], SIDE, s["fill_idx"], s["fill_price"],
                s["anchor_low"], s["anchor_high"], EXIT_ID,
            )
            net_ret = ret - COST_PCT
            sym_rets.append(net_ret); sym_holds.append(held)
            sym_trades_full.append({
                "symbol": sym,
                "signal_dt": s["signal_dt"],
                "signal_k": s["signal_k"], "signal_d": s["signal_d"],
                "signal_adx": s["signal_adx"],
                "entry_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[s["fill_idx"]]),
                "entry_price": s["fill_price"],
                "exit_dt": pd.Timestamp(s["df_ref"]["dt"].iloc[ex_idx]),
                "exit_price": ex_p,
                "exit_reason": ex_r,
                "candles_held": held,
                "gross_return_pct": round(ret, 4),
                "net_return_pct": round(net_ret, 4),
            })
        if sym_rets:
            per_sym_metrics[sym] = portfolio_metrics(sym_rets, sym_holds, period_years)
            all_trades.extend(sym_trades_full)

    say(f"Symbols with at least one trade: {len(per_sym_metrics)}")
    say(f"Total trades across all 79: {len(all_trades)}")
    say("")

    # Per-symbol CSV
    psm = pd.DataFrame([{"symbol": s, **m} for s, m in per_sym_metrics.items()])
    psm = psm.sort_values("sharpe_ann", ascending=False)
    psm.to_csv(OUT_DIR / "final_per_symbol.csv", index=False)
    say(f"Wrote {OUT_DIR / 'final_per_symbol.csv'}")

    # Distribution
    say("")
    say("=== Per-symbol Sharpe distribution ===")
    say(f"  Sharpe > 2.0 : {(psm['sharpe_ann']>2.0).sum()}")
    say(f"  Sharpe > 1.5 : {(psm['sharpe_ann']>1.5).sum()}")
    say(f"  Sharpe > 1.0 : {(psm['sharpe_ann']>1.0).sum()}")
    say(f"  Sharpe > 0.5 : {(psm['sharpe_ann']>0.5).sum()}")
    say(f"  Sharpe > 0   : {(psm['sharpe_ann']>0).sum()}")
    say(f"  Sharpe < 0   : {(psm['sharpe_ann']<0).sum()}")
    say(f"  No trades    : {len(prepped) - len(per_sym_metrics)}")
    say("")
    say("=== Top 25 symbols by per-symbol Sharpe ===")
    say(psm.head(25)[["symbol","trades","trades_per_yr","win_rate","profit_factor",
                     "expectancy_pct","annual_ret_pct","sharpe_ann","calmar","max_dd_pct"]].to_string(index=False))
    say("")
    say("=== Bottom 10 symbols ===")
    say(psm.tail(10)[["symbol","trades","trades_per_yr","win_rate","profit_factor",
                     "expectancy_pct","annual_ret_pct","sharpe_ann","calmar","max_dd_pct"]].to_string(index=False))
    say("")

    # ---- Aggregate (no curation) ----
    all_rets = [t["net_return_pct"] for t in all_trades]
    all_holds = [t["candles_held"] for t in all_trades]
    say("=== AGGREGATE — all 79 stocks (equal-risk per trade) ===")
    agg = portfolio_metrics(all_rets, all_holds, period_years)
    for k, v in agg.items():
        say(f"  {k:>20s}: {v}")
    say("")

    # ---- Curated basket: stocks with per-symbol Sharpe >= threshold ----
    curated = psm[psm["sharpe_ann"] >= CURATION_SHARPE_THRESHOLD]["symbol"].tolist()
    say(f"=== CURATED BASKET (per-symbol Sharpe >= {CURATION_SHARPE_THRESHOLD}) ===")
    say(f"  {len(curated)} stocks: {curated}")
    cur_trades = [t for t in all_trades if t["symbol"] in curated]
    cur_rets = [t["net_return_pct"] for t in cur_trades]
    cur_holds = [t["candles_held"] for t in cur_trades]
    cur_metrics = portfolio_metrics(cur_rets, cur_holds, period_years)
    for k, v in cur_metrics.items():
        say(f"  {k:>20s}: {v}")
    say("")

    # Dump curated basket trades
    cur_df = pd.DataFrame(cur_trades).sort_values("entry_dt", ascending=False)
    cur_df.to_csv(OUT_DIR / "final_curated_trades.csv", index=False)
    say(f"Wrote {OUT_DIR / 'final_curated_trades.csv'}  ({len(cur_df)} trades)")

    # Summary JSON
    summary = {
        "period_years": round(period_years, 2),
        "universe_size": len(prepped),
        "spec": {
            "side": SIDE,
            "stoch": [STOCH_K, STOCH_SK, STOCH_SD, STOCH_OS],
            "max_touch_gap": MAX_TOUCH_GAP,
            "adx_min": ADX_MIN,
            "exit_id": EXIT_ID,
            "cost_pct": COST_PCT,
        },
        "aggregate_79": agg,
        "curated_basket": {
            "sharpe_threshold": CURATION_SHARPE_THRESHOLD,
            "stocks": curated,
            "n_stocks": len(curated),
            "metrics": cur_metrics,
        },
    }
    with (OUT_DIR / "final_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    say(f"Wrote {OUT_DIR / 'final_summary.json'}")

    # Most-recent trade per curated stock for verification
    say("")
    say("=== MOST RECENT TRADE per curated stock (for verification) ===")
    latest = cur_df.groupby("symbol").head(1).sort_values("entry_dt", ascending=False)
    latest["touch_cross_gap"] = 0  # by definition (locked spec)
    cols = ["symbol", "signal_dt", "signal_k", "signal_d", "signal_adx",
            "entry_dt", "entry_price", "exit_dt", "exit_price", "exit_reason",
            "candles_held", "net_return_pct"]
    say(latest[cols].to_string(index=False))


if __name__ == "__main__":
    main()
