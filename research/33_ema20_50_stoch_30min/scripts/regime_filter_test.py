"""Regime filter test on the winning cell across 79-stock universe.

Anchor cell:
  side=long, entry=E1, stoch=(14,5,3) os=35, exit=X9_OR_X4

Regime filters tested (each applied at the trigger candle):
  F0 — no filter (baseline = Phase A result)
  F1 — 30-min ADX(14) >= 20 at trigger
  F2 — 30-min ADX(14) >= 25 at trigger
  F3 — daily close > daily EMA200 at T-1
  F4 — daily EMA200 5-day slope > 0 at T-1
  F5 — F1 AND F3 (combined)
  F6 — F2 AND F3 (stricter combined)
  F7 — F2 AND F4 (ADX strong + daily slope up)

Period: 2024-03-18 -> 2026-03-12 (24 months, 79 stocks)
Cost:   0.10% per trade (round-trip)

Output: results/regime_filter_results.csv  +  per-symbol breakdown for best filter
"""

from __future__ import annotations

import csv
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
from param_sweep import sim, cell_metrics, COST_PCT  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "results"
LOG_DIR = SCRIPT_DIR.parent / "logs"
DB_PATH = SCRIPT_DIR.parent.parent.parent / "backtest_data" / "market_data.db"

UNIVERSE_START = "2024-03-18"
UNIVERSE_END = "2026-03-12"

# Anchor cell
SIDE = "long"
STOCH_K, STOCH_SK, STOCH_SD = 14, 5, 3
STOCH_OS = 35.0
EXIT_ID = "X9_OR_X4"

ENTRY_VALIDITY_CANDLES = 10
TICK = 0.05

FILTERS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"]


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
    syms = [s for s in df["symbol"].tolist() if s not in ("NIFTY50", "BANKNIFTY")]
    return syms


def load_5min(symbol: str) -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    sql = """
        SELECT date, open, high, low, close, volume
        FROM market_data_unified
        WHERE symbol = ? AND timeframe = '5minute'
          AND date >= ? AND date <= ?
        ORDER BY date
    """
    df5 = pd.read_sql(sql, con, params=(symbol, UNIVERSE_START,
                                        UNIVERSE_END + " 23:59:59"))
    con.close()
    if df5.empty:
        return df5
    df5["date"] = pd.to_datetime(df5["date"])
    return df5.set_index("date")


def load_daily(symbol: str) -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    # Pull daily back further to allow EMA200 warmup
    sql = """
        SELECT date, open, high, low, close, volume
        FROM market_data_unified
        WHERE symbol = ? AND timeframe = 'day'
          AND date >= ? AND date <= ?
        ORDER BY date
    """
    df = pd.read_sql(sql, con, params=(symbol, "2023-01-01", UNIVERSE_END))
    con.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")
    return df


def prepare_30min(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df_30min_with_indicators, df_daily_filter_table).

    df_daily_filter_table has columns:
      - day (index, normalized)
      - close_above_ema200 (bool)
      - ema200_slope_pos (bool)
    """
    d5 = load_5min(symbol)
    if d5.empty:
        return pd.DataFrame(), pd.DataFrame()
    d30 = to_30min(d5)
    if d30.empty or len(d30) < 100:
        return pd.DataFrame(), pd.DataFrame()

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

    # Daily filter table
    daily = load_daily(symbol)
    if daily.empty or len(daily) < 200:
        # No daily data — produce empty filter table; F3/F4 will pass through as False
        daily_filter = pd.DataFrame()
    else:
        daily_filter = pd.DataFrame(index=daily.index)
        d_ema200 = ema(daily["close"], 200)
        daily_filter["close_above_ema200"] = (daily["close"] > d_ema200)
        daily_filter["ema200_slope_pos"] = (d_ema200 - d_ema200.shift(5)) > 0
        daily_filter = daily_filter.dropna()

    return df.reset_index().rename(columns={"date": "dt"}), daily_filter


def _round_tick(p: float) -> float:
    return round(round(p / TICK) * TICK, 2)


def _daily_filter_at(daily_filter: pd.DataFrame, ts: pd.Timestamp,
                     col: str) -> bool:
    """Look up the filter value for the most recent calendar day BEFORE ts."""
    if daily_filter.empty:
        return False
    target_day = pd.Timestamp(ts).normalize()
    # Most recent daily row strictly before target_day
    sub = daily_filter[daily_filter.index < target_day]
    if sub.empty:
        return False
    return bool(sub[col].iloc[-1])


def passes_filter(filter_id: str, df: pd.DataFrame, daily_filter: pd.DataFrame,
                  trigger_idx: int) -> bool:
    """Check if the trigger candle passes the filter."""
    ts = df["dt"].iloc[trigger_idx]
    adx_val = float(df["adx14"].iloc[trigger_idx])
    above_d = _daily_filter_at(daily_filter, ts, "close_above_ema200") if not daily_filter.empty else False
    slope_pos = _daily_filter_at(daily_filter, ts, "ema200_slope_pos") if not daily_filter.empty else False

    if filter_id == "F0": return True
    if filter_id == "F1": return adx_val >= 20
    if filter_id == "F2": return adx_val >= 25
    if filter_id == "F3": return above_d
    if filter_id == "F4": return slope_pos
    if filter_id == "F5": return adx_val >= 20 and above_d
    if filter_id == "F6": return adx_val >= 25 and above_d
    if filter_id == "F7": return adx_val >= 25 and slope_pos
    raise ValueError(filter_id)


def gen_setups(df: pd.DataFrame, daily_filter: pd.DataFrame, side: str,
               filter_id: str) -> list[dict]:
    st = stochastics(df["high"], df["low"], df["close"], STOCH_K, STOCH_SK, STOCH_SD)
    df = df.assign(k=st["k"], d=st["d"]).dropna(
        subset=["ema20", "ema50", "k", "d", "atr14", "rsi14", "adx14",
                "st7_3_dir", "st10_3_dir", "hh22", "ll22", "hh15", "ll15"]
    ).reset_index(drop=True)
    if df.empty:
        return []

    n = len(df)
    e20 = df["ema20"].values; e50 = df["ema50"].values
    k_arr = df["k"].values; d_arr = df["d"].values
    closes = df["close"].values; opens = df["open"].values
    highs = df["high"].values; lows = df["low"].values

    long_os = STOCH_OS
    short_ob = 100.0 - long_os

    bias = (e20 > e50) if side == "long" else (e20 < e50)
    period_starts = []
    in_period = False
    for i in range(n):
        if bias[i] and not in_period:
            period_starts.append(i); in_period = True
        elif not bias[i] and in_period:
            in_period = False

    setups: list[dict] = []
    for ema_cross_idx in period_starts:
        j_end = ema_cross_idx
        while j_end < n and bias[j_end]:
            j_end += 1
        armed = False
        for j in range(ema_cross_idx, j_end):
            if side == "long":
                if min(k_arr[j], d_arr[j]) <= long_os:
                    armed = True
                cross_event = (
                    armed and j > 0
                    and k_arr[j-1] <= d_arr[j-1] and k_arr[j] > d_arr[j]
                )
            else:
                if max(k_arr[j], d_arr[j]) >= short_ob:
                    armed = True
                cross_event = (
                    armed and j > 0
                    and k_arr[j-1] >= d_arr[j-1] and k_arr[j] < d_arr[j]
                )
            if not cross_event:
                continue

            on_correct = (closes[j] > e20[j]) if side == "long" else (closes[j] < e20[j])
            if not on_correct:
                armed = False; continue

            # Apply regime filter at trigger candle (j)
            if not passes_filter(filter_id, df, daily_filter, j):
                armed = False; continue

            anchor_high = float(highs[j]); anchor_low = float(lows[j])
            trigger_price = _round_tick(anchor_high + TICK) if side == "long" else _round_tick(anchor_low - TICK)
            end = min(j + ENTRY_VALIDITY_CANDLES, n - 1)
            filled = False; fill_idx = None; fill_price = None
            for t in range(j+1, end+1):
                if (e20[t] <= e50[t]) if side == "long" else (e20[t] >= e50[t]): break
                if side == "long":
                    if opens[t] >= trigger_price:
                        fill_price = float(opens[t]); fill_idx = t; filled = True; break
                    if highs[t] >= trigger_price:
                        fill_price = float(trigger_price); fill_idx = t; filled = True; break
                else:
                    if opens[t] <= trigger_price:
                        fill_price = float(opens[t]); fill_idx = t; filled = True; break
                    if lows[t] <= trigger_price:
                        fill_price = float(trigger_price); fill_idx = t; filled = True; break
            armed = False
            if filled:
                setups.append({
                    "fill_idx": fill_idx, "fill_price": fill_price,
                    "anchor_low": anchor_low, "anchor_high": anchor_high,
                    "df_ref": df,
                })
    return setups


def main():
    log_path = LOG_DIR / "regime_filter.log"
    log = log_path.open("w")
    def say(msg: str):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    universe = list_universe()
    say(f"=== Regime filter test (option 2) ===")
    say(f"Period: {UNIVERSE_START} -> {UNIVERSE_END}")
    say(f"Universe: {len(universe)} symbols")
    say(f"Anchor cell: side={SIDE} stoch=({STOCH_K},{STOCH_SK},{STOCH_SD}) os={STOCH_OS} exit={EXIT_ID}")
    say(f"Filters: {FILTERS}")
    say(f"Cost: {COST_PCT}%")
    say("")

    say("Loading + preparing all symbols (with daily filter table)...")
    t0 = time.time()
    prepped: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for i, sym in enumerate(universe, 1):
        df30, df_daily = prepare_30min(sym)
        if not df30.empty:
            prepped[sym] = (df30, df_daily)
        if i % 10 == 0:
            say(f"  {i}/{len(universe)}  ({time.time()-t0:.1f}s)")
    say(f"Prepared {len(prepped)} symbols in {time.time()-t0:.1f}s")
    say("")

    out_csv = OUT_DIR / "regime_filter_results.csv"
    fieldnames = ["filter", "n_symbols", "trades", "win_rate", "profit_factor",
                  "sharpe_ann", "max_dd_pct", "total_ret_pct", "mean_ret_pct", "avg_hold"]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    per_sym_by_filter: dict[str, list[dict]] = {}

    for filter_id in FILTERS:
        say(f"--- Filter {filter_id} ---")
        t_f = time.time()
        rets, holds = [], []
        sym_metrics: dict[str, dict] = {}
        for sym, (df30, df_daily) in prepped.items():
            ss = gen_setups(df30, df_daily, SIDE, filter_id)
            sym_rets, sym_holds = [], []
            for s in ss:
                _, _, _, held, ret = sim(
                    s["df_ref"], SIDE, s["fill_idx"], s["fill_price"],
                    s["anchor_low"], s["anchor_high"], EXIT_ID,
                )
                sym_rets.append(ret - COST_PCT); sym_holds.append(held)
            rets.extend(sym_rets); holds.extend(sym_holds)
            if sym_rets:
                sym_metrics[sym] = cell_metrics(sym_rets, sym_holds)

        m = cell_metrics(rets, holds)
        say(f"  trades={m.get('trades',0)} syms={len(sym_metrics)} "
            f"WR={m.get('win_rate','-')}%  PF={m.get('profit_factor','-')}  "
            f"Sharpe={m.get('sharpe_ann','-')}  DD={m.get('max_dd_pct','-')}%  "
            f"TR={m.get('total_ret_pct','-')}%  ({time.time()-t_f:.1f}s)")

        row = {"filter": filter_id, "n_symbols": len(sym_metrics), **m}
        for k in fieldnames:
            if k not in row: row[k] = ""
        with out_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        per_sym_by_filter[filter_id] = [
            {"symbol": sym, **mm} for sym, mm in sym_metrics.items()
        ]

    say("")
    say("=== FILTER RANKING (by aggregate Sharpe) ===")
    summary = pd.read_csv(out_csv).sort_values("sharpe_ann", ascending=False)
    say(summary.to_string(index=False))
    say("")

    # Surface per-symbol breakdown for the best filter
    best_filter = summary.iloc[0]["filter"]
    say(f"=== Per-symbol breakdown for best filter ({best_filter}) ===")
    psdf = pd.DataFrame(per_sym_by_filter[best_filter]).sort_values("sharpe_ann", ascending=False)
    psdf.to_csv(OUT_DIR / f"regime_filter_per_symbol_{best_filter}.csv", index=False)
    say(f"Wrote per-symbol breakdown for {best_filter}")
    say("")
    say("Top 15 symbols under best filter:")
    say(psdf.head(15)[["symbol","trades","win_rate","profit_factor","sharpe_ann","max_dd_pct","total_ret_pct"]].to_string(index=False))
    say("")
    say("Bottom 10 symbols under best filter:")
    say(psdf.tail(10)[["symbol","trades","win_rate","profit_factor","sharpe_ann","max_dd_pct","total_ret_pct"]].to_string(index=False))
    say("")
    say(f"Sharpe distribution under best filter ({best_filter}):")
    say(f"  > 1.0: {(psdf['sharpe_ann']>1.0).sum()} symbols")
    say(f"  > 0.5: {(psdf['sharpe_ann']>0.5).sum()} symbols")
    say(f"  > 0.0: {(psdf['sharpe_ann']>0.0).sum()} symbols")
    say(f"  < 0  : {(psdf['sharpe_ann']<0).sum()} symbols")

    log.close()


if __name__ == "__main__":
    main()
