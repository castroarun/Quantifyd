"""Phase D + E: Stoch parameter sweep + exit sub-variants on the top cell.

D — Vary Stochastics(K, smoothK, smoothD) periods + the long oversold
    threshold (and short overbought = 100 - long_os).
E — Sub-variants on the winning Chandelier exit and combined exits.

For each variant we re-run the entire long+short × E1 sweep on the 10
stocks (the E1 entry filter held up best in C, so we anchor on it).

Output:
  results/param_sweep_results.csv  — per-cell metrics across all variants
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data30 import INTRADAY_STOCKS, BT_START, BT_END, load_30min  # noqa: E402
from indicators_stoch import ema, stochastics, rsi, atr, supertrend  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "results"
LOG_DIR = SCRIPT_DIR.parent / "logs"

ENTRY_VALIDITY_CANDLES = 10
TICK = 0.05
COST_PCT = 0.10
ANCHOR_ENTRY = "E1"  # hard close > EMA20 (long) / close < EMA20 (short)


# =============================================================================
# Indicator caches per (symbol, stoch_params)
# =============================================================================

PRICE_CACHE: dict[str, pd.DataFrame] = {}


def _load_prices(symbol: str) -> pd.DataFrame:
    if symbol in PRICE_CACHE:
        return PRICE_CACHE[symbol]
    df = load_30min(symbol, BT_START, BT_END)
    if df.empty or len(df) < 100:
        PRICE_CACHE[symbol] = df
        return df
    df = df.reset_index().rename(columns={"date": "dt"})
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    st7 = supertrend(df["high"], df["low"], df["close"], 7, 3.0)
    df["st7_3_dir"] = st7["dir"]
    st10 = supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    df["st10_3_dir"] = st10["dir"]
    df["hh22"] = df["high"].rolling(22, min_periods=22).max()
    df["ll22"] = df["low"].rolling(22, min_periods=22).min()
    df["hh15"] = df["high"].rolling(15, min_periods=15).max()
    df["ll15"] = df["low"].rolling(15, min_periods=15).min()
    PRICE_CACHE[symbol] = df
    return df


def _round_tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 2)


# =============================================================================
# Setup gen with parameterized Stoch + thresholds
# =============================================================================

def gen_setups(
    df: pd.DataFrame,
    side: str,
    stoch_k: int,
    stoch_sk: int,
    stoch_sd: int,
    long_os: float,
) -> list[dict]:
    """Generate setups for one symbol+side+Stoch params."""
    short_ob = 100.0 - long_os

    st = stochastics(df["high"], df["low"], df["close"], stoch_k, stoch_sk, stoch_sd)
    df = df.assign(k=st["k"], d=st["d"]).dropna(
        subset=["ema20", "ema50", "k", "d", "atr14", "rsi14",
                "st7_3_dir", "st10_3_dir", "hh22", "ll22", "hh15", "ll15"]
    ).reset_index(drop=True)
    if df.empty:
        return []

    n = len(df)
    e20 = df["ema20"].values; e50 = df["ema50"].values
    k_arr = df["k"].values; d_arr = df["d"].values
    closes = df["close"].values; opens = df["open"].values
    highs = df["high"].values; lows = df["low"].values
    times = df["dt"].values

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

            # E1 hard filter
            on_correct = (closes[j] > e20[j]) if side == "long" else (closes[j] < e20[j])
            if not on_correct:
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


# =============================================================================
# Trade simulator with EXTENDED exit variants
# =============================================================================

EXTENDED_EXITS = [
    "X9_22_3.0",      # default Chandelier — winner from base sweep
    "X9_15_3.0",
    "X9_22_2.0",
    "X9_15_2.0",
    "X9_OR_X4",       # whichever first
    "X9_OR_X10",
    "X4",
    "X10",
]


def sim(df: pd.DataFrame, side: str, fill_idx: int, fill_price: float,
        sl_low: float, sl_high: float, exit_id: str) -> tuple[int, float, str, int, float]:
    """Returns (exit_idx, exit_price, exit_reason, candles_held, return_pct)."""
    n = len(df)
    e20 = df["ema20"].values; e50 = df["ema50"].values
    closes = df["close"].values; highs = df["high"].values; lows = df["low"].values
    rsis = df["rsi14"].values; atrs = df["atr14"].values
    hh22 = df["hh22"].values; ll22 = df["ll22"].values
    hh15 = df["hh15"].values; ll15 = df["ll15"].values

    for t in range(fill_idx + 1, n):
        c = float(closes[t]); h = float(highs[t]); l_ = float(lows[t])
        # X1 always-on
        if (e20[t] <= e50[t]) if side == "long" else (e20[t] >= e50[t]):
            return t, c, "X1", t - fill_idx, _r(side, fill_price, c)

        # exit_id specific
        x = exit_id
        # Chandelier helpers
        if x.startswith("X9_22_"):
            k = float(x.split("_")[2])
            ch = float(hh22[t]) - k*float(atrs[t]) if side == "long" else float(ll22[t]) + k*float(atrs[t])
            if (c < ch) if side == "long" else (c > ch):
                return t, c, x, t - fill_idx, _r(side, fill_price, c)
        elif x.startswith("X9_15_"):
            k = float(x.split("_")[2])
            ch = float(hh15[t]) - k*float(atrs[t]) if side == "long" else float(ll15[t]) + k*float(atrs[t])
            if (c < ch) if side == "long" else (c > ch):
                return t, c, x, t - fill_idx, _r(side, fill_price, c)
        elif x == "X9_OR_X4":
            ch = float(hh22[t]) - 3.0*float(atrs[t]) if side == "long" else float(ll22[t]) + 3.0*float(atrs[t])
            cha = (c < ch) if side == "long" else (c > ch)
            rsi_break = (rsis[t] < 50) if side == "long" else (rsis[t] > 50)
            if cha or rsi_break:
                return t, c, "X9OR4", t - fill_idx, _r(side, fill_price, c)
        elif x == "X9_OR_X10":
            ch = float(hh22[t]) - 3.0*float(atrs[t]) if side == "long" else float(ll22[t]) + 3.0*float(atrs[t])
            cha = (c < ch) if side == "long" else (c > ch)
            ema_break = (c < e20[t]) if side == "long" else (c > e20[t])
            if cha or ema_break:
                return t, c, "X9OR10", t - fill_idx, _r(side, fill_price, c)
        elif x == "X4":
            if (rsis[t] < 50) if side == "long" else (rsis[t] > 50):
                return t, c, "X4", t - fill_idx, _r(side, fill_price, c)
        elif x == "X10":
            if (c < e20[t]) if side == "long" else (c > e20[t]):
                return t, c, "X10", t - fill_idx, _r(side, fill_price, c)

    last = n - 1
    return last, float(closes[last]), "EOD", last - fill_idx, _r(side, fill_price, float(closes[last]))


def _r(side: str, e: float, x: float) -> float:
    return ((x - e) / e * 100.0) if side == "long" else ((e - x) / e * 100.0)


def cell_metrics(rets_arr: list[float], holds: list[int]) -> dict:
    if not rets_arr:
        return {"trades": 0}
    rets = np.array(rets_arr)
    wins = rets > 0; losses = rets <= 0
    pf = (rets[wins].sum() / -rets[losses].sum()) if losses.any() and rets[losses].sum() < 0 else float("inf")
    mean_r = float(rets.mean()); std_r = float(rets.std(ddof=1)) if len(rets) > 1 else 0
    avg_hold = float(np.mean(holds)) if holds else 0
    trades_per_year = 3000 / max(avg_hold, 1)
    sharpe = (mean_r/std_r * math.sqrt(trades_per_year)) if std_r > 0 else 0
    eq = np.cumsum(rets); peak = np.maximum.accumulate(eq); dd = peak - eq
    return {
        "trades": len(rets),
        "win_rate": round(float(wins.mean()*100), 2),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else "inf",
        "total_ret_pct": round(float(rets.sum()), 2),
        "mean_ret_pct": round(mean_r, 4),
        "sharpe_ann": round(sharpe, 2),
        "max_dd_pct": round(float(dd.max()) if len(dd) else 0, 2),
        "avg_hold": round(avg_hold, 1),
    }


# =============================================================================
# Main
# =============================================================================

# D — Stoch parameter grid
STOCH_PARAMS = [
    (14, 5, 3, 30.0),  # base
    (14, 5, 3, 25.0),
    (14, 5, 3, 35.0),
    (14, 5, 3, 20.0),
    (14, 5, 3, 40.0),
    (14, 3, 3, 30.0),
    (21, 5, 5, 30.0),
    (8, 3, 3, 30.0),
    (14, 3, 3, 25.0),
]

# E — exit variants tested per Stoch param set
EXIT_LIST = EXTENDED_EXITS


def main():
    log_path = LOG_DIR / "param_sweep.log"
    log = log_path.open("w")
    def say(msg: str):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    say("=== Phase D + E: Stoch param sweep + exit sub-variants ===")
    say(f"Anchored on entry variant: {ANCHOR_ENTRY}")
    say(f"Stoch param sets: {len(STOCH_PARAMS)}")
    say(f"Exit sub-variants:  {len(EXIT_LIST)}")
    say(f"Sides: long + short (2)")
    say(f"Total cells: {len(STOCH_PARAMS) * len(EXIT_LIST) * 2} = "
        f"{len(STOCH_PARAMS)} stoch × {len(EXIT_LIST)} exits × 2 sides")
    say(f"Cost applied: {COST_PCT}% per trade")
    say("")

    # Pre-cache prices+EMA+ATR per symbol
    say("Loading + caching price data...")
    t0 = time.time()
    for sym in INTRADAY_STOCKS:
        _load_prices(sym)
    say(f"  done in {time.time()-t0:.1f}s")
    say("")

    out_csv = OUT_DIR / "param_sweep_results.csv"
    fieldnames = [
        "side", "stoch_k", "stoch_sk", "stoch_sd", "stoch_os", "exit_id",
        "trades", "win_rate", "profit_factor", "sharpe_ann", "max_dd_pct",
        "total_ret_pct", "mean_ret_pct", "avg_hold",
    ]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    cell_count = 0; t_start = time.time()
    for (sk, ssk, ssd, sos) in STOCH_PARAMS:
        # Generate setups once per (side, stoch params), reuse across exits
        setups_by_side: dict[str, list[tuple]] = {"long": [], "short": []}
        for side in ("long", "short"):
            for sym in INTRADAY_STOCKS:
                df = _load_prices(sym)
                if df.empty: continue
                ss = gen_setups(df, side, sk, ssk, ssd, sos)
                for s in ss:
                    setups_by_side[side].append((sym, s))

        for exit_id in EXIT_LIST:
            for side in ("long", "short"):
                cell_count += 1
                t_cell = time.time()
                rets = []; holds = []
                for sym, s in setups_by_side[side]:
                    df = s["df_ref"]
                    _, _, _, held, ret = sim(
                        df, side, s["fill_idx"], s["fill_price"],
                        s["anchor_low"], s["anchor_high"], exit_id,
                    )
                    rets.append(ret - COST_PCT)
                    holds.append(held)
                m = cell_metrics(rets, holds)
                row = {"side": side, "stoch_k": sk, "stoch_sk": ssk, "stoch_sd": ssd,
                       "stoch_os": sos, "exit_id": exit_id, **m}
                for k in fieldnames:
                    if k not in row: row[k] = ""
                with out_csv.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
                say(f"  [{cell_count:>3}/{len(STOCH_PARAMS)*len(EXIT_LIST)*2}] side={side:<5} "
                    f"stoch=({sk},{ssk},{ssd}) os={sos:>4.0f} exit={exit_id:<12} "
                    f"trades={m.get('trades',0):>4}  Sharpe={m.get('sharpe_ann','-'):>5}  "
                    f"DD={m.get('max_dd_pct','-'):>5}%  TR={m.get('total_ret_pct','-'):>6}%  ({time.time()-t_cell:.1f}s)")

    say("")
    say(f"Sweep done in {time.time()-t_start:.1f}s. Wrote {out_csv}")
    log.close()


if __name__ == "__main__":
    main()
