"""Quick comparison of touch-to-cross gap filter values on the winning cell.

Anchor: LONG, E1 (close > EMA20 at cross), Stoch(14,5,3) os=35,
        X9_OR_X4 exit, ADX(14)>=25 regime filter, NET 0.10% costs

Universe: 10 long-history stocks (2024-01 → 2026-03)

Variants tested:
  Gap = 0    — cross must fire on the SAME candle as the touch
  Gap <= 1   — cross within 30 min of touch
  Gap <= 2   — cross within 1 hour
  Gap <= 3   — cross within 1.5 hours
  Gap <= 5   — cross within 2.5 hours
  Gap <= 8   — cross within 4 hours (~half a session)
  No filter  — current behavior (any gap)

Output: results/gap_filter_results.csv + verbose log
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
from indicators_stoch import ema, stochastics, rsi, atr, supertrend, adx  # noqa: E402
from param_sweep import sim, cell_metrics, COST_PCT  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "results"
LOG_DIR = SCRIPT_DIR.parent / "logs"

ENTRY_VALIDITY_CANDLES = 10
TICK = 0.05
SIDE = "long"
STOCH_K, STOCH_SK, STOCH_SD = 14, 5, 3
STOCH_OS = 35.0
ADX_MIN = 25.0
EXIT_ID = "X9_OR_X4"

# Gap filter values to test (None = no filter)
GAP_VALUES = [0, 1, 2, 3, 5, 8, None]


def _round_tick(p: float) -> float:
    return round(round(p / TICK) * TICK, 2)


def prepare_30min(symbol: str) -> pd.DataFrame:
    df = load_30min(symbol, BT_START, BT_END)
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    df = df.copy()
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


def gen_setups_with_gap(df: pd.DataFrame, max_gap: int | None) -> list[dict]:
    st = stochastics(df["high"], df["low"], df["close"], STOCH_K, STOCH_SK, STOCH_SD)
    df = df.assign(k=st["k"], d=st["d"]).dropna(
        subset=["ema20", "ema50", "k", "d", "atr14", "adx14",
                "st7_3_dir", "st10_3_dir", "hh22", "ll22", "hh15", "ll15"]
    ).reset_index(drop=True)
    if df.empty:
        return []

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
        if bias[i] and not in_p:
            period_starts.append(i); in_p = True
        elif not bias[i] and in_p:
            in_p = False

    setups: list[dict] = []
    for ema_cross in period_starts:
        end = ema_cross
        while end < n and bias[end]:
            end += 1
        armed = False; last_touch = None
        for j in range(ema_cross, end):
            if min(k_arr[j], d_arr[j]) <= STOCH_OS:
                armed = True; last_touch = j
            cross = j > 0 and k_arr[j-1] <= d_arr[j-1] and k_arr[j] > d_arr[j]
            if not (armed and cross):
                continue
            # E1 filter
            if not (closes[j] > e20[j]):
                armed = False; continue
            # ADX filter (F2)
            if adx_arr[j] < ADX_MIN:
                armed = False; continue
            # GAP FILTER
            gap = j - last_touch if last_touch is not None else 999
            if max_gap is not None and gap > max_gap:
                # Stale touch — leave armed (next touch can re-arm), skip this fire
                # Actually since the cross has fired and stoch is now neutral,
                # the natural next setup requires a new ≤35 touch. Reset armed=False
                # so we re-arm only on a new dip.
                armed = False
                continue

            # Place trigger
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
                    "df_ref": df, "gap": gap,
                })
    return setups


def main():
    log_path = LOG_DIR / "gap_filter.log"
    log = log_path.open("w")
    def say(msg: str):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    say("=== Touch-to-cross gap filter test (option B) ===")
    say(f"Anchor cell: LONG, E1, Stoch({STOCH_K},{STOCH_SK},{STOCH_SD}) os={STOCH_OS}, "
        f"ADX>={ADX_MIN}, Exit={EXIT_ID}")
    say(f"Universe: 10 long-history stocks ({BT_START} -> {BT_END})")
    say(f"Cost: {COST_PCT}% per trade")
    say("")

    say("Preparing 10 symbols...")
    t0 = time.time()
    prepped = {}
    for sym in INTRADAY_STOCKS:
        prepped[sym] = prepare_30min(sym)
    say(f"Prepared in {time.time()-t0:.1f}s")
    say("")

    out_csv = OUT_DIR / "gap_filter_results.csv"
    fields = ["max_gap", "trades", "trades_per_yr", "win_rate", "profit_factor",
              "expectancy_pct", "annual_ret_pct", "sharpe_ann", "sortino_ann",
              "calmar", "max_dd_pct", "avg_hold", "avg_gap"]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    period_years = (pd.Timestamp(BT_END) - pd.Timestamp(BT_START)).days / 365.25
    say(f"Period: {period_years:.2f} years")
    say("")

    rows = []
    for max_gap in GAP_VALUES:
        label = "no-filter" if max_gap is None else f"gap<={max_gap}"
        rets = []; holds = []; gaps = []
        for sym, df in prepped.items():
            ss = gen_setups_with_gap(df, max_gap)
            for s in ss:
                _, _, _, held, ret = sim(
                    s["df_ref"], SIDE, s["fill_idx"], s["fill_price"],
                    s["anchor_low"], s["anchor_high"], EXIT_ID,
                )
                rets.append(ret - COST_PCT); holds.append(held); gaps.append(s["gap"])

        if not rets:
            say(f"{label:>12s}: no trades")
            continue

        m = cell_metrics(rets, holds)
        # Add expectancy + annual + Calmar
        rets_arr = np.array(rets)
        total = rets_arr.sum()
        annual_ret = total / period_years
        max_dd = m["max_dd_pct"]
        calmar = annual_ret / max_dd if max_dd > 0 else float("inf")

        # Sortino
        down = rets_arr[rets_arr < 0]
        avg_hold = m.get("avg_hold", 1)
        candles_per_year = 3125
        trades_per_year = candles_per_year / max(avg_hold, 1)
        if down.size > 1:
            sortino = (rets_arr.mean() / down.std(ddof=1)) * math.sqrt(max(trades_per_year, 1))
        else:
            sortino = 0

        actual_tr_per_yr = len(rets) / period_years
        avg_gap = np.mean(gaps)

        row = {
            "max_gap": label,
            "trades": len(rets),
            "trades_per_yr": round(actual_tr_per_yr, 0),
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "expectancy_pct": round(rets_arr.mean(), 4),
            "annual_ret_pct": round(annual_ret, 2),
            "sharpe_ann": m["sharpe_ann"],
            "sortino_ann": round(sortino, 2),
            "calmar": round(calmar, 2) if math.isfinite(calmar) else "inf",
            "max_dd_pct": m["max_dd_pct"],
            "avg_hold": m["avg_hold"],
            "avg_gap": round(avg_gap, 1),
        }
        rows.append(row)
        with out_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        say(f"{label:>12s}: trades={len(rets):>4}  WR={m['win_rate']}%  PF={m['profit_factor']}  "
            f"Sharpe={m['sharpe_ann']}  DD={m['max_dd_pct']}%  Total={total:.1f}%  AnnRet={annual_ret:.1f}%  Calmar={row['calmar']}  AvgGap={avg_gap:.1f}")

    say("")
    say("=== FULL TABLE ===")
    say(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
