"""End-to-end backtester for the 20/50 EMA + Stoch(14,5,3) 30-min strategy.

For each (symbol, side, entry_variant, exit_variant) cell:
  1. Generate setups (3-event entry: EMA cross → Stoch zone touch → Stoch bull/bear cross)
  2. Apply the entry variant filter
  3. For each setup that fills, simulate the trade per the exit variant
  4. Aggregate to per-cell metrics

Sides:
  - LONG : EMA20>EMA50 bias, Stoch min(K,D)<=30 touch, K crosses ABOVE D, entry above close
  - SHORT: EMA20<EMA50 bias, Stoch max(K,D)>=70 touch, K crosses BELOW D, entry below close

Entry variants:
  - E0: literal — no extra price filter at Stoch cross candle
  - E1: HARD SKIP if close NOT on trend-correct side of EMA20
        (long: close > EMA20 required; short: close < EMA20 required)
  - E2: WAIT — defer entry until first candle whose close is on trend-correct
        side of EMA20, within ENTRY_VALIDITY_CANDLES window. Buy/sell stop
        placed at THAT deferred candle's close +/- tick.

Exit variants (each composed with X1 = reverse-EMA-cross as the always-on
"bias break" exit; the secondary exit fires on whichever happens first):
  - X1     : pure reverse EMA cross (no secondary)
  - X2     : X1 + hard SL @ entry candle's low - 0.5 * ATR(14)  (mirrored for short)
  - X3a    : X1 + Supertrend(7, 3.0) flip
  - X3b    : X1 + Supertrend(10, 3.0) flip
  - X4     : X1 + RSI(14) close < 50 (long) / > 50 (short)
  - X5     : X1 + fixed 1:2 R:R (R = entry - sl_anchor for long)
  - X6     : X1 + ATR trailing stop, k=2.5
  - X8     : X1 + time stop: exit if no profit after 40 candles
  - X9     : X1 + Chandelier (HHV(22) - 3*ATR for long)
  - X10    : X1 + close < EMA20 (long) / > EMA20 (short)

Per-trade P&L is in price-percent. No commission/slippage modelled in v1
(noted in STATUS doc).
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data30 import INTRADAY_STOCKS, BT_START, BT_END, load_30min  # noqa: E402
from indicators_stoch import ema, stochastics, rsi, atr, supertrend  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR = SCRIPT_DIR.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

ENTRY_VALIDITY_CANDLES = 10
TICK = 0.05
STOCH_LONG_OS = 30.0   # long-side oversold threshold
STOCH_SHORT_OB = 70.0  # short-side overbought threshold

ENTRY_VARIANTS = ["E0", "E1", "E2"]
EXIT_VARIANTS = ["X1", "X2", "X3a", "X3b", "X4", "X5", "X6", "X8", "X9", "X10"]
SIDES = ["long", "short"]


# -----------------------------------------------------------------------------
# Indicator pre-compute
# -----------------------------------------------------------------------------

def _prepare(symbol: str) -> pd.DataFrame:
    df = load_30min(symbol, BT_START, BT_END)
    if df.empty or len(df) < 100:
        return pd.DataFrame()

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    st = stochastics(df["high"], df["low"], df["close"], 14, 5, 3)
    df["k"] = st["k"]
    df["d"] = st["d"]
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    st7 = supertrend(df["high"], df["low"], df["close"], 7, 3.0)
    df["st7_3_dir"] = st7["dir"]
    df["st7_3_st"] = st7["st"]
    st10 = supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    df["st10_3_dir"] = st10["dir"]
    df["st10_3_st"] = st10["st"]
    df["hh22"] = df["high"].rolling(22, min_periods=22).max()
    df["ll22"] = df["low"].rolling(22, min_periods=22).min()

    df = df.dropna(subset=[
        "ema20", "ema50", "k", "d", "rsi14", "atr14",
        "st7_3_st", "st10_3_st", "hh22", "ll22",
    ]).reset_index().rename(columns={"date": "dt"})
    return df


# -----------------------------------------------------------------------------
# Setup generator (returns one row per setup, with fill info)
# -----------------------------------------------------------------------------

def _round_tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 2)


def generate_setups(df: pd.DataFrame, side: str, entry_variant: str) -> list[dict]:
    """Return list of setup dicts for one symbol+side+entry_variant combo."""
    n = len(df)
    if n < 50:
        return []

    e20 = df["ema20"].values
    e50 = df["ema50"].values
    k_arr = df["k"].values
    d_arr = df["d"].values
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["dt"].values

    # Bias direction
    if side == "long":
        bias = e20 > e50
    else:
        bias = e20 < e50

    # Find contiguous bias periods
    period_starts = []
    in_period = False
    for i in range(n):
        if bias[i] and not in_period:
            period_starts.append(i)
            in_period = True
        elif not bias[i] and in_period:
            in_period = False

    setups: list[dict] = []
    for ema_cross_idx in period_starts:
        j_end = ema_cross_idx
        while j_end < n and bias[j_end]:
            j_end += 1

        armed = False
        last_touch_idx = None

        for j in range(ema_cross_idx, j_end):
            if side == "long":
                # Touch: min(K, D) <= 30
                if min(k_arr[j], d_arr[j]) <= STOCH_LONG_OS:
                    armed = True
                    last_touch_idx = j
                # Cross: K crosses ABOVE D
                cross_event = (
                    armed and j > 0
                    and k_arr[j - 1] <= d_arr[j - 1]
                    and k_arr[j] > d_arr[j]
                )
            else:  # short
                # Touch: max(K, D) >= 70
                if max(k_arr[j], d_arr[j]) >= STOCH_SHORT_OB:
                    armed = True
                    last_touch_idx = j
                # Cross: K crosses BELOW D
                cross_event = (
                    armed and j > 0
                    and k_arr[j - 1] >= d_arr[j - 1]
                    and k_arr[j] < d_arr[j]
                )

            if not cross_event:
                continue

            # Entry variant filter at C* (cross candle = j)
            on_correct_side = (
                (closes[j] > e20[j]) if side == "long" else (closes[j] < e20[j])
            )

            trigger_anchor_idx = j
            if entry_variant == "E1":
                if not on_correct_side:
                    armed = False
                    continue
            elif entry_variant == "E2":
                if not on_correct_side:
                    # Wait for first subsequent candle close on correct side, while bias intact
                    found = None
                    end = min(j + ENTRY_VALIDITY_CANDLES, n - 1)
                    for w in range(j + 1, end + 1):
                        if e20[w] <= e50[w] if side == "long" else e20[w] >= e50[w]:
                            break  # bias broke
                        on_w = (closes[w] > e20[w]) if side == "long" else (closes[w] < e20[w])
                        if on_w:
                            found = w
                            break
                    if found is None:
                        armed = False
                        continue
                    trigger_anchor_idx = found

            # Build trigger from trigger_anchor_idx (signal candle).
            # Trigger = candle HIGH + 1 tick (long) / LOW - 1 tick (short).
            # Fill requires a SUBSEQUENT candle's high (long) or low (short)
            # to breach this level. Same-candle breaches don't count.
            anchor_close = float(closes[trigger_anchor_idx])
            anchor_high = float(highs[trigger_anchor_idx])
            anchor_low = float(lows[trigger_anchor_idx])
            if side == "long":
                trigger_price = _round_tick(anchor_high + TICK)
            else:
                trigger_price = _round_tick(anchor_low - TICK)

            # Walk forward to find fill
            end = min(trigger_anchor_idx + ENTRY_VALIDITY_CANDLES, n - 1)
            filled = False
            fill_idx = None
            fill_price = None
            for t in range(trigger_anchor_idx + 1, end + 1):
                # bias must hold at fill candle close
                if (e20[t] <= e50[t]) if side == "long" else (e20[t] >= e50[t]):
                    break
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

            armed = False  # reset for next dip-and-cross cycle

            if not filled:
                continue

            setups.append({
                "ema_cross_idx": ema_cross_idx,
                "stoch_touch_idx": last_touch_idx,
                "cross_idx": j,
                "trigger_anchor_idx": trigger_anchor_idx,
                "trigger_price": trigger_price,
                "fill_idx": fill_idx,
                "fill_price": fill_price,
                "anchor_close": anchor_close,
                "anchor_high": anchor_high,
                "anchor_low": anchor_low,
            })

    return setups


# -----------------------------------------------------------------------------
# Exit simulator
# -----------------------------------------------------------------------------

@dataclass
class TradeOutcome:
    entry_idx: int
    entry_price: float
    exit_idx: int
    exit_price: float
    exit_reason: str
    candles_held: int
    return_pct: float


def simulate_trade(
    df: pd.DataFrame,
    side: str,
    fill_idx: int,
    fill_price: float,
    sl_anchor_low: float,
    sl_anchor_high: float,
    exit_variant: str,
) -> TradeOutcome:
    """Walk the df forward from fill_idx+1 until any exit fires."""
    n = len(df)
    e20 = df["ema20"].values
    e50 = df["ema50"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    rsis = df["rsi14"].values
    atrs = df["atr14"].values
    st7d = df["st7_3_dir"].values
    st10d = df["st10_3_dir"].values
    hh22 = df["hh22"].values
    ll22 = df["ll22"].values
    k_arr = df["k"].values

    entry_atr = float(atrs[fill_idx])
    if side == "long":
        sl_2_price = sl_anchor_low - 0.5 * entry_atr
        # X5 fixed R:R
        risk = fill_price - sl_2_price
        tp_5 = fill_price + 2.0 * risk
        # ATR trail state
        trail_6 = fill_price - 2.5 * entry_atr
        peak_close = fill_price
    else:
        sl_2_price = sl_anchor_high + 0.5 * entry_atr
        risk = sl_2_price - fill_price
        tp_5 = fill_price - 2.0 * risk
        trail_6 = fill_price + 2.5 * entry_atr
        peak_close = fill_price

    for t in range(fill_idx + 1, n):
        c = float(closes[t])
        h = float(highs[t])
        l_ = float(lows[t])

        # X1 — reverse EMA cross (always on)
        if side == "long" and e20[t] <= e50[t]:
            return TradeOutcome(fill_idx, fill_price, t, c, "X1_rev_ema",
                                t - fill_idx, _ret(side, fill_price, c))
        if side == "short" and e20[t] >= e50[t]:
            return TradeOutcome(fill_idx, fill_price, t, c, "X1_rev_ema",
                                t - fill_idx, _ret(side, fill_price, c))

        if exit_variant == "X1":
            pass

        elif exit_variant == "X2":
            if side == "long" and l_ <= sl_2_price:
                return TradeOutcome(fill_idx, fill_price, t, sl_2_price,
                                    "X2_sl", t - fill_idx, _ret(side, fill_price, sl_2_price))
            if side == "short" and h >= sl_2_price:
                return TradeOutcome(fill_idx, fill_price, t, sl_2_price,
                                    "X2_sl", t - fill_idx, _ret(side, fill_price, sl_2_price))

        elif exit_variant == "X3a":
            if side == "long" and st7d[t] == -1:
                return TradeOutcome(fill_idx, fill_price, t, c, "X3a_st_flip",
                                    t - fill_idx, _ret(side, fill_price, c))
            if side == "short" and st7d[t] == 1:
                return TradeOutcome(fill_idx, fill_price, t, c, "X3a_st_flip",
                                    t - fill_idx, _ret(side, fill_price, c))

        elif exit_variant == "X3b":
            if side == "long" and st10d[t] == -1:
                return TradeOutcome(fill_idx, fill_price, t, c, "X3b_st_flip",
                                    t - fill_idx, _ret(side, fill_price, c))
            if side == "short" and st10d[t] == 1:
                return TradeOutcome(fill_idx, fill_price, t, c, "X3b_st_flip",
                                    t - fill_idx, _ret(side, fill_price, c))

        elif exit_variant == "X4":
            if side == "long" and rsis[t] < 50:
                return TradeOutcome(fill_idx, fill_price, t, c, "X4_rsi",
                                    t - fill_idx, _ret(side, fill_price, c))
            if side == "short" and rsis[t] > 50:
                return TradeOutcome(fill_idx, fill_price, t, c, "X4_rsi",
                                    t - fill_idx, _ret(side, fill_price, c))

        elif exit_variant == "X5":
            if side == "long":
                if h >= tp_5:
                    return TradeOutcome(fill_idx, fill_price, t, tp_5, "X5_tp",
                                        t - fill_idx, _ret(side, fill_price, tp_5))
                if l_ <= sl_2_price:
                    return TradeOutcome(fill_idx, fill_price, t, sl_2_price, "X5_sl",
                                        t - fill_idx, _ret(side, fill_price, sl_2_price))
            else:
                if l_ <= tp_5:
                    return TradeOutcome(fill_idx, fill_price, t, tp_5, "X5_tp",
                                        t - fill_idx, _ret(side, fill_price, tp_5))
                if h >= sl_2_price:
                    return TradeOutcome(fill_idx, fill_price, t, sl_2_price, "X5_sl",
                                        t - fill_idx, _ret(side, fill_price, sl_2_price))

        elif exit_variant == "X6":
            if side == "long":
                peak_close = max(peak_close, c)
                trail_6 = max(trail_6, peak_close - 2.5 * float(atrs[t]))
                if l_ <= trail_6:
                    return TradeOutcome(fill_idx, fill_price, t, trail_6, "X6_trail",
                                        t - fill_idx, _ret(side, fill_price, trail_6))
            else:
                peak_close = min(peak_close, c)
                trail_6 = min(trail_6, peak_close + 2.5 * float(atrs[t]))
                if h >= trail_6:
                    return TradeOutcome(fill_idx, fill_price, t, trail_6, "X6_trail",
                                        t - fill_idx, _ret(side, fill_price, trail_6))

        elif exit_variant == "X8":
            held = t - fill_idx
            in_profit = (c > fill_price) if side == "long" else (c < fill_price)
            if held >= 40 and not in_profit:
                return TradeOutcome(fill_idx, fill_price, t, c, "X8_time",
                                    t - fill_idx, _ret(side, fill_price, c))

        elif exit_variant == "X9":
            if side == "long":
                ch = float(hh22[t]) - 3.0 * float(atrs[t])
                if c < ch:
                    return TradeOutcome(fill_idx, fill_price, t, c, "X9_chand",
                                        t - fill_idx, _ret(side, fill_price, c))
            else:
                ch = float(ll22[t]) + 3.0 * float(atrs[t])
                if c > ch:
                    return TradeOutcome(fill_idx, fill_price, t, c, "X9_chand",
                                        t - fill_idx, _ret(side, fill_price, c))

        elif exit_variant == "X10":
            if side == "long" and c < e20[t]:
                return TradeOutcome(fill_idx, fill_price, t, c, "X10_ema_break",
                                    t - fill_idx, _ret(side, fill_price, c))
            if side == "short" and c > e20[t]:
                return TradeOutcome(fill_idx, fill_price, t, c, "X10_ema_break",
                                    t - fill_idx, _ret(side, fill_price, c))

    # End-of-data exit at last close
    last = n - 1
    return TradeOutcome(fill_idx, fill_price, last, float(closes[last]),
                        "EOD", last - fill_idx, _ret(side, fill_price, float(closes[last])))


def _ret(side: str, entry: float, exit_: float) -> float:
    if side == "long":
        return (exit_ - entry) / entry * 100.0
    return (entry - exit_) / entry * 100.0


# -----------------------------------------------------------------------------
# Cell metrics
# -----------------------------------------------------------------------------

def cell_metrics(trades: list[TradeOutcome]) -> dict:
    if not trades:
        return {"trades": 0}
    rets = np.array([t.return_pct for t in trades])
    wins = rets > 0
    losses = rets <= 0
    win_rate = float(wins.mean() * 100)
    avg_win = float(rets[wins].mean()) if wins.any() else 0.0
    avg_loss = float(rets[losses].mean()) if losses.any() else 0.0
    pf = (rets[wins].sum() / -rets[losses].sum()) if losses.any() and rets[losses].sum() < 0 else float("inf")
    total_ret = float(rets.sum())
    mean_ret = float(rets.mean())
    std_ret = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe_per_trade = (mean_ret / std_ret) if std_ret > 0 else 0.0
    # Annualize: ~250 trading days * 12 candles/day = 3000 30-min bars/year
    # Avg holding period ≈ mean candles_held → trades/year ≈ 3000 / hold
    avg_hold = float(np.mean([t.candles_held for t in trades])) if trades else 0
    trades_per_year = (3000 / max(avg_hold, 1)) if avg_hold else 0
    sharpe_ann = sharpe_per_trade * math.sqrt(max(trades_per_year, 1))
    # MaxDD on cumulative pct equity
    eq = np.cumsum(rets)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0.0
    sortino_per_trade = (mean_ret / np.std([r for r in rets if r < 0], ddof=1)) \
        if (rets < 0).sum() > 1 else 0.0
    sortino_ann = sortino_per_trade * math.sqrt(max(trades_per_year, 1))
    return {
        "trades": int(len(trades)),
        "win_rate": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else "inf",
        "total_ret_pct": round(total_ret, 2),
        "mean_ret_pct": round(mean_ret, 4),
        "std_ret_pct": round(std_ret, 4),
        "sharpe_per_trade": round(sharpe_per_trade, 3),
        "sharpe_ann": round(sharpe_ann, 2),
        "sortino_ann": round(sortino_ann, 2),
        "max_dd_pct": round(max_dd, 2),
        "avg_hold_candles": round(avg_hold, 1),
    }


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------

def main():
    log_path = LOG_DIR / "backtest.log"
    log = log_path.open("w")

    def say(msg: str):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    say(f"=== Research 33 backtest sweep ===")
    say(f"Sides: {SIDES}  EntryVariants: {ENTRY_VARIANTS}  ExitVariants: {EXIT_VARIANTS}")
    say(f"Universe: {INTRADAY_STOCKS}")
    say(f"Period: {BT_START} -> {BT_END}")
    say("")

    # Pre-compute indicators per symbol once
    say("Pre-computing indicators...")
    ts0 = time.time()
    prepared = {}
    for sym in INTRADAY_STOCKS:
        prepared[sym] = _prepare(sym)
        say(f"  {sym}  rows={len(prepared[sym])}")
    say(f"Indicators ready in {time.time()-ts0:.1f}s")
    say("")

    # Pre-generate setups per (symbol, side, entry_variant)
    say("Generating setups...")
    setups_cache = {}
    for sym in INTRADAY_STOCKS:
        df = prepared[sym]
        if df.empty:
            continue
        for side in SIDES:
            for ev in ENTRY_VARIANTS:
                key = (sym, side, ev)
                setups_cache[key] = generate_setups(df, side, ev)
    total_setups = sum(len(v) for v in setups_cache.values())
    say(f"Total filled setups across (sym, side, entry): {total_setups}")
    for side in SIDES:
        for ev in ENTRY_VARIANTS:
            n = sum(len(setups_cache[(sym, side, ev)])
                    for sym in INTRADAY_STOCKS if (sym, side, ev) in setups_cache)
            say(f"  side={side:<5}  entry={ev}  setups={n}")
    say("")

    # Run sweep
    say("Running per-cell simulation...")
    cell_csv = OUT_DIR / "sweep_results.csv"
    fieldnames = [
        "side", "entry_variant", "exit_variant", "n_symbols", "trades",
        "win_rate", "avg_win_pct", "avg_loss_pct", "profit_factor",
        "total_ret_pct", "mean_ret_pct", "std_ret_pct",
        "sharpe_per_trade", "sharpe_ann", "sortino_ann",
        "max_dd_pct", "avg_hold_candles",
    ]
    with cell_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
    cell_count = 0
    t_start = time.time()
    for side in SIDES:
        for ev in ENTRY_VARIANTS:
            for xv in EXIT_VARIANTS:
                cell_count += 1
                t_cell = time.time()
                trades_all: list[TradeOutcome] = []
                n_syms_with_trades = 0
                for sym in INTRADAY_STOCKS:
                    df = prepared[sym]
                    if df.empty: continue
                    setups = setups_cache.get((sym, side, ev), [])
                    sym_trades = []
                    for s in setups:
                        sym_trades.append(simulate_trade(
                            df, side, s["fill_idx"], s["fill_price"],
                            s["anchor_low"], s["anchor_high"], xv,
                        ))
                    if sym_trades:
                        n_syms_with_trades += 1
                    trades_all.extend(sym_trades)

                m = cell_metrics(trades_all)
                row = {"side": side, "entry_variant": ev, "exit_variant": xv,
                       "n_symbols": n_syms_with_trades, **m}
                # Fill any missing fields
                for k in fieldnames:
                    if k not in row: row[k] = ""
                with cell_csv.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
                dt = time.time() - t_cell
                say(f"  [{cell_count:>2}/60] side={side:<5} ev={ev} xv={xv:<4} "
                    f"trades={m.get('trades',0):>4}  WR={m.get('win_rate','-'):>5}%  "
                    f"PF={m.get('profit_factor','-'):>6}  Sharpe={m.get('sharpe_ann','-'):>5}  "
                    f"MaxDD={m.get('max_dd_pct','-'):>5}%  TR={m.get('total_ret_pct','-'):>6}%  "
                    f"({dt:.1f}s)")

    say("")
    say(f"Sweep complete in {time.time()-t_start:.1f}s. Results -> {cell_csv}")
    log.close()


if __name__ == "__main__":
    main()
