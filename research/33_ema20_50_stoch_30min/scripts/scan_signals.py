"""Signal scanner v2 — pullback-then-Stoch-crossover entry.

REVISED entry rules per user clarification:

  Event A : EMA20 crosses ABOVE EMA50 on close (Stoch state irrelevant here).
  Hold    : EMA20 stays > EMA50 — if bias breaks, abandon and wait for fresh A.
  Event B : Within the EMA-bullish period, Stoch min(%K, %D) <= 30 at any candle.
  Event C : After B, the FIRST bullish Stochastic crossover (%K crosses above %D
            from below). This is the "trigger candle".
  Entry   : Buy-stop at C.close + 1 tick, valid for next M=10 30-min candles.
  Fill    : First subsequent candle whose high pierces the trigger AND where
            EMA20 > EMA50 at that candle's close. Gap-throughs fill at open.

Each EMA-bullish period can host MULTIPLE B+C cycles → multiple setups.
After a setup fires, the "armed" flag resets; a fresh ≤30 touch is required
before another setup can fire.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data30 import INTRADAY_STOCKS, BT_START, BT_END, load_30min  # noqa: E402
from indicators_stoch import ema, stochastics  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR = SCRIPT_DIR.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

ENTRY_VALIDITY_CANDLES = 10
TICK = 0.05
STOCH_OVERSOLD = 30.0


def _round_tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 2)


def scan_symbol(symbol: str) -> list[dict]:
    df = load_30min(symbol, BT_START, BT_END)
    if df.empty or len(df) < 100:
        return []

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    st = stochastics(df["high"], df["low"], df["close"], 14, 5, 3)
    df["k"] = st["k"]
    df["d"] = st["d"]

    df = df.dropna(subset=["ema20", "ema50", "k", "d"]).reset_index().rename(columns={"date": "dt"})
    if df.empty:
        return []

    n = len(df)
    e20 = df["ema20"].values
    e50 = df["ema50"].values
    k_arr = df["k"].values
    d_arr = df["d"].values
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["dt"].values

    # Identify contiguous EMA-bullish periods
    bull = e20 > e50
    # period_start = index where bull goes False→True (or starts True at idx 0)
    period_starts = []
    in_period = False
    for i in range(n):
        if bull[i] and not in_period:
            period_starts.append(i)
            in_period = True
        elif not bull[i] and in_period:
            in_period = False

    rows: list[dict] = []
    for ema_cross_idx in period_starts:
        # Walk to end of this EMA-bullish period
        # Find first j > ema_cross_idx where bull[j] is False
        j_end = ema_cross_idx
        while j_end < n and bull[j_end]:
            j_end += 1
        # j_end is first non-bull (or n)

        armed = False
        last_touch_idx = None

        # Within [ema_cross_idx, j_end), iterate candle by candle
        for j in range(ema_cross_idx, j_end):
            s_min = float(min(k_arr[j], d_arr[j]))

            # Update armed flag on Stoch touch ≤30
            if s_min <= STOCH_OVERSOLD:
                armed = True
                last_touch_idx = j

            # Check for bullish Stoch crossover at THIS candle
            # (%K crosses above %D: k[j-1] <= d[j-1] AND k[j] > d[j])
            if (
                armed
                and j > 0
                and k_arr[j - 1] <= d_arr[j - 1]
                and k_arr[j] > d_arr[j]
            ):
                # Setup fires on candle j
                trigger_close = float(closes[j])
                trigger_high = float(highs[j])
                trigger_price = _round_tick(trigger_close + TICK)

                # Walk forward up to M candles for fill
                end = min(j + ENTRY_VALIDITY_CANDLES, n - 1)
                filled = False
                fill_idx = None
                fill_price = None
                fill_break_reason = ""
                for t in range(j + 1, end + 1):
                    # EMA bias must still hold at fill candle close
                    if e20[t] <= e50[t]:
                        fill_break_reason = "ema_bias_broke"
                        break
                    if opens[t] >= trigger_price:
                        fill_price = float(opens[t])
                        fill_idx = t
                        filled = True
                        break
                    if highs[t] >= trigger_price:
                        fill_price = float(trigger_price)
                        fill_idx = t
                        filled = True
                        break
                else:
                    fill_break_reason = "validity_expired"

                rows.append({
                    "symbol": symbol,
                    "ema_cross_dt": pd.Timestamp(times[ema_cross_idx]).isoformat(),
                    "ema_cross_close": float(closes[ema_cross_idx]),
                    "stoch_touch_dt": pd.Timestamp(times[last_touch_idx]).isoformat(),
                    "stoch_touch_min": float(min(k_arr[last_touch_idx], d_arr[last_touch_idx])),
                    "candles_a_to_b": int(last_touch_idx - ema_cross_idx),
                    "setup_dt": pd.Timestamp(times[j]).isoformat(),
                    "setup_open": float(opens[j]),
                    "setup_high": trigger_high,
                    "setup_low": float(lows[j]),
                    "setup_close": trigger_close,
                    "setup_ema20": float(e20[j]),
                    "setup_ema50": float(e50[j]),
                    "setup_ema_spread_pct": float((e20[j] - e50[j]) / e50[j] * 100),
                    "setup_k": float(k_arr[j]),
                    "setup_d": float(d_arr[j]),
                    "candles_b_to_c": int(j - last_touch_idx),
                    "trigger_price": trigger_price,
                    "validity_candles": ENTRY_VALIDITY_CANDLES,
                    "filled": int(filled),
                    "fill_dt": pd.Timestamp(times[fill_idx]).isoformat() if filled else "",
                    "fill_price": fill_price if filled else "",
                    "candles_to_fill": (fill_idx - j) if filled else "",
                    "no_fill_reason": "" if filled else fill_break_reason,
                })

                # Reset armed for next dip-and-cross cycle within same EMA period
                armed = False

    return rows


def main():
    log_path = LOG_DIR / "scan.log"
    log = log_path.open("w")

    def say(msg: str):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    say("=== Research 33 signal scan v2 (pullback-then-Stoch-cross) ===")
    say(f"Period: {BT_START} -> {BT_END}")
    say(f"Stoch oversold threshold: <= {STOCH_OVERSOLD}")
    say(f"Entry validity: {ENTRY_VALIDITY_CANDLES} 30-min candles after C* (Stoch cross candle)")
    say(f"Universe: {len(INTRADAY_STOCKS)} stocks")
    say("")

    all_rows: list[dict] = []
    t0 = time.time()
    for sym in INTRADAY_STOCKS:
        ts = time.time()
        rows = scan_symbol(sym)
        dt = time.time() - ts
        n_filled = sum(r["filled"] for r in rows)
        say(f"  {sym:12s}  {len(rows):>4} setups,  {n_filled:>4} filled,  {dt:.1f}s")
        all_rows.extend(rows)
    say("")
    say(f"Total setups: {len(all_rows)}, "
        f"filled: {sum(r['filled'] for r in all_rows)}, "
        f"elapsed: {time.time()-t0:.1f}s")

    if not all_rows:
        say("No candidates.")
        log.close()
        return

    fieldnames = list(all_rows[0].keys())
    out_csv = OUT_DIR / "signals.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    say(f"Wrote {out_csv}  ({len(all_rows)} rows)")

    # Curate top candidates: filled, recent, distributed across symbols
    df = pd.DataFrame(all_rows)
    df["setup_dt"] = pd.to_datetime(df["setup_dt"])
    df = df.sort_values("setup_dt", ascending=False)

    top_filled = df[df["filled"] == 1].copy()
    top = (
        top_filled.groupby("symbol", group_keys=False)
        .head(2)
        .sort_values("setup_dt", ascending=False)
        .head(20)
    )
    top_csv = OUT_DIR / "top_candidates.csv"
    top.to_csv(top_csv, index=False)
    say(f"Wrote {top_csv}  ({len(top)} curated rows for verification)")
    say("")
    say("=== TOP CANDIDATES FOR CHART VERIFICATION ===")
    show_cols = [
        "symbol", "ema_cross_dt", "stoch_touch_dt", "stoch_touch_min",
        "setup_dt", "setup_close", "setup_k", "setup_d",
        "trigger_price", "fill_dt", "fill_price", "candles_to_fill",
    ]
    say(top[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
