"""Volume-confirmed first-candle breakout backtest — runner.

For each (stock x timeframe x variant x direction) cell, generate signals via
`signals_volbreakout.vol_breakout_signals`, then walk forward intra-day and
log the result under EVERY exit policy in parallel:

  T_NO          — no exit until 15:25 IST
  T_HARD_SL     — hard SL at first-bar opposite extreme
  T_ATR_SL_k    — hard SL = entry - k*ATR(14, daily) for k in {0.3, 0.5, 1.0}
  T_CHANDELIER_k — trail SL = highest_high_since_entry - k*ATR for k in
                  {1.0, 1.5, 2.0}; no fixed target
  T_R_TARGET_xR — target = entry + x * initial_SL_distance for x in
                  {1, 1.5, 2, 3}; otherwise hold to EOD or hit hard SL
                  (initial SL = first-bar opposite extreme)
  T_STEP_TRAIL  — at +0.5R move SL to entry; at +1.5R move to +0.5R; at +3R
                  move to +1.5R; otherwise hard SL = first-bar opposite

Outputs:
  results/volbreakout_signals.csv  — per-signal-with-all-exits
  results/volbreakout_ranking.csv  — per-cell aggregate
  results/RESULTS.md               — top picks + summaries
  SWEEP-STATUS.md                  — status doc

Resumable: if results/volbreakout_signals.csv exists, skip already-completed
(symbol, tf, variant, direction, date) tuples.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from datetime import time as dtime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SCRIPTS_29 = HERE.parents[1] / "29_short_options_signal_sweep" / "scripts"
sys.path.insert(0, str(SCRIPTS_29))

from data_loader import load_5min, load_daily  # noqa: E402
from indicators import rsi as rsi_func  # noqa: E402
from signals_volbreakout import (  # noqa: E402
    VBSignal,
    build_first_bars,
    vol_breakout_signals,
)

ROOT = HERE.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SIGNAL_CSV = RESULTS / "volbreakout_signals.csv"
RANKING_CSV = RESULTS / "volbreakout_ranking.csv"
RESULTS_MD = RESULTS / "RESULTS.md"
STATUS_MD = ROOT / "SWEEP-STATUS.md"

STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
          "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "HINDUNILVR"]
TIMEFRAMES = ["5min", "15min"]
VOL_MULTS = [1.5, 2.0, 3.0]
GAP_PCTS = [0.0, 0.003, 0.005, None]   # None = filter off
RSI_MODES = [False, True]
DIRECTIONS = ["long", "short"]

PERIOD_START = pd.Timestamp("2024-03-01")
PERIOD_END = pd.Timestamp("2026-03-25")

EOD_TIME = dtime(15, 25)

# Exit policy labels (column tags in CSV)
EXIT_POLICIES = [
    "T_NO",
    "T_HARD_SL",
    "T_ATR_SL_0.3",
    "T_ATR_SL_0.5",
    "T_ATR_SL_1.0",
    "T_CHANDELIER_1.0",
    "T_CHANDELIER_1.5",
    "T_CHANDELIER_2.0",
    "T_R_TARGET_1.0R",
    "T_R_TARGET_1.5R",
    "T_R_TARGET_2.0R",
    "T_R_TARGET_3.0R",
    "T_STEP_TRAIL",
]


# ---------------------------------------------------------------------------
# ATR helpers
# ---------------------------------------------------------------------------

def daily_atr_series(daily: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder daily ATR(14). Returns a Series indexed by daily date."""
    if daily.empty:
        return pd.Series(dtype=float)
    df = daily.sort_index().copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return atr


def atr_for_signal(atr_daily: pd.Series, signal_date: pd.Timestamp) -> Optional[float]:
    """Return ATR(14) on the most recent daily bar BEFORE signal_date.
    None if not available (early in series).
    """
    if atr_daily.empty:
        return None
    prior = atr_daily.loc[atr_daily.index < signal_date]
    if prior.empty:
        return None
    v = float(prior.iloc[-1])
    if math.isnan(v) or v <= 0:
        return None
    return v


# ---------------------------------------------------------------------------
# Walk-forward / exit simulator
# ---------------------------------------------------------------------------

def simulate_exits(
    sess_after: pd.DataFrame,
    *,
    direction: str,
    entry_price: float,
    first_bar_low: float,
    first_bar_high: float,
    atr_daily_val: Optional[float],
) -> dict:
    """Walk forward 5-min by 5-min from sess_after (which begins at the bar
    after the signal). Apply EVERY exit policy in parallel.

    Returns a dict mapping policy_label -> per-policy result dict with keys:
      exit_time, exit_price, exit_reason, net_pts, net_pct, hold_min,
      mfe_pts, mae_pts (running max favorable / max adverse, signed in pts)

    All policies share the same MFE/MAE numbers (these are computed across
    the full intra-day post-signal walk; per-policy they're cropped at
    exit_time for fairness).
    """
    sign = 1 if direction == "long" else -1

    # Initialize each policy
    # State: alive=True, exit_time=None, exit_price=None, exit_reason=None
    policies: dict[str, dict] = {p: {
        "alive": True, "exit_time": None, "exit_price": None,
        "exit_reason": None, "mfe_pts": 0.0, "mae_pts": 0.0,
        "running_extreme": entry_price,  # high_since_entry for long, low_since_entry for short
    } for p in EXIT_POLICIES}

    # Hard-SL distances (in price pts, positive number)
    if direction == "long":
        hard_sl_dist = max(entry_price - first_bar_low, 0.0)
    else:
        hard_sl_dist = max(first_bar_high - entry_price, 0.0)
    if hard_sl_dist <= 0:
        # Pathological: entry on the wrong side of the first bar — use small fallback
        hard_sl_dist = entry_price * 0.001  # 0.1% safety floor

    atr_sl_dists = {}
    for k in (0.3, 0.5, 1.0):
        if atr_daily_val is None:
            atr_sl_dists[k] = None
        else:
            atr_sl_dists[k] = k * atr_daily_val

    # R-targets: target distance = x * hard_sl_dist
    r_target_dists = {1.0: hard_sl_dist, 1.5: 1.5 * hard_sl_dist,
                       2.0: 2.0 * hard_sl_dist, 3.0: 3.0 * hard_sl_dist}

    # Step trail thresholds (in R units): when MFE >= these, ratchet SL
    # (we apply at policy level, computing dynamically in the loop)

    def _close_policy(p: str, ts: pd.Timestamp, exit_price: float, reason: str):
        if not policies[p]["alive"]:
            return
        policies[p]["alive"] = False
        policies[p]["exit_time"] = ts
        policies[p]["exit_price"] = float(exit_price)
        policies[p]["exit_reason"] = reason

    if sess_after.empty:
        # No bars to walk -- everyone exits at entry (degenerate)
        for p in EXIT_POLICIES:
            _close_policy(p, pd.NaT, entry_price, "no_bars")
        return _format_results(policies, entry_price, sign, hold_minutes=0)

    # We need both the path of (high, low, close) for SL/target hits AND
    # to track running extreme for chandelier/step-trail.
    for ts, row in sess_after.iterrows():
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])

        # Update MFE/MAE on still-alive policies (and reusable)
        # Note: MFE/MAE are tracked in pts in the favorable direction
        favorable_now = (bar_high - entry_price) if direction == "long" else (entry_price - bar_low)
        adverse_now = (entry_price - bar_low) if direction == "long" else (bar_high - entry_price)
        favorable_now = max(favorable_now, 0.0)
        adverse_now = max(adverse_now, 0.0)

        for p in EXIT_POLICIES:
            if not policies[p]["alive"]:
                continue
            # Update extreme price seen during this trade for trailing exits
            if direction == "long":
                if bar_high > policies[p]["running_extreme"]:
                    policies[p]["running_extreme"] = bar_high
            else:
                if bar_low < policies[p]["running_extreme"]:
                    policies[p]["running_extreme"] = bar_low
            # Update MFE/MAE
            if favorable_now > policies[p]["mfe_pts"]:
                policies[p]["mfe_pts"] = favorable_now
            if adverse_now > policies[p]["mae_pts"]:
                policies[p]["mae_pts"] = adverse_now

        # Now check exits for each policy. Order within bar:
        #  1) hard SL hit (if intrabar low/high crossed)
        #  2) target hit
        #  Time-only policies just hold.

        # T_HARD_SL
        if policies["T_HARD_SL"]["alive"]:
            if direction == "long" and bar_low <= entry_price - hard_sl_dist:
                _close_policy("T_HARD_SL", ts, entry_price - hard_sl_dist, "hard_sl")
            elif direction == "short" and bar_high >= entry_price + hard_sl_dist:
                _close_policy("T_HARD_SL", ts, entry_price + hard_sl_dist, "hard_sl")

        # T_ATR_SL_k
        for k in (0.3, 0.5, 1.0):
            label = f"T_ATR_SL_{k}"
            if not policies[label]["alive"]:
                continue
            d = atr_sl_dists[k]
            if d is None:
                _close_policy(label, ts, bar_close, "atr_unavailable")
                continue
            if direction == "long" and bar_low <= entry_price - d:
                _close_policy(label, ts, entry_price - d, "atr_sl")
            elif direction == "short" and bar_high >= entry_price + d:
                _close_policy(label, ts, entry_price + d, "atr_sl")

        # T_CHANDELIER_k: trail SL = running_extreme - k*ATR  (long)
        #                            running_extreme + k*ATR  (short)
        for k in (1.0, 1.5, 2.0):
            label = f"T_CHANDELIER_{k}"
            if not policies[label]["alive"]:
                continue
            if atr_daily_val is None:
                _close_policy(label, ts, bar_close, "atr_unavailable")
                continue
            ext = policies[label]["running_extreme"]
            if direction == "long":
                trail_sl = ext - k * atr_daily_val
                if bar_low <= trail_sl:
                    _close_policy(label, ts, trail_sl, "chandelier")
            else:
                trail_sl = ext + k * atr_daily_val
                if bar_high >= trail_sl:
                    _close_policy(label, ts, trail_sl, "chandelier")

        # T_R_TARGET_xR: hard SL hit OR target hit (SL checked first within bar)
        for x in (1.0, 1.5, 2.0, 3.0):
            label = f"T_R_TARGET_{x}R"
            if not policies[label]["alive"]:
                continue
            tgt_d = r_target_dists[x]
            if direction == "long":
                # Conservative: if both SL and target are crossed in one bar, assume SL first
                if bar_low <= entry_price - hard_sl_dist:
                    _close_policy(label, ts, entry_price - hard_sl_dist, "hard_sl")
                    continue
                if bar_high >= entry_price + tgt_d:
                    _close_policy(label, ts, entry_price + tgt_d, "r_target")
            else:
                if bar_high >= entry_price + hard_sl_dist:
                    _close_policy(label, ts, entry_price + hard_sl_dist, "hard_sl")
                    continue
                if bar_low <= entry_price - tgt_d:
                    _close_policy(label, ts, entry_price - tgt_d, "r_target")

        # T_STEP_TRAIL: dynamic SL based on MFE-in-R-units of running_extreme
        if policies["T_STEP_TRAIL"]["alive"]:
            ext = policies["T_STEP_TRAIL"]["running_extreme"]
            mfe_R = ((ext - entry_price) if direction == "long" else (entry_price - ext)) / hard_sl_dist
            # Determine current SL distance from entry (positive distance below for long)
            if mfe_R >= 3.0:
                sl_offset_pts = 1.5 * hard_sl_dist  # SL above entry by 1.5R
            elif mfe_R >= 1.5:
                sl_offset_pts = 0.5 * hard_sl_dist
            elif mfe_R >= 0.5:
                sl_offset_pts = 0.0
            else:
                sl_offset_pts = -hard_sl_dist  # initial: SL at entry - 1R
            if direction == "long":
                sl_price = entry_price + sl_offset_pts
                if bar_low <= sl_price:
                    _close_policy("T_STEP_TRAIL", ts, sl_price,
                                  "step_trail" if sl_offset_pts > 0
                                  else ("breakeven" if sl_offset_pts == 0 else "hard_sl"))
            else:
                sl_price = entry_price - sl_offset_pts
                if bar_high >= sl_price:
                    _close_policy("T_STEP_TRAIL", ts, sl_price,
                                  "step_trail" if sl_offset_pts > 0
                                  else ("breakeven" if sl_offset_pts == 0 else "hard_sl"))

        # If everyone has exited, we can stop walking
        if not any(p["alive"] for p in policies.values()):
            break

    # Anyone still alive at end-of-walk exits at last bar's close (15:25)
    last_ts = sess_after.index[-1]
    last_close = float(sess_after["close"].iloc[-1])
    for p in EXIT_POLICIES:
        if policies[p]["alive"]:
            reason = "eod" if p == "T_NO" else "eod_no_hit"
            _close_policy(p, last_ts, last_close, reason)

    # Compute hold_min from the entry signal time -> exit_time
    return _format_results(policies, entry_price, sign,
                           hold_minutes=None,  # filled in caller (needs signal_time)
                           )


def _format_results(policies: dict, entry_price: float, sign: int,
                     hold_minutes: Optional[int]) -> dict:
    """Convert policies dict -> {policy_label: result dict}."""
    out = {}
    for p, st in policies.items():
        ep = st["exit_price"] if st["exit_price"] is not None else entry_price
        net_pts = (ep - entry_price) * sign
        net_pct = net_pts / entry_price if entry_price > 0 else 0.0
        out[p] = {
            "exit_time": st["exit_time"],
            "exit_price": ep,
            "exit_reason": st["exit_reason"] or "n/a",
            "net_pts": net_pts,
            "net_pct": net_pct,
            "mfe_pts": st["mfe_pts"],
            "mae_pts": st["mae_pts"],
        }
    return out


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def build_csv_header() -> list[str]:
    base = [
        "signal_id", "symbol", "timeframe", "variant", "direction", "date",
        "signal_time", "entry_price", "prev_day_high", "prev_day_low",
        "prev_day_close", "first_bar_open", "first_bar_high", "first_bar_low",
        "first_bar_close", "first_bar_volume", "vol_avg_20d", "vol_ratio",
        "gap_pct", "atr_daily", "rsi_at_signal",
    ]
    for p in EXIT_POLICIES:
        for c in ("exit_time", "exit_price", "exit_reason",
                   "net_pts", "net_pct", "mfe_pts", "mae_pts", "hold_min"):
            base.append(f"{p}__{c}")
    return base


def load_done_keys() -> set[tuple]:
    """Return set of (symbol, tf, variant, direction, date) already in CSV."""
    if not SIGNAL_CSV.exists():
        return set()
    done = set()
    with SIGNAL_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["symbol"], row["timeframe"], row["variant"],
                       row["direction"], row["date"]))
    return done


def run():
    print(f"== Volume-confirmed first-candle breakout sweep ==")
    print(f"Period: {PERIOD_START.date()} -> {PERIOD_END.date()}")
    print(f"Stocks: {len(STOCKS)}  TFs: {TIMEFRAMES}")
    n_variants = len(VOL_MULTS) * len(GAP_PCTS) * len(RSI_MODES)
    n_cells = len(STOCKS) * len(TIMEFRAMES) * n_variants * len(DIRECTIONS)
    print(f"Variants per direction: {n_variants}  Total cells (incl direction): {n_cells}")

    fieldnames = build_csv_header()
    if not SIGNAL_CSV.exists():
        with SIGNAL_CSV.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    done_keys = load_done_keys()
    print(f"Already completed signal rows in CSV: {len(done_keys)}")

    # Track aggregate stats in memory; we'll re-derive from CSV at the end too
    sig_id = 0
    if done_keys:
        sig_id = len(done_keys)

    cells_done = 0
    # Open CSV once for the whole sweep, keep it open for fast appends
    csv_file = SIGNAL_CSV.open("a", newline="", buffering=1)  # line-buffered
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    for sym in STOCKS:
        try:
            df5 = load_5min(sym, PERIOD_START.strftime("%Y-%m-%d"),
                            PERIOD_END.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"  [{sym}] load_5min failed: {e}")
            continue
        df5 = df5.loc[(df5.index >= PERIOD_START)
                       & (df5.index < PERIOD_END + pd.Timedelta(days=1))]
        if df5.empty:
            print(f"  [{sym}] empty after period clip")
            continue

        daily = load_daily(sym, "2023-03-01", PERIOD_END.strftime("%Y-%m-%d"))
        atr_d = daily_atr_series(daily, n=14)

        # Build first-bar tables and 5-min RSI once per symbol (avoid recomputing
        # 33K-row RSI inside every variant call)
        fb_by_tf = {tf: build_first_bars(df5, tf) for tf in TIMEFRAMES}
        rsi_5m = rsi_func(df5["close"], 14)
        # Cell-level done-keys: (sym, tf, variant, direction) -> set(date_str)
        # Allows fast "skip whole cell" if we know we already produced its signals.
        done_by_cell = {}
        for k in done_keys:
            cell = (k[0], k[1], k[2], k[3])
            done_by_cell.setdefault(cell, set()).add(k[4])
        sess_dates_total = sorted(set(d.strftime("%Y-%m-%d") for d in df5.index.normalize().unique()))
        n_sess_total = len(sess_dates_total)

        for tf in TIMEFRAMES:
            fb = fb_by_tf[tf]
            for vol_mult in VOL_MULTS:
                for gap in GAP_PCTS:
                    for use_rsi in RSI_MODES:
                        for direction in DIRECTIONS:
                            cells_done += 1
                            cell_n = 0
                            # Build a tentative variant tag mirroring signals_volbreakout's tagging
                            gap_tag = "off" if gap is None else f"{gap:.3f}"
                            rsi_tag = f"rsi{int(40)}_{int(60)}" if use_rsi else "norsi"
                            variant_tag = f"{direction[:1]}_vm{vol_mult}_gap{gap_tag}_{rsi_tag}"
                            cell_key = (sym, tf, variant_tag, direction)
                            already_dates = done_by_cell.get(cell_key, set())
                            for s in vol_breakout_signals(
                                df5, fb, daily,
                                vol_mult=vol_mult,
                                gap_pct=gap,
                                use_rsi=use_rsi,
                                direction=direction,
                                symbol=sym,
                                timeframe=tf,
                                rsi_precomputed=rsi_5m,
                            ):
                                date_str = s.date.strftime("%Y-%m-%d")
                                if date_str in already_dates:
                                    continue
                                key = (sym, tf, s.variant, direction, date_str)
                                if key in done_keys:
                                    continue
                                # Walk forward starting from candle AFTER signal_time
                                sess_day = df5.loc[df5.index.normalize() == s.date.normalize()]
                                sess_after = sess_day.loc[
                                    (sess_day.index > s.signal_time)
                                    & (sess_day.index.time <= EOD_TIME)
                                ]
                                atr_val = atr_for_signal(atr_d, s.date)
                                ex = simulate_exits(
                                    sess_after,
                                    direction=direction,
                                    entry_price=s.signal_price,
                                    first_bar_low=s.first_bar_low,
                                    first_bar_high=s.first_bar_high,
                                    atr_daily_val=atr_val,
                                )
                                # Compute hold_min relative to signal_time
                                row = {
                                    "signal_id": sig_id,
                                    "symbol": sym,
                                    "timeframe": tf,
                                    "variant": s.variant,
                                    "direction": direction,
                                    "date": s.date.strftime("%Y-%m-%d"),
                                    "signal_time": s.signal_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "entry_price": round(s.signal_price, 4),
                                    "prev_day_high": round(s.prev_day_high, 4),
                                    "prev_day_low": round(s.prev_day_low, 4),
                                    "prev_day_close": round(s.prev_day_close, 4),
                                    "first_bar_open": round(s.first_bar_open, 4),
                                    "first_bar_high": round(s.first_bar_high, 4),
                                    "first_bar_low": round(s.first_bar_low, 4),
                                    "first_bar_close": round(s.first_bar_close, 4),
                                    "first_bar_volume": int(s.first_bar_volume),
                                    "vol_avg_20d": round(s.vol_avg_20d, 1),
                                    "vol_ratio": round(s.vol_ratio, 3),
                                    "gap_pct": round(s.gap_pct, 5),
                                    "atr_daily": round(atr_val, 4) if atr_val else "",
                                    "rsi_at_signal": round(s.rsi_at_signal, 2)
                                        if s.rsi_at_signal is not None else "",
                                }
                                for p in EXIT_POLICIES:
                                    r = ex[p]
                                    et = r["exit_time"]
                                    if pd.isna(et) if et is not None else True:
                                        hold_min = 0
                                    else:
                                        try:
                                            hold_min = int((et - s.signal_time).total_seconds() // 60)
                                        except Exception:
                                            hold_min = 0
                                    row[f"{p}__exit_time"] = et.strftime("%Y-%m-%d %H:%M:%S") \
                                        if (et is not None and not pd.isna(et)) else ""
                                    row[f"{p}__exit_price"] = round(r["exit_price"], 4)
                                    row[f"{p}__exit_reason"] = r["exit_reason"]
                                    row[f"{p}__net_pts"] = round(r["net_pts"], 4)
                                    row[f"{p}__net_pct"] = round(r["net_pct"], 6)
                                    row[f"{p}__mfe_pts"] = round(r["mfe_pts"], 4)
                                    row[f"{p}__mae_pts"] = round(r["mae_pts"], 4)
                                    row[f"{p}__hold_min"] = hold_min
                                # Append immediately to the persistent writer
                                csv_writer.writerow(row)
                                sig_id += 1
                                cell_n += 1
                            if cell_n > 0:
                                print(f"  [{sym}/{tf}] vm={vol_mult} gap={gap} rsi={use_rsi} {direction} -> {cell_n} signals", flush=True)
                    # flush at the end of each (vol_mult x gap) block
                    csv_file.flush()
        # End-of-symbol checkpoint
        csv_file.flush()
        print(f"  [{sym}] done.", flush=True)

    csv_file.close()
    print(f"\nTotal cells visited: {cells_done}")
    print(f"Wrote signals CSV at {SIGNAL_CSV}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate():
    """Read signals CSV and produce ranking CSV per
    (symbol x timeframe x variant x direction x exit_policy).
    """
    if not SIGNAL_CSV.exists():
        print("No signals CSV — run() first.")
        return
    df = pd.read_csv(SIGNAL_CSV)
    if df.empty:
        print("Empty signals CSV.")
        return
    print(f"Aggregating {len(df)} signal rows...")

    rows = []
    grp_keys = ["symbol", "timeframe", "variant", "direction"]
    for keys, g in df.groupby(grp_keys, sort=False):
        n = len(g)
        if n < 5:
            continue
        for p in EXIT_POLICIES:
            net_pct = g[f"{p}__net_pct"].astype(float)
            net_pts = g[f"{p}__net_pts"].astype(float)
            mfe_pct = g[f"{p}__mfe_pts"].astype(float) / g["entry_price"].astype(float)
            mae_pct = g[f"{p}__mae_pts"].astype(float) / g["entry_price"].astype(float)
            wins = (net_pct > 0).sum()
            losses = (net_pct <= 0).sum()
            wr = wins / n
            avg_win = float(net_pct[net_pct > 0].mean()) if wins > 0 else 0.0
            avg_loss = float(net_pct[net_pct <= 0].mean()) if losses > 0 else 0.0
            payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf") if avg_win > 0 else 0.0
            mean_pct = float(net_pct.mean())
            std_pct = float(net_pct.std(ddof=0))
            sharpe = (mean_pct / std_pct * wr) if std_pct > 0 else 0.0
            expectancy = (wr * avg_win) - ((1 - wr) * abs(avg_loss))
            mean_mfe_pct = float(mfe_pct.mean()) if len(mfe_pct) > 0 else 0.0
            mean_mae_pct = float(mae_pct.mean()) if len(mae_pct) > 0 else 0.0
            cap_eff = (mean_pct / mean_mfe_pct) if mean_mfe_pct > 0 else 0.0
            rows.append({
                "symbol": keys[0],
                "timeframe": keys[1],
                "variant": keys[2],
                "direction": keys[3],
                "exit_policy": p,
                "n_signals": n,
                "mean_net_pts": round(float(net_pts.mean()), 4),
                "std_net_pts": round(float(net_pts.std(ddof=0)), 4),
                "median_net_pts": round(float(net_pts.median()), 4),
                "p25_net_pts": round(float(net_pts.quantile(0.25)), 4),
                "p75_net_pts": round(float(net_pts.quantile(0.75)), 4),
                "mean_net_pct": round(mean_pct, 6),
                "std_net_pct": round(std_pct, 6),
                "win_rate": round(wr * 100.0, 2),
                "avg_win_pct": round(avg_win * 100.0, 4),
                "avg_loss_pct": round(avg_loss * 100.0, 4),
                "payoff_ratio": round(payoff, 3) if not math.isinf(payoff) else 999.0,
                "mean_mfe_pct": round(mean_mfe_pct * 100.0, 4),
                "mean_mae_pct": round(mean_mae_pct * 100.0, 4),
                "capture_efficiency": round(cap_eff, 4),
                "sharpe_score": round(sharpe, 5),
                "expectancy_pct": round(expectancy * 100.0, 4),
            })
    rows.sort(key=lambda r: r["sharpe_score"], reverse=True)
    if not rows:
        print("No cells with n>=5.")
        return
    fieldnames = list(rows[0].keys())
    with RANKING_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} ranked cells to {RANKING_CSV}")
    return rows


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def write_results_md(rows: list[dict]):
    if not rows:
        return
    df = pd.read_csv(SIGNAL_CSV)

    # Counts
    by_dir = df.groupby("direction").size().to_dict()
    by_sym_dir = df.groupby(["symbol", "direction"]).size().unstack(fill_value=0)

    lines = ["# Volume-Confirmed First-Candle Breakout — Backtest Results\n\n"]
    lines.append("## Setup\n\n")
    lines.append(f"- Period: 2024-03-01 to 2026-03-25\n")
    lines.append(f"- Stocks (10): {', '.join(STOCKS)}\n")
    lines.append(f"- Timeframes: 5min, 15min (first candle of session)\n")
    lines.append(f"- Variant grid: vol_mult ∈ {{1.5, 2.0, 3.0}}  ×  gap_pct ∈ {{0%, 0.3%, 0.5%, off}}  ×  RSI ∈ {{off, on(40/60)}}\n")
    lines.append(f"- Direction: long & short, evaluated independently\n")
    lines.append(f"- 13 exit policies tested per signal in parallel\n")
    lines.append(f"- Total signals fired: **{len(df)}**\n")
    lines.append(f"  - Long: {by_dir.get('long', 0)}\n")
    lines.append(f"  - Short: {by_dir.get('short', 0)}\n\n")

    lines.append("## Signals fired per stock x direction\n\n")
    lines.append("| Stock | Long | Short | Total |\n")
    lines.append("|---|---:|---:|---:|\n")
    for sym in STOCKS:
        L = int(by_sym_dir.loc[sym, "long"]) if sym in by_sym_dir.index and "long" in by_sym_dir.columns else 0
        S = int(by_sym_dir.loc[sym, "short"]) if sym in by_sym_dir.index and "short" in by_sym_dir.columns else 0
        lines.append(f"| {sym} | {L} | {S} | {L + S} |\n")
    lines.append("\n")

    promote = [r for r in rows if r["n_signals"] >= 10 and r["mean_net_pct"] > 0]

    lines.append("## Top 10 configurations across all stocks (by Sharpe, n>=10, mean>0)\n\n")
    lines.append("| Symbol | TF | Variant | Dir | ExitPolicy | n | mean% | std% | WR% | Payoff | CapEff | Sharpe | Expect% |\n")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in promote[:10]:
        lines.append(
            f"| {r['symbol']} | {r['timeframe']} | {r['variant']} | {r['direction']} | "
            f"{r['exit_policy']} | {r['n_signals']} | {r['mean_net_pct']*100:.3f} | "
            f"{r['std_net_pct']*100:.3f} | {r['win_rate']:.1f} | {r['payoff_ratio']:.2f} | "
            f"{r['capture_efficiency']:.2f} | {r['sharpe_score']:.4f} | "
            f"{r['expectancy_pct']:.3f} |\n"
        )

    lines.append("\n## Top 5 by Sharpe (any n>=5)\n\n")
    lines.append("| Symbol | TF | Variant | Dir | ExitPolicy | n | mean% | WR% | Sharpe |\n")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|\n")
    for r in rows[:5]:
        lines.append(
            f"| {r['symbol']} | {r['timeframe']} | {r['variant']} | {r['direction']} | "
            f"{r['exit_policy']} | {r['n_signals']} | {r['mean_net_pct']*100:.3f} | "
            f"{r['win_rate']:.1f} | {r['sharpe_score']:.4f} |\n"
        )

    lines.append("\n## Best per-stock signal (any direction, n>=10, mean>0)\n\n")
    lines.append("| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | Sharpe | Notes |\n")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---|\n")
    by_sym_best = {}
    for r in promote:
        cur = by_sym_best.get(r["symbol"])
        if not cur or r["sharpe_score"] > cur["sharpe_score"]:
            by_sym_best[r["symbol"]] = r
    for sym in STOCKS:
        r = by_sym_best.get(sym)
        if not r:
            lines.append(f"| {sym} | — | (none viable) | | | | | | | |\n")
            continue
        lines.append(
            f"| {r['symbol']} | {r['timeframe']} | {r['variant']} | {r['direction']} | "
            f"{r['exit_policy']} | {r['n_signals']} | {r['mean_net_pct']*100:.3f} | "
            f"{r['win_rate']:.1f} | {r['sharpe_score']:.4f} | "
            f"payoff={r['payoff_ratio']:.2f}, capeff={r['capture_efficiency']:.2f} |\n"
        )

    # Best exit policy comparison (averaged across all viable cells)
    lines.append("\n## Exit policy comparison (averaged across cells with n>=10, mean>0)\n\n")
    lines.append("| ExitPolicy | n_cells | avg_mean% | avg_WR% | avg_capEff | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    pol_stats = {}
    for r in promote:
        pol_stats.setdefault(r["exit_policy"], []).append(r)
    pol_summary = []
    for p in EXIT_POLICIES:
        cells = pol_stats.get(p, [])
        if not cells:
            pol_summary.append((p, 0, 0.0, 0.0, 0.0, 0.0))
            continue
        ams = float(np.mean([c["mean_net_pct"] for c in cells])) * 100
        awr = float(np.mean([c["win_rate"] for c in cells]))
        acapeff = float(np.mean([c["capture_efficiency"] for c in cells]))
        asharpe = float(np.mean([c["sharpe_score"] for c in cells]))
        pol_summary.append((p, len(cells), ams, awr, acapeff, asharpe))
    pol_summary.sort(key=lambda x: x[5], reverse=True)
    for p, nc, ams, awr, acap, ash in pol_summary:
        lines.append(f"| {p} | {nc} | {ams:.4f} | {awr:.1f} | {acap:.2f} | {ash:.4f} |\n")

    # Direction comparison
    lines.append("\n## Direction comparison (cells with n>=10)\n\n")
    long_cells = [r for r in rows if r["direction"] == "long" and r["n_signals"] >= 10]
    short_cells = [r for r in rows if r["direction"] == "short" and r["n_signals"] >= 10]
    def _dstats(cs):
        if not cs:
            return 0, 0.0, 0.0, 0.0
        m = float(np.mean([c["mean_net_pct"] for c in cs])) * 100
        w = float(np.mean([c["win_rate"] for c in cs]))
        s = float(np.mean([c["sharpe_score"] for c in cs]))
        return len(cs), m, w, s
    Ln, Lm, Lw, Ls = _dstats(long_cells)
    Sn, Sm, Sw, Ss = _dstats(short_cells)
    lines.append("| Direction | n_cells | avg_mean% | avg_WR% | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    lines.append(f"| long  | {Ln} | {Lm:.4f} | {Lw:.1f} | {Ls:.4f} |\n")
    lines.append(f"| short | {Sn} | {Sm:.4f} | {Sw:.1f} | {Ss:.4f} |\n")

    # Volume threshold sweep insight
    lines.append("\n## Volume threshold (vm) sweep — viable cells only\n\n")
    lines.append("| vol_mult | n_cells | avg_mean% | avg_WR% | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for vm in (1.5, 2.0, 3.0):
        cs = [r for r in promote if f"vm{vm}" in r["variant"]]
        nc, m, w, s = _dstats(cs)
        lines.append(f"| {vm} | {nc} | {m:.4f} | {w:.1f} | {s:.4f} |\n")

    RESULTS_MD.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS_MD}")


def write_status_md(rows: Optional[list[dict]] = None, n_signals: int = 0,
                    completed: bool = False):
    lines = [
        "# Volume-Confirmed First-Candle Breakout Sweep — Status\n\n",
        "## Goal\n\n",
        "Test whether the first 5-min or 15-min candle closing above (below) the prior day's high (low) ",
        "with a clear volume spike is exploitable as a stand-alone intraday momentum strategy on the 10 ",
        "large-caps with 5-min intraday data.\n\n",
        "## Universe & Period\n\n",
        f"- Stocks: {', '.join(STOCKS)}\n",
        f"- Period: 2024-03-01 to 2026-03-25 (locked, same as research/29)\n",
        f"- Data: market_data_unified, 5minute timeframe; 15-min built from 09:15+09:20+09:25 5-min bars\n\n",
        "## Variant Grid\n\n",
        "| Axis | Values |\n",
        "|---|---|\n",
        f"| timeframe | {TIMEFRAMES} |\n",
        f"| vol_mult | {VOL_MULTS} (multiplier of 20-session first-bar avg) |\n",
        f"| gap_pct | {[g for g in GAP_PCTS]} (None = filter off; positive = min |gap| ratio) |\n",
        f"| use_rsi | {RSI_MODES} (5-min RSI(14) at signal_time; long>=60, short<=40) |\n",
        f"| direction | {DIRECTIONS} |\n\n",
        f"Total cells: {len(STOCKS)} stocks × {len(TIMEFRAMES)} tf × {len(VOL_MULTS)} vm × ",
        f"{len(GAP_PCTS)} gap × {len(RSI_MODES)} rsi × {len(DIRECTIONS)} dir = ",
        f"{len(STOCKS)*len(TIMEFRAMES)*len(VOL_MULTS)*len(GAP_PCTS)*len(RSI_MODES)*len(DIRECTIONS)}\n\n",
        "## Exit Policies (13)\n\n",
        "All policies are evaluated in parallel for every signal:\n\n",
        "1. **T_NO** — hold to 15:25 IST\n",
        "2. **T_HARD_SL** — fixed SL at first-bar opposite extreme\n",
        "3. **T_ATR_SL_{0.3, 0.5, 1.0}** — fixed SL = entry − k × daily ATR(14)\n",
        "4. **T_CHANDELIER_{1.0, 1.5, 2.0}** — trail SL = running extreme − k × ATR\n",
        "5. **T_R_TARGET_{1, 1.5, 2, 3}R** — target = x × hard_SL_distance, else hard SL\n",
        "6. **T_STEP_TRAIL** — at +0.5R move SL to entry; at +1.5R to +0.5R; at +3R to +1.5R\n\n",
        "## Status\n\n",
    ]
    if completed:
        lines.append(f"- COMPLETED. {n_signals} signals across all cells.\n")
        lines.append(f"- Outputs:\n")
        lines.append(f"  - `results/volbreakout_signals.csv`\n")
        lines.append(f"  - `results/volbreakout_ranking.csv`\n")
        lines.append(f"  - `results/RESULTS.md`\n\n")
    else:
        lines.append("- IN PROGRESS or NOT YET RUN.\n\n")
    lines.append("## Crash Recovery\n\n")
    lines.append("This script is **resumable**. To resume:\n\n")
    lines.append("1. Inspect `results/volbreakout_signals.csv` — count rows by ")
    lines.append("(symbol, timeframe, variant, direction, date) tuples.\n")
    lines.append("2. Re-run `python research/30_volume_breakout/scripts/run_volbreakout.py`. ")
    lines.append("Already-logged tuples are skipped automatically.\n")
    lines.append("3. After all signals are logged, the same script aggregates and ")
    lines.append("writes `volbreakout_ranking.csv` + `RESULTS.md`.\n\n")
    lines.append("## Final Aggregation\n\n")
    lines.append("Per-cell ranking is on the cross of (symbol × timeframe × variant × ")
    lines.append("direction × exit_policy). Primary rank = `sharpe_score` = ")
    lines.append("`(mean_net_pct / std_net_pct) × win_rate_fraction`. Secondary = expectancy_pct.\n")
    STATUS_MD.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {STATUS_MD}")


def main():
    write_status_md()
    run()
    rows = aggregate()
    if rows:
        write_results_md(rows)
    # Final status
    n_sig = sum(1 for _ in open(SIGNAL_CSV)) - 1 if SIGNAL_CSV.exists() else 0
    write_status_md(rows=rows, n_signals=n_sig, completed=True)


if __name__ == "__main__":
    main()
