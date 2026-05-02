"""Volume-confirmed first-candle breakout sweep — EXPANDED universe (79 stocks).

Sister of `research/30_volume_breakout`. Differences:
  - Universe: ALL 79 stocks with 5-min intraday data (Cohort A long-history +
    Cohort B 2-year), not just the 10 large-caps.
  - Timeframes: {5min, 10min, 15min} — adds 10-min as a new dimension.
  - Per-stock period: Cohort A from 2018-01-01, Cohort B from 2024-03-18,
    both capped at 2026-03-25.
  - Volume-Leaders analysis: per-stock leaderboard ranked by best Sharpe,
    counts of robust cells, direction asymmetry, timeframe sweet spot.

Reuses (without modification beyond a 10-min addition to build_first_bars):
  - `signals_volbreakout.vol_breakout_signals`  (research/30/scripts/)
  - `simulate_exits` re-implemented here (small, self-contained)

Outputs:
  - results/volbreakout_signals.csv   (gitignored — likely 50+ MB)
  - results/volbreakout_ranking.csv   (committable)
  - results/RESULTS.md
  - SWEEP-STATUS.md  (in research/30b_volume_breakout_expanded/)

Resumable: skip cells already in volbreakout_signals.csv.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time as _time
from datetime import time as dtime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESEARCH_ROOT = ROOT.parent
SCRIPTS_30 = RESEARCH_ROOT / "30_volume_breakout" / "scripts"
SCRIPTS_29 = RESEARCH_ROOT / "29_short_options_signal_sweep" / "scripts"

sys.path.insert(0, str(SCRIPTS_30))
sys.path.insert(0, str(SCRIPTS_29))

from data_loader import load_5min, load_daily  # noqa: E402  (research/29)
from indicators import rsi as rsi_func  # noqa: E402  (research/29)
from signals_volbreakout import (  # noqa: E402  (research/30, with our 10-min add)
    VBSignal,
    build_first_bars,
    vol_breakout_signals,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SIGNAL_CSV = RESULTS / "volbo_signals.csv"
RANKING_CSV = RESULTS / "volbo_ranking.csv"
RESULTS_MD = RESULTS / "RESULTS.md"
LEADERS_CSV = RESULTS / "volbo_leaders.csv"
STATUS_MD = ROOT / "NIFTY500_EXPANSION_SWEEP_STATUS.md"

COHORT_A = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "HINDUNILVR"]

COHORT_B = [
    "ADANIENT", "ADANIPORTS", "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT",
    "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BANKBARODA",
    "BEL", "BPCL", "BRITANNIA", "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE",
    "COLPAL", "CUMMINSIND", "DABUR", "DELHIVERY", "DIVISLAB", "DLF",
    "DRREDDY", "EICHERMOT", "FEDERALBNK", "GAIL", "GODREJPROP", "GRASIM",
    "HAL", "HAVELLS", "HCLTECH", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "IDFCFIRSTB", "INDUSINDBK", "IOC", "IRCTC", "JINDALSTEL", "JSWSTEEL",
    "LT", "M&M", "MARICO", "MARUTI", "MCX", "MUTHOOTFIN", "NESTLEIND",
    "NTPC", "ONGC", "PAYTM", "PERSISTENT", "PIDILITIND", "PNB", "POWERGRID",
    "SBILIFE", "SHREECEM", "SIEMENS", "SUNPHARMA", "TATACONSUM", "TATAPOWER",
    "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "VEDL", "VOLTAS",
    "WIPRO",
]

# Nifty 500 universe — read DYNAMICALLY from DB (will pick up backfilled stocks)
def _load_universe() -> list[str]:
    import sqlite3
    db = ROOT.parent.parent / "backtest_data" / "market_data.db"
    con = sqlite3.connect(db)
    syms = sorted(
        r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM market_data_unified "
            "WHERE timeframe='5minute' AND symbol NOT IN ('NIFTY50','BANKNIFTY')"
        ).fetchall()
    )
    con.close()
    return syms

ALL_STOCKS = _load_universe()
COHORT_A_SET = set(COHORT_A)
print(f"[run_volbo_500] Universe loaded: {len(ALL_STOCKS)} stocks "
      f"({len(COHORT_A_SET & set(ALL_STOCKS))} Cohort A, "
      f"{len(set(ALL_STOCKS) - COHORT_A_SET)} Cohort B+C)")


# ---------------------------------------------------------------------------
# Liquidity gate (Phase 1 / research/34)
# ---------------------------------------------------------------------------
LIQUIDITY_MIN_PRICE = 50.0           # Rs 50 minimum price
LIQUIDITY_MIN_TURNOVER = 5_00_00_000  # Rs 5 crore daily median turnover

def liquidity_ok(daily: pd.DataFrame, signal_date: pd.Timestamp,
                 lookback: int = 20) -> bool:
    """Return True if the stock passes price+turnover gate at signal_date.

    Uses last `lookback` daily bars BEFORE signal_date. Median turnover
    (close × volume) and median price both must clear the thresholds.
    """
    if daily.empty:
        return False
    prior = daily.loc[daily.index < signal_date].tail(lookback)
    if len(prior) < lookback:
        return False
    median_price = float(prior["close"].median())
    if median_price < LIQUIDITY_MIN_PRICE:
        return False
    turnover = (prior["close"] * prior["volume"]).median()
    if turnover < LIQUIDITY_MIN_TURNOVER:
        return False
    return True

TIMEFRAMES = ["5min", "10min", "15min"]
VOL_MULTS = [1.5, 2.0, 3.0]
GAP_PCTS = [0.0, 0.003, 0.005, None]   # None = filter off
RSI_MODES = [False, True]
DIRECTIONS = ["long", "short"]

PERIOD_END = pd.Timestamp("2026-03-25")
COHORT_A_START = pd.Timestamp("2018-01-01")
COHORT_B_START = pd.Timestamp("2024-03-18")

EOD_TIME = dtime(15, 25)

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


def cohort_for(symbol: str) -> str:
    return "A" if symbol in COHORT_A_SET else "B"


def period_for(symbol: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if symbol in COHORT_A_SET:
        return COHORT_A_START, PERIOD_END
    return COHORT_B_START, PERIOD_END


# ---------------------------------------------------------------------------
# ATR helpers
# ---------------------------------------------------------------------------

def daily_atr_series(daily: pd.DataFrame, n: int = 14) -> pd.Series:
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
# Walk-forward / exit simulator (mirrors research/30 logic)
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
    sign = 1 if direction == "long" else -1

    policies: dict[str, dict] = {p: {
        "alive": True, "exit_time": None, "exit_price": None,
        "exit_reason": None, "mfe_pts": 0.0, "mae_pts": 0.0,
        "running_extreme": entry_price,
    } for p in EXIT_POLICIES}

    if direction == "long":
        hard_sl_dist = max(entry_price - first_bar_low, 0.0)
    else:
        hard_sl_dist = max(first_bar_high - entry_price, 0.0)
    if hard_sl_dist <= 0:
        hard_sl_dist = entry_price * 0.001

    atr_sl_dists = {}
    for k in (0.3, 0.5, 1.0):
        atr_sl_dists[k] = None if atr_daily_val is None else k * atr_daily_val

    r_target_dists = {1.0: hard_sl_dist, 1.5: 1.5 * hard_sl_dist,
                      2.0: 2.0 * hard_sl_dist, 3.0: 3.0 * hard_sl_dist}

    def _close(p, ts, ep, reason):
        if not policies[p]["alive"]:
            return
        policies[p]["alive"] = False
        policies[p]["exit_time"] = ts
        policies[p]["exit_price"] = float(ep)
        policies[p]["exit_reason"] = reason

    if sess_after.empty:
        for p in EXIT_POLICIES:
            _close(p, pd.NaT, entry_price, "no_bars")
        return _format_results(policies, entry_price, sign)

    for ts, row in sess_after.iterrows():
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])

        favorable_now = (bar_high - entry_price) if direction == "long" else (entry_price - bar_low)
        adverse_now = (entry_price - bar_low) if direction == "long" else (bar_high - entry_price)
        favorable_now = max(favorable_now, 0.0)
        adverse_now = max(adverse_now, 0.0)

        for p in EXIT_POLICIES:
            if not policies[p]["alive"]:
                continue
            if direction == "long":
                if bar_high > policies[p]["running_extreme"]:
                    policies[p]["running_extreme"] = bar_high
            else:
                if bar_low < policies[p]["running_extreme"]:
                    policies[p]["running_extreme"] = bar_low
            if favorable_now > policies[p]["mfe_pts"]:
                policies[p]["mfe_pts"] = favorable_now
            if adverse_now > policies[p]["mae_pts"]:
                policies[p]["mae_pts"] = adverse_now

        # T_HARD_SL
        if policies["T_HARD_SL"]["alive"]:
            if direction == "long" and bar_low <= entry_price - hard_sl_dist:
                _close("T_HARD_SL", ts, entry_price - hard_sl_dist, "hard_sl")
            elif direction == "short" and bar_high >= entry_price + hard_sl_dist:
                _close("T_HARD_SL", ts, entry_price + hard_sl_dist, "hard_sl")

        # T_ATR_SL_k
        for k in (0.3, 0.5, 1.0):
            label = f"T_ATR_SL_{k}"
            if not policies[label]["alive"]:
                continue
            d = atr_sl_dists[k]
            if d is None:
                _close(label, ts, bar_close, "atr_unavailable")
                continue
            if direction == "long" and bar_low <= entry_price - d:
                _close(label, ts, entry_price - d, "atr_sl")
            elif direction == "short" and bar_high >= entry_price + d:
                _close(label, ts, entry_price + d, "atr_sl")

        # T_CHANDELIER_k
        for k in (1.0, 1.5, 2.0):
            label = f"T_CHANDELIER_{k}"
            if not policies[label]["alive"]:
                continue
            if atr_daily_val is None:
                _close(label, ts, bar_close, "atr_unavailable")
                continue
            ext = policies[label]["running_extreme"]
            if direction == "long":
                trail_sl = ext - k * atr_daily_val
                if bar_low <= trail_sl:
                    _close(label, ts, trail_sl, "chandelier")
            else:
                trail_sl = ext + k * atr_daily_val
                if bar_high >= trail_sl:
                    _close(label, ts, trail_sl, "chandelier")

        # T_R_TARGET
        for x in (1.0, 1.5, 2.0, 3.0):
            label = f"T_R_TARGET_{x}R"
            if not policies[label]["alive"]:
                continue
            tgt_d = r_target_dists[x]
            if direction == "long":
                if bar_low <= entry_price - hard_sl_dist:
                    _close(label, ts, entry_price - hard_sl_dist, "hard_sl")
                    continue
                if bar_high >= entry_price + tgt_d:
                    _close(label, ts, entry_price + tgt_d, "r_target")
            else:
                if bar_high >= entry_price + hard_sl_dist:
                    _close(label, ts, entry_price + hard_sl_dist, "hard_sl")
                    continue
                if bar_low <= entry_price - tgt_d:
                    _close(label, ts, entry_price - tgt_d, "r_target")

        # T_STEP_TRAIL
        if policies["T_STEP_TRAIL"]["alive"]:
            ext = policies["T_STEP_TRAIL"]["running_extreme"]
            mfe_R = ((ext - entry_price) if direction == "long" else (entry_price - ext)) / hard_sl_dist
            if mfe_R >= 3.0:
                sl_offset_pts = 1.5 * hard_sl_dist
            elif mfe_R >= 1.5:
                sl_offset_pts = 0.5 * hard_sl_dist
            elif mfe_R >= 0.5:
                sl_offset_pts = 0.0
            else:
                sl_offset_pts = -hard_sl_dist
            if direction == "long":
                sl_price = entry_price + sl_offset_pts
                if bar_low <= sl_price:
                    _close("T_STEP_TRAIL", ts, sl_price,
                           "step_trail" if sl_offset_pts > 0
                           else ("breakeven" if sl_offset_pts == 0 else "hard_sl"))
            else:
                sl_price = entry_price - sl_offset_pts
                if bar_high >= sl_price:
                    _close("T_STEP_TRAIL", ts, sl_price,
                           "step_trail" if sl_offset_pts > 0
                           else ("breakeven" if sl_offset_pts == 0 else "hard_sl"))

        if not any(p["alive"] for p in policies.values()):
            break

    last_ts = sess_after.index[-1]
    last_close = float(sess_after["close"].iloc[-1])
    for p in EXIT_POLICIES:
        if policies[p]["alive"]:
            reason = "eod" if p == "T_NO" else "eod_no_hit"
            _close(p, last_ts, last_close, reason)

    return _format_results(policies, entry_price, sign)


def _format_results(policies: dict, entry_price: float, sign: int) -> dict:
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
# CSV
# ---------------------------------------------------------------------------

def build_csv_header() -> list[str]:
    base = [
        "signal_id", "symbol", "cohort", "timeframe", "variant", "direction",
        "date", "signal_time", "entry_price", "prev_day_high", "prev_day_low",
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
    if not SIGNAL_CSV.exists():
        return set()
    done = set()
    with SIGNAL_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["symbol"], row["timeframe"], row["variant"],
                      row["direction"], row["date"]))
    return done


def load_done_cells() -> set[tuple]:
    """Set of (symbol, timeframe) cells where ALL variants/directions have at
    least been visited (proxy: any rows present means we've started). We use
    a simpler completion marker: store completed-stock list separately.
    """
    return set()


# ---------------------------------------------------------------------------
# Per-stock processor
# ---------------------------------------------------------------------------

def process_symbol(sym: str, csv_writer: csv.DictWriter, csv_file,
                   done_by_cell: dict, sig_id_start: int) -> tuple[int, int]:
    """Run all variant cells for a single symbol. Returns (signals_count, new_sig_id)."""
    period_start, period_end = period_for(sym)
    cohort = cohort_for(sym)

    try:
        df5 = load_5min(sym, period_start.strftime("%Y-%m-%d"),
                        period_end.strftime("%Y-%m-%d"))
    except Exception as e:
        print(f"  [{sym}] load_5min failed: {e}", flush=True)
        return 0, sig_id_start

    df5 = df5.loc[(df5.index >= period_start)
                  & (df5.index < period_end + pd.Timedelta(days=1))]
    if df5.empty:
        print(f"  [{sym}] empty 5-min after period clip", flush=True)
        return 0, sig_id_start

    daily_start = (period_start - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    daily = load_daily(sym, daily_start, period_end.strftime("%Y-%m-%d"))
    if daily.empty:
        print(f"  [{sym}] no daily data", flush=True)
        return 0, sig_id_start
    atr_d = daily_atr_series(daily, n=14)

    # Build first-bar tables and 5-min RSI once per symbol
    fb_by_tf = {tf: build_first_bars(df5, tf) for tf in TIMEFRAMES}
    rsi_5m = rsi_func(df5["close"], 14)

    sig_id = sig_id_start
    cell_signals = 0
    cells_visited = 0

    for tf in TIMEFRAMES:
        fb = fb_by_tf[tf]
        if fb.empty:
            continue
        for vol_mult in VOL_MULTS:
            for gap in GAP_PCTS:
                for use_rsi in RSI_MODES:
                    for direction in DIRECTIONS:
                        cells_visited += 1
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
                            # Liquidity gate (research/34): skip days where
                            # 20-day median price < Rs 50 OR median turnover
                            # < Rs 5 cr. Guards against penny-stock pollution.
                            if not liquidity_ok(daily, s.date):
                                continue

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
                            row = {
                                "signal_id": sig_id,
                                "symbol": sym,
                                "cohort": cohort,
                                "timeframe": tf,
                                "variant": s.variant,
                                "direction": direction,
                                "date": date_str,
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
                                if et is None or pd.isna(et):
                                    hold_min = 0
                                    et_str = ""
                                else:
                                    try:
                                        hold_min = int((et - s.signal_time).total_seconds() // 60)
                                    except Exception:
                                        hold_min = 0
                                    et_str = et.strftime("%Y-%m-%d %H:%M:%S")
                                row[f"{p}__exit_time"] = et_str
                                row[f"{p}__exit_price"] = round(r["exit_price"], 4)
                                row[f"{p}__exit_reason"] = r["exit_reason"]
                                row[f"{p}__net_pts"] = round(r["net_pts"], 4)
                                row[f"{p}__net_pct"] = round(r["net_pct"], 6)
                                row[f"{p}__mfe_pts"] = round(r["mfe_pts"], 4)
                                row[f"{p}__mae_pts"] = round(r["mae_pts"], 4)
                                row[f"{p}__hold_min"] = hold_min
                            csv_writer.writerow(row)
                            sig_id += 1
                            cell_signals += 1
                # flush per (vol_mult x gap) block
                csv_file.flush()
    csv_file.flush()
    return cell_signals, sig_id


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate():
    if not SIGNAL_CSV.exists():
        print("No signals CSV — run() first.")
        return None
    df = pd.read_csv(SIGNAL_CSV)
    if df.empty:
        print("Empty signals CSV.")
        return None
    print(f"Aggregating {len(df)} signal rows...")

    rows = []
    grp_keys = ["symbol", "cohort", "timeframe", "variant", "direction"]
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
            payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else (999.0 if avg_win > 0 else 0.0)
            mean_pct = float(net_pct.mean())
            std_pct = float(net_pct.std(ddof=0))
            sharpe = (mean_pct / std_pct * wr) if std_pct > 0 else 0.0
            expectancy = (wr * avg_win) - ((1 - wr) * abs(avg_loss))
            mean_mfe_pct = float(mfe_pct.mean()) if len(mfe_pct) > 0 else 0.0
            mean_mae_pct = float(mae_pct.mean()) if len(mae_pct) > 0 else 0.0
            cap_eff = (mean_pct / mean_mfe_pct) if mean_mfe_pct > 0 else 0.0
            rows.append({
                "symbol": keys[0],
                "cohort": keys[1],
                "timeframe": keys[2],
                "variant": keys[3],
                "direction": keys[4],
                "exit_policy": p,
                "n_signals": n,
                "mean_net_pts": round(float(net_pts.mean()), 4),
                "std_net_pts": round(float(net_pts.std(ddof=0)), 4),
                "median_net_pts": round(float(net_pts.median()), 4),
                "mean_net_pct": round(mean_pct, 6),
                "std_net_pct": round(std_pct, 6),
                "win_rate": round(wr * 100.0, 2),
                "avg_win_pct": round(avg_win * 100.0, 4),
                "avg_loss_pct": round(avg_loss * 100.0, 4),
                "payoff_ratio": round(payoff, 3),
                "mean_mfe_pct": round(mean_mfe_pct * 100.0, 4),
                "mean_mae_pct": round(mean_mae_pct * 100.0, 4),
                "capture_efficiency": round(cap_eff, 4),
                "sharpe_score": round(sharpe, 5),
                "expectancy_pct": round(expectancy * 100.0, 4),
            })
    rows.sort(key=lambda r: r["sharpe_score"], reverse=True)
    if not rows:
        print("No cells with n>=5.")
        return None
    fieldnames = list(rows[0].keys())
    with RANKING_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} ranked cells to {RANKING_CSV}")
    return rows


# ---------------------------------------------------------------------------
# Volume Leaders analysis
# ---------------------------------------------------------------------------

def compute_volume_leaders(rows: list[dict]) -> list[dict]:
    """For each stock, find best-Sharpe cell (n>=10) and robust-cell counts."""
    by_sym: dict[str, list[dict]] = {}
    for r in rows:
        if r["n_signals"] < 10:
            continue
        by_sym.setdefault(r["symbol"], []).append(r)

    leaders = []
    for sym, srows in by_sym.items():
        # Best cell by sharpe
        best = max(srows, key=lambda r: r["sharpe_score"])
        # High-quality cells: Sharpe >= 0.4 with n>=10
        high_q = [r for r in srows if r["sharpe_score"] >= 0.4]
        # Mid-quality cells: Sharpe >= 0.3 with n>=10
        mid_q = [r for r in srows if r["sharpe_score"] >= 0.3]
        # Direction asymmetry: best Sharpe per direction
        long_best = max([r for r in srows if r["direction"] == "long"],
                        key=lambda r: r["sharpe_score"], default=None)
        short_best = max([r for r in srows if r["direction"] == "short"],
                         key=lambda r: r["sharpe_score"], default=None)
        # TF sweet spot: best Sharpe per TF
        tf_best = {}
        for tf in TIMEFRAMES:
            tf_rows = [r for r in srows if r["timeframe"] == tf]
            if tf_rows:
                tf_best[tf] = max(tf_rows, key=lambda r: r["sharpe_score"])

        leaders.append({
            "symbol": sym,
            "cohort": best["cohort"],
            "best_timeframe": best["timeframe"],
            "best_variant": best["variant"],
            "best_direction": best["direction"],
            "best_exit_policy": best["exit_policy"],
            "best_n": best["n_signals"],
            "best_mean_pct": best["mean_net_pct"] * 100,
            "best_wr": best["win_rate"],
            "best_sharpe": best["sharpe_score"],
            "best_payoff": best["payoff_ratio"],
            "n_high_quality_cells": len(high_q),   # Sharpe >= 0.4
            "n_mid_quality_cells": len(mid_q),     # Sharpe >= 0.3
            "long_best_sharpe": long_best["sharpe_score"] if long_best else 0.0,
            "long_best_n": long_best["n_signals"] if long_best else 0,
            "short_best_sharpe": short_best["sharpe_score"] if short_best else 0.0,
            "short_best_n": short_best["n_signals"] if short_best else 0,
            "tf5_best_sharpe": tf_best.get("5min", {}).get("sharpe_score", 0.0),
            "tf10_best_sharpe": tf_best.get("10min", {}).get("sharpe_score", 0.0),
            "tf15_best_sharpe": tf_best.get("15min", {}).get("sharpe_score", 0.0),
        })

    leaders.sort(key=lambda r: r["best_sharpe"], reverse=True)

    # Write CSV
    if leaders:
        fieldnames = list(leaders[0].keys())
        with LEADERS_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in leaders:
                w.writerow(r)
        print(f"Wrote volume leaders to {LEADERS_CSV} ({len(leaders)} stocks)")

    return leaders


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_results_md(rows: list[dict], leaders: list[dict]):
    if not rows:
        return
    df = pd.read_csv(SIGNAL_CSV)

    by_dir = df.groupby("direction").size().to_dict()
    by_cohort = df.groupby("cohort").size().to_dict()

    lines = ["# Volume-Confirmed First-Candle Breakout — EXPANDED (79 stocks)\n\n"]
    lines.append("## Setup\n\n")
    lines.append(f"- Universe: 79 stocks (Cohort A = 10 stocks since 2018, Cohort B = 69 stocks since 2024-03-18)\n")
    lines.append(f"- Period end: 2026-03-25 (data cap)\n")
    lines.append(f"- Timeframes: 5min, **10min (NEW)**, 15min — first candle of session\n")
    lines.append(f"- Variant grid: vol_mult ∈ {{1.5, 2.0, 3.0}} × gap_pct ∈ {{0%, 0.3%, 0.5%, off}} × RSI ∈ {{off, on(40/60)}}\n")
    lines.append(f"- Direction: long & short\n")
    lines.append(f"- 13 exit policies tested per signal in parallel\n")
    lines.append(f"- Total signals fired: **{len(df):,}**\n")
    lines.append(f"  - Long: {by_dir.get('long', 0):,}\n")
    lines.append(f"  - Short: {by_dir.get('short', 0):,}\n")
    lines.append(f"  - Cohort A: {by_cohort.get('A', 0):,}\n")
    lines.append(f"  - Cohort B: {by_cohort.get('B', 0):,}\n\n")

    promote = [r for r in rows if r["n_signals"] >= 10 and r["mean_net_pct"] > 0]

    # ------------------------------------------------------------------ TOP 10
    lines.append("## Top 10 configurations across all stocks (by Sharpe, n>=10, mean>0)\n\n")
    lines.append("| Symbol | Coh | TF | Variant | Dir | ExitPolicy | n | mean% | WR% | Payoff | Sharpe |\n")
    lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|\n")
    for r in promote[:10]:
        lines.append(
            f"| {r['symbol']} | {r['cohort']} | {r['timeframe']} | {r['variant']} | {r['direction']} | "
            f"{r['exit_policy']} | {r['n_signals']} | {r['mean_net_pct']*100:.3f} | "
            f"{r['win_rate']:.1f} | {r['payoff_ratio']:.2f} | {r['sharpe_score']:.4f} |\n"
        )

    # ------------------------------------------------------------ VOLUME LEADERS
    lines.append("\n## Top 15 Volume Leaders (per-stock best Sharpe, n>=10)\n\n")
    lines.append("| Rank | Symbol | Coh | TF | Variant | Dir | Exit | n | mean% | WR% | Payoff | Sharpe | HiQ cells | MidQ cells |\n")
    lines.append("|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(leaders[:15], 1):
        lines.append(
            f"| {i} | {r['symbol']} | {r['cohort']} | {r['best_timeframe']} | {r['best_variant']} | "
            f"{r['best_direction']} | {r['best_exit_policy']} | {r['best_n']} | "
            f"{r['best_mean_pct']:.3f} | {r['best_wr']:.1f} | {r['best_payoff']:.2f} | "
            f"{r['best_sharpe']:.4f} | {r['n_high_quality_cells']} | {r['n_mid_quality_cells']} |\n"
        )

    # ----------------------------------------------------------- Promote candidates
    promote_candidates = [
        r for r in leaders
        if r["best_sharpe"] >= 0.5
        and r["best_n"] >= 15
        and r["n_mid_quality_cells"] >= 3
    ]
    lines.append("\n## Promote candidates (Sharpe>=0.5, n>=15, MidQ_cells>=3)\n\n")
    lines.append("These pass the robustness gate — best cell strong, n>=15, AND signal "
                 "consistent across at least 3 different variants (Sharpe>=0.3 each).\n\n")
    if promote_candidates:
        lines.append("| Symbol | Coh | TF | Variant | Dir | Exit | n | mean% | WR% | Sharpe | HiQ | MidQ | LongBestSharpe | ShortBestSharpe |\n")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in promote_candidates:
            lines.append(
                f"| {r['symbol']} | {r['cohort']} | {r['best_timeframe']} | {r['best_variant']} | "
                f"{r['best_direction']} | {r['best_exit_policy']} | {r['best_n']} | "
                f"{r['best_mean_pct']:.3f} | {r['best_wr']:.1f} | {r['best_sharpe']:.4f} | "
                f"{r['n_high_quality_cells']} | {r['n_mid_quality_cells']} | "
                f"{r['long_best_sharpe']:.3f} | {r['short_best_sharpe']:.3f} |\n"
            )
    else:
        lines.append("_No stocks passed the gate._\n")

    # ----------------------------------------------------------- Direction asymmetry
    lines.append("\n## Direction asymmetry (top 20 leaders)\n\n")
    lines.append("| Symbol | Long_best_Sharpe (n) | Short_best_Sharpe (n) | Bias |\n")
    lines.append("|---|---:|---:|---|\n")
    for r in leaders[:20]:
        L = r["long_best_sharpe"]
        S = r["short_best_sharpe"]
        if L > S * 1.5 and L > 0.2:
            bias = "LONG"
        elif S > L * 1.5 and S > 0.2:
            bias = "SHORT"
        elif max(L, S) < 0.2:
            bias = "weak"
        else:
            bias = "both"
        lines.append(f"| {r['symbol']} | {L:.3f} (n={r['long_best_n']}) | "
                     f"{S:.3f} (n={r['short_best_n']}) | {bias} |\n")

    # ----------------------------------------------------------- TF sweet spot
    lines.append("\n## Timeframe sweet spot (top 20 leaders)\n\n")
    lines.append("| Symbol | 5min | 10min | 15min | Best |\n")
    lines.append("|---|---:|---:|---:|---|\n")
    for r in leaders[:20]:
        scores = [("5min", r["tf5_best_sharpe"]),
                  ("10min", r["tf10_best_sharpe"]),
                  ("15min", r["tf15_best_sharpe"])]
        best_tf = max(scores, key=lambda x: x[1])[0]
        lines.append(f"| {r['symbol']} | {r['tf5_best_sharpe']:.3f} | "
                     f"{r['tf10_best_sharpe']:.3f} | {r['tf15_best_sharpe']:.3f} | {best_tf} |\n")

    # ----------------------------------------------------------- Cohort comparison
    cohort_a_leaders = [r for r in leaders if r["cohort"] == "A"]
    cohort_b_leaders = [r for r in leaders if r["cohort"] == "B"]
    lines.append("\n## Cohort A (10 stocks, 8 yrs) vs Cohort B (69 stocks, 2 yrs)\n\n")
    def _ag(cs):
        if not cs:
            return 0, 0.0, 0.0
        return len(cs), float(np.mean([c["best_sharpe"] for c in cs])), \
               float(np.mean([c["best_mean_pct"] for c in cs]))
    aN, aSh, aMn = _ag(cohort_a_leaders)
    bN, bSh, bMn = _ag(cohort_b_leaders)
    lines.append("| Cohort | n_stocks | avg_best_Sharpe | avg_best_mean% |\n")
    lines.append("|---|---:|---:|---:|\n")
    lines.append(f"| A (long history) | {aN} | {aSh:.4f} | {aMn:.4f} |\n")
    lines.append(f"| B (2-year only)  | {bN} | {bSh:.4f} | {bMn:.4f} |\n")

    # ----------------------------------------------------------- Exit policy comparison
    lines.append("\n## Exit policy comparison (across cells with n>=10, mean>0)\n\n")
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
    lines.append("| ExitPolicy | n_cells | avg_mean% | avg_WR% | avg_capEff | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for p, nc, ams, awr, acap, ash in pol_summary:
        lines.append(f"| {p} | {nc} | {ams:.4f} | {awr:.1f} | {acap:.2f} | {ash:.4f} |\n")

    # ----------------------------------------------------------- TF comparison
    lines.append("\n## Timeframe comparison (cells with n>=10, mean>0)\n\n")
    lines.append("| TF | n_cells | avg_mean% | avg_WR% | avg_Sharpe |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for tf in TIMEFRAMES:
        cs = [r for r in promote if r["timeframe"] == tf]
        if not cs:
            lines.append(f"| {tf} | 0 | - | - | - |\n")
            continue
        m = float(np.mean([c["mean_net_pct"] for c in cs])) * 100
        w = float(np.mean([c["win_rate"] for c in cs]))
        s = float(np.mean([c["sharpe_score"] for c in cs]))
        lines.append(f"| {tf} | {len(cs)} | {m:.4f} | {w:.1f} | {s:.4f} |\n")

    # ----------------------------------------------------------- Comparison to prior
    lines.append("\n## Comparison to prior 10-stock run (research/30)\n\n")
    rel_leaders = [r for r in leaders if r["symbol"] == "RELIANCE"]
    if rel_leaders:
        r = rel_leaders[0]
        lines.append(f"- RELIANCE best in this expanded run: TF={r['best_timeframe']}, "
                     f"Variant={r['best_variant']}, Dir={r['best_direction']}, "
                     f"Exit={r['best_exit_policy']}, n={r['best_n']}, "
                     f"mean%={r['best_mean_pct']:.3f}, Sharpe={r['best_sharpe']:.4f}\n")
        lines.append(f"- Prior run RELIANCE best: 15min, l_vm2.0_gap0.000_rsi40_60, T_NO, n=11, "
                     f"mean=1.094%, Sharpe=1.0451\n")
    lines.append(f"- Stocks beating the RELIANCE prior-best Sharpe (1.045): "
                 f"{[r['symbol'] for r in leaders if r['best_sharpe'] > 1.045][:10]}\n")

    RESULTS_MD.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS_MD}")


# ---------------------------------------------------------------------------
# Status doc
# ---------------------------------------------------------------------------

def write_status_md(state: str, n_signals: int = 0, n_done_stocks: int = 0,
                    started_at: Optional[str] = None,
                    last_stock: Optional[str] = None):
    lines = [
        "# Volume-Breakout Sweep (EXPANDED, 79 stocks) — Status\n\n",
        "## Goal\n\n",
        "Run the volume-confirmed first-candle breakout signal generator (research/30) "
        "across the full 79-stock universe with 5-min intraday data, including a NEW "
        "10-min timeframe, to find the universe-wide volume leaders.\n\n",
        "## Universe\n\n",
        f"- Cohort A (10 stocks since 2018-01-01): {', '.join(COHORT_A)}\n",
        f"- Cohort B (69 stocks since 2024-03-18): {', '.join(COHORT_B)}\n\n",
        "## Variant Grid\n\n",
        f"- timeframes: {TIMEFRAMES}\n",
        f"- vol_mult: {VOL_MULTS}\n",
        f"- gap_pct: {[g for g in GAP_PCTS]}\n",
        f"- rsi_modes: {RSI_MODES}\n",
        f"- directions: {DIRECTIONS}\n",
        f"- Cells per stock: {len(TIMEFRAMES) * len(VOL_MULTS) * len(GAP_PCTS) * len(RSI_MODES) * len(DIRECTIONS)}\n",
        f"- Total cells: {len(ALL_STOCKS) * len(TIMEFRAMES) * len(VOL_MULTS) * len(GAP_PCTS) * len(RSI_MODES) * len(DIRECTIONS)}\n\n",
        "## Status\n\n",
        f"- State: **{state}**\n",
    ]
    if started_at:
        lines.append(f"- Started: {started_at}\n")
    if last_stock:
        lines.append(f"- Last completed stock: {last_stock}\n")
    lines.append(f"- Stocks completed: {n_done_stocks} / {len(ALL_STOCKS)}\n")
    lines.append(f"- Signals logged: {n_signals:,}\n\n")

    lines.append("## Crash Recovery\n\n")
    lines.append("Script is **resumable**. To resume:\n\n")
    lines.append("```bash\npython research/30b_volume_breakout_expanded/scripts/run_volbreakout_expanded.py\n```\n\n")
    lines.append("It reads `results/volbreakout_signals.csv`, builds a (symbol, tf, variant, "
                 "direction, date) skip-set, and only computes signals not yet logged.\n\n")
    lines.append("If aggregation/markdown-write was the only step that failed, run with "
                 "`--aggregate-only`.\n\n")
    lines.append("## Outputs\n\n")
    lines.append("- `results/volbreakout_signals.csv` (per-signal x exit-policy, gitignored)\n")
    lines.append("- `results/volbreakout_ranking.csv` (per-cell summary)\n")
    lines.append("- `results/volume_leaders.csv` (per-stock leaderboard)\n")
    lines.append("- `results/RESULTS.md` (final report)\n\n")
    STATUS_MD.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(stocks: Optional[list[str]] = None):
    stocks = stocks or ALL_STOCKS
    write_status_md("STARTING", started_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

    fieldnames = build_csv_header()
    if not SIGNAL_CSV.exists():
        with SIGNAL_CSV.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    done_keys = load_done_keys()
    print(f"== Volume-breakout EXPANDED sweep ==")
    print(f"Universe: {len(stocks)} stocks  |  TFs: {TIMEFRAMES}")
    cells_per = len(TIMEFRAMES) * len(VOL_MULTS) * len(GAP_PCTS) * len(RSI_MODES) * len(DIRECTIONS)
    print(f"Cells/stock: {cells_per}  |  Total cells: {len(stocks) * cells_per}")
    print(f"Already-logged signal rows: {len(done_keys):,}")

    # Build a (symbol, tf, variant, direction) -> set(date) lookup
    done_by_cell: dict[tuple, set[str]] = {}
    for k in done_keys:
        cell = (k[0], k[1], k[2], k[3])
        done_by_cell.setdefault(cell, set()).add(k[4])

    sig_id = len(done_keys)

    # Track which stocks already have any rows so we can resume cleanly
    done_syms = {k[0] for k in done_keys}
    already_complete_syms = set()
    # A stock is "done" if every cell-key has been written. We approximate
    # by counting how many cells it has covered. We'll just rely on the
    # row-level skip — it's fast enough.

    csv_file = SIGNAL_CSV.open("a", newline="", buffering=1)
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    started_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    n_done = 0
    t0_global = _time.time()

    for i, sym in enumerate(stocks, 1):
        t0 = _time.time()
        n_sigs, sig_id = process_symbol(sym, csv_writer, csv_file, done_by_cell, sig_id)
        elapsed = _time.time() - t0
        n_done += 1
        # Count cells reached for this stock
        sym_cells = sum(1 for k in done_by_cell.keys() if k[0] == sym) + (
            len(set((sym, *k[1:]) for k in done_by_cell.keys() if k[0] == sym))
        )
        # simpler: just count signals just produced + previously
        prior_sigs = sum(len(v) for k, v in done_by_cell.items() if k[0] == sym)
        total_for_sym = prior_sigs + n_sigs
        print(f"[{i}/{len(stocks)}] {sym:14s} — {n_sigs} new signals "
              f"({total_for_sym} total) in {elapsed:5.1f}s "
              f"(elapsed {(_time.time() - t0_global)/60:5.1f} min)", flush=True)
        # Update status doc every 5 stocks
        if i % 5 == 0 or i == len(stocks):
            cur_n_signals = (sum(1 for _ in open(SIGNAL_CSV, encoding="utf-8")) - 1)
            write_status_md("RUNNING", n_signals=cur_n_signals,
                            n_done_stocks=i, started_at=started_at,
                            last_stock=sym)

    csv_file.close()
    cur_n_signals = (sum(1 for _ in open(SIGNAL_CSV, encoding="utf-8")) - 1)
    write_status_md("AGGREGATING", n_signals=cur_n_signals,
                    n_done_stocks=len(stocks), started_at=started_at,
                    last_stock=stocks[-1])

    print(f"\nTotal elapsed: {(_time.time() - t0_global)/60:.1f} min")
    print(f"Signals on disk: {cur_n_signals:,}")


def aggregate_and_report():
    rows = aggregate()
    if not rows:
        return
    leaders = compute_volume_leaders(rows)
    write_results_md(rows, leaders)
    cur_n_signals = (sum(1 for _ in open(SIGNAL_CSV, encoding="utf-8")) - 1)
    write_status_md("COMPLETE", n_signals=cur_n_signals,
                    n_done_stocks=len(ALL_STOCKS),
                    started_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    last_stock=ALL_STOCKS[-1])


def main():
    args = sys.argv[1:]
    if "--aggregate-only" in args:
        aggregate_and_report()
        return
    if "--stocks" in args:
        idx = args.index("--stocks")
        stocks = args[idx + 1].split(",")
        run_all(stocks)
    else:
        run_all()
    aggregate_and_report()


if __name__ == "__main__":
    main()
