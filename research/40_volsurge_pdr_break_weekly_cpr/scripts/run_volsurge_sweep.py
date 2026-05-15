"""VOLSURGE + Prev-Day/Week-Range Break, gated by Narrow Weekly CPR — sweep.

Faithfully encodes the 10 examples in
`VOLSURGE_PDR_BREAK_WEEKLY_CPR_INTRADAY_SWEEP_STATUS.md` (see signal_lib.py
for the per-rule mapping). Structure copied from research/30b
(`run_volbreakout_expanded.py`): resumable signal-gen via a skip-set from the
existing CSV, 13 exit policies scored in parallel per signal, incremental CSV
writes, streaming aggregation, RESULTS.md writer.

Universe (binding): sorted(FNO_LOT_SIZES) minus {TATAMOTORS, ZOMATO} = 79
F&O stocks. The sweep RUNS ON VPS; it reads the relative DB path so it works
unchanged there. 10/15/30/60-min are RESAMPLED from stored 5-min in-runner.

Grid (1,440 signal cells / stock; 13 exits scored in parallel per signal):
  Timeframe        : 5,10,15,30,60 min                          (5)
  Daily-trend def  : sma50, sma200, hh20                         (3)
  theta_cpr (%)    : 0.25, 0.50, 0.75, 1.00                      (4)
  Volume mult k    : 1.5, 2.0, 3.0                               (3)
  Clean-candle     : loose, strict                               (2)
  Clear-room       : off, on                                     (2)
  Carry            : same-day-close, carry<=M days               (2)
  -> 5*3*4*3*2*2*2 = 1,440 cells x 79 stocks, x13 exits at scoring.

Direction is set per day by the daily-trend selector (symmetric: up->long,
down->short, flat->no-trade), so long & short are both swept implicitly.

Usage:
  python run_volsurge_sweep.py                # full sweep then aggregate
  python run_volsurge_sweep.py --aggregate-only
  python run_volsurge_sweep.py --stocks RELIANCE,TCS
  python run_volsurge_sweep.py --smoke        # 10 snapshot stocks, tiny grid
"""

from __future__ import annotations

import csv
import logging
import math
import sqlite3
import sys
import time as _time
from datetime import time as dtime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.disable(logging.WARNING)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # research/40_volsurge_pdr_break_weekly_cpr
REPO_ROOT = HERE.parents[2]  # repo root (so `import services...` works as a script)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from signal_lib import (  # noqa: E402
    CLEAN_PRESETS,
    cpr_for_date,
    clear_room,
    daily_trend,
    is_clean_candle,
    opening_candles,
    range_escape,
    resample_5m,
    slot_baseline,
    swing_levels,
    trend_to_direction,
    volume_surge,
    weekly_cpr,
)

# ---------------------------------------------------------------------------
# DB path — relative so it works unchanged on VPS (binding rule)
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parents[3] / "backtest_data" / "market_data.db"

# ---------------------------------------------------------------------------
# Output map
# ---------------------------------------------------------------------------

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SIGNAL_CSV = RESULTS / "volsurge_signals.csv"
RANKING_CSV = RESULTS / "volsurge_ranking.csv"
LEADERS_CSV = RESULTS / "volsurge_leaders.csv"
RESULTS_MD = RESULTS / "RESULTS.md"
STATUS_MD = ROOT / "VOLSURGE_PDR_BREAK_WEEKLY_CPR_INTRADAY_SWEEP_STATUS.md"

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

TIMEFRAMES = ["5min", "10min", "15min", "30min", "60min"]
TREND_MODES = ["sma50", "sma200", "hh20"]
THETA_CPR = [0.25, 0.50, 0.75, 1.00]
VOL_MULTS = [1.5, 2.0, 3.0]
CLEAN_STRICT = ["loose", "strict"]
CLEAR_ROOM = [False, True]
CARRY = ["sameday", "carry"]

CARRY_MAX_DAYS = 5          # carry<=M days
CLEAR_ROOM_R_ATR = 1.0      # opposing-S/R margin (x daily ATR)
VOL_BASELINE_N = 20         # trailing same-slot opening-vol window

PERIOD_START = pd.Timestamp("2018-01-01")
PERIOD_END = pd.Timestamp("2026-05-15")   # VPS 5-min spans to 2026-05-15

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

SMOKE_STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "HINDUNILVR"]


def get_universe() -> list[str]:
    from services.data_manager import FNO_LOT_SIZES
    return sorted(set(FNO_LOT_SIZES) - {"TATAMOTORS", "ZOMATO"})


# ---------------------------------------------------------------------------
# Data loading (relative DB, full range — NOT the research/29 capped loader)
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"market_data.db not found at {DB_PATH}")
    return sqlite3.connect(str(DB_PATH))


@lru_cache(maxsize=8)
def load_5min(symbol: str) -> pd.DataFrame:
    sql = ("SELECT date, open, high, low, close, volume FROM market_data_unified "
           "WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=? ORDER BY date")
    with _connect() as con:
        df = pd.read_sql(sql, con, params=(
            symbol, PERIOD_START.strftime("%Y-%m-%d"),
            PERIOD_END.strftime("%Y-%m-%d") + " 23:59:59"))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


@lru_cache(maxsize=8)
def load_daily(symbol: str) -> pd.DataFrame:
    # pull from 2 years before period start so SMA200 + weekly CPR warm up
    dstart = (PERIOD_START - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    sql = ("SELECT date, open, high, low, close, volume FROM market_data_unified "
           "WHERE symbol=? AND timeframe='day' AND date>=? AND date<=? ORDER BY date")
    with _connect() as con:
        df = pd.read_sql(sql, con, params=(
            symbol, dstart, PERIOD_END.strftime("%Y-%m-%d")))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.set_index("date").sort_index()


# ---------------------------------------------------------------------------
# ATR helpers (mirror research/30b)
# ---------------------------------------------------------------------------

def daily_atr_series(daily: pd.DataFrame, n: int = 14) -> pd.Series:
    if daily.empty:
        return pd.Series(dtype=float)
    df = daily.sort_index()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def atr_for_signal(atr_daily: pd.Series, signal_date: pd.Timestamp) -> Optional[float]:
    if atr_daily.empty:
        return None
    prior = atr_daily.loc[atr_daily.index < pd.Timestamp(signal_date).normalize()]
    if prior.empty:
        return None
    v = float(prior.iloc[-1])
    if math.isnan(v) or v <= 0:
        return None
    return v


# ---------------------------------------------------------------------------
# Exit simulator — reuses research/30b's 13-policy implementation verbatim,
# generalised to carry-forward across multiple sessions (carry<=M days).
# ---------------------------------------------------------------------------

def simulate_exits(bars_after: pd.DataFrame, *, direction: str,
                   entry_price: float, stop_low: float, stop_high: float,
                   atr_daily_val: Optional[float]) -> dict:
    """`bars_after` = the 5-min bars from just after entry to the exit horizon
    (same session for sameday; up to carry<=M days for carry). stop_low/high
    define the structural stop (other side of the trigger candle)."""
    sign = 1 if direction == "long" else -1

    policies: dict[str, dict] = {p: {
        "alive": True, "exit_time": None, "exit_price": None,
        "exit_reason": None, "mfe_pts": 0.0, "mae_pts": 0.0,
        "running_extreme": entry_price,
    } for p in EXIT_POLICIES}

    if direction == "long":
        hard_sl_dist = max(entry_price - stop_low, 0.0)
    else:
        hard_sl_dist = max(stop_high - entry_price, 0.0)
    if hard_sl_dist <= 0:
        hard_sl_dist = entry_price * 0.001

    atr_sl_dists = {k: (None if atr_daily_val is None else k * atr_daily_val)
                    for k in (0.3, 0.5, 1.0)}
    r_target_dists = {1.0: hard_sl_dist, 1.5: 1.5 * hard_sl_dist,
                      2.0: 2.0 * hard_sl_dist, 3.0: 3.0 * hard_sl_dist}

    def _close(p, ts, ep, reason):
        if not policies[p]["alive"]:
            return
        policies[p].update(alive=False, exit_time=ts,
                           exit_price=float(ep), exit_reason=reason)

    if bars_after.empty:
        for p in EXIT_POLICIES:
            _close(p, pd.NaT, entry_price, "no_bars")
        return _format_results(policies, entry_price, sign)

    for ts, row in bars_after.iterrows():
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])

        favorable = (bar_high - entry_price) if direction == "long" else (entry_price - bar_low)
        adverse = (entry_price - bar_low) if direction == "long" else (bar_high - entry_price)
        favorable = max(favorable, 0.0)
        adverse = max(adverse, 0.0)

        for p in EXIT_POLICIES:
            if not policies[p]["alive"]:
                continue
            if direction == "long":
                if bar_high > policies[p]["running_extreme"]:
                    policies[p]["running_extreme"] = bar_high
            else:
                if bar_low < policies[p]["running_extreme"]:
                    policies[p]["running_extreme"] = bar_low
            if favorable > policies[p]["mfe_pts"]:
                policies[p]["mfe_pts"] = favorable
            if adverse > policies[p]["mae_pts"]:
                policies[p]["mae_pts"] = adverse

        if policies["T_HARD_SL"]["alive"]:
            if direction == "long" and bar_low <= entry_price - hard_sl_dist:
                _close("T_HARD_SL", ts, entry_price - hard_sl_dist, "hard_sl")
            elif direction == "short" and bar_high >= entry_price + hard_sl_dist:
                _close("T_HARD_SL", ts, entry_price + hard_sl_dist, "hard_sl")

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

        if policies["T_STEP_TRAIL"]["alive"]:
            ext = policies["T_STEP_TRAIL"]["running_extreme"]
            mfe_R = ((ext - entry_price) if direction == "long"
                     else (entry_price - ext)) / hard_sl_dist
            if mfe_R >= 3.0:
                sl_offset = 1.5 * hard_sl_dist
            elif mfe_R >= 1.5:
                sl_offset = 0.5 * hard_sl_dist
            elif mfe_R >= 0.5:
                sl_offset = 0.0
            else:
                sl_offset = -hard_sl_dist
            if direction == "long":
                sl_price = entry_price + sl_offset
                if bar_low <= sl_price:
                    _close("T_STEP_TRAIL", ts, sl_price,
                           "step_trail" if sl_offset > 0
                           else ("breakeven" if sl_offset == 0 else "hard_sl"))
            else:
                sl_price = entry_price - sl_offset
                if bar_high >= sl_price:
                    _close("T_STEP_TRAIL", ts, sl_price,
                           "step_trail" if sl_offset > 0
                           else ("breakeven" if sl_offset == 0 else "hard_sl"))

        if not any(p["alive"] for p in policies.values()):
            break

    last_ts = bars_after.index[-1]
    last_close = float(bars_after["close"].iloc[-1])
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
        out[p] = {
            "exit_time": st["exit_time"],
            "exit_price": ep,
            "exit_reason": st["exit_reason"] or "n/a",
            "net_pts": net_pts,
            "net_pct": net_pts / entry_price if entry_price > 0 else 0.0,
            "mfe_pts": st["mfe_pts"],
            "mae_pts": st["mae_pts"],
        }
    return out


# ---------------------------------------------------------------------------
# CSV header / resume
# ---------------------------------------------------------------------------

def build_csv_header() -> list[str]:
    base = [
        "signal_id", "symbol", "timeframe", "variant", "direction",
        "date", "signal_time", "entry_price",
        "prev_day_high", "prev_day_low", "prev_week_high", "prev_week_low",
        "cpr_width_pct", "theta_cpr", "vol_mult", "vol_ratio",
        "trend_mode", "clean", "clearroom", "carry",
        "trig_open", "trig_high", "trig_low", "trig_close", "trig_volume",
        "vol_baseline", "atr_daily",
    ]
    for p in EXIT_POLICIES:
        for c in ("exit_time", "exit_price", "exit_reason",
                  "net_pts", "net_pct", "mfe_pts", "mae_pts", "hold_min"):
            base.append(f"{p}__{c}")
    return base


def load_done_keys() -> set[tuple]:
    """Skip-set of (symbol, timeframe, variant, direction, date)."""
    if not SIGNAL_CSV.exists():
        return set()
    done = set()
    with SIGNAL_CSV.open("r", newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["symbol"], row["timeframe"], row["variant"],
                      row["direction"], row["date"]))
    return done


def variant_tag(tf: str, trend_mode: str, theta: float, k: float,
                clean: str, clearroom: bool, carry: str) -> str:
    return (f"{tf}_{trend_mode}_cpr{theta:.2f}_vm{k}_{clean}"
            f"_cr{'on' if clearroom else 'off'}_{carry}")


# ---------------------------------------------------------------------------
# Per-stock processor
# ---------------------------------------------------------------------------

def process_symbol(sym: str, writer: csv.DictWriter, fh,
                    done_by_cell: dict, sig_id: int, grid: dict) -> tuple[int, int]:
    try:
        df5 = load_5min(sym)
    except Exception as e:
        print(f"  [{sym}] load_5min failed: {e}", flush=True)
        return 0, sig_id
    if df5.empty:
        print(f"  [{sym}] empty 5-min", flush=True)
        return 0, sig_id

    daily = load_daily(sym)
    if daily.empty or len(daily) < 30:
        print(f"  [{sym}] insufficient daily", flush=True)
        return 0, sig_id

    atr_d = daily_atr_series(daily, 14)
    cpr_tbl = weekly_cpr(daily)
    if cpr_tbl.empty:
        print(f"  [{sym}] no weekly CPR table", flush=True)
        return 0, sig_id

    # prior-DAY hi/lo keyed by trading date (shifted so day D sees D-1).
    pdh = daily["high"].shift(1)
    pdl = daily["low"].shift(1)

    trend_cache = {m: daily_trend(daily, m) for m in grid["trend_modes"]}

    # Resample once per timeframe; opening candles + same-slot vol baseline.
    tf_data = {}
    for tf in grid["timeframes"]:
        dtf = resample_5m(df5, tf)
        if dtf.empty:
            continue
        oc = opening_candles(dtf)
        if oc.empty:
            continue
        oc = oc.copy()
        oc["_date"] = oc.index.normalize()
        tf_data[tf] = {"oc": oc, "ovol": oc["volume"].reset_index(drop=True)}

    n_new = 0
    for tf, td in tf_data.items():
        oc = td["oc"]
        ovol = td["ovol"]
        for ci in range(len(oc)):
            row = oc.iloc[ci]
            day = pd.Timestamp(row["_date"])
            ds = day.strftime("%Y-%m-%d")
            sig_time = oc.index[ci]

            cpr = cpr_for_date(cpr_tbl, day)
            if cpr is None:
                continue
            cpr_w = cpr["width_pct"]
            if cpr_w is None or np.isnan(cpr_w):
                continue
            pw_hi, pw_lo = cpr["prev_week_high"], cpr["prev_week_low"]

            if day not in pdh.index:
                continue
            pd_hi = pdh.loc[day]
            pd_lo = pdl.loc[day]
            if pd.isna(pd_hi) or pd.isna(pd_lo):
                continue

            o = float(row["open"]); h = float(row["high"])
            l = float(row["low"]); c = float(row["close"])
            tvol = float(row["volume"])
            baseline = slot_baseline(ovol, ci, VOL_BASELINE_N)
            if baseline is None:
                continue
            vr = tvol / baseline if baseline > 0 else 0.0

            atr_val = atr_for_signal(atr_d, day)
            sh, sl = swing_levels(daily, day, 20)

            for trend_mode in grid["trend_modes"]:
                tser = trend_cache[trend_mode]
                prior = tser.loc[tser.index < day]
                if prior.empty:
                    continue
                direction = trend_to_direction(prior.iloc[-1])
                if direction is None:                       # flat -> no trade
                    continue

                if not range_escape(c, pd_hi, pd_lo, pw_hi, pw_lo, direction):
                    continue

                for clean in grid["clean"]:
                    b_min, zone = CLEAN_PRESETS[clean]
                    if not is_clean_candle(o, h, l, c, direction, b_min, zone):
                        continue
                    for k in grid["vol_mults"]:
                        if not volume_surge(tvol, baseline, k):
                            continue
                        for theta in grid["theta_cpr"]:
                            if cpr_w > theta:               # week not narrow
                                continue
                            for clearroom in grid["clear_room"]:
                                if clearroom and not clear_room(
                                        c, direction, atr_val, sh, sl,
                                        CLEAR_ROOM_R_ATR):
                                    continue
                                for carry in grid["carry"]:
                                    vtag = variant_tag(tf, trend_mode, theta,
                                                       k, clean, clearroom, carry)
                                    ckey = (sym, tf, vtag, direction)
                                    if ds in done_by_cell.get(ckey, set()):
                                        continue

                                    # exit horizon
                                    if carry == "sameday":
                                        bars = df5.loc[
                                            (df5.index > sig_time)
                                            & (df5.index.normalize() == day)
                                            & (df5.index.time <= EOD_TIME)]
                                    else:
                                        horizon = day + pd.Timedelta(
                                            days=CARRY_MAX_DAYS + 4)
                                        bars = df5.loc[
                                            (df5.index > sig_time)
                                            & (df5.index <= horizon)]
                                        # cap to CARRY_MAX_DAYS trading days
                                        if not bars.empty:
                                            tdays = sorted(bars.index
                                                           .normalize().unique())
                                            keep = set(tdays[:CARRY_MAX_DAYS])
                                            bars = bars.loc[
                                                bars.index.normalize()
                                                .isin(keep)]
                                            bars = bars.loc[
                                                bars.index.time <= EOD_TIME]

                                    if direction == "long":
                                        stop_low, stop_high = l, h
                                    else:
                                        stop_low, stop_high = l, h

                                    ex = simulate_exits(
                                        bars, direction=direction,
                                        entry_price=c, stop_low=stop_low,
                                        stop_high=stop_high,
                                        atr_daily_val=atr_val)

                                    rec = {
                                        "signal_id": sig_id, "symbol": sym,
                                        "timeframe": tf, "variant": vtag,
                                        "direction": direction, "date": ds,
                                        "signal_time": sig_time.strftime(
                                            "%Y-%m-%d %H:%M:%S"),
                                        "entry_price": round(c, 4),
                                        "prev_day_high": round(float(pd_hi), 4),
                                        "prev_day_low": round(float(pd_lo), 4),
                                        "prev_week_high": round(pw_hi, 4),
                                        "prev_week_low": round(pw_lo, 4),
                                        "cpr_width_pct": round(float(cpr_w), 4),
                                        "theta_cpr": theta, "vol_mult": k,
                                        "vol_ratio": round(vr, 3),
                                        "trend_mode": trend_mode,
                                        "clean": clean,
                                        "clearroom": "on" if clearroom else "off",
                                        "carry": carry,
                                        "trig_open": round(o, 4),
                                        "trig_high": round(h, 4),
                                        "trig_low": round(l, 4),
                                        "trig_close": round(c, 4),
                                        "trig_volume": int(tvol),
                                        "vol_baseline": round(baseline, 1),
                                        "atr_daily": round(atr_val, 4)
                                            if atr_val else "",
                                    }
                                    for p in EXIT_POLICIES:
                                        r = ex[p]
                                        et = r["exit_time"]
                                        if et is None or pd.isna(et):
                                            hold, ets = 0, ""
                                        else:
                                            try:
                                                hold = int((et - sig_time)
                                                           .total_seconds() // 60)
                                            except Exception:
                                                hold = 0
                                            ets = et.strftime(
                                                "%Y-%m-%d %H:%M:%S")
                                        rec[f"{p}__exit_time"] = ets
                                        rec[f"{p}__exit_price"] = round(
                                            r["exit_price"], 4)
                                        rec[f"{p}__exit_reason"] = r["exit_reason"]
                                        rec[f"{p}__net_pts"] = round(
                                            r["net_pts"], 4)
                                        rec[f"{p}__net_pct"] = round(
                                            r["net_pct"], 6)
                                        rec[f"{p}__mfe_pts"] = round(
                                            r["mfe_pts"], 4)
                                        rec[f"{p}__mae_pts"] = round(
                                            r["mae_pts"], 4)
                                        rec[f"{p}__hold_min"] = hold
                                    writer.writerow(rec)
                                    sig_id += 1
                                    n_new += 1
        fh.flush()
    fh.flush()
    return n_new, sig_id


# ---------------------------------------------------------------------------
# Aggregation (streaming-friendly: groupby on the CSV)
# ---------------------------------------------------------------------------

def aggregate() -> Optional[list[dict]]:
    if not SIGNAL_CSV.exists():
        print("No signals CSV — run() first.")
        return None
    df = pd.read_csv(SIGNAL_CSV)
    if df.empty:
        print("Empty signals CSV.")
        return None
    print(f"Aggregating {len(df)} signal rows...")

    rows = []
    gkeys = ["symbol", "timeframe", "variant", "direction"]
    for keys, g in df.groupby(gkeys, sort=False):
        n = len(g)
        if n < 5:
            continue
        for p in EXIT_POLICIES:
            net_pct = g[f"{p}__net_pct"].astype(float)
            net_pts = g[f"{p}__net_pts"].astype(float)
            ep = g["entry_price"].astype(float)
            mfe_pct = g[f"{p}__mfe_pts"].astype(float) / ep
            mae_pct = g[f"{p}__mae_pts"].astype(float) / ep
            wins = int((net_pct > 0).sum())
            losses = int((net_pct <= 0).sum())
            wr = wins / n
            avg_win = float(net_pct[net_pct > 0].mean()) if wins else 0.0
            avg_loss = float(net_pct[net_pct <= 0].mean()) if losses else 0.0
            payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else (
                999.0 if avg_win > 0 else 0.0)
            mean_pct = float(net_pct.mean())
            std_pct = float(net_pct.std(ddof=0))
            # Sharpe-style score (meanR/stdR damped by WR) — per spec ranking.
            sharpe = (mean_pct / std_pct * wr) if std_pct > 0 else 0.0
            expectancy = (wr * avg_win) - ((1 - wr) * abs(avg_loss))
            mmfe = float(mfe_pct.mean()) if len(mfe_pct) else 0.0
            mmae = float(mae_pct.mean()) if len(mae_pct) else 0.0
            cap_eff = (mean_pct / mmfe) if mmfe > 0 else 0.0
            rows.append({
                "symbol": keys[0], "timeframe": keys[1],
                "variant": keys[2], "direction": keys[3],
                "exit_policy": p, "n_signals": n,
                "mean_net_pts": round(float(net_pts.mean()), 4),
                "std_net_pts": round(float(net_pts.std(ddof=0)), 4),
                "median_net_pts": round(float(net_pts.median()), 4),
                "mean_net_pct": round(mean_pct, 6),
                "std_net_pct": round(std_pct, 6),
                "win_rate": round(wr * 100.0, 2),
                "avg_win_pct": round(avg_win * 100.0, 4),
                "avg_loss_pct": round(avg_loss * 100.0, 4),
                "payoff_ratio": round(payoff, 3),
                "mean_mfe_pct": round(mmfe * 100.0, 4),
                "mean_mae_pct": round(mmae * 100.0, 4),
                "capture_efficiency": round(cap_eff, 4),
                "sharpe_score": round(sharpe, 5),
                "expectancy_pct": round(expectancy * 100.0, 4),
            })
    rows.sort(key=lambda r: r["sharpe_score"], reverse=True)
    if not rows:
        print("No cells with n>=5.")
        return None
    with RANKING_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} ranked cells -> {RANKING_CSV}")
    return rows


def compute_leaders(rows: list[dict]) -> list[dict]:
    by_sym: dict[str, list[dict]] = {}
    for r in rows:
        if r["n_signals"] < 10:
            continue
        by_sym.setdefault(r["symbol"], []).append(r)

    leaders = []
    for sym, sr in by_sym.items():
        best = max(sr, key=lambda r: r["sharpe_score"])
        hi_q = [r for r in sr if r["sharpe_score"] >= 0.4]
        mid_q = [r for r in sr if r["sharpe_score"] >= 0.3]
        lb = max([r for r in sr if r["direction"] == "long"],
                 key=lambda r: r["sharpe_score"], default=None)
        sb = max([r for r in sr if r["direction"] == "short"],
                 key=lambda r: r["sharpe_score"], default=None)
        tf_best = {}
        for tf in TIMEFRAMES:
            tr = [r for r in sr if r["timeframe"] == tf]
            if tr:
                tf_best[tf] = max(tr, key=lambda r: r["sharpe_score"])
        leaders.append({
            "symbol": sym, "best_timeframe": best["timeframe"],
            "best_variant": best["variant"],
            "best_direction": best["direction"],
            "best_exit_policy": best["exit_policy"],
            "best_n": best["n_signals"],
            "best_mean_pct": best["mean_net_pct"] * 100,
            "best_wr": best["win_rate"],
            "best_sharpe": best["sharpe_score"],
            "best_payoff": best["payoff_ratio"],
            "best_expectancy_pct": best["expectancy_pct"],
            "n_high_quality_cells": len(hi_q),
            "n_mid_quality_cells": len(mid_q),
            "long_best_sharpe": lb["sharpe_score"] if lb else 0.0,
            "long_best_n": lb["n_signals"] if lb else 0,
            "short_best_sharpe": sb["sharpe_score"] if sb else 0.0,
            "short_best_n": sb["n_signals"] if sb else 0,
            **{f"tf_{tf}_best_sharpe": tf_best.get(tf, {}).get("sharpe_score", 0.0)
               for tf in TIMEFRAMES},
        })
    leaders.sort(key=lambda r: r["best_sharpe"], reverse=True)
    if leaders:
        with LEADERS_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(leaders[0].keys()))
            w.writeheader()
            w.writerows(leaders)
        print(f"Wrote {len(leaders)} leaders -> {LEADERS_CSV}")
    return leaders


def write_results_md(rows: list[dict], leaders: list[dict]):
    if not rows:
        return
    df = pd.read_csv(SIGNAL_CSV)
    by_dir = df.groupby("direction").size().to_dict()
    L = ["# VOLSURGE + PDR/PWR-Break, Narrow Weekly CPR — Underlying Sweep\n\n"]
    L.append("## Setup\n\n")
    L.append(f"- Universe: 79 F&O stocks (FNO_LOT_SIZES minus TATAMOTORS, ZOMATO)\n")
    L.append(f"- Period: {PERIOD_START.date()} -> {PERIOD_END.date()} "
             f"(per-stock, clipped to available 5-min)\n")
    L.append(f"- Timeframes: {', '.join(TIMEFRAMES)} (10/15/30/60 resampled from 5-min)\n")
    L.append(f"- Grid: trend{TREND_MODES} x theta_cpr{THETA_CPR} x "
             f"k{VOL_MULTS} x clean{CLEAN_STRICT} x clearroom{CLEAR_ROOM} x "
             f"carry{CARRY}\n")
    L.append("- Direction = daily-trend selector (up->long, down->short, flat->skip)\n")
    L.append("- 13 exit policies scored in parallel per signal\n")
    L.append(f"- Total signal rows: **{len(df):,}** "
             f"(long {by_dir.get('long', 0):,} / short {by_dir.get('short', 0):,})\n\n")
    L.append("> Per ex#9: the confluence is a NECESSARY-but-NOT-SUFFICIENT "
             "probabilistic edge — judged on expectancy/Sharpe over a "
             "population, never on individual outcomes.\n\n")

    promote = [r for r in rows if r["n_signals"] >= 15 and r["mean_net_pct"] > 0]

    L.append("## Top 15 configs (Sharpe, n>=15, mean>0)\n\n")
    L.append("| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | "
             "Payoff | Exp% | Sharpe |\n")
    L.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in promote[:15]:
        L.append(f"| {r['symbol']} | {r['timeframe']} | {r['variant']} | "
                 f"{r['direction']} | {r['exit_policy']} | {r['n_signals']} | "
                 f"{r['mean_net_pct']*100:.3f} | {r['win_rate']:.1f} | "
                 f"{r['payoff_ratio']:.2f} | {r['expectancy_pct']:.3f} | "
                 f"{r['sharpe_score']:.4f} |\n")

    L.append("\n## Per-stock leaders (best Sharpe, n>=10)\n\n")
    L.append("| Rank | Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | "
             "Exp% | Sharpe | HiQ | MidQ |\n")
    L.append("|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(leaders[:20], 1):
        L.append(f"| {i} | {r['symbol']} | {r['best_timeframe']} | "
                 f"{r['best_variant']} | {r['best_direction']} | "
                 f"{r['best_exit_policy']} | {r['best_n']} | "
                 f"{r['best_mean_pct']:.3f} | {r['best_wr']:.1f} | "
                 f"{r['best_expectancy_pct']:.3f} | {r['best_sharpe']:.4f} | "
                 f"{r['n_high_quality_cells']} | {r['n_mid_quality_cells']} |\n")

    cand = [r for r in leaders if r["best_sharpe"] >= 0.5
            and r["best_n"] >= 15 and r["n_mid_quality_cells"] >= 3]
    L.append("\n## Promote candidates (Sharpe>=0.5, n>=15, MidQ>=3)\n\n")
    L.append("Robustness gate (research/34 style): best cell strong AND the "
             "edge survives across >=3 variants.\n\n")
    if cand:
        L.append("| Symbol | TF | Variant | Dir | Exit | n | mean% | WR% | "
                 "Sharpe | MidQ | LongSh | ShortSh |\n")
        L.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in cand:
            L.append(f"| {r['symbol']} | {r['best_timeframe']} | "
                     f"{r['best_variant']} | {r['best_direction']} | "
                     f"{r['best_exit_policy']} | {r['best_n']} | "
                     f"{r['best_mean_pct']:.3f} | {r['best_wr']:.1f} | "
                     f"{r['best_sharpe']:.4f} | {r['n_mid_quality_cells']} | "
                     f"{r['long_best_sharpe']:.3f} | "
                     f"{r['short_best_sharpe']:.3f} |\n")
    else:
        L.append("_No stocks passed the gate._\n")

    L.append("\n## Direction asymmetry (top 20 leaders)\n\n")
    L.append("| Symbol | Long Sharpe (n) | Short Sharpe (n) | Bias |\n")
    L.append("|---|---:|---:|---|\n")
    for r in leaders[:20]:
        Ls, Ss = r["long_best_sharpe"], r["short_best_sharpe"]
        if Ls > Ss * 1.5 and Ls > 0.2:
            bias = "LONG"
        elif Ss > Ls * 1.5 and Ss > 0.2:
            bias = "SHORT"
        elif max(Ls, Ss) < 0.2:
            bias = "weak"
        else:
            bias = "both"
        L.append(f"| {r['symbol']} | {Ls:.3f} (n={r['long_best_n']}) | "
                 f"{Ss:.3f} (n={r['short_best_n']}) | {bias} |\n")

    L.append("\n## Timeframe sweet-spot (cells n>=15, mean>0)\n\n")
    L.append("| TF | n_cells | avg_mean% | avg_WR% | avg_Sharpe |\n")
    L.append("|---|---:|---:|---:|---:|\n")
    for tf in TIMEFRAMES:
        cs = [r for r in promote if r["timeframe"] == tf]
        if not cs:
            L.append(f"| {tf} | 0 | - | - | - |\n")
            continue
        L.append(f"| {tf} | {len(cs)} | "
                 f"{np.mean([c['mean_net_pct'] for c in cs])*100:.4f} | "
                 f"{np.mean([c['win_rate'] for c in cs]):.1f} | "
                 f"{np.mean([c['sharpe_score'] for c in cs]):.4f} |\n")

    L.append("\n## Exit-policy comparison (cells n>=15, mean>0)\n\n")
    L.append("| ExitPolicy | n_cells | avg_mean% | avg_WR% | avg_Sharpe |\n")
    L.append("|---|---:|---:|---:|---:|\n")
    pol = {}
    for r in promote:
        pol.setdefault(r["exit_policy"], []).append(r)
    summ = []
    for p in EXIT_POLICIES:
        cs = pol.get(p, [])
        if not cs:
            summ.append((p, 0, 0.0, 0.0, 0.0))
            continue
        summ.append((p, len(cs),
                     float(np.mean([c["mean_net_pct"] for c in cs])) * 100,
                     float(np.mean([c["win_rate"] for c in cs])),
                     float(np.mean([c["sharpe_score"] for c in cs]))))
    summ.sort(key=lambda x: x[4], reverse=True)
    for p, nc, m, w, s in summ:
        L.append(f"| {p} | {nc} | {m:.4f} | {w:.1f} | {s:.4f} |\n")

    RESULTS_MD.write_text("".join(L), encoding="utf-8")
    print(f"Wrote {RESULTS_MD}")


# ---------------------------------------------------------------------------
# Status doc — append-only Crash Recovery + Files (handled by patch_status)
# ---------------------------------------------------------------------------

def update_status_state(state: str, n_signals: int = 0,
                        n_done: int = 0, n_total: int = 79):
    """Patch the live ## Status (event log) table's State line. Non-fatal."""
    try:
        txt = STATUS_MD.read_text(encoding="utf-8")
    except Exception:
        return
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    marker = "<!-- RUNNER_STATE -->"
    block = (f"{marker}\n**Runner state: {state}** — "
             f"stocks {n_done}/{n_total}, signals {n_signals:,} "
             f"(updated {stamp})\n")
    if marker in txt:
        head, _, rest = txt.partition(marker)
        _, _, tail = rest.partition("\n\n")
        txt = head + block + "\n" + tail
    else:
        txt = txt + "\n\n" + block
    try:
        STATUS_MD.write_text(txt, encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

FULL_GRID = {
    "timeframes": TIMEFRAMES, "trend_modes": TREND_MODES,
    "theta_cpr": THETA_CPR, "vol_mults": VOL_MULTS,
    "clean": CLEAN_STRICT, "clear_room": CLEAR_ROOM, "carry": CARRY,
}

SMOKE_GRID = {
    "timeframes": ["30min"], "trend_modes": ["sma50"],
    "theta_cpr": [1.00], "vol_mults": [2.0],
    "clean": ["loose"], "clear_room": [False], "carry": ["sameday"],
}


def run_all(stocks: list[str], grid: dict):
    fieldnames = build_csv_header()
    if not SIGNAL_CSV.exists():
        with SIGNAL_CSV.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    done = load_done_keys()
    done_by_cell: dict[tuple, set] = {}
    for k in done:
        done_by_cell.setdefault((k[0], k[1], k[2], k[3]), set()).add(k[4])
    sig_id = len(done)

    print(f"== VOLSURGE sweep ==  universe={len(stocks)}  "
          f"TFs={grid['timeframes']}  already-logged={len(done):,}", flush=True)
    update_status_state("RUNNING", len(done), 0, len(stocks))

    fh = SIGNAL_CSV.open("a", newline="", buffering=1)
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    t0 = _time.time()
    for i, sym in enumerate(stocks, 1):
        ts = _time.time()
        n_new, sig_id = process_symbol(sym, writer, fh, done_by_cell,
                                       sig_id, grid)
        print(f"[{i}/{len(stocks)}] {sym:12s} +{n_new} signals "
              f"in {_time.time()-ts:5.1f}s "
              f"(elapsed {(_time.time()-t0)/60:5.1f}m)", flush=True)
        if i % 5 == 0 or i == len(stocks):
            update_status_state("RUNNING", sig_id, i, len(stocks))
    fh.close()
    print(f"\nDone signal-gen in {(_time.time()-t0)/60:.1f}m  "
          f"rows={sig_id:,}", flush=True)
    update_status_state("AGGREGATING", sig_id, len(stocks), len(stocks))


def aggregate_and_report():
    rows = aggregate()
    if not rows:
        return
    leaders = compute_leaders(rows)
    write_results_md(rows, leaders)
    n = sum(1 for _ in SIGNAL_CSV.open(encoding="utf-8")) - 1
    update_status_state("COMPLETE", n, 79, 79)


def main():
    args = sys.argv[1:]
    if "--aggregate-only" in args:
        aggregate_and_report()
        return
    if "--smoke" in args:
        run_all(SMOKE_STOCKS, SMOKE_GRID)
        aggregate_and_report()
        return
    if "--stocks" in args:
        stocks = args[args.index("--stocks") + 1].split(",")
        run_all(stocks, FULL_GRID)
        aggregate_and_report()
        return
    run_all(get_universe(), FULL_GRID)
    aggregate_and_report()


if __name__ == "__main__":
    main()
