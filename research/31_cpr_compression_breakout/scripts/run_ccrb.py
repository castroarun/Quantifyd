"""CCRB — main runner.

Sweeps 79 stocks x 3 timeframes x (today_narrow x yesterday_ctx x
wide_thresh x narrow_range_thresh) x volume_mode x direction.
For each emitted signal, evaluates 13 exit policies in parallel and
appends a row to results/ccrb_signals.csv (flush-per-row, resumable).

Variant tag format:
  t{today_narrow:.2f}_ctx{ctx}_w{wide:.2f}_n{narrow:.2f}_{tf}_{vmode}_{dir[0]}

For example:
  t0.40_ctxW_w0.65_n0.70_15min_off_l

Resumable via a (symbol, tf, variant_tag, direction, date) skip-set.

Exit policies mirror research/30b for direct comparability.
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
SCRIPTS_29 = RESEARCH_ROOT / "29_short_options_signal_sweep" / "scripts"

sys.path.insert(0, str(SCRIPTS_29))
sys.path.insert(0, str(HERE))

from data_loader import load_5min, load_daily, resample  # noqa: E402
from signals_ccrb import (  # noqa: E402
    CCRBSignal,
    daily_setup_table,
    qualifying_dates,
    build_bar_pos_vol_avg,
    ccrb_signals,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SIGNAL_CSV = RESULTS / "ccrb_signals.csv"
RUN_LOG = RESULTS / "run.log"
STATUS_MD = ROOT / "CPR_COMPRESSION_BREAKOUT_SWEEP_STATUS.md"

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

ALL_STOCKS = sorted(set(COHORT_A + COHORT_B))
COHORT_A_SET = set(COHORT_A)

TIMEFRAMES = ["5min", "10min", "15min"]
TODAY_NARROWS = [0.0030, 0.0040, 0.0050]                 # 0.30 / 0.40 / 0.50%
YESTERDAY_WIDES = [0.0050, 0.0065, 0.0080]               # 0.50 / 0.65 / 0.80%
YESTERDAY_NARROW_RANGES = [0.0050, 0.0070, 0.0090]       # 0.50 / 0.70 / 0.90%
VOL_MODES = ["off", "vm1.5", "vm2.0"]
DIRECTIONS = ["long", "short"]

# yesterday_ctx variants — for each ctx, list (wide_thresh, narrow_thresh)
# pairs that are *meaningful*. W only varies by wide; N only by narrow;
# W_OR_N / W_AND_N use both (3x3=9). Total: 3 + 3 + 9 + 9 = 24 ctx variants.
def build_ctx_variants():
    out = []
    # W only
    for w in YESTERDAY_WIDES:
        # narrow doesn't matter for W; use a sentinel for label uniqueness
        out.append(("W", w, YESTERDAY_NARROW_RANGES[1]))   # mid-narrow as filler (unused)
    # N only
    for n in YESTERDAY_NARROW_RANGES:
        out.append(("N", YESTERDAY_WIDES[1], n))
    # W_OR_N (3x3)
    for w in YESTERDAY_WIDES:
        for n in YESTERDAY_NARROW_RANGES:
            out.append(("W_OR_N", w, n))
    # W_AND_N (3x3)
    for w in YESTERDAY_WIDES:
        for n in YESTERDAY_NARROW_RANGES:
            out.append(("W_AND_N", w, n))
    return out


CTX_VARIANTS = build_ctx_variants()       # length 24

PERIOD_END = pd.Timestamp("2026-03-25")
COHORT_A_START = pd.Timestamp("2018-01-01")
COHORT_B_START = pd.Timestamp("2024-03-18")

ENTRY_CUTOFF_TIME = dtime(14, 0)
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
# Walk-forward / exit simulator (CCRB-specific:
#   T_HARD_SL = yesterday's opposite extreme
#   R = abs(entry - yesterday_opposite_extreme)
# )
# ---------------------------------------------------------------------------

def simulate_exits(
    sess_after: pd.DataFrame,
    *,
    direction: str,
    entry_price: float,
    prev_day_high: float,
    prev_day_low: float,
    atr_daily_val: Optional[float],
) -> dict:
    sign = 1 if direction == "long" else -1

    policies: dict[str, dict] = {p: {
        "alive": True, "exit_time": None, "exit_price": None,
        "exit_reason": None, "mfe_pts": 0.0, "mae_pts": 0.0,
        "running_extreme": entry_price,
    } for p in EXIT_POLICIES}

    # Hard SL = yesterday's opposite extreme
    if direction == "long":
        hard_sl_dist = max(entry_price - prev_day_low, 0.0)
    else:
        hard_sl_dist = max(prev_day_high - entry_price, 0.0)
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
        "signal_id", "symbol", "cohort", "timeframe", "variant_tag", "direction",
        "date", "signal_time", "entry_price",
        "prev_day_high", "prev_day_low", "prev_day_close", "today_open",
        "today_cpr_width_pct", "yesterday_cpr_width_pct", "yesterday_range_pct",
        "bar_volume", "vol_ratio", "atr_daily",
    ]
    for p in EXIT_POLICIES:
        for c in ("exit_time", "exit_price", "exit_reason",
                  "net_pts", "net_pct", "mfe_pts", "mae_pts", "hold_min"):
            base.append(f"{p}__{c}")
    return base


def load_done_keys() -> dict[tuple, set[str]]:
    """(symbol, tf, variant_tag, direction) -> set of date strings."""
    out: dict[tuple, set[str]] = {}
    if not SIGNAL_CSV.exists():
        return out
    with SIGNAL_CSV.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            key = (row["symbol"], row["timeframe"], row["variant_tag"], row["direction"])
            out.setdefault(key, set()).add(row["date"])
    return out


# ---------------------------------------------------------------------------
# Per-stock processor
# ---------------------------------------------------------------------------

def process_symbol(sym: str, csv_writer: csv.DictWriter, csv_file,
                   done_by_cell: dict, sig_id_start: int) -> tuple[int, int, int]:
    """Run all variant cells for a single symbol.
    Returns (n_signals_logged, n_cells_visited, new_sig_id)."""
    period_start, period_end = period_for(sym)
    cohort = cohort_for(sym)

    try:
        df5 = load_5min(sym, period_start.strftime("%Y-%m-%d"),
                        period_end.strftime("%Y-%m-%d"))
    except Exception as e:
        print(f"  [{sym}] load_5min failed: {e}", flush=True)
        return 0, 0, sig_id_start

    df5 = df5.loc[(df5.index >= period_start)
                  & (df5.index < period_end + pd.Timedelta(days=1))]
    if df5.empty:
        print(f"  [{sym}] empty 5-min after period clip", flush=True)
        return 0, 0, sig_id_start

    daily_start = (period_start - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    daily = load_daily(sym, daily_start, period_end.strftime("%Y-%m-%d"))
    if daily.empty:
        print(f"  [{sym}] no daily data", flush=True)
        return 0, 0, sig_id_start

    atr_d = daily_atr_series(daily, n=14)
    setup = daily_setup_table(daily)
    if setup.empty:
        return 0, 0, sig_id_start
    setup_in_period = setup.loc[setup.index >= period_start]

    # Pre-build resampled frames + bar-position vol averages once per TF
    df_tf_by: dict[str, pd.DataFrame] = {}
    bar_avg_by: dict[str, pd.Series] = {}
    for tf in TIMEFRAMES:
        df_tf = df5 if tf == "5min" else resample(df5, tf)
        df_tf_by[tf] = df_tf
        bar_avg_by[tf] = build_bar_pos_vol_avg(df_tf, lookback=20) if not df_tf.empty else pd.Series(dtype=float)

    # Pre-build qualifying-day sets keyed by (today_narrow, ctx, w, n)
    qual_cache: dict[tuple, set[pd.Timestamp]] = {}
    for tn in TODAY_NARROWS:
        for ctx, w, n in CTX_VARIANTS:
            key = (tn, ctx, w, n)
            qual_cache[key] = qualifying_dates(
                setup_in_period,
                today_narrow=tn,
                yesterday_ctx=ctx,
                yesterday_wide_thresh=w,
                yesterday_narrow_range_thresh=n,
            )

    sig_id = sig_id_start
    n_cells = 0
    n_new_sigs = 0

    for tf in TIMEFRAMES:
        df_tf = df_tf_by[tf]
        if df_tf.empty:
            continue
        bar_avg = bar_avg_by[tf]

        for tn in TODAY_NARROWS:
            for ctx, w, n in CTX_VARIANTS:
                qset = qual_cache[(tn, ctx, w, n)]
                if not qset:
                    # No qualifying days under this filter — skip all
                    # downstream variant axes (volume modes x directions)
                    n_cells += len(VOL_MODES) * len(DIRECTIONS)
                    continue

                for vmode in VOL_MODES:
                    for direction in DIRECTIONS:
                        n_cells += 1
                        variant_tag = (
                            f"t{tn:.4f}_ctx{ctx}_w{w:.4f}_n{n:.4f}"
                            f"_{tf}_{vmode}_{direction[0]}"
                        )
                        cell_key = (sym, tf, variant_tag, direction)
                        already = done_by_cell.get(cell_key, set())
                        # If we already have any signals AND we expect at most
                        # |qset| sessions with at most one signal each, and we
                        # already covered all qualifying dates, we can skip.
                        # Otherwise, the inner generator + skip-set will avoid
                        # writing duplicates.
                        expected_dates = {d.strftime("%Y-%m-%d") for d in qset}
                        if already >= expected_dates:
                            # All eligible days already processed. Skip.
                            continue

                        for s in ccrb_signals(
                            df5, daily,
                            symbol=sym, cohort=cohort, timeframe=tf,
                            today_narrow=tn,
                            yesterday_ctx=ctx,
                            yesterday_wide_thresh=w,
                            yesterday_narrow_range_thresh=n,
                            vol_mode=vmode,
                            direction=direction,
                            setup_table=setup_in_period,
                            qualifying_set=qset,
                            df_tf_cache=df_tf,
                            bar_pos_avg_cache=bar_avg,
                        ):
                            date_str = s.date.strftime("%Y-%m-%d")
                            if date_str in already:
                                continue

                            sess_day = df_tf.loc[df_tf.index.normalize() == s.date.normalize()]
                            sess_after = sess_day.loc[
                                (sess_day.index > s.signal_time)
                                & (sess_day.index.time <= EOD_TIME)
                            ]
                            atr_val = atr_for_signal(atr_d, s.date)

                            ex = simulate_exits(
                                sess_after,
                                direction=direction,
                                entry_price=s.entry_price,
                                prev_day_high=s.prev_day_high,
                                prev_day_low=s.prev_day_low,
                                atr_daily_val=atr_val,
                            )
                            row = {
                                "signal_id": sig_id,
                                "symbol": sym,
                                "cohort": cohort,
                                "timeframe": tf,
                                "variant_tag": s.variant_tag,
                                "direction": direction,
                                "date": date_str,
                                "signal_time": s.signal_time.strftime("%Y-%m-%d %H:%M:%S"),
                                "entry_price": round(s.entry_price, 4),
                                "prev_day_high": round(s.prev_day_high, 4),
                                "prev_day_low": round(s.prev_day_low, 4),
                                "prev_day_close": round(s.prev_day_close, 4),
                                "today_open": round(s.today_open, 4),
                                "today_cpr_width_pct": round(s.today_cpr_width_pct, 6),
                                "yesterday_cpr_width_pct": round(s.yesterday_cpr_width_pct, 6),
                                "yesterday_range_pct": round(s.yesterday_range_pct, 6),
                                "bar_volume": int(s.bar_volume),
                                "vol_ratio": (round(s.vol_ratio, 3)
                                              if not (s.vol_ratio is None or
                                                      (isinstance(s.vol_ratio, float) and
                                                       np.isnan(s.vol_ratio)))
                                              else ""),
                                "atr_daily": round(atr_val, 4) if atr_val else "",
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
                            n_new_sigs += 1
                csv_file.flush()
        csv_file.flush()
    csv_file.flush()
    return n_new_sigs, n_cells, sig_id


# ---------------------------------------------------------------------------
# Status doc
# ---------------------------------------------------------------------------

def update_status(state: str, *, n_signals: int = 0, n_done_stocks: int = 0,
                  total_stocks: int = 79, started_at: Optional[str] = None,
                  last_stock: Optional[str] = None,
                  elapsed_min: float = 0.0,
                  notes: str = ""):
    """Rewrite the Status section of CPR_COMPRESSION_BREAKOUT_SWEEP_STATUS.md (sections 1-3 untouched).
    Sections 4 (Status), 5 (Crash Recovery), 6 (Files), 7 (Findings), 8
    (Comparison) live below '## 4. Status'. We rewrite from the
    Status section onward.
    """
    if not STATUS_MD.exists():
        return
    text = STATUS_MD.read_text(encoding="utf-8")
    cut = text.find("## 4. Status")
    if cut < 0:
        return
    head = text[:cut]

    progress_pct = (n_done_stocks / total_stocks * 100.0) if total_stocks else 0.0
    new = []
    new.append("## 4. Status (live running log)\n\n")
    new.append(f"**State:** {state}\n")
    if started_at:
        new.append(f"**Started:** {started_at}\n")
    if last_stock:
        new.append(f"**Last completed stock:** {last_stock}\n")
    new.append(f"**Stocks completed:** {n_done_stocks} / {total_stocks}  ({progress_pct:.1f}%)\n")
    new.append(f"**Signals logged:** {n_signals:,}\n")
    new.append(f"**Elapsed:** {elapsed_min:.1f} min\n")
    if notes:
        new.append(f"**Notes:** {notes}\n")
    new.append(f"**Last update:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} IST\n\n")

    new.append("### Files\n\n")
    new.append(f"- `results/ccrb_signals.csv` — per-signal x exit-policy rows\n")
    new.append(f"- `results/run.log` — per-stock progress\n\n")

    new.append("---\n\n")
    new.append("## 5. Crash Recovery\n\n")
    new.append("### A) Check what finished\n```bash\n")
    new.append("tail -5 research/31_cpr_compression_breakout/results/run.log\n")
    new.append("wc -l research/31_cpr_compression_breakout/results/ccrb_signals.csv\n")
    new.append("```\n\n")
    new.append("### B) Resume signal generation (resumable)\n```bash\n")
    new.append("cd /c/Users/Castro/Documents/Projects/Covered_Calls\n")
    new.append("python research/31_cpr_compression_breakout/scripts/run_ccrb.py\n")
    new.append("```\n\n")
    new.append("### C) Aggregate only\n```bash\n")
    new.append("python research/31_cpr_compression_breakout/scripts/aggregate_ccrb.py\n")
    new.append("```\n\n")
    new.append("Runner reads `ccrb_signals.csv`, builds a (symbol,tf,variant,dir,date) "
               "skip-set, only computes unfinished cells.\n\n")
    new.append("### D) Files NOT to touch\n")
    new.append("- `results/ccrb_signals.csv`\n- `results/run.log`\n- This file (auto-updated)\n\n")

    new.append("---\n\n## 6. Files (output map)\n\n")
    new.append("| File | Purpose | Committable? |\n|---|---|---|\n")
    new.append("| CPR_COMPRESSION_BREAKOUT_SWEEP_STATUS.md | This file | yes |\n")
    new.append("| scripts/signals_ccrb.py | Signal generator | yes |\n")
    new.append("| scripts/run_ccrb.py | Sweep runner | yes |\n")
    new.append("| scripts/aggregate_ccrb.py | Streaming aggregator | yes |\n")
    new.append("| results/ccrb_signals.csv | Per-signal rows (large) | gitignored |\n")
    new.append("| results/ccrb_ranking.csv | Per-cell aggregate | gitignored if >5MB |\n")
    new.append("| results/ccrb_leaders.csv | Per-stock leaderboard | yes |\n")
    new.append("| results/RESULTS.md | Final report | yes |\n\n")

    new.append("---\n\n## 7. Findings (during + final)\n\n")
    if state in ("COMPLETE", "AGGREGATED"):
        new.append("See `results/RESULTS.md`.\n\n")
    else:
        new.append("_Will populate after aggregation._\n\n")

    new.append("---\n\n## 8. Comparison to research/30b\n\n")
    if state in ("COMPLETE", "AGGREGATED"):
        new.append("See `results/RESULTS.md` -> 'Comparison vs research/30b' section.\n\n")
    else:
        new.append("_Will populate after aggregation._\n\n")

    new.append(f"**Last updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} IST\n")

    STATUS_MD.write_text(head + "".join(new), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(stocks: Optional[list[str]] = None):
    stocks = stocks or ALL_STOCKS

    fieldnames = build_csv_header()
    if not SIGNAL_CSV.exists():
        with SIGNAL_CSV.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    done_by_cell = load_done_keys()
    n_existing = sum(len(v) for v in done_by_cell.values())

    started_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"== CCRB sweep ==", flush=True)
    print(f"Universe: {len(stocks)} stocks  |  TFs: {TIMEFRAMES}", flush=True)
    print(f"Cells/stock = {len(TIMEFRAMES)} * {len(TODAY_NARROWS)} * {len(CTX_VARIANTS)} "
          f"* {len(VOL_MODES)} * {len(DIRECTIONS)} = "
          f"{len(TIMEFRAMES)*len(TODAY_NARROWS)*len(CTX_VARIANTS)*len(VOL_MODES)*len(DIRECTIONS)}", flush=True)
    print(f"Existing signal rows: {n_existing:,}", flush=True)

    update_status("RUNNING", n_signals=n_existing, n_done_stocks=0,
                  total_stocks=len(stocks), started_at=started_at)

    csv_file = SIGNAL_CSV.open("a", encoding="utf-8", newline="", buffering=1)
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    sig_id = n_existing
    n_done = 0
    t0_global = _time.time()

    log_f = RUN_LOG.open("a", encoding="utf-8", buffering=1)

    for i, sym in enumerate(stocks, 1):
        t0 = _time.time()
        try:
            n_sigs, n_cells, sig_id = process_symbol(
                sym, csv_writer, csv_file, done_by_cell, sig_id
            )
        except Exception as e:
            err = f"[{i}/{len(stocks)}] {sym} FAILED: {e}"
            print(err, flush=True)
            log_f.write(err + "\n")
            log_f.flush()
            continue
        elapsed = _time.time() - t0
        n_done += 1
        elapsed_total_min = (_time.time() - t0_global) / 60.0
        msg = (f"[{i}/{len(stocks)}] {sym:14s} — {n_sigs:5d} new signals "
               f"across {n_cells} cells in {elapsed:5.1f}s "
               f"(elapsed {elapsed_total_min:5.1f} min)")
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

        # Update status every 10 stocks
        if i % 10 == 0 or i == len(stocks):
            cur_n = (sum(1 for _ in open(SIGNAL_CSV, encoding="utf-8")) - 1)
            update_status("RUNNING", n_signals=cur_n, n_done_stocks=i,
                          total_stocks=len(stocks), started_at=started_at,
                          last_stock=sym, elapsed_min=elapsed_total_min)

    csv_file.close()
    log_f.close()

    cur_n = (sum(1 for _ in open(SIGNAL_CSV, encoding="utf-8")) - 1)
    elapsed_total_min = (_time.time() - t0_global) / 60.0
    update_status("AGGREGATING", n_signals=cur_n, n_done_stocks=len(stocks),
                  total_stocks=len(stocks), started_at=started_at,
                  last_stock=stocks[-1] if stocks else None,
                  elapsed_min=elapsed_total_min)
    print(f"\nTotal elapsed: {elapsed_total_min:.1f} min")
    print(f"Signals on disk: {cur_n:,}")


def main():
    args = sys.argv[1:]
    if "--stocks" in args:
        idx = args.index("--stocks")
        stocks = args[idx + 1].split(",")
        run_all(stocks)
    else:
        run_all()


if __name__ == "__main__":
    main()
