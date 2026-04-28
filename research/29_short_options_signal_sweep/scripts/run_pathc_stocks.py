"""Apply Path C (post-12:00 day-extreme break + range compression) to the
10 stocks with 5-min intraday data — mirror of the NIFTY Path C sweep.

Reuses signals.path_c_signals (already symbol-agnostic). Tests 5/10/15-min
timeframes and the same variant grid used on NIFTY:
  range_threshold ∈ {0.4%, 0.6%, 0.8%, 1.0%, off}  ×  use_rsi ∈ {on, off}

For each signal, walks forward to 15:25 IST, logs:
  - net_pts at EOD (T0 time-only)
  - net_pts at T1_SL (hard SL = 1×daily_ATR; scaled per-symbol)
  - max adverse excursion (MAE), max favourable (MFE)

Aggregates per (symbol × timeframe × variant) and ranks by Sharpe-style score:
  (mean_net / std_net) × win_rate_fraction

Outputs:
  results/pathc_stocks_signals.csv  — per-signal rows
  results/pathc_stocks_ranking.csv  — per-cell aggregate
  results/RESULTS_PATHC_STOCKS.md   — top picks markdown
"""
from __future__ import annotations

import csv
import sys
from dataclasses import asdict
from pathlib import Path
from datetime import time

import numpy as np
import pandas as pd

# Make signals + data_loader importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_loader import load_5min, resample as resample_5min  # noqa: E402
from signals import path_c_signals  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"

STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
          "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "HINDUNILVR"]
TIMEFRAMES = ["5min", "10min", "15min"]
RANGE_THRESHOLDS = [0.004, 0.006, 0.008, 0.010, None]  # None = "off"
RSI_MODES = [(False, 40, 60), (True, 40, 60)]

PERIOD_START = pd.Timestamp("2024-03-01")
PERIOD_END = pd.Timestamp("2026-03-25")

EOD_TIME = time(15, 25)


def daily_atr(df_5min: pd.DataFrame, n: int = 14) -> float:
    """Crude daily ATR proxy: average of (high - low) per session over last n days."""
    daily = df_5min.groupby(df_5min.index.normalize()).agg(
        high=("high", "max"), low=("low", "min")
    )
    rng = (daily["high"] - daily["low"]).dropna()
    if rng.empty:
        return 1.0
    return float(rng.tail(n).mean()) if len(rng) >= n else float(rng.mean())


def walk_forward(sess_after: pd.DataFrame, signal_price: float,
                 direction: str, sl_pts: float):
    """Walk session candles after the signal candle (sess_after starts at the
    bar AFTER the signal). Return (net_at_eod, mae_against, mfe_with, sl_hit_first).
    """
    if sess_after.empty:
        return 0.0, 0.0, 0.0, False
    sign = 1 if direction == "long" else -1
    pts_with = (sess_after["close"] - signal_price) * sign
    pts_against = -pts_with  # how far against us at any moment (use the lowest pts_with)
    mae = max(0.0, float(-pts_with.min()))  # adverse excursion in pts
    mfe = max(0.0, float(pts_with.max()))
    eod_close = float(sess_after["close"].iloc[-1])
    net = (eod_close - signal_price) * sign
    # T1_SL: did adverse excursion exceed SL before EOD?
    sl_hit = bool(mae >= sl_pts)
    return float(net), mae, mfe, sl_hit


def run():
    sig_path = RESULTS / "pathc_stocks_signals.csv"
    rank_path = RESULTS / "pathc_stocks_ranking.csv"
    md_path = RESULTS / "RESULTS_PATHC_STOCKS.md"

    sig_rows = []
    cell_results = {}  # (sym, tf, variant) -> list of rows

    sig_id = 0
    for sym in STOCKS:
        try:
            df5 = load_5min(sym)
        except Exception as e:
            print(f"  [{sym}] load_5min failed: {e}")
            continue
        df5 = df5.loc[(df5.index >= PERIOD_START) & (df5.index < PERIOD_END + pd.Timedelta(days=1))]
        if df5.empty:
            print(f"  [{sym}] empty after period clip")
            continue
        atr_pts = daily_atr(df5, n=20)
        sl_pts = atr_pts  # 1×ATR hard SL for T1
        print(f"\n[{sym}] rows={len(df5)}, daily_atr_proxy={atr_pts:.2f}, T1_sl={sl_pts:.2f}")

        for tf in TIMEFRAMES:
            df = df5 if tf == "5min" else resample_5min(df5, tf)
            for rng in RANGE_THRESHOLDS:
                for use_rsi, rsi_lo, rsi_hi in RSI_MODES:
                    cell_count_before = sig_id
                    for s in path_c_signals(
                        df, daily=pd.DataFrame(),  # path C doesn't actually need daily
                        range_threshold=rng,
                        use_rsi=use_rsi,
                        rsi_low=rsi_lo,
                        rsi_high=rsi_hi,
                        symbol=sym,
                        timeframe=tf,
                    ):
                        # Walk forward from the candle AFTER signal_time to 15:25
                        sess_day = df.loc[df.index.normalize() == s.date.normalize()]
                        sess_after = sess_day.loc[
                            (sess_day.index > s.signal_time) &
                            (sess_day.index.time <= EOD_TIME)
                        ]
                        net, mae, mfe, sl_hit = walk_forward(
                            sess_after, s.signal_price, s.direction, sl_pts
                        )
                        net_t1 = -sl_pts if sl_hit else net
                        row = {
                            "signal_id": sig_id,
                            "symbol": sym,
                            "timeframe": tf,
                            "variant": s.variant,
                            "date": s.date.strftime("%Y-%m-%d"),
                            "signal_time": s.signal_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "direction": s.direction,
                            "signal_price": round(s.signal_price, 2),
                            "level": round(s.level_for_T4, 2),
                            "rsi": round(s.extras.get("rsi") or 0, 2),
                            "range_pct": round(s.extras.get("range_pct") or 0, 5),
                            "mae_against": round(mae, 2),
                            "mfe_with": round(mfe, 2),
                            "net_t0": round(net, 2),
                            "net_t1_sl": round(net_t1, 2),
                            "sl_hit": sl_hit,
                            "atr_proxy": round(atr_pts, 2),
                        }
                        sig_rows.append(row)
                        key = (sym, tf, s.variant)
                        cell_results.setdefault(key, []).append(row)
                        sig_id += 1
                    n_in_cell = sig_id - cell_count_before
                    if n_in_cell > 0:
                        print(f"  {sym} {tf} {s.variant if 's' in dir() else 'n/a'} -> {n_in_cell} signals")

    # Write per-signal CSV
    if sig_rows:
        with sig_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sig_rows[0].keys()))
            w.writeheader()
            for r in sig_rows:
                w.writerow(r)
        print(f"\nWrote {len(sig_rows)} signals to {sig_path}")
    else:
        print("\nNo signals fired across all cells. Check data window or variants.")
        return

    # Aggregate per cell
    agg_rows = []
    for (sym, tf, variant), rows in cell_results.items():
        n = len(rows)
        if n < 5:
            continue  # too few to be meaningful
        for policy_col, policy_label in [("net_t0", "T0"), ("net_t1_sl", "T1_SL_1xATR")]:
            nets = np.array([r[policy_col] for r in rows], dtype=float)
            mean = float(nets.mean())
            std = float(nets.std(ddof=0))
            wins = int((nets > 0).sum())
            wr = 100.0 * wins / n
            sharpe = (mean / std * (wr / 100.0)) if std > 0 else 0.0
            agg_rows.append({
                "symbol": sym,
                "timeframe": tf,
                "variant": variant,
                "exit_policy": policy_label,
                "n_signals": n,
                "mean_net": round(mean, 3),
                "std_net": round(std, 3),
                "win_rate": round(wr, 1),
                "sharpe_score": round(sharpe, 4),
                "atr_proxy": round(rows[0]["atr_proxy"], 2),
            })

    agg_rows.sort(key=lambda r: r["sharpe_score"], reverse=True)
    with rank_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)
    print(f"Wrote {len(agg_rows)} ranked cells to {rank_path}")

    # Markdown summary
    lines = ["# Path C on Stocks — Compression Breakout Backtest\n"]
    lines.append("Same logic as NIFTY Path C: post-12:00 day-extreme break + ")
    lines.append("range compression filter. Applied to the 10 stocks with ")
    lines.append("5-min intraday data.\n\n")
    lines.append(f"- Period: 2024-03-01 to 2026-03-25\n")
    lines.append(f"- Universe: {', '.join(STOCKS)}\n")
    lines.append(f"- Timeframes: 5/10/15-min\n")
    lines.append(f"- Variant grid: range_threshold ∈ {{0.4%, 0.6%, 0.8%, 1.0%, off}} × RSI ∈ {{on, off}}\n")
    lines.append(f"- Total signals fired: **{len(sig_rows)}**\n")
    lines.append(f"- Ranked configurations (n>=5): **{len(agg_rows)}**\n\n")

    # Filter to mean>0 + n>=20 for the "promote" list
    promote = [r for r in agg_rows if r["mean_net"] > 0 and r["n_signals"] >= 20]
    promote.sort(key=lambda r: r["sharpe_score"], reverse=True)

    lines.append("## Top 10 by Sharpe-style score (n>=20, mean>0)\n\n")
    lines.append("| symbol | tf | variant | policy | n | mean | std | WR% | Sharpe |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for r in promote[:10]:
        lines.append(f"| {r['symbol']} | {r['timeframe']} | {r['variant']} | {r['exit_policy']} | "
                     f"{r['n_signals']} | {r['mean_net']:.3f} | {r['std_net']:.3f} | "
                     f"{r['win_rate']:.1f} | {r['sharpe_score']:.4f} |\n")

    # Per-stock best
    lines.append("\n## Best variant per stock (T0, n>=10)\n\n")
    by_sym = {}
    for r in agg_rows:
        if r["exit_policy"] != "T0" or r["n_signals"] < 10:
            continue
        cur = by_sym.get(r["symbol"])
        if not cur or r["sharpe_score"] > cur["sharpe_score"]:
            by_sym[r["symbol"]] = r
    lines.append("| symbol | tf | variant | n | mean | std | WR% | Sharpe |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for sym in STOCKS:
        r = by_sym.get(sym)
        if not r:
            lines.append(f"| {sym} | — | (no eligible cells) | | | | | |\n")
            continue
        lines.append(f"| {r['symbol']} | {r['timeframe']} | {r['variant']} | "
                     f"{r['n_signals']} | {r['mean_net']:.3f} | {r['std_net']:.3f} | "
                     f"{r['win_rate']:.1f} | {r['sharpe_score']:.4f} |\n")

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    run()
