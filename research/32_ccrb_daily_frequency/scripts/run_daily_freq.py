"""CCRB Daily Setup Frequency — universe-wide daily-only scan.

Question: "If we ran the CCRB scanner across all stocks with daily data,
how many stocks would qualify on a typical session for the daily-bar
setup filter (today narrow CPR + yesterday wide-CPR or narrow-range)?"

This script ONLY uses daily bars. No intraday data needed.

Universe: every distinct symbol with timeframe='day' in
backtest_data/market_data.db (~1,623 symbols).

Variants (9 total — skipping W_AND_N which never fires in research/31):
    today_narrow   in {0.0030, 0.0040, 0.0050}
    yesterday_ctx  in {W, N, W_OR_N}
    Per-ctx thresholds:
        W:        wide CPR pct in {0.0050, 0.0065, 0.0080}
        N:        narrow range pct in {0.0050, 0.0070, 0.0090}
        W_OR_N:   both thresholds applied (wide OR narrow-range qualifies)

Outputs (under research/32_ccrb_daily_frequency/results/):
    daily_setup_count.csv  -- (date, variant, n_qualifying_stocks)
    per_stock_freq.csv     -- (symbol, variant, n_setup_days, n_total_days, freq_pct)
    variant_summary.csv    -- (variant, mean/median/p10/p90 daily count, period stats)
    top50_stocks.csv       -- top 50 stocks by setup-frequency across variants

Reuses CPR helpers from research/31_cpr_compression_breakout/scripts/signals_ccrb.py.
"""
from __future__ import annotations

import csv
import sqlite3
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESEARCH_ROOT = ROOT.parent
PROJECT_ROOT = RESEARCH_ROOT.parent

# Reuse CPR formulas from research/31
SCRIPTS_31 = RESEARCH_ROOT / "31_cpr_compression_breakout" / "scripts"
sys.path.insert(0, str(SCRIPTS_31))
from signals_ccrb import _cpr  # noqa: E402

DB_PATH = PROJECT_ROOT / "backtest_data" / "market_data.db"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

DAILY_COUNT_CSV = RESULTS / "daily_setup_count.csv"
PER_STOCK_CSV = RESULTS / "per_stock_freq.csv"
VARIANT_SUMMARY_CSV = RESULTS / "variant_summary.csv"
TOP50_CSV = RESULTS / "top50_stocks.csv"

# 79-stock intraday universe (research/30b)
INTRADAY_79 = sorted(set([
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "HINDUNILVR",
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
]))

# ---------------------------------------------------------------------------
# Variant grid (9 variants — skip W_AND_N)
# ---------------------------------------------------------------------------

TODAY_NARROW = [0.0030, 0.0040, 0.0050]
WIDE_THRESHES = [0.0050, 0.0065, 0.0080]
NARROW_THRESHES = [0.0050, 0.0070, 0.0090]

# Pair wide/narrow thresholds positionally (loose / mid / tight)
PAIRED = list(zip(WIDE_THRESHES, NARROW_THRESHES))   # 3 pairs

VARIANTS: list[dict] = []
for tn in TODAY_NARROW:
    for w_thr, n_thr in PAIRED:
        for ctx in ("W", "N", "W_OR_N"):
            tag = f"t{tn:.4f}_ctx{ctx}_w{w_thr:.4f}_n{n_thr:.4f}"
            VARIANTS.append({
                "tag": tag,
                "today_narrow": tn,
                "yesterday_ctx": ctx,
                "wide_thresh": w_thr,
                "narrow_thresh": n_thr,
            })

# 27 variants total but we want only one of each pair-axis... user wants
# 9 (today_narrow=3 x context=3 = 9) using a SINGLE pair of thresholds.
# Reading the spec more carefully: variants are
#   today_narrow ∈ {0.30%, 0.40%, 0.50%}
#   yesterday_ctx ∈ {W (with thresh tier), N (with thresh tier), W_OR_N}
# wide_thresh and narrow_thresh each have 3 tiers. We sweep all combinations
# the spec wants — 3 today_narrow x 3 ctx x 3 thresh-tiers = 27, but for
# W_OR_N a single tier picks both thresholds together. For W only the
# wide_thresh matters; for N only the narrow_thresh matters. So unique
# (tn, ctx, thresh-tier) = 27, but W collapses on narrow (3 dups) and N on
# wide (3 dups). We dedupe below.
_seen = set()
_dedup = []
for v in VARIANTS:
    if v["yesterday_ctx"] == "W":
        key = ("W", v["today_narrow"], v["wide_thresh"])
    elif v["yesterday_ctx"] == "N":
        key = ("N", v["today_narrow"], v["narrow_thresh"])
    else:  # W_OR_N
        key = ("W_OR_N", v["today_narrow"], v["wide_thresh"], v["narrow_thresh"])
    if key in _seen:
        continue
    _seen.add(key)
    # Rebuild a clean tag
    if v["yesterday_ctx"] == "W":
        v["tag"] = f"t{v['today_narrow']:.4f}_ctxW_w{v['wide_thresh']:.4f}"
    elif v["yesterday_ctx"] == "N":
        v["tag"] = f"t{v['today_narrow']:.4f}_ctxN_n{v['narrow_thresh']:.4f}"
    _dedup.append(v)
VARIANTS = _dedup

print(f"Variant count after dedup: {len(VARIANTS)}")

# ---------------------------------------------------------------------------
# Setup table — vectorised version of research/31 daily_setup_table
# ---------------------------------------------------------------------------

def daily_setup_vec(daily: pd.DataFrame) -> pd.DataFrame:
    """Vectorised CPR setup table.

    Returns a frame indexed by today's trading date with columns:
      today_open
      today_cpr_width_pct        (CPR built from prev's HLC, normalised by today's open)
      prev_cpr_width_pct         (CPR built from day-before-prev's HLC, normalised by prev's open)
      prev_range_pct             ((prev_high - prev_low) / prev_open)

    Drops the first 2 rows (need 2 prior days for CPR(today) + CPR(yesterday)).
    """
    if daily.empty or len(daily) < 3:
        return pd.DataFrame()
    df = daily.sort_index()
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    # CPR widths at index i computed from prior bar (i-1)
    # pivot_i = (h[i-1]+l[i-1]+c[i-1])/3
    # bc_i    = (h[i-1]+l[i-1])/2
    # tc_i    = 2*pivot_i - bc_i
    # width_i = |tc_i - bc_i|
    pivot_today = (h + l + c) / 3.0  # placeholder; we shift
    # Compute on-bar CPR width for each day, then shift by 1 to use prev's
    bc_bar = (h + l) / 2.0
    tc_bar = 2.0 * ((h + l + c) / 3.0) - bc_bar
    width_bar = np.abs(tc_bar - bc_bar)            # CPR width for THIS day (built from THIS day's HLC)

    # today_cpr_width <- width_bar shifted by 1 (uses yesterday's HLC)
    today_cpr_width = np.empty_like(width_bar) * np.nan
    today_cpr_width[1:] = width_bar[:-1]
    # prev_cpr_width <- width_bar shifted by 2 (uses day-before-yesterday's HLC)
    prev_cpr_width = np.empty_like(width_bar) * np.nan
    prev_cpr_width[2:] = width_bar[:-2]

    # prev's HLC = shift by 1
    prev_open = np.empty_like(o) * np.nan
    prev_open[1:] = o[:-1]
    prev_high = np.empty_like(h) * np.nan
    prev_high[1:] = h[:-1]
    prev_low = np.empty_like(l) * np.nan
    prev_low[1:] = l[:-1]

    today_cpr_pct = np.where(o > 0, today_cpr_width / o, np.nan)
    prev_cpr_pct = np.where(prev_open > 0, prev_cpr_width / prev_open, np.nan)
    prev_range = prev_high - prev_low
    prev_range_pct = np.where(prev_open > 0, prev_range / prev_open, np.nan)

    out = pd.DataFrame({
        "today_open": o,
        "today_cpr_width_pct": today_cpr_pct,
        "prev_cpr_width_pct": prev_cpr_pct,
        "prev_range_pct": prev_range_pct,
    }, index=df.index)
    # Need both today_cpr (from i-1) AND prev_cpr (from i-2): drop first 2
    out = out.iloc[2:].copy()
    return out


# ---------------------------------------------------------------------------
# DB load
# ---------------------------------------------------------------------------

def list_symbols(con: sqlite3.Connection) -> list[str]:
    rows = con.execute(
        "SELECT symbol, COUNT(*) c FROM market_data_unified "
        "WHERE timeframe='day' GROUP BY symbol HAVING c >= 100 ORDER BY symbol"
    ).fetchall()
    return [r[0] for r in rows]


def load_daily_for(con: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT date, open, high, low, close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' ORDER BY date",
        con, params=(symbol,)
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")
    # Drop duplicate dates if any (defensive)
    df = df[~df.index.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# Per-symbol scan
# ---------------------------------------------------------------------------

def scan_symbol(setup: pd.DataFrame) -> dict[str, pd.Series]:
    """For a single symbol's setup table, return {variant_tag -> bool Series indexed by date}.

    True = setup qualifies on that date for that variant.
    """
    if setup.empty:
        return {}
    today_pct = setup["today_cpr_width_pct"]
    prev_cpr_pct = setup["prev_cpr_width_pct"]
    prev_rng_pct = setup["prev_range_pct"]

    out: dict[str, pd.Series] = {}
    for v in VARIANTS:
        tn = v["today_narrow"]
        ctx = v["yesterday_ctx"]
        w_thr = v["wide_thresh"]
        n_thr = v["narrow_thresh"]

        today_ok = today_pct <= tn
        wide_ok = prev_cpr_pct >= w_thr
        narrow_ok = prev_rng_pct <= n_thr

        if ctx == "W":
            ctx_ok = wide_ok
        elif ctx == "N":
            ctx_ok = narrow_ok
        elif ctx == "W_OR_N":
            ctx_ok = wide_ok | narrow_ok
        else:
            raise ValueError(ctx)

        mask = (today_ok & ctx_ok).fillna(False)
        out[v["tag"]] = mask
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = _time.time()
    print(f"DB: {DB_PATH}")
    con = sqlite3.connect(str(DB_PATH))
    symbols = list_symbols(con)
    print(f"Symbols with >=100 daily rows: {len(symbols)}")

    # Aggregators
    # daily_count[variant_tag][date] = count of qualifying symbols
    daily_count: dict[str, dict[pd.Timestamp, int]] = {v["tag"]: {} for v in VARIANTS}
    # per-stock totals: per_stock[symbol][variant_tag] = (n_setup_days, n_total_days)
    per_stock_setup: dict[str, dict[str, int]] = {}
    per_stock_total: dict[str, int] = {}
    # 2024-2026 sub-period stats
    per_stock_setup_recent: dict[str, dict[str, int]] = {}
    per_stock_total_recent: dict[str, int] = {}
    daily_count_recent: dict[str, dict[pd.Timestamp, int]] = {v["tag"]: {} for v in VARIANTS}

    RECENT_START = pd.Timestamp("2024-01-01")

    n_done = 0
    n_skipped = 0
    log_every = max(1, len(symbols) // 40)

    for sym in symbols:
        try:
            daily = load_daily_for(con, sym)
            if daily.empty or len(daily) < 100:
                n_skipped += 1
                continue
            setup = daily_setup_vec(daily)
            if setup.empty:
                n_skipped += 1
                continue
            masks = scan_symbol(setup)
            per_stock_setup[sym] = {}
            per_stock_setup_recent[sym] = {}
            per_stock_total[sym] = int(len(setup))
            recent_setup = setup.loc[setup.index >= RECENT_START]
            per_stock_total_recent[sym] = int(len(recent_setup))

            for tag, mask in masks.items():
                # Bump per-day counts (full period)
                hits = mask[mask]
                if not hits.empty:
                    for d in hits.index:
                        daily_count[tag][d] = daily_count[tag].get(d, 0) + 1
                per_stock_setup[sym][tag] = int(hits.sum())

                # Recent sub-period (2024+)
                mask_recent = mask.loc[mask.index >= RECENT_START]
                hits_recent = mask_recent[mask_recent]
                if not hits_recent.empty:
                    for d in hits_recent.index:
                        daily_count_recent[tag][d] = daily_count_recent[tag].get(d, 0) + 1
                per_stock_setup_recent[sym][tag] = int(hits_recent.sum())

            n_done += 1
            if n_done % log_every == 0 or n_done == len(symbols):
                elapsed = _time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(symbols) - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{len(symbols)}] {sym} | elapsed {elapsed:.0f}s | rate {rate:.1f}/s | ETA {eta:.0f}s",
                      flush=True)
        except Exception as e:
            print(f"  ERROR on {sym}: {e}", flush=True)
            n_skipped += 1
            continue

    con.close()
    print(f"Done. Scanned {n_done} symbols, skipped {n_skipped}. Elapsed: {_time.time()-t0:.0f}s")

    # ----- Write daily_setup_count.csv -----
    print(f"Writing {DAILY_COUNT_CSV}")
    with open(DAILY_COUNT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "date", "n_qualifying_stocks", "period"])
        for tag, dmap in daily_count.items():
            for d, n in sorted(dmap.items()):
                period = "recent" if d >= RECENT_START else "early"
                w.writerow([tag, d.strftime("%Y-%m-%d"), n, period])

    # ----- Write per_stock_freq.csv -----
    print(f"Writing {PER_STOCK_CSV}")
    with open(PER_STOCK_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "variant", "period",
                    "n_setup_days", "n_total_days", "freq_pct", "in_intraday_79"])
        for sym, perv in per_stock_setup.items():
            tot = per_stock_total.get(sym, 0)
            in79 = sym in INTRADAY_79
            for tag, n in perv.items():
                fp = (n / tot * 100.0) if tot > 0 else 0.0
                w.writerow([sym, tag, "full", n, tot, f"{fp:.4f}", int(in79)])
            tot_r = per_stock_total_recent.get(sym, 0)
            for tag, n in per_stock_setup_recent[sym].items():
                fp = (n / tot_r * 100.0) if tot_r > 0 else 0.0
                w.writerow([sym, tag, "recent", n, tot_r, f"{fp:.4f}", int(in79)])

    # ----- Variant summary -----
    print(f"Writing {VARIANT_SUMMARY_CSV}")
    rows = []
    for v in VARIANTS:
        tag = v["tag"]
        for period_name, dmap in (("full", daily_count[tag]), ("recent", daily_count_recent[tag])):
            if not dmap:
                rows.append({
                    "variant": tag, "today_narrow": v["today_narrow"],
                    "yesterday_ctx": v["yesterday_ctx"],
                    "wide_thresh": v["wide_thresh"], "narrow_thresh": v["narrow_thresh"],
                    "period": period_name, "n_sessions": 0,
                    "mean_per_day": 0, "median_per_day": 0, "p10": 0, "p90": 0,
                    "max_per_day": 0, "total_setup_events": 0,
                })
                continue
            arr = np.array(sorted(dmap.values()), dtype=float)
            rows.append({
                "variant": tag, "today_narrow": v["today_narrow"],
                "yesterday_ctx": v["yesterday_ctx"],
                "wide_thresh": v["wide_thresh"], "narrow_thresh": v["narrow_thresh"],
                "period": period_name,
                "n_sessions": int(len(arr)),
                "mean_per_day": float(arr.mean()),
                "median_per_day": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
                "max_per_day": float(arr.max()),
                "total_setup_events": float(arr.sum()),
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(VARIANT_SUMMARY_CSV, index=False)

    # ----- Top 50 stocks by setup frequency (recent period, mid-tier W_OR_N variant) -----
    # Pick representative variant: today_narrow=0.0040, ctx=W_OR_N, mid thresholds
    rep_tag = "t0.0040_ctxW_OR_N_w0.0065_n0.0070"
    print(f"Top 50 by representative variant: {rep_tag}")
    top_rows = []
    for sym, perv in per_stock_setup_recent.items():
        n = perv.get(rep_tag, 0)
        tot = per_stock_total_recent.get(sym, 0)
        if tot < 50:
            continue
        top_rows.append({
            "symbol": sym,
            "n_setup_days_recent": n,
            "n_total_days_recent": tot,
            "freq_pct_recent": (n / tot * 100.0) if tot > 0 else 0.0,
            "in_intraday_79": int(sym in INTRADAY_79),
        })
    top_df = pd.DataFrame(top_rows).sort_values("n_setup_days_recent", ascending=False).head(50)
    top_df.to_csv(TOP50_CSV, index=False)

    # ----- Console summary of representative variant -----
    print("\n=== Representative variant (recent period) ===")
    rep = summary_df[(summary_df["variant"] == rep_tag) & (summary_df["period"] == "recent")]
    if not rep.empty:
        print(rep.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
