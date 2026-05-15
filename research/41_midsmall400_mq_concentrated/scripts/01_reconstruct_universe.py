"""Phase A — Point-in-time Nifty-MidSmallcap-400-like universe reconstruction.

Survivorship-free: at each semi-annual rebalance date we rank ALL NSE daily
symbols by trailing-6-month median daily traded value (close*volume) using
ONLY data available up to that date, then take the rank band that excludes
large caps and keeps mid+small (≈ ranks 101..500). No use of today's index
membership anywhere -> no survivorship / look-ahead.

Outputs results/pit_universe.csv  (rebalance_date, symbol, rank, med_tv)
Sanity: latest reconstruction vs the user-supplied 100 MQ constituents.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "backtest_data" / "market_data.db"
HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

LARGE_CAP_EXCLUDE = 100      # drop top-100 by traded value (large caps)
UNIVERSE_SIZE = 400          # next 400 = MidSmallcap-400-like pool
LOOKBACK_TD = 126            # ~6 months of trading days for the liquidity proxy
MIN_HISTORY_TD = 126         # must have >= 6mo of bars before the date
START_YEAR = 2013            # rebalance grid start (depth permitting)


def rebalance_dates(start_year: int, end_date: pd.Timestamp):
    """Semi-annual, mirroring the index cadence: ~end-Jun and ~end-Dec."""
    dates = []
    y = start_year
    while True:
        for m in (6, 12):
            d = pd.Timestamp(year=y, month=m, day=28)
            if d <= end_date:
                dates.append(d)
        if pd.Timestamp(year=y, month=12, day=28) > end_date:
            break
        y += 1
    return dates


def main():
    con = sqlite3.connect(DB)
    # Pull all daily rows once (close, volume) — large but one pass.
    print("Loading daily close+volume for all symbols ...")
    df = pd.read_sql(
        "SELECT symbol, date, close, volume FROM market_data_unified "
        "WHERE timeframe='day' AND close IS NOT NULL",
        con, parse_dates=["date"],
    )
    con.close()
    print(f"  rows={len(df):,}  symbols={df.symbol.nunique()}  "
          f"range={df.date.min().date()}..{df.date.max().date()}")

    df["tv"] = df["close"] * df["volume"].fillna(0)
    df = df.sort_values(["symbol", "date"])

    rdates = rebalance_dates(START_YEAR, df.date.max())
    print(f"Rebalance dates: {len(rdates)}  ({rdates[0].date()} .. {rdates[-1].date()})")

    out_rows = []
    for rd in rdates:
        win_lo = rd - pd.Timedelta(days=int(LOOKBACK_TD * 1.6))  # calendar pad
        sub = df[(df.date <= rd) & (df.date > win_lo)]
        if sub.empty:
            continue
        g = sub.groupby("symbol")
        # eligibility: enough bars in the lookback window AND priced
        liq = g["tv"].median()
        bars = g["tv"].size()
        last_close = g["close"].last()
        elig = bars[bars >= MIN_HISTORY_TD * 0.6].index  # >=~75 bars in 6mo
        liq = liq.loc[liq.index.isin(elig)]
        liq = liq[liq > 0].sort_values(ascending=False)
        ranked = liq.reset_index()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        pool = ranked.iloc[LARGE_CAP_EXCLUDE: LARGE_CAP_EXCLUDE + UNIVERSE_SIZE].copy()
        pool["rebalance_date"] = rd.date().isoformat()
        out_rows.append(pool[["rebalance_date", "symbol", "rank", "tv"]]
                        .rename(columns={"tv": "med_tv"}))

    pit = pd.concat(out_rows, ignore_index=True)
    pit.to_csv(RESULTS / "pit_universe.csv", index=False)
    print(f"\nWrote {len(pit):,} rows -> results/pit_universe.csv")
    print("Universe size per rebalance (head/tail):")
    sz = pit.groupby("rebalance_date").size()
    print(sz.head(3).to_string()); print("..."); print(sz.tail(3).to_string())

    # ---- Sanity check: latest reconstruction vs supplied 100 ----
    supplied = [s.strip() for s in
                open(HERE / "universe_mq100_2026-05-15.csv") if s.strip()]
    # remap obvious renames so the check is fair
    remap = {"360ONE": "IIFLWAM", "UNOMINDA": "MINDAIND", "NAVA": "NAVABHARAT",
             "GVT&D": "GET&D", "APARINDS": "APARINDS", "GPIL": "GPIL",
             "JSWDULUX": "JSWDULUX", "KIRLOSBROS": "KIRLOSBROS",
             "USHAMART": "USHAMART"}
    supplied_eff = set(remap.get(s, s) for s in supplied)
    latest_rd = pit.rebalance_date.max()
    recon_latest = set(pit[pit.rebalance_date == latest_rd].symbol)
    hit = supplied_eff & recon_latest
    print(f"\n--- SANITY: latest PIT universe ({latest_rd}) vs supplied 100 ---")
    print(f"  reconstructed pool size: {len(recon_latest)}")
    print(f"  supplied (remapped) found in pool: {len(hit)}/100")
    miss = sorted(supplied_eff - recon_latest)
    print(f"  not in reconstructed pool ({len(miss)}): {miss[:30]}")
    print("\n  Interpretation: a high hit-rate means the liquidity/size proxy "
          "captures the same mid/small space the index draws from. Misses are "
          "typically very-recent IPOs (short history) or names whose traded "
          "value sits just outside the 101-500 band.")


if __name__ == "__main__":
    main()
