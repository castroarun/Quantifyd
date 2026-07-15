"""research/81 — Phase 0 data audit of market_data.db (brief section 2).

Read-only. Safe to run while the 5-min backfill writes (WAL).
Emits research/81_swing_edge_discovery/reports/00_data_audit.md.

Checks:
  A. Inventory per timeframe (symbols, span, rows).
  B. Integrity: duplicates, OHLC consistency, non-positive prices, zero volume.
  C. Daily session coverage vs a consensus session calendar.
  D. 5-min bar structure for deep-history names (bars/session, out-of-hours bars).
  E. Corporate-action suspects: overnight |gap| > 25% on daily equities.
  F. Backfill splice consistency: price jump where new history meets old rows.
  G. Survivorship & universe notes (hand-written conclusions appended).

Usage (VPS): venv/bin/python3 research/81_swing_edge_discovery/scripts/data_audit.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "backtest_data" / "market_data.db"
STUDY = ROOT / "research" / "81_swing_edge_discovery"
REPORT = STUDY / "reports" / "00_data_audit.md"
CHECKPOINT = STUDY / "results" / "backfill_checkpoint.json"

DEEP_5MIN = ["NIFTY50", "INDIAVIX", "BANKNIFTY", "HDFCBANK", "ICICIBANK",
             "RELIANCE", "INFY", "TCS", "SBIN", "ITC", "HINDUNILVR",
             "KOTAKBANK", "BHARTIARTL"]

L = []          # report lines


def w(s: str = ""):
    L.append(s)
    print(s, flush=True)


def q(con, sql, params=()):
    return con.execute(sql, params).fetchall()


def main():
    t0 = time.time()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)

    w(f"# Phase 0 Data Audit — market_data.db (research/81)")
    w()
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')} IST · "
      f"DB: `{DB}` · read-only snapshot **while 5-min history backfill is running** "
      f"(broad-universe 5-min coverage will deepen; deep/daily series audited here are final).")
    w()

    # ---------------- A. Inventory ----------------
    w("## A. Inventory")
    w()
    w("| Timeframe | Symbols | First bar | Last bar | Rows |")
    w("|---|---|---|---|---|")
    for tf, syms, mn, mx, n in q(con,
            "SELECT timeframe, COUNT(DISTINCT symbol), MIN(date), MAX(date), COUNT(*) "
            "FROM market_data_unified GROUP BY timeframe"):
        w(f"| {tf} | {syms} | {mn} | {mx} | {n:,} |")
    w()
    w("5-min start-year histogram (per-symbol earliest bar):")
    w()
    w("| Start year | Symbols |")
    w("|---|---|")
    for y, n in q(con,
            "SELECT substr(mn,1,4), COUNT(*) FROM (SELECT symbol, MIN(date) mn "
            "FROM market_data_unified WHERE timeframe='5minute' GROUP BY symbol) "
            "GROUP BY 1 ORDER BY 1"):
        w(f"| {y} | {n} |")
    w()

    # ---------------- B. Integrity ----------------
    w("## B. Integrity")
    w()
    w("| Timeframe | Dup (sym,tf,date) | H < max(O,C) | L > min(O,C) | H < L | Price ≤ 0 | Zero-vol rows (equity day) |")
    w("|---|---|---|---|---|---|---|")
    for tf in ("day", "60minute", "5minute", "30minute"):
        dup = q(con, "SELECT COUNT(*) - COUNT(DISTINCT symbol||'|'||date) "
                     "FROM market_data_unified WHERE timeframe=?", (tf,))[0][0]
        hoc = q(con, "SELECT COUNT(*) FROM market_data_unified WHERE timeframe=? "
                     "AND high < MAX(open, close) - 1e-9", (tf,))[0][0]
        loc = q(con, "SELECT COUNT(*) FROM market_data_unified WHERE timeframe=? "
                     "AND low > MIN(open, close) + 1e-9", (tf,))[0][0]
        hl = q(con, "SELECT COUNT(*) FROM market_data_unified WHERE timeframe=? "
                    "AND high < low - 1e-9", (tf,))[0][0]
        np_ = q(con, "SELECT COUNT(*) FROM market_data_unified WHERE timeframe=? "
                     "AND (open<=0 OR high<=0 OR low<=0 OR close<=0)", (tf,))[0][0]
        zv = ""
        if tf == "day":
            zv = q(con, "SELECT COUNT(*) FROM market_data_unified WHERE timeframe='day' "
                        "AND volume=0 AND symbol NOT LIKE '%NIFTY%' "
                        "AND symbol NOT IN ('INDIAVIX','SENSEX')")[0][0]
        w(f"| {tf} | {dup} | {hoc} | {loc} | {hl} | {np_} | {zv} |")
    w()

    # ---------------- C. Daily session coverage ----------------
    w("## C. Daily session coverage (vs consensus calendar)")
    w()
    day = pd.read_sql_query(
        "SELECT symbol, date FROM market_data_unified WHERE timeframe='day'", con)
    per_date = day.groupby("date")["symbol"].size()
    sessions = set(per_date[per_date >= 50].index)     # a real session = ≥50 symbols traded
    w(f"Consensus sessions (≥50 symbols traded): **{len(sessions):,}** "
      f"({min(sessions)} → {max(sessions)})")
    grp = day[day["date"].isin(sessions)].groupby("symbol")["date"]
    stats = grp.agg(["min", "max", "size"]).rename(columns={"size": "have"})
    sess_sorted = sorted(sessions)
    sess_idx = pd.Series(range(len(sess_sorted)), index=sess_sorted)
    stats["expected"] = stats.apply(
        lambda r: sess_idx[r["max"]] - sess_idx[r["min"]] + 1, axis=1)
    stats["coverage"] = stats["have"] / stats["expected"]
    bad = stats[(stats["coverage"] < 0.95) & (stats["expected"] >= 250)]
    w(f"Symbols with ≥250 expected sessions: {int((stats['expected']>=250).sum()):,}; "
      f"of those, **{len(bad)} have <95% session coverage** (gaps within their own span).")
    if len(bad):
        w()
        w("| Symbol | Span | Have | Expected | Coverage |")
        w("|---|---|---|---|---|")
        for sym, r in bad.sort_values("coverage").head(20).iterrows():
            w(f"| {sym} | {r['min']}→{r['max']} | {int(r['have'])} | "
              f"{int(r['expected'])} | {r['coverage']:.1%} |")
    w()

    # ---------------- D. 5-min bar structure (deep names) ----------------
    w("## D. 5-min bar structure — deep-history names")
    w()
    w("NSE regular session 09:15–15:30 → 75 five-min bars (09:15…15:25).")
    w()
    w("| Symbol | Sessions | Median bars/sess | Sess <70 bars | Sess <30 bars (half-day/muhurat?) | Bars outside 09:15–15:29 |")
    w("|---|---|---|---|---|---|")
    for sym in DEEP_5MIN:
        df = pd.read_sql_query(
            "SELECT date FROM market_data_unified WHERE timeframe='5minute' "
            "AND symbol=?", con, params=(sym,))
        if df.empty:
            w(f"| {sym} | 0 | — | — | — | — |")
            continue
        day_part = df["date"].str[:10]
        time_part = df["date"].str[11:16]
        cnt = df.assign(day=day_part).groupby("day").size()
        out_hours = int(((time_part < "09:15") | (time_part > "15:29")).sum())
        w(f"| {sym} | {len(cnt):,} | {int(cnt.median())} | "
          f"{int((cnt < 70).sum())} | {int((cnt < 30).sum())} | {out_hours:,} |")
    w()

    # ---------------- E. Corporate-action suspects (daily equities) ----------------
    w("## E. Corporate-action suspects — overnight |gap| > 25% (daily, equities)")
    w()
    dayc = pd.read_sql_query(
        "SELECT symbol, date, close FROM market_data_unified WHERE timeframe='day' "
        "AND symbol NOT LIKE '%NIFTY%' AND symbol NOT IN ('INDIAVIX','SENSEX') "
        "ORDER BY symbol, date", con)
    dayc["prev"] = dayc.groupby("symbol")["close"].shift(1)
    dayc["gap"] = (dayc["close"] / dayc["prev"] - 1).abs()
    big = dayc[(dayc["gap"] > 0.25) & (dayc["prev"] > 5)]
    w(f"Rows scanned: {len(dayc):,} · overnight |gap|>25% events (prev close >₹5): "
      f"**{len(big):,}** across **{big['symbol'].nunique()}** symbols.")
    top = (big.groupby("symbol").agg(events=("gap", "size"), worst=("gap", "max"))
              .sort_values("events", ascending=False).head(25))
    if len(top):
        w()
        w("| Symbol | >25% gap events | Worst gap |")
        w("|---|---|---|")
        for sym, r in top.iterrows():
            w(f"| {sym} | {int(r['events'])} | {r['worst']:.0%} |")
    w()
    w("Interpretation: these are unadjusted splits/bonuses OR genuine crashes/circuits "
      "OR illiquid junk. Any symbol used by a strategy must be checked/adjusted; "
      "headline results prefer the F&O-liquid subset where Kite adjusts splits.")
    w()

    # ---------------- F. Backfill splice consistency ----------------
    w("## F. Backfill splice consistency (new 2015–24 history vs pre-existing rows)")
    w()
    if CHECKPOINT.exists():
        ck = json.loads(CHECKPOINT.read_text())
        done = {s: v for s, v in ck.get("done", {}).items() if v.get("was")}
        flagged = []
        for sym, v in done.items():
            b = v["was"].split(" ")[0]           # first pre-existing session
            prev = q(con, "SELECT close FROM market_data_unified WHERE timeframe='5minute' "
                          "AND symbol=? AND date < ? ORDER BY date DESC LIMIT 1", (sym, b))
            nxt = q(con, "SELECT open FROM market_data_unified WHERE timeframe='5minute' "
                         "AND symbol=? AND date >= ? ORDER BY date ASC LIMIT 1", (sym, b))
            if prev and nxt and prev[0][0] and nxt[0][0]:
                jump = abs(nxt[0][0] / prev[0][0] - 1)
                if jump > 0.15:
                    flagged.append((sym, b, prev[0][0], nxt[0][0], jump))
        w(f"Backfilled symbols checked so far: {len(done)} · "
          f"**splice jumps >15%: {len(flagged)}**")
        if flagged:
            w()
            w("| Symbol | Boundary | Close (new hist) | Open (old rows) | Jump |")
            w("|---|---|---|---|---|")
            for sym, b, p, n, j in sorted(flagged, key=lambda x: -x[4]):
                w(f"| {sym} | {b} | {p} | {n} | {j:.0%} |")
            w()
            w("**Action:** these symbols split/bonused after their original download; "
              "their pre-existing rows are on the OLD adjustment basis. Fix = delete + "
              "full re-download of the symbol's 5-min series (idempotent downloader).")
        w()
        w("_Re-run this audit after the backfill completes to cover all symbols._")
    else:
        w("(checkpoint not found — backfill not started; section re-runs post-backfill)")
    w()

    # ---------------- G. Notes ----------------
    w("## G. Survivorship, sessions & standing caveats")
    w()
    w("- **Survivorship:** the 5-min universe is ~381 of *today's* liquid names — "
      "survivorship-biased for any pre-2024 cross-sectional claim. Daily table (1,642 "
      "symbols incl. many delisted/inactive) is broader. Every report must state this; "
      "headline claims prefer NIFTY/BANKNIFTY + F&O-liquid large-caps.")
    w("- **Futures:** no futures series exist. Per user decision (2026-07-15), cash/index "
      "series are the futures proxy with a futures-style cost model. Basis risk noted.")
    w("- **Sessions:** counts <75 bars are early closes/outages; sessions <30 bars are "
      "usually muhurat (evening, ~1h) — strategies must not treat them as regular days.")
    w("- **60-min table:** stale (ends 2026-03) and thin (95 syms) — the study aggregates "
      "5-min → 15/30/60-min itself; daily table covers pre-2015.")
    w()
    w(f"_Audit runtime: {time.time()-t0:.0f}s_")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWritten: {REPORT}")
    con.close()


if __name__ == "__main__":
    sys.exit(main())
