# -*- coding: utf-8 -*-
"""Rank the Nifty Total Market (750) by the LIVE book's own momentum rule, as of today.

Arun asked what the top 15 would be if the momentum book sourced from Nifty 750 instead of Nifty
200, and asked for NSE price data rather than our own database. Both matter:

  - our market_data.db only carries 409 symbols current to 2026-08-19, so ranking off it would
    silently produce a Nifty-400-ish list wearing a "750" label;
  - the constituent list is the official one from niftyindices.com (ind_niftytotalmarket_list.csv),
    and every price bar comes from Kite, which is the NSE feed itself.

The score is EXACTLY what the live book uses (services/momentum_paper.py::_rs_basket):
    rsblend = 0.5 * (6m return / benchmark 6m return) + 0.5 * (12m return / benchmark 12m return)
with the benchmark being NIFTYBEES, matching the live gate. No new formula is invented here — the
question is what the existing rule says about a wider universe.

Writes incrementally so a mid-run failure keeps the bars already fetched.
"""
import csv, json, os, sys, time
from datetime import datetime, timedelta

sys.path.insert(0, "/home/arun/quantifyd")
OUT = "/home/arun/quantifyd/research/113_nifty750_momentum_rank/results"
os.makedirs(OUT, exist_ok=True)
BARS_CSV = os.path.join(OUT, "bars_cache.csv")
RANK_CSV = os.path.join(OUT, "ranking.csv")
STATUS = "/home/arun/quantifyd/research/113_nifty750_momentum_rank/NIFTY750_MOMENTUM_DAILY_RANK_STATUS.md"

BENCH = "NIFTYBEES"
LOOKBACKS = ((126, 0.5), (252, 0.5))     # 6m and 12m, equal weight — the live book's blend

from services import momentum_paper as mp
import pandas as pd, numpy as np


def log(msg):
    print(msg, flush=True)
    try:
        with open(STATUS, "a", encoding="utf-8") as f:
            f.write(f"| {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg} |\n")
    except Exception:
        pass


def main():
    syms = []
    with open("/tmp/ind_niftytotalmarket_list.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            s = (row.get("Symbol") or "").strip()
            if s and (row.get("Series") or "EQ").strip() == "EQ":
                syms.append(s)
    syms = sorted(set(syms))
    log(f"constituents parsed: {len(syms)} EQ symbols from the official NSE list")

    k = mp._kite()
    ins = {i["tradingsymbol"]: i for i in k.instruments("NSE")
           if i.get("instrument_type") == "EQ" or i["tradingsymbol"] == BENCH}
    want = [s for s in syms if s in ins]
    missing = [s for s in syms if s not in ins]
    log(f"matched to Kite instruments: {len(want)} (unmatched: {len(missing)})")
    if BENCH not in ins:
        log(f"FATAL: benchmark {BENCH} not found on NSE"); return
    want = [BENCH] + [s for s in want if s != BENCH]

    end = datetime.now()
    start = end - timedelta(days=430)            # >12 months of calendar for 252 trading bars

    done = {}
    if os.path.exists(BARS_CSV):
        with open(BARS_CSV, encoding="utf-8") as f:
            for r in csv.reader(f):
                done.setdefault(r[0], []).append((r[1], float(r[2])))
        log(f"resuming: {len(done)} symbols already cached")

    fh = open(BARS_CSV, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    fetched, failed = 0, []
    for i, s in enumerate(want, 1):
        if s in done:
            continue
        try:
            bars = k.historical_data(ins[s]["instrument_token"], start, end, "day")
            rows = [(b["date"].strftime("%Y-%m-%d"), float(b["close"]), int(b.get("volume") or 0))
                    for b in bars if b.get("close")]
            for d, c, v in rows:
                w.writerow([s, d, c, v])
            done[s] = [(d, c) for d, c, v in rows]
            fetched += 1
            if fetched % 50 == 0:
                fh.flush()
                log(f"fetched {fetched} / {len(want)} (at {s}, {i} scanned)")
        except Exception as e:
            failed.append(s)
            if len(failed) <= 5:
                log(f"  fetch failed {s}: {str(e)[:70]}")
        time.sleep(0.35)                          # Kite historical: 3 req/sec
    fh.close()
    log(f"fetch complete: {fetched} newly fetched, {len(done)} total, {len(failed)} failed")

    # ---------- build the panel ----------
    px, vol = {}, {}
    with open(BARS_CSV, encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) < 4:
                continue
            px.setdefault(r[0], {})[r[1]] = float(r[2])
            vol.setdefault(r[0], {})[r[1]] = float(r[3])
    close = pd.DataFrame(px).sort_index()
    volume = pd.DataFrame(vol).sort_index()
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    close = close.ffill()
    log(f"panel built: {close.shape[1]} symbols x {close.shape[0]} days, last {close.index[-1].date()}")

    if BENCH not in close.columns:
        log("FATAL: benchmark missing from panel"); return

    # ---------- the live book's score ----------
    sc = None
    for L, wt in LOOKBACKS:
        if len(close) <= L:
            log(f"FATAL: only {len(close)} bars, need > {L}"); return
        p0, p1 = close.iloc[-L - 1], close.iloc[-1]
        r = (p1 / p0) / (p1[BENCH] / p0[BENCH]) * wt
        sc = r if sc is None else sc.add(r, fill_value=np.nan)
    sc = sc.drop(index=[BENCH], errors="ignore").dropna()

    adv = (close * volume).tail(126).median()     # rupee traded value, 6-month median
    r6 = (close.iloc[-1] / close.iloc[-127] - 1) * 100
    r12 = (close.iloc[-1] / close.iloc[-253] - 1) * 100
    b6 = (close[BENCH].iloc[-1] / close[BENCH].iloc[-127] - 1) * 100
    b12 = (close[BENCH].iloc[-1] / close[BENCH].iloc[-253] - 1) * 100

    rank = sc.sort_values(ascending=False)
    with open(RANK_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "symbol", "rsblend", "ret_6m_pct", "ret_12m_pct", "adv_cr", "px"])
        for n, (s, v) in enumerate(rank.items(), 1):
            wr.writerow([n, s, round(float(v), 4), round(float(r6.get(s, np.nan)), 1),
                         round(float(r12.get(s, np.nan)), 1),
                         round(float(adv.get(s, 0)) / 1e7, 2), round(float(close[s].iloc[-1]), 1)])

    log(f"ranking written: {len(rank)} symbols scored. Benchmark 6m {b6:+.1f}% / 12m {b12:+.1f}%")
    print("\n" + "=" * 92)
    print(f"TOP 15 — Nifty Total Market (750) by the live book's rsblend, as of {close.index[-1].date()}")
    print("=" * 92)
    print(f"{'#':>3} {'symbol':<14}{'rsblend':>9}{'6m %':>9}{'12m %':>9}{'ADV Rs cr':>11}{'price':>10}")
    for n, (s, v) in enumerate(list(rank.items())[:15], 1):
        print(f"{n:>3} {s:<14}{v:>9.3f}{r6.get(s, float('nan')):>9.1f}{r12.get(s, float('nan')):>9.1f}"
              f"{adv.get(s, 0)/1e7:>11.2f}{close[s].iloc[-1]:>10.1f}")
    print("=" * 92)
    print(f"benchmark {BENCH}: 6m {b6:+.1f}%, 12m {b12:+.1f}%  (rsblend 1.00 = matched the index)")
    thin = [s for s, _ in list(rank.items())[:15] if float(adv.get(s, 0)) / 1e7 < 5]
    if thin:
        print(f"NOTE thin liquidity (<Rs5cr/day median): {', '.join(thin)}")


if __name__ == "__main__":
    main()
