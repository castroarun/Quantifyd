"""Phase 05 — today's live top-15 from the recommended config.

Config = `q0.5_dd__v__REG` on the `mid_120d_N15` core (Phase 03 winner:
35.3% CAGR / -24.6% DD / Sharpe 1.53 / Calmar 1.44).

LIVE-PICK scope differs from the backtest (see STATUS-MD §2): the live
eligible universe is the USER-SUPPLIED 100 MQ100 constituents
(2026-05-15), NOT the reconstructed PIT liquidity band. We then apply the
exact same signal stack:
  1. RS_i = (P_i[t]/P_i[t-120]) / (NIFTYBEES[t]/NIFTYBEES[t-120])
  2. Quality screen: >= 50% of trailing-12m 21-session blocks positive
  3. Regime gate: if NIFTYBEES[t] < mean(NIFTYBEES last 200 sessions)
     the live strategy is FLAT (6.5% cash) — we still print the ranked
     would-hold list but mark the regime state loudly.
Rank quality-passers by RS, take top-15 equal-weight.

NOTE: on the laptop this reads the FROZEN snapshot DB. Re-run on the VPS
(canonical, current data) for a true today-dated list. The as-of date is
printed and written into the output.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "backtest_data" / "market_data.db"
RES = HERE / "results"; RES.mkdir(exist_ok=True)

L = 120
TOPN = 15
BENCH = "NIFTYBEES"
Q_POSFRAC = 0.50
SMA_N = 200
# supplied-ticker -> DB-symbol renames (from 01_reconstruct_universe.py)
REMAP = {"360ONE": "IIFLWAM", "UNOMINDA": "MINDAIND",
         "NAVA": "NAVABHARAT", "GVT&D": "GET&D"}


def own_pos_frac(s):
    s = s.dropna().tail(252)
    if len(s) < 120:
        return None
    blocks = [s.iloc[i:i + 21] for i in range(0, len(s) - 1, 21)]
    rets = [b.iloc[-1] / b.iloc[0] - 1 for b in blocks
            if len(b) >= 2 and b.iloc[0] > 0]
    return float(np.mean([r > 0 for r in rets])) if rets else 0.0


def main():
    supplied = [s.strip() for s in
                open(HERE / "universe_mq100_2026-05-15.csv") if s.strip()]
    db_syms = sorted({REMAP.get(s, s) for s in supplied} | {BENCH})
    rev = {REMAP.get(s, s): s for s in supplied}  # db -> display

    con = sqlite3.connect(DB)
    q = ("SELECT symbol,date,close FROM market_data_unified "
         "WHERE timeframe='day' AND symbol IN (%s) AND close IS NOT NULL "
         "ORDER BY symbol,date" % ",".join("?" * len(db_syms)))
    df = pd.read_sql(q, con, params=db_syms, parse_dates=["date"])
    con.close()
    close = df.pivot_table(index="date", columns="symbol",
                           values="close").sort_index()

    if BENCH not in close.columns:
        print(f"FATAL: {BENCH} not in DB"); return
    nb = close[BENCH].dropna()
    asof = nb.index.max()
    h = close.loc[:asof]
    nbh = h[BENCH].dropna()
    if len(nbh) <= L or len(nbh) < SMA_N:
        print("FATAL: insufficient benchmark history"); return

    # --- regime gate ---
    sma200 = nbh.tail(SMA_N).mean()
    nb_last = nbh.iloc[-1]
    regime_on = nb_last >= sma200

    # --- RS + quality over the supplied universe ---
    nf = nbh.iloc[-1] / nbh.iloc[-L - 1]
    rows = []
    for sym in close.columns:
        if sym == BENCH:
            continue
        s = h[sym].dropna()
        if len(s) <= L:
            continue
        rs = (s.iloc[-1] / s.iloc[-L - 1]) / nf
        pf = own_pos_frac(s)
        if pf is None:
            continue
        rows.append(dict(symbol=rev.get(sym, sym), db_symbol=sym,
                          rs=round(rs, 4), pos_frac=round(pf, 2),
                          last_close=round(float(s.iloc[-1]), 2),
                          quality_pass=pf >= Q_POSFRAC))
    r = pd.DataFrame(rows)
    passed = r[r.quality_pass].sort_values("rs", ascending=False)
    top = passed.head(TOPN).reset_index(drop=True)
    top.index += 1

    miss = sorted(set(REMAP.get(s, s) for s in supplied) - set(close.columns))
    out = RES / "live_top15.csv"
    top.to_csv(out)

    print(f"=== LIVE TOP-{TOPN} — config q0.5_dd__v__REG (mid_120d_N15 core) ===")
    print(f"As-of (snapshot) date : {asof.date()}  "
          f"[laptop=frozen; rerun on VPS for current]")
    print(f"Universe              : supplied MQ100 ({len(supplied)} names, "
          f"{len(r)} with >= {L}+1d history; {len(miss)} missing data)")
    print(f"NIFTYBEES regime      : last={nb_last:.2f} vs SMA200={sma200:.2f}"
          f"  -> {'RISK-ON (deploy)' if regime_on else 'RISK-OFF -> STRATEGY FLAT (6.5% cash)'}")
    print(f"Quality screen        : pos-month frac >= {Q_POSFRAC} "
          f"({r.quality_pass.sum()}/{len(r)} pass)")
    if not regime_on:
        print("\n*** REGIME RISK-OFF: per the rules the live book holds NO "
              "equity now (cash @6.5%). List below is the would-hold ranking "
              "for when regime flips back risk-on. ***")
    print()
    print(top[["symbol", "db_symbol", "rs", "pos_frac",
               "last_close"]].to_string())
    if miss:
        print(f"\nSupplied names with no/short DB history "
              f"({len(miss)}): {miss}")
    print(f"\nWritten -> {out}")
    print("\nNEXT: overlay current web-sourced fundamentals "
          "(ROE / D-E / profit growth) on these names as the human "
          "quality cross-check (done separately, not from price data).")


if __name__ == "__main__":
    main()
