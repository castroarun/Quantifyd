"""research/83 EXP-T2: book-level bakeoff — live breakout-book rules vs
faithful Turtle mechanics, long-only, NIFTY>200DMA-gated entries, F&O
universe, daily MTM NAV 2005-01..2023-12 (OOS 2024+ untouched).

Arms (LOCKED):
  A_livebook : Donchian-20 entry / Donchian-20 trailing exit, NO hard stop
               (research/71 winner), equal-notional 12%, cap 8.
  B_turtleN  : S1(20/10)+S2(55/20), 2N stops, N-sized (risk 0.75% of equity
               per unit on the 2N distance), unit cap 12.
  C_turtleEQ : same trades as B, equal-notional 10%, cap 12 — isolates the
               N-sizing effect (research/81 sizing-inversion lesson).
Gate applies to ENTRIES only (positions ride their exits) — same for all arms.
Selection metric (pre-declared): Calmar, tie-break Sharpe, DD must be <= live
book arm's DD * 1.2 to count as an upgrade candidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R81 = HERE.parents[1] / "81_swing_edge_discovery"
for p in (str(R81), str(R81.parents[1]), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader, metrics                     # noqa: E402
from run_turtle_probe import turtle_sim                # noqa: E402

RESULTS = HERE.parent / "results"
START, END = "2005-01-01", "2023-12-31"
MAX_POS_NOTIONAL = 0.25
BOOK_NOTIONAL_CAP = 1.5


def gate_series():
    nb = loader.load_bars("NIFTYBEES", "day", start="2004-01-01", end=END)
    return (nb["close"].shift(1) > nb["close"].rolling(200).mean().shift(1))


def collect(arm: str, bars: dict) -> pd.DataFrame:
    rows = []
    for sym, df in bars.items():
        if arm == "A_livebook":
            t = turtle_sim(df, 20, 20, 1, sym, stop_mult=None)
            t["risk_dist"] = np.nan
        else:
            parts = []
            for (ni, no) in ((20, 10), (55, 20)):
                tt = turtle_sim(df, ni, no, 1, sym, stop_mult=2.0)
                parts.append(tt)
            t = pd.concat(parts, ignore_index=True)
        rows.append(t)
    tr = pd.concat(rows, ignore_index=True)
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    tr["exit_time"] = pd.to_datetime(tr["exit_time"])
    tr = tr[(tr.entry_time >= START) & (tr.entry_time <= END)]
    return tr.sort_values("entry_time").reset_index(drop=True)


def nav_sim(trades, closes, atrs, gate, cap, sizing, label):
    trades = trades.copy()
    trades["eday"] = trades["entry_time"].dt.normalize()
    cal_set = set()
    for c in closes.values():
        cal_set |= set(c.index)
    cal = pd.DatetimeIndex(sorted(d for d in cal_set
                                  if pd.Timestamp(START) <= d <= pd.Timestamp(END)))
    eq, open_pos, ti = 1.0, [], 0
    equity = pd.Series(np.nan, index=cal)
    taken = skipped = 0
    for d in cal:
        while ti < len(trades) and trades.loc[ti, "eday"] <= d:
            tr = trades.loc[ti]
            ti += 1
            if tr["eday"] != d:
                continue
            if not bool(gate.get(d, False)):
                continue
            if len(open_pos) >= cap:
                skipped += 1
                continue
            # entry price approx = engine's slipped entry; derive raw
            ent_px_ser = closes[tr["symbol"]]
            book_notional = sum(p["qty"] * p["mark"] for p in open_pos)
            head = max(eq * BOOK_NOTIONAL_CAP - book_notional, 0.0)
            px = ent_px_ser.get(d, np.nan)
            if np.isnan(px):
                continue
            if sizing == "N":
                a = atrs[tr["symbol"]].get(d, np.nan)
                if np.isnan(a) or a <= 0:
                    continue
                qty = (eq * 0.0075) / (2 * a)
            else:
                qty = (eq * (0.12 if cap == 8 else 0.10)) / px
            qty = min(qty, (eq * MAX_POS_NOTIONAL) / px, head / px)
            if qty * px < eq * 0.005:
                skipped += 1
                continue
            open_pos.append({"symbol": tr["symbol"], "qty": qty, "mark": px,
                             "exit_day": tr["exit_time"].normalize(),
                             "net_ret": tr["net_ret"], "entry_px_mark": px,
                             "gross_ret": tr["gross_ret"]})
            taken += 1
        pnl, still = 0.0, []
        for p in open_pos:
            c = closes[p["symbol"]].get(d, np.nan)
            if d >= p["exit_day"]:
                # settle: total trade P&L = entry_mark*qty*net_ret rebased
                total = p["qty"] * p["entry_px_mark"] * p["net_ret"]
                booked = p["qty"] * (p["mark"] - p["entry_px_mark"])
                pnl += total - booked
            else:
                if not np.isnan(c):
                    pnl += p["qty"] * (c - p["mark"])
                    p["mark"] = c
                still.append(p)
        open_pos = still
        eq += pnl
        equity[d] = eq
    equity = equity.ffill()
    cs = metrics.curve_stats(equity)
    print(f"\n--- {label}: taken={taken} skipped={skipped} ---")
    for k, v in cs.items():
        print(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")
    yr = (metrics.per_year_table(equity.pct_change()) * 100).round(1)
    print(yr.T.to_string())
    equity.to_csv(RESULTS / f"t2_nav_{label}.csv")
    return cs


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    from services.data_manager import FNO_LOT_SIZES
    bars = {}
    for sym in sorted(FNO_LOT_SIZES.keys()):
        df = loader.load_bars(sym, "day", start="2004-01-01", end=END)
        if len(df) >= 300 and len(loader.ca_gap_flags(df)) <= 5:
            bars[sym] = df
    print(f"universe: {len(bars)}")
    closes = {s: b["close"] for s, b in bars.items()}
    atrs = {}
    for s, b in bars.items():
        pc = b["close"].shift(1)
        tr_ = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(),
                         (b["low"] - pc).abs()], axis=1).max(axis=1)
        atrs[s] = tr_.rolling(20).mean()
    gate = gate_series()

    tr_a = collect("A_livebook", bars)
    tr_b = collect("B_turtle", bars)
    print(f"trades: A={len(tr_a)} B/C={len(tr_b)}")
    nav_sim(tr_a, closes, atrs, gate, cap=8, sizing="EQ", label="A_livebook")
    nav_sim(tr_b, closes, atrs, gate, cap=12, sizing="N", label="B_turtleN")
    nav_sim(tr_b, closes, atrs, gate, cap=12, sizing="EQ", label="C_turtleEQ")
    bench = loader.load_bars("NIFTYBEES", "day", start=START, end=END)["close"]
    bs = metrics.curve_stats(bench / bench.iloc[0])
    print(f"\nNIFTYBEES B&H: CAGR {bs['cagr']:.2%} Sharpe {bs['sharpe']:.2f} "
          f"MaxDD {bs['max_dd']:.2%} Calmar {bs['calmar']:.2f}")
    print("\nT2 BAKEOFF COMPLETE")


if __name__ == "__main__":
    main()
