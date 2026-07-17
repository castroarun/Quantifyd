"""research/83 EXP-T1: faithful Turtle S1/S2 on Indian F&O equities, G1 probe.

Pre-registered grid (LOCKED): system {S1: 20-in/10-out, S2: 55-in/20-out} x
direction {L, S} = 4 cells, per-side reporting. Entry next open after channel
close-break; exit next open after opposite-channel close-break OR at the 2N
stop intrabar (N = ATR20 Wilder-simple at signal); no pyramiding (phase 2),
no time-stop (trailing exit IS the exit). Universe: F&O (~81, CA-guarded).
IS 2005-01..2017-12; Val/OOS untouched. Costs: futures-proxy 3bp/side.
Prior art stated: long side ~ research/71 (expected positive); the NEW cell
is SHORT at multi-week trailing holds.

Dedicated simulator (engine is time-stop based): sanity-checked below on a
hand-computable synthetic series before the sweep runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R81 = Path(__file__).resolve().parents[2] / "81_swing_edge_discovery"
for p in (str(R81), str(R81.parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader                              # noqa: E402
from engine.costs import CostConfig                    # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
IS_START, IS_END = "2005-01-01", "2017-12-31"
COST = CostConfig(product="FUTURES_PROXY")
SYSTEMS = {"S1": (20, 10), "S2": (55, 20)}


def turtle_sim(df: pd.DataFrame, n_in: int, n_out: int, d: int,
               sym: str) -> pd.DataFrame:
    """One symbol, one direction. Entry: close beyond n_in-channel -> next
    open. Exit: close beyond opposite n_out-channel -> next open; or 2N stop
    intrabar (gap-through at open). One position at a time."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    hi_in = df["high"].rolling(n_in).max().shift(1).to_numpy(float)
    lo_in = df["low"].rolling(n_in).min().shift(1).to_numpy(float)
    hi_out = df["high"].rolling(n_out).max().shift(1).to_numpy(float)
    lo_out = df["low"].rolling(n_out).min().shift(1).to_numpy(float)
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20).mean().to_numpy(float)

    trades = []
    i, n = n_in + 1, len(df)
    in_pos = False
    ent_i = ent_px = stop = None
    while i < n - 1:
        if not in_pos:
            sig = (c[i] > hi_in[i]) if d == 1 else (c[i] < lo_in[i])
            if sig and not np.isnan(atr[i]) and not np.isnan(hi_in[i]):
                ent_i = i + 1
                ent_raw = o[ent_i]
                ent_px = COST.fill_price(ent_raw, is_buy=(d == 1))
                stop = ent_raw - d * 2 * atr[i]
                in_pos = True
                i = ent_i
        else:
            # stop first (conservative), evaluated intrabar from entry bar on
            hit = (o[i] <= stop or l[i] <= stop) if d == 1 else \
                  (o[i] >= stop or h[i] >= stop)
            if hit:
                raw = o[i] if ((o[i] <= stop) if d == 1 else (o[i] >= stop)) else stop
                trades.append((idx[ent_i], idx[i], ent_raw, ent_px, raw, "STOP"))
                in_pos = False
            else:
                x = (c[i] < lo_out[i]) if d == 1 else (c[i] > hi_out[i])
                if x and i + 1 < n:
                    raw = o[i + 1]
                    trades.append((idx[ent_i], idx[i + 1], ent_raw, ent_px, raw, "TRAIL"))
                    in_pos = False
                    i += 1
        i += 1

    rows = []
    for et, xt, eraw, ep, xraw, reason in trades:
        xp = COST.fill_price(xraw, is_buy=(d == -1))
        qty = 500_000 / ep
        charges = COST.side_cost(ep, qty, d == 1) + COST.side_cost(xp, qty, d == -1)
        rows.append({"symbol": sym, "direction": d, "entry_time": et,
                     "exit_time": xt, "exit_reason": reason,
                     "hold_days": (xt - et).days,
                     "gross_ret": d * (xraw / eraw - 1),
                     "net_ret": d * (xp / ep - 1) - charges / 500_000})
    return pd.DataFrame(rows)


def sanity():
    """Hand-computable: rising series breaks 20d high, trails out on 10d low."""
    n = 80
    px = np.concatenate([np.full(30, 100.0), np.linspace(100, 130, 30),
                         np.linspace(130, 110, 20)])
    df = pd.DataFrame({"open": px, "high": px * 1.005, "low": px * 0.995,
                       "close": px},
                      index=pd.bdate_range("2020-01-01", periods=n))
    tr = turtle_sim(df, 20, 10, 1, "SYN")
    assert len(tr) >= 1, "sanity: no trade on a clean trend"
    t0 = tr.iloc[0]
    assert t0.exit_reason in ("TRAIL", "STOP")
    assert t0.gross_ret > 0.05, f"sanity: trend trade should win big, got {t0.gross_ret}"
    print(f"sanity OK: entry {t0.entry_time.date()} exit {t0.exit_time.date()} "
          f"({t0.exit_reason}) gross {t0.gross_ret:.1%} hold {t0.hold_days}d")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    sanity()
    from services.data_manager import FNO_LOT_SIZES
    universe = sorted(FNO_LOT_SIZES.keys())
    all_tr = []
    for sym in universe:
        df = loader.load_bars(sym, "day", start="2004-01-01", end=IS_END)
        if len(df) < 300:
            continue
        ca = loader.ca_gap_flags(df)
        if len(ca) > 5:
            continue
        for sysname, (n_in, n_out) in SYSTEMS.items():
            for d in (1, -1):
                t = turtle_sim(df, n_in, n_out, d, sym)
                if len(t):
                    t = t[(t.entry_time >= IS_START) & (t.entry_time <= IS_END)]
                    t["cell"] = f"{sysname}_{'L' if d == 1 else 'S'}"
                    all_tr.append(t)
    tr = pd.concat(all_tr, ignore_index=True)
    tr.to_csv(RESULTS / "t1_trades.csv", index=False)
    print("\n=== T1 Turtle probe (net 3bp, IS 2005-2017) ===")
    for cell, g in tr.groupby("cell"):
        r = g.net_ret.to_numpy(float)
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        sym_pnl = g.groupby("symbol")["net_ret"].sum()
        yr = pd.to_datetime(g.entry_time).dt.year
        yr_net = g.groupby(yr)["net_ret"].mean()
        print(f"{cell}: n={len(r)} gross={g.gross_ret.mean()*1e4:.0f}bps "
              f"net={r.mean()*1e4:.0f}bps t={t:.2f} win={(r>0).mean():.2f} "
              f"avg_hold={g.hold_days.mean():.0f}d sym_pos={(sym_pnl>0).mean():.2f} "
              f"yr_pos={(yr_net>0).mean():.2f}", flush=True)
    print("\nT1 PROBE COMPLETE")


if __name__ == "__main__":
    main()
