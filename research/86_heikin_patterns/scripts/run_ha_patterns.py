"""research/86: Heikin-Ashi continuation patterns, multi-TF (user chart idea).

Pre-registration: ../HEIKIN_PATTERNS_MULTI_TF_SWEEP_STATUS.md (grid LOCKED).
Detection on HA candles; triggers/fills on REAL prices. Entry: bar CLOSES
beyond the pattern's real extreme -> fill next open. Exit: opposite-pattern
break (same mechanics). Long and short as separate cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R81 = HERE.parents[1] / "81_swing_edge_discovery"
for p in (str(R81), str(R81.parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader                              # noqa: E402
from engine.costs import CostConfig                    # noqa: E402

RESULTS = HERE.parent / "results"
IDX_SYMS = ["NIFTY50", "BANKNIFTY"]
DEEP9 = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "TCS", "SBIN",
         "ITC", "HINDUNILVR", "BHARTIARTL"]
INTRA_IS = ("2015-02-01", "2021-09-30")
DAILY_IS = ("2005-01-01", "2017-12-31")
COST1 = CostConfig(product="FUTURES_PROXY", slippage=0.0001)
COST3 = CostConfig(product="FUTURES_PROXY", slippage=0.0003)


def heikin(df: pd.DataFrame) -> pd.DataFrame:
    ha_c = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_o = np.empty(len(df))
    ha_o[0] = df["open"].iloc[0]
    hac = ha_c.to_numpy()
    for i in range(1, len(df)):
        ha_o[i] = (ha_o[i - 1] + hac[i - 1]) / 2
    out = pd.DataFrame(index=df.index)
    out["ha_open"], out["ha_close"] = ha_o, hac
    out["green"] = out["ha_close"] > out["ha_open"]
    out["ha_low"] = np.minimum.reduce([df["low"].to_numpy(), ha_o, hac])
    out["nowick_lo"] = np.isclose(out["ha_low"], np.minimum(ha_o, hac))
    return out


def pattern_ready(ha, i, kind, d):
    """Is a d-direction pattern complete AT bar i (so trigger armed for i+1..)?"""
    g = ha["green"].to_numpy()
    if kind == "FLIP1":
        if i < 1:
            return False
        return (g[i] and not g[i - 1]) if d == 1 else (not g[i] and g[i - 1])
    n = 3 if kind == "3GREEN" else 2
    if i < n - 1:
        return False
    seq = g[i - n + 1:i + 1]
    ok = seq.all() if d == 1 else (~seq).all()
    if ok and kind == "2GREEN_NW":
        nw = ha["nowick_lo"].to_numpy()[i - 1:i + 1]
        ok = bool(nw.all()) if d == 1 else ok  # no-upper-wick mirror for shorts
    return bool(ok)


def sim(df, ha, kind, d, cost, sym, lo, hi):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    n = len(df)
    npat = 3 if kind == "3GREEN" else (1 if kind == "FLIP1" else 2)
    trades = []
    in_pos = False
    ent_i = ent_raw = ent_px = None
    trig = None            # armed entry trigger level (real extreme)
    xtrig = None           # armed exit trigger
    lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi + " 23:59:59")
    for i in range(4, n - 1):
        if not in_pos:
            if kind == "FLIP1":
                # classic: enter on the flip itself, next open
                if pattern_ready(ha, i, kind, d) and lo_ts <= idx[i] <= hi_ts:
                    ent_i = i + 1
                    ent_raw = o[ent_i]
                    ent_px = cost.fill_price(ent_raw, is_buy=(d == 1))
                    in_pos, xtrig = True, None
                continue
            if pattern_ready(ha, i, kind, d):
                ext = h[i - npat + 1:i + 1].max() if d == 1 else \
                    l[i - npat + 1:i + 1].min()
                trig = ext
            if trig is not None and lo_ts <= idx[i] <= hi_ts:
                fired = (c[i] > trig) if d == 1 else (c[i] < trig)
                if fired:
                    ent_i = i + 1
                    ent_raw = o[ent_i]
                    ent_px = cost.fill_price(ent_raw, is_buy=(d == 1))
                    in_pos, trig, xtrig = True, None, None
        else:
            if i <= ent_i:
                continue
            # arm/refresh exit trigger on opposite pattern
            if kind == "FLIP1":
                if pattern_ready(ha, i, kind, -d):
                    xr = o[i + 1]
                    xp = cost.fill_price(xr, is_buy=(d == -1))
                    qty = 500_000 / ent_px
                    ch = cost.side_cost(ent_px, qty, d == 1) \
                        + cost.side_cost(xp, qty, d == -1)
                    trades.append({"symbol": sym, "entry_time": idx[ent_i],
                                   "exit_time": idx[i + 1],
                                   "hold_bars": i + 1 - ent_i,
                                   "gross_ret": d * (xr / ent_raw - 1),
                                   "net_ret": d * (xp / ent_px - 1) - ch / 500_000})
                    in_pos = False
                continue
            opp = -d
            npo = 3 if kind == "3GREEN" else 2
            if pattern_ready(ha, i, "2GREEN" if kind == "2GREEN_NW" else kind, opp):
                xtrig = l[i - npo + 1:i + 1].min() if d == 1 else \
                    h[i - npo + 1:i + 1].max()
            if xtrig is not None:
                xfired = (c[i] < xtrig) if d == 1 else (c[i] > xtrig)
                if xfired:
                    xr = o[i + 1]
                    xp = cost.fill_price(xr, is_buy=(d == -1))
                    qty = 500_000 / ent_px
                    ch = cost.side_cost(ent_px, qty, d == 1) \
                        + cost.side_cost(xp, qty, d == -1)
                    trades.append({"symbol": sym, "entry_time": idx[ent_i],
                                   "exit_time": idx[i + 1],
                                   "hold_bars": i + 1 - ent_i,
                                   "gross_ret": d * (xr / ent_raw - 1),
                                   "net_ret": d * (xp / ent_px - 1) - ch / 500_000})
                    in_pos, xtrig = False, None
    return pd.DataFrame(trades)


def pooled(trs, label):
    tr = pd.concat(trs, ignore_index=True) if trs else pd.DataFrame()
    if len(tr) < 2:
        print(f"{label}: n={len(tr)}")
        return
    r = tr["net_ret"].to_numpy(float)
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    sym_pnl = tr.groupby("symbol")["net_ret"].sum()
    yr = pd.to_datetime(tr["entry_time"]).dt.year
    yr_net = tr.groupby(yr)["net_ret"].mean()
    print(f"{label}: n={len(r)} gross={tr.gross_ret.mean()*1e4:.0f}bps "
          f"net={r.mean()*1e4:.0f}bps t={t:.2f} win={(r>0).mean():.2f} "
          f"hold={tr.hold_bars.mean():.0f}bars sym+={(sym_pnl>0).mean():.2f} "
          f"yr+={(yr_net>0).mean():.2f}", flush=True)
    return tr


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    # data: intraday sets
    intra = {}
    for sym in IDX_SYMS + DEEP9:
        df5 = loader.load_bars(sym, "5minute", start=INTRA_IS[0], end=INTRA_IS[1])
        if df5.empty:
            continue
        intra[sym] = {15: loader.resample_intraday(df5, 15),
                      30: loader.resample_intraday(df5, 30),
                      60: loader.resample_intraday(df5, 60)}
    from services.data_manager import FNO_LOT_SIZES
    daily = {}
    for sym in sorted(FNO_LOT_SIZES.keys()):
        df = loader.load_bars(sym, "day", start="2004-06-01", end=DAILY_IS[1])
        if len(df) >= 300:
            daily[sym] = df
    print(f"intraday syms: {len(intra)} | daily syms: {len(daily)}", flush=True)

    all_rows = []
    for kind in ("FLIP1", "2GREEN", "3GREEN", "2GREEN_NW"):
        for d in (1, -1):
            for tf in (15, 30, 60):
                trs = []
                for sym, tfs in intra.items():
                    df = tfs[tf]
                    cost = COST1 if sym in IDX_SYMS else COST3
                    trs.append(sim(df, heikin(df), kind, d, cost, sym,
                                   INTRA_IS[0], INTRA_IS[1]))
                tr = pooled(trs, f"{kind}_{'L' if d==1 else 'S'}_{tf}m")
                if tr is not None:
                    tr["cell"] = f"{kind}_{'L' if d==1 else 'S'}_{tf}m"
                    all_rows.append(tr)
            trs = [sim(df, heikin(df), kind, d,
                       COST3, sym, DAILY_IS[0], DAILY_IS[1])
                   for sym, df in daily.items()]
            tr = pooled(trs, f"{kind}_{'L' if d==1 else 'S'}_D")
            if tr is not None:
                tr["cell"] = f"{kind}_{'L' if d==1 else 'S'}_D"
                all_rows.append(tr)
    pd.concat(all_rows, ignore_index=True).to_csv(RESULTS / "ha_trades.csv",
                                                  index=False)
    print("\nHA PATTERNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
