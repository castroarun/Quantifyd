"""Follow-ups battery: EXP-A8 (OR-width filter), EXP-H1 (decay-gate forensic),
EXP-B3 (deep-z short-fade 5-min timing). Pre-registration:
experiments/FOLLOWUPS_BATTERY/ORWIDTH_DECAYGATE_BTIMING_SWEEP_STATUS.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
for p in (str(STUDY), str(ROOT), str(STUDY / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader                              # noqa: E402
from engine.backtester import BTConfig, run_symbol     # noqa: E402
from engine.costs import CostConfig                    # noqa: E402
from run_g3_orb_validation import orb_entries          # noqa: E402

RESULTS = STUDY / "experiments" / "FOLLOWUPS_BATTERY" / "results"
IS_START, IS_END = "2015-02-01", "2021-09-30"
FULL_END = "2026-07-16"
CFG3 = {ts: BTConfig(cost=CostConfig(product="FUTURES_PROXY"),
                     time_stop_sessions=ts) for ts in (2, 4)}
CFG1 = BTConfig(cost=CostConfig(product="FUTURES_PROXY", slippage=0.0001),
                time_stop_sessions=4)


def pooled_line(tr, label):
    r = tr["net_ret"].to_numpy(float)
    if len(r) < 2:
        print(f"{label}: n={len(r)}")
        return
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    print(f"{label}: n={len(r)} gross={tr['gross_ret'].mean()*1e4:.1f} "
          f"net={r.mean()*1e4:.1f}bps t={t:.2f} win={(r > 0).mean():.3f}", flush=True)


def fno_bars(start, end, min_sessions=300):
    from services.data_manager import FNO_LOT_SIZES
    out = {}
    for sym in sorted(FNO_LOT_SIZES.keys()):
        df = loader.load_bars(sym, "5minute", start=start, end=end)
        if not df.empty and df.index.normalize().nunique() >= min_sessions:
            out[sym] = df
    return out


# ------------------------------------------------ A8: OR-width terciles
def a8(bars):
    print("\n===== EXP-A8: OR-width tercile filter (IS, exploratory) =====")
    nifty = loader.load_bars("NIFTY50", "5minute", start=IS_START, end=IS_END)
    sets = {"index": {"NIFTY50": nifty}, "stocks": bars}
    for name, group in sets.items():
        cfg = CFG1 if name == "index" else CFG3[4]
        trs = []
        for sym, df in group.items():
            tr = run_symbol(df, orb_entries(df, 12, IS_START, IS_END, 0.0025),
                            cfg, symbol=sym)
            if not len(tr):
                continue
            sess = df.index.normalize()
            pos = df.groupby(sess).cumcount()
            orh = df["high"].where(pos < 12).groupby(sess).transform("max")
            orl = df["low"].where(pos < 12).groupby(sess).transform("min")
            dopen = df["open"].groupby(sess).transform("first")
            width = ((orh - orl) / dopen).groupby(sess).first()
            wrank = width.rolling(252).rank(pct=True)
            ed = pd.to_datetime(tr["entry_time"]).dt.normalize()
            tr["wrank"] = wrank.reindex(ed).values
            trs.append(tr)
        allt = pd.concat(trs, ignore_index=True).dropna(subset=["wrank"])
        pooled_line(allt, f"A8 {name} UNCONDITIONAL")
        for lo, hi, lab in ((0, 1 / 3, "narrow"), (1 / 3, 2 / 3, "mid"),
                            (2 / 3, 1.01, "wide")):
            pooled_line(allt[(allt.wrank >= lo) & (allt.wrank < hi)],
                        f"A8 {name} {lab}")


# ------------------------------------------------ H1: decay-gate forensic
def h1():
    print("\n===== EXP-H1: decay-gate forensic (FULL period incl. consumed OOS) =====")
    bars = fno_bars("2015-02-01", FULL_END, min_sessions=500)
    trs = []
    for sym, df in bars.items():
        t = run_symbol(df, orb_entries(df, 12, IS_START, FULL_END, 0.0025),
                       CFG3[4], symbol=sym)
        trs.append(t)
    tr = pd.concat(trs, ignore_index=True)
    tr["month"] = pd.to_datetime(tr["entry_time"]).dt.to_period("M")
    monthly = tr.groupby("month")["net_ret"].agg(["mean", "size", "std"])
    monthly["t"] = monthly["mean"] / (monthly["std"] / np.sqrt(monthly["size"]))
    print(f"full ungated: n={len(tr)} net={tr.net_ret.mean()*1e4:.1f}bps")
    yr = tr.groupby(pd.to_datetime(tr.entry_time).dt.year)["net_ret"].mean() * 1e4
    print("ungated per-year bps:", dict(yr.round(1)))
    for win in (6, 12):
        for metric in ("mean", "t"):
            roll = monthly[metric].rolling(win).mean() if metric == "mean" \
                else monthly[metric].rolling(win).apply(
                    lambda x: x.mean(), raw=True)
            gate_on = (roll.shift(1) > 0)          # month m uses data <= m-1
            g = tr[tr["month"].map(gate_on).fillna(False)]
            r = g["net_ret"].to_numpy(float)
            months_on = int(gate_on.sum())
            late = g[pd.to_datetime(g.entry_time) >= "2025-01-01"]
            early = g[(pd.to_datetime(g.entry_time) >= "2017-01-01")
                      & (pd.to_datetime(g.entry_time) <= "2021-12-31")]
            all_late = tr[pd.to_datetime(tr.entry_time) >= "2025-01-01"]
            all_early = tr[(pd.to_datetime(tr.entry_time) >= "2017-01-01")
                           & (pd.to_datetime(tr.entry_time) <= "2021-12-31")]
            bleed_avoided = 1 - (late.net_ret.sum() / all_late.net_ret.sum()
                                 if all_late.net_ret.sum() < 0 else np.nan)
            gains_kept = (early.net_ret.sum() / all_early.net_ret.sum()
                          if all_early.net_ret.sum() > 0 else np.nan)
            flips = int((gate_on != gate_on.shift(1)).sum())
            print(f"gate {metric}>{0} trail{win}m: months_on={months_on} "
                  f"n={len(r)} net={r.mean()*1e4:.1f}bps "
                  f"2025-26 bleed avoided={bleed_avoided:.0%} "
                  f"2017-21 gains kept={gains_kept:.0%} flips={flips}", flush=True)
    tr.to_csv(RESULTS / "h1_full_trades.csv", index=False)


# ------------------------------------------------ B3: reversion 5-min timing
def b3(bars):
    print("\n===== EXP-B3: deep-z short fade, 5-min entry timing (IS) =====")
    from engine.loader import to_daily

    for sym, df in bars.items():
        pass  # bars reused below

    results = {}
    for mode in ("open", "confirm"):
        for ts in (2, 4):
            trs = []
            for sym, df5 in bars.items():
                d = to_daily(df5)
                sma20 = d["close"].rolling(20).mean()
                sd20 = d["close"].rolling(20).std()
                sma200 = d["close"].rolling(200).mean()
                pc = d["close"].shift(1)
                tr_ = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                                 (d["low"] - pc).abs()], axis=1).max(axis=1)
                atr = tr_.rolling(14).mean()
                z = (d["close"] - sma20) / sd20
                sig_days = d.index[(z >= 2.5) & (d["close"] < sma200)
                                   & atr.notna() & (sd20 > 0)]
                # entry on the NEXT session
                didx = list(d.index)
                nxt = {}
                for s in sig_days:
                    i = didx.index(s)
                    if i + 1 < len(didx):
                        nxt[didx[i + 1]] = s
                if not nxt:
                    continue
                sess = df5.index.normalize()
                pos = df5.groupby(sess).cumcount()
                rows = []
                for eday, s in nxt.items():
                    day = df5[sess == eday]
                    if day.empty:
                        continue
                    stop = d.loc[s, "close"] + 2.5 * atr.loc[s]
                    target = sma20.loc[s]
                    if mode == "open":
                        sig_bar = day.index[0]
                    else:
                        or_low = day["low"].iloc[:6].min()
                        after = day.iloc[6:]
                        below = after[after["close"] < or_low]
                        if below.empty:
                            continue
                        sig_bar = below.index[0]
                    rows.append({"idx": sig_bar, "stop": stop, "target": target})
                if not rows:
                    continue
                ent = pd.DataFrame({"direction": -1,
                                    "stop": [r["stop"] for r in rows],
                                    "target": [r["target"] for r in rows]},
                                   index=pd.DatetimeIndex([r["idx"] for r in rows]))
                ent = ent[(ent.index >= pd.Timestamp(IS_START))
                          & (ent.index <= pd.Timestamp(IS_END + " 23:59:59"))]
                trs.append(run_symbol(df5, ent, CFG3[ts], symbol=sym))
            allt = pd.concat(trs, ignore_index=True)
            results[(mode, ts)] = allt
            pooled_line(allt, f"B3 {mode} ts{ts}")
    # baseline-vs-confirm on the SAME days is implicit: confirm subset of days


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bars_is = fno_bars(IS_START, IS_END)
    print(f"F&O bars loaded: {len(bars_is)} symbols ({(time.time()-t0)/60:.1f}m)")
    a8(bars_is)
    b3(bars_is)
    h1()
    print(f"\nFOLLOWUPS BATTERY done in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
