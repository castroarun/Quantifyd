"""research/84 EXP-DIP1: drawdown dip-buy with averaging-down (x/y/z), G1 probe.

Pre-registration: ../DIP_AVERAGE_DAILY_SWEEP_STATUS.md (grid LOCKED there).
Campaign simulator with synthetic sanity check; 3 universe arms; campaign
stats + portfolio NAV (10 slots) vs NIFTYBEES.
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

from engine import loader, metrics                     # noqa: E402
from engine.costs import CostConfig                    # noqa: E402

RESULTS = HERE.parent / "results"
IS_START, IS_END = "2005-01-01", "2018-12-31"
COST = CostConfig(product="CASH_DELIVERY")
MAX_ADDS = 3
Z_STEP = 0.15
SLOTS = 10

NIFTY50 = ["ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
           "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
           "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH",
           "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR",
           "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JSWSTEEL", "KOTAKBANK",
           "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
           "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
           "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN",
           "TRENT", "ULTRACEMCO", "WIPRO"]
BLOWUPS = ["YESBANK", "RCOM", "SUZLON", "RPOWER", "JPASSOCIAT", "IDEA",
           "PCJEWELLER", "UNITECH", "IBULHSGFIN", "RELCAPITAL", "DHFL",
           "JETAIRWAYS"]


def campaign_sim(df: pd.DataFrame, ref_kind: str, x: float, y: float,
                 averaging: bool, sym: str, start: str, end: str):
    """Returns campaigns list. Tranche capital = 1 unit each (4 reserved).
    Campaign return reported on RESERVED capital (4 units) — the honest base."""
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    idx = df.index
    if ref_kind == "ATH":
        ref = df["close"].cummax().shift(1).to_numpy(float)
    else:
        ref = df["close"].rolling(252).max().shift(1).to_numpy(float)
    n = len(df)
    campaigns = []
    i = 260
    in_c = False
    fills = []      # (px_slipped, units)
    last_fill_raw = None
    ent_i = None
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    while i < n - 1:
        if not in_c:
            if not np.isnan(ref[i]) and c[i] <= ref[i] * (1 - x) \
                    and start_ts <= idx[i] <= end_ts:
                raw = o[i + 1]
                if raw <= 0:
                    i += 1
                    continue
                px = COST.fill_price(raw, is_buy=True)
                fills = [(px, 1.0)]
                last_fill_raw = raw
                ent_i = i + 1
                in_c = True
                i = ent_i
        else:
            qty = sum(u for _, u in fills)
            avg = sum(p * u for p, u in fills) / qty
            if averaging and len(fills) < 1 + MAX_ADDS \
                    and c[i] <= last_fill_raw * (1 - Z_STEP):
                raw = o[i + 1]
                if raw > 0:
                    fills.append((COST.fill_price(raw, is_buy=True), 1.0))
                    last_fill_raw = raw
                    i += 1
            elif c[i] >= avg * (1 + y):
                raw = o[i + 1]
                xp = COST.fill_price(raw, is_buy=False)
                invested = sum(p * u for p, u in fills)
                charges = sum(COST.side_cost(p, u * 500_000 / p, True) / 500_000 * p
                              for p, u in fills)  # approx per-unit charges
                pnl_frac = (xp * qty - invested) / (4.0 * fills[0][0])
                campaigns.append({
                    "symbol": sym, "entry": idx[ent_i], "exit": idx[i + 1],
                    "n_fills": len(fills), "days": (idx[i + 1] - idx[ent_i]).days,
                    "ret_reserved": pnl_frac, "open": False,
                    "ret_deployed": (xp * qty - invested) / invested})
                in_c = False
                i += 1
        i += 1
    if in_c:
        qty = sum(u for _, u in fills)
        invested = sum(p * u for p, u in fills)
        mark = c[n - 1] * qty
        campaigns.append({
            "symbol": sym, "entry": idx[ent_i], "exit": idx[n - 1],
            "n_fills": len(fills), "days": (idx[n - 1] - idx[ent_i]).days,
            "ret_reserved": (mark - invested) / (4.0 * fills[0][0]),
            "open": True,
            "ret_deployed": (mark - invested) / invested})
    return campaigns


def sanity():
    """V-shape: drop 30%, recover. ATH x=20 y=10 must buy the dip and exit +y."""
    px = np.concatenate([np.linspace(100, 110, 300),
                         np.linspace(110, 75, 60), np.linspace(75, 105, 80)])
    df = pd.DataFrame({"open": px, "close": px},
                      index=pd.bdate_range("2010-01-01", periods=len(px)))
    cs = campaign_sim(df, "ATH", 0.20, 0.10, True, "SYN", "2010-01-01", "2013-12-31")
    assert len(cs) == 1 and not cs[0]["open"], f"sanity: {cs}"
    assert cs[0]["ret_deployed"] > 0.08, f"sanity ret {cs[0]['ret_deployed']}"
    print(f"sanity OK: fills={cs[0]['n_fills']} ret_deployed={cs[0]['ret_deployed']:.1%} "
          f"days={cs[0]['days']}")


def arm_report(name, universe, grid):
    rows = []
    bars = {}
    for sym in universe:
        df = loader.load_bars(sym, "day", start="2004-01-01", end=IS_END)
        if len(df) >= 500:
            bars[sym] = df
    print(f"\n===== ARM {name}: {len(bars)} symbols =====", flush=True)
    all_cs = []
    for (ref, x, y, avg_on) in grid:
        cs = []
        for sym, df in bars.items():
            cs += campaign_sim(df, ref, x, y, avg_on, sym, IS_START, IS_END)
        if not cs:
            continue
        d = pd.DataFrame(cs)
        d["cell"] = f"{ref}_x{int(x*100)}_y{int(y*100)}_{'AVG' if avg_on else 'NOAVG'}"
        all_cs.append(d)
        r = d.ret_reserved.to_numpy(float)
        open_frac = d.open.mean()
        slot_wipe = (d.ret_reserved < -0.60).mean()
        rows.append({
            "cell": d.cell.iloc[0], "n": len(d),
            "win": round((d.ret_deployed > 0).mean(), 2),
            "mean_ret_res": round(r.mean() * 100, 2),
            "med_days": int(d.days.median()),
            "p95_days": int(d.days.quantile(0.95)),
            "open%": round(open_frac * 100, 0),
            "slot_wipe%": round(slot_wipe * 100, 1)})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False), flush=True)
    out = pd.concat(all_cs, ignore_index=True)
    out.to_csv(RESULTS / f"dip1_{name}_campaigns.csv", index=False)
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    sanity()
    from services.data_manager import FNO_LOT_SIZES
    GRID = [(ref, x, y, avg) for ref in ("ATH", "R252") for x in (0.15, 0.25, 0.35)
            for y in (0.10, 0.20) for avg in (False, True)]
    arm_report("nifty50", NIFTY50, GRID)
    arm_report("fno", sorted(FNO_LOT_SIZES.keys()), GRID)
    arm_report("blowups", BLOWUPS, GRID)
    print("\nDIP1 PROBE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
