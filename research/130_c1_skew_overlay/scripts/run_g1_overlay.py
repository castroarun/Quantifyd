#!/usr/bin/env python3
"""research/130 G1 — price the EXTRA credit-spread unit on each C1 trade.
For every liquid C1 trade (from r/127 phase_b2): put spread = sell PE@Ks_pe +
buy PE@Kp (one extra unit, same strikes as the parent's put side); call spread
= sell CE@Ks_ce + buy CE@Kc. Held over the SAME entry->exit window, priced on
bhav closes, net of costs. Indicator states at entry recorded for gating.
Output: results/g1_overlay.csv"""
import math, sqlite3, time, csv, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "89_short_monthly_straddle" / "scripts"))
import engine as E

RESULTS = HERE.parent / "results"
R127 = HERE.parent.parent / "127_stock_neutral_wings" / "results"
OUT = RESULTS / "g1_overlay.csv"
SLIP = 0.005

FIELDS = ["symbol","expiry","entry_date","exit_date","year","S0",
          "ps_net_pct","cs_net_pct","sma200","ema2050","rsi50","stochkd"]

def leg(con, sym, exp, K, ot, day):
    r = con.execute("SELECT close FROM nse_options_bhav WHERE symbol=? AND expiry_date=? "
                    "AND strike=? AND option_type=? AND trade_date=? AND close>0",
                    (sym, exp, K, ot, day)).fetchone()
    return float(r[0]) if r else None

def spread_net(con, sym, exp, Ks, Kw, ot, d_in, d_out, S0, spot_out):
    """Net %S0 of ONE extra unit: sell Ks, buy Kw (both `ot`), entry->exit closes."""
    s0 = leg(con, sym, exp, Ks, ot, d_in)
    w0 = leg(con, sym, exp, Kw, ot, d_in)
    if s0 is None or w0 is None:
        return None
    sx = leg(con, sym, exp, Ks, ot, d_out)
    wx = leg(con, sym, exp, Kw, ot, d_out)
    if sx is None:                       # settle at intrinsic (pessimistic wing=intrinsic)
        if spot_out is None:
            return None
        sx = max(0.0, (spot_out - Ks) if ot == "CE" else (Ks - spot_out))
        wx = max(0.0, (spot_out - Kw) if ot == "CE" else (Kw - spot_out))
    if wx is None:
        wx = 0.0
    credit = s0 - w0
    gross = credit - (sx - wx)
    turn = s0 + w0 + sx + wx
    net = gross - SLIP * turn - 0.0010 * s0 - 0.0005 * turn
    return net / S0

def states(d):
    c = d["close"]
    sma200 = c > c.rolling(200).mean()
    ema2050 = c.ewm(span=20, adjust=False).mean() > c.ewm(span=50, adjust=False).mean()
    dd = c.diff()
    g = dd.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    l = (-dd).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    rsi50 = (100 - 100/(1 + g/l)) > 50
    lo = d["low"].rolling(14).min(); hi = d["high"].rolling(14).max()
    k = (100*(c - lo)/(hi - lo)).rolling(3).mean()
    stochkd = k > k.rolling(3).mean()
    return pd.DataFrame(dict(sma200=sma200, ema2050=ema2050, rsi50=rsi50, stochkd=stochkd))

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(E.db_path())
    c1 = pd.read_csv(R127 / "phase_b2_trades.csv")
    c1 = c1[(c1.config == "C1_E45X21W7K25_noSL") & (c1.atm_vol >= 100) & (c1.wing_vol_min >= 10)]
    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, usecols=["symbol"])["symbol"].unique())
    hdr = not OUT.exists()
    syms = sorted(c1.symbol.unique())
    for i, s in enumerate(syms):
        if s in done:
            continue
        t0 = time.time()
        d = E.load_daily(s, conn)
        if d.empty:
            continue
        st = states(d)
        close = d["close"]
        rows = []
        for _, t in c1[c1.symbol == s].iterrows():
            d_in, d_out = t["entry_date"], t["exit_date"]
            spot_out = float(close.loc[d_out]) if d_out in close.index.strftime("%Y-%m-%d") else None
            # index lookup via string dates
            try:
                spot_out = float(close.loc[pd.Timestamp(d_out)])
            except KeyError:
                spot_out = None
            ps = spread_net(conn, s, t["expiry"], t["Ks_pe"], t["Kp"], "PE",
                            d_in, d_out, t["S0"], spot_out)
            cs = spread_net(conn, s, t["expiry"], t["Ks_ce"], t["Kc"], "CE",
                            d_in, d_out, t["S0"], spot_out)
            try:
                sr = st.loc[pd.Timestamp(d_in)]
            except KeyError:
                continue
            rows.append(dict(symbol=s, expiry=t["expiry"], entry_date=d_in,
                             exit_date=d_out, year=t["year"], S0=t["S0"],
                             ps_net_pct=round(ps, 5) if ps is not None else "",
                             cs_net_pct=round(cs, 5) if cs is not None else "",
                             sma200=int(bool(sr["sma200"])), ema2050=int(bool(sr["ema2050"])),
                             rsi50=int(bool(sr["rsi50"])), stochkd=int(bool(sr["stochkd"]))))
        with open(OUT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if hdr: w.writeheader(); hdr = False
            for r_ in rows: w.writerow(r_)
        print("[%d/%d] %s: %d (%.0fs)" % (i+1, len(syms), s, len(rows), time.time()-t0), flush=True)
    print("DONE ->", OUT, flush=True)

if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING)
    main()
