"""Pure Relative-Strength concentrated top-10 sweep.

Survivorship-free: monthly we reconstruct a point-in-time universe by
trailing-6mo median traded value (price*volume) using only data up to that
date, split into mid (rank 101-250) / small (251-500) / combo (101-500).
Within the chosen universe, rank by RS vs NIFTY50 over lookback L, hold the
top-10 equal-weight, rotate monthly with a top-15 retention buffer. Idle
cash (if ever) earns 6.5% p.a. Costs ~0.4% round-trip on turnover.

Grid: 3 universes x 5 lookbacks = 15 configs. Window 2014->2026 (includes
2018-19 small-cap bear, Mar-2020, 2022). Outputs ranking + per-year table.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "backtest_data" / "market_data.db"
HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)

START = "2014-01-01"
LARGE_EXCL = 100
BANDS = {"mid": (100, 250), "small": (250, 500), "combo": (100, 500)}
LOOKBACKS = {"55d": 55, "120d": 120, "126d_6m": 126, "252d_1y": 252, "blend_6m12m": None}
SIZES = [10, 15, 20, 25, 30]   # portfolio-size sweep axis
BUFFER_MULT = 1.5              # retention buffer = N * 1.5 (cuts churn)
RT_COST = 0.004             # ~0.4% round-trip (brokerage+STT+impact, small-cap)
LIQ_LOOKBACK_TD = 126
CASH_ANNUAL = 0.065         # idle cash -> debt 6.5%
# Benchmark for RS. NIFTY50 daily in market_data.db only spans 2023-03..2026-03
# (740 bars) -> using it silently parked the strategy in cash for 2014-2022.
# NIFTYBEES (Nifty-50 ETF) has full daily history 2005-01..2026-02; RS is a
# price *ratio* so the ETF scale cancels -> faithful Nifty-50 proxy 2014->2026.
NIFTY_SYM = "NIFTYBEES"


def load():
    con = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT symbol,date,close,volume FROM market_data_unified "
        "WHERE timeframe='day' AND close IS NOT NULL AND date>='2012-06-01' "
        "ORDER BY symbol,date", con, parse_dates=["date"])
    con.close()
    df["tv"] = df["close"] * df["volume"].fillna(0)
    close = df.pivot_table(index="date", columns="symbol", values="close")
    tv = df.pivot_table(index="date", columns="symbol", values="tv")
    close = close.sort_index(); tv = tv.sort_index()
    return close, tv


def month_ends(idx):
    s = pd.Series(idx, index=idx)
    me = s.groupby([idx.year, idx.month]).max().values
    me = pd.DatetimeIndex(sorted(me))
    return me[(me >= pd.Timestamp(START))]


def rs_scores(close, asof, L):
    h = close.loc[:asof]
    if NIFTY_SYM not in h.columns:
        return None
    if L is None:  # blended 6m+12m
        out = {}
        for LL, w in ((126, 0.5), (252, 0.5)):
            if len(h) <= LL: return None
            p0 = h.iloc[-LL-1]; p1 = h.iloc[-1]
            nf = (p1[NIFTY_SYM] / p0[NIFTY_SYM])
            r = (p1 / p0) / nf
            for s, v in r.items():
                if pd.notna(v): out[s] = out.get(s, 0) + w * v
        return pd.Series(out)
    if len(h) <= L: return None
    p0 = h.iloc[-L-1]; p1 = h.iloc[-1]
    nf = (p1[NIFTY_SYM] / p0[NIFTY_SYM])
    return ((p1 / p0) / nf).dropna()


def pit_universe(tv, close, asof, band):
    lo, hi = BANDS[band]
    w = tv.loc[:asof].tail(int(LIQ_LOOKBACK_TD * 1.6))
    cnt = w.notna().sum()
    medtv = w.median()
    elig = medtv[(cnt >= 75) & (medtv > 0)].sort_values(ascending=False)
    ranked = [s for s in elig.index if s != NIFTY_SYM]  # benchmark not investable
    return set(ranked[lo:hi])  # 0-based: lo=100 drops top-100


def backtest(close, tv, band, Lkey, topn, exclude=None):
    """Returns (nav_df, contrib) where contrib[symbol] = summed monthly
    book-return contribution across the whole run (for super-winner audit).
    `exclude` = set of symbols the strategy is NOT allowed to ever hold
    (used for leave-winners-out robustness)."""
    L = LOOKBACKS[Lkey]
    exclude = exclude or set()
    buffer = int(topn * BUFFER_MULT)
    me = month_ends(close.index)
    cash = 1.0; equity = 0.0
    held = {}
    nav = []
    contrib = {}
    daily = close
    last_px = {}
    for dt in me:
        px = daily.loc[:dt].iloc[-1]
        if held:
            tot_prev = sum(held.values())
            new_equity = 0.0
            for s, amt in held.items():
                p1 = px.get(s, np.nan)
                if pd.notna(p1) and s in last_px and last_px[s] > 0:
                    r = p1 / last_px[s] - 1
                    new_equity += amt * (1 + r)
                    contrib[s] = contrib.get(s, 0.0) + amt * r  # rupee-ish attribution
                else:
                    new_equity += amt  # carried flat if price gap
            equity = new_equity
        total = cash * (1 + CASH_ANNUAL) ** (1/12) + equity
        uni = pit_universe(tv, close, dt, band)
        sc = rs_scores(close, dt, L)
        if sc is None or not uni:
            nav.append((dt, total)); continue
        cand = [s for s in sc.index
                if s in uni and s != NIFTY_SYM and s not in exclude]
        sc = sc[cand].sort_values(ascending=False)
        ranked = list(sc.index)
        top = ranked[:topn]
        buf = set(ranked[:buffer])
        keep = [s for s in held if s in buf]
        fill = [s for s in top if s not in keep]
        target = (keep + fill)[:topn]
        prev = set(held); new = set(target)
        turnover = len(prev.symmetric_difference(new)) / max(1, 2 * topn)
        total *= (1 - RT_COST * turnover * 2)
        target = [s for s in target if pd.notna(px.get(s, np.nan))]
        if not target:
            cash = total; equity = 0.0; held = {}
            nav.append((dt, total)); continue
        w = total / len(target)
        held = {s: w for s in target}
        last_px = {s: px[s] for s in target}
        cash = 0.0; equity = total
        nav.append((dt, total))
    nav_df = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")
    return nav_df, contrib


def metrics(nav):
    n = nav["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cagr = (n.iloc[-1] / n.iloc[0]) ** (1/yrs) - 1
    mret = n.pct_change().dropna()
    sharpe = (mret.mean() / mret.std() * np.sqrt(12)) if mret.std() > 0 else 0
    roll_max = n.cummax()
    mdd = ((n - roll_max) / roll_max).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    per_yr = n.groupby(n.index.year).last().pct_change().dropna()
    return dict(cagr=cagr, sharpe=sharpe, mdd=mdd, calmar=calmar,
                yrs=round(yrs, 1), per_yr=per_yr)


def cagr_of(nav):
    n = nav["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    return (n.iloc[-1] / n.iloc[0]) ** (1/yrs) - 1


def main():
    print("Loading daily price+tv ...")
    close, tv = load()
    print(f"  {close.shape[1]} symbols, {close.index.min().date()}..{close.index.max().date()}")
    rows = []; peryr = {}; cache = {}
    for band in BANDS:
        for Lk in LOOKBACKS:
            for N in SIZES:
                nav, contrib = backtest(close, tv, band, Lk, N)
                if len(nav) < 12:
                    continue
                m = metrics(nav)
                label = f"{band}_{Lk}_N{N}"
                # super-winner concentration
                cs = pd.Series(contrib).sort_values(ascending=False)
                tot = cs[cs > 0].sum() or 1e-9
                top1 = cs.iloc[0] / tot * 100 if len(cs) else 0
                top3 = cs.iloc[:3].sum() / tot * 100 if len(cs) else 0
                rows.append(dict(config=label, universe=band, lookback=Lk, N=N,
                                 cagr=round(m['cagr']*100, 1), sharpe=round(m['sharpe'], 2),
                                 maxdd=round(m['mdd']*100, 1), calmar=round(m['calmar'], 2),
                                 top1_share=round(top1, 1), top3_share=round(top3, 1),
                                 years=m['yrs']))
                peryr[label] = m['per_yr']
                cache[label] = (band, Lk, N, cs)
                print(f"  {label:26} CAGR={m['cagr']*100:6.1f}%  Sh={m['sharpe']:.2f}  "
                      f"DD={m['mdd']*100:6.1f}%  Cal={m['calmar']:.2f}  "
                      f"top1={top1:4.1f}% top3={top3:4.1f}%")
    r = pd.DataFrame(rows).sort_values("cagr", ascending=False)

    # ---- Super-winner robustness: leave-top-1 / leave-top-3 OUT, re-run ----
    print("\n=== ROBUSTNESS: re-run top-12 configs forbidding their best names ===")
    rob = []
    for label in r.head(12)["config"]:
        band, Lk, N, cs = cache[label]
        winners = list(cs.head(3).index)
        nav1, _ = backtest(close, tv, band, Lk, N, exclude=set(winners[:1]))
        nav3, _ = backtest(close, tv, band, Lk, N, exclude=set(winners[:3]))
        base = r.loc[r.config == label, "cagr"].iloc[0]
        c1 = round(cagr_of(nav1)*100, 1)
        c3 = round(cagr_of(nav3)*100, 1)
        fragile = c3 < 20
        rob.append(dict(config=label, cagr=base, cagr_ex_top1=c1,
                        cagr_ex_top3=c3, drop3_delta=round(base - c3, 1),
                        robust_vs_20=("NO-fragile" if fragile else "yes")))
        print(f"  {label:26} base={base:5.1f}%  ex-top1={c1:5.1f}%  "
              f"ex-top3={c3:5.1f}%  -> {'FRAGILE' if fragile else 'robust'}")
    robdf = pd.DataFrame(rob)

    r.to_csv(RES / "rs_sweep_ranking.csv", index=False)
    robdf.to_csv(RES / "rs_sweep_robustness.csv", index=False)
    top5 = r.head(5)["config"].tolist()
    yt = pd.DataFrame({c: (peryr[c]*100).round(1) for c in top5})
    yt.to_csv(RES / "rs_sweep_top5_peryear.csv")

    print("\n=== RANKING by CAGR (top 20) ===")
    print(r.head(20).to_string(index=False))
    print("\n=== TOP-5 PER-YEAR RETURNS % (bear years exposed) ===")
    print(yt.to_string())
    # Honest verdict: prefer robust configs (edge survives losing best 3 names)
    robust = robdf[robdf.robust_vs_20 == "yes"].merge(
        r[["config", "sharpe", "maxdd", "N", "universe", "lookback"]], on="config")
    print("\n=== ROBUST configs that still beat 20% WITHOUT their top-3 names ===")
    if len(robust):
        print(robust.sort_values("cagr_ex_top3", ascending=False).to_string(index=False))
    else:
        print("  NONE — every high-CAGR config collapses below 20% once its "
              "best 3 names are removed. That is the false-indication signal: "
              "the apparent edge is 1-3 multibaggers, not breadth.")
    print(f"\nHurdle: ~20% (MidSmallcap400-MQ100). Raw configs >20%: "
          f"{(r.cagr>20).sum()}/{len(r)} | Robust (ex-top3 >20%): {len(robust)}")


if __name__ == "__main__":
    main()
