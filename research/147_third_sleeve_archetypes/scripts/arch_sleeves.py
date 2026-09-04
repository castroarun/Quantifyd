"""research/147 — third-sleeve archetypes: build candidate NAVs (gold / nasdaq / gilt /
GTAA mixes / index trend L-S futures / sector rotation), G1 standalone + tradeability where
applicable, then 3-sleeve blends vs the SAME-WINDOW TN+OA baseline using r/146's cached
after-tax OA-seed and TN-offset NAVs. Candidate sleeves net-of-cost, gross-of-tax
(pre-registered asymmetry: kills are bias-safe, any churny pass must be re-run with tax).
"""
from __future__ import annotations
import sqlite3, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
R146 = Path("/home/arun/quantifyd/research/146_complementary_third_sleeve/results")
DB = Path("/home/arun/quantifyd/backtest_data/market_data.db")

CASH_Y = 0.05
DAY_CASH = (1 + CASH_Y) ** (1 / 252)
W3 = [0.10, 0.15, 0.20, 0.25, 0.33]
OFFSETS = [0, 4, 8]
WIN = {"2015-16": ("2015-08-01", "2016-02-29"), "2018": ("2018-01-01", "2018-10-31"),
       "2020crash": ("2020-02-01", "2020-04-30"), "2022H1": ("2022-01-01", "2022-06-30")}
SECTORS = ["NIFTYIT", "NIFTYPHARMA", "NIFTYFMCG", "NIFTYAUTO", "NIFTYMETAL",
           "NIFTYREALTY", "NIFTYPSUBANK", "NIFTYENERGY", "NIFTYFINSRV"]


def series(sym):
    con = sqlite3.connect(str(DB))
    df = pd.read_sql("SELECT date, close FROM market_data_unified WHERE symbol=? AND "
                     "timeframe='day' AND close IS NOT NULL ORDER BY date", con,
                     params=(sym,), parse_dates=["date"])
    con.close()
    return df.set_index("date")["close"]


def month_ends(idx):
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def stats(nav, a=None, b=None):
    n = nav
    if a: n = n[n.index >= pd.Timestamp(a)]
    if b: n = n[n.index <= pd.Timestamp(b)]
    if len(n) < 60:
        return ("", "", "", "")
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cagr = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd = float((n / n.cummax() - 1).min())
    return (round(cagr * 100, 2), round(dd * 100, 2), round(sh, 2),
            round(cagr / abs(dd), 2) if dd < 0 else "")


def win_ret(nav, a, b):
    s = nav[(nav.index >= a) & (nav.index <= b)]
    return round(float(s.iloc[-1] / s.iloc[0] - 1) * 100, 1) if len(s) > 2 else ""


# ── sleeve builders (daily NAV, net of costs, gross of tax) ──
def bh(px):
    px = px.dropna()
    return px / px.iloc[0]


def tf(px, sma_d=210, cost=0.001):
    px = px.dropna()
    sig = (px > px.rolling(sma_d, min_periods=sma_d).mean())
    me = month_ends(px.index)
    pos = sig.reindex(me).astype(float).reindex(px.index).ffill().shift(1).fillna(0)
    r = px.pct_change().fillna(0)
    ret = pos * r + (1 - pos) * (DAY_CASH - 1)
    switch = pos.diff().abs().fillna(0)
    ret -= switch * cost                      # cost per full switch (sell+buy ~ 2x half)
    return (1 + ret).cumprod()


def mix(navs, cost_yr=0.0005):
    """Equal-weight mix of daily NAV series on the common DAILY grid (EW maintained;
    v1 built a calendar-month-end nav whose intersection with the daily legs silently
    zeroed ~half the months — fixed 2026-09-04, see STATUS log)."""
    idx = navs[0].index
    for n in navs[1:]:
        idx = idx.intersection(n.index)
    rets = [n.loc[idx].pct_change().fillna(0) for n in navs]
    r = sum(rets) / len(navs) - cost_yr / 252
    out = (1 + r).cumprod()
    out.index.name = "date"
    return out


def trend_ls(px, short_only=False, cost=0.0005):
    px = px.dropna()
    me = month_ends(px.index)
    mom = sum((px / px.shift(L) - 1 > 0).astype(int) for L in (63, 126, 252))
    sig_m = mom.reindex(me)
    pos_m = np.where(sig_m >= 2, 0.0 if short_only else 1.0, -1.0)
    pos = pd.Series(pos_m, index=me).reindex(px.index).ffill().shift(1).fillna(0)
    r = px.pct_change().fillna(0)
    # long via futures ~ index return; short = -index + cash on collateral; flat = cash
    ret = np.where(pos > 0, r, np.where(pos < 0, -r + (DAY_CASH - 1), DAY_CASH - 1))
    ret = pd.Series(ret, index=px.index)
    ret -= pos.diff().abs().fillna(0) * cost * 2
    return (1 + ret).cumprod()


def sector_rot(cost=0.001):
    cl = {}
    for s in SECTORS:
        cl[s] = series(s)
    df = pd.DataFrame(cl).dropna(how="all").ffill()
    me = month_ends(df.index)
    mom = df / df.shift(126) - 1
    r = df.pct_change().fillna(0)
    hold = {}
    cur = []
    ret = pd.Series(0.0, index=df.index)
    me_set = set(me)
    for d in df.index:
        if cur:
            ret[d] = r.loc[d, cur].mean()
        else:
            ret[d] = DAY_CASH - 1
        if d in me_set:
            row = mom.loc[d].dropna()
            new = list(row.sort_values(ascending=False).index[:2]) if len(row) >= 4 else []
            turn = len(set(new) ^ set(cur)) / 2
            ret[d] -= turn * cost * 2
            cur = new
    return (1 + ret).cumprod()


def main():
    t0 = time.time()
    nb = series("NIFTYBEES"); gold = series("GOLDBEES"); nas = series("MON100")
    gilt = series("LTGILTBEES")
    cands = {
        "GOLD": bh(gold), "NAS": bh(nas), "GILT": bh(gilt),
        "GOLD_TF": tf(gold), "NAS_TF": tf(nas),
        "GN5050": mix([bh(gold), bh(nas)]),
        "GNG3": mix([bh(gold), bh(nas), bh(gilt)]),
        "GTAA_EW": mix([bh(nb), bh(gold), bh(nas)]),
        "GTAA_TF": mix([tf(nb), tf(gold), tf(nas)]),
        "TLS": trend_ls(nb), "TSHORT": trend_ls(nb, short_only=True),
        "SECROT": sector_rot(),
    }
    # cash null on the 2015+ window
    base_idx = cands["GOLD"].index
    cands["CASHNULL"] = pd.Series(DAY_CASH ** np.arange(len(base_idx)),
                                  index=base_idx)

    oa = pd.read_csv(R146 / "oa_navs.csv", index_col=0, parse_dates=True)
    oa_navs = [oa[c] for c in oa.columns]
    tn = {o: pd.read_csv(R146 / f"tn_nav_off{o}.csv", index_col=0,
                         parse_dates=True).iloc[:, 0] for o in OFFSETS}

    g1_rows, blend_rows, win_rows = [], [], []
    for name, nav in cands.items():
        nav = nav.dropna()
        nav.to_csv(RES / f"nav_{name}.csv")
        c, d, sh, cal = stats(nav)
        idx = nav.index.intersection(tn[0].index)
        corr_tn = round(float(nav.loc[idx].pct_change().corr(
            tn[0].loc[idx].pct_change())), 3)
        corr_oa = round(float(np.median([
            nav.loc[nav.index.intersection(o.index)].pct_change().corr(
                o.loc[nav.index.intersection(o.index)].pct_change())
            for o in oa_navs])), 3)
        row = dict(cand=name, start=str(nav.index[0].date()), cagr=c, dd=d, sharpe=sh,
                   calmar=cal, corr_tn=corr_tn, corr_oa=corr_oa)
        for wn, (a, b) in WIN.items():
            row[f"ret_{wn}"] = win_ret(nav, a, b)
        g1_rows.append(row)
        print(row, flush=True)

        # blends on THIS candidate's common window; baseline recomputed same-window
        def blend(o_nav, t_nav, w3):
            ix = o_nav.index.intersection(t_nav.index).intersection(nav.index)
            m = [x.loc[ix].resample("ME").last().pct_change().fillna(0)
                 for x in (o_nav, t_nav, nav)]
            wl = (1 - w3) / 2
            return (1 + wl * m[0] + wl * m[1] + w3 * m[2]).cumprod()

        for off in OFFSETS:
            for w3 in [0.0] + W3:
                cs, ds, ks = [], [], []
                for o_nav in oa_navs:
                    b = blend(o_nav, tn[off], w3)
                    cc, dd_, _, kk = stats(b)
                    cs.append(cc); ds.append(dd_); ks.append(kk if kk != "" else np.nan)
                blend_rows.append(dict(cand=name, offset=off, w3=w3,
                                       cagr_med=round(float(np.median(cs)), 2),
                                       cagr_min=round(min(cs), 2),
                                       dd_med=round(float(np.median(ds)), 2),
                                       dd_worst=round(min(ds), 2),
                                       calmar_med=round(float(np.nanmedian(ks)), 2)))
        # windows: baseline vs w3=0.25 blend DD (offset 0, seed-median)
        for wn, (a, b_) in WIN.items():
            def wdd(w3):
                vals = []
                for o_nav in oa_navs:
                    bl = blend(o_nav, tn[0], w3)
                    s = bl[(bl.index >= a) & (bl.index <= b_)]
                    if len(s) > 2:
                        vals.append(float((s / s.cummax() - 1).min() * 100))
                return round(float(np.median(vals)), 2) if vals else ""
            win_rows.append(dict(cand=name, window=wn, sleeve_ret=row[f"ret_{wn}"],
                                 base_dd=wdd(0.0), w25_dd=wdd(0.25)))
    pd.DataFrame(g1_rows).to_csv(RES / "g1_archetypes.csv", index=False)
    pd.DataFrame(blend_rows).to_csv(RES / "blend_arch.csv", index=False)
    pd.DataFrame(win_rows).to_csv(RES / "windows_arch.csv", index=False)
    print(f"DONE [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
