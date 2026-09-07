"""research/156 — shared engine: data, synthetic industry baskets, signals, book simulator.

Nothing here writes to market_data.db. Read-only on the DB; all outputs go to results/.

Conventions (locked in the STATUS doc before any run):
  * signal computed on the close of day d, traded at the close of day d+1 (1-day lag)
  * 25 bps per side on traded notional
  * after-tax: 20% STCG / 12.5% LTCG (>365 calendar days), Indian FY netting settled on the
    first trading day on/after 1 April
  * idle cash compounds at 5% p.a.
  * drawdowns are always measured from the running peak of the FULL curve
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path("/home/arun/quantifyd")
STUDY = ROOT / "research" / "156_sector_rotation"
RES = STUDY / "results"
DB = ROOT / "backtest_data" / "market_data.db"
RES.mkdir(parents=True, exist_ok=True)

END = "2026-08-29"          # last full week; avoids a live partial candle
SECT_START = "2015-01-01"
IND_START = "2007-01-01"

SECT9 = ["NIFTYAUTO", "NIFTYIT", "NIFTYENERGY", "NIFTYFINSRV", "NIFTYFMCG",
         "NIFTYMETAL", "NIFTYPHARMA", "NIFTYPSUBANK", "NIFTYREALTY"]
BENCHES = ["NIFTY500", "NIFTY50", "NIFTYMIDCAP150", "NIFTYSMLCAP250", "NIFTYBEES"]

COST = 0.0025               # per side
CASH_YIELD = 0.05
STCG, LTCG = 0.20, 0.125
TV_FLOOR = 5e7              # Rs 5 cr 20d median traded value (stock universe)
MIN_CONSTITUENTS = 5


# ----------------------------------------------------------------------------- data
def _con():
    return sqlite3.connect(str(DB))


def load_close(symbols, start, end=END):
    """Close matrix, NaN-robust: each series is built on its own dates then aligned."""
    con = _con()
    q = ("SELECT symbol, date, close, volume FROM market_data_unified "
         "WHERE timeframe='day' AND date>=? AND date<=? AND symbol IN (%s)"
         % ",".join("?" * len(symbols)))
    df = pd.read_sql(q, con, params=[start, end] + list(symbols))
    con.close()
    df["date"] = pd.to_datetime(df.date)
    close = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    vol = df.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last")
    return close.sort_index(), vol.reindex_like(close)


def trading_calendar(start, end=END):
    con = _con()
    d = pd.read_sql("SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
                    "AND symbol='NIFTYBEES' AND date>=? AND date<=? ORDER BY date",
                    con, params=[start, end]).date
    con.close()
    return pd.DatetimeIndex(pd.to_datetime(d))


def industry_map():
    frames = []
    for f in ["nifty200_official.csv", "niftymidcap150_official.csv",
              "niftysmallcap250_official.csv"]:
        frames.append(pd.read_csv(ROOT / "backtest_data" / f))
    m = pd.concat(frames).drop_duplicates(subset=["Symbol"])
    m = m[m.Industry.notna() & (m.Industry != "Industry")]
    return m.set_index("Symbol").Industry.to_dict()


def load_stock_panel(start=IND_START, end=END):
    """Close + 20d-median traded value for the 500 industry-mapped names."""
    imap = industry_map()
    syms = sorted(imap)
    close, vol = load_close(syms, start, end)
    cal = trading_calendar(start, end)
    close = close.reindex(cal)
    vol = vol.reindex(cal)
    tv = (close * vol).rolling(20, min_periods=10).median()
    return close, tv, imap


def build_industry_baskets(close, tv, imap, min_n=MIN_CONSTITUENTS, tv_floor=0.0):
    """Equal-weight, daily-rebalanced basket NAV per industry.

    A stock contributes on day t only if it has a price on t and t-1 (and clears tv_floor
    on t-1 when a floor is given). Industries are formed only on days with >= min_n names.
    Returns (nav DataFrame, count DataFrame).
    """
    ret = close.pct_change()
    ok = close.notna() & close.shift(1).notna()
    if tv_floor > 0:
        ok &= (tv.shift(1) >= tv_floor)
    # guard against split-scale artefacts: a single-day move beyond these bounds is dropped
    ok &= ret.abs() < 0.40
    navs, cnts = {}, {}
    for ind in sorted(set(imap.values())):
        cols = [c for c in close.columns if imap.get(c) == ind]
        if not cols:
            continue
        r = ret[cols].where(ok[cols])
        n = ok[cols].sum(axis=1)
        avg = r.mean(axis=1, skipna=True).where(n >= min_n)
        if avg.notna().sum() < 250:
            continue
        first = avg.first_valid_index()
        seg = avg.loc[first:].fillna(0.0)
        navs[ind] = (1 + seg).cumprod().reindex(close.index)
        cnts[ind] = n
    return pd.DataFrame(navs), pd.DataFrame(cnts)


def breadth(close, imap, win):
    """Share of an industry's constituents above their own SMA(win)."""
    sma = close.rolling(win, min_periods=win // 2).mean()
    above = (close > sma)
    valid = close.notna() & sma.notna()
    out = {}
    for ind in sorted(set(imap.values())):
        cols = [c for c in close.columns if imap.get(c) == ind]
        if not cols:
            continue
        v = valid[cols].sum(axis=1)
        out[ind] = (above[cols].where(valid[cols]).sum(axis=1) / v.replace(0, np.nan))
    return pd.DataFrame(out)


# -------------------------------------------------------------------------- signals
def signal_specs(with_breadth=False):
    """name -> (kind, params). Kept declarative so p1/p2 share one definition."""
    s = []
    for L in (21, 42, 63, 126, 189, 252):
        s.append((f"ABSMOM{L}", ("mom", L, 0)))
    for L in (126, 189, 252):
        s.append((f"MOMSKIP{L}", ("mom", L, 21)))
    for L in (63, 126, 252):
        s.append((f"RISKADJ{L}", ("riskadj", L, 0)))
    s.append(("TSVOTE", ("tsvote", 0, 0)))
    for L in (126, 252):
        s.append((f"DISTHIGH{L}", ("disthigh", L, 0)))
    for L in (50, 100, 200):
        s.append((f"MADIST{L}", ("madist", L, 0)))
    s.append(("ACCEL", ("accel", 0, 0)))
    for L in (21, 63, 252):
        s.append((f"REVERSAL{L}", ("reversal", L, 0)))
    for L in (126, 252):
        s.append((f"VOLSCMOM{L}", ("volscmom", L, 0)))
    for L in (63, 252):
        s.append((f"LOWVOL{L}", ("lowvol", L, 0)))
    if with_breadth:
        s.append(("BREADTH50", ("breadth", 50, 0)))
        s.append(("BREADTH200", ("breadth", 200, 0)))
        s.append(("BREADTHCHG50", ("breadthchg", 50, 21)))
    return dict(s)


def compute_signal(px, kind, L, skip, brd=None, brd200=None):
    """px: DataFrame of asset NAV/close. Returns a same-shaped score DataFrame."""
    r = px.pct_change()
    if kind == "mom":
        return px.shift(skip) / px.shift(L) - 1
    if kind == "riskadj":
        return (px / px.shift(L) - 1) / (r.rolling(L, min_periods=L // 2).std() * np.sqrt(252))
    if kind == "tsvote":
        return sum((px / px.shift(k) - 1 > 0).astype(float) for k in (63, 126, 252))
    if kind == "disthigh":
        return px / px.rolling(L, min_periods=L // 2).max() - 1
    if kind == "madist":
        return px / px.rolling(L, min_periods=L // 2).mean() - 1
    if kind == "accel":
        return (px / px.shift(63) - 1) - (px / px.shift(252) - 1) * (63.0 / 252.0)
    if kind == "reversal":
        return -(px / px.shift(L) - 1)
    if kind == "volscmom":
        return (px / px.shift(L) - 1) / r.rolling(21, min_periods=10).std()
    if kind == "lowvol":
        return -r.rolling(L, min_periods=L // 2).std()
    if kind == "breadth":
        return (brd if L == 50 else brd200)
    if kind == "breadthchg":
        return brd - brd.shift(skip)
    raise ValueError(kind)


# ------------------------------------------------------------------- rebalance dates
def rebal_dates(index, clock, offset):
    """clock in {'M','Q','F'}; offset shifts the rebalance forward by n trading days."""
    idx = pd.DatetimeIndex(index)
    if clock == "F":                              # fortnightly ~ every 10 trading days
        base = idx[::10]
    else:
        per = idx.to_period("M" if clock == "M" else "Q")
        base = idx.to_series().groupby(per).last().values
        base = pd.DatetimeIndex(base)
    pos = idx.get_indexer(base)
    pos = np.clip(pos + offset, 0, len(idx) - 1)
    return idx[np.unique(pos)]


# ------------------------------------------------------------------------- simulator
class Book:
    """Weight-target book with FIFO lots, per-side costs, FY-netted Indian tax, cash yield."""

    def __init__(self, dates, prices, cost=COST, cash_yield=CASH_YIELD, tax=True,
                 start_nav=1_000_000.0):
        self.dates = pd.DatetimeIndex(dates)
        self.px = prices.reindex(self.dates)
        self.cost, self.cy, self.tax = cost, cash_yield, tax
        self.nav0 = start_nav

    def run(self, targets):
        """targets: dict {date -> {asset: weight}}. Trades execute at that date's close."""
        dates, px = self.dates, self.px
        cols = list(px.columns)
        cidx = {c: i for i, c in enumerate(cols)}
        P = px.values
        n = len(dates)
        qty = np.zeros(len(cols))
        lots = {c: [] for c in cols}            # (qty, price, date)
        cash = self.nav0
        daily_cash = (1 + self.cy) ** (1 / 252.0) - 1
        nav = np.empty(n)
        turnover = 0.0
        turn_frac = 0.0
        exposure = np.zeros(n)
        stcg_r = ltcg_r = 0.0                    # realised in the running FY
        fy = self._fy(dates[0])
        tgt_dates = set(pd.DatetimeIndex(list(targets)))

        for t in range(n):
            d = dates[t]
            cash *= (1 + daily_cash)
            # FY boundary: settle tax on the first trading day on/after 1 April
            f = self._fy(d)
            if f != fy:
                if self.tax:
                    net_s, net_l = stcg_r, ltcg_r
                    if net_s < 0:                # STCL offsets STCG then LTCG
                        net_l += net_s
                        net_s = 0.0
                    if net_l < 0:
                        net_l = 0.0
                    cash -= max(net_s, 0.0) * STCG + max(net_l, 0.0) * LTCG
                stcg_r = ltcg_r = 0.0
                fy = f
            prices = P[t]
            if d in tgt_dates:
                w = targets[d]
                valid = {a: v for a, v in w.items()
                         if a in cidx and np.isfinite(prices[cidx[a]])}
                held = cash + float(np.nansum(np.where(np.isfinite(prices), qty * prices, 0.0)))
                tot = sum(valid.values())
                if tot > 0:
                    valid = {a: v / tot * min(tot, 1.0) for a, v in valid.items()}
                tq = np.zeros(len(cols))
                for a, v in valid.items():
                    i = cidx[a]
                    tq[i] = held * v / prices[i]
                for i, c in enumerate(cols):
                    if not np.isfinite(prices[i]):
                        continue
                    dq = tq[i] - qty[i]
                    if abs(dq) * prices[i] < held * 1e-4:
                        tq[i] = qty[i]
                        continue
                    notional = abs(dq) * prices[i]
                    turnover += notional
                    turn_frac += notional / max(held, 1e-9)
                    fee = notional * self.cost
                    if dq > 0:
                        cash -= notional + fee
                        lots[c].append([dq, prices[i], d])
                    else:
                        cash += notional - fee
                        sell = -dq
                        while sell > 1e-12 and lots[c]:
                            lq, lp, ld = lots[c][0]
                            take = min(lq, sell)
                            gain = (prices[i] - lp) * take
                            if (d - ld).days > 365:
                                ltcg_r += gain
                            else:
                                stcg_r += gain
                            lq -= take
                            sell -= take
                            if lq <= 1e-12:
                                lots[c].pop(0)
                            else:
                                lots[c][0][0] = lq
                qty = tq
            mv = float(np.nansum(np.where(np.isfinite(prices), qty * prices, 0.0)))
            nav[t] = cash + mv
            exposure[t] = mv / nav[t] if nav[t] > 0 else 0.0
        s = pd.Series(nav, index=dates)
        yrs = (dates[-1] - dates[0]).days / 365.25
        return dict(nav=s, turnover_yr=turn_frac / max(yrs, 1e-9),
                    turnover_init=turnover / self.nav0 / max(yrs, 1e-9),
                    exposure=float(np.mean(exposure)))

    @staticmethod
    def _fy(d):
        return d.year if d.month >= 4 else d.year - 1


# --------------------------------------------------------------------------- metrics
def metrics(nav, full_peak=None):
    nav = nav.dropna()
    if len(nav) < 50:
        return dict(cagr=np.nan, maxdd=np.nan, calmar=np.nan, sharpe=np.nan, vol=np.nan)
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    peak = nav.cummax() if full_peak is None else full_peak.reindex(nav.index).ffill()
    dd = (nav / peak - 1).min()
    r = nav.pct_change().dropna()
    vol = r.std() * np.sqrt(252)
    return dict(cagr=cagr * 100, maxdd=dd * 100,
                calmar=(cagr / abs(dd)) if dd < 0 else np.nan,
                sharpe=(r.mean() * 252 - 0.05) / vol if vol > 0 else np.nan,
                vol=vol * 100)


def yearly(nav):
    """Per-calendar-year return and intra-year drawdown measured from the FULL curve peak."""
    nav = nav.dropna()
    peak = nav.cummax()
    out = {}
    for y, seg in nav.groupby(nav.index.year):
        prev = nav.loc[:seg.index[0]]
        base = prev.iloc[-2] if len(prev) > 1 else seg.iloc[0]
        ret = seg.iloc[-1] / base - 1
        dd = (seg / peak.reindex(seg.index) - 1).min()
        out[y] = (ret * 100, dd * 100)
    return out
