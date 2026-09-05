"""research/153 — IPO-Base breakout engine.

EXTENDS research/142's decoded bananapatterns engine (`bluesky_replay.py`), reusing its
validated common mechanics:
  * RS = IBD-weighted percentile (2*r63 + r126 + r189 + r252) over eligibles, shifted 1
  * liquidity floor: 20-day median traded value at t-1 >= Rs 5cr; ETFs excluded
  * entry = buy-stop AT the pivot, filled max(pivot, open)
  * exits = hard stop on the CLOSE + moving-average trail on the CLOSE
  * cash-constrained book, random-selection seed ensemble for path dependence

WHAT IS NEW HERE
  * the pivot is the IPO BASE HIGH (rolling max close over the last L bars), not the ATH
  * a vetted LISTING DATE table (results/listing_dates.csv, Phase 0) gates the universe to
    genuinely newly-listed names and MASKS all pre-listing junk rows
  * base quality: max depth, optional tightness (ATR%)
  * RS POLICY arms, because a 6-month-old listing cannot have a 252-day RS
  * take-profit exit (their "+25%" dial), risk-based sizing capped at 30% of capital
  * FY (1 April) tax netting: 20% STCG / 12.5% LTCG, STCL -> STCG -> LTCG, loss carry-forward
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
ROOT = Path("/home/arun/quantifyd")
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "142_bananapatterns_replication" / "scripts"))
import bluesky_replay as br  # noqa: E402  (reuse ETF_RE / TV_FLOOR / CAPITAL / simulate)

DB = ROOT / "backtest_data" / "market_data.db"
CAPITAL = br.CAPITAL          # Rs 10,00,000
TV_FLOOR = br.TV_FLOOR        # Rs 5 cr


# ───────────────────────────────────────────────────────────── context / data load
class Ctx:
    """Loads once; every sweep cell reuses it."""

    def __init__(self, panel_start="2005-06-01", verbose=True):
        t0 = time.time()
        ld = pd.read_csv(RES / "listing_dates.csv")
        ld = ld[ld.accepted].copy()
        self.listing = dict(zip(ld.symbol, pd.to_datetime(ld.list_date)))
        if verbose:
            print(f"[ctx] {len(self.listing)} vetted listings", flush=True)

        con = sqlite3.connect(str(DB))
        syms = [r[0] for r in con.execute(
            "select symbol from (select symbol, count(*) n from market_data_unified "
            "where timeframe='day' group by symbol) where n >= 60")]
        cols = {}
        for i, s in enumerate(syms):
            df = pd.read_sql_query(
                "select date, open, high, low, close, volume from market_data_unified "
                "where symbol=? and timeframe='day' order by date", con, params=(s,))
            if not len(df):
                continue
            df["date"] = pd.to_datetime(df["date"].str[:10])
            df = df.drop_duplicates("date").set_index("date").sort_index()
            ldt = self.listing.get(s)
            if ldt is not None:            # mask pre-listing junk rows
                df = df[df.index >= ldt]
            m = df.index >= panel_start
            if not m.any():
                continue
            cols[s] = df.loc[m]
            if verbose and (i + 1) % 800 == 0:
                print(f"[ctx]   loaded {i+1}/{len(syms)} ({time.time()-t0:.0f}s)", flush=True)
        con.close()

        def wide(k):
            return pd.DataFrame({s: v[k] for s, v in cols.items()}).astype("float32")

        self.close, self.high, self.low, self.open = (wide(k) for k in
                                                      ("close", "high", "low", "open"))
        vol = wide("volume")
        self.dates = self.close.index
        if verbose:
            print(f"[ctx] wide panel {self.close.shape} ({time.time()-t0:.0f}s)", flush=True)

        # ── eligibility (identical to r/142) ──
        tv20 = (self.close * vol).rolling(20).median()
        self.tv_prev = tv20.shift(1)
        etf = [c for c in self.close.columns if br.ETF_RE.search(c)]
        elig = self.tv_prev >= TV_FLOOR
        elig[etf] = False
        self.elig = elig

        # ── RS variants ──
        r63 = self.close / self.close.shift(63) - 1
        r126 = self.close / self.close.shift(126) - 1
        r189 = self.close / self.close.shift(189) - 1
        r252 = self.close / self.close.shift(252) - 1
        full = (2 * r63 + r126 + r189 + r252).where(elig)
        self.rs_full = (full.rank(axis=1, pct=True) * 100).shift(1)          # needs 252d
        self.rs_short = (r63.where(elig).rank(axis=1, pct=True) * 100).shift(1)  # needs 63d

        # ── bars since listing, calendar age in days ──
        notna = self.close.notna()
        self.bars = notna.cumsum().where(notna).astype("float32")
        # NOTE: DatetimeIndex.asi8 is resolution-dependent in pandas 2.x (this panel
        # comes back as datetime64[us]), so never divide asi8 by a ns constant.
        EPOCH = pd.Timestamp("1970-01-01")
        dnum = pd.Series((self.dates - EPOCH).days.values.astype("float64"),
                         index=self.dates)
        lst = pd.Series([float((self.listing[c] - EPOCH).days) if c in self.listing
                         else np.nan for c in self.close.columns],
                        index=self.close.columns)
        self.age = pd.DataFrame(
            dnum.values.astype("float32").reshape(-1, 1) - lst.values.astype("float32"),
            index=self.dates, columns=self.close.columns)

        # ── young-listing tradeable column subset ──
        young_any = ((self.age > 0) & (self.age <= 36 * 30.44) & elig).any()
        self.cols = [c for c in self.close.columns
                     if c in self.listing and bool(young_any.get(c, False))]
        if verbose:
            print(f"[ctx] {len(self.cols)} tradeable young-listing symbols", flush=True)

        # ── numpy views on the traded subset ──
        sub = lambda df: df[self.cols].values  # noqa: E731
        self.C, self.H, self.L, self.O = (sub(x) for x in
                                          (self.close, self.high, self.low, self.open))
        self.AGE = sub(self.age)
        self.BARS = np.nan_to_num(sub(self.bars), nan=0.0)
        self.ELIG = sub(self.elig.astype("float32")) > 0.5
        self.TVp = sub(self.tv_prev)
        self.RSF = sub(self.rs_full)
        self.RSS = sub(self.rs_short)
        self.PREVC = sub(self.close.shift(1))
        self._pivot_cache: dict = {}
        self._sma_cache: dict = {}
        self._atr_cache: dict = {}

        # ── ATR% for the tightness dial ──
        hi, lo_, pc = (self.high[self.cols].values, self.low[self.cols].values,
                       self.close[self.cols].shift(1).values)
        self._tr = pd.DataFrame(
            np.nanmax(np.stack([hi - lo_, np.abs(hi - pc), np.abs(lo_ - pc)]), axis=0),
            index=self.dates, columns=self.cols).astype("float32")

        # ── weak-market gate series (NIFTYBEES < SMA200), NaN-robust ──
        nb = self.close.get("NIFTYBEES")
        nbs = nb.dropna()
        weak = (nbs < nbs.rolling(200).mean()).shift(1)
        self.WEAK = weak.reindex(self.dates).ffill().fillna(False).to_numpy(dtype=bool)
        self.NOWEAK = np.zeros(len(self.dates), dtype=bool)
        if verbose:
            print(f"[ctx] ready ({time.time()-t0:.0f}s)", flush=True)

    # ── cached rolling matrices ──
    def pivot(self, L, mode="close"):
        k = (L, mode)
        if k not in self._pivot_cache:
            src = self.close if mode == "close" else self.high
            piv = src[self.cols].rolling(L).max().shift(1)
            lo = self.low[self.cols].rolling(L).min().shift(1)
            self._pivot_cache[k] = (piv.values.astype("float32"),
                                    lo.values.astype("float32"))
        return self._pivot_cache[k]

    def sma(self, n):
        if n not in self._sma_cache:
            self._sma_cache[n] = self.close[self.cols].rolling(n).mean().values.astype("float32")
        return self._sma_cache[n]

    def atrp(self, L):
        if L not in self._atr_cache:
            a = self._tr.rolling(L).mean() / self.close[self.cols]
            self._atr_cache[L] = (100 * a).values.astype("float32")
        return self._atr_cache[L]


# ────────────────────────────────────────────────────────────── trigger construction
def build_trigger(ctx, max_age_m=12, min_bars=25, L=40, max_depth=0.35,
                  rs_policy="relaxed", rs_min=70.0, tight_max=None,
                  pivot_mode="close", max_base_hi=None):
    """Returns (TRIG bool[T,N], PIVOT float[T,N], BASELOW float[T,N])."""
    piv, lo = ctx.pivot(L, pivot_mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = (piv - lo) / np.where(piv > 0, piv, np.nan)
    young = (ctx.AGE > 0) & (ctx.AGE <= max_age_m * 30.44) & (ctx.BARS >= min_bars)
    base_ok = depth <= max_depth
    if max_base_hi is not None:   # base high not more than X above the listing-day close
        base_ok &= True           # reserved dial, unused by default
    not_ext = ctx.PREVC < piv
    if rs_policy == "off":
        rsok = np.ones_like(ctx.C, dtype=bool)
    elif rs_policy == "strict":
        rsok = np.nan_to_num(ctx.RSF, nan=-1.0) >= rs_min
    elif rs_policy == "relaxed":       # apply where computable, pass where not
        rsok = np.isnan(ctx.RSF) | (np.nan_to_num(ctx.RSF, nan=-1.0) >= rs_min)
    elif rs_policy == "short":         # 3-month RS percentile (available to young names)
        rsok = np.nan_to_num(ctx.RSS, nan=-1.0) >= rs_min
    else:
        raise ValueError(rs_policy)
    trig = young & base_ok & not_ext & ctx.ELIG & rsok & (ctx.C > piv) & ~np.isnan(piv)
    if tight_max is not None:
        trig &= np.nan_to_num(ctx.atrp(L), nan=1e9) <= tight_max
    return np.nan_to_num(trig, nan=False), piv, lo


# ─────────────────────────────────────────────────────────────────── the book
def simulate_ipo(seed, days_idx, dates, C, O, PIV, LO, SMA, RS, TVp, TRIG, weak,
                 *, cost=0.0025, stop=0.08, slots=8, size_pct=0.1875,
                 risk_pct=None, size_cap=0.30, stop_mode="pct",
                 target=None, fill_close=False, fill_realistic=True,
                 stcg=0.20, ltcg=0.125, cash_yield=0.05, fy_tax=True,
                 capital=CAPITAL, collect_trades=True):
    """r/142 `bluesky_replay.simulate` extended with: take-profit, structure stop,
    risk-based sizing, FY (1-April) tax netting with loss carry-forward, trade detail.

    stop_mode 'pct'    -> stop price = fill*(1-stop)
    stop_mode 'struct' -> stop price = base low (LO), floored at fill*(1-2*stop)
    risk_pct not None  -> position value = risk_pct*equity / stop_distance, capped size_cap
    """
    rng = np.random.default_rng(seed)
    cash = float(capital)
    positions = []          # (col, entry_i, buy, qty, stop_px, tvp_at_entry)
    trades = []
    equity = np.empty(len(days_idx), dtype=float)
    invested = np.empty(len(days_idx), dtype=float)
    passed_up = 0
    y_day = 1.0 + cash_yield / 252.0
    fy_st = fy_lt = 0.0
    carry = 0.0             # carried-forward loss (<=0)

    def fy_of(d):
        return d.year if d.month >= 4 else d.year - 1

    cur_fy = fy_of(dates[days_idx[0]])

    for k, i in enumerate(days_idx):
        if cash_yield and cash > 0:
            cash *= y_day
        d = dates[i]
        if stcg and fy_of(d) != cur_fy:
            st, lt, cf = fy_st, fy_lt, carry
            if cf < 0:
                u = min(-cf, max(st, 0.0)); st -= u; cf += u
                u = min(-cf, max(lt, 0.0)); lt -= u; cf += u
            if st < 0:
                lt += st; st = 0.0
            if lt < 0:
                cf += lt; lt = 0.0
            tax = stcg * max(st, 0.0) + ltcg * max(lt, 0.0)
            cash -= tax
            carry = cf
            fy_st = fy_lt = 0.0
            cur_fy = fy_of(d)

        # ── entries ──
        if not weak[i]:
            cand = np.nonzero(TRIG[i])[0]
            if len(cand):
                mtm = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                          for c, _, b, q, _, _ in positions)
                eq = cash + mtm
                cand = rng.permutation(cand)
                for c in cand:
                    if len(positions) >= slots:
                        passed_up += 1
                        continue
                    pv = float(PIV[i, c])
                    if fill_close:
                        fill = float(C[i, c])
                    else:
                        fill = max(pv, float(O[i, c])) if fill_realistic else pv
                    if not np.isfinite(fill) or fill <= 0:
                        continue
                    if stop_mode == "struct":
                        bl = float(LO[i, c])
                        sp = bl if np.isfinite(bl) else fill * (1 - stop)
                        sp = max(sp, fill * (1 - 2 * stop))
                        sp = min(sp, fill * (1 - 0.01))
                    else:
                        sp = fill * (1 - stop)
                    if risk_pct is not None:
                        dist = max((fill - sp) / fill, 1e-4)
                        size = min(risk_pct / dist, size_cap) * eq
                    else:
                        size = min(size_pct, size_cap) * eq
                    qty = int(size / fill)
                    if qty < 1 or cash < qty * fill * (1 + cost):
                        passed_up += 1
                        continue
                    cash -= qty * fill * (1 + cost)
                    positions.append((c, i, fill, qty, sp, float(TVp[i, c])))

        # ── exits at the close ──
        still = []
        for c, ei, b, q, sp, tv in positions:
            cl = C[i, c]
            if np.isnan(cl):
                still.append((c, ei, b, q, sp, tv))
                continue
            reason = None
            if cl <= sp:
                reason = "stop"
            elif target is not None and cl >= b * (1 + target):
                reason = "target"
            elif i > ei and not np.isnan(SMA[i, c]) and cl < SMA[i, c]:
                reason = "trail"
            if reason:
                cash += q * float(cl) * (1 - cost)
                pnl = q * (float(cl) * (1 - cost) - b * (1 + cost))
                held = (dates[i] - dates[ei]).days
                if stcg:
                    if held > 365:
                        fy_lt += pnl
                    else:
                        fy_st += pnl
                if collect_trades:
                    trades.append(dict(col=int(c), ei=int(ei), xi=int(i), buy=b,
                                       sell=float(cl), qty=q, reason=reason,
                                       held=held, tv=tv,
                                       ret=float(cl) / b - 1.0, notional=q * b))
            else:
                still.append((c, ei, b, q, sp, tv))
        positions = still
        mtm = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                  for c, _, b, q, _, _ in positions)
        equity[k] = cash + mtm
        invested[k] = mtm

    last = days_idx[-1]
    for c, ei, b, q, sp, tv in positions:
        cl = C[last, c]
        px = float(cl) if not np.isnan(cl) else b
        if collect_trades:
            trades.append(dict(col=int(c), ei=int(ei), xi=int(last), buy=b, sell=px,
                               qty=q, reason="open_marked",
                               held=(dates[last] - dates[ei]).days, tv=tv,
                               ret=px / b - 1.0, notional=q * b))
    return equity, trades, passed_up, invested


# ────────────────────────────────────────────────────────────────────── statistics
def stats_from(equity, dates_used, trades, capital=CAPITAL, invested=None):
    e = pd.Series(equity, index=dates_used)
    yrs = (dates_used[-1] - dates_used[0]).days / 365.25
    cagr = (e.iloc[-1] / capital) ** (1 / yrs) - 1 if e.iloc[-1] > 0 else -1.0
    dd = float((e / e.cummax() - 1).min())
    r = np.array([t["ret"] for t in trades]) if trades else np.array([])
    closed = [t for t in trades if t["reason"] != "open_marked"]
    yearly = e.groupby(e.index.year).last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] / capital - 1
    # per-trade expectancy net of round-trip cost is already inside `ret`? no -> apply
    wins = r[r > 0]
    losses = r[r <= 0]
    out = dict(final=float(e.iloc[-1]), x=float(e.iloc[-1] / capital), cagr=cagr * 100,
               dd=dd * 100, calmar=(cagr * 100) / abs(dd * 100) if dd else np.nan,
               n=len(trades), n_closed=len(closed),
               tpy=len(trades) / yrs if yrs else 0,
               win=float((r > 0).mean() * 100) if len(r) else 0.0,
               mean=float(r.mean() * 100) if len(r) else 0.0,
               median=float(np.median(r) * 100) if len(r) else 0.0,
               avg_win=float(wins.mean() * 100) if len(wins) else 0.0,
               avg_loss=float(losses.mean() * 100) if len(losses) else 0.0,
               hold=float(np.mean([t["held"] for t in trades])) if trades else 0.0,
               invested_pct=float(np.mean(invested / equity) * 100)
               if invested is not None else np.nan,
               yearly={int(k): round(v * 100, 2) for k, v in yr.items()})
    # longest losing streak
    streak = mx = 0
    for t in sorted(trades, key=lambda z: z["xi"]):
        if t["ret"] <= 0:
            streak += 1; mx = max(mx, streak)
        else:
            streak = 0
    out["max_loss_streak"] = mx
    return out, e
