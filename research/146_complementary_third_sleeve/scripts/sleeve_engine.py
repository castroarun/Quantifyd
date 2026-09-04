"""research/146 — candidate third-sleeve engine (G1): mean-reversion / pullback families on
NSE cash equities, PIT top-500 universe, 10 slots x 10%, hard stops, no averaging,
25bps/side, FY-netted tax. Judged later on BLEND value (blend3.py); here: standalone stats,
tradeability gate (WR, avg win/loss, expectancy, max losing streak) and corr vs the TN leg.

Families: F1 KC6 (parked live system, rebuilt — services untouched), F2 Arun pullback-reversal,
F3 Connors RSI2/3 oversold-in-uptrend, F4 N-day-low washout.
"""
from __future__ import annotations
import sys, csv, time, sqlite3, pickle
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
DB = Path("/home/arun/quantifyd/backtest_data/market_data.db")
TN_NAV = Path("/home/arun/quantifyd/research/144_truenorth_reassessment/results/"
              "nav_INC_cash_n8_d15_tax1.csv")

BENCH = "NIFTYBEES"
EXCl = {"NIFTYBEES", "BANKNIFTY", "INDIAVIX", "JUNIORBEES", "GOLDBEES", "SILVERBEES",
        "LIQUIDBEES", "BANKBEES", "ICICIB22", "CPSEETF", "MON100", "MAFANG", "SETFNIF50",
        "SETFGOLD", "SETFNIFBK", "GOLDCASE", "KOTAKGOLD", "AXISGOLD", "HDFCGOLD",
        "GOLDSHARE", "GOLD1", "SILVER1", "NIFTYJR", "NIFTYMID", "NIFTYIT", "FINNIFTY",
        "MIDCPNIFTY"}
SLOTS = 10
COST = 0.0025            # per side
CASH_Y = 0.05
PANEL_START = "2005-01-01"
LOOP_START = pd.Timestamp("2006-04-01")
WINDOWS = [("w0", None, None), ("wa", "2012-01-01", None),
           ("w1", "2016-06-01", "2019-12-31"), ("w2", "2020-01-01", None)]


def _is_etf(s):
    return (s.endswith("BEES") or s.endswith("ETF") or s.endswith("GOLD") or
            s.endswith("SILVER") or s.endswith("IETF") or s.endswith("CASE") or
            s.startswith("NIFTY") or s in EXCl)


def _rsi(close, n):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


class SCtx:
    def __init__(self):
        t0 = time.time()
        con = sqlite3.connect(str(DB))
        df = pd.read_sql(
            "SELECT symbol,date,open,high,low,close,volume FROM market_data_unified "
            "WHERE timeframe='day' AND close IS NOT NULL AND date>=? ORDER BY symbol,date",
            con, params=(PANEL_START,), parse_dates=["date"])
        con.close()
        df["tv"] = df["close"] * df["volume"].fillna(0)
        piv = lambda c: df.pivot_table(index="date", columns="symbol", values=c).sort_index()
        cl = piv("close")
        self.dates = cl.index
        self.syms = list(cl.columns)
        self.sidx = {s: i for i, s in enumerate(self.syms)}
        self.C = cl.to_numpy(np.float32)
        self.O = piv("open").to_numpy(np.float32)
        self.Hi = piv("high").to_numpy(np.float32)
        self.Lo = piv("low").to_numpy(np.float32)
        self.CF = cl.ffill().to_numpy(np.float32)     # for MTM of held names
        tv = piv("tv")
        hi = pd.DataFrame(self.Hi, index=self.dates, columns=self.syms)
        lo = pd.DataFrame(self.Lo, index=self.dates, columns=self.syms)
        # indicators (float32 matrices)
        self.SMA200 = cl.rolling(200, min_periods=200).mean().to_numpy(np.float32)
        self.SMA50 = cl.rolling(50, min_periods=50).mean().to_numpy(np.float32)
        self.EMA50 = cl.ewm(span=50, adjust=False).mean().to_numpy(np.float32)
        self.SMA20 = cl.rolling(20, min_periods=20).mean().to_numpy(np.float32)
        self.SMA5 = cl.rolling(5, min_periods=5).mean().to_numpy(np.float32)
        ema6 = cl.ewm(span=6, adjust=False).mean()
        c1 = cl.shift(1)
        tr_np = np.fmax((hi - lo).to_numpy(np.float32),
                        np.fmax((hi - c1).abs().to_numpy(np.float32),
                                (lo - c1).abs().to_numpy(np.float32)))
        tr = pd.DataFrame(tr_np, index=self.dates, columns=self.syms)
        atr6 = tr.ewm(span=6, adjust=False).mean()
        self.EMA6 = ema6.to_numpy(np.float32)
        self.ATR6 = atr6.to_numpy(np.float32)
        self.RSI2 = _rsi(cl, 2).to_numpy(np.float32)
        self.RSI3 = _rsi(cl, 3).to_numpy(np.float32)
        self.MIN7 = cl.rolling(7, min_periods=7).min().to_numpy(np.float32)
        self.MIN10 = cl.rolling(10, min_periods=10).min().to_numpy(np.float32)
        self._cl = cl; self._tr = tr
        # monthly PIT top-500 membership
        s = pd.Series(np.arange(len(self.dates)), index=self.dates)
        me = [int(g.iloc[-1]) for _, g in s.groupby([self.dates.year, self.dates.month])]
        self.memb = {}                      # month-end pos -> bool array
        for p in me:
            w = tv.iloc[max(0, p - 201):p + 1]
            cnt = w.notna().sum(); med = w.median()
            elig = med[(cnt >= 75) & (med > 0)].sort_values(ascending=False)
            uni = [s2 for s2 in elig.index if not _is_etf(s2)][:500]
            m = np.zeros(len(self.syms), bool)
            for s2 in uni:
                m[self.sidx[s2]] = True
            self.memb[p] = m
        self.me_pos = sorted(self.memb)
        self.i0 = int(np.searchsorted(self.dates.values, np.datetime64(LOOP_START)))
        self._crash = None
        try:
            self.tn_nav = pd.read_csv(TN_NAV, index_col=0, parse_dates=True).iloc[:, 0]
        except Exception:
            self.tn_nav = None
        print(f"panel {self.C.shape} + indicators [{time.time()-t0:.0f}s]", flush=True)

    def memb_at(self, i):
        # membership from the last completed month-end before i
        k = np.searchsorted(self.me_pos, i) - 1
        return self.memb[self.me_pos[max(0, k)]]

    def crash_off(self):
        """Universe crash filter: median(ATR14 / SMA100(ATR14)) >= 1.3 blocks entries."""
        if self._crash is None:
            atr14 = self._tr.ewm(span=14, adjust=False).mean()
            ratio = atr14 / atr14.rolling(100, min_periods=100).mean()
            self._crash = (ratio.median(axis=1) >= 1.3).to_numpy()
        return self._crash


def wstats(nav, a, b):
    n = nav
    if a: n = n[n.index >= pd.Timestamp(a)]
    if b: n = n[n.index <= pd.Timestamp(b)]
    if len(n) < 60:
        return dict(cagr="", dd="", sharpe="", calmar="")
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cagr = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    dd = ((n - n.cummax()) / n.cummax()).min()
    return dict(cagr=round(cagr * 100, 2), dd=round(dd * 100, 2), sharpe=round(sh, 2),
                calmar=(round(cagr / abs(dd), 2) if dd < 0 else ""))


def run_sleeve(ctx, family, p, tax=False, cost=COST):
    """Event-driven sleeve backtest. Returns row dict + '_nav'."""
    t0 = time.time()
    D = ctx.dates; n_sym = len(ctx.syms)
    C, O, Hi, Lo, CF = ctx.C, ctx.O, ctx.Hi, ctx.Lo, ctx.CF
    up200 = None
    cash = 1.0
    day_cash = (1 + CASH_Y) ** (1 / 252)
    pos = {}      # j -> dict(entry, val, qtyv, stop, target, ei, lim/extra)
    pend = {}     # F2 buy-stops: j -> (trigger, stop, placed_i)
    nav_v = np.empty(len(D) - ctx.i0)
    trades = []   # net pct per trade
    st = dict(st_gain=0.0, lt_gain=0.0, tax=0.0, cost=0.0)
    crash = ctx.crash_off() if p.get("crash") else None

    def settle():
        nonlocal cash
        stg, ltg = st["st_gain"], st["lt_gain"]
        if stg < 0:
            ltg += stg; stg = 0.0
        t = 0.20 * max(0.0, stg) + 0.125 * max(0.0, ltg)
        if t > 0:
            cash -= t; st["tax"] += t
        st["st_gain"] = st["lt_gain"] = 0.0

    def close_pos(j, i, px):
        nonlocal cash
        d = pos.pop(j)
        ret = px / d["entry"]
        gross = d["qtyv"] * ret
        fee = gross * cost
        gain = gross - fee - d["qtyv"] * (1 + cost)   # net of both sides' cost
        if tax:
            st["st_gain" if (i - d["ei"]) < 252 else "lt_gain"] += gain
        st["cost"] += fee
        cash += gross - fee
        trades.append(gain / d["qtyv"])

    for i in range(ctx.i0, len(D)):
        if cash > 0:
            cash *= day_cash
        if tax and D[i].month == 4 and D[i - 1].month == 3:
            settle()
        # ---- manage exits ----
        for j in list(pos):
            d = pos[j]
            o, h, l, c = O[i, j], Hi[i, j], Lo[i, j], C[i, j]
            if c != c:                     # halt day: keep, mark flat
                continue
            days_held = i - d["ei"]
            if family == "kc6":
                lim = ctx.EMA6[i - 1, j]   # standing limit at KC mid (data <= t-1)
                if h == h and lim == lim and h >= lim:
                    close_pos(j, i, max(o, lim) if o == o else lim); continue
                if c <= d["entry"] * (1 - p["sl"]):
                    close_pos(j, i, c); continue
                if p.get("tp") and c >= d["entry"] * (1 + p["tp"]):
                    close_pos(j, i, c); continue
                if days_held >= p["hold"]:
                    close_pos(j, i, c); continue
            elif family == "pull":
                if l == l and l <= d["stop"]:                       # intraday hard stop
                    close_pos(j, i, min(o, d["stop"]) if o == o else d["stop"]); continue
                if h == h and h >= d["target"]:
                    close_pos(j, i, max(o, d["target"]) if o == o else d["target"]); continue
                if p.get("sma20_exit") and c < ctx.SMA20[i, j]:
                    close_pos(j, i, c); continue
                if days_held >= p["time"]:
                    close_pos(j, i, c); continue
            else:                          # conn / wash: close-based exits
                if c <= d["entry"] * (1 - p["sl"]):
                    close_pos(j, i, c); continue
                exit_sig = False
                if family == "conn":
                    exit_sig = (c > ctx.SMA5[i, j]) or \
                        (p.get("rsi_exit") and ctx.RSI2[i, j] > p["rsi_exit"])
                else:
                    exit_sig = Hi[i, j] == Hi[i, j] and c > Hi[i - 1, j]
                if exit_sig or days_held >= p["time"]:
                    close_pos(j, i, c); continue
        # mark to market
        for j, d in pos.items():
            if CF[i, j] == CF[i, j] and CF[i - 1, j] > 0:
                d["val"] *= CF[i, j] / CF[i - 1, j]
        # ---- F2 pending buy-stops fire ----
        if family == "pull":
            for j in list(pend):
                trig, stp, pi = pend[j]
                if i - pi > 1:
                    del pend[j]; continue
                o, h = O[i, j], Hi[i, j]
                if h == h and h > trig and j not in pos and len(pos) < SLOTS:
                    navv = cash + sum(d["val"] for d in pos.values())
                    q = navv / SLOTS
                    if cash >= q * (1 + cost):
                        px = max(o, trig) if o == o else trig
                        R = px - stp
                        if R > 0:
                            cash -= q * (1 + cost); st["cost"] += q * cost
                            pos[j] = dict(entry=px, val=q, qtyv=q, stop=stp,
                                          target=px + p["rr"] * R, ei=i)
                    del pend[j]
        # ---- new signals at close ----
        blocked = crash is not None and crash[i]
        if not blocked:
            memb = ctx.memb_at(i)
            c_row = C[i]; s200 = ctx.SMA200[i]
            cand = []
            if family == "kc6":
                lower = ctx.EMA6[i] - p["mult"] * ctx.ATR6[i]
                sig = (c_row < lower) & (c_row > s200) & memb
                for j in np.nonzero(sig)[0]:
                    if j not in pos:
                        cand.append(((lower[j] - c_row[j]) / c_row[j], j))
            elif family == "conn":
                r = ctx.RSI3[i] if p.get("rsi3") else ctx.RSI2[i]
                sig = (r < p["th"]) & (c_row > s200) & memb
                for j in np.nonzero(sig)[0]:
                    if j not in pos:
                        cand.append((-r[j], j))
            elif family == "wash":
                mn = ctx.MIN10[i] if p.get("n10") else ctx.MIN7[i]
                sig = (c_row <= mn) & (c_row > s200) & memb
                for j in np.nonzero(sig)[0]:
                    if j not in pos:
                        cand.append((1.0, j))
            elif family == "pull":
                ma = ctx.EMA50[i] if p.get("ema") else ctx.SMA50[i]
                ma_hist = ctx.EMA50 if p.get("ema") else ctx.SMA50
                green = (C[i] > O[i])
                red1 = (C[i - 1] < O[i - 1])
                touched = (Lo[i] <= ma) | (Lo[i - 1] <= ma_hist[i - 1]) | \
                          (Lo[i - 2] <= ma_hist[i - 2])
                rising = ctx.SMA50[i] > ctx.SMA50[i - 10]
                sig = green & red1 & touched & rising & (c_row > s200) & memb
                for j in np.nonzero(sig)[0]:
                    if j not in pos and j not in pend:
                        stp = min(Lo[i, j], Lo[i - 1, j])
                        if stp == stp and Hi[i, j] == Hi[i, j]:
                            pend[j] = (float(Hi[i, j]), float(stp), i)
            cand.sort(reverse=True)
            if family != "pull":
                for _, j in cand:
                    if len(pos) >= SLOTS:
                        break
                    navv = cash + sum(d["val"] for d in pos.values())
                    q = navv / SLOTS
                    if cash < q * (1 + cost):
                        break
                    px = float(c_row[j])
                    cash -= q * (1 + cost); st["cost"] += q * cost
                    pos[j] = dict(entry=px, val=q, qtyv=q, stop=0.0, target=np.inf, ei=i)
        nav_v[i - ctx.i0] = cash + sum(d["val"] for d in pos.values())
    if tax:
        settle()
        nav_v[-1] = cash + sum(d["val"] for d in pos.values())
    nav = pd.Series(nav_v, index=D[ctx.i0:])
    tr = np.array(trades)
    wins = tr[tr > 0]; losses = tr[tr <= 0]
    streak = mx = 0
    for x in tr:
        streak = streak + 1 if x <= 0 else 0
        mx = max(mx, streak)
    row = dict(family=family, params=str(p), tax=int(tax), rt=cost * 2,
               n_trades=len(tr), win_rate=round(100 * len(wins) / len(tr), 1) if len(tr) else 0,
               avg_win=round(100 * wins.mean(), 2) if len(wins) else 0,
               avg_loss=round(100 * losses.mean(), 2) if len(losses) else 0,
               expectancy=round(100 * tr.mean(), 3) if len(tr) else 0,
               max_lose_streak=mx, tax_paid=round(st["tax"] * 100, 1),
               secs=round(time.time() - t0, 1))
    for w, a, b in WINDOWS:
        for k, v in wstats(nav, a, b).items():
            row[f"{w}_{k}"] = v
    if ctx.tn_nav is not None:
        idx = nav.index.intersection(ctx.tn_nav.index)
        row["corr_tn_d"] = round(float(
            nav.loc[idx].pct_change().corr(ctx.tn_nav.loc[idx].pct_change())), 3)
    row["_nav"] = nav
    return row


VARIANTS = [
    ("kc6", "kc_base", dict(mult=1.3, sl=0.05, tp=0.15, hold=15)),
    ("kc6", "kc_sl7", dict(mult=1.3, sl=0.07, tp=0.15, hold=15)),
    ("kc6", "kc_noTP", dict(mult=1.3, sl=0.05, tp=None, hold=15)),
    ("kc6", "kc_m15", dict(mult=1.5, sl=0.05, tp=0.15, hold=15)),
    ("kc6", "kc_crash", dict(mult=1.3, sl=0.05, tp=0.15, hold=15, crash=True)),
    ("kc6", "kc_hold10", dict(mult=1.3, sl=0.05, tp=0.15, hold=10)),
    ("pull", "p_sma_2R_t10", dict(rr=2.0, time=10)),
    ("pull", "p_ema_2R_t10", dict(rr=2.0, time=10, ema=True)),
    ("pull", "p_sma_15R_t10", dict(rr=1.5, time=10)),
    ("pull", "p_sma_3R_t10", dict(rr=3.0, time=10)),
    ("pull", "p_sma_2R_t15", dict(rr=2.0, time=15)),
    ("pull", "p_sma_2R_sma20x", dict(rr=2.0, time=15, sma20_exit=True)),
    ("conn", "c_rsi2_10", dict(th=10, sl=0.07, time=7, rsi_exit=65)),
    ("conn", "c_rsi2_5", dict(th=5, sl=0.07, time=7, rsi_exit=65)),
    ("conn", "c_rsi2_15", dict(th=15, sl=0.07, time=7, rsi_exit=65)),
    ("conn", "c_rsi3_15", dict(th=15, sl=0.07, time=7, rsi_exit=65, rsi3=True)),
    ("conn", "c_rsi2_10_sl5", dict(th=10, sl=0.05, time=7, rsi_exit=65)),
    ("conn", "c_rsi2_10_sma5", dict(th=10, sl=0.07, time=10, rsi_exit=None)),
    ("wash", "w7", dict(sl=0.07, time=7)),
    ("wash", "w10", dict(sl=0.07, time=7, n10=True)),
    ("wash", "w7_sl5", dict(sl=0.05, time=7)),
    ("wash", "w7_t10", dict(sl=0.07, time=10)),
]

FIELDS = ["label", "family", "params", "tax", "rt", "n_trades", "win_rate", "avg_win",
          "avg_loss", "expectancy", "max_lose_streak", "tax_paid", "corr_tn_d", "secs"]
for w, _, _ in WINDOWS:
    FIELDS += [f"{w}_cagr", f"{w}_dd", f"{w}_sharpe", f"{w}_calmar"]


def main():
    ctx = SCtx()
    path = RES / "g1_candidates.csv"
    done = set()
    if path.exists():
        with open(path) as f:
            done = {r["label"] for r in csv.DictReader(f)}
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for family, name, p in VARIANTS:
        if only and family != only:
            continue
        for tax in (False, True):
            lbl = f"{name}_tax{int(tax)}"
            if lbl in done:
                continue
            r = run_sleeve(ctx, family, p, tax=tax)
            if tax:
                r["_nav"].to_csv(RES / f"nav_{name}_tax1.csv")
            row = {k: r.get(k, "") for k in FIELDS}
            row["label"] = lbl
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print(f"{lbl:24} n={row['n_trades']:>5} WR={row['win_rate']:>5} "
                  f"exp={row['expectancy']:>7} waCAGR={row['wa_cagr']:>7} "
                  f"waDD={row['wa_dd']:>7} corrTN={row['corr_tn_d']} "
                  f"streak={row['max_lose_streak']} [{row['secs']}s]", flush=True)
    print("G1 DONE", flush=True)


if __name__ == "__main__":
    main()
