"""research/144 — True North momentum re-assessment sweep (gate bake-off, actions, slots, exits, tax).

Deployed-faithful engine of services/momentum_paper.py (NOT the r62 re-equalizing engine):
PIT top-200-by-traded-value universe, rsblend 6m/12m RS vs NIFTYBEES, top-n equal-weight,
buffer=round(2.75n), monthly rebalance with NO trims (top-up only), weekly macro gate,
daily Donchian stop, 0.3% RT cost, idle cash 6.5% p.a., tax on realization (20% STCG <365d,
12.5% LTCG >=365d). All figures daily-marked.

Phases: smoke | A (gate series x construction) | B (action x freq) | C (slots x exits) |
D (finalists x 12 rebalance-day offsets + sensitivities). Incremental CSVs, resume-safe.
"""
from __future__ import annotations
import sys, csv, time, sqlite3, pickle
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
DB = Path("/home/arun/quantifyd/backtest_data/market_data.db")

BENCH = "NIFTYBEES"
GATE_SERIES = ["NIFTYBEES", "NIFTY50", "NIFTY500", "NIFTYMIDCAP150", "NIFTYSMLCAP250"]
CONS = ["sma100", "sma150", "sma200", "ema100", "ema150", "ema200",
        "xo50_200", "xo20_100", "dd8", "dd10", "dd12", "dd15", "mom63", "mom126"]
EXCLUDE = {"NIFTYBEES", "NIFTY50", "BANKNIFTY", "INDIAVIX", "NIFTYJR", "NIFTYMID",
           "NIFTYIT", "FINNIFTY", "MIDCPNIFTY", "JUNIORBEES",
           "GOLDBEES", "SILVERBEES", "LIQUIDBEES", "BANKBEES", "ICICIB22", "CPSEETF",
           "MON100", "MAFANG", "SETFNIF50", "SETFGOLD", "SETFNIFBK", "GOLDCASE",
           "KOTAKGOLD", "AXISGOLD", "HDFCGOLD", "GOLDSHARE", "GOLD1", "SILVER1"}
RT = 0.003
CASH_ANNUAL = 0.065
PANEL_START = "2005-01-01"
LOOP_START = pd.Timestamp("2006-04-01")
UNI_SIZE = 200
LIQ_LB = 202
RANK_KEEP = 60
WINDOWS = [("w0", None, None), ("wa", "2012-01-01", None),
           ("w1", "2016-06-01", "2019-12-31"), ("w2", "2020-01-01", None)]

FIELDS = ["label", "series", "cons", "action", "freq", "n", "exit", "offset", "tax",
          "cash_y", "rt"]
for w, _, _ in WINDOWS:
    FIELDS += [f"{w}_cagr", f"{w}_dd", f"{w}_sharpe", f"{w}_calmar"]
FIELDS += ["fills", "donch_exits", "gate_events", "cost_pct", "tax_pct", "avg_inv", "secs"]


def _is_etf(sym):
    return sym.endswith("BEES") or sym.endswith("ETF") or sym.endswith("GOLD") \
        or sym.endswith("SILVER") or sym.endswith("IETF") or sym.endswith("CASE") \
        or sym in EXCLUDE or sym in GATE_SERIES or sym.startswith("NIFTY")


class Ctx:
    def __init__(self):
        t0 = time.time()
        con = sqlite3.connect(str(DB))
        df = pd.read_sql(
            "SELECT symbol,date,close,volume FROM market_data_unified "
            "WHERE timeframe='day' AND close IS NOT NULL AND date>=? ORDER BY symbol,date",
            con, params=(PANEL_START,), parse_dates=["date"])
        con.close()
        df["tv"] = df["close"] * df["volume"].fillna(0)
        close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        self.tv = df.pivot_table(index="date", columns="symbol", values="tv").sort_index()
        self.close = close
        self.cf = close.ffill()
        self.C = self.cf.to_numpy()
        self.rawnn = close.notna().to_numpy()
        self.dates = close.index
        self.syms = list(close.columns)
        self.sidx = {s: i for i, s in enumerate(self.syms)}
        iso = self.dates.isocalendar()
        wk_last = set(pd.Series(self.dates, index=self.dates)
                      .groupby([iso.year.values, iso.week.values]).max())
        self.is_wk = np.array([d in wk_last for d in self.dates])
        self.i0 = int(np.searchsorted(self.dates.values, np.datetime64(LOOP_START)))
        self._gate = {}
        self._exitm = {}
        self._me = {}
        cache = RES / "ranks_cache.pkl"
        self.rank_cache = pickle.load(open(cache, "rb")) if cache.exists() else {}
        self._rank_cache_file = cache
        self._rank_dirty = 0
        print(f"panel {close.shape} {self.dates[0].date()}..{self.dates[-1].date()} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    def me_pos(self, offset):
        if offset not in self._me:
            s = pd.Series(np.arange(len(self.dates)), index=self.dates)
            pos = []
            for _, grp in s.groupby([self.dates.year, self.dates.month]):
                pos.append(int(grp.iloc[max(0, len(grp) - 1 - offset)]))
            self._me[offset] = sorted(p for p in pos if p >= self.i0)
        return self._me[offset]

    def gate_arr(self, series, cons):
        key = (series, cons)
        if key in self._gate:
            return self._gate[key]
        if series == "NONE":
            a = np.zeros(len(self.dates), bool)
        else:
            g = self.close[series].dropna()
            if cons.startswith("sma"):
                L = int(cons[3:]); ro = g < g.rolling(L, min_periods=L).mean()
            elif cons.startswith("ema"):
                L = int(cons[3:]); ro = g < g.ewm(span=L, adjust=False).mean()
            elif cons.startswith("xo"):
                f, s2 = (int(x) for x in cons[2:].split("_"))
                ro = g.rolling(f, min_periods=f).mean() < g.rolling(s2, min_periods=s2).mean()
            elif cons.startswith("dd"):
                x = float(cons[2:]) / 100.0
                ro = g < g.rolling(252, min_periods=252).max() * (1 - x)
            elif cons.startswith("mom"):
                L = int(cons[3:]); ro = g < g.shift(L)
            else:
                raise ValueError(cons)
            a = (ro.astype(float).reindex(self.dates).ffill().fillna(0.0) > 0.5).to_numpy()
        self._gate[key] = a
        return a

    def exit_mat(self, kind, p):
        key = (kind, p)
        if key in self._exitm:
            return self._exitm[key]
        if kind == "donch":
            m = self.cf.rolling(p, min_periods=p).min().shift(1)
        elif kind == "smatr":
            m = self.cf.rolling(p, min_periods=p).mean().shift(1)
        elif kind == "atr":
            m = self.cf.diff().abs().rolling(20, min_periods=20).mean().shift(1)
        else:
            raise ValueError(kind)
        a = m.to_numpy(np.float32)
        self._exitm[key] = a
        return a

    def ranking(self, i):
        """Ranked top-RANK_KEEP list at date-position i (causal, PIT universe)."""
        if i in self.rank_cache:
            return self.rank_cache[i]
        d = self.dates[i]
        w = self.tv.iloc[max(0, i - LIQ_LB + 1):i + 1]
        cnt = w.notna().sum(); med = w.median()
        elig = med[(cnt >= 75) & (med > 0)].sort_values(ascending=False)
        fresh = self.rawnn[max(0, i - 4):i + 1].any(axis=0)
        uni = [s for s in elig.index
               if not _is_etf(s) and fresh[self.sidx[s]]][:UNI_SIZE]
        if i <= 253 or BENCH not in self.cf.columns:
            return None
        p1 = self.C[i]; p126 = self.C[i - 126]; p252 = self.C[i - 252]
        jb = self.sidx[BENCH]
        nf126 = p1[jb] / p126[jb]; nf252 = p1[jb] / p252[jb]
        sc = {}
        for s in uni:
            j = self.sidx[s]
            a, b, c = p1[j], p126[j], p252[j]
            if a == a and b == b and c == c and b > 0 and c > 0:
                sc[s] = 0.5 * (a / b) / nf126 + 0.5 * (a / c) / nf252
        rank = [s for s, _ in sorted(sc.items(), key=lambda kv: -kv[1])][:RANK_KEEP]
        self.rank_cache[i] = rank
        self._rank_dirty += 1
        if self._rank_dirty % 200 == 0:
            self.save_ranks()
        return rank

    def save_ranks(self):
        pickle.dump(self.rank_cache, open(self._rank_cache_file, "wb"))


def wstats(nav, a, b):
    n = nav
    if a:
        n = n[n.index >= pd.Timestamp(a)]
    if b:
        n = n[n.index <= pd.Timestamp(b)]
    if len(n) < 60:
        return dict(cagr="", dd="", sharpe="", calmar="")
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cagr = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    dd = ((n - n.cummax()) / n.cummax()).min()
    cal = cagr / abs(dd) if dd < 0 else np.nan
    return dict(cagr=round(cagr * 100, 2), dd=round(dd * 100, 2), sharpe=round(sh, 2),
                calmar=(round(cal, 2) if cal == cal else ""))


def run(ctx, series="NIFTYBEES", cons="sma100", action="cash", freq="weekly",
        n=8, exit=("donch", 15), offset=0, tax=False, cash_y=CASH_ANNUAL, rt=RT):
    t0 = time.time()
    dates = ctx.dates; C = ctx.C; sidx = ctx.sidx
    gate_on = series != "NONE"
    garr = ctx.gate_arr(series, cons)
    me = set(ctx.me_pos(offset))
    buf = int(round(2.75 * n))
    ek, ep = (exit if exit else (None, None))
    em = ctx.exit_mat(ek, ep) if ek else None
    day_cash = (1 + cash_y) ** (1 / 252)
    st = dict(cost=0.0, tax=0.0, fills=0, donch=0, gate_events=0, traded=0.0,
              st_gain=0.0, lt_gain=0.0)
    held = {}          # sym -> [value, cost_basis, buy_i, peak_px]
    cash = 1.0
    risk_off = False
    frac = 1.0
    nav_v = np.empty(len(dates) - ctx.i0)
    inv_sum = 0.0

    def sell(s, i, f=1.0, why="x"):
        nonlocal cash
        v, c0, bi, pk = held[s]
        sv, sc = v * f, c0 * f
        fee = sv * rt / 2
        gain = sv - sc
        if tax:
            days = (dates[i] - dates[bi]).days
            st["st_gain" if days < 365 else "lt_gain"] += gain
        st["cost"] += fee; st["traded"] += sv
        cash += sv - fee
        if f >= 0.999:
            del held[s]
        else:
            held[s] = [v - sv, c0 - sc, bi, pk]

    def settle_tax():
        """Indian fiscal-year netting: STCL offsets STCG then LTCG; LTCL offsets LTCG only.
        No loss carry-forward (mildly conservative). Deducted from cash (may go negative)."""
        nonlocal cash
        stg, ltg = st["st_gain"], st["lt_gain"]
        if stg < 0:
            ltg += stg          # short-term loss set off against long-term gains
            stg = 0.0
        t = 0.20 * max(0.0, stg) + 0.125 * max(0.0, ltg)
        if t > 0:
            cash -= t
            st["tax"] += t
        st["st_gain"] = st["lt_gain"] = 0.0

    def do_fill(i, allow_new, tfrac):
        nonlocal cash
        rank = ctx.ranking(i)
        if rank is None:
            return
        rpos = {s: k for k, s in enumerate(rank)}
        for s in list(held):
            if rpos.get(s, 9999) >= buf:
                sell(s, i, 1.0, "evict")
        st["fills"] += 1
        if not allow_new:
            return
        names = list(held)
        for s in rank:
            if len(names) >= n:
                break
            j = sidx[s]
            if s not in held and ctx.rawnn[i, j] and C[i, j] == C[i, j]:
                names.append(s)
        if not names:
            return
        tot = sum(v[0] for v in held.values()) + cash
        w = tfrac * tot / n
        demand = {s: w - (held[s][0] if s in held else 0.0) for s in names}
        demand = {s: d for s, d in demand.items() if d > 1e-9}
        td = sum(demand.values())
        if td <= 0:
            return
        avail = max(0.0, cash) / (1 + rt / 2)
        k = min(1.0, avail / td)
        for s, dv in demand.items():
            b = dv * k
            if b <= 1e-9:
                continue
            fee = b * rt / 2
            cash -= b + fee
            st["cost"] += fee; st["traded"] += b
            if s in held:
                held[s][0] += b; held[s][1] += b
            else:
                held[s] = [b, b, i, C[i, sidx[s]]]

    for i in range(ctx.i0, len(dates)):
        row = C[i]; prow = C[i - 1]
        for s, stt in held.items():
            j = sidx[s]
            p1, p0 = row[j], prow[j]
            if p1 == p1 and p0 == p0 and p0 > 0:
                stt[0] *= p1 / p0
        if cash > 0:
            cash *= day_cash
        if tax and dates[i].month == 4 and dates[i - 1].month == 3:
            settle_tax()                      # fiscal-year end (Mar 31) tax settlement
        if em is not None and held:
            for s in list(held):
                j = sidx[s]
                p1 = row[j]
                if ek == "atr":
                    thr = held[s][3] - 3.0 * em[i, j]
                else:
                    thr = em[i, j]
                if thr == thr and p1 == p1 and p1 < thr:
                    sell(s, i, 1.0, "stop")
                    st["donch"] += 1
        for s in held:                                # update peaks after stop check
            j = sidx[s]
            if row[j] == row[j]:
                held[s][3] = max(held[s][3], row[j])
        if gate_on:
            chk = (freq == "daily") or (freq == "weekly" and ctx.is_wk[i]) or \
                  (freq == "monthly" and i in me)
            if chk:
                off = bool(garr[i])
                if off and not risk_off:
                    risk_off = True
                    st["gate_events"] += 1
                    if action == "cash" and held:
                        for s in list(held):
                            sell(s, i, 1.0, "gate")
                    elif action == "half" and held:
                        for s in list(held):
                            sell(s, i, 0.5, "gate")
                elif not off and risk_off:
                    risk_off = False
        if i in me:
            off_now = bool(garr[i]) if gate_on else False
            if action == "cash":
                if not off_now:
                    risk_off = False
                    do_fill(i, True, 1.0)
                # risk-off month-end: stay in cash, skip rebalance entirely
            elif action == "block":
                do_fill(i, not off_now, 1.0)
            elif action == "half":
                do_fill(i, True, 0.5 if off_now else 1.0)
        E = sum(v[0] for v in held.values())
        nav_v[i - ctx.i0] = E + cash
        inv_sum += E / (E + cash)
    if tax:
        settle_tax()                          # settle the final partial fiscal year
        nav_v[-1] = sum(v[0] for v in held.values()) + cash
    nav = pd.Series(nav_v, index=dates[ctx.i0:])
    row = dict(series=series, cons=cons, action=action, freq=freq, n=n,
               exit=(f"{ek}{ep}" if ek else "none"), offset=offset, tax=int(tax),
               cash_y=cash_y, rt=rt,
               fills=st["fills"], donch_exits=st["donch"], gate_events=st["gate_events"],
               cost_pct=round(st["cost"] * 100, 1), tax_pct=round(st["tax"] * 100, 1),
               avg_inv=round(inv_sum / len(nav_v), 2), secs=round(time.time() - t0, 1))
    for w, a, b in WINDOWS:
        for k, v in wstats(nav, a, b).items():
            row[f"{w}_{k}"] = v
    row["_nav"] = nav
    return row


# ───────────────────── phase harness ─────────────────────
def _csv_init(path):
    done = set()
    if path.exists():
        with open(path) as f:
            done = {r["label"] for r in csv.DictReader(f)}
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    return done


def _emit(path, label, row):
    row = {k: v for k, v in row.items() if k != "_nav"}
    row["label"] = label
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    print(f"{label:58} waCAGR={row['wa_cagr']:>7} waDD={row['wa_dd']:>7} "
          f"waCal={row['wa_calmar']:>5} w1={row['w1_cagr']:>7} w2={row['w2_cagr']:>7} "
          f"[{row['secs']}s]", flush=True)


def _cell(ctx, path, done, label, **kw):
    for tax in (False, True):
        lbl = f"{label}_tax{int(tax)}"
        if lbl in done:
            continue
        _emit(path, lbl, run(ctx, tax=tax, **kw))


def bench_row(ctx, path, done):
    if "BENCH_NIFTYBEES_tax0" in done:
        return
    nav = ctx.cf[BENCH].loc[ctx.dates[ctx.i0]:]
    row = dict(series=BENCH, cons="buyhold", action="", freq="", n=0, exit="", offset=0,
               tax=0, cash_y=0, rt=0, fills=0, donch_exits=0, gate_events=0,
               cost_pct=0, tax_pct=0, avg_inv=1.0, secs=0)
    for w, a, b in WINDOWS:
        for k, v in wstats(nav, a, b).items():
            row[f"{w}_{k}"] = v
    _emit(path, "BENCH_NIFTYBEES_tax0", row)


def phase_A(ctx):
    path = RES / "phaseA_gates.csv"
    done = _csv_init(path)
    bench_row(ctx, path, done)
    _cell(ctx, path, done, "A_NONE", series="NONE", cons="none")
    for series in GATE_SERIES:
        for cons in CONS:
            _cell(ctx, path, done, f"A_{series}_{cons}", series=series, cons=cons)
    ctx.save_ranks()


def _top_gates(k=2, dd_slack=3.0):
    """Top-k challenger gates from phase A by pre-registered metric (tax rows)."""
    rows = list(csv.DictReader(open(RES / "phaseA_gates.csv")))
    inc = next(r for r in rows if r["label"] == "A_NIFTYBEES_sma100_tax1")
    cand = [r for r in rows if r["tax"] == "1" and r["wa_cagr"] and r["label"] != "BENCH_NIFTYBEES_tax0"
            and float(r["wa_dd"]) >= float(inc["wa_dd"]) - dd_slack]
    cand.sort(key=lambda r: -float(r["wa_cagr"]))
    out = []
    for r in cand:
        g = (r["series"], r["cons"])
        if g != ("NIFTYBEES", "sma100") and g[0] != "NONE" and g not in out:
            out.append(g)
        if len(out) >= k:
            break
    return out


def phase_B(ctx):
    path = RES / "phaseB_actions.csv"
    done = _csv_init(path)
    gates = [("NIFTYBEES", "sma100")] + _top_gates(2)
    print("phase B gates:", gates, flush=True)
    for series, cons in gates:
        for action in ("cash", "block", "half"):
            for freq in ("daily", "weekly", "monthly"):
                _cell(ctx, path, done, f"B_{series}_{cons}_{action}_{freq}",
                      series=series, cons=cons, action=action, freq=freq)
    ctx.save_ranks()


def phase_C(ctx):
    path = RES / "phaseC_slots_exits.csv"
    done = _csv_init(path)
    gates = [("NIFTYBEES", "sma100")] + _top_gates(2)
    exits = [("donch", 10), ("donch", 15), ("donch", 20), ("donch", 25), None,
             ("atr", 20), ("smatr", 50), ("smatr", 100)]
    for series, cons in gates:
        for n in (5, 8, 10, 12, 16):
            for ex in exits:
                exl = f"{ex[0]}{ex[1]}" if ex else "none"
                _cell(ctx, path, done, f"C_{series}_{cons}_n{n}_{exl}",
                      series=series, cons=cons, n=n, exit=ex)
    ctx.save_ranks()


def phase_D(ctx, finalists):
    """finalists: list of (tag, kwargs-dict). Incumbent must be included by caller."""
    path = RES / "phaseD_robustness.csv"
    done = _csv_init(path)
    py = {}
    for tag, kw in finalists:
        for off in range(12):
            for txm in (False, True):
                lbl = f"D_{tag}_off{off}_tax{int(txm)}"
                if lbl not in done:
                    r = run(ctx, tax=txm, offset=off, **kw)
                    if off == 0 and txm:
                        nav = r["_nav"]
                        nav.to_csv(RES / f"nav_{tag}_tax1.csv")
                        last = nav.groupby(nav.index.year).last()
                        yr = last.pct_change() * 100
                        yr.iloc[0] = (last.iloc[0] / nav.iloc[0] - 1) * 100
                        py[tag] = yr.round(1)
                    _emit(path, lbl, r)
        for lbl2, kw2 in ((f"D_{tag}_cash5_tax1", dict(cash_y=0.05)),
                          (f"D_{tag}_rt50_tax1", dict(rt=0.005))):
            if lbl2 not in done:
                _emit(path, lbl2, run(ctx, tax=True, **kw, **kw2))
    if py:
        pd.DataFrame(py).to_csv(RES / "phaseD_peryear.csv")
    ctx.save_ranks()


def default_finalists():
    """Hand-picked after phases A-C (2026-09-03): incumbent, the two sub-threshold
    action/frequency improvers, the n5 concentration challenger (+ its block twin),
    and the donch20 neighbour."""
    g = dict(series="NIFTYBEES", cons="sma100")
    return [
        ("INC_cash_n8_d15", dict(**g, n=8, exit=("donch", 15))),
        ("BLOCK_n8_d15", dict(**g, action="block", n=8, exit=("donch", 15))),
        ("CASHMONTHLY_n8_d15", dict(**g, freq="monthly", n=8, exit=("donch", 15))),
        ("CASH_n5_d15", dict(**g, n=5, exit=("donch", 15))),
        ("BLOCK_n5_d15", dict(**g, action="block", n=5, exit=("donch", 15))),
        ("CASH_n8_d20", dict(**g, n=8, exit=("donch", 20))),
    ]


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    ctx = Ctx()
    if phase == "smoke":
        path = RES / "smoke.csv"
        done = _csv_init(path)
        bench_row(ctx, path, done)
        _cell(ctx, path, done, "SMOKE_incumbent", series="NIFTYBEES", cons="sma100")
        _cell(ctx, path, done, "SMOKE_nogate_nodonch", series="NONE", cons="none", exit=None)
        ctx.save_ranks()
    elif phase == "A":
        phase_A(ctx)
    elif phase == "B":
        phase_B(ctx)
    elif phase == "C":
        phase_C(ctx)
    elif phase == "D":
        phase_D(ctx, default_finalists())
    print("PHASE", phase, "DONE", flush=True)


if __name__ == "__main__":
    main()
