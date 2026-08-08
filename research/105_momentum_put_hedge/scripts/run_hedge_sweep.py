"""Research 105 — Put-hedge overlay instead of the cash-exit gate, on the live momentum book.

At the WEEKLY risk-off gate, instead of liquidating to cash, stay invested and buy a NIFTY put structure
sized to the portfolio. Sweeps structure x moneyness x tenor x hedge-ratio x exit. NIFTY EOD option closes
(OI>0). Tracks STCG realized (the churn cost the hedge is meant to avoid) and reports NET-OF-STCG CAGR.

Arms: cash_exit (current live baseline) | hold (naked control) | hedge (the study).
"""
from __future__ import annotations
import importlib.util, sqlite3, csv, sys, time
from pathlib import Path
from datetime import date as _date
import numpy as np, pandas as pd, logging

logging.disable(logging.WARNING)
P62 = Path("/home/arun/quantifyd/research/62_momentum_etf_subselect/scripts/62_mom30_subselect.py")
P75 = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
_a = importlib.util.spec_from_file_location("r62", str(P62)); r62 = importlib.util.module_from_spec(_a); _a.loader.exec_module(r62)
_b = importlib.util.spec_from_file_location("m75", str(P75)); m75 = importlib.util.module_from_spec(_b); _b.loader.exec_module(m75)
BENCH = r62.BENCH
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
RES = Path("/home/arun/quantifyd/research/105_momentum_put_hedge/results"); RES.mkdir(parents=True, exist_ok=True)
OUT = RES / "hedge_sweep.csv"
START = pd.Timestamp("2011-01-01")            # NIFTY monthly options start 2011
RT_LEG, CASH_Y, STCG = 0.0015, 0.065, 0.20
OPT_SLIP = 0.003
con = sqlite3.connect(DB); con.execute("PRAGMA busy_timeout=60000")

# ───────────────────────── option helpers ─────────────────────────
EXPS = sorted(r[0] for r in con.execute(
    "SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date>='2011-01-01'"))
_EXPD = [pd.Timestamp(e) for e in EXPS]
_pick_cache, _ser_cache = {}, {}


def pick_expiry(d, lo, hi, want):
    """Nearest expiry with DTE in [lo,hi], preferring `want` days out."""
    best = None
    for e, ed in zip(EXPS, _EXPD):
        dte = (ed - d).days
        if lo <= dte <= hi:
            score = abs(dte - want)
            if best is None or score < best[0]:
                best = (score, e)
    return best[1] if best else None


def pick_strike(d, E, target):
    key = (d.strftime("%Y-%m-%d"), E, int(round(target / 50.0)))
    if key in _pick_cache:
        return _pick_cache[key]
    r = con.execute(
        "SELECT strike,close FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date=? AND expiry_date=? "
        "AND option_type='PE' AND close>0 AND open_interest>0 ORDER BY ABS(strike-?) LIMIT 1",
        (d.strftime("%Y-%m-%d"), E, float(target))).fetchone()
    _pick_cache[key] = r
    return r


def series(E, K):
    key = (E, float(K))
    if key not in _ser_cache:
        _ser_cache[key] = {td: cl for td, cl in con.execute(
            "SELECT trade_date,close FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND strike=? "
            "AND option_type='PE' AND close>0", (E, float(K)))}
    return _ser_cache[key]


# ───────────────────────── data ─────────────────────────
print("loading panel ...", flush=True)
close, tv = m75.load()
idx_all = close.index
spot = con.execute("SELECT date,open,high,low,close FROM market_data_unified "
                   "WHERE symbol='NIFTY50' AND timeframe='day' AND close>0 ORDER BY date").fetchall()
S = {pd.Timestamp(d): c for d, o, h, l, c in spot}
SH = {pd.Timestamp(d): (h, l, c) for d, o, h, l, c in spot}

# NIFTY SuperTrend(10,3) — bullish flip used as an optional hedge exit
_sd = sorted(SH)
_atr, _tr, stb = {}, 0.0, {}
for i in range(1, len(_sd)):
    d, p = _sd[i], _sd[i - 1]
    h, l, c = SH[d]; pc = SH[p][2]
    t = max(h - l, abs(h - pc), abs(l - pc))
    _tr = t if i == 1 else (_tr * 9 + t) / 10
    _atr[d] = _tr
_up, _dn, _bull = {}, {}, {}
prev_bull, prev_ub, prev_lb = True, None, None
for d in _sd:
    if d not in _atr:
        continue
    h, l, c = SH[d]; mid = (h + l) / 2; a = 3 * _atr[d]
    ub, lb = mid + a, mid - a
    if prev_ub is not None:
        ub = min(ub, prev_ub) if c <= prev_ub else ub
        lb = max(lb, prev_lb) if c >= prev_lb else lb
    bull = c > (prev_ub if prev_ub is not None else ub) if not prev_bull else c >= lb
    _bull[d] = bull
    prev_bull, prev_ub, prev_lb = bull, ub, lb
ST_BULL = _bull

idx = idx_all[(idx_all >= START) & (idx_all <= pd.Timestamp("2026-08-01"))]
idx = idx[[d in S for d in idx]]
ME = sorted(set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values))
_iso = idx.isocalendar()
WK = set(pd.Series(idx, index=idx).groupby([_iso.year.values, _iso.week.values]).last().values)
ROLL_LOW = close.rolling(15, min_periods=15).min().shift(1)
NBX = close[BENCH].ffill()

print("precomputing monthly rankings (once, reused by every config) ...", flush=True)
ETF = {}
for d in ME:
    dt = pd.Timestamp(d)
    try:
        ETF[dt] = r62.eligible_etf(close, tv, dt, "rsblend", 30) or []
    except Exception:
        ETF[dt] = []
print(f"  {sum(1 for v in ETF.values() if v)}/{len(ETF)} month-ends ranked", flush=True)
ME = set(pd.Timestamp(x) for x in ME)


# ───────────────────────── the book ─────────────────────────
def run(mode, struct=None, moneyness=0.0, width=0.05, tenor="monthly", ratio=1.0,
        st_trail=False, pt=False, start=START, derisk_frac=0.0):
    """mode: cash_exit | hold | hedge.  Returns metrics dict."""
    lo, hi, want = (22, 45, 30) if tenor == "monthly" else (3, 12, 7)
    ix = idx[idx >= start]
    cash, held, prev, derisked = 1.0, {}, None, False
    hedge = None; st_blocked = False; dr_done = False
    nav_pre, nav_post = [], []          # pre-tax and net-of-STCG curves
    tax_paid = 0.0
    stt = dict(exits=0, stcg=0.0, prem=0.0, hpnl=0.0, hedges=0, rolls=0, donch=0)

    def E():
        return sum(v[0] for v in held.values())

    def sell(s, d, frac=1.0):
        """Sell `frac` of a holding; realize STCG on the gain portion."""
        nonlocal cash, tax_paid
        v, basis, bd = held[s]
        cut = v * frac
        g = (v - basis) * frac
        if g > 0 and (d - bd).days < 365:
            t = g * STCG; stt['stcg'] += t; tax_paid += t
        cash += cut * (1 - RT_LEG)
        if frac >= 0.999:
            del held[s]
        else:
            held[s] = [v - cut, basis * (1 - frac), bd]

    def fund(amount, d):
        """Raise `amount` of cash: use idle cash, else trim holdings pro-rata (realistic — the book runs
        ~fully invested, so a hedge premium must be funded)."""
        nonlocal cash
        if cash >= amount:
            cash -= amount; return amount
        need = amount - max(0.0, cash); cash = max(0.0, cash)
        eq = E()
        if eq <= 0:
            return 0.0
        frac = min(0.98, need / eq / (1 - RT_LEG))
        for s in list(held):
            sell(s, d, frac)
        got = min(amount, cash); cash -= got
        return got

    def hedge_value(d):
        if not hedge:
            return 0.0
        v = 0.0
        for (K, sgn, rt, ser, last) in hedge['legs']:
            px = ser.get(d.strftime("%Y-%m-%d"))
            if px is None:
                px = last[0]
            else:
                last[0] = px
            v += sgn * rt * px
        # no clamp — the real quoted marks ARE the value (a ratio structure's tail loss must show)
        return v * hedge['units']

    def open_hedge(d):
        nonlocal hedge, cash
        eq = E()
        if eq <= 0 or d not in S:
            return
        Ex = pick_expiry(d, lo, hi, want)
        if not Ex:
            return
        sp = S[d]
        legs_spec = [(moneyness, +1, 1.0)]
        if struct == "spread":
            legs_spec.append((moneyness - width, -1, 1.0))
        elif struct == "ratio12":
            legs_spec.append((moneyness - 0.05, -1, 2.0))
        legs, net = [], 0.0
        for (m, sgn, rt) in legs_spec:
            r = pick_strike(d, Ex, sp * (1 + m))
            if not r:
                return
            K, px = r
            legs.append((K, sgn, rt, series(Ex, K), [px]))
            net += sgn * rt * px
        if net <= 0:                         # need a net debit structure
            return
        units = ratio * eq / sp
        cost = net * units * (1 + OPT_SLIP)
        got = fund(cost, d)
        if got < cost * 0.9:                 # couldn't fund it
            return
        hedge = dict(legs=legs, units=units, expiry=Ex, entry=net, entry_cost=cost)
        stt['prem'] += cost; stt['hedges'] += 1

    def close_hedge(d, roll=False):
        nonlocal hedge, cash
        if not hedge:
            return
        val = hedge_value(d) * (1 - OPT_SLIP)
        cash += val
        stt['hpnl'] += val - hedge['entry_cost']
        hedge = None
        if roll:
            stt['rolls'] += 1

    def do_fill(d):
        """Monthly rebalance — rotate-only, let winners run (matches the live book)."""
        nonlocal cash
        etf = ETF.get(d) or []
        if not etf:
            return
        top, buf = etf[:8], set(etf[:22])
        for s in list(held):
            if s not in buf:
                sell(s, d)
        cand = (list(held) + [x for x in top if x not in held])[:8]
        target = [s for s in cand if s in close.columns and pd.notna(close.at[d, s])]
        per = (E() + cash) / 8
        for s in target:
            if s not in held and cash > 0:
                buy = min(per, cash)
                if buy <= 0:
                    break
                held[s] = [buy * (1 - RT_LEG), buy, d]; cash -= buy

    for d in ix:
        # mark stocks
        if held and prev is not None:
            for s, v in held.items():
                p1 = close.at[d, s] if s in close.columns else np.nan
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    v[0] += v[0] * (p1 / p0 - 1.0)
        if cash > 0:
            cash *= (1 + CASH_Y) ** (1 / 252)

        # per-stock Donchian EOD stop
        if held:
            lows = ROLL_LOW.loc[d]
            for s in list(held):
                p1 = close.at[d, s] if s in close.columns else np.nan
                ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                    sell(s, d); stt['donch'] += 1

        # hedge maintenance: roll at expiry, optional ST-trail / take-profit exits
        if hedge:
            if d >= pd.Timestamp(hedge['expiry']):
                still_off = NBX.loc[d] < NBX.loc[:d].tail(100).mean()
                close_hedge(d, roll=True)
                if still_off and mode == "hedge" and not st_blocked:
                    open_hedge(d)
            elif st_trail and ST_BULL.get(d, False):
                close_hedge(d); st_blocked = True
            elif pt and hedge_value(d) >= 2 * hedge['entry_cost']:
                close_hedge(d); st_blocked = True

        # weekly gate
        if d in WK:
            b = NBX.loc[:d].dropna()
            risk_off = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if risk_off:
                if mode == "cash_exit":
                    if held:
                        for s in list(held):
                            sell(s, d)
                        stt['exits'] += 1
                    derisked = True
                elif mode == "hedge":
                    if derisk_frac > 0 and not dr_done and held:   # hybrid: partial de-risk, hedge the rest
                        for s in list(held):
                            sell(s, d, derisk_frac)
                        dr_done = True
                    if hedge is None and not st_blocked:
                        open_hedge(d)
            else:
                derisked = False; st_blocked = False; dr_done = False
                if hedge:
                    close_hedge(d)

        if d in ME and not derisked:
            do_fill(d)

        hv = hedge_value(d) if hedge else 0.0
        nav_pre.append((d, E() + cash + hv))
        nav_post.append((d, E() + cash + hv - tax_paid))
        prev = d

    def blk(pairs):
        n = pd.DataFrame(pairs, columns=["d", "v"]).set_index("d")["v"]
        yrs = (n.index[-1] - n.index[0]).days / 365.25
        cagr = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
        dr = n.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
        dd = ((n - n.cummax()) / n.cummax()).min()
        return cagr * 100, dd * 100, sh, (cagr / abs(dd) if dd < 0 else 0.0), n.iloc[-1]

    c1, d1, s1, k1, f1 = blk(nav_pre)
    c2, d2, s2, k2, f2 = blk(nav_post)
    return dict(cagr=round(c1, 1), maxdd=round(d1, 1), sharpe=round(s1, 2), calmar=round(k1, 2),
                net_cagr=round(c2, 1), net_calmar=round(k2, 2), mult=round(f1, 1),
                full_exits=stt['exits'], stcg=round(stt['stcg'] * 100, 1),
                prem=round(stt['prem'] * 100, 1), hpnl=round(stt['hpnl'] * 100, 1),
                hedges=stt['hedges'], rolls=stt['rolls'])


FIELDS = ["config", "mode", "struct", "moneyness", "tenor", "ratio", "exit", "start",
          "cagr", "net_cagr", "maxdd", "sharpe", "calmar", "net_calmar", "mult",
          "full_exits", "stcg", "prem", "hpnl", "hedges", "rolls"]


def main():
    t0 = time.time()
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {r["config"] for r in csv.DictReader(f)}
        print(f"resuming — {len(done)} configs done", flush=True)
    else:
        with open(OUT, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    grid = [("A0_cash_exit", dict(mode="cash_exit")),
            ("A1_hold_naked", dict(mode="hold"))]
    for tenor in ("monthly", "weekly"):
        st0 = START if tenor == "monthly" else pd.Timestamp("2019-02-01")
        for mny, mtag in ((0.02, "ITM2"), (0.0, "ATM"), (-0.02, "OTM2"), (-0.05, "OTM5")):
            for ratio in (0.5, 1.0):
                grid.append((f"LP_{mtag}_{tenor}_r{ratio}",
                             dict(mode="hedge", struct="long", moneyness=mny, tenor=tenor,
                                  ratio=ratio, start=st0)))
        for mny, mtag in ((0.0, "ATM"), (-0.02, "OTM2")):
            for w, wtag in ((0.05, "w5"), (0.10, "w10")):
                grid.append((f"SPR_{mtag}_{wtag}_{tenor}_r1.0",
                             dict(mode="hedge", struct="spread", moneyness=mny, width=w,
                                  tenor=tenor, ratio=1.0, start=st0)))
    # exit-variant probes on the ATM monthly long put
    grid.append(("LP_ATM_monthly_r1.0_STtrail",
                 dict(mode="hedge", struct="long", moneyness=0.0, tenor="monthly", ratio=1.0, st_trail=True)))
    grid.append(("LP_ATM_monthly_r1.0_PT100",
                 dict(mode="hedge", struct="long", moneyness=0.0, tenor="monthly", ratio=1.0, pt=True)))
    grid.append(("RATIO12_ATM_monthly_r1.0_FLAGGED",
                 dict(mode="hedge", struct="ratio12", moneyness=0.0, tenor="monthly", ratio=1.0)))

    print(f"\n{len(grid)} configs\n{'config':>34} {'CAGR':>6} {'netCAGR':>8} {'MaxDD':>7} "
          f"{'Calmar':>6} {'netCal':>7} {'exits':>6} {'STCG':>6} {'prem':>6}", flush=True)
    for lbl, kw in grid:
        if lbl in done:
            continue
        ts = time.time()
        try:
            r = run(**kw)
        except Exception as e:
            print(f"  {lbl:>32} FAILED: {e}", flush=True); continue
        row = dict(config=lbl, mode=kw.get("mode"), struct=kw.get("struct", ""),
                   moneyness=kw.get("moneyness", ""), tenor=kw.get("tenor", ""),
                   ratio=kw.get("ratio", ""),
                   exit=("ST" if kw.get("st_trail") else "PT" if kw.get("pt") else "gate"),
                   start=str(kw.get("start", START).date()), **r)
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"{lbl:>34} {r['cagr']:>5.1f}% {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% "
              f"{r['calmar']:>6.2f} {r['net_calmar']:>7.2f} {r['full_exits']:>6} "
              f"{r['stcg']:>6.1f} {r['prem']:>6.1f}  [{time.time()-ts:.0f}s]", flush=True)
        sys.stdout.flush()
    print(f"\nDone {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
