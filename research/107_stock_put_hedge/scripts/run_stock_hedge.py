"""Research 107 (Ext 3) — hedge each HELD STOCK with its own single-stock PUT, instead of an index put.

Rationale: the index hedge failed because a NIFTY-notional put only neutralises the index component,
while the 8-stock momentum book falls ~2x the index (beta + idiosyncratic). Hedging each position with
its OWN put removes that mismatch entirely.

Reality constraints (all modelled, not assumed away):
  - Single-stock options in India are MONTHLY only (no weeklies) — and monthly was the tenor that failed
    for the index hedge, so this is a genuine test of whether removing the beta gap rescues it.
  - Only F&O names have options; many momentum picks (mid-caps) do not -> COVERAGE is measured, not assumed.
  - Stock options are far less liquid than index -> heavier slippage (2% of premium, vs 0.3% for index).
  - Prices are EOD closes with open_interest > 0 (binding liquidity rule).
"""
import importlib.util, sqlite3, csv, sys
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "m75", "/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)
spec2 = importlib.util.spec_from_file_location(
    "r62", "/home/arun/quantifyd/research/62_momentum_etf_subselect/scripts/62_mom30_subselect.py")
r62 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(r62)
BENCH = m75.BENCH
RES = Path("/home/arun/quantifyd/research/107_stock_put_hedge/results"); RES.mkdir(parents=True, exist_ok=True)
START = pd.Timestamp("2016-01-01")          # single-stock option data starts ~2016
RT_LEG, CASH_Y, STCG, OPT_SLIP = 0.0015, 0.065, 0.20, 0.02   # 2% option slippage (stock opts are wide)
con = sqlite3.connect("/home/arun/quantifyd/backtest_data/market_data.db")
con.execute("PRAGMA busy_timeout=60000")

print("loading panel ...", flush=True)
close, tv = m75.load()
idx = close.index[(close.index >= START) & (close.index <= pd.Timestamp("2026-08-01"))]
ME = [pd.Timestamp(x) for x in sorted(set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values))]
_iso = idx.isocalendar()
WK = set(pd.Series(idx, index=idx).groupby([_iso.year.values, _iso.week.values]).last().values)
ROLL_LOW = close.rolling(15, min_periods=15).min().shift(1); NBX = close[BENCH].ffill()
print("precomputing monthly rankings ...", flush=True)
ETF = {}
for d in ME:
    try:
        ETF[d] = r62.eligible_etf(close, tv, d, "rsblend", 30) or []
    except Exception:
        ETF[d] = []
print(f"  {sum(1 for v in ETF.values() if v)}/{len(ETF)} ranked", flush=True)

# ── single-stock option helpers ──
_exp_cache, _pick_cache, _ser_cache = {}, {}, {}


def expiries(sym):
    if sym not in _exp_cache:
        _exp_cache[sym] = sorted(r[0] for r in con.execute(
            "SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? AND expiry_date>=?",
            (sym, "2016-01-01")))
    return _exp_cache[sym]


def pick_expiry(sym, d, lo=3, hi=70):
    best = None
    for e in expiries(sym):
        dte = (pd.Timestamp(e) - d).days
        if lo <= dte <= hi:
            sc = abs(dte - 30)
            if best is None or sc < best[0]:
                best = (sc, e)
    return best[1] if best else None


def pick_put(sym, d, E, target):
    key = (sym, d.strftime("%Y-%m-%d"), E, int(round(target / 5.0)))
    if key not in _pick_cache:
        _pick_cache[key] = con.execute(
            "SELECT strike,close FROM nse_options_bhav WHERE symbol=? AND trade_date=? AND expiry_date=? "
            "AND option_type='PE' AND close>0 AND open_interest>0 ORDER BY ABS(strike-?) LIMIT 1",
            (sym, d.strftime("%Y-%m-%d"), E, float(target))).fetchone()
    return _pick_cache[key]


def put_series(sym, E, K):
    key = (sym, E, float(K))
    if key not in _ser_cache:
        _ser_cache[key] = {td: cl for td, cl in con.execute(
            "SELECT trade_date,close FROM nse_options_bhav WHERE symbol=? AND expiry_date=? AND strike=? "
            "AND option_type='PE' AND close>0", (sym, E, float(K)))}
    return _ser_cache[key]


def run(mode, ratio=1.0, moneyness=0.0):
    """mode: cash_exit | hold | stockhedge"""
    ix = idx
    cash, held, prev, derisked, tax = 1.0, {}, None, False, 0.0
    hedges = {}                      # sym -> dict(K,E,units,ser,last,cost)
    nav_pre, nav_post = [], []
    st = dict(tried=0, covered=0, prem=0.0, opened=0, rolls=0,
              f_noexp=0, f_nostrike=0, f_nocash=0)

    def E_():
        return sum(v[0] for v in held.values())

    def hv_all(d):
        tot = 0.0
        for s, h in hedges.items():
            px = h['ser'].get(d.strftime("%Y-%m-%d"))
            if px is None:
                px = h['last'][0]
            else:
                h['last'][0] = px
            tot += px * h['units']
        return tot

    def sell(s, d, frac=1.0):
        nonlocal cash, tax
        v, b, bd = held[s]; cut = v * frac; g = (v - b) * frac
        if g > 0 and (d - bd).days < 365:
            tax += g * STCG
        cash += cut * (1 - RT_LEG)
        if frac >= 0.999:
            del held[s]
        else:
            held[s] = [v - cut, b * (1 - frac), bd]

    def close_hedge(s, d):
        nonlocal cash
        h = hedges.pop(s, None)
        if not h:
            return
        px = h['ser'].get(d.strftime("%Y-%m-%d"), h['last'][0])
        cash += px * h['units'] * (1 - OPT_SLIP)

    def open_hedge(s, d):
        nonlocal cash
        if s in hedges or s not in held:
            return
        st['tried'] += 1
        E = pick_expiry(s, d)
        if not E:
            st['f_noexp'] += 1; return
        sp = close.at[d, s] if s in close.columns else np.nan
        if pd.isna(sp) or sp <= 0:
            return
        r = pick_put(s, d, E, sp * (1 + moneyness))
        if not r:
            st['f_nostrike'] += 1; return
        K, px = r
        if px <= 0:
            return
        st['covered'] += 1
        units = ratio * held[s][0] / sp
        cost = px * units * (1 + OPT_SLIP)
        if cost > cash:                       # fund by trimming that position
            need = cost - max(0.0, cash)
            fr = min(0.5, need / held[s][0]) if held[s][0] > 0 else 0
            if fr > 0:
                sell(s, d, fr)
            if cost > cash:
                st['f_nocash'] += 1; return
        cash -= cost
        hedges[s] = dict(K=K, E=E, units=units, ser=put_series(s, E, K), last=[px], cost=cost)
        st['prem'] += cost; st['opened'] += 1

    for d in ix:
        if held and prev is not None:
            for s, v in held.items():
                p1 = close.at[d, s] if s in close.columns else np.nan
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    v[0] += v[0] * (p1 / p0 - 1.0)
        if cash > 0:
            cash *= (1 + CASH_Y) ** (1 / 252)
        # Donchian stops (also drop that stock's hedge)
        if held:
            lows = ROLL_LOW.loc[d]
            for s in list(held):
                p1 = close.at[d, s] if s in close.columns else np.nan
                ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                    close_hedge(s, d); sell(s, d)
        # expiry rolls
        for s in list(hedges):
            if d >= pd.Timestamp(hedges[s]['E']):
                gate_off = NBX.loc[d] < NBX.loc[:d].tail(100).mean()
                close_hedge(s, d); st['rolls'] += 1
                if gate_off and mode == "stockhedge" and s in held:
                    open_hedge(s, d)
        if d in WK:
            b = NBX.loc[:d].dropna()
            off = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if off:
                if mode == "cash_exit":
                    for s in list(held):
                        sell(s, d)
                    derisked = True
                elif mode == "stockhedge":
                    for s in list(held):
                        open_hedge(s, d)
            else:
                derisked = False
                for s in list(hedges):
                    close_hedge(s, d)
        if d in ME and not derisked:
            etf = ETF.get(d) or []
            if etf:
                top, buf = etf[:8], set(etf[:22])
                for s in list(held):
                    if s not in buf:
                        close_hedge(s, d); sell(s, d)
                cand = (list(held) + [x for x in top if x not in held])[:8]
                tgt = [s for s in cand if s in close.columns and pd.notna(close.at[d, s])]
                per = (E_() + cash) / 8
                for s in tgt:
                    if s not in held and cash > 0:
                        buy = min(per, cash)
                        if buy > 0:
                            held[s] = [buy * (1 - RT_LEG), buy, d]; cash -= buy
        h = hv_all(d)
        nav_pre.append((d, E_() + cash + h))
        nav_post.append((d, E_() + cash + h - tax))
        prev = d

    def blk(p):
        n = pd.DataFrame(p, columns=["d", "v"]).set_index("d")["v"]
        y = (n.index[-1] - n.index[0]).days / 365.25
        c = (n.iloc[-1] / n.iloc[0]) ** (1 / y) - 1
        dr = n.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        dd = ((n - n.cummax()) / n.cummax()).min()
        return c * 100, dd * 100, sh, (c / abs(dd) if dd < 0 else 0)
    c1, d1, s1, k1 = blk(nav_pre); c2, d2, s2, k2 = blk(nav_post)
    cov = 100 * st['covered'] / st['tried'] if st['tried'] else 0
    return dict(cagr=round(c1, 1), maxdd=round(d1, 1), sharpe=round(s1, 2), calmar=round(k1, 2),
                net_cagr=round(c2, 1), net_calmar=round(k2, 2), coverage=round(cov, 1),
                opened=st['opened'], prem=round(st['prem'] * 100, 1),
                f_noexp=st['f_noexp'], f_nostrike=st['f_nostrike'], f_nocash=st['f_nocash'])


grid = [("A0_cash_exit (live baseline)", dict(mode="cash_exit")),
        ("A1_hold_naked", dict(mode="hold")),
        ("STOCKPUT_ATM_r1.0", dict(mode="stockhedge", ratio=1.0, moneyness=0.0)),
        ("STOCKPUT_ATM_r0.5", dict(mode="stockhedge", ratio=0.5, moneyness=0.0)),
        ("STOCKPUT_OTM5_r1.0", dict(mode="stockhedge", ratio=1.0, moneyness=-0.05)),
        ("STOCKPUT_ITM2_r1.0", dict(mode="stockhedge", ratio=1.0, moneyness=0.02))]
F = ["config", "cagr", "net_cagr", "maxdd", "sharpe", "calmar", "net_calmar", "coverage", "opened", "prem", "f_noexp", "f_nostrike", "f_nocash"]
w = csv.DictWriter(open(RES / "stock_hedge.csv", "w", newline=""), fieldnames=F); w.writeheader()
print(f"\n{'config':>30} {'CAGR':>6} {'netCAGR':>8} {'MaxDD':>7} {'netCal':>7} {'cover%':>7} {'#opn':>5} {'prem':>7}", flush=True)
for lbl, kw in grid:
    try:
        r = run(**kw)
    except Exception as e:
        print(f"  {lbl} FAILED {e}", flush=True); continue
    w.writerow({k: (lbl if k == "config" else r[k]) for k in F})
    print(f"{lbl:>30} {r['cagr']:>5.1f}% {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% "
          f"{r['net_calmar']:>7.2f} {r['coverage']:>6.1f}% {r['opened']:>5} {r['prem']:>7.1f}"
          f"   [fail: noexp {r['f_noexp']}, nostrike {r['f_nostrike']}, nocash {r['f_nocash']}]", flush=True)
    sys.stdout.flush()
print("done", flush=True)
