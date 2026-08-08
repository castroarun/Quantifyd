"""Research 106e — REDO the smallcap regime study on the REAL NIFTY Smallcap 250 index (now downloaded,
2011-2026), replacing the traded-value proxy that was only 68% accurate and got the finding retracted.

1. Real signal: NIFTY SMLCAP 250 / NIFTY 50, 6-month relative momentum (causal).
2. Grade the old proxy against it over the FULL 15 years.
3. Regime split: N200 book vs N500 book in smallcap-led vs largecap-led months.
4. Regime-switched book, now on the true signal — does the retracted finding survive?
"""
import importlib.util, sqlite3, csv, sys
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "m75", "/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)
BENCH, EXCLUDE = m75.BENCH, m75.EXCLUDE
RES = Path("/home/arun/quantifyd/research/106_nifty500_universe/results")
START = pd.Timestamp("2011-01-01"); CASH_Y, STCG, RT = 0.065, 0.20, 0.0015
con = sqlite3.connect("/home/arun/quantifyd/backtest_data/market_data.db")


def series(sym):
    return pd.Series({pd.Timestamp(d): c for d, c in con.execute(
        "SELECT date,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0",
        (sym,))}).sort_index()


small = series("NIFTYSMLCAP250"); large = series("NIFTY50")
print(f"REAL NIFTY SMLCAP 250: {small.index[0].date()} .. {small.index[-1].date()} ({len(small)} days)")
print(f"REAL NIFTY 50        : {large.index[0].date()} .. {large.index[-1].date()} ({len(large)} days)\n")
cm = small.index.intersection(large.index)
rs_real = (small[cm] / large[cm]).resample("ME").last()
real6 = rs_real / rs_real.shift(6) - 1
REAL_LEADS = real6 > 0
print(f"REAL signal: smallcaps led {REAL_LEADS.mean()*100:.0f}% of months (2011-2026)")

close, tv = m75.load()
idx = close.index[(close.index >= START) & (close.index <= pd.Timestamp("2026-08-01"))]
ME = [pd.Timestamp(x) for x in sorted(set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values))]
_iso = idx.isocalendar()
WK = set(pd.Series(idx, index=idx).groupby([_iso.year.values, _iso.week.values]).last().values)
ROLL_LOW = close.rolling(15, min_periods=15).min().shift(1); NBX = close[BENCH].ffill()
print("precomputing scores/ranks ...", flush=True)
SNAP = {}
LCp, SCp, prev = [], [], None
for d in ME:
    h = close.loc[:d].ffill()
    if len(h) <= 253:
        continue
    adv = tv.loc[:d].tail(126).median().dropna()
    adv = adv[[s for s in adv.index if s not in EXCLUDE and s != BENCH]]
    rank = adv.sort_values(ascending=False)
    sc = None
    for L, w in ((126, 0.5), (252, 0.5)):
        p0, p1 = h.iloc[-L - 1], h.iloc[-1]
        r = (p1 / p0) / (p1[BENCH] / p0[BENCH]) * w
        sc = r if sc is None else sc.add(r, fill_value=np.nan)
    ok = [s for s in sc.index if s in rank.index and pd.notna(sc[s])]
    SNAP[d] = (sc[ok], rank)
    if prev is not None:
        for b, store in ((rank.index[:100], LCp), (rank.index[300:500], SCp)):
            rr = [close.at[d, s] / close.at[prev, s] - 1 for s in b
                  if s in close.columns and pd.notna(close.at[prev, s]) and pd.notna(close.at[d, s])
                  and close.at[prev, s] > 0]
            store.append((d, float(np.mean(rr)) if rr else 0.0))
    prev = d
rs_prox = (1 + pd.Series(dict(SCp))).cumprod() / (1 + pd.Series(dict(LCp))).cumprod()
prox6 = rs_prox / rs_prox.shift(6) - 1
j = pd.DataFrame({"r": real6, "p": prox6}).dropna()
print(f"\n=== GRADING THE OLD PROXY over the full {len(j)} months ===")
print(f"  direction agreement: {(((j.r>0)==(j.p>0)).mean()*100):.0f}%   correlation: {j.r.corr(j.p):+.2f}")


def run(lo, hi, hold, buffer, pool, switch=None):
    cash, held, prev_, derisked, tax = 1.0, {}, None, False, 0.0
    pre, post = [], []

    def E():
        return sum(v[0] for v in held.values())

    def sell(s, d):
        nonlocal cash, tax
        v, b, bd = held.pop(s)
        if v > b and (d - bd).days < 365:
            tax += (v - b) * STCG
        cash += v * (1 - RT)
    for d in idx:
        if held and prev_ is not None:
            for s, v in held.items():
                p1 = close.at[d, s] if s in close.columns else np.nan
                p0 = close.at[prev_, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    v[0] += v[0] * (p1 / p0 - 1.0)
        if cash > 0:
            cash *= (1 + CASH_Y) ** (1 / 252)
        if held:
            lows = ROLL_LOW.loc[d]
            for s in list(held):
                p1 = close.at[d, s] if s in close.columns else np.nan
                ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                    sell(s, d)
        if d in WK:
            b = NBX.loc[:d].dropna()
            if len(b) >= 100 and b.iloc[-1] < b.tail(100).mean():
                for s in list(held):
                    sell(s, d)
                derisked = True
            else:
                derisked = False
        if d in SNAP and not derisked:
            L_, H_, HD, BF, PL = (switch.get(d, (lo, hi, hold, buffer, pool)) if switch
                                  else (lo, hi, hold, buffer, pool))
            sc, rank = SNAP[d]
            sub = sc[[s for s in rank.index[L_:H_] if s in sc.index]]
            etf = list(sub.sort_values(ascending=False).index[:PL])
            if etf:
                top, buf = etf[:HD], set(etf[:BF])
                for s in list(held):
                    if s not in buf:
                        sell(s, d)
                cand = (list(held) + [x for x in top if x not in held])[:HD]
                tgt = [s for s in cand if s in close.columns and pd.notna(close.at[d, s])]
                per = (E() + cash) / HD
                for s in tgt:
                    if s not in held and cash > 0:
                        buy = min(per, cash)
                        if buy > 0:
                            held[s] = [buy * (1 - RT), buy, d]; cash -= buy
        pre.append((d, E() + cash)); post.append((d, E() + cash - tax)); prev_ = d
    return pre, post


def stats(rows):
    n = pd.DataFrame(rows, columns=["d", "v"]).set_index("d")["v"]
    y = (n.index[-1] - n.index[0]).days / 365.25
    c = (n.iloc[-1] / n.iloc[0]) ** (1 / y) - 1
    dr = n.pct_change().dropna()
    dd = ((n - n.cummax()) / n.cummax()).min()
    return dict(cagr=c * 100, dd=dd * 100, cal=(c / abs(dd) if dd < 0 else 0),
                sharpe=dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0), n


print("\nrunning N200 (8/22/30) and N500 (16/44/60) ...", flush=True)
_, p200 = run(0, 200, 8, 22, 30)
_, p500 = run(0, 500, 16, 44, 60)
s200, n200 = stats(p200); s500, n500 = stats(p500)
m200 = n200.resample("ME").last().pct_change().dropna()
m500 = n500.resample("ME").last().pct_change().dropna()
cmn = m200.index.intersection(m500.index)
lead = REAL_LEADS.reindex(cmn, method="ffill").fillna(False)
print(f"\n=== REGIME SPLIT ON THE REAL SIGNAL (monthly, annualised) ===")
print(f"{'regime':>22} {'months':>7} {'N200':>9} {'N500':>9} {'N500-N200':>11}")
for lbl, mask in (("SMALLCAPS leading", lead), ("LARGECAPS leading", ~lead)):
    a, b = m200[cmn][mask.values], m500[cmn][mask.values]
    if len(a) < 3:
        continue
    print(f"{lbl:>22} {len(a):>7} {a.mean()*1200:>8.1f}% {b.mean()*1200:>8.1f}% {(b.mean()-a.mean())*1200:>+10.1f}%")

sw = {}
for d in SNAP:
    v = REAL_LEADS.reindex([d], method="ffill")
    f = bool(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else False
    sw[d] = (0, 500, 16, 44, 60) if f else (0, 200, 8, 22, 30)
_, psw = run(0, 200, 8, 22, 30, switch=sw)
ssw, nsw = stats(psw)
print(f"\n=== FULL PERIOD (net of STCG), REAL-signal switch ===")
for lbl, s in (("N200 (live)", s200), ("N500 16/44/60", s500), ("REGIME-SWITCHED (real)", ssw)):
    print(f"  {lbl:>24}: CAGR {s['cagr']:>5.1f}%  MaxDD {s['dd']:>6.1f}%  Calmar {s['cal']:>4.2f}  Sharpe {s['sharpe']:.2f}")
print(f"\ncurrent real 6m smallcap RS: {real6.dropna().iloc[-1]*100:+.1f}% -> "
      f"{'SMALLCAPS LEADING' if real6.dropna().iloc[-1]>0 else 'LARGECAPS LEADING'}")
print("done", flush=True)
