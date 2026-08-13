"""Deposit-timing test: when new cash arrives, deploy IMMEDIATELY (equal top-up of current holdings if
gate ON, else park) vs PARK till next monthly rebalance. Existing book (rsblend+buffer+Donchian+gate),
semi-annual deposits, 2006-2026. Money-weighted: compare final wealth for the SAME deposit stream."""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
P62 = Path("/home/arun/quantifyd/research/62_momentum_etf_subselect/scripts/62_mom30_subselect.py")
P75 = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
_a = importlib.util.spec_from_file_location("r62", str(P62)); r62 = importlib.util.module_from_spec(_a); _a.loader.exec_module(r62)
_b = importlib.util.spec_from_file_location("m75", str(P75)); m75 = importlib.util.module_from_spec(_b); _b.loader.exec_module(m75)
BENCH = r62.BENCH; START = pd.Timestamp("2006-01-01"); RT_LEG = 0.0015
def run(close, tv, mode, deposits):
    idx = close.index[close.index >= START]
    me = set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)
    iso = idx.isocalendar(); wk = set(pd.Series(idx, index=idx).groupby([iso.year.values, iso.week.values]).last().values)
    nbx = close[BENCH].ffill(); daycash = (1.065)**(1/252)
    roll_low = close.rolling(15, min_periods=15).min().shift(1)
    cash, held, prev, derisked = 1.0, {}, None, False
    def E(): return sum(v[0] for v in held.values())
    def gate_on(d): b = nbx.loc[:d].dropna(); return len(b) < 100 or b.iloc[-1] >= b.tail(100).mean()
    def sell(s):
        nonlocal cash; cash += held[s][0]*(1-RT_LEG); del held[s]
    def do_fill(d, px):
        nonlocal cash, held, derisked
        etf = r62.eligible_etf(close, tv, d, "rsblend", 30)
        if not etf: return
        top = etf[:8]; buf = set(etf[:22])
        for s in list(held):
            if s not in buf: sell(s)
        keep = [s for s in held if s in buf]; add = [s for s in top if s not in keep]
        target = [s for s in (keep + add)[:8] if pd.notna(px.get(s, np.nan))]
        if not target: return
        per = (E()+cash)/8
        for s in target:
            if s not in held:
                buy = min(per, max(0.0, cash))
                if buy <= 0: break
                held[s] = [buy*(1-RT_LEG), buy, d]; cash -= buy
        derisked = False
    navser = []
    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan); p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0: stt[0] += stt[0]*(p1/p0-1.0)
        if cash > 0: cash *= daycash
        if d in deposits:                                  # inject external cash
            amt = deposits[d]
            if mode == "immediate" and held and gate_on(d):
                w = amt/len(held)                          # equal-rupee top-up of current holdings
                for s in held: held[s][0] += w*(1-RT_LEG)
            else:                                          # park (or gate-off) -> cash, deploy next rebalance
                cash += amt
        if held:                                            # Donchian EOD
            lows = roll_low.loc[d]
            for s in list(held):
                p1 = px.get(s, np.nan); ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln: sell(s)
        if d in wk:
            if not gate_on(d):
                for s in list(held): sell(s)
                derisked = True
            else: derisked = False
        if d in me and not derisked: do_fill(d, px)
        navser.append((d, E()+cash)); prev = d
    return pd.DataFrame(navser, columns=["date","nav"]).set_index("date")["nav"]
close, tv = m75.load()
idx = close.index[close.index >= START]
# semi-annual deposits of 0.2 (i.e. 20% of the initial 1.0) on ~Jun/Dec month-ends
me = sorted(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)
dep_days = [pd.Timestamp(x) for x in me if pd.Timestamp(x).month in (6, 12)]
deposits = {d: 0.2 for d in dep_days}
tot_dep = 1.0 + sum(deposits.values())
print(f"Deposit-timing test — existing book, 2006-2026. Start 1.0 + {len(deposits)} semi-annual deposits of 0.2 = {tot_dep:.1f} total capital in.\n")
for mode in ["immediate", "park"]:
    nav = run(close, tv, mode, deposits)
    fin = nav.iloc[-1]
    dd = ((nav-nav.cummax())/nav.cummax()).min()
    print(f"  {mode:>10}: final wealth {fin:.1f}  (vs {tot_dep:.1f} deposited)  maxDD {dd*100:.1f}%")
