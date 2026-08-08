"""Export daily NAV curves for N200 vs N500 (no ADV filter) so the tearsheet can slice by date range."""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "m75", "/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)
BENCH, EXCLUDE = m75.BENCH, m75.EXCLUDE
RES = Path("/home/arun/quantifyd/research/106_nifty500_universe/results"); RES.mkdir(parents=True, exist_ok=True)
START = pd.Timestamp("2011-01-01"); CASH_Y, STCG, RT = 0.065, 0.20, 0.0015
close, tv = m75.load()
idx = close.index[(close.index >= START) & (close.index <= pd.Timestamp("2026-08-01"))]
ME = [pd.Timestamp(x) for x in sorted(set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values))]
_iso = idx.isocalendar()
WK = set(pd.Series(idx, index=idx).groupby([_iso.year.values, _iso.week.values]).last().values)
ROLL_LOW = close.rolling(15, min_periods=15).min().shift(1); NBX = close[BENCH].ffill()
SNAP = {}
for d in ME:
    h = close.loc[:d].ffill()
    if len(h) <= 253:
        continue
    adv = tv.loc[:d].tail(126).median()
    sc = None
    for L, w in ((126, 0.5), (252, 0.5)):
        p0, p1 = h.iloc[-L - 1], h.iloc[-1]
        r = (p1 / p0) / (p1[BENCH] / p0[BENCH]) * w
        sc = r if sc is None else sc.add(r, fill_value=np.nan)
    ok = [s for s in sc.index if s not in EXCLUDE and s != BENCH
          and pd.notna(sc[s]) and pd.notna(adv.get(s, np.nan))]
    SNAP[d] = (sc[ok], adv[ok])


def run(band):
    cash, held, prev, derisked, tax = 1.0, {}, None, False, 0.0
    out = []

    def E():
        return sum(v[0] for v in held.values())

    def sell(s, d):
        nonlocal cash, tax
        v, b, bd = held.pop(s)
        if v > b and (d - bd).days < 365:
            tax += (v - b) * STCG
        cash += v * (1 - RT)
    for d in idx:
        if held and prev is not None:
            for s, v in held.items():
                p1 = close.at[d, s] if s in close.columns else np.nan
                p0 = close.at[prev, s] if s in close.columns else np.nan
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
            sc, adv = SNAP[d]
            pool = adv.sort_values(ascending=False).index[:band]
            sub = sc[[s for s in pool if s in sc.index]]
            etf = list(sub.sort_values(ascending=False).index[:30])
            if etf:
                top, buf = etf[:8], set(etf[:22])
                for s in list(held):
                    if s not in buf:
                        sell(s, d)
                cand = (list(held) + [x for x in top if x not in held])[:8]
                tgt = [s for s in cand if s in close.columns and pd.notna(close.at[d, s])]
                per = (E() + cash) / 8
                for s in tgt:
                    if s not in held and cash > 0:
                        buy = min(per, cash)
                        if buy > 0:
                            held[s] = [buy * (1 - RT), buy, d]; cash -= buy
        out.append((d, E() + cash, E() + cash - tax))
        prev = d
    return out


data = {"series": {}}
for band, name in ((200, "Nifty 200"), (500, "Nifty 500")):
    print("running", name, flush=True)
    rows = run(band)
    data["series"][name] = [[d.strftime("%Y-%m-%d"), round(a, 5), round(b, 5)] for d, a, b in rows]
bn = close[BENCH].loc[close.index >= START].dropna(); bn = bn / bn.iloc[0]
data["series"]["NIFTY B&H"] = [[d.strftime("%Y-%m-%d"), round(float(v), 5), round(float(v), 5)]
                               for d, v in bn.items()]
json.dump(data, open(RES / "n200_n500_curves.json", "w"))
print("wrote n200_n500_curves.json", {k: len(v) for k, v in data["series"].items()}, flush=True)
