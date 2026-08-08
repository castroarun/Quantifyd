"""Research 106c — TEST ARUN'S THESIS: does the Nifty-500 book beat Nifty-200 specifically when
SMALLCAPS lead (early-bull after consolidation), even if it loses on the full-period average?

Method:
 1. Build smallcap vs largecap leadership from the price panel (tv-rank buckets: 1-100 = large,
    301-500 = small), no external index needed -> survivorship-free and available back to 2011.
 2. Run both books daily (N200 hold8/buf22 = live; N500 hold16/buf44 = best proportional config).
 3. Split history into SMALLCAP-LEADING vs LARGECAP-LEADING regimes and compare the books in each.
 4. Test a REGIME-SWITCHED book that holds N500 when smallcaps lead, else N200.
"""
import importlib.util, csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "m75", "/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)
BENCH, EXCLUDE = m75.BENCH, m75.EXCLUDE
RES = Path("/home/arun/quantifyd/research/106_nifty500_universe/results")
START = pd.Timestamp("2011-01-01"); CASH_Y, STCG, RT = 0.065, 0.20, 0.0015
close, tv = m75.load()
idx = close.index[(close.index >= START) & (close.index <= pd.Timestamp("2026-08-01"))]
ME = [pd.Timestamp(x) for x in sorted(set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values))]
_iso = idx.isocalendar()
WK = set(pd.Series(idx, index=idx).groupby([_iso.year.values, _iso.week.values]).last().values)
ROLL_LOW = close.rolling(15, min_periods=15).min().shift(1); NBX = close[BENCH].ffill()

# ── 1. smallcap vs largecap leadership (tv-rank buckets, rebuilt monthly) ──
print("building smallcap/largecap leadership series ...", flush=True)
SNAP, LC, SC = {}, [], []
lc_i, sc_i = 1.0, 1.0
prev_me = None
for d in ME:
    h = close.loc[:d].ffill()
    if len(h) <= 253:
        continue
    adv = tv.loc[:d].tail(126).median().dropna()
    adv = adv[[s for s in adv.index if s not in EXCLUDE and s != BENCH]]
    rank = adv.sort_values(ascending=False)
    sc_blend = None
    for L, w in ((126, 0.5), (252, 0.5)):
        p0, p1 = h.iloc[-L - 1], h.iloc[-1]
        r = (p1 / p0) / (p1[BENCH] / p0[BENCH]) * w
        sc_blend = r if sc_blend is None else sc_blend.add(r, fill_value=np.nan)
    ok = [s for s in sc_blend.index if s in rank.index and pd.notna(sc_blend[s])]
    SNAP[d] = (sc_blend[ok], rank)
    if prev_me is not None:
        for bucket, store in ((rank.index[:100], LC), (rank.index[300:500], SC)):
            rets = []
            for s in bucket:
                if s in close.columns:
                    a = close.at[prev_me, s]; b = close.at[d, s]
                    if pd.notna(a) and pd.notna(b) and a > 0:
                        rets.append(b / a - 1)
            store.append((d, float(np.mean(rets)) if rets else 0.0))
    prev_me = d
lc = pd.Series(dict(LC)); sc = pd.Series(dict(SC))
LCI = (1 + lc).cumprod(); SCI = (1 + sc).cumprod()
RS = (SCI / LCI)
RS6 = RS / RS.shift(6) - 1                    # 6-month smallcap-vs-largecap relative momentum
SMALL_LEADS = (RS6 > 0)
print(f"  smallcaps led in {SMALL_LEADS.mean()*100:.0f}% of months", flush=True)


def run(band, hold, buffer, pool, switch=None):
    """switch: None = fixed config; else a dict {month_end: (band,hold,buffer,pool)}."""
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
            bd_, hd_, bf_, pl_ = (switch.get(d, (band, hold, buffer, pool)) if switch
                                  else (band, hold, buffer, pool))
            scb, rank = SNAP[d]
            uni = rank.index[:bd_]
            sub = scb[[s for s in uni if s in scb.index]]
            etf = list(sub.sort_values(ascending=False).index[:pl_])
            if etf:
                top, buf = etf[:hd_], set(etf[:bf_])
                for s in list(held):
                    if s not in buf:
                        sell(s, d)
                cand = (list(held) + [x for x in top if x not in held])[:hd_]
                tgt = [s for s in cand if s in close.columns and pd.notna(close.at[d, s])]
                per = (E() + cash) / hd_
                for s in tgt:
                    if s not in held and cash > 0:
                        buy = min(per, cash)
                        if buy > 0:
                            held[s] = [buy * (1 - RT), buy, d]; cash -= buy
        out.append((d, E() + cash, E() + cash - tax)); prev = d
    return out


def stats(rows, col=2):
    n = pd.DataFrame(rows, columns=["d", "pre", "post"]).set_index("d").iloc[:, col - 1]
    y = (n.index[-1] - n.index[0]).days / 365.25
    c = (n.iloc[-1] / n.iloc[0]) ** (1 / y) - 1
    dr = n.pct_change().dropna()
    dd = ((n - n.cummax()) / n.cummax()).min()
    return dict(cagr=c * 100, dd=dd * 100, cal=(c / abs(dd) if dd < 0 else 0),
                sharpe=dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0), n


print("running N200 (live) and N500 (hold16/buf44) ...", flush=True)
r200 = run(200, 8, 22, 30)
r500 = run(500, 16, 44, 60)
s200, n200 = stats(r200); s500, n500 = stats(r500)

# ── 3. regime split: monthly returns of each book in smallcap-led vs largecap-led months ──
m200 = n200.resample("ME").last().pct_change().dropna()
m500 = n500.resample("ME").last().pct_change().dropna()
common = m200.index.intersection(m500.index)
lead = SMALL_LEADS.reindex(common, method="ffill").fillna(False)
rows = []
for label, mask in (("SMALLCAPS leading", lead), ("LARGECAPS leading", ~lead)):
    a, b = m200[common][mask.values], m500[common][mask.values]
    if len(a) < 3:
        continue
    rows.append((label, len(a), a.mean() * 1200, b.mean() * 1200, (b.mean() - a.mean()) * 1200,
                 (b > a).mean() * 100))
print(f"\n=== REGIME SPLIT (monthly, annualised) ===")
print(f"{'regime':>20} {'months':>7} {'N200':>9} {'N500':>9} {'N500-N200':>10} {'N500 wins':>10}")
for lbl, n, a, b, d, w in rows:
    print(f"{lbl:>20} {n:>7} {a:>8.1f}% {b:>8.1f}% {d:>+9.1f}% {w:>9.0f}%")

# ── 4. regime-SWITCHED book ──
sw = {}
for d in SNAP:
    ml = SMALL_LEADS.reindex([d], method="ffill")
    flag = bool(ml.iloc[0]) if len(ml) and pd.notna(ml.iloc[0]) else False
    sw[d] = (500, 16, 44, 60) if flag else (200, 8, 22, 30)
rsw = run(200, 8, 22, 30, switch=sw)
ssw, nsw = stats(rsw)
print(f"\n=== FULL-PERIOD (net of STCG) ===")
for lbl, s in (("N200 (live)", s200), ("N500 hold16/buf44", s500), ("REGIME-SWITCHED", ssw)):
    print(f"  {lbl:>20}: CAGR {s['cagr']:>5.1f}%  MaxDD {s['dd']:>6.1f}%  Calmar {s['cal']:>4.2f}  Sharpe {s['sharpe']:.2f}")

# yearly comparison
yr200 = n200.resample("YE").last().pct_change() * 100
yr500 = n500.resample("YE").last().pct_change() * 100
print(f"\n=== YEAR BY YEAR (net) ===\n{'year':>6} {'N200':>8} {'N500':>8} {'diff':>8} {'smallcap-led months':>21}")
for y in sorted(set(yr200.index.year) & set(yr500.index.year)):
    a = yr200[yr200.index.year == y]; b = yr500[yr500.index.year == y]
    if a.empty or b.empty or pd.isna(a.iloc[0]):
        continue
    sl = lead[[d for d in lead.index if d.year == y]]
    print(f"{y:>6} {a.iloc[0]:>7.1f}% {b.iloc[0]:>7.1f}% {b.iloc[0]-a.iloc[0]:>+7.1f}% "
          f"{(sl.mean()*100 if len(sl) else 0):>19.0f}%")
json.dump({"rs": [[str(k.date()), float(v)] for k, v in RS.items() if pd.notna(v)]},
          open(RES / "smallcap_rs.json", "w"))
print("\ndone", flush=True)
