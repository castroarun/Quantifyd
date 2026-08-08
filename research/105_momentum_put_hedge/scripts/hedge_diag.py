"""Fair-comparison baselines for the weekly arms (2019 start) + the key diagnostic:
WHEN does the drawdown accrue — while the gate is still risk-ON (before it flips), or after?
If most damage lands before the gate flips, no gate-triggered hedge can ever fix it."""
import importlib.util, sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts")
spec = importlib.util.spec_from_file_location(
    "hs", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_sweep.py")
hs = importlib.util.module_from_spec(spec); spec.loader.exec_module(hs)

print("\n=== FAIR BASELINES for the weekly arms (2019-02 start, same window) ===")
st19 = pd.Timestamp("2019-02-01")
for lbl, kw in [("A0_cash_exit_2019", dict(mode="cash_exit", start=st19)),
                ("A1_hold_naked_2019", dict(mode="hold", start=st19))]:
    r = hs.run(**kw)
    print(f"  {lbl:>20}  CAGR {r['cagr']:>5.1f}%  netCAGR {r['net_cagr']:>5.1f}%  "
          f"MaxDD {r['maxdd']:>6.1f}%  Calmar {r['calmar']:>4.2f}  netCal {r['net_calmar']:>4.2f}  "
          f"exits {r['full_exits']}")

print("\n=== DIAGNOSTIC: does the drawdown land BEFORE the gate flips? ===")
# rebuild the naked book, recording nav + gate state daily
NBX, idx, ME, WK, ROLL_LOW, close = hs.NBX, hs.idx, hs.ME, hs.WK, hs.ROLL_LOW, hs.close
gate_off = {}
for d in idx:
    b = NBX.loc[:d].dropna()
    gate_off[d] = bool(len(b) >= 100 and b.iloc[-1] < b.tail(100).mean())
# carry the weekly decision forward (gate only re-evaluated weekly)
state, cur = {}, False
for d in idx:
    if d in WK:
        cur = gate_off[d]
    state[d] = cur

cash, held, prev, nav = 1.0, {}, None, []
for d in idx:
    if held and prev is not None:
        for s, v in held.items():
            p1 = close.at[d, s] if s in close.columns else np.nan
            p0 = close.at[prev, s] if s in close.columns else np.nan
            if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                v[0] += v[0] * (p1 / p0 - 1.0)
    if cash > 0:
        cash *= (1.065) ** (1 / 252)
    if held:
        lows = ROLL_LOW.loc[d]
        for s in list(held):
            p1 = close.at[d, s] if s in close.columns else np.nan
            ln = lows.get(s, np.nan)
            if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                cash += held.pop(s)[0] * 0.9985
    if d in ME:
        etf = hs.ETF.get(d) or []
        if etf:
            top, buf = etf[:8], set(etf[:22])
            for s in list(held):
                if s not in buf:
                    cash += held.pop(s)[0] * 0.9985
            cand = (list(held) + [x for x in top if x not in held])[:8]
            tgt = [s for s in cand if s in close.columns and pd.notna(close.at[d, s])]
            per = (sum(v[0] for v in held.values()) + cash) / 8
            for s in tgt:
                if s not in held and cash > 0:
                    buy = min(per, cash)
                    if buy > 0:
                        held[s] = [buy * 0.9985, buy, d]; cash -= buy
    nav.append((d, sum(v[0] for v in held.values()) + cash))
    prev = d

n = pd.DataFrame(nav, columns=["d", "v"]).set_index("d")["v"]
dd = (n - n.cummax()) / n.cummax()
trough = dd.idxmin()
peak = n.loc[:trough].idxmax()
seg = n.loc[peak:trough]
print(f"  Worst drawdown {dd.min()*100:.1f}%  peak {peak.date()} -> trough {trough.date()} "
      f"({(trough-peak).days} days)")
# split the decline into gate-ON vs gate-OFF portions
ret = seg.pct_change().dropna()
on_r = float(np.prod([1 + r for d, r in ret.items() if not state.get(d, False)]) - 1)
off_r = float(np.prod([1 + r for d, r in ret.items() if state.get(d, False)]) - 1)
n_on = sum(1 for d in ret.index if not state.get(d, False))
n_off = sum(1 for d in ret.index if state.get(d, False))
print(f"  While gate RISK-ON  (unhedgeable — gate hasn't flipped): {on_r*100:>7.1f}%  over {n_on} days")
print(f"  While gate RISK-OFF (hedge would be active):             {off_r*100:>7.1f}%  over {n_off} days")
lag = (n.loc[peak:].index[[state.get(d, False) for d in n.loc[peak:].index]])
if len(lag):
    print(f"  Gate flipped risk-off on {lag[0].date()} — {(lag[0]-peak).days} days after the peak, "
          f"by which point the book was already {(n.loc[lag[0]]/n.loc[peak]-1)*100:.1f}% off its high")

print("\n  Top-5 drawdowns and how much of each was 'gate-ON' (unhedgeable):")
und = dd[dd < -0.05]
episodes, cur_ep = [], None
for d, v in dd.items():
    if v < -0.001:
        cur_ep = cur_ep or d
    elif cur_ep:
        episodes.append((cur_ep, d)); cur_ep = None
scored = []
for a, b in episodes:
    seg2 = n.loc[a:b]
    if len(seg2) < 3:
        continue
    tr = seg2.idxmin(); depth = seg2.min() / seg2.iloc[0] - 1
    if depth > -0.05:
        continue
    r2 = seg2.loc[:tr].pct_change().dropna()
    onr = float(np.prod([1 + x for d2, x in r2.items() if not state.get(d2, False)]) - 1)
    scored.append((depth, a.date(), tr.date(), onr))
for depth, a, tr, onr in sorted(scored)[:5]:
    print(f"    {a} -> {tr}: total {depth*100:>6.1f}%   of which gate-ON {onr*100:>6.1f}%")
