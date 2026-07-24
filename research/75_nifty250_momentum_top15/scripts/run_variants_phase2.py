"""Research 75 phase 2 — variant sweep on the faithful momentum book.

Extends the phase-1 faithful base along the axes the user asked for:
  - UNIVERSE : large+mid 250 (base) / midcap(100-250) / smallcap(250-500) / combo(100-500)
  - MOMENTUM : 12m absolute return (base) vs RELATIVE-STRENGTH vs NIFTY over 55/120/126/252d
               and the 6m+12m blend (the research/41 rs_scores angle; 'rs120' = the 120d RS the
               user referenced). Sweeps the period to find which suits each universe.
  - Held fixed: index cash-gate ON, N=15, EMA-stack OFF (phase1: harmful), net 0.3%, daily-marked.
  - Plus: cash-yield sensitivity (6.5% base vs 4% vs 0%) on the base cell, and a stack ON/OFF
    reconfirm on the two best momentum angles.

Reuses phase-1 run()/momentum()/load()/_stats via import, adding RS scores (rs2.rs_scores) and a
universe-band parameter.
"""
from __future__ import annotations
import importlib.util, time, csv, sys, logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.disable(logging.WARNING)
HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(parents=True, exist_ok=True)

# import phase-1 module (gives rs2, load, EMAs, _stats, START, MODERN, RT, DAY_CASH, EXCLUDE, BENCH)
P1 = HERE / "scripts" / "run_nifty250_momentum.py"
_s = importlib.util.spec_from_file_location("p1", str(P1))
p1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(p1)
rs2 = p1.rs2
BENCH = p1.BENCH; EXCLUDE = p1.EXCLUDE; RT = p1.RT
START = p1.START; MODERN = p1.MODERN

# ensure all bands exist (rs2 already has mid/small/combo; add n250)
rs2.BANDS.setdefault("n250", (0, 250))
RS_L = {"rs55": 55, "rs120": 120, "rs126": 126, "rs252": 252, "rsblend": None}


def momentum(close, asof, score):
    """Absolute-return scores handled by phase1; RS scores via rs2.rs_scores (relative
    strength vs NIFTYBEES over L days)."""
    if score in ("ret252", "ret252_21", "radj_z"):
        return p1.momentum(close, asof, score)
    if score in RS_L:
        return rs2.rs_scores(close, asof, RS_L[score])
    raise ValueError(score)


def run(close, tv, ema, nbx_ema100, band="n250", score="ret252", N=15,
        use_stack=False, use_gate=True, rt=RT, stcg=0.0, cash_annual=None):
    """Daily-marked backtest with a universe BAND and (RS or absolute) momentum SCORE.
    Mirrors phase1.run() exactly; the only additions are `band` and RS scoring."""
    day_cash = ((1 + cash_annual) ** (1 / 252)) if cash_annual is not None else p1.DAY_CASH
    idx = close.index[close.index >= START]
    _ss = pd.Series(idx, index=idx)
    me = set(_ss.groupby([idx.year, idx.month]).max().values)
    nbx = close[BENCH].ffill()
    e50, e100, e200 = ema[50], ema[100], ema[200]
    cash, held = 1.0, {}
    nav = []; prev = None
    st = dict(tax=0.0, cost=0.0, fills=0, turn_sum=0.0, gate_off=0)

    def E(): return sum(v[0] for v in held.values())

    def realize(stt, d):
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; st['tax'] += t; return t
        return 0.0

    def liquidate(d):
        nonlocal cash, held
        for s in list(held):
            t = realize(held[s], d); cash += held.pop(s)[0] - t

    def gate_on(d):
        if not use_gate: return True
        v = nbx_ema100.loc[d]
        return True if pd.isna(v) else (nbx.loc[d] > v)

    def do_fill(d, px):
        nonlocal cash, held
        sc = momentum(close, d, score)
        if sc is None or len(sc) == 0: return
        uni = rs2.pit_universe(tv, close, d, band)
        cand = [s for s in sc.index if s in uni and s not in EXCLUDE and s != BENCH]
        if use_stack:
            cand = [s for s in cand
                    if pd.notna(e50[s].loc[d]) and pd.notna(e200[s].loc[d])
                    and e50[s].loc[d] > e100[s].loc[d] > e200[s].loc[d]]
        cand = [s for s in cand if pd.notna(px.get(s, np.nan))]
        if not cand:
            liquidate(d); return
        top = list(sc[cand].sort_values(ascending=False).index[:N])
        tot = E() + cash; w = tot / len(top)
        turn = len(set(held).symmetric_difference(set(top))) / max(1, 2 * len(top))
        nh = {}
        for s in top:
            nh[s] = held[s] if s in held else [w, w, d]; nh[s][0] = w
        for s in list(held):
            if s not in nh: realize(held[s], d)
        held = nh
        c = tot * rt * turn * 2
        cash = tot - w * len(top) - c
        st['cost'] += c; st['fills'] += 1; st['turn_sum'] += turn

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1v = px.get(s, np.nan); p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1v) and pd.notna(p0) and p0 > 0:
                    stt[0] += stt[0] * (p1v / p0 - 1.0)
        if cash > 0: cash *= day_cash
        if use_gate and d in me and not gate_on(d) and held:
            liquidate(d); st['gate_off'] += 1
        if d in me:
            if gate_on(d): do_fill(d, px)
            elif held: liquidate(d)
        nav.append((d, E() + cash)); prev = d
    return p1._stats(nav, st, START)


FIELDS = ["config", "universe", "score", "stack", "cash_yield",
          "cagr", "net_cagr", "post_tax20", "maxdd", "sharpe", "sortino", "calmar",
          "mult", "fills", "avg_turn", "gate_off", "cost_pct", "mod_cagr", "mod_maxdd", "mod_calmar"]


def row_of(lbl, band, score, stack, cy, gross, net, tax):
    fills = net['st']['fills'] or 1; m = net['modern']
    return dict(config=lbl, universe=band, score=score, stack=int(stack),
                cash_yield=cy, cagr=round(gross['cagr'], 1), net_cagr=round(net['cagr'], 1),
                post_tax20=round(tax['cagr'], 1), maxdd=round(net['dd'], 1),
                sharpe=round(net['sharpe'], 2),
                sortino=round(net['sortino'], 2) if pd.notna(net['sortino']) else "",
                calmar=round(net['calmar'], 2) if pd.notna(net['calmar']) else "",
                mult=round(net['mult'], 1), fills=net['st']['fills'],
                avg_turn=round(net['st']['turn_sum'] / fills, 3),
                gate_off=net['st']['gate_off'], cost_pct=round(net['st']['cost'] * 100, 1),
                mod_cagr=round(m['cagr'], 1) if m else "",
                mod_maxdd=round(m['dd'], 1) if m else "",
                mod_calmar=round(m['calmar'], 2) if m else "")


def main():
    t0 = time.time()
    print("Loading daily price+tv ...", flush=True)
    close, tv = p1.load()
    print(f"  {close.shape[1]} symbols {close.index.min().date()}..{close.index.max().date()}", flush=True)
    ema = {p: close.ewm(span=p, adjust=False, min_periods=p).mean() for p in (50, 100, 200)}
    nbx_ema100 = close[BENCH].ffill().ewm(span=100, adjust=False, min_periods=100).mean()

    out = RES / "ranking_v2.csv"
    done = set()
    if out.exists():
        with open(out) as f: done = {r['config'] for r in csv.DictReader(f)}
        print(f"  resuming: {len(done)} done", flush=True)
    else:
        with open(out, "w", newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    universes = ["n250", "mid", "small", "combo"]
    scores = ["ret252", "rs55", "rs120", "rs126", "rs252", "rsblend"]
    cfgs = []
    for band in universes:                         # main grid: 4 x 6, stack OFF, cash 6.5%
        for sc in scores:
            cfgs.append((f"{band}__{sc}", band, sc, False, 6.5))
    # stack ON/OFF reconfirm on base universe, two angles
    cfgs += [("n250__ret252__stackON", "n250", "ret252", True, 6.5),
             ("n250__rs120__stackON",  "n250", "rs120",  True, 6.5)]
    # cash-yield sensitivity on the base cell
    cfgs += [("n250__ret252__cash4", "n250", "ret252", False, 4.0),
             ("n250__ret252__cash0", "n250", "ret252", False, 0.0)]

    for lbl, band, sc, stack, cy in cfgs:
        if lbl in done:
            print(f"  skip {lbl}", flush=True); continue
        ts = time.time()
        cya = cy / 100.0
        net = run(close, tv, ema, nbx_ema100, band, sc, 15, stack, True, RT, 0.0, cya)
        gross = run(close, tv, ema, nbx_ema100, band, sc, 15, stack, True, 0.0, 0.0, cya)
        tax = run(close, tv, ema, nbx_ema100, band, sc, 15, stack, True, RT, 0.20, cya)
        row = row_of(lbl, band, sc, stack, cy, gross, net, tax)
        with open(out, "a", newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"  {lbl:24} net={row['net_cagr']:5.1f}% DD={row['maxdd']:6.1f}% "
              f"Cal={row['calmar']} Sh={row['sharpe']:.2f} turn={row['avg_turn']} "
              f"mult={row['mult']}x [{time.time()-ts:.0f}s]", flush=True)
        sys.stdout.flush()
    print(f"\nDone in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
