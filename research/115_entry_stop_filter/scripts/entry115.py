# -*- coding: utf-8 -*-
"""Should a rebalance buy a name that is already below its own stop — and if not, what instead?

On 2026-08-31 the live rebalance ranked RADICO 6th and ADANIENSOL 3rd, then refused both because
each was trading below its 15-day Donchian low. RADICO had been SOLD by that same stop eleven
seconds earlier. The book finished 6/8 with two empty slots.

Arun's question: if a name is disqualified, why not keep walking down the ranked list and take the
next one that qualifies, instead of leaving the slot empty?

That is an ENTRY-rule change. The top-22 buffer is a RETENTION rule; entries have always come from
the top-8, so backfilling from #9+ is something research/62 never validated. And the guard itself
(shipped 2026-08-26) is equally untested — the original backtest has no entry filter at all.

Three arms, identical in every other respect:

  A  none      buy the top-8 regardless of where the price sits  (what r/62 and r/104 validated)
  B  skip      refuse a name below its stop, leave the slot EMPTY (what is live today)
  C  backfill  refuse it, then walk deeper into the 30-name pool for the next qualifier (proposed)

Engine, rules, costs and period are unchanged from run_lev62 at lev=1.0.
"""
import csv, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location(
    "lev62", "/home/arun/quantifyd/research/104_momentum_leverage/scripts/run_lev62.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r62, BENCH, START = m.r62, m.BENCH, m.START
OUT = Path("/home/arun/quantifyd/research/115_entry_stop_filter/results")
OUT.mkdir(parents=True, exist_ok=True)


def run(close, tv, entry_mode, N=8, buffer=22, gate=100, donchian=15, etf_size=30, rt=0.003):
    day_cash = (1.065) ** (1 / 252)
    idx = close.index[close.index >= START]
    me = set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)
    iso = idx.isocalendar()
    wk = set(pd.Series(idx, index=idx).groupby([iso.year.values, iso.week.values]).last().values)
    nbx = close[BENCH].ffill()
    roll_low = close.rolling(donchian, min_periods=donchian).min().shift(1)
    cash, held, nav, prev, derisked = 1.0, {}, [], None, False
    st = dict(fills=0, donch=0, blocked=0, backfilled=0, slots_left_empty=0, cost=0.0)

    def E():
        return sum(v[0] for v in held.values())

    def liq():
        nonlocal cash, held
        for s in list(held):
            cash += held.pop(s)[0]

    def do_fill(d, px):
        nonlocal cash, held, derisked
        etf = r62.eligible_etf(close, tv, d, "rsblend", etf_size)
        if not etf:
            return
        top = etf[:N]; buf = set(etf[:buffer])
        for s in list(held):
            if s not in buf:
                cash += held.pop(s)[0]
        keep = [s for s in held if s in buf]
        lows = roll_low.loc[d]

        def ok(s):
            if entry_mode == "none":
                return True
            p1 = px.get(s, np.nan); ln = lows.get(s, np.nan)
            return not (pd.notna(p1) and pd.notna(ln) and p1 < ln)

        if entry_mode == "backfill":
            pool = [s for s in etf if s not in keep]
            add = []
            for s in pool:
                if len(keep) + len(add) >= N:
                    break
                if pd.isna(px.get(s, np.nan)):
                    continue
                if ok(s):
                    add.append(s)
                    if etf.index(s) >= N:
                        st["backfilled"] += 1
                else:
                    st["blocked"] += 1
        else:
            cand = [s for s in top if s not in keep]
            add = []
            for s in cand:
                if ok(s):
                    add.append(s)
                else:
                    st["blocked"] += 1

        target = [s for s in (keep + add)[:N] if pd.notna(px.get(s, np.nan))]
        if not target:
            return
        st["slots_left_empty"] += max(0, N - len(target))
        tot = E() + cash
        if tot <= 0:
            liq(); return
        w = tot / len(target)
        turn = len(set(held).symmetric_difference(set(target))) / max(1, 2 * N)
        nh = {}
        for s in target:
            nh[s] = held[s] if s in held else [w, w, d, px[s]]
            nh[s][0] = w
        held = nh
        c = tot * rt * turn * 2
        cash = tot - E() - c
        st["cost"] += c; st["fills"] += 1
        derisked = False

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan)
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    stt[0] += stt[0] * (p1 / p0 - 1.0)
        if cash > 0:
            cash *= day_cash
        if held:
            lows = roll_low.loc[d]
            for s in list(held):
                p1 = px.get(s, np.nan); ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                    cash += held.pop(s)[0]; st["donch"] += 1
        if d in wk:
            b = nbx.loc[:d].dropna()
            roff = len(b) >= gate and b.iloc[-1] < b.tail(gate).mean()
            if roff:
                if held:
                    liq()
                derisked = True
            else:
                derisked = False
        if d in me and not derisked:
            do_fill(d, px)
        nav.append((d, E() + cash)); prev = d
    g = r62._stats(nav, st)
    g["st"] = st
    g["nav_series"] = nav      # keep the curve so results can be re-scored per era
    return g


def main():
    close, tv = m.mom75.load()
    print(f"  data {close.index.min().date()}..{close.index.max().date()}, from {START.date()}\n", flush=True)
    bn = close[BENCH].loc[close.index >= START].dropna(); bn = bn / bn.iloc[0]
    bm = r62._stats([(d, v) for d, v in bn.items()], {})
    print(f"  NIFTYBEES B&H: CAGR {bm['cagr']:.1f}%  DD {bm['dd']:.1f}%  Calmar {bm['calmar']:.2f}\n", flush=True)
    arms = [("A none (r/62 baseline)", "none"),
            ("B skip, slot empty (LIVE)", "skip"),
            ("C backfill deeper", "backfill")]
    print(f"  {'arm':<26}{'CAGR':>7}{'MaxDD':>9}{'Sharpe':>8}{'Calmar':>8}{'blocked':>9}{'backfil':>8}{'empty':>7}", flush=True)
    rows = []
    f = open(OUT / "entry_filter.csv", "w", newline="")
    w = csv.writer(f); w.writerow(["arm", "cagr", "maxdd", "sharpe", "calmar", "blocked", "backfilled", "slots_empty", "donch_exits"])
    for label, mode in arms:
        g = run(close, tv, mode)
        cal = g["calmar"] if pd.notna(g["calmar"]) else 0.0
        s = g["st"]
        print(f"  {label:<26}{g['cagr']:>6.1f}%{g['dd']:>8.1f}%{g['sharpe']:>8.2f}{cal:>8.2f}"
              f"{s['blocked']:>9}{s['backfilled']:>8}{s['slots_left_empty']:>7}", flush=True)
        w.writerow([label, round(g["cagr"], 1), round(g["dd"], 1), round(g["sharpe"], 2), round(cal, 2),
                    s["blocked"], s["backfilled"], s["slots_left_empty"], s["donch"]])
        f.flush(); rows.append((label, g, s))
    f.close()
    print()
    base = rows[0][1]
    for label, g, s in rows[1:]:
        print(f"  {label}: CAGR {g['cagr']-base['cagr']:+.1f}pp, DD {g['dd']-base['dd']:+.1f}pp, "
              f"Calmar {(g['calmar'] if pd.notna(g['calmar']) else 0)-base['calmar']:+.2f} vs the r/62 baseline")


if __name__ == "__main__":
    main()
