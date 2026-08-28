# -*- coding: utf-8 -*-
"""Measure the HELD basket's own daily moves, instead of assuming a beta.

Arun's objection to research/114: I justified the 3.0x margin-call risk by saying "a momentum
basket moves 1.1-1.4x the index" — while the same report shows the book drawing down -22% against
the index's -59.7%. Those cannot both be casually true, and the assumed number is doing real work:
it is the whole basis for calling 3.0x a knife-edge.

The resolution is not a guess either way. The book's shallow drawdown comes mostly from the GATE
(it sits in cash through crashes), which says nothing about how the basket behaves on the days it
IS holding. That is the number the margin check actually depends on, so measure it.

This mirrors run_lev62's loop exactly — same selection, same buffer, same Donchian, same gate — and
records only one extra thing: the value-weighted daily return of the held basket, before exits.
"""
import importlib.util, csv
from pathlib import Path
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location(
    "lev62", "/home/arun/quantifyd/research/104_momentum_leverage/scripts/run_lev62.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r62, BENCH, START = m.r62, m.BENCH, m.START
OUT = Path("/home/arun/quantifyd/research/114_mtf_leverage/results")


def measure(close, tv, N=8, buffer=22, gate=100, donchian=15, etf_size=30, rt=0.003):
    idx = close.index[close.index >= START]
    me = set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)
    iso = idx.isocalendar()
    wk = set(pd.Series(idx, index=idx).groupby([iso.year.values, iso.week.values]).last().values)
    nbx = close[BENCH].ffill()
    roll_low = close.rolling(donchian, min_periods=donchian).min().shift(1)
    cash, held, prev, derisked = 1.0, {}, None, False
    log = []

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
        add = [s for s in top if s not in keep]
        target = [s for s in (keep + add)[:N] if pd.notna(px.get(s, np.nan))]
        if not target:
            return
        tot = E() + cash
        if tot <= 0:
            liq(); return
        w = tot / len(target)
        nh = {}
        for s in target:
            nh[s] = [w, w, d, px[s]] if s not in held else held[s]
            nh[s][0] = w
        held = nh
        cash = 0.0
        derisked = False

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            g0 = E()
            for s, stt in held.items():
                p1 = px.get(s, np.nan)
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    stt[0] += stt[0] * (p1 / p0 - 1.0)
            g1 = E()
            if g0 > 0:
                bi = nbx.get(d, np.nan); b0 = nbx.get(prev, np.nan)
                ir = (bi / b0 - 1.0) if (pd.notna(bi) and pd.notna(b0) and b0 > 0) else np.nan
                log.append((d, g1 / g0 - 1.0, ir, len(held)))
        if held:
            lows = roll_low.loc[d]
            for s in list(held):
                p1 = px.get(s, np.nan); ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                    cash += held.pop(s)[0]
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
        prev = d
    return pd.DataFrame(log, columns=["d", "basket", "index", "n"]).set_index("d")


def main():
    close, tv = m.mom75.load()
    df = measure(close, tv).dropna()
    df.to_csv(OUT / "basket_daily.csv")
    print(f"  measured {len(df)} days on which the book was HOLDING (2006-2026)\n")

    print("  WORST single-day falls of the HELD basket:")
    print(f"    {'date':<12}{'basket':>9}{'index':>9}{'ratio':>8}  held")
    for d, r in df.nsmallest(8, "basket").iterrows():
        rat = r["basket"] / r["index"] if r["index"] < -0.0001 else float("nan")
        print(f"    {str(d.date()):<12}{r['basket']*100:>8.2f}%{r['index']*100:>8.2f}%{rat:>8.2f}  {int(r['n'])}")

    down = df[df["index"] <= -0.02]
    beta_all = np.polyfit(df["index"], df["basket"], 1)[0]
    beta_down = np.polyfit(down["index"], down["basket"], 1)[0] if len(down) > 2 else float("nan")
    print(f"\n  beta while HOLDING (all days)          {beta_all:.2f}   n={len(df)}")
    print(f"  beta while HOLDING (index days <= -2%) {beta_down:.2f}   n={len(down)}")
    print(f"  worst basket day while holding         {df['basket'].min()*100:.2f}%")

    print("\n  MARGIN-CALL CHECK using the MEASURED worst day, not an assumed beta:")
    worst = df["basket"].min()
    for lev in (2.0, 2.5, 3.0):
        trig = ((1.0 / lev) - 0.25) / 0.75
        head = trig + worst              # worst is negative
        verdict = "SAFE" if head > 0.02 else ("THIN" if head > 0 else "WOULD HAVE BEEN CALLED")
        print(f"    {lev}x: needs {trig*100:>5.1f}% fall · worst actual {worst*100:.2f}% "
              f"· headroom {head*100:>5.1f}pp -> {verdict}")


if __name__ == "__main__":
    main()
