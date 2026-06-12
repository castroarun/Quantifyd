"""Research 62 — Momentum-30 ETF sub-selection (daily-marked).

IDEA (user): instead of running our own selection, piggyback on a factor index.
Reconstruct the **Nifty 200 Momentum 30** from methodology (NO factsheets), then
hold a concentrated, buffered sub-basket: top-10 of the 30 by momentum score,
100% invested, retain a name while it stays inside the top-22 buffer, monthly
rotate. NO macro gate in the base (user design); optional per-stock Donchian
trailing exit. Control arm adds the research/41 NIFTYBEES-SMA gate to measure the
cost of dropping it (research/41 phase28: midcap no-gate had NO net edge; gate
"irreplaceable" — we test whether large-cap Momentum-30 differs).

Reconstruction (survivorship-free, from price+volume only):
  - Eligible pool = PIT top-200 by trailing-6mo median traded value (rs2.pit_universe
    with a new band nifty200=(0,200)). Faithful Nifty-200 proxy; benchmark excluded.
  - Momentum-30 score = NSE methodology: 6m(126d) & 12m(252d) price returns each
    risk-adjusted (÷ annualized daily-return vol), z-scored across the 200, averaged.
    The "ETF" = top-30 by score. We then sub-select within those 30.

Daily-marked NAV (honest MaxDD + Donchian). Gross AND post-tax@STCG20%. Per-year.
Reuses /home/arun/quantifyd/research/41_midsmall400_mq_concentrated/scripts/02_rs_sweep.py.
"""
from __future__ import annotations
import importlib.util, time, sys, csv
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
ENGINE = Path("/home/arun/quantifyd/research/41_midsmall400_mq_concentrated/"
              "scripts/02_rs_sweep.py")
_s = importlib.util.spec_from_file_location("rs2", str(ENGINE))
rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)

rs2.BANDS["nifty200"] = (0, 200)          # top-200 by traded value = Nifty-200 proxy
BENCH = rs2.NIFTY_SYM                       # NIFTYBEES
RT = rs2.RT_COST                            # 0.004 round-trip
DAY_CASH = (1 + rs2.CASH_ANNUAL) ** (1 / 252)
START = pd.Timestamp("2014-01-01")


def mom30_score(close, asof, universe):
    """Nifty 200 Momentum 30 methodology score over `universe` (a set of symbols).
    For each name: 6m & 12m total return, each ÷ annualized daily-return std
    (risk-adjusted momentum ratio); z-score each leg across the universe; average
    the two z-legs. Returns a Series (higher = stronger momentum). Causal: uses
    only data <= asof."""
    h = close.loc[:asof]
    if len(h) <= 252:
        return None
    cols = [s for s in universe if s in h.columns and s != BENCH]
    if len(cols) < 30:
        return None
    sub = h[cols]
    legs = []
    for L in (126, 252):
        p0 = sub.iloc[-L - 1]
        p1 = sub.iloc[-1]
        ret = p1 / p0 - 1.0
        dr = sub.iloc[-L:].pct_change()
        vol = dr.std() * np.sqrt(252)
        radj = ret / vol.replace(0, np.nan)
        z = (radj - radj.mean()) / radj.std(ddof=0)
        legs.append(z)
    score = pd.concat(legs, axis=1).mean(axis=1)
    return score.dropna()


def eligible_etf(close, tv, d, score_fn, etf_size):
    """Return the reconstructed 'ETF' = ranked list of top `etf_size` names."""
    uni = rs2.pit_universe(tv, close, d, "nifty200")
    if not uni:
        return None
    if score_fn == "mom30":
        sc = mom30_score(close, d, uni)
    else:  # 'rsblend' — research/41 6m/12m relative-strength proxy (sanity arm)
        sc = rs2.rs_scores(close, d, None)
        if sc is not None:
            sc = sc[[s for s in sc.index if s in uni and s != BENCH]]
    if sc is None or len(sc) == 0:
        return None
    sc = sc[[s for s in sc.index if s != BENCH]].sort_values(ascending=False)
    return list(sc.index[:etf_size])


def run(close, tv, score_fn="mom30", N=10, buffer=22, gate=None, donchian=None,
        etf_size=30, stcg=0.0, rt=RT, exclude=None):
    """Daily-marked backtest. held[sym] = [value, cost_basis, buydate, peak].
    `exclude` = set of symbols the book may never hold (super-winner guard)."""
    exclude = exclude or set()
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    iso = idx.isocalendar()
    wk_last = set(pd.Series(idx, index=idx).groupby(
        [iso.year.values, iso.week.values]).last().values)
    cash, held = 1.0, {}
    derisked = False
    nav, prev = [], None
    st = dict(tax=0.0, cost=0.0, fills=0, donchian_exits=0, gate_derisk=0,
              turn_sum=0.0, contrib={})
    # vectorized trailing Donchian low (prior N bars, excludes today via shift)
    roll_low = (close.rolling(donchian, min_periods=donchian).min().shift(1)
                if donchian else None)

    def E():
        return sum(v[0] for v in held.values())

    def realize(stt, d):
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; st['tax'] += t; return t
        return 0.0

    def do_fill(d, px):
        nonlocal cash, held, derisked
        etf = eligible_etf(close, tv, d, score_fn, etf_size)
        if not etf:
            return
        if exclude:
            etf = [s for s in etf if s not in exclude]
        top = etf[:N]
        buf = set(etf[:buffer])
        for s in list(held):                       # evict names out of buffer
            if s not in buf:
                t = realize(held[s], d); cash += held.pop(s)[0] - t
        keep = [s for s in held if s in buf]
        add = [s for s in top if s not in keep]
        target = (keep + add)[:N]
        target = [s for s in target if pd.notna(px.get(s, np.nan))]
        if not target:
            return
        tot = E() + cash
        w = tot / len(target)
        turn = len(set(held).symmetric_difference(set(target))) / max(1, 2 * N)
        nh = {}
        for s in target:
            if s in held:
                stt = held[s]; stt[0] = w; nh[s] = stt   # keep cost basis/peak
            else:
                nh[s] = [w, w, d, px[s]]
        held = nh
        c = tot * rt * turn * 2
        cash = tot - w * len(target) - c
        st['cost'] += c; st['fills'] += 1; st['turn_sum'] += turn
        derisked = False

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:                 # daily mark-to-market
            for s, stt in held.items():
                p1 = px.get(s, np.nan)
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    delta = stt[0] * (p1 / p0 - 1.0)
                    stt[0] += delta
                    stt[3] = max(stt[3], p1)
                    st['contrib'][s] = st['contrib'].get(s, 0.0) + delta
        if cash > 0:
            cash *= DAY_CASH
        h = close.loc[:d]

        if donchian and held:                          # per-stock Donchian exit
            lows = roll_low.loc[d]
            for s in list(held):
                p1 = px.get(s, np.nan); low_n = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(low_n) and p1 < low_n:
                    t = realize(held[s], d); cash += held.pop(s)[0] - t
                    st['donchian_exits'] += 1

        if gate and d in wk_last:                       # weekly macro gate (control)
            b = h[BENCH].dropna()
            roff = len(b) >= gate and b.iloc[-1] < b.tail(gate).mean()
            if roff and held:
                for s in list(held):
                    t = realize(held[s], d); cash += held.pop(s)[0] - t
                derisked = True; st['gate_derisk'] += 1
            elif not roff:
                derisked = False

        if d in me and not (gate and derisked):         # monthly rebalance
            do_fill(d, px)

        nav.append((d, E() + cash))
        prev = d

    return _stats(nav, st)


def bench_nav(close):
    n = close[BENCH].loc[close.index >= START].dropna()
    return n / n.iloc[0]


def _stats(nav, st):
    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cagr = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dr = n.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    dn = dr[dr < 0]
    sortino = (dr.mean() / dn.std() * np.sqrt(252)) if len(dn) and dn.std() > 0 else np.nan
    dd = ((n - n.cummax()) / n.cummax()).min()
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    last = n.groupby(n.index.year).last()
    py = last.pct_change()
    py.iloc[0] = last.iloc[0] / n.iloc[0] - 1
    return dict(cagr=cagr * 100, dd=dd * 100, sharpe=sharpe, sortino=sortino,
                calmar=calmar, st=st, nav=n, py=(py * 100).round(1))


def stats_from_nav(n):
    return _stats(list(zip(n.index, n.values)), {})


FIELDS = ["config", "score", "N", "buffer", "gate", "donchian",
          "cagr", "post_tax20", "maxdd", "sharpe", "sortino", "calmar",
          "fills", "donch_exits", "gate_derisk", "avg_turn", "cost_pct"]


def main():
    t0 = time.time()
    print("Loading daily price+tv from VPS DB ...", flush=True)
    close, tv = rs2.load()
    print(f"  {close.shape[1]} symbols, {close.index.min().date()}.."
          f"{close.index.max().date()}", flush=True)

    out = RES / "g1_probe.csv"
    pyout = RES / "g1_probe_peryear.csv"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r['config'] for r in csv.DictReader(f)}
        print(f"  resuming: {len(done)} configs already done", flush=True)
    else:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    # ---- reference benchmarks ----
    pys = {}
    bn = bench_nav(close)
    bm = stats_from_nav(bn)
    print(f"  [ref] NIFTYBEES  CAGR={bm['cagr']:.1f}%  DD={bm['dd']:.1f}%  "
          f"Sh={bm['sharpe']:.2f}  Cal={bm['calmar']:.2f}", flush=True)
    pys["NIFTYBEES"] = bm['py']

    # G1 grid: (config, score, N, buffer, gate, donchian)
    cfgs = [
        ("plain_ETF_top30",          "mom30", 30, 30, None, None),
        ("BASE_mom30_N10_buf22",     "mom30", 10, 22, None, None),
        ("mom30_N10_donch20",        "mom30", 10, 22, None, 20),
        ("mom30_N10_donch50",        "mom30", 10, 22, None, 50),
        ("mom30_N10_GATE100",        "mom30", 10, 22, 100, None),
        ("mom30_N10_GATE100_donch20","mom30", 10, 22, 100, 20),
        ("mom30_N10_GATE100_donch50","mom30", 10, 22, 100, 50),
        ("SANITY_rsblend_N10_buf22", "rsblend", 10, 22, None, None),
    ]
    rows = []
    for lbl, sf, N, buf, gate, donch in cfgs:
        if lbl in done:
            print(f"  skip {lbl} (done)", flush=True); continue
        ts = time.time()
        g = run(close, tv, sf, N, buf, gate, donch)
        t = run(close, tv, sf, N, buf, gate, donch, stcg=0.20)
        fills = g['st'].get('fills', 0) or 1
        row = dict(config=lbl, score=sf, N=N, buffer=buf,
                   gate=(gate or 0), donchian=(donch or 0),
                   cagr=round(g['cagr'], 1), post_tax20=round(t['cagr'], 1),
                   maxdd=round(g['dd'], 1), sharpe=round(g['sharpe'], 2),
                   sortino=round(g['sortino'], 2) if pd.notna(g['sortino']) else "",
                   calmar=round(g['calmar'], 2) if pd.notna(g['calmar']) else "",
                   fills=g['st']['fills'], donch_exits=g['st']['donchian_exits'],
                   gate_derisk=g['st']['gate_derisk'],
                   avg_turn=round(g['st']['turn_sum'] / fills, 3),
                   cost_pct=round(g['st']['cost'] * 100, 1))
        rows.append(row); pys[lbl] = g['py']
        with open(out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        pd.DataFrame(pys).reindex(range(2014, 2027)).to_csv(pyout)
        print(f"  {lbl:30} CAGR={row['cagr']:6.1f}% net20={row['post_tax20']:6.1f}%"
              f" DD={row['maxdd']:6.1f}% Sh={row['sharpe']:.2f} Cal={row['calmar']}"
              f" donchX={row['donch_exits']:>3} [{time.time()-ts:.0f}s]", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s. Output: {out}", flush=True)
    if rows:
        print(pd.DataFrame(rows)[["config", "cagr", "post_tax20", "maxdd",
                                  "sharpe", "calmar"]].to_string(index=False))


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)
    main()
