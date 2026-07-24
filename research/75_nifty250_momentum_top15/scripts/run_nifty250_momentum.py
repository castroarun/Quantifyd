"""Research 75 — Faithful replication of the Quantinuous "Only Momentum Strategy
You Need for Nifty 250 Stocks" video.

Rules (from the video):
  Universe : Nifty LargeMidcap 250  (PIT proxy = top-250 by trailing traded value)
  Select   : rank universe by momentum, hold TOP 15 equal-weight, monthly rebalance
  Trend    : per-stock EMA50 > EMA100 AND EMA100 > EMA200 (else ineligible)
  Regime   : NIFTYBEES close > its own EMA100, else move ALL to cash
Claim to test: ~27% CAGR, -23% MaxDD, ~20yr, ~12 trades/yr.

Daily-marked NAV (honest MaxDD). Gross AND net@0.3% round-trip. STCG20% shown
separately. Reuses rs2 (research/41 .../02_rs_sweep.py) for load / pit_universe /
month_ends / NIFTYBEES benchmark / cash & cost constants.
"""
from __future__ import annotations
import importlib.util, time, csv, sys, logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.disable(logging.WARNING)
HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(parents=True, exist_ok=True)
ENGINE = Path("/home/arun/quantifyd/research/41_midsmall400_mq_concentrated/"
              "scripts/02_rs_sweep.py")
_s = importlib.util.spec_from_file_location("rs2", str(ENGINE))
rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)

rs2.BANDS["n250"] = (0, 250)                 # top-250 by traded value = Nifty LargeMidcap 250 proxy
BENCH = rs2.NIFTY_SYM                         # NIFTYBEES (full-history Nifty-50 proxy)
RT = 0.003                                    # 0.3% round-trip (large-mid liquid)
DAY_CASH = (1 + rs2.CASH_ANNUAL) ** (1 / 252)
START = pd.Timestamp("2006-01-01")           # attempt ~20yr; also report 2014+ sub-period
MODERN = pd.Timestamp("2014-01-01")

# ETFs / index / commodity tickers that rank high by traded value but aren't equities
EXCLUDE = {"NIFTYBEES", "NIFTY50", "BANKNIFTY", "INDIAVIX", "NIFTYJR", "NIFTYMID",
           "NIFTYIT", "FINNIFTY", "MIDCPNIFTY", "JUNIORBEES", "GOLDBEES", "SILVERBEES",
           "LIQUIDBEES", "BANKBEES", "ICICIB22", "CPSEETF", "MON100", "MAFANG",
           "SETFNIF50", "SETFGOLD", "SETFNIFBK", "GOLDCASE", "KOTAKGOLD", "AXISGOLD",
           "HDFCGOLD", "GOLDSHARE", "GOLD1", "SILVER1"}


def load():
    """Like rs2.load() but pulls from 2004-01 so a 2006 start has EMA200 + 252d
    momentum warmup. close/tv pivots (date x symbol)."""
    import sqlite3
    con = sqlite3.connect(rs2.DB)
    df = pd.read_sql(
        "SELECT symbol,date,close,volume FROM market_data_unified "
        "WHERE timeframe='day' AND close IS NOT NULL AND date>='2004-01-01' "
        "ORDER BY symbol,date", con, parse_dates=["date"])
    con.close()
    df["tv"] = df["close"] * df["volume"].fillna(0)
    close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    tv = df.pivot_table(index="date", columns="symbol", values="tv").sort_index()
    return close, tv


def momentum(close, asof, score):
    """Causal momentum series over all names (higher = stronger). score in
    {'ret252','ret252_21','radj_z'}. Uses only data <= asof."""
    h = close.loc[:asof]
    if score == "ret252":
        if len(h) <= 253: return None
        return (h.iloc[-1] / h.iloc[-253] - 1.0).dropna()
    if score == "ret252_21":                          # 12-1 (skip most recent 21d)
        if len(h) <= 274: return None
        return (h.iloc[-22] / h.iloc[-274] - 1.0).dropna()
    if score == "radj_z":                             # risk-adjusted z-blend (6m & 12m)
        if len(h) <= 253: return None
        legs = []
        for L in (126, 252):
            ret = h.iloc[-1] / h.iloc[-L - 1] - 1.0
            vol = h.iloc[-L:].pct_change().std() * np.sqrt(252)
            radj = ret / vol.replace(0, np.nan)
            legs.append((radj - radj.mean()) / radj.std(ddof=0))
        return pd.concat(legs, axis=1).mean(axis=1).dropna()
    raise ValueError(score)


def run(close, tv, ema, nbx_ema100, score="ret252", N=15,
        use_stack=True, use_gate=True, rt=RT, stcg=0.0, start=START,
        gate_freq="monthly"):
    """Daily-marked backtest. held[sym] = [value, cost_basis, buydate]."""
    idx = close.index[close.index >= start]
    # month-end rebalance dates from THIS study's start (NOT rs2.month_ends, which
    # hard-codes rs2.START=2014 and would park the book in cash for 2006-2013).
    _s = pd.Series(idx, index=idx)
    me = set(_s.groupby([idx.year, idx.month]).max().values)
    iso = idx.isocalendar()
    wk_last = set(pd.Series(idx, index=idx).groupby(
        [iso.year.values, iso.week.values]).last().values)
    nbx = close[BENCH].ffill()
    e50, e100, e200 = ema[50], ema[100], ema[200]
    cash, held = 1.0, {}
    nav = []; prev = None
    st = dict(tax=0.0, cost=0.0, fills=0, turn_sum=0.0, gate_off=0, contrib={})

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

    def gate_risk_on(d):
        return (not use_gate) or (nbx.loc[d] > nbx_ema100.loc[d]
                                  if pd.notna(nbx_ema100.loc[d]) else True)

    def do_fill(d, px):
        nonlocal cash, held
        sc = momentum(close, d, score)
        if sc is None: return
        uni = rs2.pit_universe(tv, close, d, "n250")
        cand = [s for s in sc.index
                if s in uni and s not in EXCLUDE and s != BENCH]
        if use_stack:                                 # per-stock stacked EMA uptrend
            cand = [s for s in cand
                    if pd.notna(e50[s].loc[d]) and pd.notna(e200[s].loc[d])
                    and e50[s].loc[d] > e100[s].loc[d] > e200[s].loc[d]]
        cand = [s for s in cand if pd.notna(px.get(s, np.nan))]
        if not cand:                                  # nothing qualifies -> cash
            liquidate(d); return
        top = list(sc[cand].sort_values(ascending=False).index[:N])
        tot = E() + cash
        w = tot / len(top)
        turn = len(set(held).symmetric_difference(set(top))) / max(1, 2 * len(top))
        nh = {}
        for s in top:
            nh[s] = held[s] if s in held else [w, w, d]
            nh[s][0] = w
        # cost basis reset for new names already handled; realize tax on dropped
        for s in list(held):
            if s not in nh:
                realize(held[s], d)                   # tax tracked; value folded into tot
        held = nh
        c = tot * rt * turn * 2
        cash = tot - w * len(top) - c
        st['cost'] += c; st['fills'] += 1; st['turn_sum'] += turn

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:                 # daily mark-to-market
            for s, stt in held.items():
                p1 = px.get(s, np.nan); p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    delta = stt[0] * (p1 / p0 - 1.0)
                    stt[0] += delta
                    st['contrib'][s] = st['contrib'].get(s, 0.0) + delta
        if cash > 0: cash *= DAY_CASH

        # regime gate: liquidate immediately when it flips off (monthly or weekly check)
        if use_gate and ((gate_freq == "weekly" and d in wk_last) or (d in me)):
            if not gate_risk_on(d) and held:
                liquidate(d); st['gate_off'] += 1

        if d in me:                                   # monthly rebalance
            if gate_risk_on(d):
                do_fill(d, px)
            elif held:
                liquidate(d)
        nav.append((d, E() + cash)); prev = d
    return _stats(nav, st, start)


def _stats(nav, st, start):
    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    def block(ns):
        yrs = (ns.index[-1] - ns.index[0]).days / 365.25
        cagr = (ns.iloc[-1] / ns.iloc[0]) ** (1 / yrs) - 1
        dr = ns.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
        dn = dr[dr < 0]
        so = dr.mean() / dn.std() * np.sqrt(252) if len(dn) and dn.std() > 0 else np.nan
        dd = ((ns - ns.cummax()) / ns.cummax()).min()
        cal = cagr / abs(dd) if dd < 0 else np.nan
        return dict(cagr=cagr * 100, dd=dd * 100, sharpe=sh, sortino=so,
                    calmar=cal, mult=ns.iloc[-1] / ns.iloc[0], yrs=yrs)
    full = block(n)
    mod = block(n[n.index >= MODERN]) if (n.index >= MODERN).sum() > 250 else None
    last = n.groupby(n.index.year).last()
    py = last.pct_change(); py.iloc[0] = last.iloc[0] / n.iloc[0] - 1
    full["st"] = st; full["nav"] = n; full["py"] = (py * 100).round(1); full["modern"] = mod
    return full


FIELDS = ["config", "score", "N", "stack", "gate", "gate_freq", "rt_pct",
          "cagr", "net_cagr", "post_tax20", "maxdd", "sharpe", "sortino", "calmar",
          "mult", "fills", "avg_turn", "gate_off", "cost_pct",
          "mod_cagr", "mod_maxdd", "mod_calmar"]


def row_of(lbl, score, N, stack, gate, gf, rt, gross, net, tax):
    fills = net['st']['fills'] or 1
    m = net['modern']
    return dict(config=lbl, score=score, N=N, stack=int(stack), gate=int(gate),
                gate_freq=gf, rt_pct=rt * 100,
                cagr=round(gross['cagr'], 1), net_cagr=round(net['cagr'], 1),
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
    print("Loading daily price+tv from VPS DB ...", flush=True)
    close, tv = load()
    print(f"  {close.shape[1]} symbols, {close.index.min().date()}..{close.index.max().date()}",
          flush=True)
    # coverage: names available near start
    for probe in ("2006-06-30", "2010-06-30", "2014-06-30"):
        d = close.index[close.index <= probe][-1]
        uni = rs2.pit_universe(tv, close, d, "n250")
        print(f"  PIT n250 @ {d.date()}: {len(uni)} names", flush=True)

    print("Precomputing EMAs (50/100/200) on all names + NIFTYBEES ...", flush=True)
    ema = {p: close.ewm(span=p, adjust=False, min_periods=p).mean() for p in (50, 100, 200)}
    nbx_ema100 = close[BENCH].ffill().ewm(span=100, adjust=False, min_periods=100).mean()

    out = RES / "ranking.csv"; pyout = RES / "peryear.csv"
    done = set()
    if out.exists():
        with open(out) as f: done = {r['config'] for r in csv.DictReader(f)}
        print(f"  resuming: {len(done)} configs done", flush=True)
    else:
        with open(out, "w", newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    # ---- benchmark (NIFTYBEES buy&hold) ----
    bn = close[BENCH].loc[close.index >= START].dropna(); bn = bn / bn.iloc[0]
    bstat = _stats(list(zip(bn.index, bn.values)), dict(fills=0, turn_sum=0.0, gate_off=0), START)
    print(f"  [ref] NIFTYBEES B&H CAGR={bstat['cagr']:.1f}% DD={bstat['dd']:.1f}% "
          f"Cal={bstat['calmar']:.2f} mult={bstat['mult']:.1f}x", flush=True)

    # grid: (label, score, N, stack, gate, gate_freq, rt)
    cfgs = [
        ("A1_BASE_faithful",     "ret252",    15, True,  True,  "monthly", RT),
        ("A2_no_index_gate",     "ret252",    15, True,  False, "monthly", RT),
        ("A3_no_ema_stack",      "ret252",    15, False, True,  "monthly", RT),
        ("A4_pure_momentum",     "ret252",    15, False, False, "monthly", RT),
        ("B1_mom_12m1m",         "ret252_21", 15, True,  True,  "monthly", RT),
        ("B2_mom_radj_z",        "radj_z",    15, True,  True,  "monthly", RT),
        ("C1_N10",               "ret252",    10, True,  True,  "monthly", RT),
        ("C2_N20",               "ret252",    20, True,  True,  "monthly", RT),
        ("D1_cost_0.1pct",       "ret252",    15, True,  True,  "monthly", 0.001),
        ("D2_cost_0.5pct",       "ret252",    15, True,  True,  "monthly", 0.005),
        ("E1_gate_weekly",       "ret252",    15, True,  True,  "weekly",  RT),
    ]
    pys = {"NIFTYBEES": bstat['py']}
    for lbl, score, N, stack, gate, gf, rt in cfgs:
        if lbl in done:
            print(f"  skip {lbl}", flush=True); continue
        ts = time.time()
        gross = run(close, tv, ema, nbx_ema100, score, N, stack, gate, rt, 0.0, START, gf)
        net = run(close, tv, ema, nbx_ema100, score, N, stack, gate, rt, 0.0, START, gf)  # same (cost in NAV)
        tax = run(close, tv, ema, nbx_ema100, score, N, stack, gate, rt, 0.20, START, gf)
        # gross = zero-cost variant for the gross/net split
        gross = run(close, tv, ema, nbx_ema100, score, N, stack, gate, 0.0, 0.0, START, gf)
        row = row_of(lbl, score, N, stack, gate, gf, rt, gross, net, tax)
        with open(out, "a", newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        pys[lbl] = net['py']
        pd.DataFrame(pys).reindex(range(2006, 2027)).to_csv(pyout)
        if lbl == "A1_BASE_faithful":
            net['nav'].to_csv(RES / "nav_base.csv", header=["nav"])
        print(f"  {lbl:20} grossCAGR={row['cagr']:5.1f}% netCAGR={row['net_cagr']:5.1f}% "
              f"net20={row['post_tax20']:5.1f}% DD={row['maxdd']:6.1f}% "
              f"Sh={row['sharpe']:.2f} Cal={row['calmar']} turn={row['avg_turn']} "
              f"mult={row['mult']}x [{time.time()-ts:.0f}s]", flush=True)
        sys.stdout.flush()

    print(f"\nDone in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
