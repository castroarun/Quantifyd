"""Phase 21 — regime-triggered Nifty-PUT hedge: LABELLED SKETCH.

================  THIS IS NOT A BACKTEST RESULT  ================
There is NO historical Nifty option/IV data in-house (verified: 15
days of 2026 snapshots; iv_history.db empty). A put's entire P&L is
the premium, which depends on implied vol we do not have. This script
is an ASSUMPTION-DRIVEN SENSITIVITY SKETCH, not a result:

  IV proxy  = NIFTYBEES trailing 63-day realized vol (annualised)
              × a STATED risk-premium markup (1.0 / 1.15 / 1.30).
  Put price = Black–Scholes, 1-month tenor, r = 6.5%, notional = 1×
              equity (under-hedges mid-cap β>1 — same caveat as the
              short). Strike swept: ATM (K=1.0) and 5% OTM (K=0.95).
  Mechanic  = SMOOTHEST selection core (mid RS-120 + q0.5 + ATH≤10%
              + per-stock-100SMA + 12% trail, monthly, top-22 buffer).
              When the month-end regime is risk-off (Nifty < its
              100-SMA): HOLD the stocks (no cash, no short) and buy a
              1-month Nifty put on equity notional; next month settle
              its payoff vs Nifty's realised monthly move minus the
              modelled premium; roll while risk-off; stop when Nifty
              > 100-SMA. Real skew/term-structure/early-exercise/
              liquidity NOT modelled. Equity STCG@20% applied to the
              stock leg only; option taxation NOT modelled.
Purpose: a DIRECTIONAL read of whether the premium drag is tolerable
and whether it fixes the 2025 short-backfire — i.e. whether buying
real options data is worth it. Numbers here are illustrative only.
================================================================
"""
from __future__ import annotations
import importlib.util, math
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
_s = importlib.util.spec_from_file_location(
    "rs2", str(Path(__file__).resolve().parent / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)

BENCH = rs2.NIFTY_SYM
N, Q, ATH_THR, TRAIL = 15, 0.50, 0.10, 0.12
BUF = int(N * rs2.BUFFER_MULT)
CASH_M = (1 + rs2.CASH_ANNUAL) ** (1 / 12)
RF, TEN = 0.065, 1.0 / 12.0          # risk-free, 1-month tenor


def _N(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_frac(K, sigma):
    """Black–Scholes put price as a FRACTION of spot (S=1)."""
    sigma = max(sigma, 0.06)
    sT = sigma * math.sqrt(TEN)
    d1 = (math.log(1.0 / K) + (RF + 0.5 * sigma * sigma) * TEN) / sT
    d2 = d1 - sT
    return K * math.exp(-RF * TEN) * _N(-d2) - _N(-d1)


def own_pf(cs):
    if len(cs) < 120:
        return None
    bl = [cs.iloc[i:i + 21] for i in range(0, len(cs) - 1, 21)]
    r = [x.iloc[-1] / x.iloc[0] - 1 for x in bl
         if len(x) >= 2 and x.iloc[0] > 0]
    return float(np.mean([z > 0 for z in r])) if r else 0.0


def run(close, tv, mode, K=1.0, iv_mult=1.15, stcg=0.0):
    """mode: 'cash' (=SMOOTHEST ref), 'short' (=MAX-RETURN ref),
             'put' (the sketch)."""
    me = rs2.month_ends(close.index)
    me = me[(me >= pd.Timestamp("2014-01-01")) &
            (me <= pd.Timestamp("2026-12-31"))]
    nb = close[BENCH].reindex(close.index).ffill()
    rv = (nb.pct_change().rolling(63).std() * math.sqrt(252))
    cash, eq = 1.0, 0.0
    held, last = {}, {}
    nav, tax, prev_nb = [], 0.0, None
    for dt in me:
        h = close.loc[:dt]; px = h.iloc[-1]
        nbx = nb.loc[:dt].iloc[-1]
        nf_ret = (nbx / prev_nb - 1) if prev_nb else 0.0
        if held:
            ne = 0.0
            for s, st in held.items():
                p1 = px.get(s, np.nan)
                if pd.notna(p1) and s in last and last[s] > 0:
                    st[0] *= (1 + p1 / last[s] - 1)
                ne += st[0]
            eq = ne
        tot = cash * CASH_M + eq
        b = h[BENCH].dropna()
        roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()

        # settle LAST month's hedge overlay on this month's Nifty move
        if prev_nb is not None and eq > 0 and getattr(run, "_hedge", 0):
            if run._hmode == "short":
                tot += eq * (-1.0 * nf_ret)
            elif run._hmode == "put":
                payoff = max(run._K - (1.0 + nf_ret), 0.0)   # K - S_T/S_0
                tot += eq * (payoff - run._prem)              # net of premium
        run._hedge = 0

        def _real(stt):
            nonlocal tot, tax
            g = stt[0] - stt[1]
            if stcg and g > 0 and (dt - stt[2]).days < 365:
                t = g * stcg; tot -= t; tax += t

        # SMOOTHEST stock-level exits
        for s in list(held):
            cs = h[s].dropna(); p1 = px.get(s, np.nan)
            if (len(cs) >= 100 and cs.iloc[-1] < cs.tail(100).mean()) or \
               (pd.notna(p1) and held[s][3] > 0 and
                p1 <= (1 - TRAIL) * held[s][3]):
                _real(held[s]); cash += held.pop(s)[0]

        if roff and mode == "cash":
            for s in list(held):
                _real(held[s])
            cash, eq, held, last = tot, 0.0, {}, {}
            nav.append((dt, tot)); prev_nb = nbx; continue

        # arm this month's hedge overlay (settled next month)
        if roff and mode in ("short", "put"):
            run._hedge = 1; run._hmode = mode
            if mode == "put":
                sig = (rv.loc[:dt].iloc[-1] if pd.notna(
                    rv.loc[:dt].iloc[-1]) else 0.15) * iv_mult
                run._K = K
                run._prem = bs_put_frac(K, sig)

        uni = rs2.pit_universe(tv, close, dt, "mid")
        sc = rs2.rs_scores(close, dt, 120)
        if sc is None or not uni:
            nav.append((dt, tot)); prev_nb = nbx; continue
        cand = [s for s in sc.index if s in uni and s != BENCH]
        sc = sc[cand].sort_values(ascending=False)
        pick = []
        for s in sc.index:
            cs = h[s].dropna()
            pf = own_pf(cs.tail(252))
            if pf is None or pf < Q:
                continue
            if len(cs) < 100 or cs.iloc[-1] < cs.tail(100).mean():
                continue
            if px.get(s, np.nan) < (1 - ATH_THR) * cs.max():
                continue
            pick.append(s)
            if len(pick) >= BUF:
                break
        if not pick:
            nav.append((dt, tot)); prev_nb = nbx; continue
        top = pick[:N]; bufset = set(pick[:BUF])
        keep = [s for s in held if s in bufset]
        fill = [s for s in top if s not in keep]
        tgt = [s for s in (keep + fill)[:N] if pd.notna(px.get(s, np.nan))]
        if not tgt:
            nav.append((dt, tot)); prev_nb = nbx; continue
        drop = [s for s in held if s not in set(tgt)]
        for s in drop:
            _real(held[s])
        turn = len(set(held).symmetric_difference(set(tgt))) / max(1, 2 * N)
        tot *= (1 - rs2.RT_COST * turn * 2)
        w = tot / len(tgt)
        nh = {}
        for s in tgt:
            nh[s] = ([w, held[s][1], held[s][2], held[s][3]]
                     if (s in held and s not in drop) else [w, w, dt, px[s]])
        held = nh; last = {s: px[s] for s in tgt}
        cash, eq = 0.0, tot
        nav.append((dt, tot)); prev_nb = nbx
    run._hedge = 0
    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    mr = n.pct_change().dropna()
    sh = mr.mean() / mr.std() * np.sqrt(12) if mr.std() > 0 else 0
    py = n.groupby(n.index.year).last().pct_change()
    py.iloc[0] = n.groupby(n.index.year).last().iloc[0] / 1.0 - 1
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan,
                py=(py * 100).round(1))


def main():
    print("Loading ...", flush=True)
    print("*** LABELLED SKETCH — modelled IV, NOT a backtest result ***",
          flush=True)
    close, tv = rs2.load()
    rows, pys = [], {}
    refs = [("SMOOTHEST (cash) ref", "cash", 1.0, 1.0),
            ("MAX-RETURN (short) ref", "short", 1.0, 1.0)]
    sk = [(f"PUT {('ATM' if K==1.0 else '5%OTM')} ivx{m}", "put", K, m)
          for K in (1.0, 0.95) for m in (1.0, 1.15, 1.30)]
    for lbl, mode, K, m in refs + sk:
        g = run(close, tv, mode, K, m)
        t = run(close, tv, mode, K, m, stcg=0.20)
        rows.append(dict(config=lbl, cagr=round(g['cagr'], 1),
                         post_tax20=round(t['cagr'], 1),
                         maxdd=round(g['dd'], 1), sharpe=round(g['sh'], 2),
                         calmar=round(g['cal'], 2)))
        pys[lbl] = g['py']
        print(f"  {lbl:24} CAGR={g['cagr']:5.1f}% net20={t['cagr']:5.1f}%"
              f" DD={g['dd']:6.1f}% Sh={g['sh']:.2f} Cal={g['cal']:.2f}",
              flush=True)
    pd.DataFrame(rows).to_csv(RES / "phase21_put_sketch.csv", index=False)
    pd.DataFrame(pys).reindex(range(2014, 2027)).to_csv(
        RES / "phase21_put_sketch_peryear.csv")
    print("\n=== PUT-HEDGE SKETCH (ILLUSTRATIVE, NOT A RESULT) ===\n" +
          pd.DataFrame(rows).to_string(index=False))
    print("\nCAVEAT: modelled IV = realised-vol×markup; no skew/term-"
          "structure/liquidity/option-tax. Compare DIRECTIONALLY only — "
          "does the put fix 2025 vs short, and is the premium drag "
          "tolerable vs SMOOTHEST(cash)? Decision-grade needs real "
          "Nifty options data.")


if __name__ == "__main__":
    main()
