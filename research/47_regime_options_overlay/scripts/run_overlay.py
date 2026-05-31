"""Regime risk-off options/ETF overlay on the research/41 midcap-momentum core.

When the month-end regime is risk-off (NIFTYBEES < 100-SMA), apply one of several
risk-off actions instead of cash, settled the next month on Nifty's realised move,
rolled monthly while risk-off, removed on risk-on. Options priced via Black-Scholes
with IV = NIFTYBEES 63d realised vol x markup (NO historical IV -> directional only).

Reuses research/41 `02_rs_sweep.py` for data + the RS selection helpers, and mirrors
its phase-21 sketch mechanics (stock-level exits, costs, regime).
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

R41 = Path(__file__).resolve().parents[2] / "41_midsmall400_mq_concentrated" / "scripts"
_s = importlib.util.spec_from_file_location("rs2", str(R41 / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rs2)

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
RES.mkdir(exist_ok=True)

BENCH = rs2.NIFTY_SYM
N, Q, ATH_THR, TRAIL = 15, 0.50, 0.10, 0.12
BUF = int(N * rs2.BUFFER_MULT)
CASH_M = (1 + rs2.CASH_ANNUAL) ** (1 / 12)
RF, TEN = 0.065, 1.0 / 12.0

OVERLAY = {"short", "longput", "putspread"}        # hold midcaps + hedge
REPLACE = {"cash", "cc_etf", "shortput", "callspread"}  # liquidate -> park


def _Nf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put(K, sig):
    sig = max(sig, 0.06); sT = sig * math.sqrt(TEN)
    d1 = (math.log(1.0 / K) + (RF + 0.5 * sig * sig) * TEN) / sT
    return K * math.exp(-RF * TEN) * _Nf(-(d1 - sT)) - _Nf(-d1)


def bs_call(K, sig):
    sig = max(sig, 0.06); sT = sig * math.sqrt(TEN)
    d1 = (math.log(1.0 / K) + (RF + 0.5 * sig * sig) * TEN) / sT
    return _Nf(d1) - K * math.exp(-RF * TEN) * _Nf(d1 - sT)


def own_pf(cs):
    if len(cs) < 120:
        return None
    bl = [cs.iloc[i:i + 21] for i in range(0, len(cs) - 1, 21)]
    r = [x.iloc[-1] / x.iloc[0] - 1 for x in bl if len(x) >= 2 and x.iloc[0] > 0]
    return float(np.mean([z > 0 for z in r])) if r else 0.0


def arm(mode, iv, sz):
    """Return params dict for the armed action (premiums fixed at arm time)."""
    if mode == "short":
        return dict(mode=mode, sz=sz)
    if mode == "longput":
        K = 0.95 if sz < 0 else 1.0  # placeholder; K set by caller variant
        return dict(mode=mode, sz=abs(sz))
    return dict(mode=mode, sz=sz, iv=iv)


def settle_overlay(p, nf, strikes):
    """Overlay P&L as a fraction of the hedged notional, given Nifty move nf."""
    ST = 1.0 + nf
    if p["mode"] == "short":
        return -p["sz"] * nf
    if p["mode"] == "longput":
        K = strikes["put_long"]
        prem = bs_put(K, p["iv"])
        return p["sz"] * (max(K - ST, 0.0) - prem)
    if p["mode"] == "putspread":
        Kb, Ks = strikes["ps_buy"], strikes["ps_sell"]
        cost = bs_put(Kb, p["iv"]) - bs_put(Ks, p["iv"])
        payoff = max(Kb - ST, 0.0) - max(Ks - ST, 0.0)
        return p["sz"] * (payoff - cost)
    return 0.0


def park_return(p, nf, strikes):
    """Growth MULTIPLE of parked capital over the month, given Nifty move nf."""
    ST = 1.0 + nf
    if p["mode"] == "cash":
        return CASH_M
    if p["mode"] == "cc_etf":
        Kc = strikes["cc_call"]
        prem = bs_call(Kc, p["iv"])
        return min(ST, Kc) + prem                      # ETF capped at Kc + premium
    if p["mode"] == "shortput":
        K = strikes["sp_strike"]
        prem = bs_put(K, p["iv"])
        return CASH_M + p["sz"] * (prem - max(K - ST, 0.0))
    if p["mode"] == "callspread":
        Kb, Ks = strikes["cs_buy"], strikes["cs_sell"]
        cost = bs_call(Kb, p["iv"]) - bs_call(Ks, p["iv"])
        payoff = max(ST - Kb, 0.0) - max(ST - Ks, 0.0)
        return CASH_M + p["sz"] * (payoff - cost)
    return CASH_M


def run(close, tv, mode, sz=1.0, iv_mult=1.15, strikes=None, stcg=0.0, log=None):
    strikes = strikes or {}
    me = rs2.month_ends(close.index)
    me = me[(me >= pd.Timestamp("2014-01-01")) & (me <= pd.Timestamp("2026-12-31"))]
    nb = close[BENCH].reindex(close.index).ffill()
    rv = nb.pct_change().rolling(63).std() * math.sqrt(252)

    cash, eq = 1.0, 0.0
    held, last = {}, {}
    parked = None            # (value, params) when in a REPLACE structure
    armedO = None            # (notional, params) when an OVERLAY is on
    nav, prev_nb = [], None

    for dt in me:
        h = close.loc[:dt]; px = h.iloc[-1]
        nbx = nb.loc[:dt].iloc[-1]
        nf = (nbx / prev_nb - 1) if prev_nb else 0.0

        # mark held stocks
        if held:
            ne = 0.0
            for s, st in held.items():
                p1 = px.get(s, np.nan)
                if pd.notna(p1) and s in last and last[s] > 0:
                    st[0] *= (p1 / last[s])
                ne += st[0]
            eq = ne

        # ---- settle prior month's risk-off action ----
        if parked is not None:
            base, pp = parked
            pr = park_return(pp, nf, strikes)
            val = base * pr
            tot = val
            if log is not None:                       # isolate marginal vs cash
                log.append((dt, pp["mode"], nf, pr - CASH_M, base))
            cash, eq, held, last, parked = val, 0.0, {}, {}, None
        else:
            tot = cash * CASH_M + eq
            if armedO is not None:
                notion, pp = armedO
                tot += notion * settle_overlay(pp, nf, strikes)
                armedO = None

        b = h[BENCH].dropna()
        roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
        iv = (rv.loc[:dt].iloc[-1] if pd.notna(rv.loc[:dt].iloc[-1]) else 0.15) * iv_mult

        def _real(stt):
            nonlocal tot
            g = stt[0] - stt[1]
            if stcg and g > 0 and (dt - stt[2]).days < 365:
                tot -= g * stcg

        # stock-level exits (when holding)
        for s in list(held):
            cs = h[s].dropna(); p1 = px.get(s, np.nan)
            if (len(cs) >= 100 and cs.iloc[-1] < cs.tail(100).mean()) or \
               (pd.notna(p1) and held[s][3] > 0 and p1 <= (1 - TRAIL) * held[s][3]):
                _real(held[s]); cash += held.pop(s)[0]

        # ---- risk-off REPLACE: liquidate -> park ----
        if roff and mode in REPLACE:
            for s in list(held):
                _real(held[s])
            held, last, eq = {}, {}, 0.0
            parked = (tot, dict(mode=mode, sz=sz, iv=iv))
            nav.append((dt, tot)); prev_nb = nbx; cash = 0.0
            continue

        # ---- risk-off OVERLAY: arm hedge, keep holding ----
        if roff and mode in OVERLAY:
            armedO = (tot, dict(mode=mode, sz=sz, iv=iv))
        # (risk-on, or overlay risk-off: fall through to selection)

        # ---- monthly RS selection ----
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

    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    mr = n.pct_change().dropna()
    sh = mr.mean() / mr.std() * np.sqrt(12) if mr.std() > 0 else 0
    yend = n.groupby(n.index.year).last()
    py = yend.pct_change()
    py.iloc[0] = yend.iloc[0] - 1
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan, py=py * 100, nav=n)


STRK = dict(put_long=1.0, ps_buy=1.0, ps_sell=0.90,
            cc_call=1.03, sp_strike=0.95, cs_buy=1.0, cs_sell=1.05)


def main():
    print("loading research/41 panel...", flush=True)
    print("*** MODELLED IV — DIRECTIONAL SKETCH, NOT DECISION-GRADE ***", flush=True)
    close, tv = rs2.load()

    configs = [
        ("CASH (baseline)", "cash", 1.0, 1.15),
        ("SHORT 1x (baseline)", "short", 1.0, 1.15),
        ("SHORT 1.3x", "short", 1.3, 1.15),
        ("LONGPUT ATM 1x", "longput", 1.0, 1.15),
        ("LONGPUT ATM 1.3x", "longput", 1.3, 1.15),
        ("PUTSPREAD 1x", "putspread", 1.0, 1.15),
        ("PUTSPREAD 1.3x", "putspread", 1.3, 1.15),
        ("CALLSPREAD reentry", "callspread", 1.0, 1.15),
        ("SHORTPUT income", "shortput", 1.0, 1.15),
        ("CC_ETF (your idea)", "cc_etf", 1.0, 1.15),
    ]
    rows, pys = [], {}
    for lbl, mode, sz, ivm in configs:
        r = run(close, tv, mode, sz=sz, iv_mult=ivm, strikes=STRK)
        py = r["py"]
        y20 = py.get(2020, np.nan); y25 = py.get(2025, np.nan)
        rows.append(dict(config=lbl, CAGR=round(r["cagr"], 1), MaxDD=round(r["dd"], 1),
                         Sharpe=round(r["sh"], 2), Calmar=round(r["cal"], 2),
                         y2020=round(y20, 1), y2025=round(y25, 1)))
        pys[lbl] = py.round(1)
        print(f"  {lbl:22} CAGR={r['cagr']:5.1f}% DD={r['dd']:6.1f}% "
              f"Cal={r['cal']:.2f} Sh={r['sh']:.2f} 2020={y20:+.0f}% 2025={y25:+.0f}%",
              flush=True)

    # IV sensitivity on the option modes
    print("\nIV markup sensitivity (CAGR / MaxDD):", flush=True)
    for mode, lbl in [("longput", "LONGPUT ATM 1x"), ("putspread", "PUTSPREAD 1x"),
                      ("cc_etf", "CC_ETF"), ("shortput", "SHORTPUT")]:
        cells = []
        for ivm in (1.0, 1.15, 1.30):
            r = run(close, tv, mode, sz=1.0, iv_mult=ivm, strikes=STRK)
            cells.append(f"x{ivm}: {r['cagr']:.1f}%/{r['dd']:.0f}%")
        print(f"  {lbl:16} " + "  ".join(cells), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "overlay_compare.csv", index=False)
    pd.DataFrame(pys).reindex(range(2014, 2027)).round(1).to_csv(RES / "overlay_peryear.csv")
    print("\n=== RISK-OFF ACTION COMPARISON (modelled IV x1.15) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
