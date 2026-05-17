"""Phase 25 — keep-top8 + CONDITIONAL REFILL (user spec).

On a weekly risk-off (Nifty < 100-SMA): keep the 8 highest-RS of
CURRENT holdings (unconditional — our survivors), sell the rest, then
try to refill the freed slots with non-held names that STILL pass the
FULL gate (q0.5 + above-own-100SMA + within-10%-of-ATH), RS-ranked.
Deploy ~NAV/15 per qualifying name (the per-slot size), capped at
`cap` total invested names; unfilled slots stay cash. The strict gate
is the panic circuit-breaker: in a true crash ≈nothing qualifies →
collapses to cash; in a shallow dip it parks in genuinely strong names.

References use the UNMODIFIED validated Phase-22 engine so the
baselines are byte-identical; only the refill path is new code.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
SC = Path(__file__).resolve().parent
_p = importlib.util.spec_from_file_location(
    "p22", str(SC / "22_smoothest_variants.py"))
p22 = importlib.util.module_from_spec(_p); _p.loader.exec_module(p22)
rs2 = p22.rs2
BENCH, N, Q, ATH_THR, TRAIL = (rs2.NIFTY_SYM, p22.N, p22.Q,
                               p22.ATH_THR, p22.TRAIL)
BUF, DAY_CASH, RT = p22.BUF, p22.DAY_CASH, p22.RT
START = p22.START
own_pf = p22.own_pf


def run_refill(close, tv, K=8, cap=15, L_sma=100, stcg=0.0):
    """keep-top-K of holdings on weekly risk-off, then gated-refill the
    freed slots (full filter) up to `cap` invested names, rest cash."""
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    wk = pd.Series(idx, index=idx)
    wk_last = set(wk.groupby([idx.isocalendar().year,
                              idx.isocalendar().week]).last().values)
    cash, held, prev = 1.0, {}, None        # sym->[val,cost,buydt,peak]
    nav, tax = [], 0.0

    def E():
        return sum(v[0] for v in held.values())

    def realize(stt, d):
        nonlocal tax
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; tax += t; return t
        return 0.0

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, st in held.items():
                p1 = px.get(s, np.nan)
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    st[0] *= p1 / p0
                    st[3] = max(st[3], p1)
        if cash > 0:
            cash *= DAY_CASH
        h = close.loc[:d]

        # ---- WEEKLY regime action ----
        if d in wk_last:
            b = h[BENCH].dropna()
            roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if roff:
                # 1) keep top-K of current holdings (unconditional)
                if len(held) > K:
                    sc = rs2.rs_scores(close, d, 120)
                    rk = {s: (sc.get(s, -9e9) if sc is not None else 0)
                          for s in held}
                    keep = set(sorted(rk, key=rk.get, reverse=True)[:K])
                    for s in list(held):
                        if s not in keep:
                            t = realize(held[s], d)
                            cash += held.pop(s)[0] - t
                # 2) gated refill of freed slots, capped at `cap`
                slots = cap - len(held)
                if slots > 0:
                    sc = rs2.rs_scores(close, d, 120)
                    uni = rs2.pit_universe(tv, close, d, "mid")
                    if sc is not None and uni:
                        tot = E() + cash
                        per = tot / N            # one slot ≈ NAV/15
                        for s in sc.sort_values(ascending=False).index:
                            if slots <= 0 or cash < per * 0.5:
                                break
                            if (s == BENCH or s in held or s not in uni):
                                continue
                            cs = h[s].dropna()
                            pf = own_pf(cs.tail(252))
                            if pf is None or pf < Q:
                                continue
                            if len(cs) < L_sma or \
                               cs.iloc[-1] < cs.tail(L_sma).mean():
                                continue
                            if px.get(s, np.nan) < (1 - ATH_THR) * cs.max():
                                continue
                            buy = min(per, cash)
                            cash -= buy + buy * RT
                            held[s] = [buy, buy, d, px[s]]
                            slots -= 1

        # ---- MONTHLY selection (rebuild to N when risk-on) ----
        if d in me:
            for s in list(held):                 # stock-level exits
                cs = h[s].dropna(); p1 = px.get(s, np.nan)
                if (len(cs) >= L_sma and
                        cs.iloc[-1] < cs.tail(L_sma).mean()) or \
                   (pd.notna(p1) and held[s][3] > 0 and
                        p1 <= (1 - TRAIL) * held[s][3]):
                    t = realize(held[s], d); cash += held.pop(s)[0] - t
            b = h[BENCH].dropna()
            roff_now = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if not roff_now:                     # full rebuild only risk-on
                uni = rs2.pit_universe(tv, close, d, "mid")
                sc = rs2.rs_scores(close, d, 120)
                elig = []
                if sc is not None and uni:
                    for s in sc.sort_values(ascending=False).index:
                        if s == BENCH or s not in uni:
                            continue
                        cs = h[s].dropna()
                        pf = own_pf(cs.tail(252))
                        if pf is None or pf < Q:
                            continue
                        if len(cs) < L_sma or \
                           cs.iloc[-1] < cs.tail(L_sma).mean():
                            continue
                        if px.get(s, np.nan) < (1 - ATH_THR) * cs.max():
                            continue
                        elig.append(s)
                        if len(elig) >= BUF:
                            break
                if elig:
                    bufset = set(elig[:BUF])
                    for s in list(held):
                        if s not in bufset:
                            t = realize(held[s], d)
                            cash += held.pop(s)[0] - t
                    tot = E() + cash
                    names = ([s for s in held] +
                             [s for s in elig[:N] if s not in held])[:N]
                    if names:
                        w = tot / len(names)
                        turn = (len(set(held).symmetric_difference(
                            set(names))) / max(1, 2 * N))
                        nh = {}
                        for s in names:
                            nh[s] = (held[s] if s in held
                                     else [0, 0, d, px[s]])
                            nh[s] = [w, (nh[s][1] if s in held else w),
                                     (nh[s][2] if s in held else d),
                                     (nh[s][3] if s in held else px[s])]
                        held = nh
                        cash = tot - w * len(names)
                        cash -= tot * RT * turn * 2
        nav.append((d, E() + cash))
        prev = d

    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    py = n.groupby(n.index.year).last().pct_change()
    py.iloc[0] = n.groupby(n.index.year).last().iloc[0] / n.iloc[0] - 1
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan,
                py=(py * 100).round(1))


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    rows, pys = [], {}

    def add(lbl, g, t):
        rows.append(dict(config=lbl, cagr=round(g['cagr'], 1),
                         post_tax20=round(t['cagr'], 1),
                         maxdd=round(g['dd'], 1), sharpe=round(g['sh'], 2),
                         calmar=round(g['cal'], 2)))
        pys[lbl] = g['py']
        print(f"  {lbl:34} CAGR={g['cagr']:5.1f}% net20={t['cagr']:5.1f}%"
              f" DD={g['dd']:6.1f}% Sh={g['sh']:.2f} Cal={g['cal']:.2f}",
              flush=True)

    print("ref-A BASE allcash (validated Ph22 engine) ...", flush=True)
    add("ref-A BASE SMOOTHEST(allcash)",
        p22.run(close, tv, "allcash", 0.0, 0, 100),
        p22.run(close, tv, "allcash", 0.0, 0, 100, stcg=0.20))
    print("ref-B keep-top8 holdings-only (validated) ...", flush=True)
    add("ref-B keep-top8 (holdings-only)",
        p22.run(close, tv, "keeptop", 0.0, 8, 100),
        p22.run(close, tv, "keeptop", 0.0, 8, 100, stcg=0.20))
    for cap, tag in ((15, "V1 cond-refill cap-15"),
                     (12, "V2 cond-refill cap-12"),
                     (10, "V3 cond-refill cap-10")):
        print(f"{tag} ...", flush=True)
        add(tag, run_refill(close, tv, 8, cap, 100),
            run_refill(close, tv, 8, cap, 100, stcg=0.20))

    pd.DataFrame(rows).to_csv(RES / "phase25_cond_refill.csv", index=False)
    pd.DataFrame(pys).reindex(range(2014, 2027)).to_csv(
        RES / "phase25_cond_refill_peryear.csv")
    print("\n=== PHASE 25 — keep-top8 + conditional refill ===\n" +
          pd.DataFrame(rows).to_string(index=False))
    print("\nVERDICT RULE: a refill cap wins only if it beats ref-B "
          "keep-top8 on Calmar without losing post-tax CAGR.")


if __name__ == "__main__":
    main()
