"""Phase 09 — (A) alternative regime filters vs the lagging SMA200, and
(B) layering the ATH-proximity + 20%-trailing system on top of RS.

Core fixed = mid_120d_N15 + q0.5 quality (the locked winner's stack),
monthly, top-22 buffer, 0.4% RT, 6.5% bear-cash, NIFTYBEES benchmark,
2014-2026. We hold the SELECTION constant and only swap the market-
regime rule (Part A) or add ATH entry/exit layers (Part B), so every
delta is attributable to the change. Honest per-year + metrics + verdict.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
_s = importlib.util.spec_from_file_location(
    "rs2", str(Path(__file__).resolve().parent / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)

BENCH = rs2.NIFTY_SYM
N, Q = 15, 0.50
BUF = int(N * rs2.BUFFER_MULT)
CASH_M = (1 + rs2.CASH_ANNUAL) ** (1 / 12)


# ---------- regime rules: return True = RISK-OFF (go to cash) ----------
def reg_off(h):           return False
def reg_sma(h, n):
    b = h[BENCH].dropna()
    return len(b) >= n and b.iloc[-1] < b.tail(n).mean()
def reg_cross(h):
    b = h[BENCH].dropna()
    return len(b) >= 200 and b.tail(50).mean() < b.tail(200).mean()
def reg_dd1y(h, thr=0.10):
    b = h[BENCH].dropna()
    return len(b) >= 252 and b.iloc[-1] < (1 - thr) * b.tail(252).max()
def reg_mom3m(h):
    b = h[BENCH].dropna()
    return len(b) > 63 and b.iloc[-1] / b.iloc[-64] - 1 < 0
def _atr_ratio(h, n=14):
    b = h[BENCH].dropna()
    if len(b) < 260:
        return None
    tr = b.diff().abs()                     # close-to-close proxy (no OHLC)
    atr = tr.rolling(n).mean()
    rat = (atr / b)
    return rat.iloc[-1], rat.tail(252).median()
def reg_vol(h):
    r = _atr_ratio(h)
    return r is not None and r[0] > 1.3 * r[1]
def reg_sma_and_vol(h):                     # risk-off if EITHER fires
    return reg_sma(h, 200) or reg_vol(h)

REGIMES = {
    "OFF":            reg_off,
    "SMA200(base)":   lambda h: reg_sma(h, 200),
    "SMA100":         lambda h: reg_sma(h, 100),
    "SMA50":          lambda h: reg_sma(h, 50),
    "cross50/200":    reg_cross,
    "DDfrom1yHi>10%": reg_dd1y,
    "mom3m<0":        reg_mom3m,
    "volspike(ATR)":  reg_vol,
    "SMA200+vol":     reg_sma_and_vol,
}


def own_pf(h, s):
    ss = h[s].dropna().tail(252)
    if len(ss) < 120:
        return None
    bl = [ss.iloc[i:i + 21] for i in range(0, len(ss) - 1, 21)]
    r = [x.iloc[-1] / x.iloc[0] - 1 for x in bl
         if len(x) >= 2 and x.iloc[0] > 0]
    return float(np.mean([z > 0 for z in r])) if r else 0.0


def backtest(close, tv, regime_fn, ath_entry=False, trail_exit=False,
             ath_thr=0.10, trail=0.20, stcg=0.0):
    me = rs2.month_ends(close.index)
    me = me[(me >= pd.Timestamp("2014-01-01")) &
            (me <= pd.Timestamp("2026-12-31"))]
    cash, eq = 1.0, 0.0
    held = {}            # sym -> [val, cost, buydate, peakpx]
    last = {}
    nav, tax = [], 0.0
    for dt in me:
        h = close.loc[:dt]
        px = h.iloc[-1]
        if held:
            ne = 0.0
            for s, st in held.items():
                p1 = px.get(s, np.nan)
                if pd.notna(p1) and s in last and last[s] > 0:
                    st[0] *= (1 + p1 / last[s] - 1)
                if pd.notna(p1):
                    st[3] = max(st[3], p1)        # track peak px
                ne += st[0]
            eq = ne
        tot = cash * CASH_M + eq

        # trailing-stop exits (drop names >trail below peak-since-entry)
        if trail_exit and held:
            for s in list(held):
                p1 = px.get(s, np.nan)
                if pd.notna(p1) and held[s][3] > 0 and \
                   p1 <= (1 - trail) * held[s][3]:
                    st = held.pop(s)
                    if stcg:
                        g = st[0] - st[1]
                        if g > 0 and (dt - st[2]).days < 365:
                            tot -= g * stcg; tax += g * stcg

        if regime_fn(h):
            if stcg and held:
                for s, st in held.items():
                    g = st[0] - st[1]
                    if g > 0 and (dt - st[2]).days < 365:
                        tot -= g * stcg; tax += g * stcg
            cash, eq, held, last = tot, 0.0, {}, {}
            nav.append((dt, tot)); continue

        uni = rs2.pit_universe(tv, close, dt, "mid")
        sc = rs2.rs_scores(close, dt, 120)
        if sc is None or not uni:
            nav.append((dt, tot)); continue
        cand = [s for s in sc.index if s in uni and s != BENCH]
        sc = sc[cand].sort_values(ascending=False)
        pick = []
        for s in sc.index:
            pf = own_pf(h, s)
            if pf is None or pf < Q:
                continue
            if ath_entry:
                hh = h[s].dropna()
                if len(hh) < 60 or px.get(s, np.nan) < \
                   (1 - ath_thr) * hh.max():       # within 10% of PIT ATH
                    continue
            pick.append(s)
            if len(pick) >= BUF:
                break
        if not pick:
            cash, eq, held, last = tot, 0.0, {}, {}
            nav.append((dt, tot)); continue
        top = pick[:N]; buf = set(pick[:BUF])
        keep = [s for s in held if s in buf]
        fill = [s for s in top if s not in keep]
        tgt = [s for s in (keep + fill)[:N]
               if pd.notna(px.get(s, np.nan))]
        if not tgt:
            nav.append((dt, tot)); continue
        drop = [s for s in held if s not in set(tgt)]
        if stcg:
            for s in drop:
                st = held[s]; g = st[0] - st[1]
                if g > 0 and (dt - st[2]).days < 365:
                    tot -= g * stcg; tax += g * stcg
        turn = len(set(held).symmetric_difference(set(tgt))) / max(1, 2 * N)
        tot *= (1 - rs2.RT_COST * turn * 2)
        w = tot / len(tgt)
        nh = {}
        for s in tgt:
            if s in held and s not in drop:
                nh[s] = [w, held[s][1], held[s][2], held[s][3]]
            else:
                nh[s] = [w, w, dt, px[s]]
        held = nh
        last = {s: px[s] for s in tgt}
        cash, eq = 0.0, tot
        nav.append((dt, tot))
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
    close, tv = rs2.load()

    print("\n=== PART A: regime-filter alternatives "
          "(core mid_120d_N15 + q0.5) ===", flush=True)
    rowsA, pyA = [], {}
    for name, fn in REGIMES.items():
        m = backtest(close, tv, fn)
        mt = backtest(close, tv, fn, stcg=0.20)
        rowsA.append(dict(regime=name, cagr=round(m['cagr'], 1),
                          post_tax20=round(mt['cagr'], 1),
                          maxdd=round(m['dd'], 1),
                          sharpe=round(m['sh'], 2),
                          calmar=round(m['cal'], 2)))
        pyA[name] = m['py']
        print(f"  {name:16} CAGR={m['cagr']:5.1f}% "
              f"net20={mt['cagr']:5.1f}% DD={m['dd']:6.1f}% "
              f"Sh={m['sh']:.2f} Cal={m['cal']:.2f}", flush=True)
    dfA = pd.DataFrame(rowsA)
    dfA.to_csv(RES / "phase09_regime_alts.csv", index=False)

    # best non-OFF regime by Calmar -> use under the ATH-layer tests
    cand = dfA[dfA.regime != "OFF"].sort_values("calmar",
                                                ascending=False)
    best_name = cand.iloc[0]["regime"]
    best_fn = REGIMES[best_name]
    print(f"\n  Best-Calmar regime = {best_name}", flush=True)

    print("\n=== PART B: ATH-proximity layer on RS "
          f"(regime = {best_name}) ===", flush=True)
    rowsB, pyB = [], {}
    variants = [
        ("base (no ATH layer)", False, False),
        ("+ATH<=10% entry",     True,  False),
        ("+20% trail exit",     False, True),
        ("+ATH entry +trail",   True,  True),
    ]
    for lbl, ae, te in variants:
        m = backtest(close, tv, best_fn, ath_entry=ae, trail_exit=te)
        mt = backtest(close, tv, best_fn, ath_entry=ae, trail_exit=te,
                      stcg=0.20)
        rowsB.append(dict(variant=lbl, cagr=round(m['cagr'], 1),
                          post_tax20=round(mt['cagr'], 1),
                          maxdd=round(m['dd'], 1),
                          sharpe=round(m['sh'], 2),
                          calmar=round(m['cal'], 2)))
        pyB[lbl] = m['py']
        print(f"  {lbl:22} CAGR={m['cagr']:5.1f}% "
              f"net20={mt['cagr']:5.1f}% DD={m['dd']:6.1f}% "
              f"Sh={m['sh']:.2f} Cal={m['cal']:.2f}", flush=True)
    dfB = pd.DataFrame(rowsB)
    dfB.to_csv(RES / "phase09_ath_layer.csv", index=False)

    yrs = sorted(set(range(2014, 2027)))
    ptA = pd.DataFrame({k: pyA[k] for k in pyA}).reindex(yrs)
    ptB = pd.DataFrame({k: pyB[k] for k in pyB}).reindex(yrs)
    ptA.to_csv(RES / "phase09_regime_peryear.csv")
    ptB.to_csv(RES / "phase09_ath_peryear.csv")
    print("\n=== PART A per-year % ===\n" + ptA.to_string())
    print("\n=== PART B per-year % ===\n" + ptB.to_string())
    print("\nBaseline ref: SMA200(base) gross 35.3 / net20 ~29 / DD -24.6 "
          "/ Cal 1.44 ; OFF 37.0 / -29.6 / 1.25")


if __name__ == "__main__":
    main()
