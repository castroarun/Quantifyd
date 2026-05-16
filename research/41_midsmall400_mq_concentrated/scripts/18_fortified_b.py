"""Phase 18 — FORTIFIED-B: emergent-cash variant, NO market regime gate.

User design. There is NO Nifty 100-SMA market gate. Rules:
  * A holding is SOLD only if (its close < its own 100-day SMA) OR
    (price <= 0.88 * peak-since-entry, i.e. 12% trailing stop).
  * It is NOT sold merely because its RS rank slipped — winners are
    held until they actually break (no forced RS rotation / no
    top-22 buffer churn).
  * Freed slots are refilled from RS-ranked names passing q0.5 AND
    within 10% of all-time-high AND above their own 100-day SMA, that
    aren't already held, up to 15 names total.
  * EMERGENT DE-RISK: in a crash, holdings hit their stops and almost
    nothing is near its ATH -> few/no eligible replacements -> the
    freed capital simply stays in CASH (6.5% p.a.). No explicit timer.
  * New buys are equal-weight at total/15 per slot; surviving holdings
    are NOT trimmed (let winners run). Cash = total - sum(held).
  * Checks (exits + refills) run on a chosen cadence: MONTHLY or
    WEEKLY (both tested; pick the better).

Core universe/signal identical to the family: PIT mid-cap band
(rank 101-250 by 6mo median traded value), RS-120 vs NIFTYBEES,
q0.5 quality, ATH<=10% entry, 0.4% round-trip cost, 6.5% cash,
2014-2026. Daily-marked NAV (true daily MaxDD; compare WITHIN this
table / to Phase-15 daily figures, not to month-end Ph09-13).
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
N, Q, ATH_THR, TRAIL = 15, 0.50, 0.10, 0.12
DAY_CASH = (1 + rs2.CASH_ANNUAL) ** (1 / 252)
START = pd.Timestamp("2014-01-01")


def own_pf(cs_tail):
    if len(cs_tail) < 120:
        return None
    bl = [cs_tail.iloc[i:i + 21] for i in range(0, len(cs_tail) - 1, 21)]
    r = [x.iloc[-1] / x.iloc[0] - 1 for x in bl
         if len(x) >= 2 and x.iloc[0] > 0]
    return float(np.mean([z > 0 for z in r])) if r else 0.0


def run(close, tv, cadence, force_rotation, stcg=0.0):
    """No market gate. cadence M/W. force_rotation:
       True  (R) = FORTIFIED selection: also force RS rotation (drop
                   held names out of the top-22 buffer each cadence).
       False (H) = hold-until-break: a name is sold ONLY by the
                   stock-level exits; survivors never dropped on RS.
    Both: stock-level exits (own-100SMA breach OR 12% trail); refill
    from RS leaders passing q0.5 + ATH<=10% + above-own-100SMA;
    unfilled slots stay CASH (emergent de-risk)."""
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    wk = pd.Series(idx, index=idx)
    wk_last = set(wk.groupby([idx.isocalendar().year,
                              idx.isocalendar().week]).last().values)
    BUF = int(N * rs2.BUFFER_MULT)
    cash = 1.0
    held = {}                       # sym -> [val, cost, buydate, peak]
    nav, tax, prev_d, sells = [], 0.0, None, 0

    def _sell(s, d):
        nonlocal cash, tax, sells
        st = held.pop(s)
        g = st[0] - st[1]
        t = (g * stcg if stcg and g > 0 and (d - st[2]).days < 365
             else 0.0)
        cash += st[0] - t; tax += t; sells += 1

    for d in idx:
        px = close.loc[d]
        if held and prev_d is not None:
            for s, st in held.items():
                p1 = px.get(s, np.nan)
                p0 = close.at[prev_d, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    st[0] *= p1 / p0
                    st[3] = max(st[3], p1)
        if cash > 0:
            cash *= DAY_CASH

        is_check = (d in me) if cadence == "M" else (d in wk_last)
        if is_check:
            h = close.loc[:d]
            # 1) stock-level exits (the only sells in variant H)
            for s in list(held):
                cs = h[s].dropna()
                p1 = px.get(s, np.nan)
                brk_sma = len(cs) >= 100 and cs.iloc[-1] < cs.tail(100).mean()
                brk_tr = (pd.notna(p1) and held[s][3] > 0 and
                          p1 <= (1 - TRAIL) * held[s][3])
                if brk_sma or brk_tr:
                    _sell(s, d)
            # 2) build RS-ranked eligible list (q0.5 + ATH<=10% + own100SMA)
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
                    if len(cs) < 100 or cs.iloc[-1] < cs.tail(100).mean():
                        continue
                    if px.get(s, np.nan) < (1 - ATH_THR) * cs.max():
                        continue
                    elig.append(s)
                    if len(elig) >= BUF:
                        break
            # 2a) forced RS rotation (variant R only): drop held names
            #     no longer in the top-22 eligible buffer
            if force_rotation and elig:
                bufset = set(elig[:BUF])
                for s in list(held):
                    if s not in bufset:
                        _sell(s, d)
            # 3) refill empty slots from top eligible not already held;
            #    unfilled -> cash (emergent de-risk)
            if elig and len(held) < N:
                total = cash + sum(v[0] for v in held.values())
                slot = total / N
                for s in elig:
                    if len(held) >= N:
                        break
                    if s in held:
                        continue
                    buy = min(slot, cash)
                    if buy <= 1e-9:
                        break
                    cash -= buy + buy * rs2.RT_COST
                    held[s] = [buy, buy, d, px[s]]
        nav.append((d, cash + sum(v[0] for v in held.values())))
        prev_d = d
    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    pyr = n.groupby(n.index.year).last().pct_change()
    pyr.iloc[0] = n.groupby(n.index.year).last().iloc[0] / n.iloc[0] - 1
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan,
                sells=sells, pyr=(pyr * 100).round(1))


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    rows, pys = [], {}
    variants = [("R", True), ("H", False)]   # R=forced RS rotation, H=hold
    for vk, fr in variants:
        for cad in ("M", "W"):
            lbl = f"FORTIFIED-B {vk}-{cad}"
            g = run(close, tv, cad, fr)
            t = run(close, tv, cad, fr, stcg=0.20)
            rows.append(dict(variant=lbl,
                             rule=("forced-RS-rot" if fr else "hold-til-break"),
                             cadence=cad, cagr=round(g['cagr'], 1),
                             post_tax20=round(t['cagr'], 1),
                             maxdd_daily=round(g['dd'], 1),
                             sharpe=round(g['sh'], 2),
                             calmar=round(g['cal'], 2),
                             total_sells=g['sells']))
            pys[lbl] = g['pyr']
            print(f"  {lbl:16} CAGR={g['cagr']:5.1f}% "
                  f"net20={t['cagr']:5.1f}% dailyDD={g['dd']:6.1f}% "
                  f"Sh={g['sh']:.2f} Cal={g['cal']:.2f} "
                  f"sells={g['sells']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "phase18_fortified_b.csv", index=False)
    pt = pd.DataFrame(pys).reindex(range(2014, 2027))
    pt.to_csv(RES / "phase18_fortified_b_peryear.csv")
    print("\n=== FORTIFIED-B ===\n" + df.to_string(index=False))
    print("\n=== per-year % ===\n" + pt.to_string())
    print("\nReference (daily-marked engine, Phase 15): "
          "SMOOTHEST(weekly) Cal 1.65/DD-22.2 ; MAX-RETURN(M) Cal 1.00/"
          "DD-32.6 ; FORTIFIED(M) Cal 1.02/DD-31.7. "
          "VERDICT: FORTIFIED-B wins if it beats SMOOTHEST(weekly) on "
          "Calmar, i.e. emergent cash >= explicit regime gate.")


if __name__ == "__main__":
    main()
