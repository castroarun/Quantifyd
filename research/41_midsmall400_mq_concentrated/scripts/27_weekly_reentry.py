"""Phase 27 — Weekly RE-ENTRY vs month-end re-entry (fast-out / fast-in test).

User question: the locked SMOOTHEST engine SELLS on the weekly clock but only
BUYS back at month-end (fast out / slow in). That asymmetry was never A/B
tested — every prior phase varied EXIT speed (Ph15 daily/weekly/monthly gate,
Ph19 staggered exit); re-entry was always month-end.

This adds ONE switch, `weekly_reentry`: on the first weekly risk-ON bar AFTER a
de-risk, immediately rebuild to full (15 names, current eligibles) using the
EXACT same fill code as the month-end path — instead of waiting for the next
monthly rebalance. Nothing else changes. With weekly_reentry=False the engine
is byte-faithful to Phase 22 (sanity-gated below).

Core (unchanged): PIT mid band, RS-120 vs NIFTYBEES, q0.5, ATH<=10% entry,
N=15, top-22 buffer, per-stock-100SMA + 12% trail at month-ends, weekly 100-SMA
market gate, 0.4% RT, 6.5% cash, 2014-2026, daily-marked. Each config: gross +
post-tax@20% + per-year + re-entry-event / fill / cost counts.
"""
from __future__ import annotations
import importlib.util, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
SCR = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("rs2", str(SCR / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)
_p = importlib.util.spec_from_file_location(
    "p22", str(SCR / "22_smoothest_variants.py"))
p22 = importlib.util.module_from_spec(_p); _p.loader.exec_module(p22)

BENCH = rs2.NIFTY_SYM
N, Q, ATH_THR, TRAIL, L_SMA = 15, 0.50, 0.10, 0.12, 100
BUF = int(N * rs2.BUFFER_MULT)
DAY_CASH = (1 + rs2.CASH_ANNUAL) ** (1 / 252)
RT = rs2.RT_COST
START = pd.Timestamp("2014-01-01")
own_pf = p22.own_pf


def run(close, tv, mode, K=0, weekly_reentry=False, stcg=0.0):
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    wk_last = set(pd.Series(idx, index=idx).groupby(
        [idx.isocalendar().year, idx.isocalendar().week]).last().values)
    cash, held = 1.0, {}     # sym -> [val, cost, buydt, peak]
    tgt = 1.0
    derisked = False
    nav, prev = [], None
    st = dict(tax=0.0, cost=0.0, fills=0, wk_reentry=0, derisk_ev=0)

    def E():
        return sum(v[0] for v in held.values())

    def realize(stt, d):
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; st['tax'] += t; return t
        return 0.0

    def fill(d, px, h):
        # EXACT month-end fill logic (buffer rotation + full deploy + cost).
        # Used by BOTH the month-end path and the weekly re-entry path so the
        # only difference between baseline and variant is WHEN this fires.
        nonlocal cash, held, derisked
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
                if len(cs) < L_SMA or cs.iloc[-1] < cs.tail(L_SMA).mean():
                    continue
                if px.get(s, np.nan) < (1 - ATH_THR) * cs.max():
                    continue
                elig.append(s)
                if len(elig) >= BUF:
                    break
        if not elig:
            return
        bufset = set(elig[:BUF])
        for s in list(held):
            if s not in bufset:
                t = realize(held[s], d); cash += held.pop(s)[0] - t
        tot = E() + cash
        inv = tot                      # full deploy (gate ensures risk-on)
        names = ([s for s in held] +
                 [s for s in elig[:N] if s not in held])[:N]
        if not names:
            return
        w = inv / len(names)
        turn = (len(set(held).symmetric_difference(set(names)))
                / max(1, 2 * N))
        nh = {}
        for s in names:
            base = held[s] if s in held else [0, 0, d, px[s]]
            nh[s] = [w, (base[1] if s in held else w),
                     (base[2] if s in held else d),
                     (base[3] if s in held else px[s])]
        held = nh
        cash = tot - inv
        c = inv * RT * turn * 2
        cash -= c
        st['cost'] += c
        st['fills'] += 1
        derisked = False

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan)
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    stt[0] *= p1 / p0
                    stt[3] = max(stt[3], p1)
        if cash > 0:
            cash *= DAY_CASH
        h = close.loc[:d]

        # ---- WEEKLY regime action ----
        if d in wk_last:
            b = h[BENCH].dropna()
            roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if mode == "allcash":
                if roff and held:
                    for s in list(held):
                        t = realize(held[s], d); cash += held.pop(s)[0] - t
                    tgt = 0.0; derisked = True; st['derisk_ev'] += 1
                elif not roff:
                    tgt = 1.0
            elif mode == "keeptop":
                if roff and len(held) > K:
                    sc = rs2.rs_scores(close, d, 120)
                    if sc is not None:
                        rk = {s: sc.get(s, -9e9) for s in held}
                        keep = sorted(rk, key=rk.get, reverse=True)[:K]
                        for s in list(held):
                            if s not in keep:
                                t = realize(held[s], d)
                                cash += held.pop(s)[0] - t
                    tgt = 0.0 if not held else 1.0
                    derisked = True; st['derisk_ev'] += 1
                elif not roff:
                    tgt = 1.0
            # ---- the NEW axis: weekly re-entry ----
            if weekly_reentry and derisked and not roff:
                fill(d, px, h)
                st['wk_reentry'] += 1

        # ---- MONTHLY selection (baseline re-entry path, unchanged) ----
        if d in me:
            for s in list(held):                  # stock-level exits
                cs = h[s].dropna(); p1 = px.get(s, np.nan)
                if (len(cs) >= L_SMA and
                        cs.iloc[-1] < cs.tail(L_SMA).mean()) or \
                   (pd.notna(p1) and held[s][3] > 0 and
                        p1 <= (1 - TRAIL) * held[s][3]):
                    t = realize(held[s], d); cash += held.pop(s)[0] - t
            do_fill = True
            if mode == "allcash" and tgt == 0.0:
                do_fill = False
            if mode == "keeptop":
                b = h[BENCH].dropna()
                if len(b) >= 100 and b.iloc[-1] < b.tail(100).mean():
                    do_fill = False               # stay at K while risk-off
            if do_fill:
                fill(d, px, h)
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
                py=(py * 100).round(1), st=st, nav=n)


def main():
    t0 = time.time()
    print("Loading market data ...", flush=True)
    close, tv = rs2.load()
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)
    cfgs = [
        ("allcash  month-end (BASE)", "allcash", 0, False),
        ("allcash  WEEKLY-reentry",   "allcash", 0, True),
        ("keeptop8 month-end (BASE)", "keeptop", 8, False),
        ("keeptop8 WEEKLY-reentry",   "keeptop", 8, True),
    ]
    rows, pys = [], {}
    for lbl, m, K, wr in cfgs:
        ts = time.time()
        g = run(close, tv, m, K, wr)
        t = run(close, tv, m, K, wr, stcg=0.20)
        row = dict(config=lbl, cagr=round(g['cagr'], 1),
                   post_tax20=round(t['cagr'], 1), maxdd=round(g['dd'], 1),
                   sharpe=round(g['sh'], 2), calmar=round(g['cal'], 2),
                   reentry_ev=g['st']['wk_reentry'], fills=g['st']['fills'],
                   derisk_ev=g['st']['derisk_ev'],
                   cost_pct=round(g['st']['cost'] * 100, 1))
        rows.append(row); pys[lbl] = g['py']
        print(f"  {lbl:28} CAGR={row['cagr']:5.1f}% net20={row['post_tax20']:5.1f}%"
              f" DD={row['maxdd']:6.1f}% Sh={row['sharpe']:.2f}"
              f" Cal={row['calmar']:.2f} reentry={row['reentry_ev']:>2}"
              f" fills={row['fills']:>3} [{time.time() - ts:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(RES / "phase27_weekly_reentry.csv", index=False)
        pd.DataFrame(pys).reindex(range(2014, 2027)).to_csv(
            RES / "phase27_weekly_reentry_peryear.csv")
    print("\n=== WEEKLY RE-ENTRY vs MONTH-END ===\n" +
          pd.DataFrame(rows).to_string(index=False))
    print("\nSANITY: 'allcash month-end' should ~match Phase22 BASE "
          "(~34.2%/-22.2%); 'keeptop8 month-end' ~ (33.6%/-20.2%).")
    print(f"Total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
