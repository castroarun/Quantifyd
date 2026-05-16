"""Phase 19/20 — SMOOTHEST with STAGGERED (chunked) regime exit + an
event timeline chart.

User: liquidating 100% on one weekly risk-off signal is drastic. Scale
out gradually instead: each weekly check while Nifty < its 100-SMA,
move another `chunk` of the book to cash (e.g. 25%/wk → 4 weeks to
flat). The moment Nifty closes back above its 100-SMA, STOP de-risking;
full redeploy happens at the next monthly rebalance (normal SMOOTHEST
selection). chunk = 1.0 == the current all-at-once SMOOTHEST (parity).

Core = SMOOTHEST: PIT mid-cap band, RS-120 vs NIFTYBEES, q0.5, ATH≤10%
entry, N=15, top-22 buffer, per-stock-100SMA exit + 12% trail at
month-ends, monthly RS rotation. Daily-marked engine, WEEKLY regime
(SMOOTHEST cadence). 0.4% round-trip, 6.5% cash, 2014-2026.

Also logs every event (ENTRY / EXIT_REGIME_CHUNK / EXIT_PERSTOCK_SMA /
EXIT_TRAIL12 / EXIT_RS_ROTATION) and renders a timeline chart for the
0.25 config (exposure% backdrop + distinct markers per event type).
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
_s = importlib.util.spec_from_file_location(
    "rs2", str(Path(__file__).resolve().parent / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)

BENCH = rs2.NIFTY_SYM
N, Q, ATH_THR, TRAIL = 15, 0.50, 0.10, 0.12
BUF = int(N * rs2.BUFFER_MULT)
DAY_CASH = (1 + rs2.CASH_ANNUAL) ** (1 / 252)
RT = rs2.RT_COST
START = pd.Timestamp("2014-01-01")


def own_pf(cs):
    if len(cs) < 120:
        return None
    bl = [cs.iloc[i:i + 21] for i in range(0, len(cs) - 1, 21)]
    r = [x.iloc[-1] / x.iloc[0] - 1 for x in bl
         if len(x) >= 2 and x.iloc[0] > 0]
    return float(np.mean([z > 0 for z in r])) if r else 0.0


def run(close, tv, chunk, stcg=0.0, log=False):
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    wk = pd.Series(idx, index=idx)
    wk_last = set(wk.groupby([idx.isocalendar().year,
                              idx.isocalendar().week]).last().values)
    cash = 1.0
    held = {}                       # sym -> [val, cost, buydt, peak]
    tgt = 1.0                       # target invested fraction
    nav, expo, tax, prev = [], [], 0.0, None
    ev = []

    def E():
        return sum(v[0] for v in held.values())

    def realize(val, cost, bd, d):
        nonlocal tax
        g = val - cost
        if stcg and g > 0 and (d - bd).days < 365:
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

        # ---- WEEKLY regime check: staggered scale-out ----
        if d in wk_last:
            b = h[BENCH].dropna()
            roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if roff:
                tgt = max(0.0, round(tgt - chunk, 6))
                eq, c = E(), cash
                tot = eq + c
                want_eq = tgt * tot
                if eq > want_eq + 1e-9 and eq > 0:
                    k = want_eq / eq                  # keep fraction
                    proceeds = 0.0
                    for s, st in list(held.items()):
                        sell_val = st[0] * (1 - k)
                        t = realize(sell_val, st[1] * (1 - k),
                                    st[2], d)
                        st[0] *= k; st[1] *= k
                        proceeds += sell_val - t
                        if st[0] < 1e-9:
                            held.pop(s)
                    proceeds -= (eq - want_eq) * RT      # exit cost
                    cash += proceeds
                    if log:
                        ev.append((d, "EXIT_REGIME_CHUNK",
                                   f"to_{tgt:.0%}"))
            else:
                tgt = 1.0                                # re-enable

        # ---- MONTHLY selection / stock-level exits ----
        if d in me:
            # per-stock-SMA + 12% trail exits
            for s in list(held):
                cs = h[s].dropna()
                p1 = px.get(s, np.nan)
                if len(cs) >= 100 and cs.iloc[-1] < cs.tail(100).mean():
                    t = realize(held[s][0], held[s][1], held[s][2], d)
                    cash += held.pop(s)[0] - t
                    if log:
                        ev.append((d, "EXIT_PERSTOCK_SMA", s))
                elif (pd.notna(p1) and held[s][3] > 0 and
                      p1 <= (1 - TRAIL) * held[s][3]):
                    t = realize(held[s][0], held[s][1], held[s][2], d)
                    cash += held.pop(s)[0] - t
                    if log:
                        ev.append((d, "EXIT_TRAIL12", s))
            if tgt > 0:
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
                if elig:
                    bufset = set(elig[:BUF])
                    for s in list(held):
                        if s not in bufset:
                            t = realize(held[s][0], held[s][1],
                                        held[s][2], d)
                            cash += held.pop(s)[0] - t
                            if log:
                                ev.append((d, "EXIT_RS_ROTATION", s))
                    tot = E() + cash
                    inv = tgt * tot
                    tgt_names = ([s for s in held] +
                                 [s for s in elig[:N] if s not in held])
                    tgt_names = tgt_names[:N]
                    if tgt_names:
                        w = inv / len(tgt_names)
                        prevset = set(held)
                        turn = (len(prevset.symmetric_difference(
                            set(tgt_names))) / max(1, 2 * N))
                        nh = {}
                        for s in tgt_names:
                            if s in held:
                                nh[s] = held[s]
                                nh[s][0] = w           # rebal to target
                            else:
                                nh[s] = [w, w, d, px[s]]
                                if log:
                                    ev.append((d, "ENTRY", s))
                        held = nh
                        cash = tot - inv
                        cash -= inv * RT * turn * 2     # turnover cost
        nav.append((d, E() + cash))
        tt = E() + cash
        expo.append((d, (E() / tt) if tt > 0 else 0.0))
        prev = d

    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    ex = pd.DataFrame(expo, columns=["date", "expo"]).set_index("date")["expo"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    pyr = n.groupby(n.index.year).last().pct_change()
    pyr.iloc[0] = n.groupby(n.index.year).last().iloc[0] / n.iloc[0] - 1
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan,
                pyr=(pyr * 100).round(1), nav=n, expo=ex,
                events=pd.DataFrame(ev, columns=["date", "type", "ref"]))


def timeline(res, out):
    ex = res["expo"]; ev = res["events"]
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.fill_between(ex.index, ex.values * 100, 0, color="#bcd",
                    alpha=0.55, label="Equity exposure %")
    ax.set_ylabel("Exposure %"); ax.set_ylim(-8, 108)
    styles = {
        "ENTRY": ("^", "#159e3b", "Entry (new holding)"),
        "EXIT_REGIME_CHUNK": ("v", "#d62728",
                              "Regime de-risk chunk"),
        "EXIT_PERSTOCK_SMA": ("x", "#ff7f0e",
                              "Exit: stock < own 100-SMA"),
        "EXIT_TRAIL12": ("D", "#8e44ad", "Exit: 12% trailing stop"),
        "EXIT_RS_ROTATION": (".", "#777777", "Exit: RS rotation"),
    }
    lanes = {"ENTRY": 101, "EXIT_REGIME_CHUNK": 95,
             "EXIT_PERSTOCK_SMA": 89, "EXIT_TRAIL12": 84,
             "EXIT_RS_ROTATION": 79}
    for typ, (mk, col, lab) in styles.items():
        sub = ev[ev["type"] == typ]
        if len(sub):
            ax.scatter(sub["date"], [lanes[typ]] * len(sub),
                       marker=mk, c=col, s=34, label=f"{lab} ({len(sub)})",
                       zorder=3, linewidths=1.1)
    ax.set_title("SMOOTHEST (25%/wk staggered exit) — event timeline & "
                 "exposure, 2014–2026", fontsize=11)
    ax.legend(loc="lower center", ncol=3, fontsize=8,
              framealpha=0.92, bbox_to_anchor=(0.5, -0.32))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved timeline -> {out.name}")


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    rows, pys = [], {}
    for ch in (1.0, 0.50, 0.33, 0.25):
        g = run(close, tv, ch)
        t = run(close, tv, ch, stcg=0.20)
        lbl = ("ALL-AT-ONCE (base)" if ch == 1.0
               else f"stagger {int(ch*100)}%/wk")
        rows.append(dict(config=lbl, chunk=ch, cagr=round(g['cagr'], 1),
                         post_tax20=round(t['cagr'], 1),
                         maxdd_daily=round(g['dd'], 1),
                         sharpe=round(g['sh'], 2),
                         calmar=round(g['cal'], 2)))
        pys[lbl] = g['pyr']
        print(f"  {lbl:20} CAGR={g['cagr']:5.1f}% net20={t['cagr']:5.1f}%"
              f" dailyDD={g['dd']:6.1f}% Sh={g['sh']:.2f} "
              f"Cal={g['cal']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "phase19_staggered.csv", index=False)
    pd.DataFrame(pys).reindex(range(2014, 2027)).to_csv(
        RES / "phase19_staggered_peryear.csv")
    print("\n=== STAGGERED-EXIT SWEEP ===\n" +
          pd.DataFrame(rows).to_string(index=False))
    print("\nRendering event timeline for the 25%/wk config ...",
          flush=True)
    r25 = run(close, tv, 0.25, log=True)
    r25["events"].to_csv(RES / "phase19_events_25pct.csv", index=False)
    timeline(r25, RES / "smoothest_staggered_timeline.png")
    ec = r25["events"]["type"].value_counts().to_dict()
    print("event counts (25%/wk):", ec)
    print("Ref: SMOOTHEST all-at-once weekly (Ph15 daily) Cal 1.65 / "
          "DD -22.2. VERDICT: staggering helps if Calmar/DD improve "
          "without material CAGR loss.")


if __name__ == "__main__":
    main()
