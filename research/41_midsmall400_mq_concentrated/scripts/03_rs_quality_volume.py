"""Phase 03 — Drawdown-control overlays on the winning RS core.

RS-alone (run #2) already robustly beats the ~20% hurdle: the chosen core
`mid_120d_N15` does 38.3% CAGR / Sharpe 1.39 / MaxDD -29.8% / Calmar 1.29,
and survives losing its best-3 lifetime names. Phase 03 does NOT try to lift
CAGR. Goal: **shave the -30% drawdown toward the index's ~-24% while keeping
CAGR >= 35%**, by layering on the SAME core three filters:

  1. Trend-quality screen   — reject RS-ranked names whose OWN trailing-12m
     path is junk: positive-month fraction too low OR own max-DD too deep.
  2. Volume-breakout confirm — only hold names whose recent 20d avg volume
     >= k * prior 60d avg volume (momentum backed by participation).
  3. Regime filter           — if NIFTYBEES < its SMA200 at the rebalance,
     sit in 6.5% cash that month instead of buying the dip into a bear.

Core is fixed (band=mid, L=120d, N=15, monthly, buffer 1.5x, 0.4% RT cost,
NIFTYBEES benchmark). We sweep the overlay params, each vs the RS-alone
baseline, and report whether DD shrinks without CAGR falling below 35%.
Honest: STCG still NOT netted; quality here is price-path proxy, not
fundamentals; no OOS split (Phase 04).
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)

# Reuse the vetted loaders/helpers from the corrected Phase 02 module.
_spec = importlib.util.spec_from_file_location(
    "rs2", str(Path(__file__).resolve().parent / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rs2)

BAND = "mid"
L = 120                       # 120 trading-day RS lookback (winning core)
N = 15
BUFFER = int(N * rs2.BUFFER_MULT)
RT_COST = rs2.RT_COST
CASH_M = (1 + rs2.CASH_ANNUAL) ** (1 / 12)
BENCH = rs2.NIFTY_SYM         # NIFTYBEES

# Overlay sweep axes (None = filter OFF).
Q_POSFRAC = [None, 0.50, 0.58]      # min fraction of positive trailing 12m months
Q_MAXDD   = [None, -0.50, -0.40]    # reject if name's own 12m max-DD worse than this
V_MULT    = [None, 1.0, 1.2]        # 20d avg vol >= k * prior 60d avg vol
REGIME    = [False, True]           # cash out when NIFTYBEES < SMA200


def _own_quality(close_hist, sym):
    """(pos_month_fraction, own_max_drawdown) over trailing ~252 sessions."""
    s = close_hist[sym].dropna().tail(252)
    if len(s) < 120:
        return None, None
    m = s.resample("ME").last().pct_change().dropna() if False else None
    # cheap monthly proxy: 21-session blocks
    blocks = [s.iloc[i:i + 21] for i in range(0, len(s) - 1, 21)]
    rets = [b.iloc[-1] / b.iloc[0] - 1 for b in blocks if len(b) >= 2 and b.iloc[0] > 0]
    pos_frac = (np.mean([r > 0 for r in rets]) if rets else 0.0)
    roll_max = s.cummax()
    own_dd = ((s - roll_max) / roll_max).min()
    return pos_frac, own_dd


def backtest(close, tv, vol, q_pos, q_dd, v_mult, regime):
    me = rs2.month_ends(close.index)
    cash = 1.0; equity = 0.0
    held = {}; last_px = {}; nav = []
    for dt in me:
        h = close.loc[:dt]
        px = h.iloc[-1]
        if held:
            ne = 0.0
            for s, amt in held.items():
                p1 = px.get(s, np.nan)
                if pd.notna(p1) and s in last_px and last_px[s] > 0:
                    ne += amt * (1 + p1 / last_px[s] - 1)
                else:
                    ne += amt
            equity = ne
        total = cash * CASH_M + equity

        # --- regime filter: bear -> all cash this month ---
        if regime and BENCH in h.columns:
            b = h[BENCH].dropna()
            if len(b) >= 200 and b.iloc[-1] < b.tail(200).mean():
                cash = total; equity = 0.0; held = {}; last_px = {}
                nav.append((dt, total)); continue

        uni = rs2.pit_universe(tv, close, dt, BAND)
        sc = rs2.rs_scores(close, dt, L)
        if sc is None or not uni:
            nav.append((dt, total)); continue
        cand = [s for s in sc.index if s in uni and s != BENCH]
        sc = sc[cand].sort_values(ascending=False)

        # --- quality + volume gating, in RS-rank order, until N filled ---
        picked = []
        for s in sc.index:
            if q_pos is not None or q_dd is not None:
                pf, od = _own_quality(h, s)
                if pf is None:
                    continue
                if q_pos is not None and pf < q_pos:
                    continue
                if q_dd is not None and (od is None or od < q_dd):
                    continue
            if v_mult is not None and s in vol.columns:
                vv = vol[s].loc[:dt].dropna()
                if len(vv) >= 80:
                    a20 = vv.tail(20).mean(); a60 = vv.iloc[-80:-20].mean()
                    if not (a60 > 0 and a20 >= v_mult * a60):
                        continue
            picked.append(s)
            if len(picked) >= BUFFER:
                break
        if not picked:
            cash = total; equity = 0.0; held = {}; last_px = {}
            nav.append((dt, total)); continue

        top = picked[:N]; buf = set(picked[:BUFFER])
        keep = [s for s in held if s in buf]
        fill = [s for s in top if s not in keep]
        target = (keep + fill)[:N]
        prev, new = set(held), set(target)
        turn = len(prev.symmetric_difference(new)) / max(1, 2 * N)
        total *= (1 - RT_COST * turn * 2)
        target = [s for s in target if pd.notna(px.get(s, np.nan))]
        if not target:
            cash = total; equity = 0.0; held = {}; last_px = {}
            nav.append((dt, total)); continue
        w = total / len(target)
        held = {s: w for s in target}
        last_px = {s: px[s] for s in target}
        cash = 0.0; equity = total
        nav.append((dt, total))
    return pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")


def main():
    print("Loading price+tv (Phase 02 loader) ...", flush=True)
    close, tv = rs2.load()
    # volume pivot for the breakout confirm
    import sqlite3
    con = sqlite3.connect(rs2.DB)
    vdf = pd.read_sql(
        "SELECT symbol,date,volume FROM market_data_unified "
        "WHERE timeframe='day' AND date>='2012-06-01' ORDER BY symbol,date",
        con, parse_dates=["date"]); con.close()
    vol = vdf.pivot_table(index="date", columns="symbol", values="volume").sort_index()
    print(f"  {close.shape[1]} symbols; benchmark={BENCH}", flush=True)

    base_nav = backtest(close, tv, vol, None, None, None, False)
    bm = rs2.metrics(base_nav)
    print(f"\nBASELINE mid_120d_N15 (RS-alone, Phase02 parity): "
          f"CAGR={bm['cagr']*100:.1f}% Sh={bm['sharpe']:.2f} "
          f"DD={bm['mdd']*100:.1f}% Cal={bm['calmar']:.2f}", flush=True)
    base_cagr, base_dd = bm['cagr'] * 100, bm['mdd'] * 100

    rows = [dict(config="BASELINE_rs_alone", q_pos="-", q_dd="-", v_mult="-",
                 regime="-", cagr=round(base_cagr, 1),
                 sharpe=round(bm['sharpe'], 2), maxdd=round(base_dd, 1),
                 calmar=round(bm['calmar'], 2), dd_improve=0.0,
                 cagr_give=0.0)]
    n = 0
    for qp in Q_POSFRAC:
        for qd in Q_MAXDD:
            for vm in V_MULT:
                for rg in REGIME:
                    if qp is None and qd is None and vm is None and not rg:
                        continue  # == baseline
                    n += 1
                    nav = backtest(close, tv, vol, qp, qd, vm, rg)
                    if len(nav) < 12:
                        continue
                    m = rs2.metrics(nav)
                    cg, dd = m['cagr'] * 100, m['mdd'] * 100
                    lbl = (f"q{qp or '_'}_dd{qd or '_'}_v{vm or '_'}"
                           f"_{'REG' if rg else 'nor'}")
                    rows.append(dict(
                        config=lbl, q_pos=qp, q_dd=qd, v_mult=vm,
                        regime=rg, cagr=round(cg, 1),
                        sharpe=round(m['sharpe'], 2), maxdd=round(dd, 1),
                        calmar=round(m['calmar'], 2),
                        dd_improve=round(dd - base_dd, 1),    # +ve = shallower
                        cagr_give=round(cg - base_cagr, 1)))   # -ve = gave up
                    print(f"  [{n:2}] {lbl:30} CAGR={cg:5.1f}% "
                          f"DD={dd:6.1f}% Cal={m['calmar']:.2f} "
                          f"(dDD={dd-base_dd:+.1f} dCAGR={cg-base_cagr:+.1f})",
                          flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "phase03_overlay_sweep.csv", index=False)

    # Goal test: DD shallower than baseline AND CAGR still >= 35%.
    good = df[(df.config != "BASELINE_rs_alone") &
              (df.maxdd > base_dd) & (df.cagr >= 35.0)].copy()
    good = good.sort_values("maxdd", ascending=False)
    print("\n=== OVERLAYS that shrink drawdown AND keep CAGR>=35% ===")
    if len(good):
        print(good.to_string(index=False))
    else:
        print("  NONE — no overlay shaves drawdown while holding 35% CAGR. "
              "Honest read: the -30% hole is intrinsic to concentrated "
              "small/mid momentum; controlling it costs real CAGR.")
    # Best Calmar overall (even if it gives some CAGR)
    bestcal = df[df.config != "BASELINE_rs_alone"].sort_values(
        "calmar", ascending=False).head(8)
    print("\n=== TOP-8 by Calmar (drawdown-efficient, CAGR may be lower) ===")
    print(bestcal.to_string(index=False))
    print(f"\nBaseline: CAGR={base_cagr:.1f}% DD={base_dd:.1f}% "
          f"Cal={bm['calmar']:.2f}  | index hurdle ~20% CAGR, ~-24% DD")


if __name__ == "__main__":
    main()
