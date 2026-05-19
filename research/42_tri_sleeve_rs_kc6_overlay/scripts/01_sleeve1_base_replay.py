"""Sleeve 1 — replay the LOCKED research/41 RS-momentum base, verbatim.

This is NOT a re-optimisation. It reproduces the research/41 winning config
`q0.5_dd__v__REG` on `mid_120d_N15` exactly (same loaders, same loop as
research/41/scripts/03_rs_quality_volume.py with q_pos=0.50, q_dd=None,
v_mult=None, regime=True), and additionally records a per-month TIMELINE
that the KC6-options (Sleeve 2) and short/hedge (Sleeve 3) overlays hook
into for the single-book combined backtest.

Phase-0 fidelity GATE (binding, see STATUS-MD §4): the replayed NAV must
reproduce research/41's reported 35.3% CAGR / -24.6% MaxDD to within
+/-0.5pp. If it does NOT, STOP and fix this driver before any overlay is
added — the whole comparison depends on Sleeve 1 being faithful.

Run (smoke-test on laptop snapshot DB; real sweep runs on VPS per the
canonical-host rule):
    python research/42_tri_sleeve_rs_kc6_overlay/scripts/01_sleeve1_base_replay.py
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)

# Reuse the vetted loaders/helpers from research/41's corrected Phase-02
# module (load, month_ends, rs_scores, pit_universe, metrics, constants).
_R41 = (Path(__file__).resolve().parents[2]
        / "41_midsmall400_mq_concentrated" / "scripts" / "02_rs_sweep.py")
_spec = importlib.util.spec_from_file_location("rs2", str(_R41))
rs2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rs2)

# LOCKED Sleeve-1 config (research/41 winner `q0.5_dd__v__REG`) — DO NOT TUNE.
BAND = "mid"
L = 120
N = 15
BUFFER = int(N * rs2.BUFFER_MULT)          # top-22 retention buffer
RT_COST = rs2.RT_COST                      # 0.4% round-trip
CASH_M = (1 + rs2.CASH_ANNUAL) ** (1 / 12)  # 6.5% p.a. bear/idle cash, monthly
BENCH = rs2.NIFTY_SYM                       # NIFTYBEES
Q_POS = 0.50                               # quality screen: >=50% positive 21d blocks
Q_DD = None                                # own-DD cap OFF (headline)
V_MULT = None                              # volume confirm OFF (poison)
REGIME = True                              # NIFTYBEES<SMA200 -> all cash

# research/41 reported headline for the Phase-0 gate.
TARGET_CAGR = 35.3       # %
TARGET_MDD = -24.6       # %
GATE_TOL = 0.5           # +/- pp


def _own_quality(close_hist, sym):
    """(pos_month_fraction, own_max_drawdown) over trailing ~252 sessions.
    Copied verbatim from research/41/scripts/03_rs_quality_volume.py so the
    quality screen is bit-identical."""
    s = close_hist[sym].dropna().tail(252)
    if len(s) < 120:
        return None, None
    blocks = [s.iloc[i:i + 21] for i in range(0, len(s) - 1, 21)]
    rets = [b.iloc[-1] / b.iloc[0] - 1
            for b in blocks if len(b) >= 2 and b.iloc[0] > 0]
    pos_frac = (np.mean([r > 0 for r in rets]) if rets else 0.0)
    roll_max = s.cummax()
    own_dd = ((s - roll_max) / roll_max).min()
    return pos_frac, own_dd


def run_sleeve1(close, tv, vol):
    """Replay the locked base. Returns (nav_df, timeline).

    timeline = list of per-month-end dicts, the spine the overlays attach to:
      date        : month-end Timestamp (rebalance bar)
      regime_on   : bool  (True = RISK-ON / invested; False = bear -> all cash)
      total       : book NAV at this rebalance (NAV units, start=1.0)
      cash        : cash portion after rebalance (full NAV when regime_on=False)
      equity      : equity portion after rebalance
      holdings    : {sym: rupee_value_in_NAV_units}  (empty when in cash)
      prices      : {sym: close_at_rebalance} for held names
      rs_rank     : pd.Series RS-120 score, full band, desc — for Sleeve-3C
                    (weakest-RS shorts) and Sleeve-2 universe context
    """
    me = rs2.month_ends(close.index)
    cash = 1.0; equity = 0.0
    held = {}; last_px = {}
    nav = []; timeline = []
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

        # --- regime gate: NIFTYBEES < SMA200 -> whole book to cash ---
        regime_on = True
        if REGIME and BENCH in h.columns:
            b = h[BENCH].dropna()
            if len(b) >= 200 and b.iloc[-1] < b.tail(200).mean():
                regime_on = False
                cash = total; equity = 0.0; held = {}; last_px = {}
                nav.append((dt, total))
                timeline.append(dict(date=dt, regime_on=False, total=total,
                                     cash=total, equity=0.0, holdings={},
                                     prices={}, rs_rank=pd.Series(dtype=float)))
                continue

        uni = rs2.pit_universe(tv, close, dt, BAND)
        sc = rs2.rs_scores(close, dt, L)
        if sc is None or not uni:
            nav.append((dt, total))
            timeline.append(dict(date=dt, regime_on=regime_on, total=total,
                                 cash=cash, equity=equity, holdings=dict(held),
                                 prices={s: px.get(s) for s in held},
                                 rs_rank=pd.Series(dtype=float)))
            continue
        cand = [s for s in sc.index if s in uni and s != BENCH]
        sc = sc[cand].sort_values(ascending=False)

        # --- quality gating in RS-rank order until BUFFER filled ---
        picked = []
        for s in sc.index:
            if Q_POS is not None or Q_DD is not None:
                pf, od = _own_quality(h, s)
                if pf is None:
                    continue
                if Q_POS is not None and pf < Q_POS:
                    continue
                if Q_DD is not None and (od is None or od < Q_DD):
                    continue
            if V_MULT is not None and s in vol.columns:
                vv = vol[s].loc[:dt].dropna()
                if len(vv) >= 80:
                    a20 = vv.tail(20).mean(); a60 = vv.iloc[-80:-20].mean()
                    if not (a60 > 0 and a20 >= V_MULT * a60):
                        continue
            picked.append(s)
            if len(picked) >= BUFFER:
                break
        if not picked:
            cash = total; equity = 0.0; held = {}; last_px = {}
            nav.append((dt, total))
            timeline.append(dict(date=dt, regime_on=regime_on, total=total,
                                 cash=total, equity=0.0, holdings={},
                                 prices={}, rs_rank=sc))
            continue

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
            nav.append((dt, total))
            timeline.append(dict(date=dt, regime_on=regime_on, total=total,
                                 cash=total, equity=0.0, holdings={},
                                 prices={}, rs_rank=sc))
            continue
        w = total / len(target)
        held = {s: w for s in target}
        last_px = {s: px[s] for s in target}
        cash = 0.0; equity = total
        nav.append((dt, total))
        timeline.append(dict(date=dt, regime_on=True, total=total,
                             cash=0.0, equity=total, holdings=dict(held),
                             prices={s: float(px[s]) for s in target},
                             rs_rank=sc))
    nav_df = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")
    return nav_df, timeline


def main():
    print("Sleeve-1 base replay — loading daily price+tv (research/41 loader) ...",
          flush=True)
    close, tv = rs2.load()
    import sqlite3
    con = sqlite3.connect(rs2.DB)
    vdf = pd.read_sql(
        "SELECT symbol,date,volume FROM market_data_unified "
        "WHERE timeframe='day' AND date>='2012-06-01' ORDER BY symbol,date",
        con, parse_dates=["date"]); con.close()
    vol = vdf.pivot_table(index="date", columns="symbol",
                          values="volume").sort_index()
    print(f"  {close.shape[1]} symbols; benchmark={BENCH}", flush=True)

    nav, timeline = run_sleeve1(close, tv, vol)
    m = rs2.metrics(nav)
    cagr, mdd = m["cagr"] * 100, m["mdd"] * 100
    print(f"\nSleeve-1 replayed (locked q0.5_dd__v__REG, mid_120d_N15):")
    print(f"  CAGR  = {cagr:6.2f}%   (research/41 target {TARGET_CAGR}%)")
    print(f"  MaxDD = {mdd:6.2f}%   (research/41 target {TARGET_MDD}%)")
    print(f"  Sharpe={m['sharpe']:.2f}  Calmar={m['calmar']:.2f}  "
          f"years={m['yrs']}  months={len(timeline)}", flush=True)

    cagr_ok = abs(cagr - TARGET_CAGR) <= GATE_TOL
    mdd_ok = abs(mdd - TARGET_MDD) <= GATE_TOL
    n_on = sum(1 for t in timeline if t["regime_on"])
    print(f"  regime RISK-ON months: {n_on}/{len(timeline)} "
          f"(bear/cash months: {len(timeline) - n_on})")

    nav.to_csv(RES / "sleeve1_base_nav.csv")
    print(f"  wrote {RES / 'sleeve1_base_nav.csv'}")

    if cagr_ok and mdd_ok:
        print("\n  PHASE-0 GATE: PASS — Sleeve-1 is faithful. "
              "Safe to layer Sleeve 2/3 on this timeline.")
    else:
        print("\n  PHASE-0 GATE: *** FAIL *** — replayed numbers diverge from "
              f"research/41 by > {GATE_TOL}pp "
              f"(CAGR ok={cagr_ok}, MDD ok={mdd_ok}). "
              "DO NOT add overlays until this is reconciled "
              "(likely a stale laptop snapshot DB vs the VPS canonical DB — "
              "re-run on VPS per the canonical-host rule before concluding).")


if __name__ == "__main__":
    main()
