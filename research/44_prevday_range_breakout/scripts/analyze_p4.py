"""Phase 4: market-regime filter + portfolio risk controls, from trades_long.csv
and market_regime.csv.

A. Regime effect on per-trade edge (NR7, beta cutoffs, 2R): OFF vs reg20/reg50/reg200.
B. Per-year, regime ON vs OFF.
C. Portfolio sim with regime gate + portfolio-HEAT cap (total concurrent risk).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
TRADES = RESULTS / "trades_long.csv"
REGIME = RESULTS / "market_regime.csv"


def load_trades():
    with open(TRADES) as f:
        return list(csv.DictReader(f))


def load_regime():
    reg = {}
    with open(REGIME) as f:
        for r in csv.DictReader(f):
            reg[r["date"]] = r
    return reg


def stats(results):
    n = len(results)
    if n == 0:
        return None
    wins = [r for r in results if r > 0]
    loss_r = sum(r for r in results if r <= 0)
    pf = sum(wins) / abs(loss_r) if loss_r < 0 else float("inf")
    return dict(n=n, wr=100 * len(wins) / n, exp=sum(results) / n, pf=pf,
                totR=sum(results))


def line(label, s):
    if s is None:
        print(f"   {label:30s} (no trades)")
        return
    pf = f"{s['pf']:.3f}" if s['pf'] != float("inf") else "inf"
    print(f"   {label:30s} n={s['n']:6d}  wr={s['wr']:5.1f}%  exp={s['exp']:+.3f}R  "
          f"PF={pf:>6s}  totR={s['totR']:+.0f}")


def reg_ok(reg, d, key):
    r = reg.get(d)
    return r is not None and r.get(key, "0") == "1"


def section_A(rows, reg):
    print("\n=== A. Regime effect on edge — NR7 long, 2R ===")
    for cut in (1.0, 1.2):
        sub = [r for r in rows if r["nr7"] == "1" and float(r["beta"]) >= cut]
        print(f"\n -- beta>={cut} --")
        line("regime OFF (all)", stats([float(r["result_2R"]) for r in sub]))
        for key in ("reg20", "reg50", "reg200"):
            res = [float(r["result_2R"]) for r in sub if reg_ok(reg, r["entry_date"], key)]
            line(f"regime {key} ON", stats(res))


def section_B(rows, reg, cut=1.0, key="reg50"):
    print(f"\n=== B. Per-year — NR7 beta>={cut}, 2R: OFF vs {key} ON ===")
    sub = [r for r in rows if r["nr7"] == "1" and float(r["beta"]) >= cut]
    yrs = sorted({r["year"] for r in sub})
    print(f"   {'year':6s}{'OFF n/wr/PF':28s}{'ON n/wr/PF':28s}")
    for y in yrs:
        off = stats([float(r["result_2R"]) for r in sub if r["year"] == y])
        on = stats([float(r["result_2R"]) for r in sub
                    if r["year"] == y and reg_ok(reg, r["entry_date"], key)])
        def fmt(s):
            if not s:
                return "      -"
            pf = f"{s['pf']:.2f}" if s['pf'] != float("inf") else "inf"
            return f"n={s['n']:4d} wr={s['wr']:4.0f}% PF={pf:>5s}"
        print(f"   {y:6s}{fmt(off):28s}{fmt(on):28s}")


def _d(s):
    y, m, dd = s.split("-")
    return date(int(y), int(m), int(dd))


def portfolio(rows, reg, regkey, cut=1.0, tgt="2R", base_risk=0.01,
              max_heat=0.06, maxc=20, C0=10_000_000, year_min="2024"):
    sub = [r for r in rows if r["nr7"] == "1" and float(r["beta"]) >= cut
           and r["entry_date"] >= year_min
           and (regkey is None or reg_ok(reg, r["entry_date"], regkey))]
    trades = sorted([(r["entry_date"], r[f"exit_{tgt}"], float(r[f"result_{tgt}"]))
                     for r in sub], key=lambda t: t[0])
    if not trades:
        print(f"   (no trades: regime={regkey})")
        return
    capital = C0
    openpos = []           # (exit_date, risk_amt, result_R)
    equity = [(trades[0][0], capital)]
    taken = skipped = 0

    def realize(d):
        nonlocal capital, openpos
        due = sorted([p for p in openpos if p[0] <= d], key=lambda p: p[0])
        for ex, risk_amt, r in due:
            capital += risk_amt * r
            equity.append((ex, capital))
        openpos = [p for p in openpos if p[0] > d]

    for e, x, r in trades:
        realize(e)
        open_heat = sum(p[1] for p in openpos) / capital
        if len(openpos) >= maxc or open_heat + base_risk > max_heat:
            skipped += 1
            continue
        openpos.append((x, capital * base_risk, r))
        taken += 1
    for ex, risk_amt, r in sorted(openpos, key=lambda p: p[0]):
        capital += risk_amt * r
        equity.append((ex, capital))

    eq = sorted(equity, key=lambda p: p[0])
    final = eq[-1][1]
    yrs = (_d(eq[-1][0]) - _d(eq[0][0])).days / 365.25
    cagr = (final / C0) ** (1 / yrs) - 1 if yrs > 0 and final > 0 else float("nan")
    peak, maxdd = -1e18, 0.0
    for _, v in eq:
        peak = max(peak, v)
        maxdd = min(maxdd, v / peak - 1)
    bm = {}
    for d, v in eq:
        bm[d[:7]] = v
    months = sorted(bm)
    prev, rets = C0, []
    for m in months:
        rets.append(bm[m] / prev - 1)
        prev = bm[m]
    sharpe = (st.mean(rets) / st.pstdev(rets) * (12 ** 0.5)) if len(rets) > 1 and st.pstdev(rets) > 0 else float("nan")
    tag = f"regime={regkey or 'OFF'}"
    print(f"   {tag:14s} heat={max_heat*100:.0f}% maxc={maxc:2d}: "
          f"{final/1e7:.2f}cr CAGR={cagr*100:+5.1f}% MaxDD={maxdd*100:5.1f}% "
          f"Sh~{sharpe:.2f} took={taken} skip={skipped}")


def main():
    rows = load_trades()
    reg = load_regime()
    print(f"loaded {len(rows)} trades, {len(reg)} regime days")
    section_A(rows, reg)
    section_B(rows, reg, cut=1.0, key="reg50")
    print("\n=== C. Portfolio sim (2024+, NR7 beta>=1.0, 2R, 1% risk) ===")
    print("   -- baseline vs regime gates, heat cap 6%, max 20 concurrent --")
    for rk in (None, "reg20", "reg50", "reg200"):
        portfolio(rows, reg, rk, cut=1.0, max_heat=0.06, maxc=20)
    print("   -- best regime (reg50): heat-cap sweep --")
    for heat in (0.03, 0.06, 0.10, 0.20):
        portfolio(rows, reg, "reg50", cut=1.0, max_heat=heat, maxc=30)
    print("   -- reg50, beta>=1.2, heat 6% --")
    portfolio(rows, reg, "reg50", cut=1.2, max_heat=0.06, maxc=20)


if __name__ == "__main__":
    main()
