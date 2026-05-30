"""Phase 3 analysis from results/trades_long.csv (causal-beta long trade log):
  1. Causal-beta confirmation + beta-cutoff sweep (NR7, PCT25; 2R, 3R)
  2. Per-year stability of the winner (NR7, beta>=1.2, 2R)
  3. Portfolio sim -> equity curve: CAGR, MaxDD, monthly Sharpe
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
TRADES = RESULTS / "trades_long.csv"


def load():
    with open(TRADES) as f:
        return list(csv.DictReader(f))


def stats(results):
    n = len(results)
    if n == 0:
        return None
    wins = [r for r in results if r > 0]
    losses = [r for r in results if r <= 0]
    win_r = sum(wins)
    loss_r = sum(losses)
    pf = win_r / abs(loss_r) if loss_r < 0 else float("inf")
    return dict(n=n, win_pct=100 * len(wins) / n, exp=sum(results) / n,
                pf=pf, totR=sum(results))


def line(label, s):
    if s is None:
        print(f"   {label:34s} (no trades)")
        return
    pf = f"{s['pf']:.3f}" if s['pf'] != float('inf') else "inf"
    print(f"   {label:34s} n={s['n']:6d}  wr={s['win_pct']:5.1f}%  "
          f"exp={s['exp']:+.3f}R  PF={pf:>6s}  totR={s['totR']:+.0f}")


def beta_sweep(rows):
    print("\n=== 1. Causal-beta confirmation + beta-cutoff sweep ===")
    for comp in ("nr7", "pct25"):
        sub = [r for r in rows if r[comp] == "1"]
        print(f"\n -- compression={comp.upper()} (causal trailing-252d beta) --")
        for cut in (0.0, 1.0, 1.2, 1.4):
            for tgt in ("2R", "3R"):
                res = [float(r[f"result_{tgt}"]) for r in sub if float(r["beta"]) >= cut]
                line(f"beta>={cut:<3} | {tgt}", stats(res))


def per_year(rows):
    print("\n=== 2. Per-year stability — NR7, beta>=1.2, 2R (long-only) ===")
    sub = [r for r in rows if r["nr7"] == "1" and float(r["beta"]) >= 1.2]
    by_year = defaultdict(list)
    for r in sub:
        by_year[r["year"]].append(float(r["result_2R"]))
    for y in sorted(by_year):
        line(f"{y}", stats(by_year[y]))
    print("   " + "-" * 60)
    line("ALL YEARS", stats([float(r["result_2R"]) for r in sub]))
    # also 3R per-year summary line
    line("ALL YEARS (3R)", stats([float(r["result_3R"]) for r in sub]))


def _d(s):
    y, m, dd = s.split("-")
    return date(int(y), int(m), int(dd))


def portfolio(rows, tgt="2R", cut=1.2, comp="nr7",
              C0=10_000_000, risk_pct=0.01, maxc=10):
    sub = [r for r in rows if r[comp] == "1" and float(r["beta"]) >= cut]
    trades = sorted(
        [(r["entry_date"], r[f"exit_{tgt}"], float(r[f"result_{tgt}"])) for r in sub],
        key=lambda t: t[0])
    if not trades:
        print("no trades for portfolio")
        return None
    capital = C0
    openpos = []           # list of (exit_date, pnl)
    equity = []            # (date, capital) at each realization
    taken = skipped = 0

    def realize_until(d):
        nonlocal capital, openpos
        due = sorted([p for p in openpos if p[0] <= d], key=lambda p: p[0])
        for ex, pnl in due:
            capital += pnl
            equity.append((ex, capital))
        openpos = [p for p in openpos if p[0] > d]

    equity.append((trades[0][0], capital))
    for e, x, r in trades:
        realize_until(e)
        if len(openpos) >= maxc:
            skipped += 1
            continue
        pnl = capital * risk_pct * r
        openpos.append((x, pnl))
        taken += 1
    for ex, pnl in sorted(openpos, key=lambda p: p[0]):
        capital += pnl
        equity.append((ex, capital))

    # metrics
    eq = sorted(equity, key=lambda p: p[0])
    final = eq[-1][1]
    yrs = (_d(eq[-1][0]) - _d(eq[0][0])).days / 365.25
    cagr = (final / C0) ** (1 / yrs) - 1 if yrs > 0 and final > 0 else float("nan")
    peak = -1e18
    maxdd = 0.0
    for _, v in eq:
        peak = max(peak, v)
        maxdd = min(maxdd, v / peak - 1)
    # monthly Sharpe (approx): realized pnl bucketed by exit month
    mret = defaultdict(float)
    run = C0
    month_start = defaultdict(lambda: None)
    # rebuild monthly equity
    by_month_end = {}
    for d, v in eq:
        by_month_end[d[:7]] = v
    months = sorted(by_month_end)
    prev = C0
    rets = []
    for m in months:
        v = by_month_end[m]
        rets.append(v / prev - 1)
        prev = v
    import statistics as st
    sharpe = (st.mean(rets) / st.pstdev(rets) * (12 ** 0.5)) if len(rets) > 1 and st.pstdev(rets) > 0 else float("nan")

    print(f"\n=== 3. Portfolio sim — {comp.upper()}, beta>={cut}, {tgt}, "
          f"risk={risk_pct*100:.0f}%/trade, max {maxc} concurrent ===")
    print(f"   capital {C0:,.0f} -> {final:,.0f}  over {yrs:.1f}y")
    print(f"   CAGR={cagr*100:.1f}%   MaxDD={maxdd*100:.1f}%   "
          f"monthlySharpe~{sharpe:.2f}   trades taken={taken} skipped={skipped}")
    return dict(cagr=cagr, maxdd=maxdd, final=final, taken=taken, skipped=skipped)


def main():
    rows = load()
    print(f"loaded {len(rows)} long trades from {TRADES.name}")
    beta_sweep(rows)
    per_year(rows)
    for tgt in ("2R", "3R"):
        for maxc in (5, 10, 20):
            portfolio(rows, tgt=tgt, maxc=maxc)


if __name__ == "__main__":
    main()
