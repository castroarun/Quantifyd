"""
GTAA variant sweep (research/63). Incremental, resumable.

Run on VPS:  venv/bin/python research/63_gtaa_etf_rotation/scripts/run_gtaa_sweep.py
Output:      research/63_gtaa_etf_rotation/results/gtaa_sweep.csv  (append-per-config)
"""
import csv
import itertools
from pathlib import Path

from gtaa_engine import load_monthly_closes, run_gtaa, GTAAConfig

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
RES.mkdir(exist_ok=True)
OUT = RES / "gtaa_sweep.csv"

CASH = "LIQUIDBEES"
UNIVERSES = {
    "core3": ["NIFTYBEES", "GOLDBEES", "MON100"],            # the slide's 3 (2016->)
    "ext5":  ["NIFTYBEES", "GOLDBEES", "MON100", "JUNIORBEES", "BANKBEES"],  # +Next50,Bank (2016->)
}
TOP_N = [1, 2, 3]
CASH_LEG = [False, True]
ROC = {"roc12": (12,), "roc6": (6,), "blend": (3, 6, 12)}
MA = {"ma6": 6, "noma": 0, "ma10": 10}
COST_BPS = 20.0  # sweep at one realistic cost; cost-sensitivity done on winners

FIELDS = ["label", "universe", "top_n", "cash_leg", "roc", "ma", "cost_bps",
          "cagr", "sharpe", "sortino", "max_drawdown", "calmar", "vol",
          "turnover_annual", "pct_in_cash", "n_months", "start", "end", "final_mult"]


def main():
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {r["label"] for r in csv.DictReader(f)}
        print(f"Resuming: {len(done)} configs already done")
    else:
        with open(OUT, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    # preload panels (all symbols once)
    all_syms = sorted({s for u in UNIVERSES.values() for s in u} | {CASH})
    panel = load_monthly_closes(all_syms)
    print("panel:", panel.index.min().date(), "->", panel.index.max().date(),
          "cols:", list(panel.columns))

    combos = list(itertools.product(UNIVERSES, TOP_N, CASH_LEG, ROC, MA))
    print(f"{len(combos)} configs")
    for i, (uname, n, cl, rk, mk) in enumerate(combos, 1):
        label = f"{uname}_top{n}_{'cash' if cl else 'nocash'}_{rk}_{mk}"
        if label in done:
            continue
        cfg = GTAAConfig(
            risk_assets=UNIVERSES[uname], cash_asset=CASH, top_n=n,
            roc_months=ROC[rk], ma_months=MA[mk], cash_leg=cl,
            require_pos_roc=True, cost_bps=COST_BPS, label=label,
        )
        try:
            r = run_gtaa(panel, cfg)
        except Exception as e:
            print(f"[{i}/{len(combos)}] {label}: ERR {e}")
            continue
        row = dict(label=label, universe=uname, top_n=n, cash_leg=cl, roc=rk, ma=mk,
                   cost_bps=COST_BPS, cagr=round(r.cagr * 100, 2),
                   sharpe=round(r.sharpe, 3), sortino=round(r.sortino, 3),
                   max_drawdown=round(r.max_drawdown * 100, 2), calmar=round(r.calmar, 3),
                   vol=round(r.vol * 100, 2), turnover_annual=round(r.turnover_annual, 2),
                   pct_in_cash=round(r.pct_in_cash * 100, 1), n_months=r.n_months,
                   start=r.start, end=r.end, final_mult=round(r.final_mult, 3))
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"[{i}/{len(combos)}] {label}: CAGR={row['cagr']}% "
              f"DD={row['max_drawdown']}% Calmar={row['calmar']} "
              f"Sharpe={row['sharpe']} cash={row['pct_in_cash']}%", flush=True)

    print("\nDONE ->", OUT)


if __name__ == "__main__":
    main()
