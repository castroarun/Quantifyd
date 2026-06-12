"""Phase 31 — debt-yield tax sensitivity. The 6.5% cash yield is currently added
GROSS; debt-fund returns are slab-taxed (post-Apr-2023). all-cash parks 100% in
debt during risk-off while keep-top8 keeps 53% in equity, so taxing the cash yield
should hit all-cash harder and NARROW the finalist gap. Quantify it.

Method: monkeypatch the engines' daily cash-growth factor to an AFTER-TAX debt
rate (6.5%*(1-slab)); equity STCG stays 20%. Compare net CAGR across slabs 0/20/30%.
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
_w = importlib.util.spec_from_file_location(
    "p27", str(SCR / "27_weekly_reentry.py"))
p27 = importlib.util.module_from_spec(_w); _w.loader.exec_module(p27)

CASH = rs2.CASH_ANNUAL


def set_cash_tax(rate):
    f = (1 + CASH * (1 - rate)) ** (1 / 252)
    p22.DAY_CASH = f
    p27.DAY_CASH = f


def run_variant(close, tv, which, stcg):
    if which == "all-cash+weekly":
        return p27.run(close, tv, "allcash", 0, True, stcg=stcg)
    if which == "keep-top8":
        return p22.run(close, tv, "keeptop", 0.0, 8, 100, stcg=stcg)
    if which == "all-cash base":
        return p22.run(close, tv, "allcash", 0.0, 0, 100, stcg=stcg)


def main():
    t0 = time.time()
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print(f"  loaded {time.time()-t0:.0f}s", flush=True)
    variants = ["all-cash+weekly", "keep-top8", "all-cash base"]

    rows = []
    for v in variants:
        set_cash_tax(0.0)                 # gross factor for the gross run
        g = run_variant(close, tv, v, 0.0)
        set_cash_tax(0.0)
        n0 = run_variant(close, tv, v, 0.20)   # equity20, debt untaxed (current)
        set_cash_tax(0.20)
        n20 = run_variant(close, tv, v, 0.20)  # equity20 + debt@20%
        set_cash_tax(0.30)
        n30 = run_variant(close, tv, v, 0.20)  # equity20 + debt@30%
        row = dict(variant=v,
                   gross=round(g['cagr'], 1),
                   net_debt0=round(n0['cagr'], 1),
                   net_debt20=round(n20['cagr'], 1),
                   net_debt30=round(n30['cagr'], 1),
                   maxdd=round(g['dd'], 1), calmar=round(g['cal'], 2))
        rows.append(row)
        print(f"  {v:18} gross={row['gross']:.1f}  net(debt0)={row['net_debt0']:.1f}"
              f"  net(debt20)={row['net_debt20']:.1f}  net(debt30)={row['net_debt30']:.1f}",
              flush=True)
        pd.DataFrame(rows).to_csv(RES / "phase31_debt_tax.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== DEBT-YIELD TAX SENSITIVITY (net CAGR %) ===\n" +
          df.to_string(index=False))
    # finalist gap at each slab
    acw = df[df.variant == "all-cash+weekly"].iloc[0]
    kt8 = df[df.variant == "keep-top8"].iloc[0]
    print("\nFinalist gap (all-cash+weekly minus keep-top8), net CAGR:")
    for c, lbl in [("net_debt0", "debt 0%"), ("net_debt20", "debt 20%"),
                   ("net_debt30", "debt 30%")]:
        print(f"  {lbl:9}: {acw[c]-kt8[c]:+.1f} pp", flush=True)
    print(f"\nTotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
