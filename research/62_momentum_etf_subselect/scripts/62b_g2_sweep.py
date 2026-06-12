"""Research 62 — G2 robustness sweep on the gate+Donchian momentum sub-basket.

G1 winner = mom30 N10 buf22 + GATE100 + Donchian-20 (DD -19%, Sharpe 1.55, Cal 1.33).
G2 sweeps the stack to test: score (mom30 vs punchy rsblend), concentration (N),
buffer, Donchian window, gate SMA. Then on the top net-post-tax survivors: cost
sensitivity (0.2/0.4/0.6%) + super-winner guard (drop top-3 contributors) + per-year.

Daily-marked, gross + post-tax@20%. Incremental + resumable CSV. Reuses 62_mom30_subselect.
"""
from __future__ import annotations
import importlib.util, time, csv, logging
from pathlib import Path
import numpy as np
import pandas as pd
logging.disable(logging.WARNING)

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
ENG = HERE / "scripts" / "62_mom30_subselect.py"
_s = importlib.util.spec_from_file_location("m62", str(ENG))
m = importlib.util.module_from_spec(_s); _s.loader.exec_module(m)

SCORES = ["mom30", "rsblend"]
NS = [5, 8, 10, 15]
BUFFERS = [18, 22, 26]
DONCH = [0, 15, 20, 30]
GATES = [0, 100, 200]

FIELDS = ["config", "score", "N", "buffer", "gate", "donchian",
          "cagr", "net20", "maxdd", "sharpe", "sortino", "calmar", "net_calmar",
          "fills", "donch_exits", "gate_derisk", "avg_turn", "cost_pct"]


def metrics_row(label, sf, N, buf, gate, donch, g, t):
    fills = g['st'].get('fills', 0) or 1
    net_cal = (t['cagr'] / abs(g['dd'])) if g['dd'] < 0 else np.nan
    return dict(
        config=label, score=sf, N=N, buffer=buf, gate=gate, donchian=donch,
        cagr=round(g['cagr'], 1), net20=round(t['cagr'], 1),
        maxdd=round(g['dd'], 1), sharpe=round(g['sharpe'], 2),
        sortino=round(g['sortino'], 2) if pd.notna(g['sortino']) else "",
        calmar=round(g['calmar'], 2) if pd.notna(g['calmar']) else "",
        net_calmar=round(net_cal, 2) if pd.notna(net_cal) else "",
        fills=g['st']['fills'], donch_exits=g['st']['donchian_exits'],
        gate_derisk=g['st']['gate_derisk'],
        avg_turn=round(g['st']['turn_sum'] / fills, 3),
        cost_pct=round(g['st']['cost'] * 100, 1))


def main():
    t0 = time.time()
    print("Loading ...", flush=True)
    close, tv = m.rs2.load()
    print(f"  {close.shape[1]} syms, {close.index.min().date()}..{close.index.max().date()}", flush=True)

    out = RES / "g2_sweep.csv"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r['config'] for r in csv.DictReader(f)}
        print(f"  resume: {len(done)} done", flush=True)
    else:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    grid = [(sf, N, buf, g, dn) for sf in SCORES for N in NS for buf in BUFFERS
            for g in GATES for dn in DONCH if buf >= N]
    total = len(grid)
    print(f"  grid = {total} cells", flush=True)
    i = 0
    for sf, N, buf, gate, dn in grid:
        i += 1
        label = f"{sf}_N{N}_buf{buf}_g{gate}_d{dn}"
        if label in done:
            continue
        ts = time.time()
        gg = m.run(close, tv, sf, N, buf, gate or None, dn or None)
        tt = m.run(close, tv, sf, N, buf, gate or None, dn or None, stcg=0.20)
        row = metrics_row(label, sf, N, buf, gate, dn, gg, tt)
        with open(out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        if i % 10 == 0 or time.time() - ts > 8:
            print(f"  [{i}/{total}] {label:30} CAGR={row['cagr']:5.1f}% "
                  f"DD={row['maxdd']:6.1f}% netCal={row['net_calmar']} "
                  f"[{time.time()-ts:.0f}s] elapsed {time.time()-t0:.0f}s", flush=True)
    print(f"\nSWEEP DONE {total} cells in {time.time()-t0:.0f}s", flush=True)

    # ---- rank by net post-tax Calmar among investable-DD configs ----
    df = pd.read_csv(out)
    df['net_calmar'] = pd.to_numeric(df['net_calmar'], errors='coerce')
    inv = df[(df['maxdd'] > -25) & (df['gate'] > 0)].copy()
    top = inv.sort_values('net_calmar', ascending=False).head(6)
    print("\n=== TOP-6 by NET Calmar (investable DD > -25%, gated) ===", flush=True)
    print(top[['config', 'cagr', 'net20', 'maxdd', 'sharpe', 'calmar', 'net_calmar']]
          .to_string(index=False), flush=True)

    # ---- cost-sensitivity + super-winner guard + per-year on the top survivors ----
    cost_rows = []; guard_rows = []; pys = {}
    bm = m.stats_from_nav(m.bench_nav(close)); pys['NIFTYBEES'] = bm['py']
    for _, r in top.iterrows():
        sf, N, buf = r['score'], int(r['N']), int(r['buffer'])
        gate, dn = int(r['gate']) or None, int(r['donchian']) or None
        lbl = r['config']
        # cost sensitivity (net post-tax CAGR & net Calmar)
        cs = {}
        for rt in (0.002, 0.004, 0.006):
            tt = m.run(close, tv, sf, N, buf, gate, dn, stcg=0.20, rt=rt)
            cs[rt] = (round(tt['cagr'], 1),
                      round(tt['cagr'] / abs(tt['dd']), 2) if tt['dd'] < 0 else np.nan)
        cost_rows.append(dict(config=lbl, net_cagr_20bps=cs[0.002][0],
                              net_cagr_40bps=cs[0.004][0], net_cagr_60bps=cs[0.006][0],
                              netCal_20=cs[0.002][1], netCal_40=cs[0.004][1],
                              netCal_60=cs[0.006][1]))
        # super-winner guard: drop top-3 lifetime contributors, re-run
        base = m.run(close, tv, sf, N, buf, gate, dn)
        pys[lbl] = base['py']
        contrib = pd.Series(base['st']['contrib']).sort_values(ascending=False)
        top3 = list(contrib.head(3).index)
        ex = m.run(close, tv, sf, N, buf, gate, dn, exclude=set(top3))
        guard_rows.append(dict(config=lbl, base_cagr=round(base['cagr'], 1),
                               ex_top3_cagr=round(ex['cagr'], 1),
                               ex_top3_dd=round(ex['dd'], 1),
                               ex_top3_calmar=round(ex['calmar'], 2) if ex['dd'] < 0 else np.nan,
                               top3=",".join(top3)))
    pd.DataFrame(cost_rows).to_csv(RES / "g2_cost_sensitivity.csv", index=False)
    pd.DataFrame(guard_rows).to_csv(RES / "g2_superwinner_guard.csv", index=False)
    pd.DataFrame(pys).reindex(range(2014, 2027)).to_csv(RES / "g2_top_peryear.csv")
    top.to_csv(RES / "g2_top_survivors.csv", index=False)
    print("\n=== COST SENSITIVITY (net post-tax CAGR) ===", flush=True)
    print(pd.DataFrame(cost_rows).to_string(index=False), flush=True)
    print("\n=== SUPER-WINNER GUARD (drop top-3 contributors) ===", flush=True)
    print(pd.DataFrame(guard_rows).to_string(index=False), flush=True)
    print(f"\nALL DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
