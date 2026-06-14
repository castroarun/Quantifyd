"""Add the master 'all combinations' table + real-ETF confirmation to research/64 study."""
from pathlib import Path
P = Path("/home/arun/quantifyd/frontend/src/data/backtests.ts")
ts = P.read_text()
if "All combinations" in ts:
    print("already added"); raise SystemExit

anchor = """          rows: [
            ['Value (clean) — best, lowest-DD', '16.8%', '−9.5%', '1.77'],
            ['Quality (cleaned-indicative)', '16.3%', '−9.4%', '1.74'],
            ['Momentum (clean)', '19.1%', '−12.5%', '1.53'],
            ['Alpha (clean)', '19.9%', '−13.7%', '1.46'],
            ['Nifty (baseline)', '16.4%', '−11.3%', '1.46'],
          ],
          highlightRows: [0],
          heatmap: false,
        },"""
add = anchor + """
        {
          title: 'All combinations — solo / factor-only / +1 asset / +Gold+Nasdaq (clean, 2015–26)',
          caption: 'Do factor indices ALONE do better? No. Factor-only books (solo or combined) sit at Calmar ~0.45–0.55 — all the same equity beta (−23% to −33% DD). Gold+Nasdaq are what create the edge; you need BOTH.',
          columns: ['Book', 'Type', 'CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['Value + Gold + Nasdaq', 'factor + 2 assets', '16.8%', '−9.5%', '1.77'],
            ['Momentum + Gold + Nasdaq', 'factor + 2 assets', '19.1%', '−12.5%', '1.53'],
            ['Nifty + Gold + Nasdaq (original)', '3-asset', '17.2%', '−11.3%', '1.52'],
            ['Momentum + Gold', 'factor + 1 asset', '15.4%', '−11.3%', '1.36'],
            ['Value + Gold', 'factor + 1 asset', '12.7%', '−10.8%', '1.18'],
            ['Value+Momentum+Alpha', 'factor-only', '13.6%', '−24.9%', '0.55'],
            ['Alpha only', 'solo', '17.1%', '−32.3%', '0.53'],
            ['Momentum only', 'solo', '14.7%', '−29.0%', '0.51'],
            ['Value only', 'solo', '10.1%', '−22.7%', '0.45'],
            ['Nifty only', 'solo', '9.0%', '−29.3%', '0.31'],
          ],
          highlightRows: [0, 5],
          heatmap: false,
        },
        {
          title: 'Replace-Nifty on REAL factor ETFs (2022-08→2026-06, ~3.8y BULL — read ranks, not levels)',
          caption: 'Pulled the real factor ETFs (max history) to test Low-Vol/Quality properly. Short bull window inflates all Calmars (3.5–4.9); the RANKING and the real Low-Vol stats are the takeaways. Real Low-Vol = 13.6% vol & 0.93 corr to Nifty → NOT a diversifier (buries the corrupt-index 0.42–0.47 claim). Factor-only stays poor: Calmar 0.61 vs 4.49 baseline.',
          columns: ['Equity sleeve (+Gold+Nasdaq, equal)', 'CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['Value (MOVALUE)', '36%', '−7%', '4.92'],
            ['Low-Vol (LOWVOL1)', '29%', '−6%', '4.71'],
            ['Nifty (baseline)', '28%', '−6%', '4.49'],
            ['Quality (SBIETFQLTY)', '28%', '−7%', '3.95'],
            ['Momentum (MOMOMENTUM)', '29%', '−8%', '3.53'],
          ],
          highlightRows: [0],
          heatmap: false,
        },"""
assert anchor in ts, "replace-nifty anchor missing"
ts = ts.replace(anchor, add, 1)
P.write_text(ts)
print("added master + real-ETF tables")
