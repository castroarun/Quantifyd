"""
Test: on a mid-month Donchian exit, IMMEDIATELY redeploy the freed cash into the next-best
momentum name (instead of waiting to month-end). Winner config rsblend N8 buf22 donch15 gate100.
Gross AND net-post-tax(STCG20%). Compare base (wait for month-end) vs redeploy_on_exit=True.
"""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location('mom62', os.path.join(HERE, '62_mom30_subselect.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

WIN = dict(score_fn='rsblend', N=8, buffer=22, gate=100, donchian=15)
print('Loading price+tv ...', flush=True)
close, tv = m.rs2.load()
bn = m.bench_nav(close); bm = m.stats_from_nav(bn)
print(f"[ref] NIFTYBEES CAGR={bm['cagr']:.1f}% DD={bm['dd']:.1f}% Calmar={bm['calmar']:.2f}\n", flush=True)

print(f"{'Variant':<34}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>7}{'Sharpe':>7}{'fills':>7}{'donchX':>7}{'redep':>7}{'turn':>7}")
def row(lbl, g):
    st = g['st']
    print(f"{lbl:<34}{g['cagr']:>6.1f}%{g['dd']:>7.1f}%{g['calmar']:>7.2f}{g['sharpe']:>7.2f}"
          f"{st['fills']:>7}{st['donchian_exits']:>7}{st.get('redeploys',0):>7}{st['turn_sum']/max(1,st['fills']):>7.2f}", flush=True)
    return g

for lbl, redep, tax in [('BASE (cash till month-end) gross', False, 0.0),
                        ('BASE net-tax(20%)',                False, 0.20),
                        ('REDEPLOY-on-exit gross',           True,  0.0),
                        ('REDEPLOY-on-exit net-tax(20%)',    True,  0.20)]:
    g = m.run(close, tv, WIN['score_fn'], WIN['N'], WIN['buffer'], gate=WIN['gate'],
              donchian=WIN['donchian'], stcg=tax, redeploy_on_exit=redep)
    row(lbl, g)

# per-year gross: base vs redeploy
print('\nPer-year gross CAGR%: BASE | REDEPLOY')
gb = m.run(close, tv, WIN['score_fn'], WIN['N'], WIN['buffer'], gate=WIN['gate'], donchian=WIN['donchian'], redeploy_on_exit=False)
gr = m.run(close, tv, WIN['score_fn'], WIN['N'], WIN['buffer'], gate=WIN['gate'], donchian=WIN['donchian'], redeploy_on_exit=True)
import pandas as pd
for y in sorted(set(gb['py'].index) | set(gr['py'].index)):
    print(f"  {y}  {gb['py'].get(y, float('nan')):6.1f} | {gr['py'].get(y, float('nan')):6.1f}")
print('DONE')
