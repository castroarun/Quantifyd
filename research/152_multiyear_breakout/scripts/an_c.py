"""research/152 — Phase C read-out: per-family OFAT tables, plateau check, and the
pre-registered standalone bar (after-tax CAGR >= 20% AND Calmar >= 0.8 on W2, both windows
above NIFTYBEES)."""
import pandas as pd
import numpy as np

R = '/home/arun/quantifyd/research/152_multiyear_breakout/results/'
d = pd.read_csv(R + 'phaseC_g2.csv')
w2 = d[d.window == 'W2'].set_index(['family', 'arm'])
w1 = d[d.window == 'W1'].set_index(['family', 'arm'])
t = pd.DataFrame({
    'cagr2': w2.cagr_med, 'min2': w2.cagr_min, 'dd2': w2.dd_med, 'ddw': w2.dd_worst,
    'cal2': w2.calmar_med, 'tr_yr': w2.trades_yr, 'win': w2.win_med,
    'mean_tr': w2.mean_tr, 'awin': w2.avg_win, 'aloss': w2.avg_loss,
    'strk': w2.streak, 'hold': w2.avg_hold, 'axis': w2.axis})
t['cagr1'] = w1.cagr_med
t['dd1'] = w1.dd_med
t['cal1'] = w1.calmar_med
t['b2'] = w2.bench_cagr
t['b1'] = w1.bench_cagr
t['PASS'] = (t.cagr2 >= 20) & (t.cal2 >= 0.80) & (t.cagr2 > t.b2) & (t.cagr1 > t.b1)

print('cells run:', len(d), ' (family x arm x window)')
for fam in t.index.get_level_values(0).unique():
    s = t.loc[fam].sort_values('cal2', ascending=False)
    print(f'\n================ {fam} ================')
    print(s[['axis', 'cagr2', 'min2', 'dd2', 'ddw', 'cal2', 'cagr1', 'dd1', 'cal1',
             'tr_yr', 'win', 'mean_tr', 'awin', 'aloss', 'strk', 'hold', 'PASS']]
          .to_string())

print('\n\n================ CELLS CLEARING THE PRE-REGISTERED STANDALONE BAR ================')
p = t[t.PASS].sort_values('cal2', ascending=False)
print(p[['cagr2', 'min2', 'dd2', 'cal2', 'cagr1', 'dd1', 'tr_yr', 'win', 'mean_tr']]
      .to_string() if len(p) else 'NONE')

print('\n\n================ EXIT-FAMILY PLATEAU (trail SMA dose-response) ================')
ex = t[t.axis.isin(['exit', 'base'])]
print(ex[['cagr2', 'dd2', 'cal2', 'cagr1', 'tr_yr']].to_string())

print('\n\n================ SIZING AXIS ================')
sz = t[t.axis == 'sizing']
print(sz[['cagr2', 'min2', 'dd2', 'ddw', 'cal2', 'cagr1', 'tr_yr']].to_string())

print('\n\n================ COST LADDER ================')
print(t[t.axis.isin(['cost', 'base'])][['cagr2', 'dd2', 'cal2']].to_string())

print('\n\n================ GATE / FILL / BASE QUALITY ================')
print(t[t.axis.isin(['gate', 'fill', 'basequality', 'base'])]
      [['cagr2', 'dd2', 'cal2', 'cagr1', 'tr_yr', 'mean_tr']].to_string())
