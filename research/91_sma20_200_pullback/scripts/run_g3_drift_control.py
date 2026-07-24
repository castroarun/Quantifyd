"""
G3 drift control (research/91): the daily LONG cells showed positive (but t<2)
expectancy. Is that the pause-break SETUP, or just survivor drift = being long a
good large-cap in an uptrend? Control: for each name, compare the setup's gross
per-trade return vs the SAME up-regime's average H-day forward return from
(a) ALL regime bars and (b) RANDOM regime bars (matched count). If setup <= drift,
the pause/near mechanics add nothing -> NO EDGE (drift only).
"""
import os, sys, csv
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
from sma_pullback_engine import load_bars, add_indicators, backtest

rng = np.random.default_rng(20260724)
NAMES = ['NIFTY50', 'BANKNIFTY', 'HDFCBANK', 'ICICIBANK', 'RELIANCE', 'INFY',
         'TCS', 'SBIN', 'ITC', 'HINDUNILVR', 'KOTAKBANK', 'BHARTIARTL']

setup_r, drift_all, drift_rand = [], [], []
n_setup = 0
for nm in NAMES:
    df = load_bars(nm, 'day')
    if df.empty or len(df) < 250:
        continue
    df = add_indicators(df)
    c = df['close'].values; smf = df['sma_f'].values; sms = df['sma_s'].values
    n = len(df)
    # setup trades (best daily long cell: ext35, overnight, slope_lb=3)
    tr = backtest(df, direction='long', slope_lb=3, near_pct=0.005, atr_mult=3.5,
                  entry_win=3, hold='overnight', cost_bps=0.0, exit_mode='extend')
    holds = [t['hold_bars'] for t in tr if t['hold_bars'] > 0]
    if not tr:
        continue
    setup_r.extend([t['gross_pct'] / 100.0 for t in tr])
    n_setup += len(tr)
    H = max(1, int(round(np.mean(holds)))) if holds else 3
    # up-regime bars (same condition as the long setup regime)
    reg = [i for i in range(200, n - H)
           if not np.isnan(smf[i]) and not np.isnan(sms[i])
           and smf[i] > sms[i] and smf[i] > smf[i - 3] and c[i] > sms[i]]
    if not reg:
        continue
    fwd = np.array([c[i + H] / c[i] - 1.0 for i in reg])
    drift_all.extend(fwd.tolist())
    k = min(len(tr), len(reg))
    pick = rng.choice(len(reg), size=k, replace=False)
    drift_rand.extend(fwd[pick].tolist())


def stat(x, label):
    x = np.array(x)
    n = len(x)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n)) if n > 1 and x.std() > 0 else 0
    print(f'{label:32s} n={n:6d}  mean_gross={x.mean()*100:+.4f}%  t={t:+.2f}', flush=True)
    return x.mean()


print('=== G3 DRIFT CONTROL — daily, 12 deep names, LONG, ext35 overnight ===\n', flush=True)
s = stat(setup_r, 'SETUP (pause-break)')
a = stat(drift_all, 'DRIFT: all up-regime bars')
r = stat(drift_rand, 'DRIFT: random up-regime (matched)')
print(f'\nSetup edge OVER drift (all): {(s-a)*100:+.4f}%/trade', flush=True)
print(f'Setup edge OVER drift (random-matched): {(s-r)*100:+.4f}%/trade', flush=True)
print('\nIf setup ~<= drift, the pattern is just being long in an uptrend = NO EDGE.', flush=True)
