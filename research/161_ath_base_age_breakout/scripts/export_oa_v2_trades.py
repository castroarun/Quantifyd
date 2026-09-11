"""
research/161 — export OA V2.0 backtested trade details for the app.

Writes:
  results/oa_v2_trades_median_seed.csv  one row per trade on the MEDIAN-CAGR seed
  results/oa_v2_seed_summary.csv        30 rows: seed, CAGR, DD, trades, win rate
"""
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt_core as B

RES = Path(__file__).resolve().parents[1] / 'results'
DB = Path(__file__).resolve().parents[3] / 'backtest_data' / 'market_data.db'
SEEDS = list(range(1, 31))
REARM = 60
CFG = dict(X=60, dep=20.0, K=0.0, sau=0, ex='ST_14_4', hard=False, liq=2.0)


def rearm(df):
    keep = []
    for _, g in df.sort_values(['symbol', 'hist_bars']).groupby('symbol', sort=False):
        last = -10 ** 9
        for idx, hb in zip(g.index, g['hist_bars'].to_numpy()):
            if hb - last >= REARM:
                keep.append(idx); last = hb
    return df.loc[keep]


con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
cal = [d for d in pd.read_sql_query(
    "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
    "AND symbol='NIFTYBEES' ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']
raw = pd.read_csv(RES / 'ath_events.csv')
raw = raw[raw['tv20_cr'] >= CFG['liq']]

# panel cache is gitignored-by-size; rebuild it if absent (~60 s)
if (RES / 'panel161.pkl').exists():
    panel = pickle.load(open(RES / 'panel161.pkl', 'rb'))
else:
    print('panel cache missing - rebuilding...')
    panel = B.Panel(con, sorted(raw['symbol'].unique()), cal)
    pickle.dump(panel, open(RES / 'panel161.pkl', 'wb'), protocol=4)
    print('panel rebuilt: %d symbols' % len(panel.close))
con.close()

raw = raw[raw['symbol'].isin(panel.close) & raw['entry_date'].isin(panel.pos)]
raw['entry_i'] = raw['entry_date'].map(panel.pos)
s = raw[(raw['tv20_cr'] >= CFG['liq']) & (raw['x_bars'] >= CFG['X'])
        & (raw['depth_pct'] >= CFG['dep'])]
s = rearm(s)
events = [dict(symbol=a, entry_i=int(b)) for a, b in zip(s['symbol'], s['entry_i'])]
print('OA V2.0 events: %d' % len(events))

cfg = dict(exit=CFG['ex'], hard_stop=CFG['hard'], time_stop=0, cost_bps=25.0, gate_ok=None)
rows, trade_sets, navs = [], {}, {}
for sd in SEEDS:
    nav, tr = B.simulate(events, panel, cfg, sd)
    m = B.metrics(nav, cal, tr)
    rows.append(dict(seed=sd, cagr_pct=round(m['cagr'], 2), max_drawdown_pct=round(m['maxdd'], 2),
                     calmar=round(m['calmar'], 3), sharpe=round(m['sharpe'], 3),
                     trades=int(m['trades']), win_rate_pct=round(m['win_rate'], 1),
                     avg_win_pct=round(m['avg_win'], 2), avg_loss_pct=round(m['avg_loss'], 2),
                     expectancy_pct=round(m['expectancy'], 3),
                     max_losing_streak=int(m['max_loss_streak']),
                     final_value_inr=int(m['final'])))
    trade_sets[sd] = tr
    navs[sd] = m['cagr']

summ = pd.DataFrame(rows).sort_values('seed')
summ.to_csv(RES / 'oa_v2_seed_summary.csv', index=False)
print('wrote oa_v2_seed_summary.csv (%d seeds)' % len(summ))

med_seed = int(summ.iloc[(summ['cagr_pct'] - summ['cagr_pct'].median()).abs().argsort().iloc[0]]['seed'])
print('median-CAGR seed = %d (CAGR %.2f%%)' % (med_seed, navs[med_seed]))

t = pd.DataFrame(trade_sets[med_seed])
t = t.rename(columns={'entry_px': 'entry_price', 'exit_px': 'exit_price',
                      'ret_pct': 'return_pct', 'pnl': 'pnl_inr', 'bars': 'bars_held',
                      'reason': 'exit_reason'})
t['seed'] = med_seed
t['entry_price'] = t['entry_price'].round(2)
t['exit_price'] = t['exit_price'].round(2)
t['return_pct'] = t['return_pct'].round(2)
t['pnl_inr'] = t['pnl_inr'].round(0).astype(int)
t['exit_reason'] = t['exit_reason'].replace({'ST_14_4': 'SuperTrend(14,4) trail',
                                             'EOD': 'open at final bar'})
t = t[['symbol', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'exit_reason',
       'bars_held', 'return_pct', 'pnl_inr', 'shares', 'seed']]
t = t.sort_values('entry_date')
t.to_csv(RES / 'oa_v2_trades_median_seed.csv', index=False)
print('wrote oa_v2_trades_median_seed.csv (%d trades)' % len(t))
print(t.head(8).to_string(index=False))
print('\nwinners %d / losers %d | best %+.1f%% | worst %+.1f%% | total P&L Rs%s'
      % ((t.return_pct > 0).sum(), (t.return_pct <= 0).sum(), t.return_pct.max(),
         t.return_pct.min(), format(int(t.pnl_inr.sum()), ',')))
