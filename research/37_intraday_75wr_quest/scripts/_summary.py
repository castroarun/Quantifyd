import os
import pandas as pd

CSV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'results', '11b_long_trend_pullback_ranking.csv'))

df = pd.read_csv(CSV)
df_filt = df[df['n_trades'] >= 30].copy()
df_filt = df_filt.sort_values('win_rate', ascending=False)

print('TOP 5 by WR (n>=30):')
for i, row in df_filt.head(5).iterrows():
    print(f"  WR={row['win_rate']:.3f} n={int(row['n_trades'])} PF={row['profit_factor']:.2f} "
          f"DD={row['max_dd_pct']:.1f}% Sh={row['sharpe']:.2f} tpsy={row['trades_per_stock_year']:.1f} "
          f"tp={row['tp_pct']} sl={row['sl_pct']} {row['params']}")

print()
print('Systems with WR>=0.78 AND PF>=2.0 AND tpsy>=4 AND n>=30:')
qual = df_filt[(df_filt['win_rate']>=0.78) & (df_filt['profit_factor']>=2.0) & (df_filt['trades_per_stock_year']>=4)]
for i, row in qual.iterrows():
    print(f"  WR={row['win_rate']:.3f} n={int(row['n_trades'])} PF={row['profit_factor']:.2f} "
          f"DD={row['max_dd_pct']:.1f}% Sh={row['sharpe']:.2f} tpsy={row['trades_per_stock_year']:.1f} "
          f"tp={row['tp_pct']} sl={row['sl_pct']} {row['params']}")

print()
print('Systems passing strict gates (WR>=75%, PF>=2, tpsy>=8, n>=30):')
strict = df_filt[(df_filt['win_rate']>=0.75) & (df_filt['profit_factor']>=2.0) & (df_filt['trades_per_stock_year']>=8) & (df_filt['n_trades']>=30)]
print(f'  count: {len(strict)}')
for i, row in strict.iterrows():
    print(f"  WR={row['win_rate']:.3f} n={int(row['n_trades'])} PF={row['profit_factor']:.2f} "
          f"DD={row['max_dd_pct']:.1f}% Sh={row['sharpe']:.2f} tpsy={row['trades_per_stock_year']:.1f} "
          f"tp={row['tp_pct']} sl={row['sl_pct']} {row['params']}")

print()
print('Best balance: WR>=78% AND tpsy>=10:')
bal = df_filt[(df_filt['win_rate']>=0.78) & (df_filt['trades_per_stock_year']>=10)]
for i, row in bal.iterrows():
    print(f"  WR={row['win_rate']:.3f} n={int(row['n_trades'])} PF={row['profit_factor']:.2f} "
          f"DD={row['max_dd_pct']:.1f}% Sh={row['sharpe']:.2f} tpsy={row['trades_per_stock_year']:.1f} "
          f"tp={row['tp_pct']} sl={row['sl_pct']} {row['params']}")
