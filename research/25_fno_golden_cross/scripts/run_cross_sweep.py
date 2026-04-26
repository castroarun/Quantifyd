"""Signal 4 — Golden / Death cross sweep (both directions).

LONG side (golden cross — fast MA crosses ABOVE slow MA):
  gc_sma_50_200          classic
  gc_ema_50_200          EMA 50/200 instead of SMA
  gc_ema_20_50           faster EMA 20/50
  gc_sma_50_200_stoch    classic + stochastic K/D crossing up from <30
  gc_sma_50_200_adx      classic + ADX > 25
  gc_sma_50_200_confluence  classic + stoch + ADX + 200-SMA above

SHORT side (death cross mirror): same 6 variants with `dc_` prefix.
Cross must be RECENT — within last `lookback_bars` (default 5).
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SHARED = HERE.parents[1] / '22_fno_bearish_breakdown' / 'scripts'
sys.path.insert(0, str(SHARED))

from _shared_engine import (   # type: ignore
    load_universe, load_all_bars, run_engine, compute_metrics,
    write_trades, write_summary, heartbeat, START_DATE, END_DATE,
)

RES = HERE.parent / 'results'
LOGS = HERE.parent / 'logs'
RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
HB = LOGS / 'sweep_heartbeat.txt'
UNI_CSV = RES / 'universe.csv'


def make_cfg(name, ma_type='sma', fast=50, slow=200, lookback_bars=5,
             use_stoch=False, use_adx=False, adx_thr=25, use_above_sma200=False,
             target_pct=0.25, hard_stop_pct=0.08, cost_pct=0.0020):
    return dict(name=name, ma_type=ma_type, fast=fast, slow=slow,
                lookback_bars=lookback_bars, use_stoch=use_stoch,
                use_adx=use_adx, adx_thr=adx_thr,
                use_above_sma200=use_above_sma200,
                target_pct=target_pct, hard_stop_pct=hard_stop_pct, cost_pct=cost_pct)


VARIANTS_LONG = [
    make_cfg('gc_sma_50_200',           ma_type='sma', fast=50,  slow=200),
    make_cfg('gc_ema_50_200',           ma_type='ema', fast=50,  slow=200),
    make_cfg('gc_ema_20_50',            ma_type='ema', fast=20,  slow=50),
    make_cfg('gc_sma_50_200_stoch',     ma_type='sma', fast=50,  slow=200, use_stoch=True),
    make_cfg('gc_sma_50_200_adx',       ma_type='sma', fast=50,  slow=200, use_adx=True),
    make_cfg('gc_sma_50_200_confluence',ma_type='sma', fast=50,  slow=200,
             use_stoch=True, use_adx=True, use_above_sma200=True),
]
# Mirror for short / death cross
VARIANTS_SHORT = [
    {**v, 'name': v['name'].replace('gc_', 'dc_', 1)} for v in VARIANTS_LONG
]


def _get_ma(df_or_row, ma_type, period, kind='row'):
    """Look up the right precomputed MA. Shared engine has sma_50, sma_200, ema_20, ema_50, ema_200."""
    key_map = {
        ('sma', 50):  'sma_50',
        ('sma', 200): 'sma_200',
        ('ema', 20):  'ema_20',
        ('ema', 50):  'ema_50',
        ('ema', 200): 'ema_200',
    }
    key = key_map.get((ma_type, period))
    if key is None:
        return None
    return df_or_row.get(key) if kind == 'row' else df_or_row[key]


# To detect a "recent cross" we need history, not just today's row.
# We pre-build a per-symbol cross-flag column once (cheap), then signal_fn just reads it.
def _annotate_crosses(bars: dict[str, pd.DataFrame]) -> None:
    """Add cross-detection columns:
       gc_<fast>_<slow>: True if fast crossed above slow within last 5 bars.
       dc_<fast>_<slow>: True if fast crossed below slow within last 5 bars.
       Also: stoch_cross_up (k crossing up from <30)
    """
    for sym, df in bars.items():
        # Compute cross flags
        for ma_type, fast, slow in [('sma', 50, 200), ('ema', 50, 200), ('ema', 20, 50)]:
            f_key = {('sma',50):'sma_50',('ema',50):'ema_50',('ema',20):'ema_20'}[(ma_type, fast)]
            s_key = {('sma',200):'sma_200',('ema',200):'ema_200',('ema',50):'ema_50'}[(ma_type, slow)]
            f_ser = df[f_key]; s_ser = df[s_key]
            # cross-up: prev <= and current >
            cross_up = (f_ser.shift(1) <= s_ser.shift(1)) & (f_ser > s_ser)
            cross_dn = (f_ser.shift(1) >= s_ser.shift(1)) & (f_ser < s_ser)
            df[f'gc_{ma_type}_{fast}_{slow}'] = cross_up.rolling(5).sum() > 0
            df[f'dc_{ma_type}_{fast}_{slow}'] = cross_dn.rolling(5).sum() > 0

        # Stochastic %K crossing up from <30 (and crossing down from >70)
        k = df['stoch_k']; d = df['stoch_d']
        k_up = (k.shift(1) <= d.shift(1)) & (k > d) & (k.shift(1) < 30)
        k_dn = (k.shift(1) >= d.shift(1)) & (k < d) & (k.shift(1) > 70)
        df['stoch_cross_up']   = k_up.rolling(5).sum() > 0
        df['stoch_cross_down'] = k_dn.rolling(5).sum() > 0


def signal_fn_long(row: pd.Series, cfg: dict) -> bool:
    flag_key = f'gc_{cfg["ma_type"]}_{cfg["fast"]}_{cfg["slow"]}'
    if not bool(row.get(flag_key, False)):
        return False
    if cfg.get('use_stoch') and not bool(row.get('stoch_cross_up', False)):
        return False
    if cfg.get('use_adx'):
        adx = row.get('adx_14')
        if pd.isna(adx) or adx <= cfg.get('adx_thr', 25):
            return False
    if cfg.get('use_above_sma200'):
        sma200 = row.get('sma_200')
        if pd.isna(sma200) or row['close'] <= sma200:
            return False
    return True


def signal_fn_short(row: pd.Series, cfg: dict) -> bool:
    flag_key = f'dc_{cfg["ma_type"]}_{cfg["fast"]}_{cfg["slow"]}'
    if not bool(row.get(flag_key, False)):
        return False
    if cfg.get('use_stoch') and not bool(row.get('stoch_cross_down', False)):
        return False
    if cfg.get('use_adx'):
        adx = row.get('adx_14')
        if pd.isna(adx) or adx <= cfg.get('adx_thr', 25):
            return False
    if cfg.get('use_above_sma200'):
        sma200 = row.get('sma_200')
        if pd.isna(sma200) or row['close'] >= sma200:
            return False
    return True


def main():
    t_start = time.time()
    heartbeat(HB, 'start')
    uni = load_universe(UNI_CSV); bars = load_all_bars(uni)
    print(f'Loaded {len(bars)} stocks; period {START_DATE} -> {END_DATE}', flush=True)
    print('Annotating cross flags...', flush=True)
    _annotate_crosses(bars)

    summary_rows = []
    print(f'\n{"Variant":>30} {"Dir":>5} {"Trades":>7} {"WR%":>6} {"PF":>6} {"AvgD":>6} '
          f'{"CAGR%":>7} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7}', flush=True)
    print('-' * 100, flush=True)

    best_pf, best_var, best_trades, best_dir = -1.0, None, None, None
    for direction, vlist, sfn in [('LONG', VARIANTS_LONG, signal_fn_long),
                                  ('SHORT', VARIANTS_SHORT, signal_fn_short)]:
        for cfg in vlist:
            heartbeat(HB, f'running {direction} {cfg["name"]}')
            t0 = time.time()
            trades, eq = run_engine(bars, cfg, sfn, direction)
            m = compute_metrics(trades, eq)
            summary_rows.append({'variant': cfg['name'], 'direction': direction, 'metrics': m})
            print(f'{cfg["name"]:>30} {direction:>5} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} '
                  f'{m["profit_factor"]:>6.2f} {m["avg_days_held"]:>6.1f} '
                  f'{m["cagr_pct"]:>+7.2f} {m["sharpe"]:>7.2f} '
                  f'{m["max_dd_pct"]:>7.2f} {m["calmar"]:>7.2f}  ({time.time()-t0:.1f}s)', flush=True)
            eq.to_csv(RES / f'equity_{cfg["name"]}_{direction}.csv', header=['equity'])
            if m['profit_factor'] > best_pf and m['trades'] >= 5:
                best_pf, best_var, best_trades, best_dir = m['profit_factor'], cfg['name'], trades, direction

    write_summary(summary_rows, RES / 'summary.csv')
    if best_trades is not None:
        write_trades(best_trades, RES / f'trades_{best_var}_{best_dir}.csv')
    print(f'\nBest by PF: {best_var} ({best_dir}) PF={best_pf:.2f}')
    print(f'Total runtime: {time.time()-t_start:.1f}s', flush=True)
    heartbeat(HB, f'DONE | best={best_var}/{best_dir} PF={best_pf:.2f}')


if __name__ == '__main__':
    sys.exit(main())
