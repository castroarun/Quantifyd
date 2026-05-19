"""Pair trading - Stage A: cointegration screen.

Goal: from the 80-stock F&O universe, identify the top pairs that are
stably cointegrated on the TRAIN window 2018-01-01 to 2023-12-31, with
mean-reversion half-lives in the [3, 30]-day range and stable hedge
ratios.

Method:
  - Engle-Granger cointegration test (statsmodels.tsa.stattools.coint)
    on log-prices of the train window only (no look-ahead).
  - For survivors at p<0.05, fit OLS hedge ratio (beta) on log-prices.
  - Compute residual spread = log(P_a) - alpha - beta * log(P_b).
  - Estimate half-life of mean-reversion by AR(1) on the spread.
  - Filter: 3 <= half_life <= 30 days, |corr_pearson| >= 0.7.
  - Save survivors -> results/05_pair_universe.csv (incremental).

Resume: skips already-scored pairs. Safe to re-run.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine_daily import FNO_UNIVERSE, load_daily

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
LOGS = os.path.join(ROOT, 'logs')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

OUT_CSV = os.path.join(RESULTS, '05_pair_universe.csv')
LOG_PATH = os.path.join(LOGS, '05_pair_universe.log')

START = '2018-01-01'
TRAIN_END = '2023-12-31'

# Filters for "tradable" pairs
P_VALUE_MAX = 0.05
HALF_LIFE_MIN_DAYS = 3
HALF_LIFE_MAX_DAYS = 30
CORR_MIN = 0.70
MIN_OBS = 750  # require >~3 years of overlap on train window

FIELDNAMES = [
    'symA', 'symB', 'n_obs', 'pvalue', 'tstat', 'corr',
    'beta', 'alpha', 'spread_mean', 'spread_std', 'half_life',
    'spread_min', 'spread_max', 'last_z',
]


def log(msg: str) -> None:
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} | {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def _existing_pairs() -> set:
    if not os.path.exists(OUT_CSV):
        return set()
    done = set()
    with open(OUT_CSV, 'r', newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            done.add((row['symA'], row['symB']))
    return done


def _ensure_header() -> None:
    if not os.path.exists(OUT_CSV):
        with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
            csv.DictWriter(fh, fieldnames=FIELDNAMES).writeheader()


def _append(row: Dict) -> None:
    with open(OUT_CSV, 'a', newline='', encoding='utf-8') as fh:
        csv.DictWriter(fh, fieldnames=FIELDNAMES).writerow(row)


def _half_life(spread: np.ndarray) -> float:
    """Estimate AR(1) half-life of mean-reversion. Returns large number if non-stationary."""
    s = spread[np.isfinite(spread)]
    if len(s) < 30:
        return 9999.0
    ds = np.diff(s)
    s_lag = s[:-1]
    s_lag = s_lag - s_lag.mean()
    # ds = lambda * s_lag + eps  -> half_life = -ln(2)/lambda
    try:
        slope, _, _, _ = np.linalg.lstsq(s_lag.reshape(-1, 1), ds, rcond=None)
        lam = float(slope[0])
    except Exception:
        return 9999.0
    if lam >= 0:
        # not mean-reverting
        return 9999.0
    hl = -np.log(2) / lam
    if not np.isfinite(hl) or hl <= 0:
        return 9999.0
    return float(hl)


def screen_pair(la: pd.Series, lb: pd.Series, symA: str, symB: str) -> Dict | None:
    """Compute cointegration metrics on log-price series. Returns dict
    if pair clears the screen, else None.

    All inputs are TRAIN-WINDOW ONLY (no look-ahead).
    """
    df = pd.concat([la, lb], axis=1, join='inner').dropna()
    df.columns = ['la', 'lb']
    if len(df) < MIN_OBS:
        return None

    # Engle-Granger
    try:
        tstat, pvalue, _ = coint(df['la'].values, df['lb'].values)
    except Exception:
        return None
    if not np.isfinite(pvalue) or pvalue > P_VALUE_MAX:
        return None

    # Correlation
    corr = float(df['la'].corr(df['lb']))
    if abs(corr) < CORR_MIN:
        return None

    # OLS: la = alpha + beta * lb + e
    X = sm.add_constant(df['lb'].values)
    try:
        ols = sm.OLS(df['la'].values, X).fit()
    except Exception:
        return None
    alpha = float(ols.params[0])
    beta = float(ols.params[1])
    spread = df['la'].values - alpha - beta * df['lb'].values

    hl = _half_life(spread)
    if hl < HALF_LIFE_MIN_DAYS or hl > HALF_LIFE_MAX_DAYS:
        return None

    smean = float(np.mean(spread))
    sstd = float(np.std(spread, ddof=1))
    if sstd <= 0:
        return None
    last_z = float((spread[-1] - smean) / sstd)

    return dict(
        symA=symA,
        symB=symB,
        n_obs=int(len(df)),
        pvalue=float(pvalue),
        tstat=float(tstat),
        corr=corr,
        beta=beta,
        alpha=alpha,
        spread_mean=smean,
        spread_std=sstd,
        half_life=hl,
        spread_min=float(np.min(spread)),
        spread_max=float(np.max(spread)),
        last_z=last_z,
    )


def main() -> None:
    log(f'Pair universe screener starting. Universe size {len(FNO_UNIVERSE)}.')
    _ensure_header()
    done = _existing_pairs()
    log(f'Found {len(done)} already-screened pairs in {OUT_CSV}.')

    # Load daily series for all symbols, then trim to train window
    log('Loading daily prices for all symbols...')
    series: Dict[str, pd.Series] = {}
    for sym in FNO_UNIVERSE:
        df = load_daily(sym, start=START, end=TRAIN_END)
        if df.empty or len(df) < MIN_OBS:
            log(f'  SKIP {sym}: too few train rows ({len(df)})')
            continue
        # log price
        s = np.log(df['close'].astype(float))
        s.name = sym
        series[sym] = s
    log(f'Loaded {len(series)} symbols with sufficient train data.')

    syms = sorted(series.keys())
    pairs = list(combinations(syms, 2))
    log(f'Screening {len(pairs)} candidate pairs.')

    survivors = 0
    examined = 0
    t0 = time.time()
    for i, (a, b) in enumerate(pairs):
        if (a, b) in done:
            continue
        examined += 1
        try:
            res = screen_pair(series[a], series[b], a, b)
        except Exception as e:
            log(f'  ERROR on ({a},{b}): {e}')
            continue
        if res is None:
            continue
        _append(res)
        survivors += 1
        if survivors % 5 == 0:
            log(f'[{i+1}/{len(pairs)}] survivors={survivors}, last={a}-{b} '
                f'p={res["pvalue"]:.4f} hl={res["half_life"]:.1f} corr={res["corr"]:.2f}')
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            log(f'progress: {i+1}/{len(pairs)} | survivors={survivors} | '
                f'examined={examined} | {elapsed:.0f}s')

    elapsed = time.time() - t0
    log(f'DONE. examined={examined}, survivors={survivors}, elapsed={elapsed:.0f}s')


if __name__ == '__main__':
    main()
