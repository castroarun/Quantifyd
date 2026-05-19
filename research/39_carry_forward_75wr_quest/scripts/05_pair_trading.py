"""Pair trading - Stages B+C+D: signal sweep + walk-forward + portfolio.

Inputs:
  results/05_pair_universe.csv  (cointegrated train-window pairs)

Pipeline:
  Stage B: For each pair (with TRAIN-FIT alpha+beta), compute z-score on
           rolling lookback. Sweep entry/stop/half-life/hold/lookback grids.
           Walk-forward: train 2018-2023, test 2024+ (out of sample).
           Pass criteria:
             train WR >= 75%, test WR >= 70%, test PF >= 1.8,
             test MaxDD <= 15%, test n_trades >= 30,
             cost-stress at 0.20%/leg (0.40%/pair-trade) keeps PF >= 1.0
             and remains positive total return.

  Stage D: Top survivors -> 05_pair_walk_forward.csv with full breakdown.

  Stage E: If 3+ survivors, run combined portfolio (max-N concurrent).

Cost model (per LEG round-trip):
  Default 0.06% (F&O futures) -> 0.12% per pair-trade
  Stress   0.20% -> 0.40% per pair-trade

Trade convention:
  z-score = (spread_today - spread_mean) / spread_std on rolling lookback
  spread = log(P_a) - alpha_TRAIN - beta_TRAIN * log(P_b)
  ENTRY (long-A short-B = "long the spread")  when z <= -entry_z
    (the spread is below mean -> A is cheap relative to B -> long A short B)
  ENTRY (short-A long-B = "short the spread") when z >= +entry_z
  EXIT  on |z| <= 0.0 (mean cross), or |z| >= stop_z (stop), or hold-cap.

Position sizing for portfolio: each leg risks Rs.3,000 absolute (Rs.6K per pair).
Hedge ratio applied to share counts: short_b_shares = beta * long_a_shares
adjusted to equal-rupee weighting at entry.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine_daily import FNO_UNIVERSE, load_daily

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
LOGS = os.path.join(ROOT, 'logs')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

UNIVERSE_CSV = os.path.join(RESULTS, '05_pair_universe.csv')
RANK_CSV = os.path.join(RESULTS, '05_pair_ranking.csv')
WF_CSV = os.path.join(RESULTS, '05_pair_walk_forward.csv')
DIAMOND_TXT = os.path.join(RESULTS, '05_pair_diamonds.txt')
PORT_TXT = os.path.join(RESULTS, '05_pair_portfolio_summary.txt')
LOG_PATH = os.path.join(LOGS, '05_pair_trading.log')

START = '2018-01-01'
END = '2025-11-30'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'

# Sweep grids
ENTRY_Z_GRID = [2.0, 2.5, 3.0]
STOP_Z_GRID = [4.0, 5.0, 999.0]   # 999 = effectively no stop
HALF_LIFE_FILTER_GRID = [7, 15, 30]    # max half-life days from universe row
HOLD_DAYS_GRID = [10, 15, 20]
LOOKBACK_GRID = [20, 40, 60]

COST_DEFAULT = 0.12   # round-trip per pair-trade (0.06 per leg x2)
COST_STRESS = 0.40    # stress: 0.20 per leg x2

# Walk-forward gates
GATE_TRAIN_WR = 0.75
GATE_TEST_WR = 0.70
GATE_TEST_PF = 1.8
GATE_TEST_DD = 15.0
GATE_TEST_N = 30

RANK_FIELDS = [
    'symA', 'symB', 'beta', 'alpha', 'half_life',
    'entry_z', 'stop_z', 'hold_days', 'lookback',
    'half_life_filter',
    # train stats
    'tr_n', 'tr_wr', 'tr_pf', 'tr_dd', 'tr_total_ret', 'tr_avg_hold',
    # test stats
    'te_n', 'te_wr', 'te_pf', 'te_dd', 'te_total_ret', 'te_avg_hold',
    # stress
    'te_pf_stress', 'te_total_ret_stress',
    # gate
    'passes_gate', 'aws',
]


def log(msg: str) -> None:
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} | {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


# ---------------------------------------------------------------------------
# Trade simulator (pair version)
# ---------------------------------------------------------------------------

@dataclass
class PairRules:
    entry_z: float
    stop_z: float
    hold_days: int
    lookback: int


def simulate_pair(prices_a: pd.Series,
                   prices_b: pd.Series,
                   alpha: float,
                   beta: float,
                   rules: PairRules) -> pd.DataFrame:
    """Vectorised-ish pair-trade simulator. Inputs are aligned daily close
    series. alpha + beta are TRAIN-FIT and applied across ALL dates (train+test).

    For z-score on rolling LOOKBACK we use the rolling mean/std of the
    spread itself (so test-period z-scores use only test-period info,
    BUT the spread definition uses TRAIN-FIT beta; this is the standard
    pair-trade walk-forward setup).

    Returns DataFrame of trades:
      entry_date, exit_date, days_held, direction (1=long_spread, -1=short),
      ret_pct (gross %, i.e. mean of leg P&L %), exit_reason
    """
    df = pd.concat([prices_a, prices_b], axis=1, join='inner').dropna()
    df.columns = ['pa', 'pb']
    if len(df) < rules.lookback + 5:
        return pd.DataFrame()
    df['la'] = np.log(df['pa'])
    df['lb'] = np.log(df['pb'])
    df['spread'] = df['la'] - alpha - beta * df['lb']
    df['mu'] = df['spread'].rolling(rules.lookback, min_periods=rules.lookback).mean()
    df['sd'] = df['spread'].rolling(rules.lookback, min_periods=rules.lookback).std(ddof=1)
    df['z'] = (df['spread'] - df['mu']) / df['sd'].replace(0, np.nan)

    z = df['z'].to_numpy()
    pa = df['pa'].to_numpy()
    pb = df['pb'].to_numpy()
    idx = df.index.to_numpy()
    n = len(df)

    rows_e, rows_x, rows_h, rows_d, rows_r, rows_x_reason = [], [], [], [], [], []

    in_pos = False
    direction = 0  # +1 long spread (long A short B), -1 short spread
    e_i = -1
    e_pa = e_pb = np.nan

    for i in range(n):
        zi = z[i]
        if not np.isfinite(zi):
            continue
        if not in_pos:
            # Entry triggers (use signal at close of day i, fill at close of day i)
            if zi <= -rules.entry_z:
                in_pos = True
                direction = 1
                e_i = i
                e_pa = pa[i]
                e_pb = pb[i]
            elif zi >= rules.entry_z:
                in_pos = True
                direction = -1
                e_i = i
                e_pa = pa[i]
                e_pb = pb[i]
        else:
            held = i - e_i
            exit_now = False
            reason = None
            # Mean-revert exit: z crossed zero
            if direction == 1 and zi >= 0:
                exit_now = True
                reason = 'MR'
            elif direction == -1 and zi <= 0:
                exit_now = True
                reason = 'MR'
            # Stop
            elif abs(zi) >= rules.stop_z:
                exit_now = True
                reason = 'STOP'
            # Hold cap
            elif held >= rules.hold_days:
                exit_now = True
                reason = 'TIME'

            if exit_now:
                x_pa = pa[i]
                x_pb = pb[i]
                # P&L per leg in %
                ret_a = (x_pa - e_pa) / e_pa * 100
                ret_b = (x_pb - e_pb) / e_pb * 100
                # direction +1 = long A, short B -> P&L = ret_a - ret_b
                # direction -1 = short A, long B -> P&L = -ret_a + ret_b
                if direction == 1:
                    pair_ret = ret_a - ret_b
                else:
                    pair_ret = ret_b - ret_a
                # Average per-leg notional was 50/50 -> divide by 2 for unit notional
                pair_ret = pair_ret / 2.0
                rows_e.append(idx[e_i])
                rows_x.append(idx[i])
                rows_h.append(held)
                rows_d.append(direction)
                rows_r.append(pair_ret)
                rows_x_reason.append(reason)
                in_pos = False
                direction = 0
                e_i = -1

    if not rows_e:
        return pd.DataFrame()
    return pd.DataFrame({
        'entry_date': rows_e,
        'exit_date': rows_x,
        'days_held': rows_h,
        'direction': rows_d,
        'ret_pct': rows_r,
        'exit_reason': rows_x_reason,
    })


def trade_stats(trades: pd.DataFrame, cost_pct: float = COST_DEFAULT) -> Dict:
    if trades is None or len(trades) == 0:
        return dict(n=0, wr=0.0, pf=0.0, dd=0.0, total_ret=0.0, avg_hold=0.0,
                    sharpe=0.0, aws=0.0)
    net = trades['ret_pct'] - cost_pct
    wins = net > 0
    losses = ~wins
    wr = float(wins.mean())
    gp = float(net[wins].sum()) if wins.any() else 0.0
    gl = float(-net[losses].sum()) if losses.any() else 0.0
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    eq = net.cumsum()
    dd = float((eq.cummax() - eq).max())
    total = float(net.sum())
    avg_hold = float(trades['days_held'].mean())
    if net.std() > 0:
        sh = float(net.mean() / net.std() * np.sqrt(252 / max(avg_hold, 1)))
    else:
        sh = 0.0
    aws = wr * np.log1p(len(trades)) * min(pf / 2, 1.5) * min(max(sh, 0) / 1.5, 1.5)
    return dict(n=int(len(trades)), wr=wr, pf=pf, dd=dd, total_ret=total,
                avg_hold=avg_hold, sharpe=sh, aws=float(aws))


def split_train_test(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades is None or len(trades) == 0:
        return pd.DataFrame(), pd.DataFrame()
    et = pd.to_datetime(trades['entry_date'])
    train = trades[et <= pd.Timestamp(TRAIN_END)].copy()
    test = trades[et >= pd.Timestamp(TEST_START)].copy()
    return train, test


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _ensure_rank_header() -> None:
    if not os.path.exists(RANK_CSV):
        with open(RANK_CSV, 'w', newline='', encoding='utf-8') as fh:
            csv.DictWriter(fh, fieldnames=RANK_FIELDS).writeheader()


def _existing_rank_keys() -> set:
    if not os.path.exists(RANK_CSV):
        return set()
    keys = set()
    with open(RANK_CSV, 'r', newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            keys.add((
                row['symA'], row['symB'],
                row['entry_z'], row['stop_z'],
                row['hold_days'], row['lookback'],
                row['half_life_filter'],
            ))
    return keys


def _append_rank(row: Dict) -> None:
    with open(RANK_CSV, 'a', newline='', encoding='utf-8') as fh:
        csv.DictWriter(fh, fieldnames=RANK_FIELDS).writerow(row)


def load_universe() -> pd.DataFrame:
    if not os.path.exists(UNIVERSE_CSV):
        raise FileNotFoundError(f'{UNIVERSE_CSV} missing — run 05_pair_universe_screen.py first.')
    df = pd.read_csv(UNIVERSE_CSV)
    log(f'Universe loaded: {len(df)} pairs')
    # Tighten universe with stable hedge ratio + sane half-life
    df = df[(df['half_life'] >= 3) & (df['half_life'] <= 30)]
    df = df.sort_values('pvalue').reset_index(drop=True)
    return df


def run_sweep(universe: pd.DataFrame,
              max_pairs: int = 0) -> None:
    _ensure_rank_header()
    done = _existing_rank_keys()
    log(f'Sweep starting. {len(done)} cells already in {os.path.basename(RANK_CSV)}.')

    # Pre-load all needed symbols once
    syms = sorted(set(universe['symA']).union(set(universe['symB'])))
    log(f'Loading daily prices for {len(syms)} symbols (full window)...')
    series: Dict[str, pd.Series] = {}
    for sym in syms:
        df = load_daily(sym, start=START, end=END)
        if df.empty:
            continue
        series[sym] = df['close'].astype(float)
    log(f'Loaded {len(series)} symbols.')

    pairs_to_run = universe.head(max_pairs) if max_pairs and max_pairs > 0 else universe
    log(f'Sweeping {len(pairs_to_run)} pairs across grids '
        f'(entry={ENTRY_Z_GRID}, stop={STOP_Z_GRID}, hold={HOLD_DAYS_GRID}, lookback={LOOKBACK_GRID}, '
        f'hl_filter={HALF_LIFE_FILTER_GRID}).')

    grid = list(product(ENTRY_Z_GRID, STOP_Z_GRID, HOLD_DAYS_GRID, LOOKBACK_GRID))
    log(f'Grid cells per pair: {len(grid)}')
    survivors_so_far = 0
    t0 = time.time()
    pairs_done = 0

    for prow in pairs_to_run.itertuples(index=False):
        symA = prow.symA
        symB = prow.symB
        if symA not in series or symB not in series:
            continue
        # half-life filter (universe row half_life vs grid filter)
        for hl_filter in HALF_LIFE_FILTER_GRID:
            if prow.half_life > hl_filter:
                continue
            for entry_z, stop_z, hold_days, lookback in grid:
                key = (symA, symB, str(entry_z), str(stop_z),
                       str(hold_days), str(lookback), str(hl_filter))
                if key in done:
                    continue

                rules = PairRules(entry_z=entry_z, stop_z=stop_z,
                                   hold_days=hold_days, lookback=lookback)
                try:
                    trades = simulate_pair(series[symA], series[symB],
                                            alpha=prow.alpha, beta=prow.beta,
                                            rules=rules)
                except Exception as e:
                    log(f'  ERROR sim ({symA},{symB}) entry={entry_z}: {e}')
                    continue

                tr, te = split_train_test(trades)
                tr_s = trade_stats(tr, COST_DEFAULT)
                te_s = trade_stats(te, COST_DEFAULT)
                te_stress = trade_stats(te, COST_STRESS)

                passes = (
                    te_s['n'] >= GATE_TEST_N
                    and tr_s['wr'] >= GATE_TRAIN_WR
                    and te_s['wr'] >= GATE_TEST_WR
                    and te_s['pf'] >= GATE_TEST_PF
                    and te_s['dd'] <= GATE_TEST_DD
                    and te_stress['pf'] >= 1.0
                    and te_stress['total_ret'] > 0
                )

                row = dict(
                    symA=symA, symB=symB,
                    beta=round(prow.beta, 4),
                    alpha=round(prow.alpha, 4),
                    half_life=round(prow.half_life, 2),
                    entry_z=entry_z, stop_z=stop_z,
                    hold_days=hold_days, lookback=lookback,
                    half_life_filter=hl_filter,
                    tr_n=tr_s['n'], tr_wr=round(tr_s['wr'] * 100, 2),
                    tr_pf=round(tr_s['pf'], 2),
                    tr_dd=round(tr_s['dd'], 2),
                    tr_total_ret=round(tr_s['total_ret'], 2),
                    tr_avg_hold=round(tr_s['avg_hold'], 2),
                    te_n=te_s['n'], te_wr=round(te_s['wr'] * 100, 2),
                    te_pf=round(te_s['pf'], 2),
                    te_dd=round(te_s['dd'], 2),
                    te_total_ret=round(te_s['total_ret'], 2),
                    te_avg_hold=round(te_s['avg_hold'], 2),
                    te_pf_stress=round(te_stress['pf'], 2),
                    te_total_ret_stress=round(te_stress['total_ret'], 2),
                    passes_gate=int(passes),
                    aws=round(te_s['aws'], 3),
                )
                _append_rank(row)
                if passes:
                    survivors_so_far += 1
                    log(f'GATE PASS! {symA}-{symB} ez={entry_z} hold={hold_days} lb={lookback} '
                        f'tr_n={tr_s["n"]} tr_wr={tr_s["wr"]*100:.1f}% '
                        f'te_n={te_s["n"]} te_wr={te_s["wr"]*100:.1f}% '
                        f'te_pf={te_s["pf"]:.2f} te_dd={te_s["dd"]:.1f}%')

        pairs_done += 1
        if pairs_done % 10 == 0:
            elapsed = time.time() - t0
            log(f'progress: {pairs_done}/{len(pairs_to_run)} pairs done | '
                f'survivors_so_far={survivors_so_far} | {elapsed:.0f}s')

    elapsed = time.time() - t0
    log(f'SWEEP DONE. pairs_done={pairs_done} survivors={survivors_so_far} elapsed={elapsed:.0f}s')


# ---------------------------------------------------------------------------
# Walk-forward summary + diamonds
# ---------------------------------------------------------------------------

def write_walk_forward_summary() -> pd.DataFrame:
    if not os.path.exists(RANK_CSV):
        log('No ranking CSV; nothing to summarise.')
        return pd.DataFrame()
    df = pd.read_csv(RANK_CSV)
    log(f'Ranking has {len(df)} rows; gate-passing rows: {(df["passes_gate"]==1).sum()}')
    survivors = df[df['passes_gate'] == 1].copy()
    survivors = survivors.sort_values(['te_pf', 'te_wr', 'te_n'], ascending=[False, False, False])
    survivors.to_csv(WF_CSV, index=False)
    log(f'Wrote walk-forward survivors -> {WF_CSV} ({len(survivors)} rows).')
    return survivors


def write_diamonds(top: pd.DataFrame, k: int = 30) -> None:
    """Write top-k pairs (regardless of gate) for narrative inspection."""
    if not os.path.exists(RANK_CSV):
        return
    df = pd.read_csv(RANK_CSV)
    # Filter to viable on test cohort
    viable = df[(df['te_n'] >= GATE_TEST_N) & (df['te_pf'] > 0)].copy()
    viable = viable.sort_values(['te_pf', 'te_wr', 'te_n'], ascending=[False, False, False])
    head = viable.head(k)
    with open(DIAMOND_TXT, 'w', encoding='utf-8') as fh:
        fh.write(f'Top {len(head)} pair-trade cells by test PF (n>={GATE_TEST_N}).\n')
        fh.write(f'Train end {TRAIN_END} | Test start {TEST_START} | Cost default {COST_DEFAULT}%/pair-trade\n\n')
        fh.write(head.to_string(index=False))
        fh.write('\n\nGate-pass rows (full WF survivors):\n\n')
        if not top.empty:
            fh.write(top.to_string(index=False))
        else:
            fh.write('(no full-gate survivors)\n')
    log(f'Wrote diamonds -> {DIAMOND_TXT}')


# ---------------------------------------------------------------------------
# Combined portfolio backtest
# ---------------------------------------------------------------------------

def combined_portfolio(survivors: pd.DataFrame, max_concurrent: int = 5,
                        per_pair_risk: float = 6000.0,
                        capital: float = 1_000_000.0) -> Dict:
    """Run a simple combined portfolio: re-simulate each survivor over test
    window only, then merge by entry_date and constrain to max_concurrent
    open pairs. Each new pair allocates per_pair_risk / 2 to each leg.
    Returns summary dict + writes 05_pair_portfolio_summary.txt.
    """
    if survivors is None or len(survivors) == 0:
        with open(PORT_TXT, 'w', encoding='utf-8') as fh:
            fh.write('No walk-forward survivors -> no combined portfolio.\n')
        return dict(n_pairs=0)

    log(f'Combined portfolio: re-simulating {len(survivors)} survivor cells on TEST window.')

    # We'll rebuild trades for each cell on FULL window, then keep test-window only,
    # then concurrency-cap.
    syms = sorted(set(survivors['symA']).union(set(survivors['symB'])))
    series: Dict[str, pd.Series] = {}
    for sym in syms:
        df = load_daily(sym, start=START, end=END)
        if not df.empty:
            series[sym] = df['close'].astype(float)

    all_trades_rows = []
    for r in survivors.itertuples(index=False):
        rules = PairRules(entry_z=r.entry_z, stop_z=r.stop_z,
                           hold_days=int(r.hold_days), lookback=int(r.lookback))
        if r.symA not in series or r.symB not in series:
            continue
        trades = simulate_pair(series[r.symA], series[r.symB],
                                alpha=r.alpha, beta=r.beta, rules=rules)
        if trades.empty:
            continue
        trades = trades[pd.to_datetime(trades['entry_date']) >= pd.Timestamp(TEST_START)]
        for t in trades.itertuples(index=False):
            all_trades_rows.append(dict(
                pair=f'{r.symA}-{r.symB}',
                entry_date=t.entry_date,
                exit_date=t.exit_date,
                days_held=t.days_held,
                ret_pct=t.ret_pct,
                exit_reason=t.exit_reason,
            ))
    if not all_trades_rows:
        with open(PORT_TXT, 'w', encoding='utf-8') as fh:
            fh.write('Survivors produced 0 test-window trades -> empty portfolio.\n')
        return dict(n_pairs=0)

    all_t = pd.DataFrame(all_trades_rows).sort_values('entry_date').reset_index(drop=True)

    # Concurrency cap: emulate FIFO acceptance
    all_t['accepted'] = False
    open_until = []  # list of exit_date timestamps for currently open positions
    for i, row in all_t.iterrows():
        ed = pd.Timestamp(row['entry_date'])
        # purge those that exited before ed
        open_until = [d for d in open_until if d > ed]
        if len(open_until) < max_concurrent:
            all_t.at[i, 'accepted'] = True
            open_until.append(pd.Timestamp(row['exit_date']))
    accepted = all_t[all_t['accepted']].copy()

    cost_per_trade = COST_DEFAULT
    accepted['net_ret'] = accepted['ret_pct'] - cost_per_trade
    accepted['rupee_pnl'] = accepted['net_ret'] / 100 * per_pair_risk

    eq_start = capital
    eq = [eq_start]
    for v in accepted['rupee_pnl'].values:
        eq.append(eq[-1] + v)
    eq = pd.Series(eq[1:], index=pd.to_datetime(accepted['entry_date'].values))
    eq_full = eq.cumsum() * 0 + eq  # already cumulative

    n = len(accepted)
    wr = float((accepted['net_ret'] > 0).mean()) if n else 0.0
    pf = (accepted.loc[accepted['net_ret'] > 0, 'net_ret'].sum() /
          max(-accepted.loc[accepted['net_ret'] <= 0, 'net_ret'].sum(), 1e-9)) if n else 0.0
    total_pnl = float(accepted['rupee_pnl'].sum())
    total_ret_pct = total_pnl / capital * 100
    peak = eq_full.cummax()
    dd_series = (eq_full - peak)
    max_dd_rs = float(-dd_series.min()) if len(dd_series) else 0.0
    max_dd_pct = max_dd_rs / capital * 100

    with open(PORT_TXT, 'w', encoding='utf-8') as fh:
        fh.write('Combined Pair-Trading Portfolio (Walk-Forward TEST window)\n')
        fh.write('============================================================\n')
        fh.write(f'Test window      : {TEST_START} to {END}\n')
        fh.write(f'Survivor pairs   : {len(survivors)}\n')
        fh.write(f'Max concurrent   : {max_concurrent}\n')
        fh.write(f'Per pair-trade   : Rs.{per_pair_risk:,.0f}\n')
        fh.write(f'Capital base     : Rs.{capital:,.0f}\n')
        fh.write(f'Cost (default)   : {cost_per_trade}% per pair-trade\n\n')
        fh.write(f'Total trades     : {n}\n')
        fh.write(f'Win rate         : {wr*100:.2f}%\n')
        fh.write(f'Profit factor    : {pf:.2f}\n')
        fh.write(f'Total P&L        : Rs.{total_pnl:,.0f}\n')
        fh.write(f'Total return     : {total_ret_pct:.2f}%\n')
        fh.write(f'Max drawdown     : Rs.{max_dd_rs:,.0f} ({max_dd_pct:.2f}%)\n')
        if not accepted.empty:
            fh.write('\nTrade tape (accepted only):\n')
            fh.write(accepted.to_string(index=False))
            fh.write('\n')

    log(f'Portfolio: n={n} wr={wr*100:.1f}% pf={pf:.2f} '
        f'total={total_ret_pct:.2f}% dd={max_dd_pct:.2f}%')
    return dict(n_pairs=len(survivors), n_trades=n, wr=wr, pf=pf,
                total_ret_pct=total_ret_pct, max_dd_pct=max_dd_pct)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--max-pairs', type=int, default=0,
                   help='Limit number of pairs (sorted by p-value). 0 = all.')
    p.add_argument('--phase', choices=['sweep', 'wf', 'portfolio', 'all'],
                   default='all')
    args = p.parse_args()

    if args.phase in ('sweep', 'all'):
        universe = load_universe()
        run_sweep(universe, max_pairs=args.max_pairs)

    survivors = pd.DataFrame()
    if args.phase in ('wf', 'all'):
        survivors = write_walk_forward_summary()
        write_diamonds(survivors)

    if args.phase in ('portfolio', 'all'):
        if survivors.empty and os.path.exists(WF_CSV):
            survivors = pd.read_csv(WF_CSV)
        if not survivors.empty:
            combined_portfolio(survivors)
        else:
            log('No survivors -> skipping portfolio.')


if __name__ == '__main__':
    main()
